"""Phishing URL API (Phase 4).

POST /predict -> {label, score, threshold, risk_level, reasons[],
source, override, disclaimer}. The model artifact is loaded ONCE at
startup; feature extraction and reasons come only from ml/features.py.
A small offline allowlist (backend/allowlist.txt) overrides the model
verdict for well-known domains — score and model reasons are still
returned for transparency.

Run from the repo root:
  uvicorn backend.main:app --reload --port 8000
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Allow running/importing from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ml.features import (
    FEATURE_NAMES,
    MAX_URL_LENGTH,
    extract_features,
    reasons_from_features,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "phishing_rf.joblib"))
ALLOWLIST_PATH = Path(
    os.getenv("ALLOWLIST_PATH", ROOT / "backend" / "allowlist.txt")
)
METRICS_PATH = Path(
    os.getenv("METRICS_PATH", ROOT / "models" / "metrics.json")
)
MAX_BODY_BYTES = 4096
RISK_HIGH_CUT = 0.65

DISCLAIMER_BASE = (
    "Lexical heuristic only, not a safety guarantee. "
    "The score is the share of model trees voting phishing, "
    "not a calibrated probability."
)
DISCLAIMER_ALLOWLIST = (
    " Allowlist match means the domain is well-known; it does not "
    "guarantee this specific page is safe (legitimate sites can be "
    "compromised)."
)
ALLOWLIST_REASON = "Well-known domain on the allowlist"


class PredictRequest(BaseModel):
    url: str


def _request_host(url: str) -> str:
    """Host part of a URL for allowlist routing only (never features).

    Strips one http(s) scheme, cuts at the first / ? #, lowercases,
    drops userinfo (right of the last @ is the real host), drops a
    single :port, strips a trailing dot.
    """
    text = url.strip()
    lowered = text.lower()
    if lowered.startswith("http://"):
        text = text[len("http://"):]
    elif lowered.startswith("https://"):
        text = text[len("https://"):]
    end = len(text)
    for sep in ("?", "#", "/"):
        pos = text.find(sep)
        if pos != -1 and pos < end:
            end = pos
    host = text[:end].lower()
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if host.startswith("["):
        bracket = host.find("]")
        host = host[1:bracket] if bracket != -1 else host
    elif host.count(":") == 1:
        host = host.split(":", 1)[0]
    return host.strip().rstrip(".")


def _allowlist_match(host: str, allowlist: set) -> str | None:
    """Matched allowlist domain, or None.

    Exact host or www.<domain> only — never arbitrary subdomains, so
    docs.google.com or google.com.evil.com never match google.com.
    """
    if host in allowlist:
        return host
    if host.startswith("www.") and host[len("www."):] in allowlist:
        return host[len("www."):]
    return None


def _risk_level(score: float, threshold: float) -> str:
    if score < threshold:
        return "low"
    if score < RISK_HIGH_CUT:
        return "medium"
    return "high"


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(MODEL_PATH)
    for key in ("model", "threshold", "feature_names"):
        if key not in bundle:
            raise RuntimeError(f"model artifact missing key: {key}")
    if list(bundle["feature_names"]) != FEATURE_NAMES:
        raise RuntimeError("model features do not match ml/features.py")
    app.state.model = bundle["model"]
    app.state.threshold = float(bundle["threshold"])
    app.state.allowlist = {
        line.strip().lower()
        for line in ALLOWLIST_PATH.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    if not app.state.allowlist:
        raise RuntimeError(f"allowlist empty: {ALLOWLIST_PATH}")
    app.state.metrics = (
        json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists()
        else None
    )
    yield


app = FastAPI(title="Phishing URL detector", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    if request.url.path == "/predict":
        length = request.headers.get("content-length")
        if length and int(length) > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": "request body too large "
                                   f"(max {MAX_BODY_BYTES} bytes)"},
            )
    return await call_next(request)


def _invalid(message: str) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": message})


@app.post("/predict")
def predict(req: PredictRequest):
    text = req.url.strip()
    if not text:
        return _invalid("url must be a non-empty string")
    if len(text) > MAX_URL_LENGTH:
        return _invalid(
            f"url too long (max {MAX_URL_LENGTH} characters)")
    try:
        feats = extract_features(text)
    except ValueError as exc:
        return _invalid(str(exc))
    if len(feats) != len(FEATURE_NAMES):
        return JSONResponse(
            status_code=500,
            content={"detail": "feature drift: model and "
                               "ml/features.py disagree"},
        )
    score = float(
        app.state.model.predict_proba(
            np.array([feats], dtype=float))[0, 1])
    threshold = app.state.threshold
    model_says_phishing = score >= threshold
    reasons = reasons_from_features(feats)

    matched = _allowlist_match(_request_host(text), app.state.allowlist)
    if matched is not None:
        return {
            "label": "safe",
            "score": score,
            "threshold": threshold,
            "risk_level": "low",
            "reasons": [f"{ALLOWLIST_REASON} (matched {matched} — "
                        "verdict overridden)."] + reasons,
            "source": "allowlist",
            "override": bool(model_says_phishing),
            "disclaimer": DISCLAIMER_BASE + DISCLAIMER_ALLOWLIST,
        }
    return {
        "label": "phishing" if model_says_phishing else "safe",
        "score": score,
        "threshold": threshold,
        "risk_level": _risk_level(score, threshold),
        "reasons": reasons,
        "source": "model",
        "override": False,
        "disclaimer": DISCLAIMER_BASE,
    }


@app.get("/health")
def health():
    return {"status": "ok",
            "model_loaded": hasattr(app.state, "model")}


@app.get("/metrics")
def metrics():
    if app.state.metrics is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "metrics not available "
                               "(run ml/train.py first)"},
        )
    return app.state.metrics
