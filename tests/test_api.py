"""Tests for backend/main.py (Phase 4.2).

Requires the trained artifact (run ml/train.py first); the whole
module skips otherwise. Uses FastAPI TestClient (lifespan loads the
model + allowlist once).
"""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import ALLOWLIST_PATH, ALLOWLIST_REASON, app
from ml.features import (
    FALLBACK_REASON,
    FEATURE_NAMES,
    extract_features,
    reason_strength,
    reasons_from_features,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "phishing_rf.joblib"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="run ml/train.py first")

BANNED_FROM_ALLOWLIST = {
    # shared hosting / user content
    "github.io", "blogspot.com", "herokuapp.com", "netlify.app",
    "web.app", "pages.dev", "vercel.app", "workers.dev", "appspot.com",
    "azurewebsites.net", "sharepoint.com",
    # URL shorteners
    "bit.ly", "t.co", "goo.gl", "tinyurl.com", "ow.ly", "is.gd",
    "cutt.ly",
    # file hosting
    "storage.googleapis.com", "s3.amazonaws.com", "onedrive.live.com",
    "dropbox.com", "drive.google.com", "sites.google.com",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _check_shape(body: dict):
    assert body["label"] in ("phishing", "safe")
    assert 0.0 <= body["score"] <= 1.0
    assert body["threshold"] == pytest.approx(0.35)
    assert body["risk_level"] in ("low", "medium", "high")
    assert len(body["reasons"]) >= 1
    assert isinstance(body["strong_signals"], bool)
    assert "specific_signals" not in body
    # strong_signals reflects the model reasons only: the fallback and
    # the allowlist routing note never count.
    model_reasons = [r for r in body["reasons"]
                     if not r.startswith(ALLOWLIST_REASON)]
    assert body["strong_signals"] == any(
        reason_strength(r) == "strong" for r in model_reasons)
    assert body["source"] in ("model", "allowlist")
    assert isinstance(body["override"], bool)
    assert "lexical heuristic" in body["disclaimer"].lower()
    assert body["label"] == (
        "phishing" if body["score"] >= body["threshold"] else "safe"
    ) or body["source"] == "allowlist"


def test_predict_clean_model_url(client):
    r = client.post("/predict",
                    json={"url": "http://example.com/about/team/contact-us"})
    assert r.status_code == 200
    body = r.json()
    _check_shape(body)
    assert body["label"] == "safe"
    assert body["source"] == "model"
    assert body["override"] is False
    assert body["strong_signals"] is False  # fallback reason only


def test_strong_signals_flag(client):
    clean = client.post(
        "/predict", json={"url": "http://example.com/about/team/contact-us"}
    ).json()
    assert clean["reasons"] == [FALLBACK_REASON]
    assert clean["strong_signals"] is False

    phish = client.post(
        "/predict",
        json={"url": "http://paypal-login-secure-update.tk/signin"},
    ).json()
    assert phish["label"] == "phishing"
    assert phish["strong_signals"] is True  # lure words + cheap TLD
    assert any(reason_strength(r) == "strong" for r in phish["reasons"])

    # High score from weak reasons only: long ID, deep path, entropy.
    forms = client.post(
        "/predict",
        json={"url": "https://docs.google.com/forms/d/e/"
                     "1FAIpQLSdaBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789abcdef/"
                     "viewform"},
    ).json()
    assert forms["label"] == "phishing"
    assert forms["risk_level"] == "high"
    assert forms["source"] == "model"  # docs.google.com never allowlisted
    assert forms["strong_signals"] is False
    assert all(reason_strength(r) == "weak" for r in forms["reasons"])


def test_allowlist_bare_and_path(client):
    bare = client.post("/predict", json={"url": "google.com"}).json()
    assert bare["label"] == "safe"
    assert bare["source"] == "allowlist"
    assert bare["override"] is False  # model also says safe (0.035)
    assert "allowlist" in bare["reasons"][0].lower()
    assert "verdict overridden" not in bare["reasons"][0].lower()
    assert FALLBACK_REASON not in bare["reasons"]
    assert bare["strong_signals"] is False  # model reasons are fallback-only

    path = client.post("/predict", json={"url": "google.com/about"}).json()
    assert path["label"] == "safe"
    assert path["source"] == "allowlist"
    assert path["override"] is True  # model says phishing (0.677)
    assert "allowlist" in path["reasons"][0].lower()
    assert "verdict overridden" in path["reasons"][0].lower()
    assert FALLBACK_REASON not in path["reasons"]
    assert path["strong_signals"] is False  # model reasons are fallback-only
    assert "does not guarantee" in path["disclaimer"]


def test_allowlist_www_and_trailing_dot(client):
    www = client.post("/predict", json={"url": "www.google.com"}).json()
    assert www["source"] == "allowlist"
    assert www["label"] == "safe"
    dot = client.post("/predict", json={"url": "google.com."}).json()
    assert dot["source"] == "allowlist"
    assert dot["label"] == "safe"


def test_allowlist_amazon_in_variants(client):
    for url in ("amazon.in", "www.amazon.in/"):
        body = client.post("/predict", json={"url": url}).json()
        assert body["source"] == "allowlist", url
        assert body["label"] == "safe", url


def test_amazon_in_lookalike_not_matched(client):
    body = client.post(
        "/predict", json={"url": "amazon.in.evil.com"}
    ).json()
    assert body["source"] == "model"
    assert body["strong_signals"] is True  # brand-mismatch is strong


def test_evil_subdomain_not_matched(client):
    body = client.post(
        "/predict", json={"url": "http://paypal.login.evil.com/verify"}
    ).json()
    assert body["source"] == "model"
    assert body["label"] == "phishing"


def test_lookalike_domains_not_matched(client):
    for url in ("google.com.evil.com",
                "docs.google.com/spreadsheet/viewform",
                "foo.github.io",
                "xn--ggle-0nda.com"):
        body = client.post("/predict", json={"url": url}).json()
        assert body["source"] == "model", url


def test_userinfo_host_not_matched(client):
    body = client.post(
        "/predict", json={"url": "http://google.com@evil.com/login"}
    ).json()
    # Real host is evil.com -> must go to the model, never the allowlist.
    assert body["source"] == "model"


def test_empty_and_garbage_422(client):
    for payload in ({"url": ""}, {"url": "   "}, {"url": 123},
                    {"url": "x" * 2049}):
        r = client.post("/predict", json=payload)
        assert r.status_code == 422, payload
        assert "detail" in r.json()


def test_oversized_body_413(client):
    r = client.post("/predict", json={"url": "x" * 5000})
    assert r.status_code == 413


def test_scheme_invariance(client):
    a = client.post(
        "/predict", json={"url": "http://example.com/about/team/contact-us"}
    ).json()
    b = client.post(
        "/predict", json={"url": "example.com/about/team/contact-us"}
    ).json()
    # approx: parallel tree inference (n_jobs=-1) can flip the last bit.
    assert a["score"] == pytest.approx(b["score"])


def test_no_duplicated_feature_logic(client):
    url = "http://paypal.login.evil.com/verify"
    body = client.post("/predict", json={"url": url}).json()
    feats = extract_features(url)
    assert body["reasons"] == reasons_from_features(feats)
    assert len(feats) == len(FEATURE_NAMES)


def test_health_and_metrics(client):
    assert client.get("/health").json()["status"] == "ok"
    metrics = client.get("/metrics").json()
    assert "v2" in metrics and "v3" in metrics


def test_allowlist_hygiene():
    lines = [ln.strip() for ln in ALLOWLIST_PATH.read_text().splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    assert len(lines) >= 180  # about 200
    assert len(set(lines)) == len(lines)
    assert all(ln == ln.lower() for ln in lines)
    assert all("://" not in ln and "/" not in ln for ln in lines)
    assert not (set(lines) & BANNED_FROM_ALLOWLIST)
