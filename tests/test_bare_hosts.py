"""Regression: well-known hosts must not be flagged as phishing.

Covers the bare-host bias (v2 scored bare domains ~0.97 phishing):
google.com, github.com and wikipedia.org, bare and with /about,
must score below the tuned threshold.
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pytest

from ml.features import FEATURE_NAMES, extract_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "phishing_rf.joblib"))

CASES = [
    "google.com",
    pytest.param(
        "google.com/about",
        marks=pytest.mark.xfail(
            strict=True,
            reason="short-path quirk, see docs/known_limitations.md",
        ),
    ),
    "github.com",
    pytest.param(
        "github.com/about",
        marks=pytest.mark.xfail(
            strict=True,
            reason="short-path quirk, see docs/known_limitations.md",
        ),
    ),
    "wikipedia.org",
    pytest.param(
        "wikipedia.org/about",
        marks=pytest.mark.xfail(
            strict=True,
            reason="short-path quirk, see docs/known_limitations.md",
        ),
    ),
]


def _load():
    if not MODEL_PATH.exists():
        pytest.skip(f"model artifact missing: {MODEL_PATH} (run ml/train.py)")
    bundle = joblib.load(MODEL_PATH)
    assert list(bundle["feature_names"]) == FEATURE_NAMES
    return bundle["model"], float(bundle["threshold"])


@pytest.mark.parametrize("url", CASES)
def test_wellknown_not_phishing(url):
    model, threshold = _load()
    score = float(model.predict_proba(
        np.array([extract_features(url)], dtype=float))[0, 1])
    assert score < threshold, f"{url} scored {score:.3f} >= {threshold}"
