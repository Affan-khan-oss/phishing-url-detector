"""Tests for ALLOWED_ORIGINS CORS env var (free-tier hosting).

Does not need the trained model except for the n_jobs check, which
skips when the artifact is missing.
"""

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.main as main_mod
from backend.main import get_allowed_origins

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "phishing_rf.joblib"


def test_default_origin(monkeypatch):
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    assert get_allowed_origins() == ["http://localhost:3000"]


def test_comma_separated_with_spaces(monkeypatch):
    monkeypatch.setenv(
        "ALLOWED_ORIGINS",
        "https://a.example.com, https://b.example.com ",
    )
    assert get_allowed_origins() == [
        "https://a.example.com",
        "https://b.example.com",
    ]


def test_blank_entries_ignored(monkeypatch):
    monkeypatch.setenv(
        "ALLOWED_ORIGINS",
        " https://a.example.com ,, , http://localhost:3000 ",
    )
    assert get_allowed_origins() == [
        "https://a.example.com",
        "http://localhost:3000",
    ]


def test_blank_value_falls_back_to_default(monkeypatch):
    for blank in ("", "   ", " , , "):
        monkeypatch.setenv("ALLOWED_ORIGINS", blank)
        assert get_allowed_origins() == ["http://localhost:3000"]


def _cors_allow_origins(app):
    for mw in app.user_middleware:
        kwargs = getattr(mw, "kwargs", None) or {}
        if getattr(mw.cls, "__name__", "") == "CORSMiddleware":
            return kwargs.get("allow_origins")
    return None


def test_app_middleware_uses_env(monkeypatch):
    """Reloading backend.main picks up ALLOWED_ORIGINS for CORSMiddleware."""
    monkeypatch.setenv(
        "ALLOWED_ORIGINS",
        "https://shop.example.com,https://app.example.com",
    )
    try:
        reloaded = importlib.reload(main_mod)
        assert reloaded.ALLOWED_ORIGINS == [
            "https://shop.example.com",
            "https://app.example.com",
        ]
        assert _cors_allow_origins(reloaded.app) == [
            "https://shop.example.com",
            "https://app.example.com",
        ]
    finally:
        monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
        importlib.reload(main_mod)


def test_app_middleware_default(monkeypatch):
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    try:
        reloaded = importlib.reload(main_mod)
        assert reloaded.ALLOWED_ORIGINS == ["http://localhost:3000"]
        assert _cors_allow_origins(reloaded.app) == ["http://localhost:3000"]
    finally:
        importlib.reload(main_mod)


@pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="run ml/train.py first")
def test_inference_single_threaded():
    """Lifespan pins model.n_jobs = 1 for stable free-tier inference."""
    from backend.main import app

    with TestClient(app):
        assert getattr(app.state.model, "n_jobs", 1) == 1
