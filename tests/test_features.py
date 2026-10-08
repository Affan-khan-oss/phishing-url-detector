"""Tests for ml/features.py (Phase 2.2)."""

import pytest

from ml.features import (
    FALLBACK_REASON,
    FEATURE_NAMES,
    extract_features,
    reasons_from_features,
)


def _index(name: str) -> int:
    return FEATURE_NAMES.index(name)


def test_same_url_same_vector():
    url = "http://paypal.secure-login.evil.com/verify?account=123"
    assert extract_features(url) == extract_features(url)


def test_stable_feature_order_and_length():
    assert len(FEATURE_NAMES) == 14
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)  # no duplicates
    first = extract_features("http://example.com/login")
    second = extract_features("http://paypal.secure-login.evil.com/verify")
    assert len(first) == len(FEATURE_NAMES)
    assert len(second) == len(FEATURE_NAMES)
    # Spot-check order: has_ip sits at its documented index.
    assert first[_index("has_ip")] == 0


def test_ip_in_host_flagged():
    feats = extract_features("http://192.168.0.1/login")
    assert feats[_index("has_ip")] == 1
    reasons = reasons_from_features(feats)
    assert any("IP address" in r for r in reasons)


def test_scheme_and_no_scheme_identical():
    with_scheme = extract_features("http://example.com/login")
    without_scheme = extract_features("example.com/login")
    https_scheme = extract_features("https://example.com/login")
    assert with_scheme == without_scheme == https_scheme


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        extract_features("")
    with pytest.raises(ValueError):
        extract_features("   ")
    with pytest.raises(ValueError):
        extract_features(None)
    with pytest.raises(ValueError):
        extract_features(123)
    with pytest.raises(ValueError):
        extract_features("x" * 2049)


def test_reasons_nonempty_and_fallback():
    phishing = extract_features("http://192.168.0.1/login")
    assert len(reasons_from_features(phishing)) >= 1
    clean = extract_features("example.com")
    assert reasons_from_features(clean) == [FALLBACK_REASON]
