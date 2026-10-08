"""Tests for ml/features.py (Phase 2.2, v2: 24 features)."""

import pytest

from ml.features import (
    FALLBACK_REASON,
    FEATURE_NAMES,
    STRONG_REASONS,
    extract_features,
    reason_strength,
    reasons_from_features,
)

N_FEATURES = 24


def _index(name: str) -> int:
    return FEATURE_NAMES.index(name)


def test_same_url_same_vector():
    url = "http://paypal.secure-login.evil.com/verify?account=123"
    assert extract_features(url) == extract_features(url)


def test_stable_feature_order_and_length():
    assert len(FEATURE_NAMES) == N_FEATURES
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


def test_scheme_invariance_with_v2_features():
    url = "https://paypal.login.evil-files.tk/a/b/c/signin.php?x=1&y=2"
    bare = url.split("://", 1)[1]
    assert extract_features(url) == extract_features(bare)


def test_brand_mismatch_flagged():
    spoof = extract_features("http://paypal.login.evil.com/verify")
    assert spoof[_index("brand_mismatch")] == 1
    assert any("brand" in r for r in reasons_from_features(spoof))
    genuine = extract_features("http://paypal.com/login")
    assert genuine[_index("brand_mismatch")] == 0


def test_risky_ext_and_entropies():
    feats = extract_features("http://evil.com/files/invoice.exe")
    assert feats[_index("has_risky_ext")] == 1
    assert feats[_index("host_entropy")] >= 0.0
    assert feats[_index("path_entropy")] >= 0.0
    assert isinstance(feats[_index("digit_ratio")], float)


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


def test_reason_strength_tags():
    assert reason_strength(FALLBACK_REASON) == "none"
    assert reason_strength("URL is unusually long.") == "weak"
    assert (
        reason_strength("Uses an IP address instead of a domain name.")
        == "strong"
    )
    # Every reason the helper can emit is tagged, and each of the six
    # strong reasons fires on at least one battery URL.
    battery = [
        "http://192.168.0.1/login",
        "http://evil.com/a@b",
        "http://xn--ggle-0nda.com/login",
        "http://paypal-login-secure-update.tk/signin",
        "http://paypal.login.evil.com/verify",
        "http://evil.com/files/invoice.exe",
        "https://docs.google.com/forms/d/e/1FAIpQLSdaBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789abcdef/viewform",
        "example.com/about/team/contact-us",
    ]
    seen_strong = set()
    for url in battery:
        for r in reasons_from_features(extract_features(url)):
            strength = reason_strength(r)
            assert strength in ("strong", "weak", "none"), r
            if strength == "strong":
                seen_strong.add(r)
    assert seen_strong == set(STRONG_REASONS)
