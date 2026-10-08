"""Lexical URL features for phishing detection (v1).

Single source of truth for feature extraction. Used by both
training (ml/train.py) and the API (backend/main.py).

Rules enforced here (per PRD/RULES):
- No scheme-based feature: http:// and https:// are stripped before
  any feature is computed, so predictions are scheme-invariant.
- No network calls. tldextract runs offline via
  TLDExtract(suffix_list_urls=()) using its bundled snapshot.
"""

import re

import tldextract

MAX_URL_LENGTH = 2048

FEATURE_NAMES = [
    "url_len",
    "host_len",
    "path_len",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_digits",
    "num_specials",
    "has_at",
    "has_ip",
    "num_subdomains",
    "has_suspicious_word",
    "has_punycode",
    "has_percent_encoding",
]

SUSPICIOUS_WORDS = [
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "free",
    "bonus",
    "paypal",
    "bank",
    "confirm",
    "signin",
]

FALLBACK_REASON = (
    "No single strong signal; the model judged by a combination of patterns."
)

# Offline extractor: never fetches the public-suffix list from the network.
_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

_IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_PERCENT_RE = re.compile(r"%[0-9a-fA-F]{2}")
_SPECIAL_CHARS = set("%$&=?+")


def _strip_scheme(url: str) -> str:
    """Remove one leading http:// or https:// prefix (case-insensitive)."""
    lowered = url.lower()
    if lowered.startswith("http://"):
        return url[len("http://"):]
    if lowered.startswith("https://"):
        return url[len("https://"):]
    return url


def _split_host_rest(stripped: str) -> tuple[str, str]:
    """Split stripped URL into (host, rest) at the first / ? #.

    Only the host is lowercased; the path/query keeps its case.
    """
    idx = len(stripped)
    for sep in ("?", "#", "/"):
        pos = stripped.find(sep)
        if pos != -1 and pos < idx:
            idx = pos
    if idx == len(stripped):
        return stripped.lower(), ""
    return stripped[:idx].lower(), stripped[idx:]


def _actual_host(host: str) -> str:
    """Return the real host without userinfo, port, or brackets."""
    # Drop userinfo: http://user@example.com -> example.com (host still
    # contains '@'; the real host is right of the last '@').
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    # Bracketed IPv6: [::1] or [::1]:8080.
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            return host[1:end].lower()
        return host.lower()
    # Single colon -> host:port; multiple colons -> bare IPv6, keep as-is.
    if host.count(":") == 1:
        host = host.split(":", 1)[0]
    return host.strip().rstrip(".").lower()


def _is_ip(host: str) -> bool:
    """True for IPv4 literals and simple (bracketed or bare) IPv6."""
    if _IPV4_RE.match(host):
        try:
            return all(0 <= int(part) <= 255 for part in host.split("."))
        except ValueError:
            return False
    # IPv6 heuristic: at least two colons with only hex digits/colons/dots.
    if host.count(":") >= 2 and re.fullmatch(r"[0-9a-fA-F:.]+", host):
        return True
    return False


def extract_features(url: str) -> list[int]:
    """Extract the fixed 14-feature vector for a URL.

    Raises:
        ValueError: for empty/non-string input or URLs over 2048 chars.
    """
    if not isinstance(url, str):
        raise ValueError("url must be a non-empty string")
    if len(url) > MAX_URL_LENGTH:
        raise ValueError(f"url longer than {MAX_URL_LENGTH} characters")
    text = url.strip()
    if not text:
        raise ValueError("url must be a non-empty string")
    if len(text) > MAX_URL_LENGTH:
        raise ValueError(f"url longer than {MAX_URL_LENGTH} characters")

    stripped = _strip_scheme(text)
    host, rest = _split_host_rest(stripped)
    real_host = _actual_host(host)
    lowered_full = stripped.lower()

    url_len = len(stripped)
    host_len = len(host)
    path_len = len(rest)
    num_dots = stripped.count(".")
    num_hyphens = stripped.count("-")
    num_underscores = stripped.count("_")
    num_digits = sum(ch.isdigit() for ch in stripped)
    num_specials = sum(ch in _SPECIAL_CHARS for ch in stripped)
    has_at = 1 if "@" in stripped else 0
    has_ip = 1 if _is_ip(real_host) else 0

    ext = _EXTRACTOR(real_host)
    num_subdomains = len(ext.subdomain.split(".")) if ext.subdomain else 0

    has_suspicious_word = (
        1 if any(word in lowered_full for word in SUSPICIOUS_WORDS) else 0
    )
    has_punycode = 1 if "xn--" in host else 0
    has_percent_encoding = 1 if _PERCENT_RE.search(stripped) else 0

    return [
        url_len,
        host_len,
        path_len,
        num_dots,
        num_hyphens,
        num_underscores,
        num_digits,
        num_specials,
        has_at,
        has_ip,
        num_subdomains,
        has_suspicious_word,
        has_punycode,
        has_percent_encoding,
    ]


def _features_to_dict(features) -> dict:
    """Accept a dict or an ordered vector; return a name -> value dict."""
    if isinstance(features, dict):
        return features
    values = list(features)
    if len(values) != len(FEATURE_NAMES):
        raise ValueError(
            f"expected {len(FEATURE_NAMES)} features, got {len(values)}"
        )
    return dict(zip(FEATURE_NAMES, values))


def reasons_from_features(features) -> list[str]:
    """Turn a feature vector into plain-English reasons.

    Accepts either a dict (name -> value) or an ordered vector matching
    FEATURE_NAMES. Always returns at least one reason.
    """
    f = _features_to_dict(features)
    reasons: list[str] = []

    if f.get("has_ip"):
        reasons.append("Uses an IP address instead of a domain name.")
    if f.get("has_at"):
        reasons.append("Contains '@', which can hide the real destination.")
    if f.get("has_punycode"):
        reasons.append("Uses encoded characters that can mimic a trusted domain.")
    if f.get("has_percent_encoding"):
        reasons.append("Contains encoded characters that hide the real address.")
    if f.get("has_suspicious_word"):
        reasons.append("Contains urgent lures like 'verify' or 'login'.")
    if f.get("url_len", 0) > 75:
        reasons.append("URL is unusually long.")
    if f.get("host_len", 0) > 30:
        reasons.append("Domain part is unusually long.")
    if f.get("path_len", 0) > 50:
        reasons.append("Extra-long path, often used to hide the real page.")
    if f.get("num_dots", 0) > 3:
        reasons.append("Has many dots, a sign of fake subdomains.")
    # NOTE: num_hyphens and num_underscores are model features but have
    # no reasons: threshold validation (docs/threshold_validation.md)
    # showed both fire more often on legit URLs than on phishing ones.
    if f.get("num_digits", 0) > 5:
        reasons.append(
            "Contains many numbers, typical of auto-generated phishing links."
        )
    if f.get("num_specials", 0) >= 3:
        reasons.append(
            "Contains many special symbols, typical of redirect tricks."
        )
    if f.get("num_subdomains", 0) >= 3:
        reasons.append("Uses many subdomains to impersonate a trusted site.")

    if not reasons:
        reasons.append(FALLBACK_REASON)
    return reasons
