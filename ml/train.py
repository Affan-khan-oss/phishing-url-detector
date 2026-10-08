"""Train v3: v2-RF + bare-host augmentation (fixes "no path = phishing").

Pipeline: load ml/data/clean.csv -> featurize via ml/features.py
(cached in ml/data/features.csv, git-ignored) -> domain-grouped
train/val/test split (same seeds as v1/v2, so same rows) ->
AUGMENT train+val legit rows with bare-host variants (post-split,
same groups; test stays un-augmented) -> train RF -> held-out
metrics on the un-augmented test -> threshold tuning on augmented
VAL (0.92 target) -> bare-host slice diagnostic (v2 vs v3, test rows
only, never trained on) -> hand-made probe set (sanity check only,
never tuned on) -> save model + metrics.

Run from the repo root:
  .venv\\Scripts\\python.exe ml/train.py
"""

import json
import sys
import time
from pathlib import Path

# Allow running as a script: `python ml/train.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import tldextract
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit

from ml.features import FEATURE_NAMES, extract_features

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "ml" / "data" / "clean.csv"
CACHE_PATH = ROOT / "ml" / "data" / "features.csv"
MODEL_PATH = ROOT / "models" / "phishing_rf.joblib"
MODEL_V2_PATH = ROOT / "models" / "phishing_rf_v2.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
METRICS_V2_PATH = ROOT / "models" / "metrics_v2.json"
PROBE_PATH = ROOT / "docs" / "probe_set.md"

SEED_SPLIT1 = 42  # same as v1/v2 -> same grouped rows
SEED_SPLIT2 = 43
SEED_MODEL = 42
SEED_AUG_TRAIN = 42
SEED_AUG_VAL = 43
RECALL_TARGET = 0.92  # safety margin: v1 lost ~0.015 val->test
AUG_FRAC = 0.10  # cap: ~10% of legit rows per augmented set

# Offline extractor for grouping only (split logic, not model features).
_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

# Hand-made probe set: sanity check only, NEVER tuned on, NOT a benchmark.
PROBE_WELLKNOWN = [
    "google.com", "youtube.com", "facebook.com", "wikipedia.org",
    "amazon.com", "apple.com", "microsoft.com", "netflix.com",
    "instagram.com", "linkedin.com", "x.com", "reddit.com",
    "yahoo.com", "bing.com", "office.com", "icloud.com",
    "github.com", "stackoverflow.com", "bbc.com", "nytimes.com",
    "cnn.com", "ebay.com", "paypal.com", "chase.com",
    "bankofamerica.com", "wellsfargo.com", "outlook.com", "dropbox.com",
    "spotify.com", "bbc.co.uk",
]
PROBE_PHISHING = [
    "http://192.168.0.1/login",
    "http://10.0.0.5/secure/update.php",
    "http://172.16.9.4:8080/paypal/signin",
    "paypal.login.evil.com/verify",
    "appleid.verify-login.tk/signin",
    "secure-chase-online.gq/login?user=1&s=2",
    "netflix-billing-update.ml/account",
    "amaz0n.payments-verify.top/gp/cart",
    "wellsfargo.signin.verify.cf/login",
    "instagram.verify-login.xyz/accounts",
    "coinbase.wallet-check.click/verify",
    "ebay.motors.deals.buzz/signin?x=1&y=2",
    "linkedin.jobs.verify.shop/login",
    "steamcommunity.trade-confirm.ga/login",
    "discord.nitro-free.top/claim",
    "binance.api-verify.tk/login",
    "alipay.secure-check.ml/account",
    "whatsapp.web-login.xyz/verify",
    "google.docs-share.evil.com/document",
    "outlook.office365-verify.tk/login",
    "dropbox.shared-file.evil.com/s/abc123",
    "spotify.premium-free.gq/claim",
    "reddit.gold-verify.cf/login",
    "x.verify-badge.ml/account",
    "bbc.news-update.tk/article?id=5&s=1",
    "cnn.breaking-news.xyz/story?x=9&y=9",
    "chase.online-access.evil.com/logon",
    "bankofamerica.secure.verify.gq/signin",
    "xn--paypl-7qa.evil.com/login",
    "steam.login.verify-login.tk/signin?session=abc123&token=xyz789",
]


def registered_domain(url: str) -> str:
    """Registrable domain for grouping; falls back to the raw host."""
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
    host = host.strip().rstrip(".")
    ext = _EXTRACTOR(host)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    if ext.domain:
        return ext.domain.lower()
    return host


def bare_host(url: str) -> str:
    """Host-only variant of a URL (no scheme, path, userinfo)."""
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
    return host.strip().rstrip(".")


def load_or_build_features() -> pd.DataFrame:
    """Load the cached feature matrix, or build it from clean.csv."""
    df = pd.read_csv(CLEAN_PATH, usecols=["url", "label"])
    print(f"clean rows: {len(df)}")
    if CACHE_PATH.exists():
        cached = pd.read_csv(CACHE_PATH)
        expected = ["url", "label"] + FEATURE_NAMES
        if (
            list(cached.columns) == expected
            and len(cached) == len(df)
            and bool((cached["url"].values == df["url"].values).all())
        ):
            print(f"using cached features: {CACHE_PATH}")
            return cached
        print("cache mismatch (rows/urls/columns) -> rebuilding")
    else:
        print("no feature cache -> building")

    urls = df["url"].astype(str).tolist()
    X_rows: list[list] = []
    keep: list[bool] = []
    skipped = 0
    t0 = time.time()
    for i, u in enumerate(urls, start=1):
        try:
            X_rows.append(extract_features(u))
            keep.append(True)
        except ValueError:
            keep.append(False)
            skipped += 1
        if i % 50000 == 0:
            print(f"  featurized {i}/{len(urls)} "
                  f"({time.time() - t0:.1f}s)")
    kept = df[pd.Series(keep, index=df.index)].reset_index(drop=True)
    feat = pd.DataFrame(X_rows, columns=FEATURE_NAMES)
    out = pd.concat([kept, feat], axis=1)
    out.to_csv(CACHE_PATH, index=False)
    print(f"featurized {len(out)}/{len(df)} rows "
          f"(skipped {skipped}) in {time.time() - t0:.1f}s")
    print(f"wrote cache: {CACHE_PATH}")
    return out


def metrics_at(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_phish": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_phish": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_phish": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def print_metrics(title: str, m: dict) -> None:
    print(f"{title}: accuracy={m['accuracy']:.4f} "
          f"precision={m['precision_phish']:.4f} "
          f"recall={m['recall_phish']:.4f} f1={m['f1_phish']:.4f}")


def balance_text(name: str, y) -> str:
    y = np.asarray(y)
    n = len(y)
    p = int((y == 1).sum())
    l = n - p
    return (f"{name}: n={n} legit={l} phishing={p} "
            f"phishing_pct={100.0 * p / n:.2f}%")


def augment_bare_legit(urls, y, groups, name: str, seed: int):
    """Bare-host variants of legit rows (label 0, parent's group).

    Unique hosts, capped at ~10% of legit rows. Call AFTER the grouped
    split so test rows are never touched and groups stay set-local.
    """
    legit_pos = [i for i, v in enumerate(y) if v == 0]
    host_to_pos: dict[str, int] = {}
    for i in legit_pos:
        h = bare_host(urls[i])
        if h and h not in host_to_pos:
            host_to_pos[h] = i
    existing = set(urls)
    candidates = [h for h in host_to_pos if h not in existing]
    cap = int(AUG_FRAC * len(legit_pos))
    rng = np.random.default_rng(seed)
    k = min(cap, len(candidates))
    picked = sorted(rng.choice(candidates, size=k, replace=False).tolist())
    X_aug, kept_hosts, skipped = [], [], 0
    for h in picked:
        try:
            X_aug.append(extract_features(h))
            kept_hosts.append(h)
        except ValueError:
            skipped += 1
    g_aug = np.array([groups[host_to_pos[h]] for h in kept_hosts])
    print(f"augment {name}: {len(host_to_pos)} unique legit hosts, "
          f"cap {cap}, added {len(X_aug)} bare-legit rows "
          f"(skipped {skipped})")
    return (np.array(X_aug, dtype=float), np.zeros(len(X_aug), dtype=int),
            g_aug, picked)


def tune_threshold(val_proba, y_val, label: str) -> float:
    """Highest threshold with val recall >= target (VAL only)."""
    coarse = [round(float(t), 2) for t in np.arange(0.05, 1.0, 0.05)]
    print(f"{label} val sweep (threshold -> recall, precision):")
    recalls: dict[float, float] = {}
    for t in coarse:
        pred = (val_proba >= t).astype(int)
        r = float(recall_score(y_val, pred, pos_label=1, zero_division=0))
        p = float(precision_score(y_val, pred, pos_label=1, zero_division=0))
        recalls[t] = r
        print(f"  thr={t:.2f} recall={r:.4f} precision={p:.4f}")
    candidates = [t for t in coarse if recalls[t] >= RECALL_TARGET]
    if not candidates:
        print(f"WARNING: target recall {RECALL_TARGET} NOT reachable on "
              f"val for {label}; falling back to 0.05.")
        return 0.05
    tuned = max(candidates)
    t = round(tuned + 0.01, 2)  # refine upward while target holds
    while t < 1.0:
        pred = (val_proba >= t).astype(int)
        r = float(recall_score(y_val, pred, pos_label=1, zero_division=0))
        if r < RECALL_TARGET:
            break
        tuned = t
        t = round(t + 0.01, 2)
    tuned = float(tuned)
    r = float(recall_score(y_val, (val_proba >= tuned).astype(int),
                           pos_label=1, zero_division=0))
    print(f"{label} tuned threshold={tuned:.2f} (val recall {r:.4f} "
          f"reaches target {RECALL_TARGET})")
    return tuned


def slice_report(name: str, y_true, y_pred) -> dict:
    """Overall slice metrics + per-side rates (legit FP rate, phish recall)."""
    y_true = np.asarray(y_true)
    m = metrics_at(y_true, y_pred)
    legit = y_true == 0
    phish = y_true == 1
    fp_rate = float((y_pred[legit] == 1).mean()) if legit.sum() else 0.0
    ph_recall = float((y_pred[phish] == 1).mean()) if phish.sum() else 0.0
    print(f"{name} bare slice: n={len(y_true)} "
          f"legit={int(legit.sum())} phish={int(phish.sum())} "
          f"fp_rate_legit={fp_rate:.4f} recall_phish={ph_recall:.4f} "
          f"prec={m['precision_phish']:.4f} f1={m['f1_phish']:.4f}")
    return {"n": int(len(y_true)), "n_legit": int(legit.sum()),
            "n_phish": int(phish.sum()), "fp_rate_legit": fp_rate,
            "recall_phish": ph_recall, **m}


def write_probe_set() -> None:
    """Write the hand-made probe list (URLs only, one per line)."""
    lines = ["# Probe set (hand-made sanity check, NOT a benchmark)",
             "",
             "Never tune on these URLs. They are biased by construction:",
             "30 well-known domains (bare + with `/about`) plus 30",
             "phishing-style URLs.",
             "",
             "## well-known bare (30)",
             "",
             "```",
             *[f"https://{d}" for d in PROBE_WELLKNOWN],
             "```",
             "",
             "## well-known with path (30)",
             "",
             "```",
             *[f"https://{d}/about" for d in PROBE_WELLKNOWN],
             "```",
             "",
             "## phishing-style (30)",
             "",
             "```",
             *PROBE_PHISHING,
             "```",
             ""]
    PROBE_PATH.write_text("\n".join(lines))
    print(f"saved {PROBE_PATH} (90 probe URLs)")


def main() -> None:
    frozen = json.loads(METRICS_V2_PATH.read_text()) \
        if METRICS_V2_PATH.exists() else None
    if frozen is None:
        print("WARNING: models/metrics_v2.json missing; "
              "v2 block will be null")
    v2_rf_test = (frozen["v2"]["rf"]["tuned_threshold_test"]
                  if frozen and "v2" in frozen else {})
    v2_thr = float(frozen["v2"]["rf"]["threshold"]) if frozen else 0.35

    data = load_or_build_features()
    print(balance_text("full data", data["label"].values))

    X = data[FEATURE_NAMES].to_numpy(dtype=float)
    y = data["label"].to_numpy()
    urls_all = data["url"].tolist()

    t0 = time.time()
    groups = np.array([registered_domain(u) for u in urls_all])
    print(f"grouped {len(groups)} urls into "
          f"{len(set(groups))} domains ({time.time() - t0:.1f}s)")

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20,
                             random_state=SEED_SPLIT1)
    train_val_idx, test_idx = next(gss1.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25,
                             random_state=SEED_SPLIT2)
    rel_train, rel_val = next(
        gss2.split(X[train_val_idx], y[train_val_idx],
                   groups[train_val_idx]))
    train_idx = train_val_idx[rel_train]
    val_idx = train_val_idx[rel_val]

    # No domain may appear in more than one set.
    assert not (set(groups[train_idx]) & set(groups[val_idx]))
    assert not (set(groups[train_idx]) & set(groups[test_idx]))
    assert not (set(groups[val_idx]) & set(groups[test_idx]))
    print(f"split sizes: train={len(train_idx)} val={len(val_idx)} "
          f"test={len(test_idx)} (seeds {SEED_SPLIT1}/{SEED_SPLIT2})")
    print(balance_text("train", y[train_idx]))
    print(balance_text("val  ", y[val_idx]))
    print(balance_text("test ", y[test_idx]))

    # Augment train+val ONLY (post-split, same groups). Test untouched.
    train_urls = [urls_all[i] for i in train_idx]
    val_urls = [urls_all[i] for i in val_idx]
    X_tr_a, y_tr_a, g_tr_a, _ = augment_bare_legit(
        train_urls, y[train_idx], groups[train_idx], "train",
        SEED_AUG_TRAIN)
    X_va_a, y_va_a, g_va_a, _ = augment_bare_legit(
        val_urls, y[val_idx], groups[val_idx], "val", SEED_AUG_VAL)

    X_train = np.vstack([X[train_idx], X_tr_a])
    y_train = np.concatenate([y[train_idx], y_tr_a])
    X_val = np.vstack([X[val_idx], X_va_a])
    y_val = np.concatenate([y[val_idx], y_va_a])
    X_test, y_test = X[test_idx], y[test_idx]  # un-augmented headline set
    print(balance_text("train+aug", y_train))
    print(balance_text("val+aug", y_val))
    print(balance_text("test (un-augmented)", y_test))

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    base_m = metrics_at(y_test, dummy.predict(X_test))
    print_metrics("baseline (always legit) test", base_m)

    rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=SEED_MODEL,
        n_jobs=-1,
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    print(f"trained RandomForest in {time.time() - t0:.1f}s")

    default_m = metrics_at(y_test, rf.predict(X_test))
    print_metrics("v3 RF test @0.50", default_m)
    tuned = tune_threshold(rf.predict_proba(X_val)[:, 1], y_val, "v3 RF")
    val_m = metrics_at(y_val, (rf.predict_proba(X_val)[:, 1] >= tuned))
    print_metrics(f"v3 RF val @{tuned:.2f}", val_m)
    test_pred = (rf.predict_proba(X_test)[:, 1] >= tuned).astype(int)
    tuned_m = metrics_at(y_test, test_pred)
    print_metrics(f"v3 RF test @{tuned:.2f}", tuned_m)
    cm = confusion_matrix(y_test, test_pred)
    print(f"v3 RF confusion matrix test @{tuned:.2f}:")
    print(cm)

    importances = sorted(zip(FEATURE_NAMES, rf.feature_importances_),
                         key=lambda kv: kv[1], reverse=True)
    print("v3 RF top 10 feature importances:")
    for rank, (name, imp) in enumerate(importances[:10], start=1):
        print(f"  {rank:2d}. {name:<20s} {imp:.4f}")

    # Bare-host slice diagnostic: test rows ONLY, never trained on.
    test_urls = [urls_all[i] for i in test_idx]
    host_labels: dict[str, list[int]] = {}
    for u, v in zip(test_urls, y_test):
        h = bare_host(u)
        if h:
            host_labels.setdefault(h, []).append(int(v))
    slice_hosts, slice_y, dropped = [], [], 0
    for h, vs in host_labels.items():
        if vs.count(0) == vs.count(1):
            dropped += 1
            continue
        slice_hosts.append(h)
        slice_y.append(0 if vs.count(0) > vs.count(1) else 1)
    X_slice = np.array([extract_features(h) for h in slice_hosts],
                       dtype=float)
    y_slice = np.array(slice_y)
    print(f"bare slice: {len(slice_hosts)} unique hosts "
          f"({dropped} tied hosts dropped)")
    slice_res: dict[str, dict] = {}
    if MODEL_V2_PATH.exists():
        v2 = joblib.load(MODEL_V2_PATH)
        v2_pred = (v2["model"].predict_proba(X_slice)[:, 1]
                   >= float(v2["threshold"])).astype(int)
        slice_res["v2"] = {"threshold": float(v2["threshold"]),
                           **slice_report("v2", y_slice, v2_pred)}
    else:
        print("WARNING: v2 joblib copy missing; v2 slice skipped")
    v3_pred = (rf.predict_proba(X_slice)[:, 1] >= tuned).astype(int)
    slice_res["v3"] = {"threshold": float(tuned),
                       **slice_report("v3", y_slice, v3_pred)}

    # Probe set: sanity check only, never tuned on.
    write_probe_set()
    train_groups = set(groups[train_idx].tolist())
    probe_urls = ([f"https://{d}" for d in PROBE_WELLKNOWN]
                  + [f"https://{d}/about" for d in PROBE_WELLKNOWN]
                  + PROBE_PHISHING)
    X_probe = np.array([extract_features(u) for u in probe_urls],
                       dtype=float)
    probe_res: dict[str, dict] = {}
    models = {"v3": (rf, float(tuned))}
    if MODEL_V2_PATH.exists():
        v2 = joblib.load(MODEL_V2_PATH)
        models["v2"] = (v2["model"], float(v2["threshold"]))
    for mname, (m, thr) in models.items():
        pred = (m.predict_proba(X_probe)[:, 1] >= thr).astype(int)
        wk_bare = pred[0:30]
        wk_path = pred[30:60]
        ph = pred[60:90]
        seen = [registered_domain(u) in train_groups for u in probe_urls]
        entry = {
            "threshold": thr,
            "wellknown_bare_safe": f"{int((wk_bare == 0).sum())}/30",
            "wellknown_path_safe": f"{int((wk_path == 0).sum())}/30",
            "phishing_caught": f"{int((ph == 1).sum())}/30",
        }
        for part, sl in (("wellknown_bare", slice(0, 30)),
                         ("wellknown_path", slice(30, 60)),
                         ("phishing", slice(60, 90))):
            idx = list(range(sl.start, sl.stop))
            s_seen = [i for i in idx if seen[i]]
            s_new = [i for i in idx if not seen[i]]
            good = (pred == 0) if part != "phishing" else (pred == 1)
            entry[f"{part}_seen"] = (
                f"{int(good[s_seen].sum())}/{len(s_seen)}" if s_seen
                else "n/a")
            entry[f"{part}_unseen"] = (
                f"{int(good[s_new].sum())}/{len(s_new)}" if s_new
                else "n/a")
        probe_res[mname] = entry
        print(f"probe {mname} (thr={thr:.2f}): well-known bare safe "
              f"{entry['wellknown_bare_safe']} "
              f"(seen {entry['wellknown_bare_seen']}, "
              f"unseen {entry['wellknown_bare_unseen']}); with path safe "
              f"{entry['wellknown_path_safe']} "
              f"(seen {entry['wellknown_path_seen']}, "
              f"unseen {entry['wellknown_path_unseen']}); phishing caught "
              f"{entry['phishing_caught']} "
              f"(seen {entry['phishing_seen']}, "
              f"unseen {entry['phishing_unseen']})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": rf, "threshold": float(tuned),
                 "feature_names": FEATURE_NAMES,
                 "random_state": SEED_MODEL},
                MODEL_PATH, compress=3)
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"saved {MODEL_PATH} ({size_mb:.2f} MB)")

    v3 = {
        "threshold": float(tuned),
        "default_threshold_test": {"threshold": 0.5, **default_m},
        "tuned_val": {"threshold": float(tuned), **val_m},
        "tuned_threshold_test": {"threshold": float(tuned), **tuned_m},
        "confusion_matrix_test_tuned": cm.tolist(),
        "feature_importances_top10": [
            {"feature": n, "importance": float(i)}
            for n, i in importances[:10]
        ],
        "augmentation": {"train_added": int(len(X_tr_a)),
                         "val_added": int(len(X_va_a))},
        "bare_slice": slice_res,
        "probe": probe_res,
        "model_size_mb": round(size_mb, 2),
    }
    metrics = {
        "random_state_split1": SEED_SPLIT1,
        "random_state_split2": SEED_SPLIT2,
        "random_state_model": SEED_MODEL,
        "recall_target": RECALL_TARGET,
        "n_features": len(FEATURE_NAMES),
        "splits": {
            "train": {"n": int(len(train_idx))},
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "baseline_test": base_m,
        "v2": frozen,
        "v3": v3,
        "selected": "rf",
        "model_file": MODEL_PATH.name,
        "model_size_mb": round(size_mb, 2),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"saved {METRICS_PATH}")

    r = v3["tuned_threshold_test"]
    print("v2 vs v3 (@val-tuned threshold, un-augmented test set):")
    print("model      thr     acc    prec   recall   f1")
    if v2_rf_test:
        print(f"v2 RF      {v2_rf_test['threshold']:.2f}  "
              f"{v2_rf_test['accuracy']:.4f} "
              f"{v2_rf_test['precision_phish']:.4f} "
              f"{v2_rf_test['recall_phish']:.4f} "
              f"{v2_rf_test['f1_phish']:.4f}")
    print(f"v3 RF      {r['threshold']:.2f}  "
          f"{r['accuracy']:.4f} {r['precision_phish']:.4f} "
          f"{r['recall_phish']:.4f} {r['f1_phish']:.4f}")


if __name__ == "__main__":
    main()
