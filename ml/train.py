"""Train the URL phishing Random Forest (Phase 3).

Pipeline: load ml/data/clean.csv -> featurize via ml/features.py
(cached in ml/data/features.csv, git-ignored) -> domain-grouped
train/val/test split -> RandomForest -> held-out metrics ->
threshold tuning on VAL only -> save model + metrics.

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
METRICS_PATH = ROOT / "models" / "metrics.json"

SEED_SPLIT1 = 42
SEED_SPLIT2 = 43
SEED_MODEL = 42
RECALL_TARGET = 0.90

# Offline extractor for grouping only (split logic, not model features).
_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())


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
    X_rows: list[list[int]] = []
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


def main() -> None:
    data = load_or_build_features()
    print(balance_text("full data", data["label"].values))

    X = data[FEATURE_NAMES].to_numpy()
    y = data["label"].to_numpy()

    t0 = time.time()
    groups = np.array([registered_domain(u) for u in data["url"].tolist()])
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

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    base_m = metrics_at(y_test, dummy.predict(X_test))
    print_metrics("baseline (always legit) test", base_m)

    clf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=SEED_MODEL,
        n_jobs=-1,
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"trained RandomForest in {time.time() - t0:.1f}s")

    default_m = metrics_at(y_test, clf.predict(X_test))
    print_metrics("test @0.50", default_m)
    print("confusion matrix test @0.50 "
          "(rows=true legit/phish, cols=pred legit/phish):")
    print(confusion_matrix(y_test, clf.predict(X_test)))

    # Threshold sweep on VAL only (never the test set).
    val_proba = clf.predict_proba(X_val)[:, 1]
    print("val sweep (threshold -> recall, precision):")
    coarse = [round(float(t), 2) for t in np.arange(0.05, 1.0, 0.05)]
    val_recall: dict[float, float] = {}
    for t in coarse:
        pred = (val_proba >= t).astype(int)
        r = float(recall_score(y_val, pred, pos_label=1, zero_division=0))
        p = float(precision_score(y_val, pred, pos_label=1, zero_division=0))
        val_recall[t] = r
        print(f"  thr={t:.2f} recall={r:.4f} precision={p:.4f}")
    candidates = [t for t in coarse if val_recall[t] >= RECALL_TARGET]
    if candidates:
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
        tuned_val_recall = float(recall_score(
            y_val, (val_proba >= tuned).astype(int),
            pos_label=1, zero_division=0))
        print(f"tuned threshold={tuned:.2f} (val recall "
              f"{tuned_val_recall:.4f} reaches target {RECALL_TARGET})")
    else:
        tuned = min(coarse)
        print(f"WARNING: target recall {RECALL_TARGET} NOT reachable on "
              f"val; falling back to threshold={tuned:.2f}. The target "
              f"should be lowered with a written reason (see PRD).")
    tuned = float(tuned)

    val_pred = (val_proba >= tuned).astype(int)
    val_m = metrics_at(y_val, val_pred)
    print_metrics(f"val @{tuned:.2f}", val_m)

    test_proba = clf.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= tuned).astype(int)
    tuned_m = metrics_at(y_test, test_pred)
    print_metrics(f"test @{tuned:.2f}", tuned_m)
    print(f"confusion matrix test @{tuned:.2f}:")
    cm_tuned = confusion_matrix(y_test, test_pred)
    print(cm_tuned)

    importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                         key=lambda kv: kv[1], reverse=True)
    print("top 10 feature importances:")
    for rank, (name, imp) in enumerate(importances[:10], start=1):
        print(f"  {rank:2d}. {name:<20s} {imp:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "threshold": tuned,
                 "feature_names": FEATURE_NAMES,
                 "random_state": SEED_MODEL},
                MODEL_PATH, compress=3)
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"saved {MODEL_PATH} ({size_mb:.2f} MB)")

    metrics = {
        "random_state_split1": SEED_SPLIT1,
        "random_state_split2": SEED_SPLIT2,
        "random_state_model": SEED_MODEL,
        "recall_target": RECALL_TARGET,
        "splits": {
            "train": {"n": int(len(train_idx))},
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "baseline_test": base_m,
        "default_threshold_test": {"threshold": 0.5, **default_m},
        "tuned_val": {"threshold": tuned, **val_m},
        "tuned_threshold_test": {"threshold": tuned, **tuned_m},
        "confusion_matrix_test_tuned": cm_tuned.tolist(),
        "feature_importances_top10": [
            {"feature": n, "importance": float(i)}
            for n, i in importances[:10]
        ],
        "model_file": str(MODEL_PATH.name),
        "model_size_mb": round(size_mb, 2),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"saved {METRICS_PATH}")


if __name__ == "__main__":
    main()
