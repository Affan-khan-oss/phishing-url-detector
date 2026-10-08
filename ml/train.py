"""Train v2 models: RandomForest vs XGBoost on 24 features (v2 track).

Pipeline: load ml/data/clean.csv -> featurize via ml/features.py
(cached in ml/data/features.csv, git-ignored) -> domain-grouped
train/val/test split (same seeds as v1, so same rows) -> train RF +
XGB -> held-out metrics -> per-model threshold tuning on VAL only
(target 0.92 val recall as safety margin) -> save winner + metrics.

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
from xgboost import XGBClassifier

from ml.features import FEATURE_NAMES, extract_features

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "ml" / "data" / "clean.csv"
CACHE_PATH = ROOT / "ml" / "data" / "features.csv"
MODEL_RF_PATH = ROOT / "models" / "phishing_rf.joblib"
MODEL_XGB_PATH = ROOT / "models" / "phishing_xgb.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
METRICS_V1_PATH = ROOT / "models" / "metrics_v1.json"

SEED_SPLIT1 = 42  # same as v1 -> same grouped rows
SEED_SPLIT2 = 43
SEED_MODEL = 42
RECALL_TARGET = 0.92  # safety margin: v1 lost ~0.015 val->test

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


def evaluate(name: str, clf, X_val, y_val, X_test, y_test) -> dict:
    """Default + val-tuned test metrics for one fitted model."""
    default_m = metrics_at(y_test, clf.predict(X_test))
    print_metrics(f"{name} test @0.50", default_m)
    tuned = tune_threshold(clf.predict_proba(X_val)[:, 1], y_val, name)
    val_m = metrics_at(y_val, (clf.predict_proba(X_val)[:, 1] >= tuned))
    print_metrics(f"{name} val @{tuned:.2f}", val_m)
    test_pred = (clf.predict_proba(X_test)[:, 1] >= tuned).astype(int)
    tuned_m = metrics_at(y_test, test_pred)
    print_metrics(f"{name} test @{tuned:.2f}", tuned_m)
    cm = confusion_matrix(y_test, test_pred)
    print(f"{name} confusion matrix test @{tuned:.2f}:")
    print(cm)
    return {
        "threshold": tuned,
        "default_threshold_test": {"threshold": 0.5, **default_m},
        "tuned_val": {"threshold": tuned, **val_m},
        "tuned_threshold_test": {"threshold": tuned, **tuned_m},
        "confusion_matrix_test_tuned": cm.tolist(),
    }


def main() -> None:
    v1 = json.loads(METRICS_V1_PATH.read_text()) if METRICS_V1_PATH.exists() \
        else None
    if v1 is None:
        print("WARNING: models/metrics_v1.json missing; v1 block will be null")

    data = load_or_build_features()
    print(balance_text("full data", data["label"].values))

    X = data[FEATURE_NAMES].to_numpy(dtype=float)
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
    rf_res = evaluate("RF", rf, X_val, y_val, X_test, y_test)
    rf_imp = sorted(zip(FEATURE_NAMES, rf.feature_importances_),
                    key=lambda kv: kv[1], reverse=True)
    print("RF top 10 feature importances:")
    for rank, (name, imp) in enumerate(rf_imp[:10], start=1):
        print(f"  {rank:2d}. {name:<20s} {imp:.4f}")
    rf_res["feature_importances_top10"] = [
        {"feature": n, "importance": float(i)} for n, i in rf_imp[:10]]

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        scale_pos_weight=neg / pos,
        random_state=SEED_MODEL,
        tree_method="hist",
        n_jobs=-1,
    )
    t0 = time.time()
    xgb.fit(X_train, y_train)
    print(f"trained XGBoost in {time.time() - t0:.1f}s")
    xgb_res = evaluate("XGB", xgb, X_val, y_val, X_test, y_test)
    gain = xgb.get_booster().get_score(importance_type="gain")
    xgb_imp = sorted(
        ((FEATURE_NAMES[int(k[1:])], v) for k, v in gain.items()),
        key=lambda kv: kv[1], reverse=True)
    print("XGB top 10 feature importances (gain):")
    for rank, (name, imp) in enumerate(xgb_imp[:10], start=1):
        print(f"  {rank:2d}. {name:<20s} {imp:.4f}")
    xgb_res["feature_importances_top10"] = [
        {"feature": n, "importance": float(i)} for n, i in xgb_imp[:10]]

    # Winner: test recall >= 0.90 first, then higher test F1.
    scored = {
        "rf": (rf_res["tuned_threshold_test"]["recall_phish"],
               rf_res["tuned_threshold_test"]["f1_phish"]),
        "xgb": (xgb_res["tuned_threshold_test"]["recall_phish"],
                xgb_res["tuned_threshold_test"]["f1_phish"]),
    }
    ok = [k for k, (r, _) in scored.items() if r >= 0.90]
    pool = ok if ok else list(scored)
    selected = max(pool, key=lambda k: scored[k][1])
    print(f"selected model: {selected} "
          f"(test recall/F1: {scored[selected][0]:.4f}/"
          f"{scored[selected][1]:.4f})")

    winner, winner_path = (rf, MODEL_RF_PATH) if selected == "rf" \
        else (xgb, MODEL_XGB_PATH)
    loser_path = MODEL_XGB_PATH if selected == "rf" else MODEL_RF_PATH
    winner_thr = rf_res["threshold"] if selected == "rf" \
        else xgb_res["threshold"]
    joblib.dump({"model": winner, "threshold": float(winner_thr),
                 "feature_names": FEATURE_NAMES,
                 "random_state": SEED_MODEL},
                winner_path, compress=3)
    size_mb = winner_path.stat().st_size / (1024 * 1024)
    print(f"saved {winner_path} ({size_mb:.2f} MB)")
    if loser_path.exists():
        loser_path.unlink()
        print(f"removed stale {loser_path}")
    (rf_res if selected == "rf" else xgb_res)["model_size_mb"] = \
        round(size_mb, 2)

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
        "v1": v1,
        "v2": {"rf": rf_res, "xgb": xgb_res},
        "selected": selected,
        "model_file": winner_path.name,
        "model_size_mb": round(size_mb, 2),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"saved {METRICS_PATH}")

    v1t = v1["tuned_threshold_test"] if v1 else {}
    r, x = rf_res["tuned_threshold_test"], xgb_res["tuned_threshold_test"]
    print("v1 vs v2 (@val-tuned threshold, test set):")
    print("model      thr     acc    prec   recall   f1")
    if v1t:
        print(f"v1 RF      {v1t['threshold']:.2f}  "
              f"{v1t['accuracy']:.4f} {v1t['precision_phish']:.4f} "
              f"{v1t['recall_phish']:.4f} {v1t['f1_phish']:.4f}")
    print(f"v2 RF      {r['threshold']:.2f}  "
          f"{r['accuracy']:.4f} {r['precision_phish']:.4f} "
          f"{r['recall_phish']:.4f} {r['f1_phish']:.4f}")
    print(f"v2 XGB     {x['threshold']:.2f}  "
          f"{x['accuracy']:.4f} {x['precision_phish']:.4f} "
          f"{x['recall_phish']:.4f} {x['f1_phish']:.4f}")


if __name__ == "__main__":
    main()
