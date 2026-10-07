# PRD: URL Phishing Detection System

## Problem

Users can't reliably tell phishing URLs from legitimate ones by eye.
Goal: a local full-stack app where you paste a URL and get a risk verdict with human-readable reasons.

## User

- Beginner builder (you): learning Python, ML, FastAPI, Next.js end to end.
- End user: pastes a URL into a single web page, sees verdict + reasons.

## MVP (v1)

Stack: Python, scikit-learn (Random Forest; XGBoost is a v2 candidate), FastAPI backend, Next.js + TypeScript frontend.

Flow: dataset → features → model → `/predict` API → one Next.js page.

- FR1: Load a labeled URL dataset from `ml/data/` (CSV with `url,label`).
- FR2: Extract a fixed feature vector per URL via `ml/features.py` only. `extract_features` must strip `http://`/`https://` before computing features; `uses_https` is NOT a model feature because the dataset has almost no scheme information.
- FR3: Split train/test **before** fitting (grouped so the same domain never appears in both sets), train Random Forest, report held-out metrics.
- FR4: Save model artifact locally (e.g. `models/phishing_rf.joblib`, never committed).
- FR5: `POST /predict {url}` returns `{label, probability, reasons[]}` — `reasons[]` always present and human-readable.
- FR6: One Next.js + TypeScript page: input → calls API → shows verdict + probability + reasons.

## Out of scope for v1

Chrome extension, MongoDB, dashboard, login, XGBoost.

## Success criteria (v1 done when all true)

1. Training reports **accuracy, precision, recall, and F1** on the held-out test set.
2. Target: **phishing-class recall ≥ 0.90**. If the dataset makes this unrealistic, lower it and explain why (e.g. small/imbalanced data, noisy labels) — don't silently drop the target.
3. API and training import the same `extract_features` from `ml/features.py` — zero duplicated feature logic.
4. `/predict` on 5 known URLs returns plausible labels, each with ≥ 1 reason.
5. Frontend page works end-to-end against the local backend.
6. Fresh clone + instructions gets a second person (or future you) running in < 15 min.

## Limitations

- **URL-only features can miss new or short-lived phishing domains.** Lexical signals (length, symbols, keywords) don't see page content, hosting reputation, or freshly registered domains with clean-looking URLs.
- **Dataset bias can inflate accuracy.** If `ml/data/` over-represents certain brands, TLDs, or old campaigns, test scores look better than real-world performance. Treat metrics as comparative, not as a safety guarantee.
- **No HTTPS signal in v1.** `uses_https` is not a model feature: only 107 of 549k dataset URLs carry a scheme, so the model cannot learn anything reliable about HTTPS. `extract_features` strips `http://`/`https://` before computing features so predictions behave the same with or without a scheme.
