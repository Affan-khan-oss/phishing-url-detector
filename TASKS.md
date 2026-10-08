# TASKS: URL Phishing Detection System (v1)

Phase-wise, beginner-sized. Do in order; don't skip ahead. No code until Phase 2.

## Phase 0 — Setup

- [ ] 0.1 Confirm dataset source (your CSV vs. a suggested public one).
- [ ] 0.2 Create layout: `ml/`, `ml/data/`, `models/`, `backend/`, `frontend/`, `tests/`. Done when: structure matches `RULES.md`.

## Phase 1 — Dataset

- [ ] 1.1 Place raw CSV in `ml/data/` (keep it unchanged) and run `ml/clean.py` → `ml/data/clean.csv` with columns `url` and `label` (0 = legit/good, 1 = phishing/bad); record row count, source, and label split, and document which original value meant what. Note: `extract_features` must strip `http://`/`https://` before computing features, and `uses_https` is NOT a model feature because the dataset has almost no scheme information. Done when: class balance is known (e.g. X legit / Y phishing).

## Phase 2 — Features

- [ ] 2.1 Implement `extract_features(url)` in `ml/features.py` only (~8–12 lexical features: length, `@`, `-`, dots, subdomains, IP-in-host, https, suspicious words, etc.), plus a feature-names list and a `reasons_from_features()` helper in the same file. Done when: one function powers both training and explanations.
- [ ] 2.2 Add `tests/test_features.py`: same URL always gives same vector; feature order/length stable; spot-check 2–3 known URLs (e.g. IP-in-host flagged, reasons non-empty). Done when: tests pass.

## Phase 3 — Model

- [ ] 3.1 Write `ml/train.py`: load CSV from `ml/data/` → featurize via `ml/features.py` → domain-grouped train/test split (same domain never in both) → train Random Forest → print accuracy, precision, recall, F1 → save `models/phishing_rf.joblib`. Done when: metrics come from the held-out set and phishing recall is reported against the ≥ 0.90 target (or adjusted with a written reason).

## Phase 4 — API

- [ ] 4.1 FastAPI `POST /predict` in `backend/main.py`: load `.joblib` once, reuse `extract_features` + reasons helper, return `{label, probability, reasons[]}` with 422 on invalid URLs; enable CORS for `http://localhost:3000` and enforce a URL length limit (2048 chars) returning 422. Request bodies over 4 KB are rejected (413). Note — before deploy: add per-IP rate limiting. Done when: good/bad/garbage/oversized URLs all return sensible responses.
- [ ] 4.2 Add `tests/test_api.py`: valid URL returns label + probability + non-empty reasons; garbage URL returns 422; training-only feature logic is not duplicated in the API. Done when: tests pass.

## Phase 5 — Frontend

- [ ] 5.1 One Next.js + TS page: URL input → loading state → verdict + probability + reasons. Done when: works against local `POST /predict`.

## Phase 6 — End-to-end

- [ ] 6.1 Manual check: 5 test URLs through the UI; record results. Done when: all 5 return a verdict with reasons.
- [ ] 6.2 Write run instructions (backend + frontend commands) and update `AGENTS.md` with the exact commands. README must say where to download the dataset and where to place it (`ml/data/`).

## Explicitly not in v1

Chrome extension, MongoDB, dashboard, login, XGBoost.
