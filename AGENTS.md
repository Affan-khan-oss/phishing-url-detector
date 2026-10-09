# AGENTS.md

Greenfield repo — no source, README, or build/test config yet (only `.gitignore` + `.venv/`).

## Verified facts

- Toolchain: Python 3.13.0 (see `.venv/pyvenv.cfg`), venv at `.venv/` (Windows: `.venv\Scripts\activate.bat`).
- Ignored via `.gitignore`: `.venv/`, `.env`, `__pycache__/`, `node_modules/`, `*.joblib` (except the tracked deploy model via `!models/phishing_rf.joblib`).
- `*.joblib` ignore suggests ML model artifacts are meant to stay out of git — keep it that way.

## Working rules

- No commands or entrypoints to preserve yet; do not assume pytest/ruff/black — check what gets added.
- When a stack, layout, or workflow is introduced, update this file with exact commands and entrypoints.

## Commands (all from the repo root, Windows paths)

- Tests: `.\.venv\Scripts\python.exe -m pytest tests -v`
- Train: `.\.venv\Scripts\python.exe ml/train.py` (needs `ml/data/clean.csv`; writes `models/phishing_rf.joblib` + `models/metrics.json` (tracked)). Note: the tracked `models/phishing_rf.joblib` is the small deploy model (see `deploy` in `models/metrics.json`); `ml/train.py` overwrites it with the full-size model, and `models/phishing_rf_full.joblib` is the local-only backup.
- API: `.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000` (needs the trained `.joblib`; `MODEL_PATH` env overrides the model location)
