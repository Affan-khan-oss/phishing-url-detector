# RULES: AI coding on this project

AI must follow these rules. When in doubt, stop and ask. Written for a beginner codebase — keep things small and explicit.

## Golden rules (never break)

1. Feature extraction lives **only** in `ml/features.py` and is shared by training and the API. No feature logic anywhere else.
2. **Split train/test before fitting.** Group the split so **the same domain never appears in both sets** (e.g. group by registrable domain). Never fit on full data; use a fixed seed for reproducibility.
3. **Every prediction must return human-readable `reasons[]`.** Reasons derive from the same feature vector the model saw — no separate URL parsing in the API layer.

## Stack / layout

- Backend: FastAPI. Frontend: Next.js + TypeScript. ML: scikit-learn Random Forest (XGBoost is v2 only).
- Layout: `ml/features.py`, `ml/train.py`, `ml/data/` (dataset), `models/` (artifacts), `backend/main.py`, `frontend/`, `tests/`.
- Keep v1 flat and beginner-readable; don't add packages or folders without asking.

## Data / secrets (never commit)

- **Never commit datasets, models, or `.env` files.** Datasets live in `ml/data/`, model artifacts (`*.joblib`) in `models/`, secrets in `.env` — all stay local-only.
- No hardcoded absolute paths; model path configurable, default `models/phishing_rf.joblib`.

## Python / ML

- Use repo `.venv` (Python 3.13.0; Windows: `.venv\Scripts\activate.bat`).
- Report accuracy, precision, recall, and F1 on the held-out set every time you train; don't claim performance without a domain-grouped test split.

## API / frontend

- `POST /predict` request: `{url}`; response: `{label, probability, reasons[]}`. Validate input, return 422 + message on garbage URLs, never crash.
- Frontend: one page, one input, calls `POST /predict`, displays verdict + probability + reasons list. No login/dashboard state.

## AI workflow

- Small steps, one phase of `TASKS.md` at a time. Explain what/why before each change.
- Don't invent commands (no pytest/ruff/black until added). Update `AGENTS.md` when a real command or entrypoint is introduced.
