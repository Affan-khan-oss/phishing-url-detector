# AGENTS.md

Greenfield repo — no source, README, or build/test config yet (only `.gitignore` + `.venv/`).

## Verified facts

- Toolchain: Python 3.13.0 (see `.venv/pyvenv.cfg`), venv at `.venv/` (Windows: `.venv\Scripts\activate.bat`).
- Ignored via `.gitignore`: `.venv/`, `.env`, `__pycache__/`, `node_modules/`, `*.joblib`.
- `*.joblib` ignore suggests ML model artifacts are meant to stay out of git — keep it that way.

## Working rules

- No commands or entrypoints to preserve yet; do not assume pytest/ruff/black — check what gets added.
- When a stack, layout, or workflow is introduced, update this file with exact commands and entrypoints.
