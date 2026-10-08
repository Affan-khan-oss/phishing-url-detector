# URL Phishing Detection System

Paste a URL, get a phishing verdict with human-readable reasons — powered by a scikit-learn model behind a FastAPI backend and a Next.js page.

## Run

Backend (from the repo root, needs `models/phishing_rf.joblib` — run `ml/train.py` first):

```bat
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

Frontend (from the repo root; API base from `NEXT_PUBLIC_API_URL`, default `http://localhost:8000`):

```bat
cd frontend
npm install
npm run dev
```

Open http://localhost:3000. See `frontend/.env.example` for the API URL variable.

