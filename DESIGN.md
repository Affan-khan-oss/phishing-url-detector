# DESIGN.md — One-page Checker UI (Next.js + TypeScript + Tailwind)

## 1. Goal

One page: paste URL → `POST /predict` → verdict + signals.
Beginner-readable, small, calm tone. No login, no dashboard, no router.

## 2. API contract (real, per `backend/main.py`)

`POST {NEXT_PUBLIC_API_URL, default http://localhost:8000}/predict`
Req: `{ "url": string }`
Res:

```json
{
  "label": "safe" | "phishing",
  "score": 0.0,
  "threshold": 0.35,
  "risk_level": "low" | "medium" | "high",
  "reasons": ["string"],
  "strong_signals": true,
  "disclaimer": "string",
  "source": "model" | "allowlist",
  "override": false
}
```

`strong_signals` is true when at least one strong reason fired on the
model path; the fallback text and allowlist routing notes never count.

Reason strength (`ml/features.py`, `reason_strength()`; feature values
and `FEATURE_NAMES` unchanged):

- Strong: IP address as host, `@`, punycode, urgent lure words
  (verify/login), trusted brand in a different domain, suspicious TLD.
- Weak: URL/host/path length, many digits/dots/specials, digit/letter
  ratios, entropy, long random token, subdomain count, path depth,
  query params, risky extension, percent-encoding.

Errors: `422 + {detail}` on garbage URL; `413` on body > 4KB; network failure / timeout = API down.

Rules:

- Render `disclaimer` verbatim from API.
- Never render any URL from the API as `<a href>`. Plain `<p>/<code>` only.
- Never display score as "% probability". Only as "Risk score" bar (`0.72 / 1.00`).
- Fetch has a 10s timeout (`AbortController`); on timeout show the friendly timeout error below.

## 3. Layout (mobile-first, single column, `max-w-xl mx-auto`)

```text
<Header>           title + 1-line subhead
<main>
  <UrlForm>        <label> + input + Check button (in <form>)
  <Examples>       4 <button type="button">, shown as text, fills input
  <ResultCard>     states below
</main>
<Footer>           Limitations section (static copy, §8)
```

Header copy: "Phishing URL Checker" / "Paste a URL to check lexical risk signals. Experimental — not a safety guarantee."

Examples are `<button type="button">`, never `<a href>`. Click fills the input (does not auto-submit). Captions below are what the API really returned via one-off `TestClient` probe (v3 model, threshold 0.35; exact score may shift slightly after retraining, shape is stable):

1. `google.com` [allowlist]
   → `label=safe, source=allowlist, override=false, risk_level=low, score~0.03`.
   Caption: "Known safe site — allowlist, low risk."
2. `github.com/about` [allowlist, safe domain with a short path]
   → `label=safe, source=allowlist, override=true, risk_level=low, score~0.75`.
   Caption: "Known safe site — allowlist (model scored 0.75, overridden). Shows the short-path edge case."
3. `example.com/about/team/contact-us` [model-served safe]
   → `label=safe, source=model, override=false, risk_level=low, score~0.21`.
   Caption: "Model safe — low risk, longer path."
4. `http://paypal-login-secure-update.tk/signin` [phishing-style]
   → `label=phishing, source=model, override=false, risk_level=high, score~0.90`.
   Caption: "Likely phishing — high risk (urgent lure + cheap TLD)."

## 4. States + copy

All states live in `<ResultCard>` region (`aria-live="polite"`). Form stays mounted.

- **empty** (initial): muted panel: "No check yet. Enter a URL above and press Check."
- **loading**: input + button disabled (`Checking…`, `aria-busy="true"`), skeleton bar + "Checking URL…". Abort previous fetch on resubmit.
- **allowlist** (`source == "allowlist"`, any `override`): badge `ℹ Known safe site (allowlist)`. Sub: "This domain is well-known, so the verdict comes from the allowlist." If `override == true`, add second line: "model score was overridden." Always show score bar + model score for transparency. `risk_level` is `low` by contract on this path.
- **model low** (`source == "model"`, `risk_level == "low"`): badge `✓ Looks safe`. Sub: "No strong phishing signals found. Still verify the sender before entering credentials."
- **model high with strong signals** (`source == "model"`, `label == "phishing"`, `risk_level == "high"`, `strong_signals == true`): badge `⚠ Likely phishing`. Sub: "This URL has patterns strongly associated with phishing. Don't enter credentials. Verify via a known source." Calm: contained card only, no full-screen red. "Likely phishing" appears only here.
- **model phishing otherwise** (medium risk, or high risk without strong signals): badge `⚠ Suspicious` — with strong signals: "Suspicious: some phishing-like patterns" ("This URL has patterns often seen in phishing. Don't enter credentials. Verify via a known source."); without: "Suspicious: flagged by the overall pattern, no single strong signal" ("The model flagged the overall pattern but found no single strong signal. Don't enter credentials. Verify via a known source.")
- **borderline note** (any result): when `|score − threshold| ≤ 0.05`, show a small "Borderline: score is close to the threshold." line under the score bar.
- **safe with signals**: when `label == "safe"` and the list is non-empty, title it "Signals found in the URL (not enough to flag)".
- **email-looking input** (`name@domain.tld`, no slash, no scheme): show above the result "This looks like an email address, not a URL. This tool checks links, so treat this result as not meaningful." Never block submission.
- **error-422**: badge `ⓘ Can't check that URL`. Sub: API `detail` message verbatim, e.g. "Enter a valid URL, e.g. example.com/login." Keep input value.
- **error-down**: badge `ⓘ Checker unavailable`. Sub: "Can't reach the API at {API_URL}. Start the backend (`uvicorn backend.main:app --port 8000`) and try again."
- **error-timeout** (fetch > 10s): badge `ⓘ Request timed out`. Sub: "The check took longer than 10 seconds. The backend may be busy — try again."

Color is never the only signal: every verdict has icon + text + risk_level word.

## 5. ResultCard spec

```text
[verdict badge: icon + heading + risk_level word]
Risk score [bar + "0.72 / 1.00 · threshold 0.35"]
[optional small note when |score − threshold| ≤ 0.05: "Borderline: score is close to the threshold."]
[optional email note when input looks like name@domain.tld: "This looks like an email address, ..."]
Signals found in the URL:   (or "Signals found in the URL (not enough to flag)" when label is safe)
  - <li> reason 1 …
Source note: "Source: model" / "Source: allowlist" [+ " — model score was overridden." if override]
<small> {disclaimer from API} </small>
<code> checked URL as plain text (no link) </code>
```

- Verdict order: allowlist → model low ("Looks safe") → model phishing + high risk + `strong_signals` ("Likely phishing") → any other model phishing ("Suspicious", copy varies by `strong_signals`).

- Title is "Signals found in the URL" — NOT "why the model decided". For a safe verdict with a non-empty list it becomes "Signals found in the URL (not enough to flag)".
- Score bar: `<div role="progressbar" aria-valuenow={score} aria-valuemin={0} aria-valuemax={1} aria-label="Risk score">` inner width `score*100%`, plus threshold tick at `threshold*100%`. Text alongside: `Risk score 0.72 / 1.00 · threshold 0.35`. No `% probability` anywhere.
- `reasons[]` empty → "No specific signals returned." (shouldn't happen — backend always returns ≥1 reason).
- `risk_level` shown as word (`low` / `medium` / `high`) alongside badge.
- Checked URL rendered in `<code>` / `<p>`, never `<a>`.

## 6. Components (keep small, `frontend/app/page.tsx` + few files)

- `Header.tsx` — static title + subhead.
- `UrlForm.tsx` — `<form onSubmit>`, `<label htmlFor="url-input">`, controlled input (`type="text" inputMode="url" autoComplete="off" spellCheck={false}`), submit on Enter natively, `Check` button (`type="submit"`).
- `ExampleChips.tsx` — 4 × `<button type="button">` chips from §3.
- `ResultCard.tsx` (+ tiny `ScoreBar` inside) — switch on states in §4.
- `Footer.tsx` — Limitations section (`id="limitations"`).
- `lib/api.ts` — `PredictResponse` TS type + one `predictUrl(url, { timeoutMs: 10000 })` fetch helper using `AbortController`. `API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"`.
- No router, no state lib; `useState` only.

## 7. Colors + typography

Tailwind only, `dark:` variants (`color-scheme: light dark`).
Base: light `bg-slate-50 text-slate-900`, card `bg-white border-slate-200`; dark `bg-slate-950 text-slate-100`, card `bg-slate-900 border-slate-800`.
- low / allowlist-safe: `emerald-700 on emerald-50 border-emerald-300` / dark `emerald-300 on emerald-950 border-emerald-800`
- medium / suspicious: `amber-800 on amber-50 border-amber-300` / dark `amber-200 on amber-950 border-amber-800`
- high / phishing: `rose-800 on rose-50 border-rose-300` (contained card, muted — no full-page red) / dark `rose-200 on rose-950 border-rose-800`
- allowlist info accent: `sky-800 on sky-50 border-sky-300` / dark `sky-200 on sky-950 border-sky-800`
- error/neutral: `slate-700 on slate-100 border-slate-300` / dark `slate-200 on slate-900 border-slate-700`
All pairs chosen for ≥4.5:1 contrast; verify in both modes.

Typography: system stack only (`font-sans`), no webfont. Scale: h1 `text-2xl font-semibold`, body `text-sm/base`, code/reasons `text-sm`, URL `font-mono`. One accent weight for badge.

Dark/light: respect OS via `dark:` + `color-scheme`. No manual toggle in v1.

## 8. Footer / Limitations (static copy from PRD + `docs/known_limitations.md`)

> **Limitations — experimental, URL text only.**
>
> - Checks URL text only; can't see page content, hosting, or new domains.
> - No HTTPS signal in v1; `http(s)://` is stripped before scoring.
> - Dataset bias: scores reflect URL structure as much as intent.
> - Headline metrics overstate reality: v3 gets ~0.79 acc / 0.91 recall at thr 0.35, but misses ~77% of bare-phishing hosts and flags `/about`, `/login` paths even on well-known domains.
> - Allowlist match means the domain is well-known; it does not guarantee this specific page is safe.
> - Not a safety guarantee. When in doubt, navigate via a known source.

## 9. UX / a11y checklist

- `<form onSubmit>` so Enter submits; button `type="submit"`, examples `type="button"`.
- Visible `<label>`, `aria-live="polite"` on result region, `aria-busy` while loading, focus moved to result heading on resolve.
- Touch targets ≥44px, single column stacks on <640px.
- Calm tone: no "DANGER", no fullscreen red, no % certainty claims.
- 10s timeout with distinct timeout copy (§4); abort stale requests.

## 10. Files / env

`frontend/app/page.tsx`, `frontend/components/{Header,UrlForm,ExampleChips,ResultCard,Footer}.tsx`, `frontend/lib/api.ts`. Env: `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`). `frontend/.env.example` documents it.
