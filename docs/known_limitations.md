# Known limitations (measured, v3)

Headline metrics overstate real-world performance. Details below;
see also `docs/threshold_validation.md` and `docs/probe_set.md`.

## v2 vs v3 headline (un-augmented test set, thr 0.35)

| model | acc | prec | recall | F1 |
|---|---|---:|---:|---:|
| v2 RF | 0.8007 | 0.5407 | 0.9070 | 0.6775 |
| v3 RF | 0.7939 | 0.5314 | 0.9074 | 0.6703 |

## Deploy model (512 MB host budget)

The tracked `models/phishing_rf.joblib` is a smaller RandomForest
(`deploy-m`: 60 trees, `max_depth` 24, `min_samples_leaf` 10, thr 0.38)
trained on the same grouped splits/seeds/augmentation as v3 with the
same 0.92 val-recall tuning. The previous artifact is kept locally as
`models/phishing_rf_full.joblib` (git-ignored). Full numbers live in
`models/metrics.json` under `deploy`.

| config (trees/depth/leaf) | thr | size MB | RAM after load MB | latency ms | test recall | prec | F1 | meets all? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| current (64/24/5) | 0.37 | 18.38 | 216 | 18.9 | 0.9083 | 0.5497 | 0.6849 | — (reference) |
| deploy-s (60/20/20) | 0.38 | 7.45 | 166 | 14.8 | 0.9120 | 0.5010 | 0.6467 | no (F1 0.038 below current) |
| deploy-m (60/24/10) | 0.38 | 12.63 | 193 | 15.3 | 0.9091 | 0.5272 | 0.6673 | **yes — shipped** |
| deploy-l (100/24/5) | 0.37 | 28.77 | 256 | 23.1 | 0.9080 | 0.5356 | 0.6738 | no (RAM 256 > 250 MB) |

Rule: smallest config with test recall ≥ 0.90, F1 within 0.02 of
current (≥ 0.6649), and RAM after load ≤ 250 MB. File is 12.63 MB
(well under the 50 MB cap). The bare-host and short-path blind spots
below apply to the deploy model too — it uses the same features and
augmentation recipe.

## Bare-host slice (43,414 unique test hosts, never trained on)

| model | FP rate on bare-legit (n=30,708) | recall on bare-phishing (n=12,706) | slice prec / F1 |
|---|---|---|---|
| v2 | 0.9998 | 1.0000 | 0.2927 / 0.4529 |
| v3 | 0.0256 | 0.2262 | 0.7850 / 0.3512 |

v3 fixed bare-legit false positives but now misses most bare-phishing
hosts — the augmentation overcorrected toward "bare = legit".

## Probe results (90 hand-made URLs, sanity check only, never tuned on)

| model | well-known bare safe | well-known +path safe | phishing caught |
|---|---|---|---|
| v2 | 0/30 | 3/30 | 30/30 |
| v3 | 30/30 | 3/30 | 30/30 |

## Findings

(a) The dataset's legit and phishing rows come from different sources,
so part of the score reflects URL structure, not phishing intent.
Legit rows are mostly full URLs with paths while phishing feeds list
many bare domains — the model learned "no path = phishing" (v2 flagged
~100% of bare-legit test hosts), and after bare-host augmentation it
leans the other way on bare hosts (v3 catches only 23% of bare
phishing). The headline test numbers do not show this slice behaviour.

(b) Short single-segment paths like `/about` and `/login` score as
phishing even on well-known domains (`google.com/about` 0.677,
`github.com/about` 0.745, `wikipedia.org/about` 0.799, all above the
0.35 threshold). Structurally they resemble classic phishing paths, and
no host signal in the current 24 features overrides that. Tracked by
`tests/test_bare_hosts.py` (`xfail(strict=True)` until fixed).

## Real-world spot checks (not a benchmark)

One-off `/predict` probes (v3 model, thr 0.35), recorded 2026-10-08.
A handful of hand-picked URLs — sanity checks only, never tuned on.

- `www.amazon.in/` → model score 0.663 (high) with `strong_signals`
  false (fallback reason only); served `safe` solely via the allowlist
  override. Same trailing-slash/short-path false-positive pattern as
  `figma.com/about` (0.741) and `figma.com/design` (0.874).
- `amazon.in/` → model score 0.342, within 0.008 of the threshold:
  borderline-safe at model level, served via allowlist.
- `figma.com` (bare) → model score 0.048, comfortably safe — the
  suspected "figma borderline" did not reproduce on the bare domain;
  only pathed forms (`/about`, `/design`, `/login`) score high.
- `amazon.in.evil.com` → model path (`source=model`, score 0.536,
  brand-mismatch signal): exact-or-www matching correctly refuses the
  lookalike.

## More real-world spot checks (not a benchmark)

One-off `/predict` probes (v3 model, thr 0.35), recorded 2026-10-08.
Hand-picked URLs — sanity checks only, never tuned on.

- `amazon.in` — model false positive on `www.amazon.in/` (score 0.663,
  high, fallback-only so `strong_signals` false), fixed via the
  allowlist override. Now also exercises the verdict rule: high score
  with no strong reason shows Suspicious, never "Likely phishing".
- `muthootfinance.com` (bare) → score 0.030, safe — the suspected 0.61
  Suspicious did NOT reproduce on the bare, `www.`, or trailing-slash
  forms (all below 0.10). Only a `/login` path flags (0.736, strong
  lure word). Listed so the non-reproduction is on record.
- `chat.whatsapp.com/<invite-code>` → score ~0.38–0.49 depending on the
  code (medium, weak-only long-token/entropy reasons, `strong_signals`
  false) → Suspicious. A subdomain of an allowlisted domain, so it
  goes to the model by design.
- `flow.google.com/` → score 0.382 (medium, fallback-only) →
  Suspicious plus the borderline note (0.032 from the threshold). Bare
  `flow.google.com` scores 0.021, safe.
- Google Forms link (`docs.google.com/forms/d/e/.../viewform`) →
  score 0.886 (high) with six weak reasons and `strong_signals` false
  → Suspicious, never "Likely phishing". Long IDs and deep paths look
  "random" to lexical features even on trusted hosts.

`docs.google.com` is deliberately NOT allowlisted: forms are widely
abused for phishing, so every forms/docs URL goes to the model.
Subdomains of allowlisted domains (`chat.whatsapp.com`,
`flow.google.com`, `docs.google.com`) always go to the model by
design — only the exact host or `www.` ever matches.
