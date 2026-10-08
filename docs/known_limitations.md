# Known limitations (measured, v3)

Headline metrics overstate real-world performance. Details below;
see also `docs/threshold_validation.md` and `docs/probe_set.md`.

## v2 vs v3 headline (un-augmented test set, thr 0.35)

| model | acc | prec | recall | F1 |
|---|---|---:|---:|---:|
| v2 RF | 0.8007 | 0.5407 | 0.9070 | 0.6775 |
| v3 RF | 0.7939 | 0.5314 | 0.9074 | 0.6703 |

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
