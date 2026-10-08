# Threshold validation for `reasons_from_features`

## Method

- Data: `ml/data/clean.csv` (507,191 rows: 392,896 legit / 114,295 phishing).
- Random sample of **50,000 rows** with fixed seed **42** (38,691 legit / 11,309 phishing in-sample).
- Each URL featurized with `extract_features` from `ml/features.py`.
- For each of the 14 features: mean value per label, plus the share of rows
  where its reason threshold fires per label (legit vs phishing).

## Results

| feature | reason threshold | mean legit | mean phish | fire legit | fire phish |
|---|---|---:|---:|---:|---:|
| url_len | > 75 | 45.694 | 71.326 | 0.1039 | 0.2864 |
| host_len | > 30 | 15.823 | 21.928 | 0.0185 | 0.0997 |
| path_len | > 50 | 29.871 | 49.399 | 0.1533 | 0.2976 |
| num_dots | > 3 | 1.785 | 3.041 | 0.0299 | 0.2422 |
| num_hyphens | > 2 | 1.317 | 0.733 | 0.1777 | 0.0723 |
| num_underscores | ≥ 1 | 0.426 | 0.345 | 0.1976 | 0.1643 |
| num_digits | > 5 | 3.152 | 10.344 | 0.2535 | 0.3189 |
| num_specials | ≥ 3 | 0.530 | 1.421 | 0.0539 | 0.1644 |
| has_at | == 1 | 0.001 | 0.015 | 0.0005 | 0.0149 |
| has_ip | == 1 | 0.000 | 0.034 | 0.0001 | 0.0345 |
| num_subdomains | ≥ 3 | 0.361 | 0.781 | 0.0028 | 0.0660 |
| has_suspicious_word | == 1 | 0.019 | 0.294 | 0.0193 | 0.2945 |
| has_punycode | == 1 | 0.000 | 0.001 | 0.0000 | 0.0010 |
| has_percent_encoding | == 1 | 0.025 | 0.040 | 0.0245 | 0.0399 |

## Findings

1. **`num_hyphens` (> 2) fires more on legit than on phishing**
   (0.1777 vs 0.0723, diff +0.1054). Legit URLs in this dataset use
   hyphens more (mean 1.317 vs 0.733), so a "many hyphens" reason would
   mislead. Reason removed; feature kept in `FEATURE_NAMES`.
2. **`num_underscores` (≥ 1) fires more on legit than on phishing**
   (0.1976 vs 0.1643, diff +0.0333). Same action: reason removed,
   feature kept.

All other 12 reasons fire more on phishing, as intended. Strongest
phishing-side separators: `has_suspicious_word` (0.2945 vs 0.0193),
`num_dots` (0.2422 vs 0.0299), `url_len` (0.2864 vs 0.1039).
`has_punycode` is nearly absent in both classes — harmless but near-zero
signal on this sample.
