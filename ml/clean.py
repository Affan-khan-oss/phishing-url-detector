"""Clean the raw phishing URL dataset.

Reads (never modified):  ml/data/phishing_site_urls.csv  (columns xURL, Label)
Writes:                  ml/data/clean.csv               (columns url, label)

Cleaning steps:
  1. Rename xURL -> url.
  2. Map labels: good -> 0 (legit), bad -> 1 (phishing).
  3. Strip leading/trailing whitespace from each URL.
  4. Drop exact duplicate rows.
  5. Drop ALL rows for URLs that have conflicting labels.

Run from the repo root:
  .venv\\Scripts\\python.exe ml/clean.py
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "ml" / "data" / "phishing_site_urls.csv"
CLEAN_PATH = ROOT / "ml" / "data" / "clean.csv"

LABEL_MAP = {"good": 0, "bad": 1}  # 0 = legit, 1 = phishing


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    raw_hash_before = sha256_of(RAW_PATH)

    df = pd.read_csv(RAW_PATH)
    raw_rows = len(df)

    # 1. Rename to the canonical column names.
    df = df.rename(columns={"xURL": "url", "Label": "label"})

    # 2. Map text labels to 0/1; fail loudly on anything unexpected.
    unknown = set(df["label"].unique()) - set(LABEL_MAP)
    if unknown:
        sys.exit(f"ERROR: unknown label values in raw data: {sorted(unknown)}")
    df["label"] = df["label"].map(LABEL_MAP).astype(int)

    # 3. Strip whitespace (scheme text itself is kept as-is).
    df["url"] = df["url"].astype(str).str.strip()

    # Drop unusable empty URLs after stripping (reported below).
    empty_urls = int((df["url"] == "").sum())
    df = df[df["url"] != ""]

    # 4. Drop exact duplicate rows.
    dup_rows = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # 5. Drop ALL rows for URLs with conflicting labels.
    label_counts = df.groupby("url")["label"].nunique()
    conflicting_urls = label_counts[label_counts > 1].index
    conflict_rows = int(df["url"].isin(conflicting_urls).sum())
    df = df[~df["url"].isin(conflicting_urls)]

    df.to_csv(CLEAN_PATH, index=False)

    # Confirm the raw file was not modified.
    raw_hash_after = sha256_of(RAW_PATH)

    total_removed = raw_rows - len(df)
    print(f"raw rows:            {raw_rows}")
    print(f"clean rows:          {len(df)}")
    print(f"class balance:       label=0 (legit) {(df['label'] == 0).sum()}, "
          f"label=1 (phishing) {(df['label'] == 1).sum()}")
    print(f"rows removed:        {total_removed} "
          f"(exact duplicates: {dup_rows}, "
          f"conflicting-label rows: {conflict_rows} "
          f"across {len(conflicting_urls)} url(s), "
          f"empty urls: {empty_urls})")
    print(f"raw sha256 before:   {raw_hash_before}")
    print(f"raw sha256 after:    {raw_hash_after}")
    print(f"raw file unchanged:  {raw_hash_before == raw_hash_after}")
    print(f"wrote:               {CLEAN_PATH}")


if __name__ == "__main__":
    main()
