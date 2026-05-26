"""
Member A – StockX data cleaning pipeline.

Input : data/raw/StockX-Data-Contest-2019-3.csv
Outputs: data/processed/stockx_cleaned_all.csv
         data/processed/stockx_price_cleaned_member_a.csv
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT_DIR / "data" / "raw" / "StockX-Data-Contest-2019-3.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

CLEANED_ALL = PROCESSED_DIR / "stockx_cleaned_all.csv"
CLEANED_MEMBER_A = PROCESSED_DIR / "stockx_price_cleaned_member_a.csv"

# ── Column name mapping ────────────────────────────────────────────────────────
RENAME_MAP = {
    "Order Date": "date",
    "Brand": "brand",
    "Sneaker Name": "shoe_name",
    "Sale Price": "sale_price",
    "Retail Price": "retail_price",
    "Release Date": "release_date",
    "Shoe Size": "size",
    "Buyer Region": "buyer_region",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_currency(series: pd.Series) -> pd.Series:
    """Strip $ and commas then coerce to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[\$,]", "", regex=True).str.strip(),
        errors="coerce",
    )


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Raw rows       : {len(df)}")
    print(f"Raw columns    : {list(df.columns)}")

    # 1. Standardize column names
    df = df.rename(columns=RENAME_MAP)

    # 2. Strip leading/trailing whitespace from all string columns
    str_cols = [c for c in df.columns if df[c].dtype == "object"]
    for col in str_cols:
        df[col] = df[col].str.strip()

    # 3. Parse date – source format is M/D/YY (e.g. 9/1/17 → 2017-09-01)
    df["date"] = (
        pd.to_datetime(df["date"], format="%m/%d/%y", errors="coerce")
        .dt.strftime("%Y-%m-%d")
    )

    # 4. Parse currency columns
    df["sale_price"] = _parse_currency(df["sale_price"])
    df["retail_price"] = _parse_currency(df["retail_price"])

    # 5. Parse size to numeric
    df["size"] = pd.to_numeric(df["size"], errors="coerce")

    # 6. Add colorway (not present in source dataset)
    df["colorway"] = "Unknown"

    # ── Diagnostics before filtering ──────────────────────────────────────────
    critical_cols = ["date", "shoe_name", "sale_price", "retail_price", "size"]
    print(f"\nMissing values (before cleaning):")
    print(df[critical_cols].isnull().sum().to_string())

    dup_before = df.duplicated().sum()
    print(f"\nDuplicate rows (before removal): {dup_before}")

    # 7. Remove rows with missing critical values
    df = df.dropna(subset=critical_cols)

    # 8. Remove exact duplicates
    df = df.drop_duplicates()

    # 9. Remove invalid numeric values (zero or negative)
    df = df[(df["sale_price"] > 0) & (df["retail_price"] > 0) & (df["size"] > 0)]

    # 10. Sort by shoe_name then date
    df = df.sort_values(["shoe_name", "date"]).reset_index(drop=True)

    return df


def _report(df: pd.DataFrame, label: str) -> None:
    missing = df.isnull().sum()
    missing_nonzero = missing[missing > 0]
    dups = df.duplicated().sum()

    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  Rows             : {len(df)}")
    print(f"  Columns          : {list(df.columns)}")
    print(f"  Date range       : {df['date'].min()} → {df['date'].max()}")
    print(f"  Unique sneakers  : {df['shoe_name'].nunique()}")
    print(f"  Missing values   : {dict(missing_nonzero) if not missing_nonzero.empty else 'None'}")
    print(f"  Duplicates       : {dups}")


def main() -> None:
    if not RAW_FILE.exists():
        print(f"ERROR: raw file not found: {RAW_FILE}", file=sys.stderr)
        sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {RAW_FILE}\n")
    raw = load_raw(RAW_FILE)
    df = clean(raw.copy())

    # ── Output 1: full cleaned version for downstream members ─────────────────
    all_cols = [
        "date", "shoe_name", "sale_price", "retail_price",
        "size", "colorway", "brand", "release_date", "buyer_region",
    ]
    df_all = df[[c for c in all_cols if c in df.columns]].copy()
    df_all.to_csv(CLEANED_ALL, index=False)
    _report(df_all, "stockx_cleaned_all.csv")

    # ── Output 2: Member A price schema ───────────────────────────────────────
    # Drop duplicates again after narrowing columns – rows that differ only in
    # buyer_region / retail_price become identical in the reduced schema.
    member_a_cols = ["date", "shoe_name", "sale_price", "size", "colorway"]
    df_member_a = df[member_a_cols].drop_duplicates().reset_index(drop=True)
    df_member_a.to_csv(CLEANED_MEMBER_A, index=False)
    _report(df_member_a, "stockx_price_cleaned_member_a.csv")

    print(f"\nOutputs written to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
