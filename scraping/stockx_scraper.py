"""
StockX Historical Transaction Scraper

Collects historical sale prices, dates, and shoe metadata
from StockX for target sneaker models.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from config import RAW_DIR


def scrape_stockx(shoe_url: str, max_pages: int = 50) -> pd.DataFrame:
    """
    Scrape historical transaction data for a specific sneaker.

    Args:
        shoe_url: StockX product URL or slug
        max_pages: Maximum number of result pages to fetch

    Returns:
        DataFrame with columns: [date, price, size, shoe_name]
    """
    raise NotImplementedError("Live scraping disabled — use load_kaggle_dataset() instead")


def load_kaggle_dataset(filepath: str) -> pd.DataFrame:
    """
    Load and clean the StockX Data Contest 2019 dataset.

    Args:
        filepath: Path to downloaded CSV

    Returns:
        Cleaned DataFrame with normalized schema
    """
    df = pd.read_csv(filepath, encoding="utf-8-sig")

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    if "order_date" in df.columns:
        df["date"] = pd.to_datetime(df["order_date"], format="mixed")
    elif "date" not in df.columns:
        date_cols = [c for c in df.columns if "date" in c.lower()]
        if date_cols:
            df["date"] = pd.to_datetime(df[date_cols[0]], format="mixed")

    for col in ["sale_price", "retail_price"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[\$,]", "", regex=True)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "shoe_size" in df.columns:
        df["shoe_size"] = pd.to_numeric(df["shoe_size"], errors="coerce")

    df = df.dropna(subset=["sale_price", "date"])

    if "sale_price" in df.columns:
        df = df.rename(columns={"sale_price": "price"})

    if "sneaker_name" in df.columns:
        df = df.rename(columns={"sneaker_name": "shoe_name"})

    df = df.sort_values("date").reset_index(drop=True)

    df["price_premium"] = df["price"] - df.get("retail_price", 0)

    return df


def get_daily_prices(df: pd.DataFrame, shoe_name: str = None, brand: str = None) -> pd.DataFrame:
    """
    Aggregate transaction-level data into daily average prices.

    Args:
        df: Cleaned StockX DataFrame
        shoe_name: Filter to specific shoe (optional)
        brand: Filter to specific brand (optional)

    Returns:
        DataFrame with columns: [date, price, volume, price_std]
    """
    filtered = df.copy()

    if shoe_name:
        filtered = filtered[filtered["shoe_name"].str.contains(shoe_name, case=False, na=False)]
    if brand:
        filtered = filtered[filtered["brand"].str.strip().str.lower() == brand.strip().lower()]

    daily = (
        filtered.groupby(filtered["date"].dt.date)
        .agg(
            price=("price", "mean"),
            volume=("price", "count"),
            price_std=("price", "std"),
        )
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["price_std"] = daily["price_std"].fillna(0)

    return daily


def save_raw_data(df: pd.DataFrame, shoe_name: str) -> Path:
    """Save scraped data to raw data directory."""
    output_path = RAW_DIR / f"{shoe_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    from config import ROOT_DIR

    csv_path = ROOT_DIR / "StockX-Data-Contest-2019-3.csv"
    if csv_path.exists():
        df = load_kaggle_dataset(str(csv_path))
        print(f"Loaded {len(df)} transactions")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Brands: {df['brand'].unique().tolist() if 'brand' in df.columns else 'N/A'}")
        print(f"\nSample:\n{df.head()}")
    else:
        print(f"CSV not found at {csv_path}")
