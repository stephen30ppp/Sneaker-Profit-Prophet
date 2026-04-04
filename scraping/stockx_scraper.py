"""
StockX Historical Transaction Scraper

Collects historical sale prices, dates, and shoe metadata
from StockX for target sneaker models.
"""
import pandas as pd
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
    # TODO: Implement with selenium or requests
    raise NotImplementedError


def load_kaggle_dataset(filepath: str) -> pd.DataFrame:
    """
    Alternative: Load pre-existing StockX dataset from Kaggle.
    Dataset: https://www.kaggle.com/datasets/hudsonstuck/stockx-data-contest

    Args:
        filepath: Path to downloaded CSV

    Returns:
        Cleaned DataFrame
    """
    df = pd.read_csv(filepath)
    return df


def save_raw_data(df: pd.DataFrame, shoe_name: str) -> Path:
    """Save scraped data to raw data directory."""
    output_path = RAW_DIR / f"{shoe_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    # Example usage
    print("StockX Scraper - run with target shoe URL")
