"""
Member B — VADER Sentiment Scorer

Rule-based sentiment analysis optimised for social media text.
Fast, no GPU required. Good baseline for comparison with FinBERT.

Run from project root:
    python sentiment/vader_scorer.py

Input:
    data/processed/social_text_cleaned.csv      (real data — preferred)
    data/processed/social_text_cleaned_DEMO.csv  (demo fallback — for testing only)

Output:
    data/sentiment/vader_scored_posts.csv
    Columns: date, text, likes, retweets, vader_neg, vader_neu, vader_pos, vader_compound
"""

import sys
from pathlib import Path

# Allow 'from config import ...' when running as: python sentiment/vader_scorer.py
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("\nERROR: vaderSentiment is not installed.")
    print("Fix it by running:  pip install vaderSentiment\n")
    sys.exit(1)

from config import PROCESSED_DIR, SENTIMENT_DIR

SOCIAL_FILE = PROCESSED_DIR / "social_text_cleaned.csv"
DEMO_FILE   = PROCESSED_DIR / "social_text_cleaned_DEMO.csv"
OUTPUT_FILE = SENTIMENT_DIR / "vader_scored_posts.csv"

REQUIRED_COLS = {"date", "text", "likes", "retweets"}


def load_social_data() -> pd.DataFrame:
    """Load social text data, falling back to demo file if real data is missing."""
    if SOCIAL_FILE.exists():
        df = pd.read_csv(SOCIAL_FILE)
        print(f"[OK] Loaded real social data: {len(df):,} rows  ({SOCIAL_FILE})")
        using_demo = False
    elif DEMO_FILE.exists():
        print("\n" + "=" * 60)
        print("WARNING: social_text_cleaned.csv NOT FOUND.")
        print("Using DEMO data for pipeline testing only.")
        print("DEMO data is NOT real social media data.")
        print(f"Demo file: {DEMO_FILE}")
        print("=" * 60 + "\n")
        df = pd.read_csv(DEMO_FILE)
        print(f"[DEMO] Loaded {len(df)} demo rows.")
        using_demo = True
    else:
        print("\nERROR: No social text data found.")
        print(f"  Expected real file : {SOCIAL_FILE}")
        print(f"  Expected demo file : {DEMO_FILE}")
        print("Member B requires social_text_cleaned.csv. It is missing.")
        sys.exit(1)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"ERROR: Input file is missing columns: {missing}")
        sys.exit(1)

    return df


def score_with_vader(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add VADER sentiment columns to the DataFrame.

    VADER scores each text and returns four values:
      vader_neg      — negative sentiment (0 to 1)
      vader_neu      — neutral  sentiment (0 to 1)
      vader_pos      — positive sentiment (0 to 1)
      vader_compound — overall score     (-1 to +1)
                       > 0.05  = positive
                       < -0.05 = negative
                       otherwise neutral
    """
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for text in df["text"].fillna(""):
        scores = analyzer.polarity_scores(str(text))
        results.append({
            "vader_neg":      scores["neg"],
            "vader_neu":      scores["neu"],
            "vader_pos":      scores["pos"],
            "vader_compound": scores["compound"],
        })
    return pd.DataFrame(results)


def main():
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_social_data()

    print("Scoring texts with VADER...")
    scores = score_with_vader(df)

    out = pd.concat(
        [df[["date", "text", "likes", "retweets"]].reset_index(drop=True), scores],
        axis=1,
    )

    # Normalise date format to YYYY-MM-DD
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")

    out.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] Saved {len(out):,} rows → {OUTPUT_FILE}")
    print("\nSample output:")
    print(out[["date", "vader_compound", "vader_pos", "vader_neg"]].head(5).to_string(index=False))
    print(f"\nCompound score stats:")
    print(f"  mean = {out['vader_compound'].mean():.4f}")
    print(f"  min  = {out['vader_compound'].min():.4f}")
    print(f"  max  = {out['vader_compound'].max():.4f}")


if __name__ == "__main__":
    main()
