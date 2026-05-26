"""
Member B — FinBERT Sentiment Scorer

Transformer-based sentiment analysis fine-tuned on financial text.
Higher accuracy than VADER for finance/market language, but requires
transformers + torch (large install, ~3-5 GB with GPU support).

Run from project root:
    python sentiment/finbert_scorer.py

Input:
    data/processed/social_text_cleaned.csv      (real data — preferred)
    data/processed/social_text_cleaned_DEMO.csv  (demo fallback — for testing only)

Output:
    data/sentiment/finbert_scored_posts.csv
    Columns: date, text, likes, retweets, finbert_label, finbert_score

finbert_label: "positive" / "neutral" / "negative"
finbert_score: numeric mapping  positive=+1, neutral=0, negative=-1

If transformers/torch are not installed, scores are set to NaN and labels to
"unavailable". Install with:
    pip install transformers torch
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import PROCESSED_DIR, SENTIMENT_DIR

SOCIAL_FILE = PROCESSED_DIR / "social_text_cleaned.csv"
DEMO_FILE   = PROCESSED_DIR / "social_text_cleaned_DEMO.csv"
OUTPUT_FILE = SENTIMENT_DIR / "finbert_scored_posts.csv"

REQUIRED_COLS = {"date", "text", "likes", "retweets"}

LABEL_TO_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

# Maximum text length FinBERT can handle (in characters, safe estimate for BERT's 512 tokens)
MAX_TEXT_LEN = 512


def load_social_data() -> pd.DataFrame:
    """Load social text data, falling back to demo file if real data is missing."""
    if SOCIAL_FILE.exists():
        df = pd.read_csv(SOCIAL_FILE)
        print(f"[OK] Loaded real social data: {len(df):,} rows  ({SOCIAL_FILE})")
    elif DEMO_FILE.exists():
        print("\n" + "=" * 60)
        print("WARNING: social_text_cleaned.csv NOT FOUND.")
        print("Using DEMO data for pipeline testing only.")
        print("DEMO data is NOT real social media data.")
        print(f"Demo file: {DEMO_FILE}")
        print("=" * 60 + "\n")
        df = pd.read_csv(DEMO_FILE)
        print(f"[DEMO] Loaded {len(df)} demo rows.")
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


def _try_load_finbert():
    """
    Try to load the FinBERT pipeline.
    Returns the pipeline object if successful, or None if dependencies are missing.
    """
    try:
        from transformers import pipeline
        print("Loading FinBERT model (ProsusAI/finbert) — first run downloads ~500 MB...")
        pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        print("[OK] FinBERT model loaded.")
        return pipe
    except ImportError:
        return None
    except Exception as exc:
        print(f"WARNING: FinBERT model failed to load: {exc}")
        return None


def score_with_finbert(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score texts with FinBERT.

    If transformers/torch are not installed, all scores are set to NaN
    and labels to 'unavailable'. The output CSV is still valid — it just
    cannot be used for real sentiment analysis until FinBERT is installed.
    """
    pipe = _try_load_finbert()

    if pipe is None:
        print("\n" + "=" * 60)
        print("WARNING: transformers or torch is NOT installed.")
        print("FinBERT scoring is UNAVAILABLE.")
        print("All finbert_score values will be NaN (honest 'I don't know').")
        print("\nTo enable real FinBERT scoring, install:")
        print("    pip install transformers torch")
        print("Then re-run: python sentiment/finbert_scorer.py")
        print("=" * 60 + "\n")

        n = len(df)
        labels = ["unavailable"] * n
        scores = [float("nan")] * n
        return pd.DataFrame({"finbert_label": labels, "finbert_score": scores})

    # FinBERT available — score all texts
    print("Scoring texts with FinBERT (this may take a while on CPU)...")
    texts = df["text"].fillna("").str[:MAX_TEXT_LEN].tolist()

    labels = []
    scores = []
    for i, text in enumerate(texts):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(texts)}")
        result = pipe(text)[0]
        label = result["label"].lower()
        labels.append(label)
        scores.append(LABEL_TO_SCORE.get(label, float("nan")))

    return pd.DataFrame({"finbert_label": labels, "finbert_score": scores})


def main():
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_social_data()

    score_df = score_with_finbert(df)

    out = pd.concat(
        [df[["date", "text", "likes", "retweets"]].reset_index(drop=True), score_df],
        axis=1,
    )

    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")

    out.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] Saved {len(out):,} rows → {OUTPUT_FILE}")
    print("\nSample output:")
    print(out[["date", "finbert_label", "finbert_score"]].head(5).to_string(index=False))

    available = (out["finbert_score"].notna().sum())
    print(f"\nRows with real FinBERT scores: {available}/{len(out)}")
    if available == 0:
        print("NOTE: No real FinBERT scores. Install transformers + torch and re-run.")


if __name__ == "__main__":
    main()
