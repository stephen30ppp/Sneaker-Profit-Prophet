"""
Member A — Social Text Preprocessing
=====================================
Collects sneaker-related posts from Reddit's public JSON API
(no API key or paid subscription required) and cleans them into a
standard schema for Member B's sentiment analysis.

Run from project root:
    python scraping/member_a_social_text_preprocessing.py

What this script does
---------------------
1. Checks data/raw/ for any existing social/tweet CSV files.
2. If none found, fetches posts from Reddit's public JSON endpoint
   using multiple sneaker search terms.
3. Cleans and normalises all text into the required schema.
4. Exports to data/processed/social_text_cleaned.csv.
5. If all collection fails (network error, rate-limit), writes a
   clearly-labelled demo file instead and explains what to do next.

Output schema (social_text_cleaned.csv)
-----------------------------------------
  date       YYYY-MM-DD   post date
  text       string       post title + body (combined)
  likes      int          Reddit upvote score
  retweets   int          always 0 for Reddit (no retweet concept)
  source     string       "reddit"
  query      string       which search term found the post
  shoe_name  string       best-matching shoe name from search term

Why Reddit instead of Twitter/X
---------------------------------
Twitter API requires a paid developer account (Basic tier ~$100/month).
Reddit's JSON API is public, free, and needs no authentication.
All posts collected here are genuine user-generated text about sneakers.
"""

import sys
import time
import json
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import RAW_DIR, PROCESSED_DIR

# ── Configuration ─────────────────────────────────────────────────────────────

# Reddit's JSON API endpoint (public, no key required)
REDDIT_BASE = "https://www.reddit.com"

# User-Agent is required by Reddit to avoid 429 rate-limit blocks
USER_AGENT = "SneakerProfitProphet/1.0 (academic sentiment analysis project)"

# Subreddits most likely to have sneaker market discussion
SUBREDDITS = ["sneakers", "Sneakers", "streetwear", "adidas", "Nike"]

# Search terms aligned with the StockX Kaggle dataset (Yeezy + Off-White focus)
SEARCH_QUERIES = [
    ("Yeezy Boost 350",       "Adidas-Yeezy-Boost-350"),
    ("Yeezy Boost 700",       "Adidas-Yeezy-Boost-700"),
    ("Off-White Jordan 1",    "Nike-Air-Jordan-1-Retro-High-Off-White"),
    ("Nike Dunk resale",      "Nike-Dunk"),
    ("StockX sneaker market", "StockX"),
    ("sneaker resale profit", "sneaker-resale"),
    ("Off-White Nike collab", "Nike-Off-White"),
]

POSTS_PER_QUERY = 100   # Reddit API maximum per page
SLEEP_BETWEEN   = 1.5   # seconds between requests to respect rate limits
REQUEST_TIMEOUT = 12    # seconds

OUTPUT_REAL = PROCESSED_DIR / "social_text_cleaned.csv"
OUTPUT_DEMO = PROCESSED_DIR / "social_text_cleaned_DEMO.csv"


# ── Step 1 — Check for existing raw social CSV files ─────────────────────────

def find_existing_social_csv() -> pd.DataFrame | None:
    """
    Scan data/raw/ for CSV files that might contain social text data.
    Returns a cleaned DataFrame if found, None otherwise.
    """
    if not RAW_DIR.exists():
        return None

    candidate_files = list(RAW_DIR.glob("*.csv")) + list(RAW_DIR.glob("*.tsv"))
    social_files = []
    for f in candidate_files:
        name_lower = f.name.lower()
        # Skip the known StockX price file
        if "stockx" in name_lower and "contest" in name_lower:
            continue
        # Heuristic: file name suggests social/tweet/reddit content
        if any(kw in name_lower for kw in ["tweet", "reddit", "social", "text", "post", "comment"]):
            social_files.append(f)

    for path in social_files:
        print(f"Found candidate social file: {path.name}")
        try:
            df = pd.read_csv(path, nrows=5)
            cols = {c.lower().strip() for c in df.columns}
            has_text  = bool(cols & {"text", "tweet", "content", "title", "body", "selftext"})
            has_date  = bool(cols & {"date", "created_at", "timestamp", "created_utc"})
            if has_text and has_date:
                print(f"  → Looks like social data! Loading full file...")
                df_full = pd.read_csv(path)
                return normalise_raw_social_csv(df_full)
        except Exception as e:
            print(f"  Could not read {path.name}: {e}")

    return None


def normalise_raw_social_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map whatever columns exist in a raw social CSV to the standard schema.
    Supports common column-name variants used by Twitter exports, Reddit dumps, etc.
    """
    col_map = {c.lower().strip(): c for c in df.columns}

    def pick(candidates):
        for c in candidates:
            if c in col_map:
                return col_map[c]
        return None

    date_col     = pick(["date", "created_at", "timestamp", "created_utc"])
    text_col     = pick(["text", "tweet", "content", "title", "body", "selftext"])
    likes_col    = pick(["likes", "like_count", "score", "upvotes", "favorite_count"])
    retweets_col = pick(["retweets", "retweet_count", "reposts", "shares"])

    if not date_col or not text_col:
        raise ValueError("Cannot identify date/text columns in raw social CSV.")

    out = pd.DataFrame()
    out["date"]     = df[date_col]
    out["text"]     = df[text_col].fillna("").astype(str)
    out["likes"]    = pd.to_numeric(df[likes_col],    errors="coerce").fillna(0).astype(int) if likes_col    else 0
    out["retweets"] = pd.to_numeric(df[retweets_col], errors="coerce").fillna(0).astype(int) if retweets_col else 0
    out["source"]   = "raw_csv"
    out["query"]    = ""
    out["shoe_name"]= ""
    return out


# ── Step 2 — Reddit collection ───────────────────────────────────────────────

def _fetch_json(url: str) -> dict | None:
    """Fetch a URL and return parsed JSON, or None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} on {url}")
        return None
    except Exception as e:
        print(f"  Request failed: {e}")
        return None


def fetch_reddit_posts(query: str, shoe_name: str, subreddit: str) -> list[dict]:
    """
    Fetch up to POSTS_PER_QUERY posts matching `query` from one subreddit.

    Uses Reddit's public search JSON endpoint — no login needed.
    """
    url = (
        f"{REDDIT_BASE}/r/{subreddit}/search.json"
        f"?q={urllib.parse.quote(query)}"
        f"&sort=relevance&t=all&restrict_sr=1&limit={POSTS_PER_QUERY}"
    )

    data = _fetch_json(url)
    if not data:
        return []

    posts = []
    try:
        for child in data["data"]["children"]:
            d = child["data"]
            title    = d.get("title", "") or ""
            selftext = d.get("selftext", "") or ""
            # Skip deleted/removed posts
            if selftext in ("[deleted]", "[removed]"):
                selftext = ""
            combined = (title + " " + selftext).strip()
            if not combined:
                continue
            posts.append({
                "date":      datetime.fromtimestamp(d["created_utc"], tz=timezone.utc)
                                     .strftime("%Y-%m-%d"),
                "text":      combined,
                "likes":     int(d.get("score", 0)),
                "retweets":  0,
                "source":    f"reddit/r/{subreddit}",
                "query":     query,
                "shoe_name": shoe_name,
            })
    except (KeyError, TypeError) as e:
        print(f"  JSON parse error: {e}")

    return posts


def collect_from_reddit() -> pd.DataFrame | None:
    """
    Collect posts from Reddit across multiple queries and subreddits.
    Returns a cleaned DataFrame, or None if nothing was collected.
    """
    # urllib.parse imported here to keep imports near usage
    import urllib.parse  # noqa: F401 (registered above in fetch_reddit_posts)

    all_posts = []
    total_requests = len(SEARCH_QUERIES) * len(SUBREDDITS)
    done = 0

    print(f"\nCollecting from Reddit ({total_requests} requests, {SLEEP_BETWEEN}s between each)...")

    for query, shoe_name in SEARCH_QUERIES:
        for subreddit in SUBREDDITS:
            done += 1
            print(f"  [{done}/{total_requests}] r/{subreddit} ← '{query}'", end=" ... ", flush=True)
            posts = fetch_reddit_posts(query, shoe_name, subreddit)
            print(f"{len(posts)} posts")
            all_posts.extend(posts)
            time.sleep(SLEEP_BETWEEN)

    if not all_posts:
        print("No posts collected from Reddit.")
        return None

    df = pd.DataFrame(all_posts)
    print(f"\nRaw collected: {len(df):,} posts (before deduplication)")
    return df


# ── Step 3 — Cleaning ────────────────────────────────────────────────────────

def clean_social_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalise a raw social DataFrame into the final schema.

    Steps:
      1. Parse date → YYYY-MM-DD, drop unparseable rows
      2. Remove empty text
      3. Remove duplicate text
      4. Ensure likes/retweets are non-negative integers
      5. Sort by date ascending
    """
    print("\n--- Cleaning ---")
    n0 = len(df)

    # 1. Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    bad_dates  = df["date"].isna().sum()
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    print(f"  Dropped invalid dates     : {bad_dates}")

    # 2. Remove empty text
    df["text"] = df["text"].fillna("").str.strip()
    # Also remove very short text (< 10 chars — likely noise)
    empty = (df["text"].str.len() < 10).sum()
    df = df[df["text"].str.len() >= 10]
    print(f"  Dropped empty/short text  : {empty}")

    # 3. Remove exact duplicate text
    dupes = df.duplicated(subset=["text"]).sum()
    df = df.drop_duplicates(subset=["text"])
    print(f"  Dropped duplicate text    : {dupes}")

    # 4. Numeric coercion
    df["likes"]    = pd.to_numeric(df["likes"],    errors="coerce").fillna(0).clip(lower=0).astype(int)
    df["retweets"] = pd.to_numeric(df["retweets"], errors="coerce").fillna(0).clip(lower=0).astype(int)

    # 5. Sort
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  Rows before cleaning      : {n0:,}")
    print(f"  Rows after  cleaning      : {len(df):,}")
    print(f"  Date range                : {df['date'].min()} → {df['date'].max()}")
    print(f"  Unique dates              : {df['date'].nunique()}")

    return df


# ── Step 4 — Demo fallback ───────────────────────────────────────────────────

DEMO_ROWS = [
    ("2017-09-07", "Just copped Yeezy Boost 350 Moonrock on StockX! Prices rising fast, great investment opportunity", 45, 12, "demo", "Yeezy Boost 350", "Adidas-Yeezy-Boost-350-Low-Moonrock"),
    ("2017-09-15", "Yeezy market is absolutely on fire right now. Seeing huge premiums over retail everywhere", 89, 34, "demo", "Yeezy Boost 350", "Adidas-Yeezy-Boost-350"),
    ("2017-09-26", "Off-White Nike collab dropping soon, prices will explode. Already up 40 percent on secondary market", 67, 28, "demo", "Off-White Jordan", "Nike-Air-Jordan-1-Retro-High-Off-White"),
    ("2017-10-04", "Disappointed with StockX fees eating into profits. Margins getting tight on budget sneakers lately", 8, 2, "demo", "StockX sneaker market", "StockX"),
    ("2017-10-18", "Yeezy 350 Beluga just restocked. Resale value dropping because of oversupply problem", 15, 5, "demo", "Yeezy Boost 350", "Adidas-Yeezy-Boost-350-Low-Beluga"),
    ("2017-11-03", "StockX showing strong sales volume today. Sneaker market sentiment very bullish overall", 56, 21, "demo", "StockX sneaker market", "StockX"),
    ("2017-11-20", "Nike Off-White collab is overhyped. Too many fakes flooding the market, trust is low right now", 12, 4, "demo", "Off-White Nike collab", "Nike-Off-White"),
    ("2017-12-05", "Incredible day on StockX, sold three pairs of Yeezys at massive premium. Market is extremely hot", 102, 45, "demo", "Yeezy Boost 350", "Adidas-Yeezy-Boost-350"),
    ("2017-12-19", "Sneaker bubble concerns are rising. Prices unsustainably high for average collectors right now", 23, 9, "demo", "sneaker resale profit", "sneaker-resale"),
    ("2018-01-08", "New year new gains! Yeezy prices holding strong. Bullish on the resale market overall this year", 78, 31, "demo", "Yeezy Boost 350", "Adidas-Yeezy-Boost-350"),
    ("2018-01-22", "Off-White Nike prices crashed 20 percent in one week. Market correction happening right now suddenly", 19, 7, "demo", "Off-White Nike collab", "Nike-Off-White"),
    ("2018-02-10", "StockX platform improvements are great. Faster authentication, more buyer confidence in platform", 44, 16, "demo", "StockX sneaker market", "StockX"),
    ("2018-02-28", "Yeezy 700 Wave Runner just dropped, incredible hype. Prices already 3x retail on resale market", 133, 58, "demo", "Yeezy Boost 700", "Adidas-Yeezy-Boost-700-Wave-Runner"),
    ("2018-03-15", "Market sentiment turning negative. Too many Yeezy releases diluting the brand value significantly", 31, 11, "demo", "Yeezy Boost 350", "Adidas-Yeezy-Boost-350"),
    ("2018-04-02", "Off-White collab with Jordan Brand announcement. Prices going crazy on secondary market today", 97, 42, "demo", "Off-White Jordan 1", "Nike-Air-Jordan-1-Retro-High-Off-White"),
    ("2018-04-20", "Sneaker market crash fears overblown. Fundamentals still strong for premium brands like Yeezy", 52, 19, "demo", "sneaker resale profit", "sneaker-resale"),
    ("2018-05-10", "Bad experience with counterfeit Yeezys this week. Trust in StockX authentication needs improvement", 7, 2, "demo", "StockX sneaker market", "StockX"),
    ("2018-06-15", "Summer slowdown on sneaker resale market. Prices stable but volume declining across all platforms", 29, 8, "demo", "sneaker resale profit", "sneaker-resale"),
    ("2018-07-04", "Jordan 1 Off-White Chicago is the grail of this year. Ridiculous price premium expected at release", 88, 37, "demo", "Off-White Jordan 1", "Nike-Air-Jordan-1-Retro-High-Off-White-Chicago"),
    ("2018-08-20", "Adidas oversupply killing Yeezy resale margins. Too many pairs available, prices dropping fast", 18, 6, "demo", "Yeezy Boost 350", "Adidas-Yeezy-Boost-350"),
    ("2018-09-12", "Nike pulling back on Off-White collabs is great for scarcity. Bullish for existing pairs value", 63, 24, "demo", "Off-White Nike collab", "Nike-Off-White"),
    ("2018-10-08", "StockX hitting record transaction volume this week. Sneaker market still very much alive and kicking", 74, 30, "demo", "StockX sneaker market", "StockX"),
    ("2018-11-15", "Yeezy 350 Butter price holding surprisingly well. Strong collector demand at all price levels", 48, 17, "demo", "Yeezy Boost 350", "Adidas-Yeezy-Boost-350-Butter"),
    ("2018-12-01", "Holiday season boosting sneaker prices significantly. Great time to flip inventory for profit", 91, 39, "demo", "sneaker resale profit", "sneaker-resale"),
    ("2019-01-15", "New year speculation on Yeezy 350 colorways. Market positioning for big releases expected ahead", 55, 22, "demo", "Yeezy Boost 350", "Adidas-Yeezy-Boost-350"),
]

DEMO_COLS = ["date", "text", "likes", "retweets", "source", "query", "shoe_name"]


def create_demo_file() -> None:
    """Write the clearly-labelled demo fallback file."""
    df = pd.DataFrame(DEMO_ROWS, columns=DEMO_COLS)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DEMO, index=False)
    print(f"\n[DEMO] Saved {len(df)} demo rows → {OUTPUT_DEMO}")
    print("NOTE: This is NOT real social media data. It is a hand-crafted demo")
    print("      for pipeline testing only. Real data collection still required.")


# ── Main ─────────────────────────────────────────────────────────────────────

def print_quality_report(df: pd.DataFrame, label: str) -> None:
    print(f"\n=== Quality Report: {label} ===")
    print(f"  rows          : {len(df):,}")
    print(f"  columns       : {list(df.columns)}")
    print(f"  date range    : {df['date'].min()} → {df['date'].max()}")
    print(f"  unique dates  : {df['date'].nunique()}")
    print(f"  missing values:\n{df[['date','text','likes','retweets']].isnull().sum().to_string()}")
    print(f"  duplicate rows: {df.duplicated(subset=['text']).sum()}")
    print(f"  avg likes     : {df['likes'].mean():.1f}")
    print(f"  top queries   :")
    if "query" in df.columns:
        for q, n in df["query"].value_counts().head(5).items():
            print(f"    {n:4d}  {q}")


def main():
    import urllib.parse  # ensure available in scope used by fetch_reddit_posts

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Member A — Social Text Preprocessing")
    print("=" * 65)

    # ── Option A: existing raw social CSV ─────────────────────────────
    print("\n[Step 1] Checking data/raw/ for existing social CSV files...")
    df_existing = find_existing_social_csv()
    if df_existing is not None:
        df_clean = clean_social_df(df_existing)
        df_clean.to_csv(OUTPUT_REAL, index=False)
        print_quality_report(df_clean, "social_text_cleaned.csv (from raw CSV)")
        print(f"\n[OK] Saved real data → {OUTPUT_REAL}")
        return

    print("  No existing social CSV found in data/raw/.")

    # ── Option B: Reddit collection ───────────────────────────────────
    print("\n[Step 2] Attempting Reddit collection...")
    try:
        import urllib.parse  # noqa: F811
        df_raw = collect_from_reddit()
    except Exception as e:
        print(f"Reddit collection raised an unexpected error: {e}")
        df_raw = None

    if df_raw is not None and len(df_raw) > 0:
        df_clean = clean_social_df(df_raw)

        if len(df_clean) >= 10:
            df_clean.to_csv(OUTPUT_REAL, index=False)
            print_quality_report(df_clean, "social_text_cleaned.csv (from Reddit)")
            print(f"\n[OK] Real social data saved → {OUTPUT_REAL}")

            # Important date-alignment note
            price_start, price_end = "2017-09-01", "2019-02-13"
            social_start = df_clean["date"].min()
            social_end   = df_clean["date"].max()
            print("\n[DATE ALIGNMENT NOTE]")
            print(f"  StockX price data  : {price_start} → {price_end}")
            print(f"  Reddit posts       : {social_start} → {social_end}")
            overlap = df_clean[(df_clean["date"] >= price_start) &
                               (df_clean["date"] <= price_end)]
            print(f"  Overlapping rows   : {len(overlap):,} / {len(df_clean):,}")
            if len(overlap) < len(df_clean) * 0.1:
                print("  WARNING: Minimal date overlap with price data.")
                print("  The price–sentiment merge will have many NaN rows.")
                print("  This is expected: Reddit's public API returns recent posts.")
                print("  Real impact: sentiment analysis pipeline is validated on")
                print("  genuine text; the merged DataFrame will need real historical")
                print("  social data for full predictive modelling.")
            return
        else:
            print(f"  Only {len(df_clean)} clean rows collected — too few. Falling back to demo.")

    # ── Demo fallback ─────────────────────────────────────────────────
    print("\n[Step 3] All real collection methods exhausted. Creating demo file.")
    create_demo_file()
    print(f"\nACTION REQUIRED:")
    print(f"  Real social data is still needed at: {OUTPUT_REAL}")
    print(f"  Options:")
    print(f"    A) Place a raw social CSV in data/raw/ and re-run this script.")
    print(f"    B) Obtain historical Reddit data via Pushshift or academic datasets.")


if __name__ == "__main__":
    main()
