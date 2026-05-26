# Member B — Sentiment Analysis Notes

## Responsibility

Member B is responsible for sentiment analysis of sneaker-related social media text and merging those signals with the price data prepared by Member A. Scope: sentiment scoring (VADER + FinBERT), daily aggregation, and price–sentiment merge. No UI work, backtesting, or model training is included here.

---

## Input Files

| File | Source | Description |
|------|--------|-------------|
| `data/processed/social_text_cleaned.csv` | Member A | **1,967 real Reddit posts**, 2011–2026 |
| `data/processed/stockx_price_cleaned_member_a.csv` | Member A | 83,750 StockX price transactions, 2017–2019 |

### Social text schema (confirmed)

| Column | Type | Notes |
|--------|------|-------|
| date | YYYY-MM-DD | Reddit post date |
| text | string | Post title + body |
| likes | int | Upvote score |
| retweets | int | Always 0 (Reddit has no retweet concept) |
| source | string | `reddit/r/{subreddit}` |
| query | string | Search term used to find this post |
| shoe_name | string | Best-matching shoe slug |

---

## VADER Method (`sentiment/vader_scorer.py`)

**What is VADER?**
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based tool that assigns sentiment scores based on a hand-crafted lexicon plus punctuation/capitalization rules. It needs no training, no GPU, and runs in seconds. It is a standard baseline for social media text.

**How it works — step by step:**
1. Load `data/processed/social_text_cleaned.csv`.
2. For each post, call `SentimentIntensityAnalyzer().polarity_scores(text)`.
3. This returns four numbers:
   - `vader_neg` — negativity (0 to 1)
   - `vader_neu` — neutrality (0 to 1)
   - `vader_pos` — positivity (0 to 1)
   - `vader_compound` — overall score (–1 = very negative, +1 = very positive)
4. Append those four columns and export.

**Interpreting compound:**
- `> 0.05` → positive sentiment
- `< –0.05` → negative sentiment
- otherwise → neutral

**Observed results on our Reddit dataset:**
- Mean compound = **+0.317** (overall positive tone — typical for enthusiast communities)
- Min = –0.994, Max = +0.9998
- 1,967 posts scored with zero nulls

**Install:**
```bash
pip install vaderSentiment
```

**Run:**
```bash
python sentiment/vader_scorer.py
```

**Output:** `data/sentiment/vader_scored_posts.csv` — 1,967 rows

---

## FinBERT Method (`sentiment/finbert_scorer.py`)

**What is FinBERT?**
FinBERT is a BERT transformer model fine-tuned on financial news text (earnings calls, analyst reports, market news). It classifies text as *positive*, *neutral*, or *negative* with higher accuracy than VADER on financial/market language — but requires PyTorch and Hugging Face Transformers to run.

**How it works:**
1. Load `data/processed/social_text_cleaned.csv`.
2. Load the `ProsusAI/finbert` model via `transformers.pipeline`.
3. For each post, the model outputs a label + confidence.
4. Map labels to numeric values: `positive=+1`, `neutral=0`, `negative=–1`.
5. Export columns: `finbert_label` (string) and `finbert_score` (numeric –1/0/+1).

**Safe fallback when torch is not installed:**
The script checks for `transformers` / `torch` at runtime. If they are missing, it:
- Prints a clear warning with installation instructions.
- Saves `finbert_score = NaN` and `finbert_label = "unavailable"` for all rows.
- Does **not** crash silently or present fake scores.
- Still saves a valid CSV so the rest of the pipeline can run.

**Current status:** `transformers` and `torch` are NOT installed. All FinBERT scores are NaN.

**To enable real FinBERT scoring:**
```bash
pip install transformers torch
python sentiment/finbert_scorer.py   # re-run after install
```

Note: `torch` CPU-only is ~1 GB. GPU-accelerated is ~3–5 GB. On CPU, scoring 1,967 posts takes ~10–30 minutes depending on hardware.

**Output:** `data/sentiment/finbert_scored_posts.csv` — 1,967 rows

---

## Daily Aggregation Method (`sentiment/aggregate_sentiment.py`)

**What it does:**
Collapses one row per post into one row per calendar day.

**VADER aggregation:**
- Group by `date`.
- Compute `mean` and `std` of `vader_compound` → `sentiment_mean`, `sentiment_std`.
- Count posts → `tweet_count`.

**FinBERT aggregation:**
- Group by `date`.
- Compute `mean` and `std` of `finbert_score` (the –1/0/+1 numeric).
- `NaN` values are automatically excluded by pandas, so `tweet_count` is 0 for days where FinBERT was unavailable.

**Why `sentiment_std` is NaN for some days:**
Standard deviation is mathematically undefined for a single value. Days with exactly one post will always have `NaN` in `sentiment_std`. This is correct behaviour — not a bug.

**Run:**
```bash
python sentiment/aggregate_sentiment.py
```

**Outputs:**

| File | Rows | Columns |
|------|------|---------|
| `data/sentiment/vader_daily_sentiment.csv` | 1,507 unique dates | date, sentiment_mean, sentiment_std, tweet_count |
| `data/sentiment/finbert_daily_sentiment.csv` | 1,507 unique dates | date, sentiment_mean, sentiment_std, tweet_count |

Of the 1,507 daily rows, 1,127 have `NaN` sentiment_std (those dates had only one post each).

---

## Price–Sentiment Merge (`sentiment/merge_sentiment_price.py`)

**What it does:**
Joins daily sentiment onto the per-transaction price data using a **left join on date**.

**Why left join?**
- Keeps all 83,750 price rows — no data loss.
- Rows on dates with no matching sentiment get `NaN` in sentiment columns.
- Explicit NaN is more honest than dropping or zero-filling.

**Run:**
```bash
python sentiment/merge_sentiment_price.py
```

**Output:** `data/processed/price_sentiment_merged.csv` — 83,750 rows

**Merge quality (actual results):**

| Metric | Value |
|--------|-------|
| Total price rows | 83,750 |
| Price rows WITH VADER sentiment | 32,834 (39.2%) |
| Price rows WITHOUT any sentiment | 50,916 (60.8%) |
| Price rows WITH FinBERT sentiment | 0 (FinBERT not installed) |
| Unique price dates | 531 |
| Unique VADER daily dates | 1,507 |
| Overlapping dates | 243 of 531 price dates |

**Why 60.8% of price rows are NaN in sentiment:**
The Reddit API is biased toward more-recent content. Most Reddit posts about Yeezy and Off-White from 2017–2019 were not surfaced by the public search endpoint, which has limited historical reach. Only 243 of the 531 price dates (2017-09-01 → 2019-02-13) have a matching Reddit post.

**Output schema:**

| Column | Source |
|--------|--------|
| date | Price data |
| shoe_name | Price data |
| sale_price | Price data |
| size | Price data |
| colorway | Price data |
| vader_sentiment_mean | VADER daily |
| vader_sentiment_std | VADER daily |
| vader_tweet_count | VADER daily |
| finbert_sentiment_mean | FinBERT daily (NaN until installed) |
| finbert_sentiment_std | FinBERT daily (NaN until installed) |
| finbert_tweet_count | FinBERT daily |

---

## All Output Files

| File | Rows | Real/Demo | Notes |
|------|------|-----------|-------|
| `data/sentiment/vader_scored_posts.csv` | 1,967 | **Real Reddit** | 0 nulls, 0 dupes |
| `data/sentiment/finbert_scored_posts.csv` | 1,967 | Real input, NaN scores | Install torch to fix |
| `data/sentiment/vader_daily_sentiment.csv` | 1,507 | **Real** | sentiment_std NaN on single-post days |
| `data/sentiment/finbert_daily_sentiment.csv` | 1,507 | NaN means | Correct until FinBERT installed |
| `data/processed/price_sentiment_merged.csv` | 83,750 | **Real** | 39.2% rows have VADER signal |

> All `data/sentiment/` and `data/processed/` files are in `.gitignore` and are NOT committed. Re-generate by running the four scripts in order.

---

## Limitations

1. **Date gap**: Reddit posts span 2011–2026; the StockX price data is 2017–2019. Only 243 of 531 price dates overlap. The merged file has NaN sentiment for 60.8% of rows.

2. **Reddit ≠ Twitter**: Reddit posts are longer, community-moderated, and less time-sensitive than tweets. The sentiment signal may lag real market events by hours or days.

3. **FinBERT unavailable**: `torch` and `transformers` are not installed. All FinBERT scores are NaN. The pipeline structure is fully implemented and ready — just needs the install.

4. **Sentiment std NaN on single-post days**: 1,127 of 1,507 daily rows have only one post, making `sentiment_std` mathematically undefined. This improves with denser social data.

5. **Reddit upvotes ≠ reach**: A post with score 1,000 was upvoted by many users over its lifetime, but this does not mean it was seen by 1,000 people in a single day.

---

## What Member C Can Use Next

- **`data/processed/price_sentiment_merged.csv`** — the main multivariate DataFrame for LSTM/GRU training. Contains all 83,750 price rows. For rows with sentiment, use `vader_sentiment_mean` as the sentiment feature. Replace with `finbert_sentiment_mean` once FinBERT is installed.
- **`data/sentiment/vader_daily_sentiment.csv`** — daily time series for visualisation in the Streamlit dashboard.
- **VADER vs FinBERT comparison** — once FinBERT is installed, compute the Pearson correlation between `vader_sentiment_mean` and `finbert_sentiment_mean` to measure agreement.

### Recommended next steps for Member C

1. `pip install transformers torch` → re-run `finbert_scorer.py`, `aggregate_sentiment.py`, `merge_sentiment_price.py`.
2. Load `price_sentiment_merged.csv` for model input. Use `vader_sentiment_mean` as an additional feature alongside `sale_price`.
3. For Streamlit: plot `vader_daily_sentiment.csv` as a line chart overlaid on price. Days with high positive sentiment but dropping price = potential buy signal.
