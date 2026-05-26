# Member C Quickstart — Using the Sentiment Pipeline Output

This document tells Member C (and any downstream teammate) how to reproduce and consume Member B's sentiment features for LSTM/GRU price modelling.

## One-command reproduction

From the project root, after Member A's outputs are in place:

```bash
python sentiment/run_all.py
```

This runs all five steps end-to-end (~1–2 minutes on CPU; mostly FinBERT).

## Prerequisites

| File | Produced by | Required |
|------|-------------|----------|
| `data/processed/social_text_cleaned.csv` | Member A — `scraping/member_a_social_text_preprocessing.py` | always |
| `data/processed/stockx_price_cleaned_member_a.csv` | Member A — `scraping/member_a_stockx_cleaning.py` | step 4 |
| `vaderSentiment` | `pip install -r requirements.txt` | step 1 |
| `torch`, `transformers` | `pip install torch transformers` | step 2 |

If a prerequisite is missing, `run_all.py` prints a clear error before doing any work.

## Pipeline steps

| # | Script | Reads | Writes |
|---|--------|-------|--------|
| 1 | `sentiment/vader_scorer.py` | `social_text_cleaned.csv` | `data/sentiment/vader_scored_posts.csv` |
| 2 | `sentiment/finbert_scorer.py` | `social_text_cleaned.csv` | `data/sentiment/finbert_scored_posts.csv` |
| 3 | `sentiment/aggregate_sentiment.py` | both `*_scored_posts.csv` | `vader_daily_sentiment.csv`, `finbert_daily_sentiment.csv` |
| 4 | `sentiment/merge_sentiment_price.py` | daily files + price file | `data/processed/price_sentiment_merged.csv` |
| 5 | `sentiment/compare_vader_finbert.py` | both `*_scored_posts.csv` + daily files | `data/sentiment/comparison/*` |

All intermediate files share `date` (YYYY-MM-DD) as the join key.

## Common runs

```bash
# Full pipeline
python sentiment/run_all.py

# Skip FinBERT (much faster, but no comparison)
python sentiment/run_all.py --skip-finbert

# Skip the comparison only
python sentiment/run_all.py --skip-compare

# Re-run only specific steps (e.g. after fixing the merge logic)
python sentiment/run_all.py --only 4 5
```

## Your input file

**`data/processed/price_sentiment_merged.csv`** — 83,750 rows × 11 columns.

Schema:

```
date, shoe_name, sale_price, size, colorway,
vader_sentiment_mean, vader_sentiment_std, vader_tweet_count,
finbert_sentiment_mean, finbert_sentiment_std, finbert_tweet_count
```

Coverage: 32,834 / 83,750 (39.2 %) rows have a matching daily VADER signal.
The remaining 60.8 % carry NaN sentiment — Reddit's public search API does not return enough posts for the 2017–2019 price window. Forward-fill or zero-fill as you see fit; document the choice.

## Feature recommendation for LSTM/GRU

- **Use `vader_sentiment_mean`** as the sentiment feature.
- **Skip `finbert_sentiment_mean`**: FinBERT labels 93.7 % of sneaker Reddit posts as neutral (near-zero variance), so it adds little signal and may slow convergence.
- See [`member_b_vader_vs_finbert.md`](member_b_vader_vs_finbert.md) for the full comparison and reasoning.

## Handling NaN sentiment rows

Suggested approaches (pick one and document it):

1. **Forward-fill** — carry the last known sentiment forward into NaN gaps.
2. **Zero-fill** — treat missing as neutral (sentiment_mean = 0).
3. **Drop** — train only on the 32,834 rows that have sentiment (reduces dataset by 60 %).

Option 1 is generally safest for time-series models.

## Related documentation

- [`member_b_sentiment_notes.md`](member_b_sentiment_notes.md) — full notes on each step, schemas, and limitations.
- [`member_b_vader_vs_finbert.md`](member_b_vader_vs_finbert.md) — VADER vs FinBERT comparison report.

## Reproducibility

All output directories (`data/processed/`, `data/sentiment/`) are git-ignored. Outputs are regenerated from source by re-running the pipeline; the scripts are idempotent (existing files are overwritten).
