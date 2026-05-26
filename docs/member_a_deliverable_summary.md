# Member A – Deliverable Summary

## Scope

Member A's responsibility is data collection and preprocessing. No modelling, sentiment scoring, backtesting, or UI work is included.

---

## Deliverables

### 1. Cleaning Script

**File:** `scraping/member_a_stockx_cleaning.py`

Loads the raw Kaggle StockX CSV, applies all cleaning steps, and exports two processed CSVs. Run with:

```bash
python scraping/member_a_stockx_cleaning.py
```

### 2. Processed Data

| File | Rows | Schema |
|------|------|--------|
| `data/processed/stockx_cleaned_all.csv` | 97,116 | date, shoe_name, sale_price, retail_price, size, colorway, brand, release_date, buyer_region |
| `data/processed/stockx_price_cleaned_member_a.csv` | 83,750 | date, shoe_name, sale_price, size, colorway |

> **Note:** Both `data/raw/` and `data/processed/` are listed in `.gitignore`. The CSVs are not committed to the repository. Re-generate them by running the script above after placing the raw file at `data/raw/StockX-Data-Contest-2019-3.csv`.

### 3. Documentation

| File | Purpose |
|------|---------|
| `docs/member_a_stockx_data_notes.md` | Detailed notes on dataset, cleaning steps, schema, assumptions, and limitations |
| `docs/member_a_deliverable_summary.md` | This file – top-level summary for team review |

---

## Final Schemas

### `stockx_price_cleaned_member_a.csv` (Member A canonical schema)

| Column | Type | Description |
|--------|------|-------------|
| date | string (YYYY-MM-DD) | Transaction order date |
| shoe_name | string | Hyphen-separated sneaker slug |
| sale_price | float | Resale sale price in USD |
| size | float | US shoe size |
| colorway | string | Always `"Unknown"` (not in source) |

### `stockx_cleaned_all.csv` (full cleaned version for downstream)

| Column | Type | Description |
|--------|------|-------------|
| date | string (YYYY-MM-DD) | Transaction order date |
| shoe_name | string | Hyphen-separated sneaker slug |
| sale_price | float | Resale sale price in USD |
| retail_price | float | Original retail price in USD |
| size | float | US shoe size |
| colorway | string | Always `"Unknown"` (not in source) |
| brand | string | Brand (e.g. Yeezy, Off-White) |
| release_date | string | Original release date (YYYY-MM-DD) |
| buyer_region | string | US state of buyer |

---

## Data Quality Summary

| Check | stockx_cleaned_all | stockx_price_cleaned_member_a |
|-------|-------------------|-------------------------------|
| Missing critical values | 0 | 0 |
| Duplicate rows | 0 | 0 |
| Invalid sale_price (≤0) | 0 | 0 |
| Invalid retail_price (≤0) | 0 | — |
| Invalid size (≤0) | 0 | 0 |
| Date format (YYYY-MM-DD) | Valid | Valid |
| Date range | 2017-09-01 → 2019-02-13 | 2017-09-01 → 2019-02-13 |
| Unique sneakers | 50 | 50 |

---

### 4. Social Text Collection & Preprocessing

**File:** `scraping/member_a_social_text_preprocessing.py`

Collects sneaker-related posts from Reddit's public JSON API (no API key required), cleans them, and exports to a standard schema for Member B's sentiment analysis.

Run with:
```bash
python scraping/member_a_social_text_preprocessing.py
```

**Output:**

| File | Rows | Date range | Real or Demo |
|------|------|------------|--------------|
| `data/processed/social_text_cleaned.csv` | 1,967 | 2011-09-15 → 2026-05-25 | **Real Reddit data** |

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| date | string (YYYY-MM-DD) | Reddit post date |
| text | string | Post title + body combined |
| likes | int | Reddit upvote score |
| retweets | int | Always 0 (Reddit has no retweets) |
| source | string | e.g. `reddit/r/sneakers` |
| query | string | Search term that found this post |
| shoe_name | string | Best-matching shoe name |

**Why Reddit instead of Twitter/X:** Twitter API requires a paid developer account. Reddit's public JSON endpoint is free and needs no authentication. See `docs/member_a_stockx_data_notes.md` for full collection methodology.

**Date alignment:** Of 1,967 posts, 243 fall within the StockX price date range (Sep 2017 – Feb 2019). This produces 32,834 / 83,750 price rows with a matching daily VADER sentiment score in the merged output. Full coverage requires historical Reddit data from 2017–2019.

---

## What Was NOT Done (Out of Scope for Member A)

- Sentiment scoring (VADER / FinBERT) – Member B
- LSTM / GRU modeling – Member B/C
- Trading simulation / backtesting – Member B/C
- Streamlit UI – Member C
- Additional datasets (e.g. GOAT, Flight Club) – future iteration
