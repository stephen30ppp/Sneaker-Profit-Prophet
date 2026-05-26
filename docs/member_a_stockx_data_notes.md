# Member A – StockX Data Notes

## Responsibility

Member A is responsible for data collection and preprocessing. This covers sourcing the sneaker transaction dataset, cleaning it, normalising the schema, and exporting reproducible CSV outputs for downstream use by Member B (modeling) and Member C (sentiment / UI).

---

## Dataset Source

**Dataset:** Kaggle – StockX Sneaker Data Contest 2019  
**Raw file:** `data/raw/StockX-Data-Contest-2019-3.csv`  
**Rows (raw):** 99,956  
**Coverage:** September 2017 – February 2019

### Why Kaggle instead of scraping StockX directly

StockX's website and API do not provide a public bulk data endpoint. Scraping the site at scale would:

- Violate StockX's Terms of Service (automated data collection is prohibited).
- Require persistent browser automation that is brittle and rate-limited.
- Produce an incomplete, inconsistent dataset.

The Kaggle StockX Sneaker Data Contest dataset is a publicly released, curated export that covers ~100 k verified resale transactions. It is reproducible, well-structured, and sufficient for the modelling goals of this project.

---

## Raw Schema

| Raw column    | Type   | Notes                                       |
|---------------|--------|---------------------------------------------|
| Order Date    | string | `M/D/YY` format (e.g. `9/1/17`)             |
| Brand         | string | e.g. `Yeezy`, `Off-White` (leading space)   |
| Sneaker Name  | string | hyphen-separated slug                       |
| Sale Price    | string | Dollar-formatted, e.g. `$1,097`             |
| Retail Price  | string | Dollar-formatted, e.g. `$220`               |
| Release Date  | string | `M/D/YY` format                             |
| Shoe Size     | float  | US sizing                                   |
| Buyer Region  | string | US state                                    |

No colorway column exists in the source data.

---

## Cleaning Steps

Performed by `scraping/member_a_stockx_cleaning.py`:

1. **Rename columns** – snake_case mapping via `RENAME_MAP`.
2. **Strip whitespace** – leading/trailing spaces removed from all string columns (fixes `" Yeezy"` → `"Yeezy"` etc.).
3. **Parse dates** – `pd.to_datetime(..., format="%m/%d/%y")` converts `9/1/17` → `2017-09-01`. Output stored as `YYYY-MM-DD` string.
4. **Parse currency** – regex strips `$` and `,` then `pd.to_numeric(..., errors="coerce")` converts to float.
5. **Parse size** – `pd.to_numeric(..., errors="coerce")` converts to float.
6. **Add colorway** – fixed value `"Unknown"` added since the source has no colorway column.
7. **Drop missing critical values** – rows with NaN in `date`, `shoe_name`, `sale_price`, `retail_price`, or `size` are removed.
8. **Remove duplicates** – exact-row duplicates removed (2,840 removed from the raw 99,956).
9. **Remove invalid numerics** – rows where `sale_price`, `retail_price`, or `size` ≤ 0 are removed.
10. **Sort** – ascending by `shoe_name`, then `date`.
11. **Re-deduplicate Member A subset** – after narrowing to the five-column schema, rows that differ only in `buyer_region` or `retail_price` become identical; a second `drop_duplicates()` is applied.

---

## Output Files

| File | Rows | Columns |
|------|------|---------|
| `data/processed/stockx_cleaned_all.csv` | 97,116 | date, shoe_name, sale_price, retail_price, size, colorway, brand, release_date, buyer_region |
| `data/processed/stockx_price_cleaned_member_a.csv` | 83,750 | date, shoe_name, sale_price, size, colorway |

---

## Assumptions

- All dates in the `Order Date` column follow `M/D/YY` format; no rows needed fallback parsing.
- `Shoe Size` uses US sizing conventions. No unit conversion is applied.
- A `colorway` of `"Unknown"` is semantically acceptable for downstream models that accept this field; it can be enriched later if a supplementary dataset with colorway data is merged.
- `retail_price` is treated as a required field (no rows had it missing in this dataset).

---

## Limitations

- **No colorway** – the source dataset contains no colorway information. All records carry `"Unknown"`. This limits any colour-based analysis unless external data is joined.
- **50 unique sneakers** – the Kaggle dataset covers only 50 sneaker models, heavily skewed toward Yeezy and Off-White × Nike collaborations. Generalisation to other models requires additional data.
- **US buyers only** – `buyer_region` covers US states. International demand is not represented.
- **Date range** – transactions span Sep 2017 – Feb 2019. Newer market dynamics are not captured.

---

## What Member B / C Can Use

- **Member B (modeling / LSTM / GRU):** Use `stockx_cleaned_all.csv`. It retains `retail_price`, `brand`, and `release_date` which are useful as features. The `sale_price` column is the regression target.
- **Member C (sentiment / UI):** Use `stockx_price_cleaned_member_a.csv` as the price backbone for joining with sentiment scores. The five-column schema is intentionally minimal for easy merging.

---

## Social Text Data (Member A Extension)

**Script:** `scraping/member_a_social_text_preprocessing.py`

### Why Reddit instead of Twitter/X

Twitter/X API now requires a paid developer subscription (Basic tier ≈ $100 USD/month). The public v1.1 endpoints were shut down in 2023. Reddit's public JSON search endpoint is free, requires no authentication, and provides genuine user-generated discussion about sneakers and the resale market.

### Collection Method

Reddit posts were fetched using the public JSON endpoint:
```
https://www.reddit.com/r/{subreddit}/search.json?q={query}&sort=relevance&t=all&limit=100
```

**Search queries used:** Yeezy Boost 350, Yeezy Boost 700, Off-White Jordan 1, Nike Dunk resale, StockX sneaker market, sneaker resale profit, Off-White Nike collab

**Subreddits searched:** r/sneakers, r/Sneakers, r/streetwear, r/adidas, r/Nike

**Text constructed from:** post title + post body (selftext) combined into one string. Deleted/removed posts excluded.

### Cleaning Steps

1. Convert `created_utc` (Unix timestamp) → `YYYY-MM-DD`.
2. Combine `title` + `selftext` into the `text` column.
3. Remove posts with fewer than 10 characters (noise).
4. Remove exact duplicate text across all subreddit×query combinations.
5. Convert `score` (Reddit upvotes) → `likes`; set `retweets = 0`.
6. Sort ascending by date.

### Social Text Output

| File | Rows | Date range | Columns |
|------|------|------------|---------|
| `data/processed/social_text_cleaned.csv` | 1,967 | 2011-09-15 → 2026-05-25 | date, text, likes, retweets, source, query, shoe_name |

### Date Alignment Note

The StockX price data covers **Sep 2017 – Feb 2019**. Reddit's public search endpoint returns posts across all time, but is biased toward more-recent content. Of the 1,967 collected posts, **243 fall within the price data date range** — resulting in **32,834 / 83,750 price rows** having a matching daily sentiment score in the merged output.

The remaining 50,916 price rows have `NaN` in all sentiment columns. This is honest: no Reddit post was found for those specific dates. Once real historical social data covering 2017–2019 is available (e.g. via Pushshift academic dataset), it should replace `social_text_cleaned.csv` and re-running the pipeline will produce full coverage.

### Social Text Limitations

- Reddit posts are not the same as Twitter. Reddit tends toward longer, more deliberate text, while Twitter is short-form and more immediate.
- Post dates reflect when a Reddit post was created, not when a transaction occurred.
- `retweets` is always 0 for Reddit (the concept does not apply).
- Upvote scores reflect long-term community agreement, not immediate viral reaction.
