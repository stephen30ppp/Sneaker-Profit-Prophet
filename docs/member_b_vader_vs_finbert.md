# Member B — VADER vs FinBERT Comparison

## Scope

Quantitative comparison of two sentiment scorers on the same 1,967 real Reddit posts about sneakers and the resale market. The aim is to decide which scorer (or combination) Member C should use as a feature for LSTM/GRU price forecasting.

## Inputs

| File | Rows | Source |
|------|------|--------|
| `data/sentiment/vader_scored_posts.csv` | 1,967 | VADER (rule-based, social-media tuned) |
| `data/sentiment/finbert_scored_posts.csv` | 1,967 | FinBERT (`ProsusAI/finbert`, finance-tuned) |

## Methodology

- **VADER → label**: `compound > +0.05` → positive; `compound < −0.05` → negative; else neutral.
- **FinBERT → label**: native `positive / neutral / negative` from the pipeline.
- **Score axis**: VADER `compound ∈ [−1, +1]` vs FinBERT `score ∈ {−1, 0, +1}`.
- **Agreement rate**: % of posts where both models output the same label.
- **Correlation**: Pearson and Spearman on post-level scores; Pearson on daily means over the 1,508 overlapping dates.
- **Disagreement ranking**: posts sorted by `|vader_compound − finbert_score|`.

## Results

### 1. Label distribution

| Label | VADER | VADER % | FinBERT | FinBERT % |
|-------|-------|---------|---------|-----------|
| negative | 296 | 15.05 | 73 | 3.71 |
| neutral | 459 | 23.34 | **1,843** | **93.70** |
| positive | 1,212 | 61.62 | 51 | 2.59 |

### 2. Agreement

- Overall label agreement: **26.74 %** (526 / 1,967 posts).

### 3. Confusion matrix (VADER row × FinBERT col)

| | negative | neutral | positive |
|---|---|---|---|
| negative | 32 | 263 | 1 |
| neutral | 7 | 448 | 4 |
| positive | 34 | 1,132 | 46 |

> 1,132 posts that VADER calls **positive** are flattened to **neutral** by FinBERT — this single cell is the dominant source of disagreement.

### 4. Score correlation

| Level | Metric | Value | p-value |
|-------|--------|-------|---------|
| Post-level | Pearson r | **+0.127** | 1.79 × 10⁻⁸ |
| Post-level | Spearman ρ | **+0.107** | 2.14 × 10⁻⁶ |
| Daily-level (1,508 dates) | Pearson r | **+0.131** | — |

Statistically significant but practically very weak: the two models share less than 2 % of variance.

### 5. Plots

All written to `data/sentiment/comparison/`:

- `label_distribution.png` — per-model label counts.
- `confusion_matrix.png` — VADER × FinBERT heatmap.
- `daily_scatter.png` — daily mean scatter, with `y = x` reference.
- `daily_timeline.png` — daily mean overlaid over time.
- `top_disagreements.csv` — top 20 posts ranked by absolute score gap.

## Key findings

1. **FinBERT collapses to neutral on sneaker text.** 93.7 % of posts are labelled neutral with near-zero variance. As a model feature this column is effectively a constant and contributes no signal.
2. **VADER produces a usable signal.** Distribution is well-spread (15 / 23 / 62), variance is non-trivial, and the score is continuous in `[−1, +1]`.
3. **The two models do not corroborate each other.** 26.7 % agreement and r ≈ 0.13 mean we cannot use them as mutual validation.
4. **Domain mismatch is the most likely cause.** FinBERT was fine-tuned on financial news (earnings calls, analyst reports). It does not see sneaker slang ("fire", "cop", "W", "L", "cook", "🔥"), and treats most informal posts as neutral.

## Recommendation for Member C

- **Use `vader_sentiment_mean` as the sentiment feature** in the LSTM/GRU input.
- **Do not include `finbert_sentiment_mean` as a feature.** Its near-constant value will not help the model and may slow convergence.
- Keep the FinBERT columns in `price_sentiment_merged.csv` for transparency and possible later experiments, but exclude them from the feature list passed to `SneakerPriceDataset`.
- For the 60.8 % of price rows with no matching daily VADER score (date-coverage gap, see `member_b_sentiment_notes.md`), forward-fill or zero-fill is acceptable; document the choice.

## Limitations

- Single dataset (Reddit, 1,967 posts). Conclusion may not generalise to Twitter or news data.
- Cutoffs `±0.05` are the published VADER defaults; they were not tuned for sneaker text.
- FinBERT was used in the off-the-shelf `ProsusAI/finbert` form. Fine-tuning on sneaker data is out of scope here but would be the obvious next step.

## Reproducibility

```bash
# 1. score posts
python sentiment/vader_scorer.py
python sentiment/finbert_scorer.py     # requires torch + transformers

# 2. aggregate to daily
python sentiment/aggregate_sentiment.py

# 3. run the comparison
python sentiment/compare_vader_finbert.py
```

All artifacts are written to `data/sentiment/comparison/`. The directory is git-ignored — re-generate locally.
