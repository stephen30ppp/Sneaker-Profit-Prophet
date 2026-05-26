"""
Member B — VADER vs FinBERT Comparison

Quantitative comparison of the two sentiment scorers on the SAME 1,967
Reddit posts. Required by the Member B deliverable list.

Run from project root (after vader_scorer.py and finbert_scorer.py have
both produced real outputs):
    python sentiment/compare_vader_finbert.py

Inputs:
    data/sentiment/vader_scored_posts.csv
    data/sentiment/finbert_scored_posts.csv

Outputs (under data/sentiment/comparison/):
    label_distribution.png      Bar chart of pos/neu/neg counts per model
    confusion_matrix.png        VADER label x FinBERT label heatmap
    daily_scatter.png           Daily sentiment_mean scatter plot
    daily_timeline.png          Daily mean over time, both models overlaid
    top_disagreements.csv       Top-N posts where the two models disagree most
    comparison_report.md        Plain-text summary with all key numbers

Methodology:
    - VADER produces a continuous 'vader_compound' in [-1, +1].
      Map to label using the standard cutoffs:
          compound >  0.05  -> positive
          compound < -0.05  -> negative
          otherwise         -> neutral
    - FinBERT directly produces a {-1, 0, +1} score and a label.
    - Agreement rate     = % posts where both models give the same label.
    - Pearson / Spearman = correlation between vader_compound and
                           finbert_score on the post level.
    - Daily correlation  = correlation between the two daily mean series
                           on overlapping dates.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend; safe on headless machines
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib is not installed. Run: pip install matplotlib")
    sys.exit(1)

from scipy.stats import pearsonr, spearmanr

from config import SENTIMENT_DIR

# --- Paths ---
VADER_POSTS    = SENTIMENT_DIR / "vader_scored_posts.csv"
FINBERT_POSTS  = SENTIMENT_DIR / "finbert_scored_posts.csv"
VADER_DAILY    = SENTIMENT_DIR / "vader_daily_sentiment.csv"
FINBERT_DAILY  = SENTIMENT_DIR / "finbert_daily_sentiment.csv"

OUT_DIR = SENTIMENT_DIR / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_DIST_PNG  = OUT_DIR / "label_distribution.png"
CONF_MATRIX_PNG = OUT_DIR / "confusion_matrix.png"
DAILY_SCATTER   = OUT_DIR / "daily_scatter.png"
DAILY_TIMELINE  = OUT_DIR / "daily_timeline.png"
TOP_DISAGREE    = OUT_DIR / "top_disagreements.csv"
REPORT_MD       = OUT_DIR / "comparison_report.md"

# Standard VADER thresholds — see vaderSentiment README
VADER_POS_CUTOFF =  0.05
VADER_NEG_CUTOFF = -0.05

LABEL_ORDER = ["negative", "neutral", "positive"]
TOP_N_DISAGREE = 20


# ------------------------------------------------------------
# Loading & label mapping
# ------------------------------------------------------------
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not VADER_POSTS.exists():
        sys.exit(f"ERROR: {VADER_POSTS} not found. Run vader_scorer.py first.")
    if not FINBERT_POSTS.exists():
        sys.exit(f"ERROR: {FINBERT_POSTS} not found. Run finbert_scorer.py first.")

    vader   = pd.read_csv(VADER_POSTS)
    finbert = pd.read_csv(FINBERT_POSTS)

    if len(vader) != len(finbert):
        print(f"WARNING: row count mismatch — VADER {len(vader)} vs FinBERT {len(finbert)}")

    # FinBERT may legitimately be all-NaN if torch was missing on the run.
    real_fb = finbert["finbert_score"].notna().sum()
    if real_fb == 0:
        sys.exit(
            "ERROR: All finbert_score values are NaN. Install torch+transformers and re-run "
            "finbert_scorer.py before doing the comparison."
        )

    return vader, finbert


def vader_compound_to_label(c: float) -> str:
    if c >  VADER_POS_CUTOFF:
        return "positive"
    if c <  VADER_NEG_CUTOFF:
        return "negative"
    return "neutral"


def build_joined(vader: pd.DataFrame, finbert: pd.DataFrame) -> pd.DataFrame:
    """Align the two scorings row-by-row on the shared 'text' column."""
    df = vader.merge(
        finbert[["text", "finbert_label", "finbert_score"]],
        on="text",
        how="inner",
    )
    df["vader_label"] = df["vader_compound"].apply(vader_compound_to_label)
    df = df.dropna(subset=["finbert_score"]).reset_index(drop=True)
    return df


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    v = df["vader_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    f = df["finbert_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    out = pd.DataFrame({"VADER": v, "FinBERT": f})
    out["VADER_pct"]   = (out["VADER"]   / out["VADER"].sum()   * 100).round(2)
    out["FinBERT_pct"] = (out["FinBERT"] / out["FinBERT"].sum() * 100).round(2)
    return out


def agreement_rate(df: pd.DataFrame) -> float:
    return (df["vader_label"] == df["finbert_label"]).mean() * 100


def confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cm = pd.crosstab(
        df["vader_label"], df["finbert_label"],
        rownames=["VADER"], colnames=["FinBERT"],
    )
    return cm.reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)


def post_level_correlations(df: pd.DataFrame) -> dict:
    p_r, p_p = pearsonr(df["vader_compound"],  df["finbert_score"])
    s_r, s_p = spearmanr(df["vader_compound"], df["finbert_score"])
    return {
        "pearson_r":   float(p_r),
        "pearson_p":   float(p_p),
        "spearman_r":  float(s_r),
        "spearman_p":  float(s_p),
    }


def daily_correlation() -> dict:
    if not (VADER_DAILY.exists() and FINBERT_DAILY.exists()):
        return {"available": False}

    vd = pd.read_csv(VADER_DAILY)
    fd = pd.read_csv(FINBERT_DAILY)
    merged = vd.merge(fd, on="date", suffixes=("_vader", "_finbert"))
    merged = merged.dropna(subset=["sentiment_mean_vader", "sentiment_mean_finbert"])

    if len(merged) < 3:
        return {"available": False, "n": len(merged)}

    p_r, p_p = pearsonr(merged["sentiment_mean_vader"], merged["sentiment_mean_finbert"])
    return {
        "available": True,
        "n":          int(len(merged)),
        "pearson_r":  float(p_r),
        "pearson_p":  float(p_p),
        "merged":     merged,
    }


def top_disagreements(df: pd.DataFrame, n: int = TOP_N_DISAGREE) -> pd.DataFrame:
    """Posts where the two models disagree the most (largest |vader - finbert|)."""
    out = df.copy()
    out["disagreement"] = (out["vader_compound"] - out["finbert_score"]).abs()
    cols = ["date", "vader_label", "vader_compound",
            "finbert_label", "finbert_score", "disagreement", "text"]
    return (
        out.sort_values("disagreement", ascending=False)
           .head(n)[cols]
           .reset_index(drop=True)
    )


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
def plot_label_distribution(dist: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(LABEL_ORDER))
    width = 0.35
    ax.bar(x - width / 2, dist["VADER"],   width, label="VADER",   color="#3b82f6")
    ax.bar(x + width / 2, dist["FinBERT"], width, label="FinBERT", color="#ef4444")
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_ylabel("Number of posts")
    ax.set_title("Sentiment Label Distribution — VADER vs FinBERT")
    ax.legend()
    fig.tight_layout()
    fig.savefig(LABEL_DIST_PNG, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(LABEL_ORDER)))
    ax.set_yticks(range(len(LABEL_ORDER)))
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel("FinBERT label")
    ax.set_ylabel("VADER label")
    ax.set_title("Confusion Matrix (VADER row × FinBERT col)")
    for i in range(len(LABEL_ORDER)):
        for j in range(len(LABEL_ORDER)):
            v = cm.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.values.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CONF_MATRIX_PNG, dpi=120)
    plt.close(fig)


def plot_daily(daily_info: dict) -> None:
    if not daily_info.get("available"):
        return
    merged = daily_info["merged"].copy()
    merged["date"] = pd.to_datetime(merged["date"])

    # Scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(merged["sentiment_mean_vader"], merged["sentiment_mean_finbert"],
               alpha=0.4, s=14, color="#6366f1")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.plot([-1, 1], [-1, 1], color="red", linewidth=0.7, linestyle="--", label="y = x")
    ax.set_xlabel("VADER daily mean (compound)")
    ax.set_ylabel("FinBERT daily mean (-1/0/+1)")
    ax.set_title(f"Daily Sentiment Mean — Pearson r = {daily_info['pearson_r']:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DAILY_SCATTER, dpi=120)
    plt.close(fig)

    # Timeline
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(merged["date"], merged["sentiment_mean_vader"],   label="VADER",   color="#3b82f6", linewidth=0.9)
    ax.plot(merged["date"], merged["sentiment_mean_finbert"], label="FinBERT", color="#ef4444", linewidth=0.9, alpha=0.8)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_ylabel("Daily mean sentiment")
    ax.set_title("Daily Sentiment Time Series — VADER vs FinBERT")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DAILY_TIMELINE, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------
def _df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown-table renderer (avoids tabulate dependency)."""
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for idx, row in df.iterrows():
        cells = [str(idx)] + [str(v) for v in row.values]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def write_report(
    n_posts: int,
    dist: pd.DataFrame,
    agreement: float,
    cm: pd.DataFrame,
    corr: dict,
    daily: dict,
) -> None:
    lines = []
    lines.append("# VADER vs FinBERT — Comparison Report")
    lines.append("")
    lines.append(f"- Posts compared (rows with real scores in BOTH models): **{n_posts:,}**")
    lines.append("")

    lines.append("## 1. Label distribution")
    lines.append("")
    lines.append(_df_to_markdown(dist))
    lines.append("")

    lines.append("## 2. Agreement")
    lines.append("")
    lines.append(f"- Overall label-agreement rate: **{agreement:.2f}%**")
    lines.append("")

    lines.append("## 3. Confusion matrix (VADER row × FinBERT col)")
    lines.append("")
    lines.append(_df_to_markdown(cm))
    lines.append("")

    lines.append("## 4. Post-level score correlation")
    lines.append("")
    lines.append(f"- Pearson  r = **{corr['pearson_r']:+.4f}**  (p = {corr['pearson_p']:.2e})")
    lines.append(f"- Spearman ρ = **{corr['spearman_r']:+.4f}**  (p = {corr['spearman_p']:.2e})")
    lines.append("")

    lines.append("## 5. Daily-level correlation")
    lines.append("")
    if daily.get("available"):
        lines.append(f"- Overlapping dates: **{daily['n']}**")
        lines.append(f"- Pearson r on daily means: **{daily['pearson_r']:+.4f}**  (p = {daily['pearson_p']:.2e})")
    else:
        lines.append("- Daily files unavailable; skipped.")
    lines.append("")

    lines.append("## 6. Outputs")
    lines.append("")
    lines.append("- label_distribution.png")
    lines.append("- confusion_matrix.png")
    lines.append("- daily_scatter.png")
    lines.append("- daily_timeline.png")
    lines.append("- top_disagreements.csv  (top 20 posts where the two models disagree most)")
    lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("Loading inputs...")
    vader, finbert = load_inputs()

    df = build_joined(vader, finbert)
    print(f"[OK] Joined posts (real scores both sides): {len(df):,}")

    print("Computing label distribution...")
    dist = label_distribution(df)
    print(dist)

    agreement = agreement_rate(df)
    print(f"\nAgreement rate: {agreement:.2f}%")

    cm = confusion_matrix(df)
    print("\nConfusion matrix (VADER × FinBERT):")
    print(cm)

    corr = post_level_correlations(df)
    print(f"\nPost-level Pearson  r = {corr['pearson_r']:+.4f}  (p = {corr['pearson_p']:.2e})")
    print(f"Post-level Spearman ρ = {corr['spearman_r']:+.4f}  (p = {corr['spearman_p']:.2e})")

    daily = daily_correlation()
    if daily.get("available"):
        print(f"\nDaily Pearson r = {daily['pearson_r']:+.4f} on {daily['n']} overlapping dates")
    else:
        print("\nDaily-level correlation skipped (daily CSVs missing or too sparse).")

    print("\nWriting plots...")
    plot_label_distribution(dist)
    plot_confusion_matrix(cm)
    plot_daily(daily)

    print("Writing top disagreements CSV...")
    top = top_disagreements(df)
    top.to_csv(TOP_DISAGREE, index=False)

    print("Writing markdown report...")
    write_report(len(df), dist, agreement, cm, corr, daily)

    print("\n--- Done ---")
    print(f"All artifacts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
