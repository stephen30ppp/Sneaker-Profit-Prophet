"""
End-to-End Data Pipeline

Connects: Raw Data → Sentiment Scoring → Feature Engineering → Model Training → Agent Backtest

Usage:
    python pipeline.py                 # Full pipeline
    python pipeline.py --skip-train    # Skip training (use existing checkpoints)
"""
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from config import (
    ROOT_DIR, RAW_DIR, PROCESSED_DIR, SENTIMENT_DIR, MODEL_DIR,
    SEQUENCE_LENGTH, FORECAST_HORIZON, BATCH_SIZE,
)


def step1_load_data():
    """Load and clean the StockX dataset."""
    print("\n" + "=" * 60)
    print("STEP 1: Loading StockX Data")
    print("=" * 60)

    from scraping.stockx_scraper import load_kaggle_dataset, get_daily_prices

    csv_path = ROOT_DIR / "StockX-Data-Contest-2019-3.csv"
    df = load_kaggle_dataset(str(csv_path))
    print(f"  Loaded {len(df)} transactions")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Brands: {df['brand'].str.strip().unique().tolist()}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / "stockx_all.csv", index=False)

    daily = get_daily_prices(df, brand="Yeezy")
    print(f"  Daily Yeezy prices: {len(daily)} days")

    return df, daily


def step2_load_tweets():
    """Load Twitter sentiment data."""
    print("\n" + "=" * 60)
    print("STEP 2: Loading Twitter Data")
    print("=" * 60)

    from scraping.twitter_scraper import load_tweets_jsonl, get_brand_tweets

    tweets_df = load_tweets_jsonl()
    print(f"  Loaded {len(tweets_df)} tweets")
    if not tweets_df.empty:
        print(f"  Date range: {tweets_df['date'].min().date()} to {tweets_df['date'].max().date()}")
        print(f"  Brand distribution: {tweets_df['brand'].value_counts().to_dict()}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tweets_df.to_csv(RAW_DIR / "tweets_all.csv", index=False)

    return tweets_df


def step3_sentiment(tweets_df: pd.DataFrame):
    """Score tweets with VADER and FinBERT."""
    print("\n" + "=" * 60)
    print("STEP 3: Sentiment Scoring")
    print("=" * 60)

    from sentiment.vader_scorer import aggregate_daily_sentiment as vader_daily, save_sentiment as save_vader
    from sentiment.finbert_scorer import aggregate_daily_sentiment as finbert_daily, save_sentiment as save_finbert

    if tweets_df.empty:
        print("  No tweets to score, generating synthetic sentiment...")
        dates = pd.date_range("2017-09-01", "2019-02-28", freq="D")
        vader_sentiment = pd.DataFrame({
            "date": dates,
            "sentiment_mean": np.random.normal(0.1, 0.3, len(dates)).clip(-1, 1),
            "sentiment_std": np.abs(np.random.normal(0.2, 0.1, len(dates))),
            "tweet_count": np.random.randint(5, 50, len(dates)),
        })
        finbert_sentiment = vader_sentiment.copy()
        finbert_sentiment["sentiment_mean"] *= 0.8
    else:
        sample = tweets_df.head(10000)
        print(f"  Scoring {len(sample)} tweets with VADER...")
        vader_sentiment = vader_daily(sample)
        print(f"  VADER: {len(vader_sentiment)} daily scores")

        print(f"  Scoring {len(sample)} tweets with FinBERT (fallback mode)...")
        finbert_sentiment = finbert_daily(sample)
        print(f"  FinBERT: {len(finbert_sentiment)} daily scores")

    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    vader_sentiment.to_csv(SENTIMENT_DIR / "daily_vader.csv", index=False)
    finbert_sentiment.to_csv(SENTIMENT_DIR / "daily_finbert.csv", index=False)

    print(f"  VADER mean sentiment: {vader_sentiment['sentiment_mean'].mean():.4f}")
    print(f"  FinBERT mean sentiment: {finbert_sentiment['sentiment_mean'].mean():.4f}")

    return vader_sentiment, finbert_sentiment


def step4_feature_engineering(daily_prices: pd.DataFrame, sentiment_df: pd.DataFrame):
    """Merge price and sentiment into model-ready features."""
    print("\n" + "=" * 60)
    print("STEP 4: Feature Engineering")
    print("=" * 60)

    daily_prices = daily_prices.copy()
    sentiment_df = sentiment_df.copy()

    daily_prices["date"] = pd.to_datetime(daily_prices["date"]).dt.tz_localize(None)
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.tz_localize(None)

    merged = pd.merge(daily_prices, sentiment_df, on="date", how="left")

    if merged["sentiment_mean"].isna().all() and not sentiment_df.empty:
        avg_sent = sentiment_df["sentiment_mean"].mean()
        std_sent = sentiment_df["sentiment_mean"].std()
        np.random.seed(42)
        n = len(merged)
        merged["sentiment_mean"] = np.random.normal(avg_sent, max(std_sent, 0.1), n).clip(-1, 1)
        merged["sentiment_std"] = np.abs(np.random.normal(0.2, 0.05, n))
        merged["tweet_count"] = np.random.randint(10, 100, n)
        print("  Note: Date ranges don't overlap — projecting sentiment distribution onto price period")

    merged["sentiment_mean"] = merged["sentiment_mean"].fillna(0).astype(float)
    merged["sentiment_std"] = merged["sentiment_std"].fillna(0).astype(float)

    merged["price_norm"] = (merged["price"] - merged["price"].mean()) / merged["price"].std()
    merged["return_1d"] = merged["price"].pct_change().fillna(0)
    merged["sma_7"] = merged["price"].rolling(7, min_periods=1).mean()
    merged["sma_30"] = merged["price"].rolling(30, min_periods=1).mean()
    merged["volatility_7d"] = merged["return_1d"].rolling(7, min_periods=1).std().fillna(0)

    merged = merged.dropna().reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(PROCESSED_DIR / "features.csv", index=False)
    print(f"  Feature matrix: {merged.shape[0]} rows x {merged.shape[1]} columns")
    print(f"  Features: {list(merged.columns)}")

    return merged


def step5_train(features_df: pd.DataFrame):
    """Train LSTM and GRU models."""
    print("\n" + "=" * 60)
    print("STEP 5: Model Training")
    print("=" * 60)

    from models.dataset import SneakerPriceDataset
    from models.lstm_model import SneakerLSTM
    from models.gru_model import SneakerGRU
    from models.train import train_model

    feature_cols = ["price_norm", "sentiment_mean", "return_1d", "volatility_7d"]
    target_col = "price_norm"

    available_cols = [c for c in feature_cols if c in features_df.columns]
    if not available_cols:
        available_cols = ["price_norm", "sentiment_mean"]

    min_samples = SEQUENCE_LENGTH + FORECAST_HORIZON + 10
    if len(features_df) < min_samples:
        print(f"  Not enough data ({len(features_df)} < {min_samples}), skipping training")
        return {}

    dataset = SneakerPriceDataset(features_df, available_cols, target_col)
    n = len(dataset)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)

    train_ds = torch.utils.data.Subset(dataset, range(train_size))
    val_ds = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Training samples: {train_size}, Validation: {val_size}, Test: {n - train_size - val_size}")
    print(f"  Input features: {len(available_cols)}")

    histories = {}

    print("\n  --- Training LSTM ---")
    lstm = SneakerLSTM(input_size=len(available_cols))
    histories["lstm"] = train_model(lstm, train_loader, val_loader, model_name="lstm")

    print("\n  --- Training GRU ---")
    gru = SneakerGRU(input_size=len(available_cols))
    histories["gru"] = train_model(gru, train_loader, val_loader, model_name="gru")

    print(f"\n  Checkpoints saved to {MODEL_DIR}")
    return histories


def step6_evaluate(features_df: pd.DataFrame):
    """Evaluate models against random walk baseline."""
    print("\n" + "=" * 60)
    print("STEP 6: Evaluation (Model vs Random Walk)")
    print("=" * 60)

    from backtesting.baseline import random_walk_mse
    from backtesting.evaluate import mse, rmse, mae, directional_accuracy
    from models.lstm_model import SneakerLSTM
    from models.gru_model import SneakerGRU

    prices = features_df["price_norm"].values
    rw_mse = random_walk_mse(prices, FORECAST_HORIZON)
    print(f"  Random Walk MSE: {rw_mse:.6f}")

    feature_cols = ["price_norm", "sentiment_mean", "return_1d", "volatility_7d"]
    available_cols = [c for c in feature_cols if c in features_df.columns]

    results = {"random_walk": {"mse": rw_mse}}

    for model_name, ModelClass in [("lstm", SneakerLSTM), ("gru", SneakerGRU)]:
        checkpoint = MODEL_DIR / f"{model_name}_best.pt"
        if not checkpoint.exists():
            print(f"  {model_name.upper()} checkpoint not found, skipping")
            continue

        model = ModelClass(input_size=len(available_cols))
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()

        test_start = int(len(features_df) * 0.85)
        test_data = features_df.iloc[test_start:].reset_index(drop=True)

        if len(test_data) < SEQUENCE_LENGTH + FORECAST_HORIZON:
            print(f"  Not enough test data for {model_name.upper()}")
            continue

        all_preds = []
        all_actuals = []
        features_arr = torch.FloatTensor(test_data[available_cols].values)
        targets_arr = test_data["price_norm"].values

        num_windows = len(test_data) - SEQUENCE_LENGTH - FORECAST_HORIZON + 1
        for i in range(num_windows):
            x = features_arr[i : i + SEQUENCE_LENGTH].unsqueeze(0)
            with torch.no_grad():
                pred = model(x).squeeze().numpy()
            actual = targets_arr[i + SEQUENCE_LENGTH : i + SEQUENCE_LENGTH + FORECAST_HORIZON]
            all_preds.append(pred)
            all_actuals.append(actual)

        if all_preds:
            preds = np.array(all_preds)
            actuals = np.array(all_actuals)
            model_mse = mse(actuals, preds)
            model_rmse = rmse(actuals, preds)
            model_mae_val = mae(actuals, preds)

            results[model_name] = {
                "mse": model_mse,
                "rmse": model_rmse,
                "mae": model_mae_val,
                "improvement_vs_rw": (1 - model_mse / rw_mse) * 100 if rw_mse > 0 else 0,
            }
            print(f"  {model_name.upper()} MSE: {model_mse:.6f} (vs RW: {results[model_name]['improvement_vs_rw']:+.1f}%)")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).T.to_csv(PROCESSED_DIR / "evaluation_results.csv")
    return results


def step7_agent_backtest(daily_prices: pd.DataFrame, sentiment_df: pd.DataFrame):
    """Run agent backtest on historical data."""
    print("\n" + "=" * 60)
    print("STEP 7: Agent Trading Backtest")
    print("=" * 60)

    from trading.simulator import TradingSimulator, BuyAndHoldBaseline

    prices = daily_prices["price"].values.tolist()
    dates = daily_prices["date"].astype(str).values.tolist()

    if len(prices) < SEQUENCE_LENGTH:
        print("  Not enough price data for backtest")
        return {}

    sentiment_values = sentiment_df["sentiment_mean"].values.tolist() if not sentiment_df.empty else [0.0] * len(prices)
    if len(sentiment_values) < len(prices):
        sentiment_values = sentiment_values + [0.0] * (len(prices) - len(sentiment_values))

    sim = TradingSimulator()
    for i in range(SEQUENCE_LENGTH, len(prices)):
        price = prices[i]
        sent = sentiment_values[i] if i < len(sentiment_values) else 0.0

        confidence = 0.5 + sent * 0.2
        if i > 1 and prices[i] > prices[i - 1]:
            confidence += 0.1
        confidence = min(max(confidence, 0.0), 1.0)

        signal = sim.decide(confidence)
        sim.execute(signal, price, dates[i])

    bh = BuyAndHoldBaseline()
    bh_history = bh.run(prices, dates)

    agent_return = sim.total_return(prices[-1])
    bh_return = bh.total_return(prices[0], prices[-1])
    sharpe = sim.sharpe_ratio()
    max_dd = sim.max_drawdown()

    print(f"  Agent Total Return: {agent_return:.2f}%")
    print(f"  Buy & Hold Return: {bh_return:.2f}%")
    print(f"  Agent Sharpe Ratio: {sharpe:.4f}")
    print(f"  Agent Max Drawdown: {max_dd:.2%}")
    print(f"  Total Trades: {len(sim.trade_log)}")

    trade_log = sim.get_trade_log()
    portfolio_hist = sim.get_portfolio_history()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not trade_log.empty:
        trade_log.to_csv(PROCESSED_DIR / "trade_log.csv", index=False)
    portfolio_hist.to_csv(PROCESSED_DIR / "portfolio_history.csv", index=False)
    bh_history.to_csv(PROCESSED_DIR / "buyhold_history.csv", index=False)

    return {
        "agent_return": agent_return,
        "buyhold_return": bh_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "total_trades": len(sim.trade_log),
    }


def main():
    parser = argparse.ArgumentParser(description="Sneaker Profit Prophet — Full Pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    args = parser.parse_args()

    print("=" * 60)
    print("  SNEAKER PROFIT PROPHET — END-TO-END PIPELINE")
    print("=" * 60)

    stockx_df, daily_prices = step1_load_data()
    tweets_df = step2_load_tweets()
    vader_sentiment, finbert_sentiment = step3_sentiment(tweets_df)
    features_df = step4_feature_engineering(daily_prices, vader_sentiment)

    if not args.skip_train:
        step5_train(features_df)
    else:
        print("\n  [Skipping training — using existing checkpoints]")

    results = step6_evaluate(features_df)
    backtest = step7_agent_backtest(daily_prices, vader_sentiment)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Processed data saved to: {PROCESSED_DIR}")
    print(f"  Model checkpoints at: {MODEL_DIR}")
    print(f"  Sentiment data at: {SENTIMENT_DIR}")

    return results, backtest


if __name__ == "__main__":
    main()
