"""
Sneaker Profit Prophet - Financial Terminal UI

Run with: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    ROOT_DIR, PROCESSED_DIR, SENTIMENT_DIR, MODEL_DIR,
    SEQUENCE_LENGTH, FORECAST_HORIZON,
)

st.set_page_config(page_title="Sneaker Profit Prophet", page_icon="📈", layout="wide")

st.title("Sneaker Profit Prophet")
st.markdown("*Predictive Modeling of Secondary Market Assets*")


@st.cache_data
def load_stockx_data():
    from scraping.stockx_scraper import load_kaggle_dataset, get_daily_prices
    csv_path = ROOT_DIR / "StockX-Data-Contest-2019-3.csv"
    if not csv_path.exists():
        return None, None, []
    df = load_kaggle_dataset(str(csv_path))
    brands = sorted(df["brand"].str.strip().unique().tolist())
    return df, None, brands


@st.cache_data
def get_brand_daily(brand):
    from scraping.stockx_scraper import load_kaggle_dataset, get_daily_prices
    csv_path = ROOT_DIR / "StockX-Data-Contest-2019-3.csv"
    df = load_kaggle_dataset(str(csv_path))
    return get_daily_prices(df, brand=brand)


@st.cache_data
def load_sentiment():
    vader_path = SENTIMENT_DIR / "daily_vader.csv"
    finbert_path = SENTIMENT_DIR / "daily_finbert.csv"
    vader = pd.read_csv(vader_path, parse_dates=["date"]) if vader_path.exists() else pd.DataFrame()
    finbert = pd.read_csv(finbert_path, parse_dates=["date"]) if finbert_path.exists() else pd.DataFrame()
    return vader, finbert


@st.cache_data
def load_backtest_results():
    portfolio_path = PROCESSED_DIR / "portfolio_history.csv"
    bh_path = PROCESSED_DIR / "buyhold_history.csv"
    trade_path = PROCESSED_DIR / "trade_log.csv"
    eval_path = PROCESSED_DIR / "evaluation_results.csv"

    portfolio = pd.read_csv(portfolio_path) if portfolio_path.exists() else pd.DataFrame()
    bh = pd.read_csv(bh_path) if bh_path.exists() else pd.DataFrame()
    trades = pd.read_csv(trade_path) if trade_path.exists() else pd.DataFrame()
    eval_results = pd.read_csv(eval_path, index_col=0) if eval_path.exists() else pd.DataFrame()
    return portfolio, bh, trades, eval_results


def run_forecast(prices, sentiments, model_type):
    import torch

    prices_arr = np.asarray(prices, dtype=np.float32)
    prices_arr = np.nan_to_num(prices_arr, nan=0.0, posinf=0.0, neginf=0.0)
    if prices_arr.size == 0:
        raise ValueError("No price history available for forecast")

    sentiments_arr = np.asarray(sentiments or [], dtype=np.float32)
    sentiments_arr = np.nan_to_num(sentiments_arr, nan=0.0, posinf=0.0, neginf=0.0)

    prices_window = prices_arr[-SEQUENCE_LENGTH:]
    if prices_window.size < SEQUENCE_LENGTH:
        pad_len = SEQUENCE_LENGTH - prices_window.size
        prices_window = np.concatenate([
            np.full(pad_len, float(prices_window[0]), dtype=np.float32),
            prices_window,
        ])

    sents_window = sentiments_arr[-SEQUENCE_LENGTH:]
    if sents_window.size < SEQUENCE_LENGTH:
        pad_len = SEQUENCE_LENGTH - sents_window.size
        sents_window = np.concatenate([np.zeros(pad_len, dtype=np.float32), sents_window])

    price_mean = float(prices_window.mean())
    price_std = float(prices_window.std())
    if price_std > 1e-8:
        price_norm = (prices_window - price_mean) / price_std
    else:
        price_norm = np.zeros_like(prices_window)

    return_1d = np.zeros_like(price_norm)
    prev = price_norm[:-1]
    delta = np.diff(price_norm)
    valid = np.abs(prev) > 1e-8
    return_1d[1:] = np.where(valid, delta / prev, 0.0)

    volatility_7d = np.zeros_like(return_1d)
    for i in range(return_1d.size):
        start = max(0, i - 6)
        window = return_1d[start : i + 1]
        volatility_7d[i] = float(window.std(ddof=1)) if window.size > 1 else 0.0

    seq = np.column_stack([price_norm, sents_window, return_1d, volatility_7d]).astype(np.float32)
    x = torch.from_numpy(seq).unsqueeze(0)

    if model_type == "LSTM":
        from models.lstm_model import SneakerLSTM
        model = SneakerLSTM(input_size=4)
        ckpt = MODEL_DIR / "lstm_best.pt"
    else:
        from models.gru_model import SneakerGRU
        model = SneakerGRU(input_size=4)
        ckpt = MODEL_DIR / "gru_best.pt"

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))

    model.eval()

    with torch.no_grad():
        pred_norm = model(x).squeeze().numpy()

    if np.ndim(pred_norm) == 0:
        pred_norm = np.array([float(pred_norm)])

    if price_std > 1e-8:
        return pred_norm * price_std + price_mean
    return np.full_like(pred_norm, price_mean)


# ── Sidebar ────────────────────────────────────────────
st.sidebar.header("Configuration")

stockx_df, _, brands = load_stockx_data()

if brands:
    brand = st.sidebar.selectbox("Select Brand", brands, index=0)
else:
    brand = "Yeezy"
    st.sidebar.warning("No data loaded. Run `python pipeline.py` first.")

model_type = st.sidebar.radio("Forecast Model", ["LSTM", "GRU"])
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 60, 30)
show_sentiment = st.sidebar.checkbox("Show Sentiment Overlay", value=True)

# ── Load Data ──────────────────────────────────────────
if brands:
    daily = get_brand_daily(brand)
    vader_sent, finbert_sent = load_sentiment()
    portfolio, bh, trades, eval_results = load_backtest_results()
else:
    daily = pd.DataFrame()
    vader_sent = pd.DataFrame()
    finbert_sent = pd.DataFrame()
    portfolio = pd.DataFrame()
    bh = pd.DataFrame()
    trades = pd.DataFrame()
    eval_results = pd.DataFrame()

# ── KPI Metrics ────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

if not daily.empty:
    current_price = daily["price"].iloc[-1]
    price_change = current_price - daily["price"].iloc[-2] if len(daily) > 1 else 0
    pct_change = (price_change / daily["price"].iloc[-2] * 100) if len(daily) > 1 else 0

    with col1:
        st.metric("Current Avg Price", f"${current_price:.0f}", f"{pct_change:+.1f}%")
    with col2:
        if not vader_sent.empty:
            latest_sent = vader_sent["sentiment_mean"].iloc[-1]
            st.metric("VADER Sentiment", f"{latest_sent:+.3f}",
                     "Positive" if latest_sent > 0.05 else "Negative" if latest_sent < -0.05 else "Neutral")
        else:
            st.metric("Sentiment", "N/A")
    with col3:
        if not portfolio.empty:
            final_val = portfolio["portfolio_value"].iloc[-1]
            ret = (final_val - 10000) / 10000 * 100
            st.metric("Agent Return", f"{ret:+.1f}%", f"${final_val:,.0f}")
        else:
            st.metric("Agent Return", "N/A")
    with col4:
        if not eval_results.empty and "mse" in eval_results.columns:
            model_key = model_type.lower()
            if model_key in eval_results.index:
                model_mse = eval_results.loc[model_key, "mse"]
                st.metric(f"{model_type} MSE", f"{model_mse:.6f}")
            else:
                st.metric(f"{model_type} MSE", "N/A")
        else:
            st.metric(f"{model_type} MSE", "Run pipeline")
else:
    with col1:
        st.metric("Current Price", "$--")
    with col2:
        st.metric("Sentiment", "--")
    with col3:
        st.metric("Agent Return", "--")
    with col4:
        st.metric("Model MSE", "--")

# ── Price Chart with Forecast ──────────────────────────
st.subheader("Price History & Forecast")

if not daily.empty:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price & Forecast", "Daily Volume"),
    )

    fig.add_trace(
        go.Scatter(
            x=daily["date"], y=daily["price"],
            name="Avg Sale Price",
            line=dict(color="#1f77b4", width=2),
        ),
        row=1, col=1,
    )

    if len(daily) >= SEQUENCE_LENGTH:
        prices = daily["price"].values
        sents = vader_sent["sentiment_mean"].values if not vader_sent.empty else np.zeros(len(prices))

        try:
            pred_prices = run_forecast(prices.tolist(), sents.tolist(), model_type)

            last_date = daily["date"].iloc[-1]
            forecast_dates = pd.date_range(last_date, periods=len(pred_prices) + 1, freq="D")[1:]

            fig.add_trace(
                go.Scatter(
                    x=forecast_dates, y=pred_prices,
                    name=f"{model_type} Forecast",
                    line=dict(color="#ff7f0e", width=2, dash="dash"),
                ),
                row=1, col=1,
            )
        except Exception as e:
            st.warning(f"Forecast unavailable: {e}")

    fig.add_trace(
        go.Bar(
            x=daily["date"], y=daily["volume"],
            name="Trade Volume",
            marker_color="rgba(31, 119, 180, 0.3)",
        ),
        row=2, col=1,
    )

    fig.update_layout(height=500, showlegend=True, template="plotly_white")
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No price data loaded. Run `python pipeline.py` to generate data.")

# ── Sentiment Timeline ─────────────────────────────────
if show_sentiment and not vader_sent.empty:
    st.subheader("Sentiment Analysis")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        fig_sent = go.Figure()
        fig_sent.add_trace(go.Scatter(
            x=vader_sent["date"], y=vader_sent["sentiment_mean"],
            name="VADER",
            line=dict(color="#2ca02c", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(44, 160, 44, 0.1)",
        ))
        if not finbert_sent.empty:
            fig_sent.add_trace(go.Scatter(
                x=finbert_sent["date"], y=finbert_sent["sentiment_mean"],
                name="FinBERT",
                line=dict(color="#d62728", width=1.5),
            ))
        fig_sent.update_layout(
            title="Daily Sentiment Score",
            yaxis_title="Sentiment [-1, 1]",
            height=300,
            template="plotly_white",
        )
        fig_sent.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_sent, use_container_width=True)

    with col_s2:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=vader_sent["date"], y=vader_sent["tweet_count"],
            name="Tweet Volume",
            marker_color="rgba(148, 103, 189, 0.6)",
        ))
        fig_vol.update_layout(
            title="Daily Tweet Volume",
            yaxis_title="Tweets",
            height=300,
            template="plotly_white",
        )
        st.plotly_chart(fig_vol, use_container_width=True)

# ── Trading Agent Performance ──────────────────────────
st.subheader("Agent Trading Performance")

if not portfolio.empty and not bh.empty:
    fig_perf = go.Figure()

    fig_perf.add_trace(go.Scatter(
        x=portfolio["date"], y=portfolio["portfolio_value"],
        name="Agent Strategy",
        line=dict(color="#1f77b4", width=2),
    ))
    fig_perf.add_trace(go.Scatter(
        x=bh["date"], y=bh["portfolio_value"],
        name="Buy & Hold",
        line=dict(color="#ff7f0e", width=2, dash="dot"),
    ))
    fig_perf.add_hline(y=10000, line_dash="dash", line_color="gray", annotation_text="Initial Capital")
    fig_perf.update_layout(
        yaxis_title="Portfolio Value ($)",
        height=350,
        template="plotly_white",
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    if not trades.empty:
        st.markdown("**Recent Trades**")
        st.dataframe(trades.tail(10), use_container_width=True)
else:
    st.info("Run `python pipeline.py` to generate backtest results.")

# ── Model Evaluation ───────────────────────────────────
st.subheader("Model vs Baseline Evaluation")

if not eval_results.empty:
    st.dataframe(eval_results.style.format("{:.6f}"), use_container_width=True)

    if "mse" in eval_results.columns:
        fig_mse = go.Figure()
        for idx in eval_results.index:
            fig_mse.add_trace(go.Bar(
                x=[idx.upper()],
                y=[eval_results.loc[idx, "mse"]],
                name=idx.upper(),
            ))
        fig_mse.update_layout(
            title="MSE Comparison: Models vs Random Walk",
            yaxis_title="MSE",
            height=300,
            template="plotly_white",
            showlegend=False,
        )
        st.plotly_chart(fig_mse, use_container_width=True)
else:
    st.info("Run `python pipeline.py` to generate evaluation metrics.")

# ── Footer ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "*Sneaker Profit Prophet* — LSTM/GRU forecasting + VADER/FinBERT sentiment + ReAct agent trading | "
    "Data: StockX Contest 2019 + Twitter"
)
