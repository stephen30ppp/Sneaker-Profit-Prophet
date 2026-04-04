"""
Sneaker Profit Prophet - Financial Terminal UI

Run with: streamlit run app/streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="Sneaker Profit Prophet", page_icon="📈", layout="wide")

st.title("Sneaker Profit Prophet")
st.markdown("*Predictive Modeling of Secondary Market Assets*")

# ── Sidebar ────────────────────────────────────────────
st.sidebar.header("Configuration")
shoe_model = st.sidebar.selectbox("Select Sneaker", ["-- Select --"])
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 30, 90, 30)
model_type = st.sidebar.radio("Model", ["LSTM", "GRU"])

# ── Main Dashboard ─────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Price", "$--", "--")
with col2:
    st.metric("Predicted Price (30d)", "$--", "--")
with col3:
    st.metric("Sentiment Score", "--", "--")

st.subheader("Price History & Forecast")
st.info("Load data and train model to see predictions.")

# TODO: Add price chart with plotly
# TODO: Add sentiment timeline
# TODO: Add trading signals visualization
# TODO: Add backtesting results section
