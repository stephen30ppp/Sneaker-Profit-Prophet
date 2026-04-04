# Sneaker Profit Prophet

Predictive modeling of secondary market sneaker assets using multivariate time-series analysis, sentiment scoring, and deep learning.

## Getting Started

### Prerequisites

- Python 3.10+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/stephen30ppp/Sneaker-Profit-Prophet.git
cd Sneaker-Profit-Prophet
```

### 2. Create a Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys:
```
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
```

### 5. Verify Installation

```bash
python -c "import torch; import pandas; import streamlit; print('All good!')"
```

### 6. Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
├── scraping/          # StockX & Twitter data collection
├── sentiment/         # VADER & FinBERT sentiment scoring
├── models/            # LSTM & GRU forecasting models (PyTorch)
├── backtesting/       # Baseline comparison & evaluation metrics
├── trading/           # Buy/Hold/Sell trading simulator
├── app/               # Streamlit UI
├── data/              # Raw, processed, and sentiment data (git-ignored)
├── notebooks/         # Exploratory analysis
├── config.py          # Global configuration & hyperparameters
└── requirements.txt   # Python dependencies
```
