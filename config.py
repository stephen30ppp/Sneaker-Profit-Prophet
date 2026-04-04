"""
Sneaker Profit Prophet - Global Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SENTIMENT_DIR = DATA_DIR / "sentiment"
MODEL_DIR = ROOT_DIR / "models" / "checkpoints"

# ── API Keys (from .env) ──────────────────────────────
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

# ── Model Hyperparameters ─────────────────────────────
SEQUENCE_LENGTH = 30          # lookback window (days)
FORECAST_HORIZON = 30         # prediction horizon (days)
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# ── Trading Simulator ────────────────────────────────
BUY_THRESHOLD = 0.6           # confidence above this → Buy
SELL_THRESHOLD = 0.4          # confidence below this → Sell
INITIAL_CAPITAL = 10000.0     # mock portfolio starting capital ($)

# ── Sentiment ────────────────────────────────────────
SENTIMENT_MODEL = "vader"     # "vader" or "finbert"
