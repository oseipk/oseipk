"""Configuration for the Football Prediction Agent."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# League configurations
LEAGUES = {
    "ARG": {
        "name": "Argentina Primera Division",
        "country": "Argentina",
        "url": "https://www.football-data.co.uk/new/ARG.csv",
    },
    "BRA": {
        "name": "Brazil Serie A",
        "country": "Brazil",
        "url": "https://www.football-data.co.uk/new/BRA.csv",
    },
    "MEX": {
        "name": "Mexico Liga MX",
        "country": "Mexico",
        "url": "https://www.football-data.co.uk/new/MEX.csv",
    },
}

# Feature engineering
ROLLING_WINDOW = 10
MIN_MATCHES_FOR_PREDICTION = 5
ELO_K_FACTOR = 20
ELO_INITIAL_RATING = 1500

# Model settings
MAX_GOALS = 7  # Maximum goals to consider in Poisson matrix (0-7)
ENSEMBLE_WEIGHTS = {
    "poisson": 0.4,
    "ml": 0.6,
}
SEASONS_TO_USE = 5  # Number of recent seasons for training

# Data caching
CACHE_TTL_HOURS = 24

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "high": 0.55,    # Highest predicted probability > 55%
    "medium": 0.40,  # Highest predicted probability > 40%
    "low": 0.0,      # Everything else
}

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
