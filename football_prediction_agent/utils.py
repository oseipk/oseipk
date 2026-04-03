"""Utility functions for the Football Prediction Agent."""

import logging
import re
import unicodedata
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_team_name(name: str) -> str:
    """Normalize team name by removing accents and standardizing whitespace."""
    if not isinstance(name, str):
        return str(name)
    # Remove accents
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ASCII", "ignore").decode("ASCII")
    # Standardize whitespace
    ascii_name = re.sub(r"\s+", " ", ascii_name).strip()
    return ascii_name


def parse_date(date_str: str) -> datetime | None:
    """Parse date string from football-data.co.uk CSV (DD/MM/YYYY)."""
    if not isinstance(date_str, str) or not date_str.strip():
        return None
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    logger.warning(f"Could not parse date: {date_str}")
    return None


def implied_probability(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if not isinstance(odds, (int, float)) or np.isnan(odds) or odds <= 1.0:
        return np.nan
    return 1.0 / odds


def remove_overround(probs: dict[str, float]) -> dict[str, float]:
    """Remove bookmaker overround from implied probabilities to get fair probs."""
    total = sum(p for p in probs.values() if not np.isnan(p))
    if total <= 0:
        return probs
    return {k: v / total for k, v in probs.items()}


def calculate_elo_change(
    rating_a: float, rating_b: float, result: float, k: float = 20.0
) -> tuple[float, float]:
    """Calculate Elo rating changes.

    Args:
        rating_a: Current rating of team A
        rating_b: Current rating of team B
        result: 1.0 for A win, 0.5 for draw, 0.0 for A loss
        k: K-factor controlling volatility

    Returns:
        Tuple of (new_rating_a, new_rating_b)
    """
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1.0 - expected_a

    new_a = rating_a + k * (result - expected_a)
    new_b = rating_b + k * ((1.0 - result) - expected_b)

    return new_a, new_b


def points_from_result(result: str, perspective: str = "home") -> int:
    """Convert result code (H/D/A) to points for given perspective."""
    if perspective == "home":
        return {"H": 3, "D": 1, "A": 0}.get(result, 0)
    else:
        return {"H": 0, "D": 1, "A": 3}.get(result, 0)


def get_confidence_level(max_prob: float) -> str:
    """Determine confidence level based on highest predicted probability."""
    from . import config
    if max_prob >= config.CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    elif max_prob >= config.CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    return "low"
