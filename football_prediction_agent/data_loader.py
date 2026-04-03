"""Data loading and caching for football match data from football-data.co.uk."""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config
from .utils import normalize_team_name, parse_date

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Date", "Home", "Away", "HG", "AG", "Res"]


def download_csv(url: str, dest: Path, max_retries: int = 3) -> Path:
    """Download a CSV file with retry logic and caching."""
    # Check cache freshness
    if dest.exists():
        mtime = datetime.fromtimestamp(dest.stat().st_mtime)
        if datetime.now() - mtime < timedelta(hours=config.CACHE_TTL_HOURS):
            logger.info(f"Using cached file: {dest}")
            return dest

    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(response.content)
            logger.info(f"Saved to {dest} ({len(response.content)} bytes)")
            return dest
        except requests.RequestException as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)

    if dest.exists():
        logger.warning(f"All download attempts failed. Using stale cache: {dest}")
        return dest

    # Fall back to generating sample data
    logger.warning(f"Download failed and no cache. Generating sample data for testing.")
    try:
        from .sample_data import save_sample_data
        league_code = dest.stem  # e.g. "ARG" from "ARG.csv"
        save_sample_data(league_code)
        if dest.exists():
            return dest
    except Exception as e:
        logger.error(f"Sample data generation also failed: {e}")

    raise RuntimeError(f"Failed to download {url} and no cache available")


def load_league_data(league_code: str) -> pd.DataFrame:
    """Load and preprocess data for a single league.

    Args:
        league_code: League identifier (ARG, BRA, MEX)

    Returns:
        Cleaned DataFrame with match data
    """
    if league_code not in config.LEAGUES:
        raise ValueError(f"Unknown league: {league_code}. Available: {list(config.LEAGUES)}")

    league_cfg = config.LEAGUES[league_code]
    csv_path = config.CACHE_DIR / f"{league_code}.csv"

    download_csv(league_cfg["url"], csv_path)

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows for {league_cfg['name']}")

    df = _clean_data(df, league_code)
    return df


def _clean_data(df: pd.DataFrame, league_code: str) -> pd.DataFrame:
    """Clean and validate match data."""
    # Validate required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {league_code}: {missing}")

    # Parse dates
    df["Date"] = df["Date"].apply(parse_date)
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Convert goals to numeric, drop invalid
    for col in ["HG", "AG"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["HG", "AG"])
    df["HG"] = df["HG"].astype(int)
    df["AG"] = df["AG"].astype(int)

    # Validate result column
    df["Res"] = df["Res"].str.strip().str.upper()
    df = df[df["Res"].isin(["H", "D", "A"])]

    # Normalize team names
    df["Home"] = df["Home"].apply(normalize_team_name)
    df["Away"] = df["Away"].apply(normalize_team_name)

    # Convert odds columns to numeric (may have missing values)
    odds_cols = ["PH", "PD", "PA", "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA"]
    for col in odds_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter to recent seasons
    if "Season" in df.columns:
        seasons = sorted(df["Season"].dropna().unique())
        if len(seasons) > config.SEASONS_TO_USE:
            recent = seasons[-config.SEASONS_TO_USE:]
            df = df[df["Season"].isin(recent)]
            logger.info(f"Filtered to seasons: {recent}")

    # Remove duplicate matches
    df = df.drop_duplicates(subset=["Date", "Home", "Away"], keep="last")

    # Add computed columns
    df["TotalGoals"] = df["HG"] + df["AG"]
    df["GoalDiff"] = df["HG"] - df["AG"]
    df["BTTS"] = ((df["HG"] > 0) & (df["AG"] > 0)).astype(int)
    df["LeagueCode"] = league_code

    logger.info(f"After cleaning: {len(df)} matches for {league_code}")
    return df


def load_all_leagues() -> dict[str, pd.DataFrame]:
    """Load data for all configured leagues.

    Returns:
        Dictionary mapping league code to DataFrame
    """
    data = {}
    for code in config.LEAGUES:
        try:
            data[code] = load_league_data(code)
        except Exception as e:
            logger.error(f"Failed to load {code}: {e}")
    return data


def get_upcoming_matches(df: pd.DataFrame, team: str | None = None) -> pd.DataFrame:
    """Get the most recent matches that could represent upcoming fixtures.

    Since football-data.co.uk provides historical data only, we identify the latest
    season's teams and create potential fixture pairs for prediction.
    """
    if "Season" in df.columns:
        latest_season = df["Season"].iloc[-1]
        season_df = df[df["Season"] == latest_season]
    else:
        # Use last 3 months of data
        cutoff = df["Date"].max() - timedelta(days=90)
        season_df = df[df["Date"] >= cutoff]

    teams = sorted(set(season_df["Home"].unique()) | set(season_df["Away"].unique()))

    if team:
        teams = [t for t in teams if team.lower() in t.lower()]

    return teams, season_df


def get_league_summary(df: pd.DataFrame) -> dict:
    """Generate a summary of the loaded league data."""
    return {
        "total_matches": len(df),
        "date_range": f"{df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d}",
        "seasons": sorted(df["Season"].unique().tolist()) if "Season" in df.columns else [],
        "teams": sorted(set(df["Home"].unique()) | set(df["Away"].unique())),
        "avg_goals_per_match": round(df["TotalGoals"].mean(), 2),
        "home_win_pct": round((df["Res"] == "H").mean() * 100, 1),
        "draw_pct": round((df["Res"] == "D").mean() * 100, 1),
        "away_win_pct": round((df["Res"] == "A").mean() * 100, 1),
        "btts_pct": round(df["BTTS"].mean() * 100, 1),
    }
