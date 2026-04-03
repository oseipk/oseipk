"""Main orchestrator for the Football Prediction Agent."""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data_loader import get_league_summary, load_all_leagues, load_league_data
from .feature_engineering import FeatureBuilder
from .markets import MarketPredictor
from .models import EnsemblePredictor
from .utils import get_confidence_level, normalize_team_name, setup_logging

logger = logging.getLogger(__name__)


class FootballPredictionAgent:
    """End-to-end football match prediction agent.

    Orchestrates data loading, model training, and prediction generation
    across multiple Latin American leagues.
    """

    def __init__(self):
        self.league_data: dict[str, pd.DataFrame] = {}
        self.models: dict[str, EnsemblePredictor] = {}
        self.market_predictor = MarketPredictor()
        self._initialized = False

    def initialize(self, leagues: list[str] | None = None) -> None:
        """Load data and train models for specified leagues.

        Args:
            leagues: List of league codes. If None, uses all configured leagues.
        """
        setup_logging(config.LOG_LEVEL)
        target_leagues = leagues or list(config.LEAGUES.keys())

        logger.info(f"Initializing agent for leagues: {target_leagues}")

        # Step 1: Load data
        logger.info("=" * 60)
        logger.info("STEP 1: Loading historical match data")
        logger.info("=" * 60)

        for code in target_leagues:
            try:
                self.league_data[code] = load_league_data(code)
                summary = get_league_summary(self.league_data[code])
                logger.info(
                    f"  {config.LEAGUES[code]['name']}: {summary['total_matches']} matches, "
                    f"{summary['date_range']}, "
                    f"avg {summary['avg_goals_per_match']} goals/match"
                )
            except Exception as e:
                logger.error(f"  Failed to load {code}: {e}")

        if not self.league_data:
            raise RuntimeError("No league data could be loaded")

        # Step 2: Train models
        logger.info("=" * 60)
        logger.info("STEP 2: Training prediction models")
        logger.info("=" * 60)

        for code, df in self.league_data.items():
            logger.info(f"  Training models for {config.LEAGUES[code]['name']}...")
            model = EnsemblePredictor()
            model.fit(df)
            self.models[code] = model

        self._initialized = True
        logger.info("=" * 60)
        logger.info("Agent initialization complete!")
        logger.info("=" * 60)

    def predict_match(self, league_code: str, home_team: str, away_team: str) -> dict:
        """Generate full prediction for a single match.

        Args:
            league_code: League identifier (ARG, BRA, MEX)
            home_team: Home team name
            away_team: Away team name

        Returns:
            Complete prediction with all markets
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        if league_code not in self.models:
            raise ValueError(f"No model for league {league_code}")

        # Normalize names
        home_team = normalize_team_name(home_team)
        away_team = normalize_team_name(away_team)

        # Find best matching team names
        home_team = self._find_team(league_code, home_team)
        away_team = self._find_team(league_code, away_team)

        logger.info(f"Predicting: {home_team} vs {away_team} ({config.LEAGUES[league_code]['name']})")

        # Get model prediction
        model = self.models[league_code]
        model_output = model.predict_match(home_team, away_team)

        # Generate all market predictions
        markets = self.market_predictor.predict_all_markets(model_output)

        # Determine confidence
        max_prob = max(markets["match_result"].values())
        confidence = get_confidence_level(max_prob)

        prediction = {
            "match": {
                "league": config.LEAGUES[league_code]["name"],
                "league_code": league_code,
                "home_team": home_team,
                "away_team": away_team,
                "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "predictions": markets,
            "confidence": confidence,
            "model_metadata": {
                "poisson_lambda_home": model_output["lambda_home"],
                "poisson_lambda_away": model_output["lambda_away"],
                "ensemble_weights": {
                    "poisson": model.poisson_weight,
                    "ml": model.ml_weight,
                },
                "training_matches": len(self.league_data[league_code]),
                "poisson_result": model_output.get("poisson_result"),
                "ml_result": model_output.get("ml_result"),
            },
        }

        return prediction

    def predict_league_fixtures(self, league_code: str, n_fixtures: int = 10) -> list[dict]:
        """Predict upcoming fixtures for a league.

        Uses the latest season's teams to generate fixture predictions
        for the most recent unplayed matchups.
        """
        if league_code not in self.league_data:
            raise ValueError(f"No data for league {league_code}")

        df = self.league_data[league_code]

        # Get latest season teams
        if "Season" in df.columns:
            latest_season = df["Season"].iloc[-1]
            season_df = df[df["Season"] == latest_season]
        else:
            season_df = df.tail(200)

        teams = sorted(set(season_df["Home"].unique()) | set(season_df["Away"].unique()))

        # Find matchups from the latest round
        recent = season_df.tail(n_fixtures * 2)
        played_pairs = set()
        for _, row in recent.iterrows():
            played_pairs.add((row["Home"], row["Away"]))

        # Generate predictions for plausible next fixtures
        predictions = []
        fixture_count = 0

        for i, home in enumerate(teams):
            for away in teams[i + 1:]:
                if fixture_count >= n_fixtures:
                    break
                if (home, away) not in played_pairs:
                    try:
                        pred = self.predict_match(league_code, home, away)
                        predictions.append(pred)
                        fixture_count += 1
                    except Exception as e:
                        logger.warning(f"Skipping {home} vs {away}: {e}")
            if fixture_count >= n_fixtures:
                break

        return predictions

    def evaluate_model(self, league_code: str) -> dict:
        """Run backtesting evaluation for a league's model."""
        if league_code not in self.league_data:
            raise ValueError(f"No data for league {league_code}")

        model = self.models.get(league_code)
        if not model:
            raise ValueError(f"No model trained for {league_code}")

        logger.info(f"Evaluating model for {config.LEAGUES[league_code]['name']}...")
        return model.evaluate(self.league_data[league_code])

    def get_league_teams(self, league_code: str) -> list[str]:
        """Get list of teams in a league's current season."""
        if league_code not in self.league_data:
            raise ValueError(f"No data for league {league_code}")

        df = self.league_data[league_code]
        if "Season" in df.columns:
            latest = df["Season"].iloc[-1]
            df = df[df["Season"] == latest]

        return sorted(set(df["Home"].unique()) | set(df["Away"].unique()))

    def _find_team(self, league_code: str, team_name: str) -> str:
        """Find the closest matching team name in the league."""
        df = self.league_data[league_code]
        all_teams = set(df["Home"].unique()) | set(df["Away"].unique())

        # Exact match
        if team_name in all_teams:
            return team_name

        # Case-insensitive match
        for t in all_teams:
            if t.lower() == team_name.lower():
                return t

        # Substring match
        matches = [t for t in all_teams if team_name.lower() in t.lower()]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            logger.warning(f"Multiple matches for '{team_name}': {matches}. Using first.")
            return matches[0]

        raise ValueError(
            f"Team '{team_name}' not found in {league_code}. "
            f"Available teams: {sorted(all_teams)}"
        )

    def save_predictions(self, predictions: list[dict], filepath: str | None = None) -> Path:
        """Save predictions to a JSON file."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = config.OUTPUT_DIR / f"predictions_{timestamp}.json"
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON serialization
        clean = json.loads(json.dumps(predictions, default=_json_serializer))

        with open(filepath, "w") as f:
            json.dump(clean, f, indent=2)

        logger.info(f"Predictions saved to {filepath}")
        return filepath


def format_prediction(pred: dict) -> str:
    """Format a single prediction as a human-readable string."""
    match = pred["match"]
    p = pred["predictions"]
    meta = pred["model_metadata"]

    lines = [
        f"\n{'='*70}",
        f"  {match['home_team']} vs {match['away_team']}",
        f"  {match['league']} | Confidence: {pred['confidence'].upper()}",
        f"{'='*70}",
        "",
        f"  Match Result (1X2):",
        f"    Home: {p['match_result']['home']:.1%}  |  "
        f"Draw: {p['match_result']['draw']:.1%}  |  "
        f"Away: {p['match_result']['away']:.1%}",
        "",
        f"  Correct Score (Top 5):",
    ]

    for cs in p["correct_score"][:5]:
        bar = "#" * int(cs["probability"] * 100)
        lines.append(f"    {cs['score']:>5s}  {cs['probability']:6.1%}  {bar}")

    lines.extend([
        "",
        f"  Over/Under Goals:",
        f"    O1.5: {p['over_under']['over_1.5']:.1%}  |  "
        f"O2.5: {p['over_under']['over_2.5']:.1%}  |  "
        f"O3.5: {p['over_under']['over_3.5']:.1%}",
        "",
        f"  BTTS: Yes {p['btts']['yes']:.1%}  |  No {p['btts']['no']:.1%}",
        "",
        f"  Double Chance:",
        f"    1X: {p['double_chance']['1X']:.1%}  |  "
        f"X2: {p['double_chance']['X2']:.1%}  |  "
        f"12: {p['double_chance']['12']:.1%}",
        "",
        f"  Draw No Bet:",
        f"    Home: {p['draw_no_bet']['home']:.1%}  |  Away: {p['draw_no_bet']['away']:.1%}",
        "",
        f"  Goal Range:",
    ])
    for rng, prob in p["goal_range"].items():
        lines.append(f"    {rng}: {prob:.1%}")

    lines.extend([
        "",
        f"  Winning Margin:",
    ])
    for margin, prob in p["winning_margin"].items():
        lines.append(f"    {margin}: {prob:.1%}")

    if p.get("asian_handicap"):
        lines.extend(["", "  Asian Handicap:"])
        for line_key in ["home_-1.5", "home_-0.5", "home_+0.5", "home_+1.5"]:
            if line_key in p["asian_handicap"]:
                lines.append(f"    {line_key}: {p['asian_handicap'][line_key]:.1%}")

    lines.extend([
        "",
        f"  Model Info:",
        f"    Expected Goals: {meta['poisson_lambda_home']:.2f} - {meta['poisson_lambda_away']:.2f}",
        f"    Weights: Poisson {meta['ensemble_weights']['poisson']:.0%} / "
        f"ML {meta['ensemble_weights']['ml']:.0%}",
        f"    Training data: {meta['training_matches']} matches",
        "",
    ])

    return "\n".join(lines)


def _json_serializer(obj):
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
