"""Feature engineering for football match prediction."""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from . import config
from .utils import calculate_elo_change, implied_probability, points_from_result, remove_overround

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Builds features for each match based on historical team performance."""

    def __init__(self, rolling_window: int = config.ROLLING_WINDOW):
        self.window = rolling_window
        self.elo_ratings: dict[str, float] = defaultdict(lambda: config.ELO_INITIAL_RATING)
        self.team_stats: dict[str, list[dict]] = defaultdict(list)

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features for all matches in the DataFrame.

        Features are computed using only data available BEFORE each match (no leakage).
        """
        df = df.sort_values("Date").reset_index(drop=True)
        features_list = []

        # Reset state
        self.elo_ratings = defaultdict(lambda: config.ELO_INITIAL_RATING)
        self.team_stats = defaultdict(list)

        for idx, row in df.iterrows():
            home = row["Home"]
            away = row["Away"]

            # Compute features BEFORE updating stats with this match
            feats = self._compute_match_features(home, away, row)
            features_list.append(feats)

            # Update team stats with this match result
            self._update_stats(row)

        features_df = pd.DataFrame(features_list, index=df.index)
        return pd.concat([df, features_df], axis=1)

    def get_prediction_features(self, home: str, away: str, league_df: pd.DataFrame) -> dict:
        """Get features for a new prediction (after building from historical data)."""
        # Build all historical features first to populate state
        self.build_features(league_df)

        # Now compute features using current state
        dummy_row = pd.Series({
            "Date": pd.Timestamp.now(),
            "Home": home,
            "Away": away,
        })
        return self._compute_match_features(home, away, dummy_row)

    def _compute_match_features(self, home: str, away: str, row: pd.Series) -> dict:
        """Compute all features for a single match."""
        feats = {}

        # Team form features
        home_stats = self.team_stats[home]
        away_stats = self.team_stats[away]

        home_recent = home_stats[-self.window:] if home_stats else []
        away_recent = away_stats[-self.window:] if away_stats else []

        # Home team offensive features
        feats["home_goals_scored_avg"] = self._avg(home_recent, "goals_for")
        feats["home_goals_conceded_avg"] = self._avg(home_recent, "goals_against")
        feats["home_goals_scored_home_avg"] = self._avg(
            [s for s in home_recent if s["venue"] == "home"], "goals_for"
        )
        feats["home_goals_conceded_home_avg"] = self._avg(
            [s for s in home_recent if s["venue"] == "home"], "goals_against"
        )

        # Away team offensive features
        feats["away_goals_scored_avg"] = self._avg(away_recent, "goals_for")
        feats["away_goals_conceded_avg"] = self._avg(away_recent, "goals_against")
        feats["away_goals_scored_away_avg"] = self._avg(
            [s for s in away_recent if s["venue"] == "away"], "goals_for"
        )
        feats["away_goals_conceded_away_avg"] = self._avg(
            [s for s in away_recent if s["venue"] == "away"], "goals_against"
        )

        # Form (points)
        feats["home_points_last_n"] = sum(s["points"] for s in home_recent)
        feats["away_points_last_n"] = sum(s["points"] for s in away_recent)

        # Streaks
        feats["home_win_streak"] = self._current_streak(home_recent, "W")
        feats["home_loss_streak"] = self._current_streak(home_recent, "L")
        feats["home_unbeaten_run"] = self._unbeaten_run(home_recent)
        feats["away_win_streak"] = self._current_streak(away_recent, "W")
        feats["away_loss_streak"] = self._current_streak(away_recent, "L")
        feats["away_unbeaten_run"] = self._unbeaten_run(away_recent)

        # Clean sheets
        feats["home_clean_sheet_pct"] = self._pct(home_recent, "clean_sheet")
        feats["away_clean_sheet_pct"] = self._pct(away_recent, "clean_sheet")

        # BTTS rate
        feats["home_btts_pct"] = self._pct(home_recent, "btts")
        feats["away_btts_pct"] = self._pct(away_recent, "btts")

        # Over 2.5 rate
        feats["home_over25_pct"] = self._pct(home_recent, "over25")
        feats["away_over25_pct"] = self._pct(away_recent, "over25")

        # Elo ratings
        feats["home_elo"] = self.elo_ratings[home]
        feats["away_elo"] = self.elo_ratings[away]
        feats["elo_diff"] = feats["home_elo"] - feats["away_elo"]

        # Head-to-head
        h2h = self._get_h2h(home, away)
        feats["h2h_home_wins"] = h2h["home_wins"]
        feats["h2h_draws"] = h2h["draws"]
        feats["h2h_away_wins"] = h2h["away_wins"]
        feats["h2h_avg_goals"] = h2h["avg_goals"]
        feats["h2h_matches"] = h2h["total"]

        # League averages (attack/defense strength)
        all_stats = [s for stats_list in self.team_stats.values() for s in stats_list[-20:]]
        if all_stats:
            league_avg_goals = np.mean([s["goals_for"] for s in all_stats])
            if league_avg_goals > 0:
                feats["home_attack_strength"] = feats["home_goals_scored_avg"] / league_avg_goals
                feats["home_defense_strength"] = feats["home_goals_conceded_avg"] / league_avg_goals
                feats["away_attack_strength"] = feats["away_goals_scored_avg"] / league_avg_goals
                feats["away_defense_strength"] = feats["away_goals_conceded_avg"] / league_avg_goals
            else:
                feats["home_attack_strength"] = 1.0
                feats["home_defense_strength"] = 1.0
                feats["away_attack_strength"] = 1.0
                feats["away_defense_strength"] = 1.0
        else:
            feats["home_attack_strength"] = 1.0
            feats["home_defense_strength"] = 1.0
            feats["away_attack_strength"] = 1.0
            feats["away_defense_strength"] = 1.0

        # Odds-derived features (if available)
        feats.update(self._compute_odds_features(row))

        # Match count (sample size indicator)
        feats["home_match_count"] = len(home_stats)
        feats["away_match_count"] = len(away_stats)

        return feats

    def _update_stats(self, row: pd.Series) -> None:
        """Update team statistics after a match."""
        home, away = row["Home"], row["Away"]
        hg, ag = int(row["HG"]), int(row["AG"])
        result = row["Res"]

        # Determine result from each perspective
        if result == "H":
            home_result, away_result = "W", "L"
            elo_result = 1.0
        elif result == "D":
            home_result, away_result = "D", "D"
            elo_result = 0.5
        else:
            home_result, away_result = "L", "W"
            elo_result = 0.0

        # Update Elo
        new_home_elo, new_away_elo = calculate_elo_change(
            self.elo_ratings[home], self.elo_ratings[away],
            elo_result, config.ELO_K_FACTOR
        )
        self.elo_ratings[home] = new_home_elo
        self.elo_ratings[away] = new_away_elo

        total_goals = hg + ag

        # Home team stats
        self.team_stats[home].append({
            "date": row["Date"],
            "venue": "home",
            "opponent": away,
            "goals_for": hg,
            "goals_against": ag,
            "result": home_result,
            "points": points_from_result(result, "home"),
            "clean_sheet": 1 if ag == 0 else 0,
            "btts": 1 if hg > 0 and ag > 0 else 0,
            "over25": 1 if total_goals > 2.5 else 0,
        })

        # Away team stats
        self.team_stats[away].append({
            "date": row["Date"],
            "venue": "away",
            "opponent": home,
            "goals_for": ag,
            "goals_against": hg,
            "result": away_result,
            "points": points_from_result(result, "away"),
            "clean_sheet": 1 if hg == 0 else 0,
            "btts": 1 if hg > 0 and ag > 0 else 0,
            "over25": 1 if total_goals > 2.5 else 0,
        })

    def _get_h2h(self, home: str, away: str) -> dict:
        """Get head-to-head record between two teams."""
        home_stats = self.team_stats[home]
        h2h_matches = [s for s in home_stats if s["opponent"] == away]

        if not h2h_matches:
            return {"home_wins": 0, "draws": 0, "away_wins": 0, "avg_goals": 0.0, "total": 0}

        home_wins = sum(1 for s in h2h_matches if s["result"] == "W")
        draws = sum(1 for s in h2h_matches if s["result"] == "D")
        away_wins = sum(1 for s in h2h_matches if s["result"] == "L")
        avg_goals = np.mean([s["goals_for"] + s["goals_against"] for s in h2h_matches])

        return {
            "home_wins": home_wins,
            "draws": draws,
            "away_wins": away_wins,
            "avg_goals": round(avg_goals, 2),
            "total": len(h2h_matches),
        }

    def _compute_odds_features(self, row: pd.Series) -> dict:
        """Extract features from betting odds if available."""
        feats = {}
        odds_available = all(
            col in row.index and pd.notna(row.get(col))
            for col in ["AvgH", "AvgD", "AvgA"]
        )

        if odds_available:
            probs = {
                "home": implied_probability(row["AvgH"]),
                "draw": implied_probability(row["AvgD"]),
                "away": implied_probability(row["AvgA"]),
            }
            fair = remove_overround(probs)
            feats["odds_implied_home"] = fair.get("home", np.nan)
            feats["odds_implied_draw"] = fair.get("draw", np.nan)
            feats["odds_implied_away"] = fair.get("away", np.nan)
        else:
            feats["odds_implied_home"] = np.nan
            feats["odds_implied_draw"] = np.nan
            feats["odds_implied_away"] = np.nan

        return feats

    @staticmethod
    def _avg(stats: list[dict], key: str) -> float:
        """Calculate average of a stat across matches."""
        if not stats:
            return 0.0
        return np.mean([s[key] for s in stats])

    @staticmethod
    def _pct(stats: list[dict], key: str) -> float:
        """Calculate percentage of matches where a condition was true."""
        if not stats:
            return 0.0
        return np.mean([s[key] for s in stats])

    @staticmethod
    def _current_streak(stats: list[dict], result_type: str) -> int:
        """Count current streak of a specific result type (W/D/L)."""
        streak = 0
        for s in reversed(stats):
            if s["result"] == result_type:
                streak += 1
            else:
                break
        return streak

    @staticmethod
    def _unbeaten_run(stats: list[dict]) -> int:
        """Count matches since last defeat."""
        run = 0
        for s in reversed(stats):
            if s["result"] != "L":
                run += 1
            else:
                break
        return run


def get_feature_columns() -> list[str]:
    """Return list of feature column names used for ML training."""
    return [
        "home_goals_scored_avg", "home_goals_conceded_avg",
        "home_goals_scored_home_avg", "home_goals_conceded_home_avg",
        "away_goals_scored_avg", "away_goals_conceded_avg",
        "away_goals_scored_away_avg", "away_goals_conceded_away_avg",
        "home_points_last_n", "away_points_last_n",
        "home_win_streak", "home_loss_streak", "home_unbeaten_run",
        "away_win_streak", "away_loss_streak", "away_unbeaten_run",
        "home_clean_sheet_pct", "away_clean_sheet_pct",
        "home_btts_pct", "away_btts_pct",
        "home_over25_pct", "away_over25_pct",
        "home_elo", "away_elo", "elo_diff",
        "h2h_home_wins", "h2h_draws", "h2h_away_wins",
        "h2h_avg_goals", "h2h_matches",
        "home_attack_strength", "home_defense_strength",
        "away_attack_strength", "away_defense_strength",
        "home_match_count", "away_match_count",
    ]
