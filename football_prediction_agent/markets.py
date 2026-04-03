"""Betting market predictions derived from model outputs."""

import logging

import numpy as np
from scipy.stats import poisson

from . import config

logger = logging.getLogger(__name__)


class MarketPredictor:
    """Generates predictions for various betting markets from model output."""

    def __init__(self, max_goals: int = config.MAX_GOALS):
        self.max_goals = max_goals

    def predict_all_markets(self, model_output: dict) -> dict:
        """Generate predictions for all supported markets.

        Args:
            model_output: Output from EnsemblePredictor.predict_match()

        Returns:
            Dictionary with predictions for each market
        """
        score_matrix = model_output["score_matrix"]
        match_result = model_output["match_result"]
        lambda_home = model_output["lambda_home"]
        lambda_away = model_output["lambda_away"]

        markets = {}

        # 1X2 (Match Result)
        markets["match_result"] = match_result

        # Correct Score
        markets["correct_score"] = self._correct_score(score_matrix)

        # Over/Under Goals
        markets["over_under"] = self._over_under(score_matrix)

        # Both Teams to Score
        markets["btts"] = self._btts(score_matrix)

        # Double Chance
        markets["double_chance"] = self._double_chance(match_result)

        # Draw No Bet
        markets["draw_no_bet"] = self._draw_no_bet(match_result)

        # Asian Handicap
        markets["asian_handicap"] = self._asian_handicap(score_matrix)

        # Half-Time / Full-Time (approximation using Poisson)
        markets["ht_ft"] = self._ht_ft(lambda_home, lambda_away)

        # Goal Range
        markets["goal_range"] = self._goal_range(score_matrix)

        # Exact Total Goals
        markets["exact_total_goals"] = self._exact_total_goals(score_matrix)

        # First Team to Score (approximation)
        markets["first_to_score"] = self._first_to_score(lambda_home, lambda_away)

        # Winning Margin
        markets["winning_margin"] = self._winning_margin(score_matrix)

        return markets

    def _correct_score(self, matrix: np.ndarray, top_n: int = 10) -> list[dict]:
        """Get top N most likely correct scores."""
        scores = []
        max_g = matrix.shape[0]

        for i in range(max_g):
            for j in range(max_g):
                scores.append({
                    "score": f"{i}-{j}",
                    "home_goals": i,
                    "away_goals": j,
                    "probability": round(float(matrix[i][j]), 4),
                })

        scores.sort(key=lambda x: x["probability"], reverse=True)
        return scores[:top_n]

    def _over_under(self, matrix: np.ndarray) -> dict:
        """Calculate over/under probabilities for various goal lines."""
        max_g = matrix.shape[0]
        result = {}

        for line in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
            over_prob = 0.0
            for i in range(max_g):
                for j in range(max_g):
                    if i + j > line:
                        over_prob += matrix[i][j]
            result[f"over_{line}"] = round(float(over_prob), 4)
            result[f"under_{line}"] = round(1.0 - float(over_prob), 4)

        return result

    def _btts(self, matrix: np.ndarray) -> dict:
        """Both Teams to Score probability."""
        max_g = matrix.shape[0]
        btts_prob = 0.0

        for i in range(1, max_g):
            for j in range(1, max_g):
                btts_prob += matrix[i][j]

        return {
            "yes": round(float(btts_prob), 4),
            "no": round(1.0 - float(btts_prob), 4),
        }

    def _double_chance(self, match_result: dict) -> dict:
        """Double chance market derived from 1X2."""
        return {
            "1X": round(match_result["home"] + match_result["draw"], 4),
            "X2": round(match_result["draw"] + match_result["away"], 4),
            "12": round(match_result["home"] + match_result["away"], 4),
        }

    def _draw_no_bet(self, match_result: dict) -> dict:
        """Draw no bet market (excluding draw probability)."""
        non_draw = match_result["home"] + match_result["away"]
        if non_draw <= 0:
            return {"home": 0.5, "away": 0.5}
        return {
            "home": round(match_result["home"] / non_draw, 4),
            "away": round(match_result["away"] / non_draw, 4),
        }

    def _asian_handicap(self, matrix: np.ndarray) -> dict:
        """Asian handicap probabilities for common lines."""
        max_g = matrix.shape[0]
        result = {}

        for handicap in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
            home_cover = 0.0
            for i in range(max_g):
                for j in range(max_g):
                    goal_diff = i - j
                    if goal_diff + handicap > 0:
                        home_cover += matrix[i][j]

            label = f"home_{handicap:+.1f}"
            result[label] = round(float(home_cover), 4)
            result[f"away_{-handicap:+.1f}"] = round(1.0 - float(home_cover), 4)

        return result

    def _ht_ft(self, lambda_home: float, lambda_away: float) -> list[dict]:
        """Half-time / Full-time predictions using half-match Poisson.

        Approximates HT goals as half the full-time lambda.
        """
        ht_lambda_h = lambda_home / 2
        ht_lambda_a = lambda_away / 2

        combinations = []
        for ht_res in ["H", "D", "A"]:
            for ft_res in ["H", "D", "A"]:
                prob = self._ht_ft_prob(ht_lambda_h, ht_lambda_a, ht_res, ft_res,
                                         lambda_home, lambda_away)
                combinations.append({
                    "ht": ht_res,
                    "ft": ft_res,
                    "label": f"{ht_res}/{ft_res}",
                    "probability": round(prob, 4),
                })

        combinations.sort(key=lambda x: x["probability"], reverse=True)
        return combinations

    def _ht_ft_prob(
        self, ht_lam_h: float, ht_lam_a: float, ht_res: str, ft_res: str,
        ft_lam_h: float, ft_lam_a: float
    ) -> float:
        """Calculate probability of a specific HT/FT combination."""
        max_g = 5  # Reduced for HT computation

        # HT probability
        ht_prob = 0.0
        for i in range(max_g):
            for j in range(max_g):
                if self._matches_result(i, j, ht_res):
                    ht_prob += poisson.pmf(i, ht_lam_h) * poisson.pmf(j, ht_lam_a)

        # FT probability (independent approximation)
        ft_prob = 0.0
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                if self._matches_result(i, j, ft_res):
                    ft_prob += poisson.pmf(i, ft_lam_h) * poisson.pmf(j, ft_lam_a)

        # Rough joint probability (simplified)
        return ht_prob * ft_prob

    @staticmethod
    def _matches_result(home_goals: int, away_goals: int, result: str) -> bool:
        """Check if a scoreline matches a result code."""
        if result == "H":
            return home_goals > away_goals
        elif result == "D":
            return home_goals == away_goals
        else:
            return home_goals < away_goals

    def _goal_range(self, matrix: np.ndarray) -> dict:
        """Probability of total goals falling in specific ranges."""
        max_g = matrix.shape[0]
        ranges = {"0-1": 0, "2-3": 0, "4-5": 0, "6+": 0}

        for i in range(max_g):
            for j in range(max_g):
                total = i + j
                if total <= 1:
                    ranges["0-1"] += matrix[i][j]
                elif total <= 3:
                    ranges["2-3"] += matrix[i][j]
                elif total <= 5:
                    ranges["4-5"] += matrix[i][j]
                else:
                    ranges["6+"] += matrix[i][j]

        return {k: round(float(v), 4) for k, v in ranges.items()}

    def _exact_total_goals(self, matrix: np.ndarray) -> dict:
        """Probability of exact total goals (0 through 6+)."""
        max_g = matrix.shape[0]
        result = {}

        for total_target in range(7):
            prob = 0.0
            for i in range(max_g):
                j = total_target - i
                if 0 <= j < max_g:
                    prob += matrix[i][j]
            result[str(total_target)] = round(float(prob), 4)

        # 7+ goals
        prob_7plus = 0.0
        for i in range(max_g):
            for j in range(max_g):
                if i + j >= 7:
                    prob_7plus += matrix[i][j]
        result["7+"] = round(float(prob_7plus), 4)

        return result

    def _first_to_score(self, lambda_home: float, lambda_away: float) -> dict:
        """Approximate probability of which team scores first."""
        total_lambda = lambda_home + lambda_away
        if total_lambda <= 0:
            return {"home": 0.5, "away": 0.5, "no_goal": 0.0}

        # Probability of 0-0
        no_goal = float(poisson.pmf(0, lambda_home) * poisson.pmf(0, lambda_away))

        # Among games with goals, proportion based on scoring rates
        home_first = (lambda_home / total_lambda) * (1 - no_goal)
        away_first = (lambda_away / total_lambda) * (1 - no_goal)

        return {
            "home": round(home_first, 4),
            "away": round(away_first, 4),
            "no_goal": round(no_goal, 4),
        }

    def _winning_margin(self, matrix: np.ndarray) -> dict:
        """Probability distribution of winning margin."""
        max_g = matrix.shape[0]
        margins = {}

        for margin_key, label in [
            ("home_1", "Home by 1"), ("home_2", "Home by 2"), ("home_3+", "Home by 3+"),
            ("draw", "Draw"),
            ("away_1", "Away by 1"), ("away_2", "Away by 2"), ("away_3+", "Away by 3+"),
        ]:
            prob = 0.0
            for i in range(max_g):
                for j in range(max_g):
                    diff = i - j
                    if margin_key == "home_1" and diff == 1:
                        prob += matrix[i][j]
                    elif margin_key == "home_2" and diff == 2:
                        prob += matrix[i][j]
                    elif margin_key == "home_3+" and diff >= 3:
                        prob += matrix[i][j]
                    elif margin_key == "draw" and diff == 0:
                        prob += matrix[i][j]
                    elif margin_key == "away_1" and diff == -1:
                        prob += matrix[i][j]
                    elif margin_key == "away_2" and diff == -2:
                        prob += matrix[i][j]
                    elif margin_key == "away_3+" and diff <= -3:
                        prob += matrix[i][j]
            margins[label] = round(float(prob), 4)

        return margins
