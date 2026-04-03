"""Prediction models: Poisson regression and Gradient Boosting ensemble."""

import logging
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

from . import config
from .feature_engineering import FeatureBuilder, get_feature_columns

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)


class BasePredictor(ABC):
    """Abstract base class for prediction models."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Train the model on historical match data."""

    @abstractmethod
    def predict_match(self, home: str, away: str) -> dict:
        """Predict probabilities for a single match."""


class PoissonModel(BasePredictor):
    """Bivariate Poisson model for goal prediction.

    Estimates expected goals for each team based on attack strength,
    defense strength, and home advantage, then generates a probability
    matrix for all scoreline combinations.
    """

    def __init__(self, max_goals: int = config.MAX_GOALS):
        self.max_goals = max_goals
        self.league_avg_home_goals = 1.5
        self.league_avg_away_goals = 1.2
        self.home_advantage = 1.0
        self.feature_builder = FeatureBuilder()
        self._fitted = False
        self._df = None

    def fit(self, df: pd.DataFrame) -> None:
        """Compute league-wide averages and team strengths from historical data."""
        self._df = df.copy()

        # League averages
        self.league_avg_home_goals = df["HG"].mean()
        self.league_avg_away_goals = df["AG"].mean()
        self.home_advantage = self.league_avg_home_goals / max(self.league_avg_away_goals, 0.1)

        # Build features to populate Elo and team stats
        self.feature_builder = FeatureBuilder()
        self.feature_builder.build_features(df)

        self._fitted = True
        logger.info(
            f"Poisson model fitted: avg home goals={self.league_avg_home_goals:.2f}, "
            f"avg away goals={self.league_avg_away_goals:.2f}, "
            f"home advantage={self.home_advantage:.2f}"
        )

    def predict_match(self, home: str, away: str) -> dict:
        """Predict match using Poisson model."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get team strengths from feature builder state
        home_stats = self.feature_builder.team_stats.get(home, [])
        away_stats = self.feature_builder.team_stats.get(away, [])

        if len(home_stats) < config.MIN_MATCHES_FOR_PREDICTION:
            logger.warning(f"{home} has only {len(home_stats)} matches, using league average")
        if len(away_stats) < config.MIN_MATCHES_FOR_PREDICTION:
            logger.warning(f"{away} has only {len(away_stats)} matches, using league average")

        # Calculate attack and defense strengths
        home_attack = self._attack_strength(home_stats, "home")
        home_defense = self._defense_strength(home_stats, "home")
        away_attack = self._attack_strength(away_stats, "away")
        away_defense = self._defense_strength(away_stats, "away")

        # Expected goals
        lambda_home = self.league_avg_home_goals * home_attack * away_defense
        lambda_away = self.league_avg_away_goals * away_attack * home_defense

        # Clamp to reasonable range
        lambda_home = np.clip(lambda_home, 0.2, 5.0)
        lambda_away = np.clip(lambda_away, 0.2, 5.0)

        # Build score probability matrix
        score_matrix = self._build_score_matrix(lambda_home, lambda_away)

        # Derive probabilities from matrix
        home_win_prob = np.sum(np.tril(score_matrix, -1))
        draw_prob = np.sum(np.diag(score_matrix))
        away_win_prob = np.sum(np.triu(score_matrix, 1))

        # Normalize
        total = home_win_prob + draw_prob + away_win_prob
        if total > 0:
            home_win_prob /= total
            draw_prob /= total
            away_win_prob /= total

        return {
            "match_result": {
                "home": round(home_win_prob, 4),
                "draw": round(draw_prob, 4),
                "away": round(away_win_prob, 4),
            },
            "score_matrix": score_matrix,
            "lambda_home": round(lambda_home, 3),
            "lambda_away": round(lambda_away, 3),
        }

    def _build_score_matrix(self, lambda_home: float, lambda_away: float) -> np.ndarray:
        """Build probability matrix for all scoreline combinations."""
        max_g = self.max_goals + 1
        matrix = np.zeros((max_g, max_g))

        for i in range(max_g):
            for j in range(max_g):
                matrix[i][j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)

        return matrix

    def _attack_strength(self, stats: list[dict], venue: str) -> float:
        """Calculate attack strength relative to league average."""
        recent = [s for s in stats[-config.ROLLING_WINDOW:] if s["venue"] == venue]
        if not recent:
            recent = stats[-config.ROLLING_WINDOW:]
        if not recent:
            return 1.0

        avg_goals = np.mean([s["goals_for"] for s in recent])
        league_avg = self.league_avg_home_goals if venue == "home" else self.league_avg_away_goals
        return avg_goals / max(league_avg, 0.1)

    def _defense_strength(self, stats: list[dict], venue: str) -> float:
        """Calculate defense strength (goals conceded relative to league avg)."""
        recent = [s for s in stats[-config.ROLLING_WINDOW:] if s["venue"] == venue]
        if not recent:
            recent = stats[-config.ROLLING_WINDOW:]
        if not recent:
            return 1.0

        avg_conceded = np.mean([s["goals_against"] for s in recent])
        league_avg = self.league_avg_away_goals if venue == "home" else self.league_avg_home_goals
        return avg_conceded / max(league_avg, 0.1)


class GradientBoostingModel(BasePredictor):
    """XGBoost-based model for match outcome and goal prediction."""

    def __init__(self):
        self.result_model = None
        self.goals_model = None
        self.feature_builder = FeatureBuilder()
        self.label_encoder = LabelEncoder()
        self._fitted = False
        self._df = None
        self._feature_cols = get_feature_columns()

    def fit(self, df: pd.DataFrame) -> None:
        """Train gradient boosting models on feature-enriched data."""
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError:
            logger.warning("XGBoost not available, falling back to sklearn")
            from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
            from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor

        self._df = df.copy()

        # Build features
        self.feature_builder = FeatureBuilder()
        featured_df = self.feature_builder.build_features(df)

        # Filter to matches where teams have enough history
        mask = (
            (featured_df["home_match_count"] >= config.MIN_MATCHES_FOR_PREDICTION) &
            (featured_df["away_match_count"] >= config.MIN_MATCHES_FOR_PREDICTION)
        )
        train_df = featured_df[mask].copy()

        if len(train_df) < 50:
            logger.warning(f"Only {len(train_df)} training samples, model may be unreliable")

        # Prepare features - exclude odds features (may not be available for prediction)
        available_cols = [c for c in self._feature_cols if c in train_df.columns]
        X = train_df[available_cols].fillna(0)
        y_result = self.label_encoder.fit_transform(train_df["Res"])
        y_goals = train_df["TotalGoals"]

        # Train result classifier
        try:
            self.result_model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="mlogloss",
            )
            self.result_model.fit(X, y_result)
        except TypeError:
            # Fallback for sklearn GradientBoostingClassifier
            self.result_model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                min_samples_leaf=3,
                subsample=0.8,
                random_state=42,
            )
            self.result_model.fit(X, y_result)

        # Train total goals regressor
        try:
            self.goals_model = XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
            self.goals_model.fit(X, y_goals)
        except TypeError:
            self.goals_model = XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                min_samples_leaf=3,
                subsample=0.8,
                random_state=42,
            )
            self.goals_model.fit(X, y_goals)

        self._fitted = True
        logger.info(f"Gradient Boosting model fitted on {len(train_df)} matches")

    def predict_match(self, home: str, away: str) -> dict:
        """Predict match result and total goals."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get features for this matchup
        feats = self.feature_builder.get_prediction_features(home, away, self._df)

        available_cols = [c for c in self._feature_cols if c in feats]
        X = pd.DataFrame([{c: feats.get(c, 0) for c in available_cols}]).fillna(0)

        # Predict result probabilities
        probs = self.result_model.predict_proba(X)[0]
        classes = self.label_encoder.classes_

        result_probs = {}
        for cls, prob in zip(classes, probs):
            if cls == "A":
                result_probs["away"] = round(float(prob), 4)
            elif cls == "D":
                result_probs["draw"] = round(float(prob), 4)
            elif cls == "H":
                result_probs["home"] = round(float(prob), 4)

        # Predict total goals
        predicted_goals = float(self.goals_model.predict(X)[0])
        predicted_goals = max(0, predicted_goals)

        return {
            "match_result": result_probs,
            "predicted_total_goals": round(predicted_goals, 2),
        }


class EnsemblePredictor:
    """Combines Poisson and Gradient Boosting predictions."""

    def __init__(
        self,
        poisson_weight: float = config.ENSEMBLE_WEIGHTS["poisson"],
        ml_weight: float = config.ENSEMBLE_WEIGHTS["ml"],
    ):
        self.poisson_model = PoissonModel()
        self.ml_model = GradientBoostingModel()
        self.poisson_weight = poisson_weight
        self.ml_weight = ml_weight
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """Fit both models on the same dataset."""
        logger.info("Fitting ensemble models...")
        self.poisson_model.fit(df)

        try:
            self.ml_model.fit(df)
        except Exception as e:
            logger.warning(f"ML model fitting failed: {e}. Using Poisson only.")
            self.poisson_weight = 1.0
            self.ml_weight = 0.0

        self._fitted = True
        logger.info("Ensemble fitting complete")

    def predict_match(self, home: str, away: str) -> dict:
        """Generate ensemble prediction for a match."""
        if not self._fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        # Poisson prediction
        poisson_pred = self.poisson_model.predict_match(home, away)

        # ML prediction (with fallback)
        ml_pred = None
        if self.ml_weight > 0:
            try:
                ml_pred = self.ml_model.predict_match(home, away)
            except Exception as e:
                logger.warning(f"ML prediction failed for {home} vs {away}: {e}")

        # Combine predictions
        if ml_pred and self.ml_weight > 0:
            combined_result = {}
            for key in ["home", "draw", "away"]:
                p_prob = poisson_pred["match_result"].get(key, 0.33)
                m_prob = ml_pred["match_result"].get(key, 0.33)
                combined_result[key] = round(
                    self.poisson_weight * p_prob + self.ml_weight * m_prob, 4
                )
        else:
            combined_result = poisson_pred["match_result"]

        # Normalize
        total = sum(combined_result.values())
        if total > 0:
            combined_result = {k: round(v / total, 4) for k, v in combined_result.items()}

        return {
            "match_result": combined_result,
            "score_matrix": poisson_pred["score_matrix"],
            "lambda_home": poisson_pred["lambda_home"],
            "lambda_away": poisson_pred["lambda_away"],
            "predicted_total_goals": ml_pred["predicted_total_goals"] if ml_pred else None,
            "poisson_result": poisson_pred["match_result"],
            "ml_result": ml_pred["match_result"] if ml_pred else None,
        }

    def evaluate(self, df: pd.DataFrame, n_splits: int = 3) -> dict:
        """Evaluate model using time-series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        accuracies = []
        log_losses = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            # Fit on training data
            temp_ensemble = EnsemblePredictor(self.poisson_weight, self.ml_weight)
            try:
                temp_ensemble.fit(train_df)
            except Exception as e:
                logger.warning(f"Fold {fold} fitting failed: {e}")
                continue

            # Evaluate on test data
            correct = 0
            total = 0
            for _, row in test_df.iterrows():
                try:
                    pred = temp_ensemble.predict_match(row["Home"], row["Away"])
                    predicted = max(pred["match_result"], key=pred["match_result"].get)
                    result_map = {"home": "H", "draw": "D", "away": "A"}
                    if result_map.get(predicted) == row["Res"]:
                        correct += 1
                    total += 1
                except Exception:
                    continue

            if total > 0:
                accuracy = correct / total
                accuracies.append(accuracy)
                logger.info(f"Fold {fold}: accuracy={accuracy:.3f} ({correct}/{total})")

        return {
            "mean_accuracy": round(np.mean(accuracies), 4) if accuracies else 0,
            "std_accuracy": round(np.std(accuracies), 4) if accuracies else 0,
            "folds": len(accuracies),
        }
