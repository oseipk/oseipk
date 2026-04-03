# Football Match Prediction Agent - Skill Guide

## Overview

This skill guide teaches an AI agent how to build, operate, and maintain an end-to-end football match prediction system. The agent predicts match outcomes across three Latin American leagues using historical data from football-data.co.uk, employing statistical models (Poisson regression) and machine learning (gradient boosting) in an ensemble approach.

## Target Leagues

| League | Country | Code | Source URL |
|--------|---------|------|-----------|
| Argentina Primera Division | Argentina | ARG | `https://www.football-data.co.uk/new/ARG.csv` |
| Brazil Serie A | Brazil | BRA | `https://www.football-data.co.uk/new/BRA.csv` |
| Mexico Liga MX | Mexico | MEX | `https://www.football-data.co.uk/new/MEX.csv` |

## Data Source: football-data.co.uk

### URL Pattern
Non-European ("new") leagues use a single CSV per league containing all seasons:
```
https://www.football-data.co.uk/new/{LEAGUE_CODE}.csv
```

### CSV Column Definitions

| Column | Description |
|--------|-------------|
| `Country` | Country name |
| `League` | League name |
| `Season` | Season identifier (e.g., `2023/2024`) |
| `Date` | Match date (DD/MM/YYYY) |
| `Time` | Kick-off time (HH:MM) |
| `Home` | Home team name |
| `Away` | Away team name |
| `HG` | Home goals scored |
| `AG` | Away goals scored |
| `Res` | Full-time result: `H` (Home win), `D` (Draw), `A` (Away win) |
| `PH` | Pinnacle home win odds |
| `PD` | Pinnacle draw odds |
| `PA` | Pinnacle away win odds |
| `MaxH` | Maximum home win odds (across bookmakers) |
| `MaxD` | Maximum draw odds |
| `MaxA` | Maximum away win odds |
| `AvgH` | Average home win odds |
| `AvgD` | Average draw odds |
| `AvgA` | Average away win odds |

## Architecture

```
football_prediction_agent/
    __init__.py          # Package init
    config.py            # League configs, model hyperparameters, constants
    data_loader.py       # Download, cache, clean, and validate CSV data
    feature_engineering.py  # Transform raw match data into ML features
    models.py            # Poisson regression + Gradient Boosting ensemble
    markets.py           # Betting market predictions (correct score, O/U, BTTS, 1X2)
    agent.py             # Main orchestrator that ties everything together
    utils.py             # Shared helpers (date parsing, team name normalization)
main.py                  # CLI entry point
requirements.txt         # Python dependencies
Skill.md                 # This file
```

## Step-by-Step Agent Workflow

### Step 1: Data Acquisition
1. Download CSV files for each league from football-data.co.uk
2. Cache locally to `data/` directory to avoid redundant downloads
3. Parse dates, handle missing values, validate schema
4. Filter to recent seasons (default: last 5 seasons) for model training

### Step 2: Feature Engineering
For each team, compute rolling/cumulative statistics over a configurable window (default: last 10 matches):

**Offensive Features:**
- `goals_scored_avg` - Average goals scored (home/away split)
- `goals_scored_home_avg` / `goals_scored_away_avg` - Venue-specific scoring
- `shots_on_target_avg` (if available)

**Defensive Features:**
- `goals_conceded_avg` - Average goals conceded
- `goals_conceded_home_avg` / `goals_conceded_away_avg`
- `clean_sheet_pct` - Clean sheet percentage

**Form Features:**
- `points_last_n` - Points accumulated in last N matches
- `win_streak` / `loss_streak` - Current streak length
- `form_string` - W/D/L pattern (e.g., "WWDLW")
- `unbeaten_run` - Matches since last defeat

**Head-to-Head Features:**
- `h2h_home_wins` - Historical H2H results
- `h2h_avg_goals` - Average goals in H2H meetings
- `h2h_last_result` - Most recent H2H outcome

**Strength Metrics:**
- `attack_strength` - Team's scoring rate / league average scoring rate
- `defense_strength` - Team's conceding rate / league average conceding rate
- `home_advantage` - Team-specific home advantage factor
- `elo_rating` - Dynamic Elo rating updated per match

**Odds-Derived Features:**
- `implied_prob_home` / `implied_prob_draw` / `implied_prob_away` - From Pinnacle odds
- `odds_value` - Discrepancy between model probability and market probability

### Step 3: Model Training

#### Model 1: Poisson Regression (Statistical)
- Predicts expected goals for home and away teams independently
- Uses attack strength, defense strength, and home advantage
- Generates a bivariate probability matrix for all scoreline combinations (0-0 through 7-7)
- Key advantage: Naturally produces correct score probabilities

```python
# Poisson probability for k goals given expected lambda
P(X=k) = (lambda^k * e^(-lambda)) / k!

# Expected home goals
lambda_home = avg_home_goals * home_attack_strength * away_defense_strength * home_advantage

# Expected away goals  
lambda_away = avg_away_goals * away_attack_strength * home_defense_strength
```

#### Model 2: Gradient Boosting Classifier (ML)
- Trained on engineered features to predict match outcome (H/D/A)
- Also trained as regressor to predict total goals
- Captures non-linear relationships and feature interactions
- Uses XGBoost or LightGBM for efficiency

#### Ensemble Strategy
- Combine Poisson probabilities with ML probabilities using weighted average
- Default weights: 40% Poisson, 60% ML (tunable)
- Correct score predictions primarily from Poisson model
- Match outcome predictions primarily from ML model

### Step 4: Market Predictions

The agent produces predictions for the following betting markets:

| Market | Method | Output |
|--------|--------|--------|
| **1X2 (Match Result)** | Ensemble probability | Home/Draw/Away probabilities |
| **Correct Score** | Poisson bivariate matrix | Top 5 most likely scorelines with probabilities |
| **Over/Under Goals** | Poisson CDF | O/U 0.5, 1.5, 2.5, 3.5, 4.5 probabilities |
| **Both Teams to Score (BTTS)** | Poisson P(HG>0) * P(AG>0) | Yes/No probability |
| **Double Chance** | Derived from 1X2 | 1X, X2, 12 probabilities |
| **Draw No Bet** | Derived from 1X2 | Home/Away (excluding draw) |
| **Half-Time/Full-Time** | Historical distribution | Most likely HT/FT combinations |
| **Asian Handicap** | Poisson goal difference distribution | Probabilities for common handicap lines |

### Step 5: Prediction Output

Output format for each predicted match:
```json
{
  "match": {
    "league": "Argentina Primera Division",
    "home_team": "Boca Juniors",
    "away_team": "River Plate",
    "date": "2026-04-10"
  },
  "predictions": {
    "match_result": {"home": 0.42, "draw": 0.28, "away": 0.30},
    "correct_score": [
      {"score": "1-0", "probability": 0.12},
      {"score": "1-1", "probability": 0.11},
      {"score": "2-1", "probability": 0.10},
      {"score": "0-0", "probability": 0.08},
      {"score": "2-0", "probability": 0.07}
    ],
    "over_under": {
      "over_0.5": 0.87, "over_1.5": 0.65, "over_2.5": 0.42,
      "over_3.5": 0.22, "over_4.5": 0.09
    },
    "btts": {"yes": 0.52, "no": 0.48},
    "double_chance": {"1X": 0.70, "X2": 0.58, "12": 0.72},
    "asian_handicap": {
      "home_-0.5": 0.42, "home_-1.5": 0.22,
      "away_+0.5": 0.58, "away_+1.5": 0.78
    }
  },
  "confidence": "medium",
  "model_metadata": {
    "poisson_lambda_home": 1.35,
    "poisson_lambda_away": 1.10,
    "ensemble_weights": {"poisson": 0.4, "ml": 0.6},
    "training_matches": 1520
  }
}
```

## Key Implementation Rules

1. **Always validate data** - Check for NaN goals, impossible results, duplicate matches
2. **Season-aware splits** - Never leak future data into training; use time-based CV
3. **Team name consistency** - Normalize team names (handle accent marks, abbreviations)
4. **Recency weighting** - More recent matches should have higher weight in features
5. **Minimum sample size** - Require at least 5 matches before generating predictions for a team
6. **Odds validation** - If odds columns are missing/NaN, skip odds-derived features gracefully
7. **Cache management** - Re-download data at most once per day; use local cache otherwise
8. **Graceful degradation** - If one model fails, fall back to the other instead of crashing
9. **Logging** - Log all predictions with timestamps for backtesting and accountability

## Model Evaluation (Backtesting)

Evaluate model quality using:
- **Accuracy** - % of correct 1X2 predictions
- **Log Loss** - Calibration of probability estimates
- **Brier Score** - Mean squared error of probability predictions
- **ROI Simulation** - Simulated return on investment vs closing odds
- **Correct Score Hit Rate** - % of times the top predicted scoreline was correct

## Configuration Defaults

```python
CONFIG = {
    "leagues": {
        "ARG": {"name": "Argentina Primera Division", "url": "https://www.football-data.co.uk/new/ARG.csv"},
        "BRA": {"name": "Brazil Serie A", "url": "https://www.football-data.co.uk/new/BRA.csv"},
        "MEX": {"name": "Mexico Liga MX", "url": "https://www.football-data.co.uk/new/MEX.csv"},
    },
    "rolling_window": 10,
    "min_matches": 5,
    "max_score": 7,
    "ensemble_weights": {"poisson": 0.4, "ml": 0.6},
    "seasons_to_use": 5,
    "elo_k_factor": 20,
    "elo_initial": 1500,
    "cache_dir": "data/",
    "cache_ttl_hours": 24,
}
```

## Error Handling Patterns

```python
# Pattern 1: Graceful data loading
try:
    df = download_league_data(league_code)
except requests.RequestException as e:
    logger.warning(f"Download failed for {league_code}, using cached data: {e}")
    df = load_cached_data(league_code)

# Pattern 2: Feature fallback
if "PH" not in df.columns or df["PH"].isna().all():
    logger.info("Pinnacle odds not available, skipping odds features")
    features = compute_features_without_odds(df)
else:
    features = compute_all_features(df)

# Pattern 3: Model fallback
try:
    ml_probs = ml_model.predict_proba(features)
except Exception as e:
    logger.warning(f"ML model failed: {e}, using Poisson-only predictions")
    ml_probs = None
    ensemble_weights = {"poisson": 1.0, "ml": 0.0}
```

## Dependencies

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.11
xgboost>=2.0
requests>=2.31
```

## Extension Points

An agent can extend this system by:
1. **Adding leagues** - Add new entries to `config.LEAGUES` dict with the country code
2. **Adding markets** - Implement new market calculators in `markets.py`
3. **Adding models** - Implement new model classes following the `BasePredictor` interface
4. **Adding features** - Add new feature computation functions in `feature_engineering.py`
5. **Live data integration** - Extend `data_loader.py` to fetch fixture lists from APIs
6. **Scheduling** - Wrap `agent.py` in a cron job or scheduler for daily predictions
