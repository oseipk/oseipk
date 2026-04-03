#!/usr/bin/env python3
"""
Football Match Prediction Agent - CLI Entry Point

Predicts football match outcomes across Latin American leagues:
  - Argentina Primera Division
  - Brazil Serie A
  - Mexico Liga MX

Usage:
    python main.py                          # Predict fixtures for all leagues
    python main.py --league ARG             # Predict fixtures for Argentina only
    python main.py --match ARG "Boca Juniors" "River Plate"  # Predict a specific match
    python main.py --evaluate               # Run model backtesting
    python main.py --teams ARG              # List teams in a league
"""

import argparse
import json
import sys

from football_prediction_agent.agent import (
    FootballPredictionAgent,
    format_prediction,
)
from football_prediction_agent.config import LEAGUES, OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Football Match Prediction Agent for Latin American Leagues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                      Predict fixtures for all leagues
  python main.py --league ARG                         Argentina fixtures only
  python main.py --league ARG BRA                     Argentina and Brazil
  python main.py --match ARG "Boca Juniors" "River Plate"    Specific match
  python main.py --evaluate                           Backtest model accuracy
  python main.py --teams ARG                          List teams in Argentina
  python main.py --fixtures 5                         Predict 5 fixtures per league
  python main.py --json                               Output as JSON
        """,
    )

    parser.add_argument(
        "--league", nargs="+", default=None,
        help=f"League codes to process (default: all). Options: {list(LEAGUES.keys())}",
    )
    parser.add_argument(
        "--match", nargs=3, metavar=("LEAGUE", "HOME", "AWAY"),
        help="Predict a specific match: LEAGUE HOME_TEAM AWAY_TEAM",
    )
    parser.add_argument(
        "--fixtures", type=int, default=10,
        help="Number of fixtures to predict per league (default: 10)",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run backtesting evaluation on models",
    )
    parser.add_argument(
        "--teams", type=str, default=None,
        help="List available teams for a league code",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output predictions as JSON",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save predictions to a specific file path",
    )

    args = parser.parse_args()

    # Determine which leagues to use
    leagues = args.league or list(LEAGUES.keys())
    if args.match:
        leagues = [args.match[0]]
    if args.teams:
        leagues = [args.teams]

    # Validate league codes
    for code in leagues:
        if code not in LEAGUES:
            print(f"Error: Unknown league '{code}'. Available: {list(LEAGUES.keys())}")
            sys.exit(1)

    # Initialize agent
    print("\n" + "=" * 70)
    print("  FOOTBALL MATCH PREDICTION AGENT")
    print("  Latin American Leagues Edition")
    print("=" * 70)
    print()

    agent = FootballPredictionAgent()
    agent.initialize(leagues)

    # List teams
    if args.teams:
        teams = agent.get_league_teams(args.teams)
        print(f"\nTeams in {LEAGUES[args.teams]['name']} (current season):")
        print("-" * 40)
        for i, team in enumerate(teams, 1):
            print(f"  {i:2d}. {team}")
        print(f"\nTotal: {len(teams)} teams")
        return

    # Evaluate models
    if args.evaluate:
        print("\n" + "=" * 70)
        print("  MODEL EVALUATION (Backtesting)")
        print("=" * 70)
        for code in leagues:
            result = agent.evaluate_model(code)
            print(f"\n  {LEAGUES[code]['name']}:")
            print(f"    Mean Accuracy: {result['mean_accuracy']:.1%}")
            print(f"    Std Deviation: {result['std_accuracy']:.1%}")
            print(f"    CV Folds:      {result['folds']}")
        return

    # Predict specific match
    if args.match:
        league_code, home, away = args.match
        prediction = agent.predict_match(league_code, home, away)

        if args.json:
            print(json.dumps(prediction, indent=2, default=str))
        else:
            print(format_prediction(prediction))

        if args.save:
            agent.save_predictions([prediction], args.save)
        return

    # Predict fixtures for each league
    all_predictions = []
    for code in leagues:
        print(f"\n{'='*70}")
        print(f"  Predictions for {LEAGUES[code]['name']}")
        print(f"{'='*70}")

        predictions = agent.predict_league_fixtures(code, n_fixtures=args.fixtures)
        all_predictions.extend(predictions)

        if args.json:
            for pred in predictions:
                print(json.dumps(pred, indent=2, default=str))
        else:
            for pred in predictions:
                print(format_prediction(pred))

    # Summary
    if not args.json:
        print("\n" + "=" * 70)
        print(f"  SUMMARY: {len(all_predictions)} matches predicted across {len(leagues)} leagues")
        print("=" * 70)

        for code in leagues:
            league_preds = [p for p in all_predictions if p["match"]["league_code"] == code]
            if league_preds:
                avg_conf = sum(
                    max(p["predictions"]["match_result"].values()) for p in league_preds
                ) / len(league_preds)
                high = sum(1 for p in league_preds if p["confidence"] == "high")
                med = sum(1 for p in league_preds if p["confidence"] == "medium")
                low = sum(1 for p in league_preds if p["confidence"] == "low")
                print(
                    f"  {LEAGUES[code]['name']}: {len(league_preds)} matches | "
                    f"High: {high} | Medium: {med} | Low: {low} | "
                    f"Avg top prob: {avg_conf:.1%}"
                )

    # Save predictions
    save_path = args.save or str(OUTPUT_DIR / "latest_predictions.json")
    agent.save_predictions(all_predictions, save_path)
    print(f"\n  Predictions saved to: {save_path}")


if __name__ == "__main__":
    main()
