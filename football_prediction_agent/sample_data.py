"""Generate realistic sample data for testing when live download is unavailable."""

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from . import config

# Realistic team names for each league
LEAGUE_TEAMS = {
    "ARG": [
        "Boca Juniors", "River Plate", "Racing Club", "Independiente",
        "San Lorenzo", "Huracan", "Velez Sarsfield", "Argentinos Juniors",
        "Estudiantes", "Gimnasia LP", "Lanus", "Banfield",
        "Defensa y Justicia", "Talleres Cordoba", "Godoy Cruz", "Union",
        "Central Cordoba", "Platense", "Barracas Central", "Instituto",
        "Belgrano", "Newells Old Boys", "Rosario Central", "Tigre",
        "Sarmiento", "Atletico Tucuman", "Colon",  "Arsenal Sarandi",
    ],
    "BRA": [
        "Flamengo", "Palmeiras", "Atletico Mineiro", "Fluminense",
        "Botafogo", "Sao Paulo", "Corinthians", "Internacional",
        "Gremio", "Cruzeiro", "Atletico Paranaense", "Fortaleza",
        "Bahia", "Vasco da Gama", "Santos", "Bragantino",
        "Cuiaba", "Goias", "Coritiba", "America Mineiro",
    ],
    "MEX": [
        "Club America", "Guadalajara", "Cruz Azul", "Pumas UNAM",
        "Monterrey", "Tigres UANL", "Santos Laguna", "Leon",
        "Toluca", "Pachuca", "Atlas", "Queretaro",
        "Puebla", "Necaxa", "Mazatlan", "Tijuana",
        "Juarez", "San Luis",
    ],
}

LEAGUE_NAMES = {
    "ARG": "Liga Profesional",
    "BRA": "Serie A",
    "MEX": "Liga MX",
}

COUNTRY_NAMES = {
    "ARG": "Argentina",
    "BRA": "Brazil",
    "MEX": "Mexico",
}


def generate_sample_data(league_code: str, seasons: int = 5) -> pd.DataFrame:
    """Generate realistic sample match data for a league.

    Generates full round-robin seasons with realistic goal distributions
    and home advantage effect.
    """
    random.seed(42 + hash(league_code))
    teams = LEAGUE_TEAMS[league_code]
    rows = []

    base_date = datetime(2020, 1, 15)

    for season_idx in range(seasons):
        season_start = base_date + timedelta(days=season_idx * 365)
        season_label = f"{season_start.year}/{season_start.year + 1}"
        match_day = 0

        # Full round-robin (each team plays each other twice: home and away)
        matchups = []
        for i, home in enumerate(teams):
            for j, away in enumerate(teams):
                if i != j:
                    matchups.append((home, away))

        random.shuffle(matchups)

        for idx, (home, away) in enumerate(matchups):
            match_date = season_start + timedelta(days=(idx * 3) // len(teams) * 7 + random.randint(0, 2))

            # Realistic goal generation with home advantage
            home_lambda = 1.45 + random.gauss(0, 0.3)  # Home advantage
            away_lambda = 1.10 + random.gauss(0, 0.3)

            # Team strength variation
            home_strength = 0.8 + (hash(home) % 5) * 0.1
            away_strength = 0.8 + (hash(away) % 5) * 0.1

            hg = max(0, int(random.expovariate(1.0 / (home_lambda * home_strength / 1.0))))
            ag = max(0, int(random.expovariate(1.0 / (away_lambda * away_strength / 1.0))))

            # Cap at reasonable values
            hg = min(hg, 7)
            ag = min(ag, 6)

            if hg > ag:
                res = "H"
            elif hg == ag:
                res = "D"
            else:
                res = "A"

            # Generate realistic odds
            if res == "H":
                ph = round(1.5 + random.random() * 0.8, 2)
                pd_odds = round(3.2 + random.random() * 0.5, 2)
                pa = round(4.0 + random.random() * 3.0, 2)
            elif res == "D":
                ph = round(2.0 + random.random() * 1.0, 2)
                pd_odds = round(2.8 + random.random() * 0.6, 2)
                pa = round(3.0 + random.random() * 2.0, 2)
            else:
                ph = round(3.0 + random.random() * 3.0, 2)
                pd_odds = round(3.2 + random.random() * 0.5, 2)
                pa = round(1.5 + random.random() * 0.8, 2)

            rows.append({
                "Country": COUNTRY_NAMES[league_code],
                "League": LEAGUE_NAMES[league_code],
                "Season": season_label,
                "Date": match_date.strftime("%d/%m/%Y"),
                "Time": f"{random.choice([15, 17, 19, 21])}:00",
                "Home": home,
                "Away": away,
                "HG": hg,
                "AG": ag,
                "Res": res,
                "PH": ph,
                "PD": pd_odds,
                "PA": pa,
                "MaxH": round(ph * (1 + random.random() * 0.05), 2),
                "MaxD": round(pd_odds * (1 + random.random() * 0.05), 2),
                "MaxA": round(pa * (1 + random.random() * 0.05), 2),
                "AvgH": round(ph * (1 - random.random() * 0.03), 2),
                "AvgD": round(pd_odds * (1 - random.random() * 0.03), 2),
                "AvgA": round(pa * (1 - random.random() * 0.03), 2),
            })

    df = pd.DataFrame(rows)
    return df


def save_sample_data(league_code: str) -> Path:
    """Generate and save sample data to the cache directory."""
    df = generate_sample_data(league_code)
    path = config.CACHE_DIR / f"{league_code}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def ensure_data_available() -> None:
    """Ensure sample data exists for all leagues (used as fallback)."""
    for code in config.LEAGUES:
        csv_path = config.CACHE_DIR / f"{code}.csv"
        if not csv_path.exists():
            print(f"Generating sample data for {code}...")
            save_sample_data(code)
