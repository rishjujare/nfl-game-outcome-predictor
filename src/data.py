"""Download and standardize historical completed games."""
import pandas as pd
import requests
from .config import DATA_URL, RAW_CSV, LEAGUE


def download_data(force=False):
    if RAW_CSV.exists() and not force:
        return RAW_CSV
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(DATA_URL, timeout=60, headers={"User-Agent": "sports-portfolio-educational/1.0"})
    response.raise_for_status()
    if len(response.content) < 200:
        raise ValueError("Data download appears incomplete")
    RAW_CSV.write_bytes(response.content)
    return RAW_CSV


def load_games(path=None):
    df = pd.read_csv(path or download_data(), low_memory=False)
    if LEAGUE == "NBA":
        required = {"date", "home", "away", "h_pts", "a_pts"}
        if not required.issubset(df.columns):
            raise ValueError(f"NBA source changed. Missing: {required - set(df.columns)}")
        if "type" in df.columns:
            # The publisher uses 'type' for game category; use all completed games rather than guessing its labels.
            pass
        data = df.rename(columns={"home": "home_team", "away": "away_team", "h_pts": "home_score", "a_pts": "away_score"})
        data["season"] = pd.to_datetime(data["date"], errors="coerce").dt.year
    else:
        required = {"gameday", "home_team", "away_team", "home_score", "away_score", "season", "game_type"}
        if not required.issubset(df.columns):
            raise ValueError(f"NFL source changed. Missing: {required - set(df.columns)}")
        data = df[df["game_type"].eq("REG")].copy()
        data = data.rename(columns={"gameday": "date"})
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True).dt.tz_localize(None)
    for col in ["home_score", "away_score"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score", "season"])
    data = data[data["home_score"] != data["away_score"]].copy()  # No ties in binary classification.
    data["home_team"] = data["home_team"].astype(str).str.strip()
    data["away_team"] = data["away_team"].astype(str).str.strip()
    data = data[(data["home_team"] != "") & (data["away_team"] != "")]
    # Multiple rows from the same game would be leakage/duplicate training cases.
    data = data.drop_duplicates(subset=["date", "home_team", "away_team"])
    data = data.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    data["home_win"] = (data["home_score"] > data["away_score"]).astype(int)
    if len(data) < 100:
        raise ValueError(f"Only {len(data)} completed games found. Need at least 100.")
    return data[["date", "season", "home_team", "away_team", "home_score", "away_score", "home_win"]]
