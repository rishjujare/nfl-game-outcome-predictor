"""Leakage-aware features: every statistic is computed from previous games only."""
import numpy as np
import pandas as pd

FEATURES = ["home_prev_win_rate", "away_prev_win_rate", "home_prev_margin", "away_prev_margin", "home_prev_games", "away_prev_games", "home_rest_days", "away_rest_days"]


def make_features(games, window=5):
    if window < 1:
        raise ValueError("window must be positive")
    if games.empty:
        raise ValueError("games cannot be empty")
    base = games.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True).copy()
    base["game_id"] = np.arange(len(base))
    home = base[["game_id", "date", "season", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "points_for", "away_score": "points_against"})
    away = base[["game_id", "date", "season", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "points_for", "home_score": "points_against"})
    home["side"] = "home"
    away["side"] = "away"
    history = pd.concat([home, away], ignore_index=True)
    history["win"] = (history["points_for"] > history["points_against"]).astype(int)
    history["margin"] = history["points_for"] - history["points_against"]
    history = history.sort_values(["season", "team", "date", "game_id"]).copy()
    grouped = history.groupby(["season", "team"], sort=False)
    # Shift *before* computing rolling features. The game's actual outcome is NEVER in its predictors.
    history["prev_win_rate"] = grouped["win"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    history["prev_margin"] = grouped["margin"].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    history["prev_games"] = grouped.cumcount().clip(upper=window)
    history["rest_days"] = grouped["date"].diff().dt.days.clip(lower=0, upper=21)
    pieces = []
    for side in ("home", "away"):
        part = history[history["side"] == side][["game_id", "prev_win_rate", "prev_margin", "prev_games", "rest_days"]].copy()
        part = part.rename(columns={x: f"{side}_{x}" for x in ["prev_win_rate", "prev_margin", "prev_games", "rest_days"]})
        pieces.append(part)
    result = base.merge(pieces[0], on="game_id", validate="one_to_one").merge(pieces[1], on="game_id", validate="one_to_one")
    return result.sort_values("date").reset_index(drop=True)
