"""Tests use fabricated mini games solely to verify feature computation."""
import unittest
import pandas as pd
from src.features import make_features, FEATURES


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.games = pd.DataFrame([
            {"date": "2024-01-01", "season": 2024, "home_team": "A", "away_team": "B", "home_score": 100, "away_score": 90, "home_win": 1},
            {"date": "2024-01-05", "season": 2024, "home_team": "B", "away_team": "A", "home_score": 80, "away_score": 95, "home_win": 0},
            {"date": "2024-01-10", "season": 2024, "home_team": "A", "away_team": "B", "home_score": 70, "away_score": 85, "home_win": 0},
        ])
        self.games["date"] = pd.to_datetime(self.games["date"])

    def test_first_game_has_no_prior_win_rate(self):
        x = make_features(self.games)
        self.assertTrue(pd.isna(x.loc[0, "home_prev_win_rate"]))
        self.assertEqual(int(x.loc[0, "home_prev_games"]), 0)

    def test_pregame_rolling_history(self):
        x = make_features(self.games)
        self.assertAlmostEqual(x.loc[1, "away_prev_win_rate"], 1.0)  # A won prior game
        self.assertAlmostEqual(x.loc[1, "home_prev_win_rate"], 0.0)  # B lost prior game
        self.assertAlmostEqual(x.loc[2, "home_prev_margin"], 12.5)
        self.assertEqual(len(FEATURES), 8)

    def test_ignores_current_result(self):
        modified = self.games.copy()
        modified.loc[2, "home_score"] = 1000
        left, right = make_features(self.games), make_features(modified)
        for feature in FEATURES:
            a, b = left.loc[2, feature], right.loc[2, feature]
            self.assertTrue((pd.isna(a) and pd.isna(b)) or a == b, feature)


if __name__ == "__main__":
    unittest.main()
