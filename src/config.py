"""Project settings and dataset attribution."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
LEAGUE = 'NFL'
DATA_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
RAW_CSV = ROOT / 'data' / 'nfl_games.csv'
RESULTS = ROOT / 'results'
RANDOM_STATE = 42
