# NFL Game Outcome Predictor

**Python machine learning project | historical, pregame home-win classification**

Predict NFL home-team wins using only pregame historical team form. The pipeline ingests *real historical games* from a public data repository, constructs rolling team features from earlier games only, compares simple models, and evaluates on the **most recent 20% of game dates**.

> **Status:** Working source-code implementation. Run it locally with the upstream data to generate actual measurements. No performance results are claimed in this repository until the pipeline has been run and the results verified. Code drafted with AI assistance; data and evaluation should be independently checked before presenting them professionally.

## Why this project matters

- End-to-end Python data workflow: data ingestion, cleaning, feature engineering, training, evaluation, and plots.
- Reduces a common ML error: **target leakage**. Each team's win-rate and score-margin features use `shift(1)` so the game being predicted is not included.
- Evaluates a majority-class baseline, logistic regression, and random forest.
- Uses chronological separation of training and testing, rather than randomly mixing historical games.

## Setup (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.pipeline
```

The last command **requires internet access** on the first run to download a public CSV. Cached CSV is saved under `data/`; use `python -m src.pipeline --refresh` to download again.

Expected outputs after a successful real-data run:
- `results/metrics.json`: actual test metrics and train/test sizes
- `results/model_accuracy.png`: baseline vs model comparison
- `results/confusion_matrix.png`: logistic-regression confusion matrix

Those results are ignored by Git until you've reviewed them; if you want to feature genuine metrics/plots in the repo, intentionally add those files by editing `.gitignore`, then commit the verified outputs. Never type made-up metrics into this README.

## Data source and attribution

- NFL raw dataset: [https://github.com/nflverse/nfldata/blob/master/data/games.csv](https://github.com/nflverse/nfldata/blob/master/data/games.csv)
- CSV used by code: `https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv`
- Original dataset is attributed to its publisher; this project is an independent educational analysis, not affiliated with the league or publisher.
- Data publisher may change columns, move files, or impose additional license terms. Check the upstream repository and its license before redistributing datasets. We do not commit the raw dataset.

## Model features

Features are calculated separately for each team *and season*:

| Feature | Meaning |
| --- | --- |
| Previous win rate | Winning fraction over up to 5 earlier games |
| Previous point margin | Average points for minus against over up to 5 earlier games |
| Previous games | Number of games played this season, capped at 5 |
| Rest days | Days since prior game, capped at 21 |

Each statistic is included for **both home and away teams**. Values at the beginning of a season can be missing; the classifier imputes those from training-set values. Same-day games stay together in the holdout split.

## Limitations

- A simple historical baseline, **not a real-time prediction service** or a betting model.
- Does not incorporate injuries, starting lineups, trades, quarterback availability, travel, betting odds, or detailed possession/play-by-play information.
- A rolling pregame average is not a causal explanation of wins.
- Sports schedules/data quality can change; verify dates, team names, and counts on every run.
- This project's initial design/code was AI-assisted and should be reviewed, tested, and improved by its owner.

## Project structure

```text
nfl-game-outcome-predictor/
├── src/
│   ├── config.py       # source URLs / constants
│   ├── data.py         # download and clean completed games
│   ├── features.py     # pregame rolling features
│   └── pipeline.py     # train/evaluate/export plots
├── tests/
│   └── test_features.py
├── data/README.md
├── results/README.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Good next improvements

1. Validate season boundaries and team-name changes.
2. Add calibration curves and performance broken down by year/team.
3. Try a no-rest-days ablation and compare with the full model.
4. Add a simple command-line prediction for an upcoming scheduled match using strictly pregame data.
