"""Run pregame win-prediction baselines, temporal holdout evaluation, and plots."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from .config import LEAGUE, RESULTS
from .data import load_games, download_data
from .features import make_features, FEATURES


def chronological_split(df, frac=.8):
    """Hold out the newest ~20% by date; do not split same-day games."""
    distinct = np.array(sorted(df["date"].unique()))
    if len(distinct) < 10:
        raise ValueError("Need at least 10 distinct game dates for time split")
    cutoff = distinct[min(len(distinct) - 1, int(len(distinct) * frac))]
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    if len(train) < 50 or len(test) < 10:
        raise ValueError("Not enough observations for chronological train/test")
    return train, test, cutoff


def evaluate(train, test):
    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(SimpleImputer(strategy="median", add_indicator=True), LogisticRegression(max_iter=2000, random_state=42)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median", add_indicator=True), RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf=5, random_state=42, n_jobs=-1)),
    }
    x_train, x_test = train[FEATURES], test[FEATURES]
    y_train, y_test = train["home_win"], test["home_win"]
    output = {}
    probs = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
        prediction = (probability >= .5).astype(int)
        output[name] = {
            "accuracy": round(float(accuracy_score(y_test, prediction)), 4),
            "brier_score": round(float(brier_score_loss(y_test, probability)), 4),
            "log_loss": round(float(log_loss(y_test, np.clip(probability, 1e-7, 1-1e-7))), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probability)), 4) if y_test.nunique() == 2 else None,
        }
        probs[name] = probability
    return output, probs


def plot_results(results, test, probabilities, destination):
    names = list(results)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, [results[k]["accuracy"] for k in names])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy on chronological holdout")
    ax.set_title(f"{LEAGUE} game outcome prediction: comparison")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(destination / "model_accuracy.png", dpi=160)
    plt.close(fig)
    prediction = (probabilities["logistic_regression"] >= 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ConfusionMatrixDisplay.from_predictions(test["home_win"], prediction, ax=ax, colorbar=False)
    ax.set_title("Logistic regression: holdout confusion matrix")
    fig.tight_layout()
    fig.savefig(destination / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def run(path=None, force=False):
    if path is None:
        download_data(force=force)
    games = load_games(path)
    feature_table = make_features(games)
    train, test, cutoff = chronological_split(feature_table)
    metrics, probabilities = evaluate(train, test)
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {
        "league": LEAGUE, "games": len(games), "train_games": len(train), "test_games": len(test),
        "test_starts_on": str(pd.Timestamp(cutoff).date()),
        "feature_names": FEATURES, "metrics": metrics,
        "note": "Holdout evaluation on historical games, not predictions of future events. No betting use intended."
    }
    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_results(metrics, test, probabilities, RESULTS)
    print(json.dumps(report, indent=2))
    print(f"Charts saved to {RESULTS}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Historical sports win-prediction analysis")
    parser.add_argument("--csv", type=Path, default=None, help="Optional local CSV instead of public download")
    parser.add_argument("--refresh", action="store_true", help="Redownload upstream CSV")
    args = parser.parse_args()
    run(path=args.csv, force=args.refresh)
