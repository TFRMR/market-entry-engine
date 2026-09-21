"""Compare CORE, no-close-position, and candle-geometry-only models."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

INPUT = Path("data/processed/momentum_model_1r_10_binary.csv")


FEATURE_SETS = {
    "CORE": [
        "candle_range",
        "candle_body",
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "close_position",
        "true_range",
        "atr_14",
        "range_to_atr",
    ],
    "NO_CLOSE_POSITION": [
        "candle_range",
        "candle_body",
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "true_range",
        "atr_14",
        "range_to_atr",
    ],
    "GEOMETRY_ONLY": [
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "close_position",
    ],
}

TARGET = "target"


# ---------------------------------------------------------------------------
# Load and validate
# ---------------------------------------------------------------------------

df = pd.read_csv(INPUT, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

required = {
    "timestamp",
    "is_bullish",
    "is_bearish",
    TARGET,
    *{
        feature
        for features in FEATURE_SETS.values()
        for feature in features
    },
}

missing = sorted(required - set(df.columns))

if missing:
    raise SystemExit(f"Missing columns: {missing}")

if not df["timestamp"].is_monotonic_increasing:
    raise SystemExit("Timestamp order is not chronological.")

if df["timestamp"].duplicated().any():
    raise SystemExit("Duplicate timestamps found.")

if (df["is_bullish"] == df["is_bearish"]).any():
    raise SystemExit("Invalid bullish/bearish direction.")

df["direction"] = np.where(
    df["is_bullish"],
    "LONG",
    "SHORT",
)


# ---------------------------------------------------------------------------
# Four chronological folds
# ---------------------------------------------------------------------------

n = len(df)
test_size = n // 5

if n != 1950:
    raise SystemExit(
        f"Expected 1950 rows for the established walk-forward, got {n}"
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_model():
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=20,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        n_jobs=2,
        eval_metric="logloss",
    )


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

results = []

for feature_set_name, features in FEATURE_SETS.items():

    for fold in range(4):

        train_end = (fold + 1) * test_size
        test_start = train_end
        test_end = test_start + test_size

        train = df.iloc[:train_end]
        test = df.iloc[test_start:test_end]

        model = make_model()

        model.fit(
            train[features],
            train[TARGET],
        )

        probability = model.predict_proba(
            test[features]
        )[:, 1]

        prediction = (
            probability >= 0.50
        ).astype(int)

        for direction in ["ALL", "LONG", "SHORT"]:

            if direction == "ALL":
                mask = np.ones(
                    len(test),
                    dtype=bool,
                )
            else:
                mask = (
                    test["direction"].to_numpy()
                    == direction
                )

            y_true = test[TARGET].to_numpy()[mask]
            y_prob = probability[mask]
            y_pred = prediction[mask]

            auc = roc_auc_score(
                y_true,
                y_prob,
            )

            ll = log_loss(
                y_true,
                np.clip(
                    y_prob,
                    1e-6,
                    1 - 1e-6,
                ),
                labels=[0, 1],
            )

            baseline = log_loss(
                y_true,
                np.full(
                    len(y_true),
                    y_true.mean(),
                ),
                labels=[0, 1],
            )

            results.append(
                {
                    "feature_set": feature_set_name,
                    "fold": fold + 1,
                    "period": f"P{fold + 1}",
                    "direction": direction,
                    "n": len(y_true),
                    "auc": auc,
                    "logloss": ll,
                    "baseline_logloss": baseline,
                    "logloss_delta": ll - baseline,
                    "accuracy": accuracy_score(
                        y_true,
                        y_pred,
                    ),
                }
            )


results = pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

print("=== SIGNAL DECOMPOSITION: ALL ===")
print()

all_results = results[
    results["direction"] == "ALL"
]

summary = (
    all_results
    .groupby("feature_set")
    .agg(
        mean_auc=("auc", "mean"),
        mean_logloss=("logloss", "mean"),
        mean_baseline=("baseline_logloss", "mean"),
        mean_delta=("logloss_delta", "mean"),
        mean_accuracy=("accuracy", "mean"),
    )
)

print(
    summary.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ---------------------------------------------------------------------------
# AUC by period
# ---------------------------------------------------------------------------

print()
print("=== AUC BY PERIOD ===")

auc_matrix = results.pivot_table(
    index=["feature_set", "period"],
    columns="direction",
    values="auc",
)

print(
    auc_matrix.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ---------------------------------------------------------------------------
# Logloss delta by period
# ---------------------------------------------------------------------------

print()
print("=== LOGLOSS DELTA BY PERIOD ===")
print("(negative = better than period baseline)")

delta_matrix = results.pivot_table(
    index=["feature_set", "period"],
    columns="direction",
    values="logloss_delta",
)

print(
    delta_matrix.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ---------------------------------------------------------------------------
# Feature-set deltas vs CORE
# ---------------------------------------------------------------------------

print()
print("=== DELTA VS CORE: ALL ===")

core = summary.loc["CORE"]

for name in ["NO_CLOSE_POSITION", "GEOMETRY_ONLY"]:

    row = summary.loc[name]

    print(
        f"{name}: "
        f"AUC={row['mean_auc'] - core['mean_auc']:+.4f}, "
        f"logloss={row['mean_logloss'] - core['mean_logloss']:+.4f}, "
        f"accuracy={row['mean_accuracy'] - core['mean_accuracy']:+.4f}"
    )


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

print()
print("=== CHECKS ===")

expected_rows = len(FEATURE_SETS) * 4 * 3

if len(results) != expected_rows:
    raise SystemExit(
        f"Expected {expected_rows} rows, got {len(results)}"
    )

if results["n"].le(0).any():
    raise SystemExit("Empty evaluation group found.")

print("Checks passed.")
