"""Walk-forward LightGBM comparison across feature groups."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


INPUT = Path("data/processed/momentum_model_1r_10_binary.csv")


MOMENTUM_VOLATILITY = [
    "candle_range",
    "candle_body",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_position",
    "true_range",
    "atr_14",
    "range_to_atr",
]


EMA_5_20 = [
    "ema_5",
    "ema_20",
    "price_vs_ema_5",
    "price_vs_ema_20",
    "ema_5_vs_20",
    "ema_alignment",
]


SUPPORT_RESISTANCE = [
    "previous_high_20",
    "previous_low_20",
    "previous_high_50",
    "previous_low_50",
    "distance_to_high_20",
    "distance_to_low_20",
    "distance_to_high_50",
    "distance_to_low_50",
    "breakout_above_20",
    "breakout_below_20",
    "breakout_above_50",
    "breakout_below_50",
]


RECENT_MOVEMENT_VOLUME = [
    "return_3",
    "return_6",
    "return_12",
    "volume_ratio_20",
    "volume_change_1",
]


MICRO_STRUCTURE = [
    "high_vs_previous_high_1",
    "high_vs_previous_high_2",
    "high_vs_previous_high_3",
    "low_vs_previous_low_1",
    "low_vs_previous_low_2",
    "low_vs_previous_low_3",
    "range_vs_avg_3",
    "range_vs_avg_5",
]


FEATURE_SETS = {
    "A_MOMENTUM_VOL": (
        MOMENTUM_VOLATILITY
    ),
    "B_PLUS_EMA_5_20": (
        MOMENTUM_VOLATILITY
        + EMA_5_20
    ),
    "C_PLUS_CONTEXT": (
        MOMENTUM_VOLATILITY
        + EMA_5_20
        + SUPPORT_RESISTANCE
        + RECENT_MOVEMENT_VOLUME
    ),
    "D_PLUS_MICRO": (
        MOMENTUM_VOLATILITY
        + EMA_5_20
        + SUPPORT_RESISTANCE
        + RECENT_MOVEMENT_VOLUME
        + MICRO_STRUCTURE
    ),
}


def make_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        max_depth=-1,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        n_jobs=2,
        verbosity=-1,
    )


df = pd.read_csv(INPUT, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

required = {"timestamp", "target"}

for columns in FEATURE_SETS.values():
    required.update(columns)

missing = sorted(required - set(df.columns))

if missing:
    raise SystemExit(f"Missing columns: {missing}")

if df["target"].isna().any():
    raise SystemExit("Missing target values.")

if not df["timestamp"].is_monotonic_increasing:
    raise SystemExit("Timestamp order is not chronological.")

if df["target"].nunique() != 2:
    raise SystemExit("Target must contain exactly two classes.")


# ---------------------------------------------------------------------------
# Expanding chronological walk-forward
# ---------------------------------------------------------------------------

n = len(df)
test_size = n // 5

if test_size < 1:
    raise SystemExit("Dataset too small.")

folds = []

for fold in range(4):
    train_end = test_size * (fold + 1)
    test_start = train_end
    test_end = min(test_start + test_size, n)

    if test_start >= n:
        break

    folds.append(
        (
            fold + 1,
            train_end,
            test_start,
            test_end,
        )
    )


results = []


print("=== Feature Set Walk-Forward Audit ===")
print(f"Rows: {n}")
print(f"Test size per fold: {test_size}")
print()


for set_name, features in FEATURE_SETS.items():

    print(f"### {set_name}")
    print(f"Features: {len(features)}")

    for fold, train_end, test_start, test_end in folds:

        train = df.iloc[:train_end]
        test = df.iloc[test_start:test_end]

        X_train = train[features]
        y_train = train["target"]

        X_test = test[features]
        y_test = test["target"]

        model = make_model()
        model.fit(X_train, y_train)

        probability = model.predict_proba(X_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)

        auc = roc_auc_score(y_test, probability)
        ll = log_loss(y_test, probability)
        acc = accuracy_score(y_test, prediction)

        baseline_probability = np.full(
            len(y_test),
            y_train.mean(),
        )

        baseline_ll = log_loss(
            y_test,
            baseline_probability,
        )

        results.append(
            {
                "feature_set": set_name,
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "auc": auc,
                "logloss": ll,
                "baseline_logloss": baseline_ll,
                "accuracy": acc,
            }
        )

        print(
            f"Fold {fold}: "
            f"AUC={auc:.4f} "
            f"logloss={ll:.4f} "
            f"baseline={baseline_ll:.4f} "
            f"accuracy={acc:.4f}"
        )

    print()


results = pd.DataFrame(results)


summary = (
    results
    .groupby("feature_set")
    .agg(
        mean_auc=("auc", "mean"),
        median_auc=("auc", "median"),
        mean_logloss=("logloss", "mean"),
        mean_baseline_logloss=("baseline_logloss", "mean"),
        mean_accuracy=("accuracy", "mean"),
    )
    .reset_index()
)


print("=== Aggregate ===")
print(summary.to_string(index=False))


print()
print("=== Incremental Change ===")

ordered = list(FEATURE_SETS)

for i in range(1, len(ordered)):

    previous = summary[
        summary["feature_set"] == ordered[i - 1]
    ].iloc[0]

    current = summary[
        summary["feature_set"] == ordered[i]
    ].iloc[0]

    print(
        f"{ordered[i]} vs {ordered[i - 1]}: "
        f"AUC {current.mean_auc - previous.mean_auc:+.4f}, "
        f"logloss "
        f"{current.mean_logloss - previous.mean_logloss:+.4f}, "
        f"accuracy "
        f"{current.mean_accuracy - previous.mean_accuracy:+.4f}"
    )


print()
print("=== Feature Count ===")

for name, features in FEATURE_SETS.items():
    print(f"{name}: {len(features)}")


print()
print("Checks passed.")
