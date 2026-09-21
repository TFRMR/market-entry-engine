"""Walk-forward LightGBM ablation test against the core feature set."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

INPUT = Path("data/processed/momentum_model_1r_10_binary.csv")


CORE = [
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

GROUPS = {
    "CORE": [],
    "CORE_PLUS_EMA": [
        "ema_5",
        "ema_20",
        "price_vs_ema_5",
        "price_vs_ema_20",
        "ema_5_vs_20",
        "ema_alignment",
    ],
    "CORE_PLUS_SR": [
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
    ],
    "CORE_PLUS_MOVEMENT": [
        "return_3",
        "return_6",
        "return_12",
    ],
    "CORE_PLUS_VOLUME": [
        "volume_ratio_20",
        "volume_change_1",
    ],
    "CORE_PLUS_MICRO": [
        "high_vs_previous_high_1",
        "high_vs_previous_high_2",
        "high_vs_previous_high_3",
        "low_vs_previous_low_1",
        "low_vs_previous_low_2",
        "low_vs_previous_low_3",
        "range_vs_avg_3",
        "range_vs_avg_5",
    ],
}


def make_model():
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

required = {"timestamp", "target"}
for group in GROUPS.values():
    required.update(CORE + group)

missing = sorted(required - set(df.columns))
if missing:
    raise SystemExit(f"Missing columns: {missing}")

if not df["timestamp"].is_monotonic_increasing:
    raise SystemExit("Timestamp order is not chronological.")

if df["target"].isna().any():
    raise SystemExit("Missing target values.")

n = len(df)
test_size = n // 5

folds = []

for fold in range(4):
    train_end = test_size * (fold + 1)
    test_start = train_end
    test_end = min(test_start + test_size, n)

    if test_start >= n:
        break

    folds.append(
        (fold + 1, train_end, test_start, test_end)
    )


results = []

print("=== Feature Ablation Walk-Forward Audit ===")
print(f"Rows: {n}")
print(f"Test size per fold: {test_size}")
print()


for name, group in GROUPS.items():

    features = CORE + group

    print(f"### {name}")
    print(f"Features: {len(features)}")

    for fold, train_end, test_start, test_end in folds:

        train = df.iloc[:train_end]
        test = df.iloc[test_start:test_end]

        model = make_model()

        model.fit(
            train[features],
            train["target"],
        )

        probability = model.predict_proba(
            test[features]
        )[:, 1]

        prediction = (probability >= 0.5).astype(int)

        auc = roc_auc_score(
            test["target"],
            probability,
        )

        logloss = log_loss(
            test["target"],
            probability,
        )

        accuracy = accuracy_score(
            test["target"],
            prediction,
        )

        baseline_probability = np.full(
            len(test),
            train["target"].mean(),
        )

        baseline_logloss = log_loss(
            test["target"],
            baseline_probability,
        )

        results.append(
            {
                "group": name,
                "fold": fold,
                "auc": auc,
                "logloss": logloss,
                "baseline_logloss": baseline_logloss,
                "accuracy": accuracy,
            }
        )

        print(
            f"Fold {fold}: "
            f"AUC={auc:.4f} "
            f"logloss={logloss:.4f} "
            f"baseline={baseline_logloss:.4f} "
            f"accuracy={accuracy:.4f}"
        )

    print()


results = pd.DataFrame(results)

summary = (
    results
    .groupby("group")
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

core = summary[
    summary["group"] == "CORE"
].iloc[0]

print()
print("=== Delta vs CORE ===")

for _, row in summary.iterrows():

    if row["group"] == "CORE":
        continue

    print(
        f"{row['group']}: "
        f"AUC {row['mean_auc'] - core['mean_auc']:+.4f}, "
        f"logloss "
        f"{row['mean_logloss'] - core['mean_logloss']:+.4f}, "
        f"accuracy "
        f"{row['mean_accuracy'] - core['mean_accuracy']:+.4f}"
    )

print()
print("Checks passed.")
