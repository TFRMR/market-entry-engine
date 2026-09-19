"""Walk-forward comparison of LightGBM and XGBoost on core features."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


INPUT = Path("data/processed/momentum_model_1r_10_binary.csv")

FEATURES = [
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


def make_lgbm():
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        max_depth=-1,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        n_jobs=2,
        verbosity=-1,
    )


def make_xgb():
    return xgb.XGBClassifier(
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


df = pd.read_csv(INPUT, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

required = {"timestamp", "target", *FEATURES}
missing = sorted(required - set(df.columns))

if missing:
    raise SystemExit(f"Missing columns: {missing}")

if not df["timestamp"].is_monotonic_increasing:
    raise SystemExit("Timestamp order is not chronological.")

if df["target"].isna().any():
    raise SystemExit("Missing target values.")

n = len(df)
test_size = n // 5

results = []

print("=== LightGBM vs XGBoost: CORE FEATURES ===")
print(f"Rows: {n}")
print(f"Features: {len(FEATURES)}")
print(f"Test size per fold: {test_size}")
print()

for fold in range(4):
    train_end = test_size * (fold + 1)
    test_start = train_end
    test_end = min(test_start + test_size, n)

    train = df.iloc[:train_end]
    test = df.iloc[test_start:test_end]

    X_train = train[FEATURES]
    y_train = train["target"]
    X_test = test[FEATURES]
    y_test = test["target"]

    baseline_probability = np.full(
        len(test),
        y_train.mean(),
    )

    baseline_logloss = log_loss(
        y_test,
        baseline_probability,
    )

    for model_name, make_model in [
        ("LightGBM", make_lgbm),
        ("XGBoost", make_xgb),
    ]:
        model = make_model()

        model.fit(X_train, y_train)

        probability = model.predict_proba(X_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)

        auc = roc_auc_score(y_test, probability)
        logloss = log_loss(y_test, probability)
        accuracy = accuracy_score(y_test, prediction)

        results.append(
            {
                "model": model_name,
                "fold": fold + 1,
                "auc": auc,
                "logloss": logloss,
                "baseline_logloss": baseline_logloss,
                "accuracy": accuracy,
            }
        )

        print(
            f"Fold {fold + 1} {model_name}: "
            f"AUC={auc:.4f} "
            f"logloss={logloss:.4f} "
            f"baseline={baseline_logloss:.4f} "
            f"accuracy={accuracy:.4f}"
        )

    print()

results = pd.DataFrame(results)

summary = (
    results
    .groupby("model")
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
print("=== Per-Fold AUC ===")
print(
    results
    .pivot(index="fold", columns="model", values="auc")
    .to_string()
)

print()
print("=== Per-Fold Logloss ===")
print(
    results
    .pivot(index="fold", columns="model", values="logloss")
    .to_string()
)

print()
print("Checks passed.")
