"""Train a chronological LightGBM baseline."""

from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
)

INPUT_PATH = Path(
    "data/processed/momentum_model_1r_10_binary.csv"
)

FEATURE_COLUMNS = [
    "candle_range",
    "candle_body",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_position",
    "is_bullish",
    "is_bearish",
    "true_range",
    "atr_14",
    "range_to_atr",
    "ema_5",
    "ema_20",
    "ema_50",
    "price_vs_ema_5",
    "price_vs_ema_20",
    "price_vs_ema_50",
    "ema_5_vs_20",
    "ema_20_vs_50",
    "ema_alignment",
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
    "return_3",
    "return_6",
    "return_12",
    "volume_ratio_20",
    "volume_change_1",
]


def main() -> None:
    print("=== LightGBM Chronological Baseline ===")

    frame = pd.read_csv(INPUT_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    split_index = int(len(frame) * 0.80)

    train = frame.iloc[:split_index].copy()
    test = frame.iloc[split_index:].copy()

    X_train = train[FEATURE_COLUMNS]
    y_train = train["target"]

    X_test = test[FEATURE_COLUMNS]
    y_test = test["target"]

    print(f"Total samples: {len(frame)}")
    print(f"Train samples: {len(train)}")
    print(f"Test samples:  {len(test)}")
    print()

    print("Train period:")
    print(f"  {train['timestamp'].min()}")
    print(f"  {train['timestamp'].max()}")
    print()

    print("Test period:")
    print(f"  {test['timestamp'].min()}")
    print(f"  {test['timestamp'].max()}")
    print()

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=20,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        n_jobs=2,
        verbosity=-1,
    )

    model.fit(X_train, y_train)

    train_probability = model.predict_proba(X_train)[:, 1]
    test_probability = model.predict_proba(X_test)[:, 1]

    train_prediction = (
        train_probability >= 0.50
    ).astype(int)

    test_prediction = (
        test_probability >= 0.50
    ).astype(int)

    train_auc = roc_auc_score(
        y_train,
        train_probability,
    )

    test_auc = roc_auc_score(
        y_test,
        test_probability,
    )

    train_logloss = log_loss(
        y_train,
        train_probability,
    )

    test_logloss = log_loss(
        y_test,
        test_probability,
    )

    train_accuracy = accuracy_score(
        y_train,
        train_prediction,
    )

    test_accuracy = accuracy_score(
        y_test,
        test_prediction,
    )

    baseline_probability = y_train.mean()

    baseline_test_probability = [
        baseline_probability
    ] * len(y_test)

    baseline_logloss = log_loss(
        y_test,
        baseline_test_probability,
    )

    print("Metrics:")
    print(
        f"  Train ROC-AUC:   {train_auc:.4f}"
    )
    print(
        f"  Test ROC-AUC:    {test_auc:.4f}"
    )
    print(
        f"  Train log loss:  {train_logloss:.4f}"
    )
    print(
        f"  Test log loss:   {test_logloss:.4f}"
    )
    print(
        f"  Train accuracy:  {train_accuracy:.4f}"
    )
    print(
        f"  Test accuracy:   {test_accuracy:.4f}"
    )
    print(
        f"  Baseline logloss:{baseline_logloss:.4f}"
    )
    print()

    print("Test probability distribution:")
    print(
        pd.Series(test_probability)
        .describe()
        .to_string()
    )

    print()
    print("Test target distribution:")
    print(
        y_test.value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("=== Baseline training complete ===")


if __name__ == "__main__":
    main()
