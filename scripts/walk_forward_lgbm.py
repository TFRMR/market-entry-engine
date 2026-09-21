"""Evaluate LightGBM with expanding-window chronological validation."""

from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

INPUT_PATH = Path(
    "data/processed/momentum_model_1r_10_binary.csv"
)

BASE_FEATURE_COLUMNS = [
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

MICRO_STRUCTURE_FEATURE_COLUMNS = [
    "high_vs_previous_high_1",
    "high_vs_previous_high_2",
    "high_vs_previous_high_3",
    "low_vs_previous_low_1",
    "low_vs_previous_low_2",
    "low_vs_previous_low_3",
    "range_vs_avg_3",
    "range_vs_avg_5",
]


def make_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
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


def evaluate_fold(
    frame: pd.DataFrame,
    feature_columns: list[str],
    train_end: int,
    test_end: int,
    fold_number: int,
) -> dict:
    train = frame.iloc[:train_end]
    test = frame.iloc[train_end:test_end]

    X_train = train[feature_columns]
    y_train = train["target"]

    X_test = test[feature_columns]
    y_test = test["target"]

    model = make_model()
    model.fit(X_train, y_train)

    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.50).astype(int)

    train_base_rate = y_train.mean()

    baseline_probability = [
        train_base_rate
    ] * len(y_test)

    return {
        "fold": fold_number,
        "train_n": len(train),
        "test_n": len(test),
        "train_start": train["timestamp"].min(),
        "train_end": train["timestamp"].max(),
        "test_start": test["timestamp"].min(),
        "test_end": test["timestamp"].max(),
        "train_base_rate": train_base_rate,
        "test_rate": y_test.mean(),
        "auc": roc_auc_score(y_test, probability),
        "logloss": log_loss(y_test, probability),
        "baseline_logloss": log_loss(
            y_test,
            baseline_probability,
        ),
        "accuracy": accuracy_score(
            y_test,
            prediction,
        ),
    }


def main() -> None:
    print("=== LightGBM Walk-Forward Validation ===")

    frame = pd.read_csv(INPUT_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    n = len(frame)

    # Four expanding-window folds.
    test_size = n // 5

    folds = [
        (test_size * 1, test_size * 2),
        (test_size * 2, test_size * 3),
        (test_size * 3, test_size * 4),
        (test_size * 4, n),
    ]

    feature_sets = {
        "baseline": BASE_FEATURE_COLUMNS,
        "baseline_plus_micro": (
            BASE_FEATURE_COLUMNS + MICRO_STRUCTURE_FEATURE_COLUMNS
        ),
    }

    all_results = []

    for model_name, feature_columns in feature_sets.items():
        print()
        print(f"=== {model_name} ===")

        results = []

        for fold_number, (train_end, test_end) in enumerate(
            folds,
            start=1,
        ):
            result = evaluate_fold(
                frame,
                feature_columns,
                train_end,
                test_end,
                fold_number,
            )

            results.append(result)

            print()
            print(f"Fold {fold_number}")
            print(
                f"  Train: {result['train_n']} samples "
                f"{result['train_start']} -> {result['train_end']}"
            )
            print(
                f"  Test:  {result['test_n']} samples "
                f"{result['test_start']} -> {result['test_end']}"
            )
            print(f"  Test AUC:          {result['auc']:.4f}")
            print(f"  Test log loss:     {result['logloss']:.4f}")
            print(
                f"  Baseline log loss: {result['baseline_logloss']:.4f}"
            )
            print(f"  Test accuracy:     {result['accuracy']:.4f}")

            all_results.append(
                {
                    "model": model_name,
                    **result,
                }
            )

        result_frame = pd.DataFrame(results)

        print()
        print("Aggregate")
        print(
            f"  Mean AUC:              "
            f"{result_frame['auc'].mean():.4f}"
        )
        print(
            f"  Median AUC:            "
            f"{result_frame['auc'].median():.4f}"
        )
        print(
            f"  Mean log loss:         "
            f"{result_frame['logloss'].mean():.4f}"
        )
        print(
            f"  Mean baseline logloss: "
            f"{result_frame['baseline_logloss'].mean():.4f}"
        )
        print(
            f"  Mean accuracy:         "
            f"{result_frame['accuracy'].mean():.4f}"
        )

    comparison = pd.DataFrame(all_results)

    print()
    print("=== Model Comparison ===")
    print(
        comparison[
            [
                "model",
                "fold",
                "auc",
                "logloss",
                "baseline_logloss",
                "accuracy",
            ]
        ].to_string(index=False)
    )

    print()
    print("=== Comparison Aggregate ===")
    print(
        comparison.groupby("model")[
            ["auc", "logloss", "baseline_logloss", "accuracy"]
        ].mean().to_string()
    )

    print()
    print("Interpretation:")
    print("  AUC near 0.50 means weak ranking ability.")
    print("  Compare model log loss against the fold baseline.")
    print("  Do not tune hyperparameters from these results yet.")
    print()
    print("=== Walk-forward validation complete ===")


if __name__ == "__main__":
    main()
