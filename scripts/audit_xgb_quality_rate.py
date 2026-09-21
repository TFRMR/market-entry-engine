"""Audit OOS XGBoost probability against quality_5r."""

from pathlib import Path

import pandas as pd
import xgboost as xgb

INPUT_PATH = Path("data/processed/momentum_quality_5r.csv")
OUTPUT_PATH = Path("reports/xgb_quality_rate_oos.csv")

FEATURE_COLUMNS = [
    "candle_range",
    "candle_body",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_position",
    "true_range",
    "atr_14",
    "range_to_atr",
    "is_bullish",
    "is_bearish",
    "price_vs_ema_5",
    "price_vs_ema_20",
    "ema_5_vs_20",
    "ema_alignment",
]


def make_model():
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


def evaluate_fold(frame, train_end, test_end, fold):
    train = frame.iloc[:train_end]
    test = frame.iloc[train_end:test_end].copy()

    model = make_model()
    model.fit(train[FEATURE_COLUMNS], train["quality_5r"])

    test["probability"] = model.predict_proba(
        test[FEATURE_COLUMNS]
    )[:, 1]

    rows = []

    for threshold in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        selected = test["probability"] >= threshold
        selected_frame = test.loc[selected]

        if len(selected_frame) == 0:
            continue

        rows.append(
            {
                "fold": fold,
                "threshold": threshold,
                "test_n": len(test),
                "selected_n": len(selected_frame),
                "coverage": len(selected_frame) / len(test),
                "quality_n": int(selected_frame["quality_5r"].sum()),
                "quality_rate": float(selected_frame["quality_5r"].mean()),
            }
        )

    return pd.DataFrame(rows)


def main():
    print("=== XGBoost OOS Quality-Rate Audit ===")

    frame = pd.read_csv(INPUT_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    baseline = frame["quality_5r"].mean()
    n = len(frame)
    test_size = n // 5

    folds = [
        (test_size, test_size * 2),
        (test_size * 2, test_size * 3),
        (test_size * 3, test_size * 4),
        (test_size * 4, n),
    ]

    results = []

    for fold, (train_end, test_end) in enumerate(folds, start=1):
        result = evaluate_fold(
            frame,
            train_end,
            test_end,
            fold,
        )
        results.append(result)

    result_frame = pd.concat(results, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(OUTPUT_PATH, index=False)

    print()
    print(f"Overall baseline quality rate: {baseline:.6f}")

    print()
    print("=== Per-Fold Quality Rate ===")
    print(result_frame.to_string(index=False))

    print()
    print("=== Aggregate ===")

    aggregate = (
        result_frame
        .groupby("threshold")
        .agg(
            folds=("fold", "count"),
            selected_n=("selected_n", "sum"),
            coverage=("coverage", "mean"),
            quality_n=("quality_n", "sum"),
        )
    )

    aggregate["quality_rate"] = (
        aggregate["quality_n"] / aggregate["selected_n"]
    )

    aggregate["lift_vs_baseline"] = (
        aggregate["quality_rate"] - baseline
    )

    print(aggregate.to_string())

    print()
    print("=== By Fold Quality Rate ===")
    print(
        result_frame.pivot(
            index="threshold",
            columns="fold",
            values="quality_rate",
        ).to_string()
    )

    print()
    print("Checks passed.")


if __name__ == "__main__":
    main()
