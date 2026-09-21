"""Audit OOS LightGBM probabilities by selectivity threshold."""

from pathlib import Path

import lightgbm as lgb
import pandas as pd

INPUT_PATH = Path(
    "data/processed/momentum_model_1r_10_binary.csv"
)

OUTPUT_PATH = Path(
    "reports/lgbm_selectivity_oos.csv"
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
    train_end: int,
    test_end: int,
    fold_number: int,
) -> pd.DataFrame:
    train = frame.iloc[:train_end]
    test = frame.iloc[train_end:test_end].copy()

    model = make_model()
    model.fit(
        train[FEATURE_COLUMNS],
        train["target"],
    )

    test["probability"] = model.predict_proba(
        test[FEATURE_COLUMNS]
    )[:, 1]

    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    rows = []

    for threshold in thresholds:
        long_mask = test["probability"] >= threshold
        short_mask = test["probability"] <= (1.0 - threshold)

        selected = long_mask | short_mask

        selected_frame = test.loc[selected]

        if len(selected_frame) == 0:
            continue

        predicted_long = selected_frame["probability"] >= threshold

        directional_correct = (
            selected_frame["target"].eq(
                predicted_long.astype("int8")
            )
        )

        rows.append(
            {
                "fold": fold_number,
                "threshold": threshold,
                "test_n": len(test),
                "selected_n": len(selected_frame),
                "coverage": len(selected_frame) / len(test),
                "long_n": int(long_mask.sum()),
                "short_n": int(short_mask.sum()),
                "selected_tp_rate": (
                    directional_correct.mean()
                ),
                "selected_tp": int(directional_correct.sum()),
                "selected_sl": int(
                    len(selected_frame)
                    - directional_correct.sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    print("=== LightGBM OOS Selectivity Audit ===")

    frame = pd.read_csv(INPUT_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    n = len(frame)
    test_size = n // 5

    folds = [
        (test_size * 1, test_size * 2),
        (test_size * 2, test_size * 3),
        (test_size * 3, test_size * 4),
        (test_size * 4, n),
    ]

    results = []

    for fold_number, (train_end, test_end) in enumerate(
        folds,
        start=1,
    ):
        fold_result = evaluate_fold(
            frame,
            train_end,
            test_end,
            fold_number,
        )

        results.append(fold_result)

    result_frame = pd.concat(
        results,
        ignore_index=True,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=== Per-Fold Results ===")
    print(
        result_frame.to_string(index=False)
    )

    print()
    print("=== Aggregate ===")

    aggregate = (
        result_frame
        .groupby("threshold")
        .agg(
            folds=("fold", "count"),
            selected_n=("selected_n", "sum"),
            coverage=("coverage", "mean"),
            selected_tp=("selected_tp", "sum"),
            selected_sl=("selected_sl", "sum"),
        )
    )

    aggregate["tp_rate"] = (
        aggregate["selected_tp"]
        / (
            aggregate["selected_tp"]
            + aggregate["selected_sl"]
        )
    )

    print(aggregate.to_string())

    print()
    print("=== By Fold TP Rate ===")

    pivot = result_frame.pivot(
        index="threshold",
        columns="fold",
        values="selected_tp_rate",
    )

    print(pivot.to_string())

    print()
    print("=== Audit Checks ===")

    if result_frame["selected_n"].le(0).any():
        raise RuntimeError(
            "Found empty selectivity bucket."
        )

    if not result_frame["coverage"].between(
        0.0, 1.0
    ).all():
        raise RuntimeError(
            "Coverage outside [0, 1]."
        )

    if result_frame["selected_tp"].add(
        result_frame["selected_sl"]
    ).ne(result_frame["selected_n"]).any():
        raise RuntimeError(
            "Selected TP/SL counts do not sum to selected rows."
        )

    print("  All selectivity checks passed.")
    print()
    print("=== Selectivity audit complete ===")


if __name__ == "__main__":
    main()
