"""Audit OOS LightGBM selectivity by LONG/SHORT direction."""

from pathlib import Path

import lightgbm as lgb
import pandas as pd


MODEL_INPUT_PATH = Path(
    "data/processed/momentum_model_1r_10_binary.csv"
)

TRAINING_INPUT_PATH = Path(
    "data/processed/momentum_training_1r_10.csv"
)

OUTPUT_PATH = Path(
    "reports/lgbm_direction_oos.csv"
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

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    rows = []

    for threshold in thresholds:
        for direction in ("LONG", "SHORT"):
            if direction == "LONG":
                selected = test["probability"] >= threshold
                predicted_target = 1
            else:
                selected = test["probability"] <= (1.0 - threshold)
                predicted_target = 0

            selected_frame = test.loc[
                selected & test["direction"].eq(direction)
            ]

            if len(selected_frame) == 0:
                continue

            correct = selected_frame["target"].eq(
                predicted_target
            )

            rows.append(
                {
                    "fold": fold_number,
                    "threshold": threshold,
                    "direction": direction,
                    "test_n": len(test),
                    "selected_n": len(selected_frame),
                    "coverage": len(selected_frame) / len(test),
                    "tp": int(correct.sum()),
                    "sl": int(
                        len(selected_frame) - correct.sum()
                    ),
                    "tp_rate": correct.mean(),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    print("=== LightGBM OOS Direction Audit ===")

    model_frame = pd.read_csv(MODEL_INPUT_PATH)
    training_frame = pd.read_csv(TRAINING_INPUT_PATH)

    model_frame["timestamp"] = pd.to_datetime(
        model_frame["timestamp"]
    )
    training_frame["timestamp"] = pd.to_datetime(
        training_frame["timestamp"]
    )

    required_training = {
        "timestamp",
        "is_bullish",
        "is_bearish",
    }

    missing_training = (
        required_training - set(training_frame.columns)
    )

    if missing_training:
        raise RuntimeError(
            "Training dataset is missing required columns: "
            f"{sorted(missing_training)}"
        )

    direction_map = (
        training_frame[
            ["timestamp", "is_bullish", "is_bearish"]
        ]
        .drop_duplicates("timestamp")
    )

    frame = model_frame.merge(
        direction_map,
        on="timestamp",
        how="left",
        validate="one_to_one",
        suffixes=("", "_training"),
    )

    if frame["is_bullish_training"].isna().any():
        raise RuntimeError(
            "Some model rows could not be matched to direction."
        )

    invalid_direction = (
        frame["is_bullish_training"]
        == frame["is_bearish_training"]
    )

    if invalid_direction.any():
        raise RuntimeError(
            "Found rows where bullish/bearish direction "
            "is ambiguous."
        )

    frame["direction"] = frame[
        "is_bullish_training"
    ].map(
        {
            True: "LONG",
            False: "SHORT",
        }
    )

    if frame["direction"].isna().any():
        raise RuntimeError(
            "Could not map all rows to LONG/SHORT."
        )

    if "is_bullish.1" in frame.columns:
        if not frame["is_bullish"].eq(
            frame["is_bullish.1"]
        ).all():
            raise RuntimeError(
                "Duplicate bullish columns disagree."
            )

    if "is_bearish.1" in frame.columns:
        if not frame["is_bearish"].eq(
            frame["is_bearish.1"]
        ).all():
            raise RuntimeError(
                "Duplicate bearish columns disagree."
            )

    if not frame["timestamp"].is_monotonic_increasing:
        raise RuntimeError(
            "Input timestamps are not chronological."
        )

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
        results.append(
            evaluate_fold(
                frame,
                train_end,
                test_end,
                fold_number,
            )
        )

    result_frame = pd.concat(
        results,
        ignore_index=True,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_frame.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=== Per-Fold Results ===")
    print(result_frame.to_string(index=False))

    print()
    print("=== Aggregate by Direction ===")

    aggregate = (
        result_frame
        .groupby(["threshold", "direction"])
        .agg(
            folds=("fold", "count"),
            selected_n=("selected_n", "sum"),
            coverage=("coverage", "mean"),
            tp=("tp", "sum"),
            sl=("sl", "sum"),
        )
    )

    aggregate["tp_rate"] = (
        aggregate["tp"]
        / (aggregate["tp"] + aggregate["sl"])
    )

    print(aggregate.to_string())

    print()
    print("=== TP Rate by Fold ===")

    pivot = result_frame.pivot_table(
        index=["threshold", "direction"],
        columns="fold",
        values="tp_rate",
    )

    print(pivot.to_string())

    print()
    print("=== Direction Selection Count ===")

    balance = (
        result_frame
        .groupby(["threshold", "direction"])["selected_n"]
        .sum()
        .unstack()
    )

    print(balance.to_string())

    print()
    print("=== Audit Checks ===")

    if not result_frame["coverage"].between(
        0.0, 1.0
    ).all():
        raise RuntimeError(
            "Coverage outside [0, 1]."
        )

    if result_frame["tp"].add(
        result_frame["sl"]
    ).ne(result_frame["selected_n"]).any():
        raise RuntimeError(
            "TP/SL counts do not sum to selected rows."
        )

    if (result_frame["selected_n"] <= 0).any():
        raise RuntimeError(
            "Found empty direction bucket."
        )

    print("  Direction mapping: passed.")
    print("  Bullish/bearish consistency: passed.")
    print("  Chronology: passed.")
    print("  TP/SL accounting: passed.")
    print("  Coverage bounds: passed.")
    print()
    print("=== Direction audit complete ===")


if __name__ == "__main__":
    main()
