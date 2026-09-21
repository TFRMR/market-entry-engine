"""Build the binary modeling dataset from barrier outcomes."""

from pathlib import Path

import pandas as pd

from market_engine.labels import MODEL_LABELS, SL_FIRST, TP_FIRST

INPUT_PATH = Path("data/processed/momentum_training_1r_10.csv")
OUTPUT_PATH = Path("data/processed/momentum_model_1r_10_binary.csv")

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
    "high_vs_previous_high_1",
    "high_vs_previous_high_2",
    "high_vs_previous_high_3",
    "low_vs_previous_low_1",
    "low_vs_previous_low_2",
    "low_vs_previous_low_3",
    "range_vs_avg_3",
    "range_vs_avg_5",
]


def main() -> None:
    print("=== Build Binary Model Dataset ===")

    source = pd.read_csv(INPUT_PATH)

    label_column = "barrier_1r_10"

    valid_labels = MODEL_LABELS

    dataset = source.loc[
        source[label_column].isin(valid_labels)
    ].copy()

    dataset["target"] = (
        dataset[label_column] == TP_FIRST
    ).astype("int8")

    columns = [
        "timestamp",
        "is_bullish",
        "is_bearish",
        *FEATURE_COLUMNS,
        "target",
    ]

    dataset = dataset[columns].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"Input rows:       {len(source)}")
    print(f"Model rows:       {len(dataset)}")
    print(f"Excluded rows:    {len(source) - len(dataset)}")
    print()

    print("Excluded labels:")
    excluded = source.loc[
        ~source[label_column].isin(valid_labels),
        label_column,
    ]
    print(excluded.value_counts().to_string())
    print()

    print("Target distribution:")
    counts = dataset["target"].value_counts().sort_index()

    for target, count in counts.items():
        percentage = count / len(dataset) * 100
        name = SL_FIRST if target == 0 else TP_FIRST
        print(f"  {target} = {name:8s}: {count:4d} ({percentage:6.2f}%)")

    print()
    print("Chronology:")
    print(f"  Start: {dataset['timestamp'].min()}")
    print(f"  End:   {dataset['timestamp'].max()}")

    if dataset["timestamp"].duplicated().any():
        raise RuntimeError("Duplicate timestamps found.")

    if not dataset["timestamp"].is_monotonic_increasing:
        raise RuntimeError("Timestamp order was not preserved.")

    if dataset["target"].isna().any():
        raise RuntimeError("Missing target found.")

    if dataset[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Missing feature found.")

    print()
    print("Dataset checks:")
    print("  No duplicate timestamps.")
    print("  Chronological order preserved.")
    print("  No missing target.")
    print("  No missing feature values.")
    print("  Only TP_FIRST and SL_FIRST are modeling labels.")
    print()
    print("=== Model dataset build successful ===")


if __name__ == "__main__":
    main()
