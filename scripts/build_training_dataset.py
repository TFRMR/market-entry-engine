"""Build a candidate + barrier-label dataset for model training."""

from pathlib import Path

from market_engine.data import load_mt5_csv
from market_engine.features import build_features
from market_engine.outcomes import add_barrier_outcomes

DATA_PATH = Path(
    "data/raw/XAUUSDc_M30_202409012200_202609182030.csv"
)

OUTPUT_PATH = Path("data/processed/momentum_training_1r_10.csv")


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
    print("=== Build Training Dataset ===")

    raw = load_mt5_csv(DATA_PATH)
    features = build_features(raw)

    outcomes = add_barrier_outcomes(
        features,
        horizons=(10,),
        targets_r=(1.0,),
    )

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "spread",
        "is_momentum_candle",
        "barrier_entry",
        "barrier_stop",
        "barrier_risk",
        "target_1r",
        "barrier_1r_10",
    ]

    missing = [
        column
        for column in required + FEATURE_COLUMNS
        if column not in outcomes.columns
    ]

    if missing:
        raise RuntimeError(
            "Required columns missing: " + ", ".join(sorted(set(missing)))
        )

    candidates = outcomes.loc[
        outcomes["is_momentum_candle"]
    ].copy()

    label_column = "barrier_1r_10"

    candidates = candidates.loc[
        candidates[label_column].notna()
    ].copy()

    candidates = candidates.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "spread",
        "is_momentum_candle",
        "barrier_entry",
        "barrier_stop",
        "barrier_risk",
        "target_1r",
        *FEATURE_COLUMNS,
        label_column,
    ]

    dataset = candidates[columns].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"Output:           {OUTPUT_PATH}")
    print(f"Rows:             {len(dataset)}")
    print()

    print("Label distribution:")
    print(
        dataset[label_column]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print("Direction:")
    print(
        dataset.assign(
            direction=dataset.apply(
                lambda row: (
                    "LONG"
                    if row["is_bullish"]
                    else "SHORT"
                ),
                axis=1,
            )
        )["direction"]
        .value_counts()
        .to_string()
    )

    print()
    print("Date range:")
    print(f"  Start: {dataset['timestamp'].min()}")
    print(f"  End:   {dataset['timestamp'].max()}")

    print()
    print("Dataset checks:")

    if dataset["timestamp"].duplicated().any():
        raise RuntimeError("Duplicate timestamps found.")

    if not dataset["timestamp"].is_monotonic_increasing:
        raise RuntimeError("Dataset is not chronologically ordered.")

    if dataset[label_column].isna().any():
        raise RuntimeError("Missing labels found.")

    if dataset[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Missing feature values found.")

    print("  No duplicate timestamps.")
    print("  Chronological order preserved.")
    print("  No missing labels.")
    print("  No missing feature values.")
    print("  Future outcome columns are not model features.")

    print()
    print("=== Dataset build successful ===")


if __name__ == "__main__":
    main()
