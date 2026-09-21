"""Build a candidate + barrier-label dataset for model training."""

from pathlib import Path

import pandas as pd

from market_engine.data import load_mt5_csv
from market_engine.features import build_features

DATA_PATH = Path(
    "data/raw/XAUUSDc_M30_202409012200_202609182030.csv"
)

OUTPUT_PATH = Path("data/processed/momentum_quality_5r.csv")


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


def main():
    print("=== Build Quality Dataset +5R ===")

    raw = load_mt5_csv(DATA_PATH)
    features = build_features(raw)

    required = [
        "timestamp", "open", "high", "low", "close", "spread",
        *FEATURE_COLUMNS,
    ]

    missing = [c for c in required if c not in features.columns]
    if missing:
        raise RuntimeError("Required columns missing: " + ", ".join(missing))

    rows = []
    for i in range(len(features) - 5):
        row = features.iloc[i]
        if not row["is_momentum_candle"]:
            continue

        future = features.iloc[i + 5]
        risk = float(row["candle_range"])
        entry = float(row["close"])

        if risk <= 0:
            continue

        favorable_r = (
            (float(future["close"]) - entry) / risk
            if row["is_bullish"]
            else (entry - float(future["close"])) / risk
        )

        item = row[[
            "timestamp", "open", "high", "low", "close", "spread",
            "is_momentum_candle", *FEATURE_COLUMNS
        ]].to_dict()

        item["favorable_5r"] = favorable_r
        item["quality_5r"] = int(favorable_r >= 0.5)
        rows.append(item)

    dataset = pd.DataFrame(rows)
    dataset = dataset.dropna(subset=FEATURE_COLUMNS).copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"Output: {OUTPUT_PATH}")
    print(f"Rows:   {len(dataset)}")
    print()
    print("Target distribution:")
    print(dataset["quality_5r"].value_counts().sort_index().to_string())
    print()
    print("Direction:")
    print(
        dataset.assign(
            direction=dataset["is_bullish"].map({True: "LONG", False: "SHORT"})
        )["direction"].value_counts().to_string()
    )
    print()
    print("Start: {}".format(dataset["timestamp"].min()))
    print("End: {}".format(dataset["timestamp"].max()))

    if dataset.empty:
        raise RuntimeError("Quality dataset is empty.")
    if dataset["timestamp"].duplicated().any():
        raise RuntimeError("Duplicate timestamps found.")
    if not dataset["timestamp"].is_monotonic_increasing:
        raise RuntimeError("Dataset is not chronological.")
    if dataset[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Missing feature values found.")
    if not dataset["quality_5r"].isin([0, 1]).all():
        raise RuntimeError("Invalid target values found.")

    print()
    print("Checks passed.")
    print("=== Quality dataset build successful ===")


if __name__ == "__main__":
    main()
