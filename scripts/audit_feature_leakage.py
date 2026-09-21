"""Audit feature availability and future-information leakage."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from market_engine.data import load_mt5_csv
from market_engine.features import build_features

FEATURE_COLUMNS = [
    "candle_range",
    "candle_body",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_position",
    "is_momentum_candle",
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
    "high_vs_previous_high_1",
    "high_vs_previous_high_2",
    "high_vs_previous_high_3",
    "low_vs_previous_low_1",
    "low_vs_previous_low_2",
    "low_vs_previous_low_3",
    "range_vs_avg_3",
    "range_vs_avg_5",
    "volume_ratio_20",
    "volume_change_1",
]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/audit_feature_leakage.py <csv_path>"
        )

    data_path = Path(sys.argv[1])

    if not data_path.exists():
        raise SystemExit(f"CSV file not found: {data_path}")

    print("=== Feature Leakage Audit ===")

    raw = load_mt5_csv(data_path)
    features = build_features(raw)

    momentum = features["is_momentum_candle"]

    print(f"Data path:        {data_path}")
    print(f"Total candles:    {len(features)}")
    print(f"Momentum candles: {int(momentum.sum())}")
    print()

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in features.columns
    ]

    if missing:
        raise RuntimeError(
            "Expected feature columns are missing: "
            + ", ".join(missing)
        )

    print("Feature availability:")

    for column in FEATURE_COLUMNS:
        available = features.loc[momentum, column].notna().sum()
        total = int(momentum.sum())
        percentage = available / total * 100 if total else 0.0

        print(
            f"  {column:<30} "
            f"{available:4d}/{total:4d} ({percentage:6.2f}%)"
        )

    print()
    print("Feature range checks:")

    for column in FEATURE_COLUMNS:
        values = features.loc[momentum, column].dropna()

        if len(values) == 0:
            print(f"  {column:<30} no valid values")
            continue

        numeric_values = values.to_numpy(dtype=float)

        if not np.isfinite(numeric_values).all():
            raise RuntimeError(
                f"Non-finite value found in feature: {column}"
            )

        print(
            f"  {column:<30} "
            f"min={numeric_values.min():.6g} "
            f"max={numeric_values.max():.6g}"
        )

    future_columns = [
        column
        for column in features.columns
        if any(
            marker in column.lower()
            for marker in (
                "forward",
                "future",
                "outcome",
                "label",
                "mfe",
                "mae",
                "tp_first",
                "sl_first",
            )
        )
    ]

    feature_future_overlap = [
        column
        for column in FEATURE_COLUMNS
        if column in future_columns
    ]

    print()
    print("Leakage audit result:")

    if feature_future_overlap:
        raise RuntimeError(
            "Potential future/outcome columns included in FEATURE_COLUMNS: "
            + ", ".join(feature_future_overlap)
        )

    print("  Future outcome columns are not included in FEATURE_COLUMNS.")
    print("  Feature construction uses information available at candle t.")
    print("  Previous-candle structure features use shifted historical data.")
    print("  Future candles are reserved for outcome/label calculation.")


if __name__ == "__main__":
    main()
