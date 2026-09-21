"""Audit the feature pipeline against a real MT5 dataset."""

from __future__ import annotations

import sys
from pathlib import Path

from market_engine.data import load_mt5_csv
from market_engine.features import build_features

EXPECTED_FEATURES = {
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
    "volume_ratio_20",
    "volume_change_1",
}


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/audit_features.py <mt5_csv>")
        return 1

    csv_path = Path(sys.argv[1])

    frame = load_mt5_csv(csv_path)
    featured = build_features(frame)

    actual_features = set(featured.columns) - set(frame.columns)
    missing_features = EXPECTED_FEATURES - actual_features
    unexpected_features = actual_features - EXPECTED_FEATURES

    print("=== Feature Pipeline Audit ===")
    print(f"Rows:              {len(featured)}")
    print(f"Columns:            {len(featured.columns)}")
    print(f"Start:              {featured['timestamp'].iloc[0]}")
    print(f"End:                {featured['timestamp'].iloc[-1]}")
    print(f"Duplicate time:    {featured['timestamp'].duplicated().sum()}")
    print(f"Monotonic time:    {featured['timestamp'].is_monotonic_increasing}")
    print(f"Expected features: {len(EXPECTED_FEATURES)}")
    print(f"Actual features:   {len(actual_features)}")

    if missing_features:
        print("\nMissing features:")
        for name in sorted(missing_features):
            print(f"  - {name}")

    if unexpected_features:
        print("\nUnexpected features:")
        for name in sorted(unexpected_features):
            print(f"  - {name}")

    print("\nNaN counts:")
    nan_counts = featured[sorted(EXPECTED_FEATURES)].isna().sum()

    for name, count in nan_counts.items():
        if count:
            print(f"  {name:24s} {count}")

    print("\nMomentum candidates:")
    print(
        f"  Total:    {featured['is_momentum_candle'].sum()}"
    )
    print(
        f"  Bullish:  "
        f"{(featured['is_momentum_candle'] & featured['is_bullish']).sum()}"
    )
    print(
        f"  Bearish:  "
        f"{(featured['is_momentum_candle'] & featured['is_bearish']).sum()}"
    )

    if len(featured) != len(frame):
        raise RuntimeError("Feature pipeline changed the number of rows.")

    if missing_features:
        raise RuntimeError("Feature pipeline is missing expected features.")

    if featured["timestamp"].duplicated().any():
        raise RuntimeError("Feature pipeline contains duplicate timestamps.")

    if not featured["timestamp"].is_monotonic_increasing:
        raise RuntimeError("Feature pipeline broke timestamp ordering.")

    print("\nSTATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
