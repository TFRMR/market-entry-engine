"""Analyze momentum-candle outcomes by existing market context."""

from __future__ import annotations

import sys
from pathlib import Path

from market_engine.data import load_mt5_csv
from market_engine.features import build_features
from market_engine.outcomes import add_forward_returns


HORIZONS = (3, 5, 10)


def summarize(name, frame):
    print(f"\n{name}")
    print(f"  Samples: {len(frame)}")

    for horizon in HORIZONS:
        column = f"forward_return_{horizon}"
        values = frame[column].dropna()

        if values.empty:
            print(f"  +{horizon}: no data")
            continue

        print(
            f"  +{horizon}: "
            f"mean={values.mean():.6%}, "
            f"median={values.median():.6%}, "
            f"positive={(values > 0).mean():.2%}"
        )


def analyze_binary_context(frame, column, positive_name, negative_name):
    positive = frame[frame[column]]
    negative = frame[~frame[column]]

    summarize(positive_name, positive)
    summarize(negative_name, negative)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_momentum_context.py <mt5_csv>")
        return 1

    csv_path = Path(sys.argv[1])

    frame = load_mt5_csv(csv_path)
    frame = build_features(frame)
    frame = add_forward_returns(frame)

    momentum = frame[
        frame["is_momentum_candle"]
        & frame["is_bullish"].ne(frame["is_bearish"])
    ].copy()

    print("=== Momentum Context Analysis ===")
    print(f"Momentum candles: {len(momentum)}")

    print("\n--- EMA Alignment ---")
    summarize("EMA bullish alignment", momentum[momentum["ema_alignment"] == 1])
    summarize("EMA bearish alignment", momentum[momentum["ema_alignment"] == -1])
    summarize("EMA mixed/neutral", momentum[momentum["ema_alignment"] == 0])

    print("\n--- Breakout Context ---")
    summarize(
        "Breakout above 20",
        momentum[momentum["breakout_above_20"]],
    )
    summarize(
        "No breakout above 20",
        momentum[~momentum["breakout_above_20"]],
    )
    summarize(
        "Breakout below 20",
        momentum[momentum["breakout_below_20"]],
    )
    summarize(
        "No breakout below 20",
        momentum[~momentum["breakout_below_20"]],
    )

    print("\n--- Volatility Context ---")
    low_vol = momentum["range_to_atr"] < 1.0
    normal_vol = momentum["range_to_atr"].between(1.0, 2.0, inclusive="left")
    high_vol = momentum["range_to_atr"] >= 2.0

    summarize("Range/ATR < 1.0", momentum[low_vol])
    summarize("Range/ATR 1.0-2.0", momentum[normal_vol])
    summarize("Range/ATR >= 2.0", momentum[high_vol])

    print("\n--- Recent Movement ---")
    positive_recent = momentum["return_6"] > 0
    negative_recent = momentum["return_6"] < 0
    flat_recent = momentum["return_6"] == 0

    summarize("Return-6 positive", momentum[positive_recent])
    summarize("Return-6 negative", momentum[negative_recent])
    summarize("Return-6 zero", momentum[flat_recent])

    print("\n--- Volume Context ---")
    low_volume = momentum["volume_ratio_20"] < 1.0
    normal_volume = momentum["volume_ratio_20"].between(
        1.0, 1.5, inclusive="left"
    )
    high_volume = momentum["volume_ratio_20"] >= 1.5

    summarize("Volume ratio < 1.0", momentum[low_volume])
    summarize("Volume ratio 1.0-1.5", momentum[normal_volume])
    summarize("Volume ratio >= 1.5", momentum[high_volume])

    print("\nSTATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
