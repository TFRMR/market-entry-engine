"""Analyze forward outcomes after momentum candles."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from market_engine.data import load_mt5_csv
from market_engine.features import build_features
from market_engine.outcomes import add_forward_returns


HORIZONS = (1, 3, 5, 10)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_momentum_outcomes.py <mt5_csv>")
        return 1

    csv_path = Path(sys.argv[1])

    frame = load_mt5_csv(csv_path)
    frame = build_features(frame)
    frame = add_forward_returns(frame)

    momentum = frame[
        frame["is_momentum_candle"]
        & frame["is_bullish"].ne(frame["is_bearish"])
    ].copy()

    print("=== Momentum Outcome Analysis ===")
    print(f"Total candles:       {len(frame)}")
    print(f"Momentum candles:    {len(momentum)}")
    print(
        f"Bullish momentum:    "
        f"{(momentum['is_bullish']).sum()}"
    )
    print(
        f"Bearish momentum:    "
        f"{(momentum['is_bearish']).sum()}"
    )

    print("\nForward return statistics:")

    for horizon in HORIZONS:
        column = f"forward_return_{horizon}"
        values = momentum[column].dropna()

        print(f"\n+{horizon} candle(s)")
        print(f"  Samples:   {len(values)}")
        print(f"  Mean:      {values.mean():.6%}")
        print(f"  Median:    {values.median():.6%}")
        print(f"  Std:       {values.std():.6%}")
        print(f"  Min:       {values.min():.6%}")
        print(f"  Max:       {values.max():.6%}")
        print(f"  Positive:  {(values > 0).mean():.2%}")
        print(f"  Negative:  {(values < 0).mean():.2%}")
        print(f"  Zero:      {(values == 0).mean():.2%}")

    print("\nDirectional outcome by momentum direction:")

    for direction, mask in (
        ("Bullish", momentum["is_bullish"]),
        ("Bearish", momentum["is_bearish"]),
    ):
        subset = momentum.loc[mask]

        print(f"\n{direction} momentum")
        print(f"  Samples: {len(subset)}")

        for horizon in HORIZONS:
            column = f"forward_return_{horizon}"
            values = subset[column].dropna()

            print(
                f"  +{horizon}: "
                f"mean={values.mean():.6%}, "
                f"median={values.median():.6%}, "
                f"positive={(values > 0).mean():.2%}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
