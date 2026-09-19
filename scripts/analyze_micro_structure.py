"""Audit micro-structure around momentum candles."""

from pathlib import Path
import sys

import pandas as pd

from market_engine.data import load_mt5_csv
from market_engine.features import build_features


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/analyze_micro_structure.py <csv_path>"
        )

    path = Path(sys.argv[1])

    frame = load_mt5_csv(path)
    frame = build_features(frame)

    momentum = frame[frame["is_momentum_candle"]].copy()

    print("=== MOMENTUM ===")
    print(f"Total candles:       {len(frame):,}")
    print(f"Momentum candles:    {len(momentum):,}")
    print(f"Bullish momentum:    {momentum['is_bullish'].sum():,}")
    print(f"Bearish momentum:    {momentum['is_bearish'].sum():,}")
    print(f"Momentum rate:       {pct(len(momentum) / len(frame))}")

    print("\n=== PREVIOUS CANDLE DIRECTION ===")

    direction = frame["close"] - frame["open"]

    for n in (1, 2, 3):
        previous = direction.shift(n)

        bullish = previous > 0
        bearish = previous < 0

        bullish_count = bullish.loc[momentum.index].sum()
        bearish_count = bearish.loc[momentum.index].sum()

        print(
            f"C[-{n}]: bullish={bullish_count:,} "
            f"({pct(bullish_count / len(momentum))}) | "
            f"bearish={bearish_count:,} "
            f"({pct(bearish_count / len(momentum))})"
        )

    print("\n=== CONSECUTIVE DIRECTION ===")

    bullish = direction > 0
    bearish = direction < 0

    for n in (2, 3):
        prior_bullish = pd.concat(
            [bullish.shift(i) for i in range(1, n + 1)],
            axis=1,
        ).all(axis=1)

        prior_bearish = pd.concat(
            [bearish.shift(i) for i in range(1, n + 1)],
            axis=1,
        ).all(axis=1)

        bull_count = prior_bullish.loc[momentum.index].sum()
        bear_count = prior_bearish.loc[momentum.index].sum()

        print(
            f"Previous {n} bullish: {bull_count:,} "
            f"({pct(bull_count / len(momentum))})"
        )
        print(
            f"Previous {n} bearish: {bear_count:,} "
            f"({pct(bear_count / len(momentum))})"
        )

    print("\n=== BREAKOUT OF PREVIOUS HIGH/LOW ===")

    for n in (1, 2, 3):
        previous_high = frame["high"].shift(1).rolling(
            window=n,
            min_periods=n,
        ).max()

        previous_low = frame["low"].shift(1).rolling(
            window=n,
            min_periods=n,
        ).min()

        above = (frame["high"] > previous_high).loc[momentum.index]
        below = (frame["low"] < previous_low).loc[momentum.index]

        above_count = above.sum()
        below_count = below.sum()

        print(
            f"High > previous {n} high(s): {above_count:,} "
            f"({pct(above_count / len(momentum))})"
        )
        print(
            f"Low < previous {n} low(s):   {below_count:,} "
            f"({pct(below_count / len(momentum))})"
        )

    print("\n=== RANGE EXPANSION ===")

    candle_range = frame["high"] - frame["low"]

    for n in (3, 5):
        previous_average = candle_range.shift(1).rolling(
            window=n,
            min_periods=n,
        ).mean()

        ratio = candle_range / previous_average
        values = ratio.loc[momentum.index]

        print(
            f"Range / avg previous {n}: "
            f"median={values.median():.2f}x | "
            f"mean={values.mean():.2f}x"
        )

        for threshold in (1.5, 2.0, 3.0):
            count = (values >= threshold).sum()
            print(
                f"  >= {threshold:.1f}x: {count:,} "
                f"({pct(count / len(momentum))})"
            )

    print("\n=== BODY EXPANSION ===")

    body = (frame["close"] - frame["open"]).abs()

    for n in (3, 5):
        previous_average = body.shift(1).rolling(
            window=n,
            min_periods=n,
        ).mean()

        ratio = body / previous_average
        values = ratio.loc[momentum.index]

        print(
            f"Body / avg previous {n}: "
            f"median={values.median():.2f}x | "
            f"mean={values.mean():.2f}x"
        )

        for threshold in (1.5, 2.0, 3.0):
            count = (values >= threshold).sum()
            print(
                f"  >= {threshold:.1f}x: {count:,} "
                f"({pct(count / len(momentum))})"
            )


if __name__ == "__main__":
    main()
