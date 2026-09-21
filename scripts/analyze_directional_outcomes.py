"""Analyze outcomes relative to momentum-candle direction."""

from __future__ import annotations

import sys
from pathlib import Path

from market_engine.data import load_mt5_csv
from market_engine.features import build_features
from market_engine.outcomes import add_forward_returns

HORIZONS = (1, 3, 5, 10)


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: python scripts/analyze_directional_outcomes.py <mt5_csv>"
        )
        return 1

    csv_path = Path(sys.argv[1])

    frame = load_mt5_csv(csv_path)
    frame = build_features(frame)
    frame = add_forward_returns(frame)

    momentum = frame[
        frame["is_momentum_candle"]
        & frame["is_bullish"].ne(frame["is_bearish"])
    ].copy()

    momentum["momentum_direction"] = 0
    momentum.loc[momentum["is_bullish"], "momentum_direction"] = 1
    momentum.loc[momentum["is_bearish"], "momentum_direction"] = -1

    print("=== Directional Momentum Outcome Analysis ===")
    print(f"Momentum candles: {len(momentum)}")

    for horizon in HORIZONS:
        source = f"forward_return_{horizon}"

        directional = (
            momentum[source] * momentum["momentum_direction"]
        ).dropna()

        print(f"\n+{horizon} candle(s)")
        print(f"  Samples:       {len(directional)}")
        print(f"  Mean:          {directional.mean():.6%}")
        print(f"  Median:        {directional.median():.6%}")
        print(f"  Std:           {directional.std():.6%}")
        print(f"  Min:           {directional.min():.6%}")
        print(f"  Max:           {directional.max():.6%}")
        print(f"  Follow-through: {(directional > 0).mean():.2%}")
        print(f"  Reversal:       {(directional < 0).mean():.2%}")

    print("\nBy momentum direction:")

    for name, direction in (
        ("Bullish", 1),
        ("Bearish", -1),
    ):
        subset = momentum[momentum["momentum_direction"] == direction]

        print(f"\n{name}")
        print(f"  Samples: {len(subset)}")

        for horizon in HORIZONS:
            source = f"forward_return_{horizon}"

            directional = (
                subset[source] * direction
            ).dropna()

            print(
                f"  +{horizon}: "
                f"mean={directional.mean():.6%}, "
                f"median={directional.median():.6%}, "
                f"follow-through={(directional > 0).mean():.2%}"
            )

    print("\nSTATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
