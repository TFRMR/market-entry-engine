"""Analyze 1R/10-candle outcomes by micro-structure context."""

from pathlib import Path
import sys

import pandas as pd

from market_engine.data import load_mt5_csv
from market_engine.features import build_features
from market_engine.outcomes import add_barrier_outcomes


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def show_result(name: str, mask: pd.Series, outcome: pd.Series) -> None:
    values = outcome.loc[mask]
    values = values.dropna()

    total = len(values)
    if total == 0:
        print(f"{name}: n=0")
        return

    tp = (values == "TP_FIRST").sum()
    sl = (values == "SL_FIRST").sum()
    both = (values == "BOTH_SAME_CANDLE").sum()
    unresolved = (values == "UNRESOLVED").sum()

    resolved = tp + sl

    print(
        f"{name}: n={total:,} | "
        f"TP={pct(tp / total)} | "
        f"SL={pct(sl / total)} | "
        f"BOTH={pct(both / total)} | "
        f"UNRES={pct(unresolved / total)}"
    )

    if resolved:
        print(
            f"  resolved TP rate: {pct(tp / resolved)} "
            f"(resolved n={resolved:,})"
        )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/analyze_micro_structure_outcomes.py <csv_path>"
        )

    path = Path(sys.argv[1])

    frame = load_mt5_csv(path)
    frame = build_features(frame)
    frame = add_barrier_outcomes(
        frame,
        horizons=(10,),
        targets_r=(1.0,),
    )

    momentum = frame["is_momentum_candle"]
    bullish = momentum & frame["is_bullish"]
    bearish = momentum & frame["is_bearish"]

    outcome = frame["barrier_1r_10"]

    print("=== 1R / 10-CANDLE OUTCOMES ===")
    print(f"Total candles:    {len(frame):,}")
    print(f"Momentum candles: {momentum.sum():,}")
    print(f"Bullish:          {bullish.sum():,}")
    print(f"Bearish:          {bearish.sum():,}")

    print("\n=== BASELINE ===")
    show_result("All momentum", momentum, outcome)
    show_result("Bullish momentum", bullish, outcome)
    show_result("Bearish momentum", bearish, outcome)

    print("\n=== PREVIOUS DIRECTION ===")

    direction = frame["close"] - frame["open"]
    prev_bullish = direction.shift(1) > 0
    prev_bearish = direction.shift(1) < 0

    for label, mask in [
        ("Previous 1 bullish", prev_bullish),
        ("Previous 1 bearish", prev_bearish),
        ("Previous 2 bullish", prev_bullish & (direction.shift(2) > 0)),
        ("Previous 2 bearish", prev_bearish & (direction.shift(2) < 0)),
        (
            "Previous 3 bullish",
            prev_bullish
            & (direction.shift(2) > 0)
            & (direction.shift(3) > 0),
        ),
        (
            "Previous 3 bearish",
            prev_bearish
            & (direction.shift(2) < 0)
            & (direction.shift(3) < 0),
        ),
    ]:
        show_result(label, momentum & mask, outcome)

    print("\n=== SHORT STRUCTURE BREAKOUT ===")

    for n in (1, 2, 3):
        previous_high = frame["high"].shift(1).rolling(
            window=n,
            min_periods=n,
        ).max()

        previous_low = frame["low"].shift(1).rolling(
            window=n,
            min_periods=n,
        ).min()

        high_break = frame["high"] > previous_high
        low_break = frame["low"] < previous_low

        show_result(
            f"High > previous {n} high(s)",
            momentum & high_break,
            outcome,
        )

        show_result(
            f"Low < previous {n} low(s)",
            momentum & low_break,
            outcome,
        )

    print("\n=== RANGE EXPANSION ===")

    candle_range = frame["high"] - frame["low"]

    for n in (3, 5):
        average_range = candle_range.shift(1).rolling(
            window=n,
            min_periods=n,
        ).mean()

        ratio = candle_range / average_range

        for threshold in (1.5, 2.0):
            show_result(
                f"Range >= {threshold:.1f}x avg {n}",
                momentum & ratio.ge(threshold),
                outcome,
            )

            show_result(
                f"Range < {threshold:.1f}x avg {n}",
                momentum & ratio.lt(threshold),
                outcome,
            )

    print("\n=== DIRECTIONAL STRUCTURE ===")

    for label, mask in [
        (
            "Bullish + previous 3 bullish",
            bullish
            & (direction.shift(1) > 0)
            & (direction.shift(2) > 0)
            & (direction.shift(3) > 0),
        ),
        (
            "Bearish + previous 3 bearish",
            bearish
            & (direction.shift(1) < 0)
            & (direction.shift(2) < 0)
            & (direction.shift(3) < 0),
        ),
        (
            "Bullish + high breaks previous 3",
            bullish
            & (frame["high"]
               > frame["high"].shift(1).rolling(3, min_periods=3).max()),
        ),
        (
            "Bearish + low breaks previous 3",
            bearish
            & (frame["low"]
               < frame["low"].shift(1).rolling(3, min_periods=3).min()),
        ),
    ]:
        show_result(label, momentum & mask, outcome)


if __name__ == "__main__":
    main()
