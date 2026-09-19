"""Analyze TP/SL barrier outcomes for momentum-candle setups."""

from pathlib import Path

import pandas as pd

from market_engine.data import load_mt5_csv
from market_engine.features import build_features
from market_engine.outcomes import add_barrier_outcomes


DATA_PATH = Path(
    "/home/tofarmer/Desktop/"
    "XAUUSDc_M30_202601012300_202609181730.csv"
)


HORIZONS = (1, 3, 5, 10)
TARGETS_R = (1.0, 1.5, 1.618, 2.0)


def print_distribution(
    frame: pd.DataFrame,
    column: str,
    label: str,
) -> None:
    subset = frame.loc[frame["is_momentum_candle"], column].dropna()

    counts = subset.value_counts()
    total = len(subset)

    print(f"\n{label}")
    print(f"  Samples: {total}")

    if total == 0:
        return

    for outcome in (
        "TP_FIRST",
        "SL_FIRST",
        "BOTH_SAME_CANDLE",
        "UNRESOLVED",
    ):
        count = int(counts.get(outcome, 0))
        percentage = count / total * 100

        print(
            f"  {outcome:18s}: "
            f"{count:4d} ({percentage:6.2f}%)"
        )


def main() -> None:
    print("=== Barrier Outcome Analysis ===")

    raw = load_mt5_csv(DATA_PATH)
    features = build_features(raw)

    result = add_barrier_outcomes(
        features,
        horizons=HORIZONS,
        targets_r=TARGETS_R,
    )

    momentum = result.loc[result["is_momentum_candle"]].copy()

    print(f"Total candles:       {len(result)}")
    print(f"Momentum candles:    {len(momentum)}")
    print(
        f"Bullish momentum:    "
        f"{int(momentum['is_bullish'].sum())}"
    )
    print(
        f"Bearish momentum:    "
        f"{int(momentum['is_bearish'].sum())}"
    )

    for target_r in TARGETS_R:
        print(f"\n{'=' * 60}")
        print(f"Target: {target_r:g}R")
        print(f"{'=' * 60}")

        for horizon in HORIZONS:
            column = f"barrier_{target_r:g}r_{horizon}"

            print_distribution(
                result,
                column,
                f"+{horizon} candle(s) — all momentum",
            )

            bullish = result[
                result["is_momentum_candle"]
                & result["is_bullish"]
            ]

            bearish = result[
                result["is_momentum_candle"]
                & result["is_bearish"]
            ]

            for name, subset in (
                ("bullish momentum", bullish),
                ("bearish momentum", bearish),
            ):
                values = subset[column].dropna()

                print(f"\n  {name}")
                print(f"    Samples: {len(values)}")

                counts = values.value_counts()

                for outcome in (
                    "TP_FIRST",
                    "SL_FIRST",
                    "BOTH_SAME_CANDLE",
                    "UNRESOLVED",
                ):
                    count = int(counts.get(outcome, 0))
                    percentage = (
                        count / len(values) * 100
                        if len(values)
                        else 0.0
                    )

                    print(
                        f"    {outcome:18s}: "
                        f"{count:4d} ({percentage:6.2f}%)"
                    )

            aligned = result[
                result["is_momentum_candle"]
                & (
                    (
                        result["is_bullish"]
                        & result["ema_alignment"].eq(1)
                    )
                    |
                    (
                        result["is_bearish"]
                        & result["ema_alignment"].eq(-1)
                    )
                )
            ]

            print("\n  EMA-aligned momentum")
            print(f"    Samples: {len(aligned)}")

            values = aligned[column].dropna()
            counts = values.value_counts()

            for outcome in (
                "TP_FIRST",
                "SL_FIRST",
                "BOTH_SAME_CANDLE",
                "UNRESOLVED",
            ):
                count = int(counts.get(outcome, 0))
                percentage = (
                    count / len(values) * 100
                    if len(values)
                    else 0.0
                )

                print(
                    f"    {outcome:18s}: "
                    f"{count:4d} ({percentage:6.2f}%)"
                )


if __name__ == "__main__":
    main()
