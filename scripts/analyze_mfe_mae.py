"""Analyze MFE/MAE in R for momentum-candle setups."""

from pathlib import Path

import pandas as pd

from market_engine.data import load_mt5_csv
from market_engine.features import build_features
from market_engine.outcomes import add_forward_excursions

DATA_PATH = Path(
    "/home/tofarmer/Desktop/XAUUSDc_M30_202601012300_202609181730.csv"
)


def print_stats(frame: pd.DataFrame, label: str) -> None:
    print(f"\n{label}")
    print(f"  Samples: {len(frame)}")

    if frame.empty:
        print("  No data")
        return

    for horizon in (1, 3, 5, 10):
        mfe = frame[f"mfe_r_{horizon}"].dropna()
        mae = frame[f"mae_r_{horizon}"].dropna()

        print(f"\n  +{horizon} candle(s)")

        if mfe.empty:
            print("    No data")
            continue

        print(
            f"    MFE mean:   {mfe.mean():.3f}R"
        )
        print(
            f"    MFE median: {mfe.median():.3f}R"
        )
        print(
            f"    MFE max:    {mfe.max():.3f}R"
        )
        print(
            f"    MFE >= 1R:  {(mfe >= 1.0).mean() * 100:.2f}%"
        )
        print(
            f"    MFE >= 1.5R: {(mfe >= 1.5).mean() * 100:.2f}%"
        )
        print(
            f"    MFE >= 2R:  {(mfe >= 2.0).mean() * 100:.2f}%"
        )

        print(
            f"    MAE mean:   {mae.mean():.3f}R"
        )
        print(
            f"    MAE median: {mae.median():.3f}R"
        )
        print(
            f"    MAE max:    {mae.max():.3f}R"
        )
        print(
            f"    MAE >= 1R:  {(mae >= 1.0).mean() * 100:.2f}%"
        )


def main() -> None:
    raw = load_mt5_csv(DATA_PATH)
    features = build_features(raw)

    momentum = features[features["is_momentum_candle"]].copy()

    result = add_forward_excursions(
        momentum,
        horizons=(1, 3, 5, 10),
    )

    print("=== MFE / MAE Momentum Analysis ===")
    print(f"Total candles:       {len(raw)}")
    print(f"Momentum candles:    {len(result)}")
    print(
        f"Bullish momentum:    "
        f"{(result['close'] > result['open']).sum()}"
    )
    print(
        f"Bearish momentum:    "
        f"{(result['close'] < result['open']).sum()}"
    )

    print_stats(result, "All momentum candles")

    bullish = result[result["close"] > result["open"]]
    bearish = result[result["close"] < result["open"]]

    print_stats(bullish, "Bullish momentum")
    print_stats(bearish, "Bearish momentum")

    ema_bullish = result[result["ema_alignment"] == 1]
    ema_bearish = result[result["ema_alignment"] == -1]

    print_stats(
        ema_bullish,
        "EMA bullish alignment",
    )
    print_stats(
        ema_bearish,
        "EMA bearish alignment",
    )

    bullish_aligned = result[
        (result["close"] > result["open"])
        & (result["ema_alignment"] == 1)
    ]

    bearish_aligned = result[
        (result["close"] < result["open"])
        & (result["ema_alignment"] == -1)
    ]

    print_stats(
        bullish_aligned,
        "Bullish momentum + bullish EMA alignment",
    )
    print_stats(
        bearish_aligned,
        "Bearish momentum + bearish EMA alignment",
    )

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
