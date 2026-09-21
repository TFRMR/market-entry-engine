"""Audit +5R-equivalent forward movement under different contexts."""

from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path(
    "data/processed/momentum_training_1r_10.csv"
)

HORIZON = 5


def summarize(
    frame: pd.DataFrame,
    name: str,
) -> dict:
    values = frame["favorable_r_5"].dropna()

    return {
        "group": name,
        "n": len(values),
        "mean_R": values.mean(),
        "median_R": values.median(),
        "positive_pct": (values > 0).mean(),
        "ge_0_5R_pct": (values >= 0.5).mean(),
        "ge_1R_pct": (values >= 1.0).mean(),
        "le_neg_0_5R_pct": (values <= -0.5).mean(),
        "le_neg_1R_pct": (values <= -1.0).mean(),
    }


def main() -> None:
    frame = pd.read_csv(INPUT_PATH)

    required = {
        "timestamp",
        "close",
        "barrier_risk",
        "is_bullish",
        "is_bearish",
        "ema_alignment",
        "breakout_above_20",
        "breakout_below_20",
        "breakout_above_50",
        "breakout_below_50",
        "range_vs_avg_3",
        "range_vs_avg_5",
        "volume_ratio_20",
        "return_3",
        "return_6",
        "return_12",
    }

    missing = required - set(frame.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"]
    )

    bullish = frame["is_bullish"].astype(bool)
    bearish = frame["is_bearish"].astype(bool)

    if (bullish == bearish).any():
        raise RuntimeError(
            "Invalid direction mapping."
        )

    future_close = frame["close"].shift(-HORIZON)

    raw_return = future_close - frame["close"]

    frame["favorable_r_5"] = np.where(
        bullish,
        raw_return / frame["barrier_risk"],
        -raw_return / frame["barrier_risk"],
    )

    groups = {
        "ALL": pd.Series(True, index=frame.index),

        "LONG": bullish,
        "SHORT": bearish,

        "LONG_EMA_BULL": (
            bullish
            & frame["ema_alignment"].eq(1)
        ),

        "LONG_EMA_NOT_BULL": (
            bullish
            & ~frame["ema_alignment"].eq(1)
        ),

        "SHORT_EMA_BEAR": (
            bearish
            & frame["ema_alignment"].eq(-1)
        ),

        "SHORT_EMA_NOT_BEAR": (
            bearish
            & ~frame["ema_alignment"].eq(-1)
        ),

        "LONG_BREAKOUT20": (
            bullish
            & frame["breakout_above_20"].astype(bool)
        ),

        "LONG_NO_BREAKOUT20": (
            bullish
            & ~frame["breakout_above_20"].astype(bool)
        ),

        "SHORT_BREAKOUT20": (
            bearish
            & frame["breakout_below_20"].astype(bool)
        ),

        "SHORT_NO_BREAKOUT20": (
            bearish
            & ~frame["breakout_below_20"].astype(bool)
        ),

        "LONG_BREAKOUT50": (
            bullish
            & frame["breakout_above_50"].astype(bool)
        ),

        "LONG_NO_BREAKOUT50": (
            bullish
            & ~frame["breakout_above_50"].astype(bool)
        ),

        "SHORT_BREAKOUT50": (
            bearish
            & frame["breakout_below_50"].astype(bool)
        ),

        "SHORT_NO_BREAKOUT50": (
            bearish
            & ~frame["breakout_below_50"].astype(bool)
        ),

        "RANGE_EXPANSION_1_5": (
            frame["range_vs_avg_3"] >= 1.5
        ),

        "RANGE_NO_EXPANSION_1_5": (
            frame["range_vs_avg_3"] < 1.5
        ),

        "RANGE_EXPANSION_2": (
            frame["range_vs_avg_3"] >= 2.0
        ),

        "RANGE_NO_EXPANSION_2": (
            frame["range_vs_avg_3"] < 2.0
        ),

        "HIGH_VOLUME": (
            frame["volume_ratio_20"] >= 1.5
        ),

        "NORMAL_VOLUME": (
            frame["volume_ratio_20"] < 1.5
        ),
    }

    rows = []

    for name, mask in groups.items():
        rows.append(
            summarize(
                frame.loc[mask],
                name,
            )
        )

    result = pd.DataFrame(rows)

    print("=== Conditional +5 Horizon Audit ===")
    print()

    print(
        result[
            [
                "group",
                "n",
                "mean_R",
                "median_R",
                "positive_pct",
                "ge_0_5R_pct",
                "ge_1R_pct",
                "le_neg_0_5R_pct",
                "le_neg_1R_pct",
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Direction x EMA
    # ------------------------------------------------------------------

    print()
    print("=== Direction x EMA ===")
    print()

    for direction_name, direction_mask in [
        ("LONG", bullish),
        ("SHORT", bearish),
    ]:
        for ema_name, ema_mask in [
            ("EMA_ALIGNED", frame["ema_alignment"].eq(
                1 if direction_name == "LONG" else -1
            )),
            ("EMA_NOT_ALIGNED", ~frame["ema_alignment"].eq(
                1 if direction_name == "LONG" else -1
            )),
        ]:
            subset = frame[
                direction_mask & ema_mask
            ]

            values = subset[
                "favorable_r_5"
            ].dropna()

            print(
                f"{direction_name:5s} "
                f"{ema_name:16s} "
                f"n={len(values):4d} "
                f"mean={values.mean():7.3f}R "
                f"median={values.median():7.3f}R "
                f"positive={((values > 0).mean() * 100):6.2f}%"
            )

    # ------------------------------------------------------------------
    # Direction x breakout
    # ------------------------------------------------------------------

    print()
    print("=== Direction x Breakout20 ===")
    print()

    for direction_name, direction_mask, breakout_column in [
        ("LONG", bullish, "breakout_above_20"),
        ("SHORT", bearish, "breakout_below_20"),
    ]:
        breakout = frame[breakout_column].astype(bool)

        for breakout_name, breakout_mask in [
            ("BREAKOUT", breakout),
            ("NO_BREAKOUT", ~breakout),
        ]:
            subset = frame[
                direction_mask & breakout_mask
            ]

            values = subset[
                "favorable_r_5"
            ].dropna()

            print(
                f"{direction_name:5s} "
                f"{breakout_name:12s} "
                f"n={len(values):4d} "
                f"mean={values.mean():7.3f}R "
                f"median={values.median():7.3f}R "
                f"positive={((values > 0).mean() * 100):6.2f}%"
            )

    # ------------------------------------------------------------------
    # Direction x regime context
    # ------------------------------------------------------------------

    print()
    print("=== Direction x Range Expansion ===")
    print()

    for direction_name, direction_mask in [
        ("LONG", bullish),
        ("SHORT", bearish),
    ]:
        for threshold in [1.5, 2.0]:
            expansion = (
                frame["range_vs_avg_3"] >= threshold
            )

            for expansion_name, expansion_mask in [
                ("EXPANSION", expansion),
                ("NORMAL", ~expansion),
            ]:
                subset = frame[
                    direction_mask & expansion_mask
                ]

                values = subset[
                    "favorable_r_5"
                ].dropna()

                print(
                    f"{direction_name:5s} "
                    f"range>={threshold:<3.1f} "
                    f"{expansion_name:9s} "
                    f"n={len(values):4d} "
                    f"mean={values.mean():7.3f}R "
                    f"median={values.median():7.3f}R "
                    f"positive={((values > 0).mean() * 100):6.2f}%"
                )

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    print()
    print("=== Direction x Volume ===")
    print()

    for direction_name, direction_mask in [
        ("LONG", bullish),
        ("SHORT", bearish),
    ]:
        high_volume = (
            frame["volume_ratio_20"] >= 1.5
        )

        for volume_name, volume_mask in [
            ("HIGH_VOLUME", high_volume),
            ("NORMAL_VOLUME", ~high_volume),
        ]:
            subset = frame[
                direction_mask & volume_mask
            ]

            values = subset[
                "favorable_r_5"
            ].dropna()

            print(
                f"{direction_name:5s} "
                f"{volume_name:12s} "
                f"n={len(values):4d} "
                f"mean={values.mean():7.3f}R "
                f"median={values.median():7.3f}R "
                f"positive={((values > 0).mean() * 100):6.2f}%"
            )

    # ------------------------------------------------------------------
    # Recent movement
    # ------------------------------------------------------------------

    print()
    print("=== Recent Movement Distribution ===")
    print()

    for column in [
        "return_3",
        "return_6",
        "return_12",
    ]:
        print(
            f"{column:10s} "
            f"median={frame[column].median():.6f} "
            f"mean={frame[column].mean():.6f}"
        )

    # ------------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------------

    print()
    print("=== Audit Checks ===")

    if result["n"].le(0).any():
        raise RuntimeError(
            "Found empty conditional group."
        )

    if not (
        result["positive_pct"]
        .dropna()
        .between(0.0, 1.0)
        .all()
    ):
        raise RuntimeError(
            "Positive percentage outside [0, 1]."
        )

    if not (
        result["ge_1R_pct"]
        .dropna()
        .between(0.0, 1.0)
        .all()
    ):
        raise RuntimeError(
            "1R percentage outside [0, 1]."
        )

    print("  Direction validity: passed.")
    print("  Conditional groups: passed.")
    print("  Rate bounds: passed.")
    print()
    print("=== Conditional horizon audit complete ===")


if __name__ == "__main__":
    main()
