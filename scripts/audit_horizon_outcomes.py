"""Audit momentum-candle outcomes across forward horizons."""

from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path(
    "data/processed/momentum_training_1r_10.csv"
)


HORIZONS = [1, 3, 5, 10]


def build_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    # Entry is the momentum candle close.
    entry = frame["close"]

    # Risk proxy used throughout the existing 1R analysis.
    risk = frame["barrier_risk"]

    for horizon in HORIZONS:
        future_close = frame["close"].shift(-horizon)

        raw_return = future_close - entry

        frame[f"forward_return_{horizon}"] = raw_return

        frame[f"forward_return_r_{horizon}"] = (
            raw_return / risk
        )

        frame[f"favorable_r_{horizon}"] = np.where(
            frame["is_bullish"],
            raw_return / risk,
            -raw_return / risk,
        )

    return frame


def summarize(
    frame: pd.DataFrame,
    horizon: int,
) -> dict:
    values = frame[
        f"favorable_r_{horizon}"
    ].dropna()

    return {
        "n": len(values),
        "mean_R": values.mean(),
        "median_R": values.median(),
        "positive_pct": (values > 0).mean(),
        "negative_pct": (values < 0).mean(),
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
    }

    missing = required - set(frame.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    frame = build_outcomes(frame)

    bullish = frame["is_bullish"].astype(bool)
    bearish = frame["is_bearish"].astype(bool)

    if (bullish == bearish).any():
        raise RuntimeError(
            "Invalid direction mapping."
        )

    print("=== Forward Horizon Outcome Audit ===")
    print()

    rows = []

    for horizon in HORIZONS:
        for name, mask in [
            ("ALL", pd.Series(True, index=frame.index)),
            ("LONG", bullish),
            ("SHORT", bearish),
        ]:
            result = summarize(
                frame.loc[mask],
                horizon,
            )

            result["horizon"] = horizon
            result["group"] = name
            rows.append(result)

    result = pd.DataFrame(rows)

    print(
        result[
            [
                "horizon",
                "group",
                "n",
                "mean_R",
                "median_R",
                "positive_pct",
                "negative_pct",
                "ge_0_5R_pct",
                "ge_1R_pct",
                "le_neg_0_5R_pct",
                "le_neg_1R_pct",
            ]
        ].to_string(index=False)
    )

    print()
    print("=== Chronological Stability ===")
    print()

    frame["period"] = pd.qcut(
        frame["timestamp"].rank(method="first"),
        q=4,
        labels=["P1", "P2", "P3", "P4"],
    )

    stability_rows = []

    for period in ["P1", "P2", "P3", "P4"]:
        p = frame[frame["period"] == period]

        for horizon in HORIZONS:
            for name, mask in [
                ("ALL", pd.Series(True, index=p.index)),
                (
                    "LONG",
                    bullish.loc[p.index],
                ),
                (
                    "SHORT",
                    bearish.loc[p.index],
                ),
            ]:
                values = p.loc[
                    mask,
                    f"favorable_r_{horizon}",
                ].dropna()

                stability_rows.append(
                    {
                        "period": period,
                        "horizon": horizon,
                        "group": name,
                        "n": len(values),
                        "mean_R": values.mean(),
                        "median_R": values.median(),
                        "positive_pct": (
                            values > 0
                        ).mean(),
                    }
                )

    stability = pd.DataFrame(
        stability_rows
    )

    print(
        stability.to_string(index=False)
    )

    print()
    print("=== Horizon Summary ===")

    all_groups = result[
        result["group"] == "ALL"
    ].copy()

    print(
        all_groups[
            [
                "horizon",
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

    print()
    print("=== Audit Checks ===")

    if result["n"].le(0).any():
        raise RuntimeError(
            "Found empty horizon result."
        )

    if not (
        result["positive_pct"]
        .between(0.0, 1.0)
        .all()
    ):
        raise RuntimeError(
            "Positive percentage outside [0, 1]."
        )

    if not (
        result["negative_pct"]
        .between(0.0, 1.0)
        .all()
    ):
        raise RuntimeError(
            "Negative percentage outside [0, 1]."
        )

    if not (
        stability["positive_pct"]
        .dropna()
        .between(0.0, 1.0)
        .all()
    ):
        raise RuntimeError(
            "Stability positive percentage outside [0, 1]."
        )

    print("  Direction validity: passed.")
    print("  Horizon calculations: passed.")
    print("  Rate bounds: passed.")
    print()
    print("=== Horizon outcome audit complete ===")


if __name__ == "__main__":
    main()
