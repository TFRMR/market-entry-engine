"""Audit LONG/SHORT outcome stability across chronological periods."""

from pathlib import Path

import pandas as pd

INPUT_PATH = Path(
    "data/processed/momentum_training_1r_10.csv"
)


def summarize(frame: pd.DataFrame, name: str) -> dict:
    counts = frame["barrier_1r_10"].value_counts()

    tp = int(counts.get("TP_FIRST", 0))
    sl = int(counts.get("SL_FIRST", 0))
    both = int(counts.get("BOTH_SAME_CANDLE", 0))
    unresolved = int(counts.get("UNRESOLVED", 0))

    resolved = tp + sl

    return {
        "group": name,
        "n": len(frame),
        "tp_first": tp,
        "sl_first": sl,
        "both": both,
        "unresolved": unresolved,
        "resolved_n": resolved,
        "resolved_tp_rate": (
            tp / resolved if resolved else float("nan")
        ),
    }


def main() -> None:
    frame = pd.read_csv(INPUT_PATH)

    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    bullish = frame["is_bullish"].astype(bool)
    bearish = frame["is_bearish"].astype(bool)

    if (bullish == bearish).any():
        raise RuntimeError(
            "Invalid bullish/bearish direction mapping."
        )

    # Four chronological periods.
    frame["period"] = pd.qcut(
        frame["timestamp"].rank(method="first"),
        q=4,
        labels=[
            "P1",
            "P2",
            "P3",
            "P4",
        ],
    )

    rows = []

    for period in ["P1", "P2", "P3", "P4"]:
        period_frame = frame[frame["period"] == period]

        groups = [
            ("ALL", period_frame),
            (
                "LONG",
                period_frame[bullish.loc[period_frame.index]],
            ),
            (
                "SHORT",
                period_frame[bearish.loc[period_frame.index]],
            ),
            (
                "LONG_EMA_BULL",
                period_frame[
                    bullish.loc[period_frame.index]
                    & period_frame["ema_alignment"].eq(1)
                ],
            ),
            (
                "SHORT_EMA_BEAR",
                period_frame[
                    bearish.loc[period_frame.index]
                    & period_frame["ema_alignment"].eq(-1)
                ],
            ),
            (
                "LONG_BREAKOUT_20",
                period_frame[
                    bullish.loc[period_frame.index]
                    & period_frame["breakout_above_20"].astype(bool)
                ],
            ),
            (
                "SHORT_BREAKOUT_20",
                period_frame[
                    bearish.loc[period_frame.index]
                    & period_frame["breakout_below_20"].astype(bool)
                ],
            ),
        ]

        for name, subset in groups:
            result = summarize(subset, name)
            result["period"] = period
            rows.append(result)

    result = pd.DataFrame(rows)

    print("=== Direction Stability Audit ===")
    print()
    print(
        result[
            [
                "period",
                "group",
                "n",
                "resolved_n",
                "resolved_tp_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print("=== Period Date Ranges ===")

    for period in ["P1", "P2", "P3", "P4"]:
        subset = frame[frame["period"] == period]

        print(
            f"{period}: "
            f"{subset['timestamp'].min()} "
            f"-> "
            f"{subset['timestamp'].max()}"
        )

    print()
    print("=== Audit Checks ===")

    if result["n"].le(0).any():
        raise RuntimeError(
            "Found empty group."
        )

    if not (
        result["resolved_tp_rate"]
        .dropna()
        .between(0.0, 1.0)
        .all()
    ):
        raise RuntimeError(
            "Rate outside [0, 1]."
        )

    print("  Non-empty groups: passed.")
    print("  Rate bounds: passed.")
    print()
    print("=== Stability audit complete ===")


if __name__ == "__main__":
    main()
