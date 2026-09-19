"""Compare simple non-ML baselines across chronological periods."""

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/momentum_training_1r_10.csv"
)


def rate(frame: pd.DataFrame) -> float:
    tp = (frame["barrier_1r_10"] == "TP_FIRST").sum()
    sl = (frame["barrier_1r_10"] == "SL_FIRST").sum()
    resolved = tp + sl

    return tp / resolved if resolved else float("nan")


def summarize(
    period: str,
    name: str,
    frame: pd.DataFrame,
) -> dict:
    tp = (frame["barrier_1r_10"] == "TP_FIRST").sum()
    sl = (frame["barrier_1r_10"] == "SL_FIRST").sum()
    both = (frame["barrier_1r_10"] == "BOTH_SAME_CANDLE").sum()
    unresolved = (frame["barrier_1r_10"] == "UNRESOLVED").sum()

    return {
        "period": period,
        "baseline": name,
        "n": len(frame),
        "resolved_n": tp + sl,
        "tp_first": int(tp),
        "sl_first": int(sl),
        "both": int(both),
        "unresolved": int(unresolved),
        "resolved_tp_rate": rate(frame),
    }


def main() -> None:
    frame = pd.read_csv(INPUT_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    bullish = frame["is_bullish"].astype(bool)
    bearish = frame["is_bearish"].astype(bool)

    if (bullish == bearish).any():
        raise RuntimeError(
            "Invalid bullish/bearish direction."
        )

    # Same chronological four-way split used in the
    # previous stability audit.
    frame["period"] = pd.qcut(
        frame["timestamp"].rank(method="first"),
        q=4,
        labels=["P1", "P2", "P3", "P4"],
    )

    rows = []

    for period in ["P1", "P2", "P3", "P4"]:
        p = frame[frame["period"] == period]

        baselines = {
            "ALL": p,

            "LONG": p[bullish.loc[p.index]],

            "SHORT": p[bearish.loc[p.index]],

            "LONG_EMA": p[
                bullish.loc[p.index]
                & p["ema_alignment"].eq(1)
            ],

            "SHORT_EMA": p[
                bearish.loc[p.index]
                & p["ema_alignment"].eq(-1)
            ],

            "LONG_BREAKOUT20": p[
                bullish.loc[p.index]
                & p["breakout_above_20"].astype(bool)
            ],

            "SHORT_BREAKOUT20": p[
                bearish.loc[p.index]
                & p["breakout_below_20"].astype(bool)
            ],

            "LONG_BREAKOUT50": p[
                bullish.loc[p.index]
                & p["breakout_above_50"].astype(bool)
            ],

            "SHORT_BREAKOUT50": p[
                bearish.loc[p.index]
                & p["breakout_below_50"].astype(bool)
            ],
        }

        for name, subset in baselines.items():
            rows.append(
                summarize(
                    period,
                    name,
                    subset,
                )
            )

    result = pd.DataFrame(rows)

    print("=== Simple Baseline Comparison ===")
    print()
    print(
        result[
            [
                "period",
                "baseline",
                "n",
                "resolved_n",
                "resolved_tp_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print("=== Aggregate Baselines ===")

    aggregate_rows = []

    baselines = result["baseline"].unique()

    for name in baselines:
        subset = frame.copy()

        if name == "ALL":
            selected = subset

        elif name == "LONG":
            selected = subset[bullish]

        elif name == "SHORT":
            selected = subset[bearish]

        elif name == "LONG_EMA":
            selected = subset[
                bullish & subset["ema_alignment"].eq(1)
            ]

        elif name == "SHORT_EMA":
            selected = subset[
                bearish & subset["ema_alignment"].eq(-1)
            ]

        elif name == "LONG_BREAKOUT20":
            selected = subset[
                bullish
                & subset["breakout_above_20"].astype(bool)
            ]

        elif name == "SHORT_BREAKOUT20":
            selected = subset[
                bearish
                & subset["breakout_below_20"].astype(bool)
            ]

        elif name == "LONG_BREAKOUT50":
            selected = subset[
                bullish
                & subset["breakout_above_50"].astype(bool)
            ]

        elif name == "SHORT_BREAKOUT50":
            selected = subset[
                bearish
                & subset["breakout_below_50"].astype(bool)
            ]

        else:
            raise RuntimeError(
                f"Unknown baseline: {name}"
            )

        aggregate_rows.append(
            summarize(
                "ALL",
                name,
                selected,
            )
        )

    aggregate = pd.DataFrame(aggregate_rows)

    print(
        aggregate[
            [
                "baseline",
                "n",
                "resolved_n",
                "tp_first",
                "sl_first",
                "both",
                "unresolved",
                "resolved_tp_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print("=== Stability Spread ===")

    stability = (
        result
        .groupby("baseline")["resolved_tp_rate"]
        .agg(["min", "max", "mean", "std"])
        .reset_index()
    )

    stability["spread"] = (
        stability["max"] - stability["min"]
    )

    print(
        stability.to_string(index=False)
    )

    print()
    print("=== Audit Checks ===")

    for _, row in aggregate.iterrows():
        total = (
            row["tp_first"]
            + row["sl_first"]
            + row["both"]
            + row["unresolved"]
        )

        if total != row["n"]:
            raise RuntimeError(
                f"Outcome accounting failed for {row['baseline']}."
            )

    if not (
        aggregate["resolved_tp_rate"]
        .dropna()
        .between(0.0, 1.0)
        .all()
    ):
        raise RuntimeError(
            "Rate outside [0, 1]."
        )

    if not (
        stability["spread"] >= 0
    ).all():
        raise RuntimeError(
            "Invalid stability spread."
        )

    print("  Outcome accounting: passed.")
    print("  Rate bounds: passed.")
    print("  Stability calculation: passed.")
    print()
    print("=== Simple baseline audit complete ===")


if __name__ == "__main__":
    main()
