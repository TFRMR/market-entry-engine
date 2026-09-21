"""Audit raw LONG/SHORT outcome rates without using the model."""

from pathlib import Path

import pandas as pd

INPUT_PATH = Path(
    "data/processed/momentum_training_1r_10.csv"
)

OUTPUT_PATH = Path(
    "reports/direction_baseline_outcomes.csv"
)


def summarize(
    frame: pd.DataFrame,
    group_name: str,
    subset: pd.Series,
) -> dict:
    data = frame.loc[subset].copy()

    total = len(data)

    counts = data["barrier_1r_10"].value_counts()

    tp = int(counts.get("TP_FIRST", 0))
    sl = int(counts.get("SL_FIRST", 0))
    both = int(counts.get("BOTH_SAME_CANDLE", 0))
    unresolved = int(counts.get("UNRESOLVED", 0))

    resolved = tp + sl

    return {
        "group": group_name,
        "n": total,
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
    print("=== Direction Baseline Outcome Audit ===")

    frame = pd.read_csv(INPUT_PATH)

    required = {
        "timestamp",
        "is_bullish",
        "is_bearish",
        "barrier_1r_10",
        "ema_alignment",
        "breakout_above_20",
        "breakout_below_20",
        "breakout_above_50",
        "breakout_below_50",
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
            "Found ambiguous bullish/bearish direction."
        )

    groups = [
        ("ALL", pd.Series(True, index=frame.index)),
        ("LONG", bullish),
        ("SHORT", bearish),
        (
            "LONG_EMA_BULL",
            bullish & frame["ema_alignment"].eq(1),
        ),
        (
            "SHORT_EMA_BEAR",
            bearish & frame["ema_alignment"].eq(-1),
        ),
        (
            "LONG_BREAKOUT_20",
            bullish & frame["breakout_above_20"].astype(bool),
        ),
        (
            "SHORT_BREAKOUT_20",
            bearish & frame["breakout_below_20"].astype(bool),
        ),
        (
            "LONG_BREAKOUT_50",
            bullish & frame["breakout_above_50"].astype(bool),
        ),
        (
            "SHORT_BREAKOUT_50",
            bearish & frame["breakout_below_50"].astype(bool),
        ),
    ]

    rows = [
        summarize(frame, name, subset)
        for name, subset in groups
    ]

    result = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(result.to_string(index=False))

    print()
    print("=== Direction Comparison ===")

    direction = result[
        result["group"].isin(["LONG", "SHORT"])
    ][
        [
            "group",
            "n",
            "tp_first",
            "sl_first",
            "both",
            "unresolved",
            "resolved_n",
            "resolved_tp_rate",
        ]
    ]

    print(direction.to_string(index=False))

    print()
    print("=== Audit Checks ===")

    if result["n"].le(0).any():
        raise RuntimeError(
            "Found empty baseline group."
        )

    for _, row in result.iterrows():
        if (
            row["tp_first"]
            + row["sl_first"]
            + row["both"]
            + row["unresolved"]
            != row["n"]
        ):
            raise RuntimeError(
                f"Outcome counts do not sum for {row['group']}."
            )

    if not (
        result["resolved_tp_rate"]
        .dropna()
        .between(0.0, 1.0)
        .all()
    ):
        raise RuntimeError(
            "Resolved TP rate outside [0, 1]."
        )

    print("  Direction validity: passed.")
    print("  Outcome accounting: passed.")
    print("  Rate bounds: passed.")
    print()
    print("=== Baseline audit complete ===")


if __name__ == "__main__":
    main()
