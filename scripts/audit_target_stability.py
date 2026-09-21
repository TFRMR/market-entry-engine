"""Audit 1R/10 target stability across four chronological periods."""

from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/processed/momentum_training_1r_10.csv")
TARGET = "barrier_1r_10"


df = pd.read_csv(INPUT, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)


# Validate schema and chronology.
required = {"timestamp", "is_bullish", "is_bearish", TARGET}
missing = sorted(required - set(df.columns))

if missing:
    raise SystemExit(f"Missing columns: {missing}")

if not df["timestamp"].is_monotonic_increasing:
    raise SystemExit("Timestamp order is not chronological.")

if df["timestamp"].duplicated().any():
    raise SystemExit("Duplicate timestamps found.")


# Direction.
if (df["is_bullish"] == df["is_bearish"]).any():
    raise SystemExit("Invalid bullish/bearish direction.")

df["direction"] = np.where(
    df["is_bullish"],
    "LONG",
    "SHORT",
)


# Validate target labels.
expected = {
    "TP_FIRST",
    "SL_FIRST",
    "BOTH_SAME_CANDLE",
    "UNRESOLVED",
}

actual = set(df[TARGET].dropna().unique())
unexpected = sorted(actual - expected)

if unexpected:
    raise SystemExit(f"Unexpected target labels: {unexpected}")

if df[TARGET].isna().any():
    raise SystemExit("Missing target value found.")


# Four chronological periods.
df["period"] = pd.qcut(
    np.arange(len(df)),
    q=4,
    labels=["P1", "P2", "P3", "P4"],
)


def summarize(group):
    tp = int((group[TARGET] == "TP_FIRST").sum())
    sl = int((group[TARGET] == "SL_FIRST").sum())
    both = int((group[TARGET] == "BOTH_SAME_CANDLE").sum())
    unresolved = int((group[TARGET] == "UNRESOLVED").sum())

    resolved_ex_both = tp + sl

    return {
        "n": len(group),
        "tp": tp,
        "sl": sl,
        "both": both,
        "unresolved": unresolved,
        "resolved_ex_both": resolved_ex_both,
        "tp_rate": (
            tp / resolved_ex_both
            if resolved_ex_both
            else np.nan
        ),
        "unresolved_pct": unresolved / len(group),
    }


rows = []

for period in ["P1", "P2", "P3", "P4"]:
    period_df = df[df["period"] == period]

    for direction in ["ALL", "LONG", "SHORT"]:
        if direction == "ALL":
            subset = period_df
        else:
            subset = period_df[
                period_df["direction"] == direction
            ]

        row = summarize(subset)
        row["period"] = period
        row["direction"] = direction
        rows.append(row)


summary = pd.DataFrame(rows)

summary = summary[
    [
        "period",
        "direction",
        "n",
        "tp",
        "sl",
        "both",
        "unresolved",
        "resolved_ex_both",
        "tp_rate",
        "unresolved_pct",
    ]
]


print("=== TARGET STABILITY: 4 CHRONOLOGICAL PERIODS ===")
print()
print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print()
print("=== PERIOD BOUNDARIES ===")

for period in ["P1", "P2", "P3", "P4"]:
    part = df[df["period"] == period]

    print(
        f"{period}: "
        f"{part['timestamp'].min()} -> "
        f"{part['timestamp'].max()} "
        f"(n={len(part)})"
    )


print()
print("=== TP RATE BY PERIOD ===")

rates = summary.pivot(
    index="period",
    columns="direction",
    values="tp_rate",
)

print(
    rates.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


print()
print("=== TP RATE STABILITY ===")

for direction in ["ALL", "LONG", "SHORT"]:
    values = summary.loc[
        summary["direction"] == direction,
        "tp_rate",
    ].dropna()

    print(
        f"{direction}: "
        f"min={values.min():.4f} "
        f"max={values.max():.4f} "
        f"spread={values.max() - values.min():.4f} "
        f"mean={values.mean():.4f}"
    )


print()
print("=== LONG VS SHORT ===")

for period in ["P1", "P2", "P3", "P4"]:
    long_rate = summary.loc[
        (summary["period"] == period)
        & (summary["direction"] == "LONG"),
        "tp_rate",
    ].iloc[0]

    short_rate = summary.loc[
        (summary["period"] == period)
        & (summary["direction"] == "SHORT"),
        "tp_rate",
    ].iloc[0]

    print(
        f"{period}: "
        f"LONG={long_rate:.4f} "
        f"SHORT={short_rate:.4f} "
        f"LONG-SHORT={long_rate - short_rate:+.4f}"
    )


print()
print("=== ACCOUNTING CHECK ===")

counts = df[TARGET].value_counts()

total = sum(
    counts.get(label, 0)
    for label in expected
)

if total != len(df):
    raise SystemExit(
        f"Accounting mismatch: {total} != {len(df)}"
    )

print(
    f"TP={counts.get('TP_FIRST', 0)}, "
    f"SL={counts.get('SL_FIRST', 0)}, "
    f"BOTH={counts.get('BOTH_SAME_CANDLE', 0)}, "
    f"UNRESOLVED={counts.get('UNRESOLVED', 0)}"
)

print(f"Total={total}")
print("Checks passed.")
