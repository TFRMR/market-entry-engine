"""Audit close-position regime shift by period and direction."""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("data/processed/momentum_model_1r_10_binary.csv")

df = pd.read_csv(INPUT, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)


required = {
    "timestamp",
    "is_bullish",
    "is_bearish",
    "close_position",
    "target",
}

missing = sorted(required - set(df.columns))

if missing:
    raise SystemExit(f"Missing columns: {missing}")

if df["timestamp"].duplicated().any():
    raise SystemExit("Duplicate timestamps found.")

if not df["timestamp"].is_monotonic_increasing:
    raise SystemExit("Timestamp order is not chronological.")

df["direction"] = np.where(
    df["is_bullish"],
    "LONG",
    "SHORT",
)

df["period"] = pd.qcut(
    np.arange(len(df)),
    q=4,
    labels=["P1", "P2", "P3", "P4"],
)


print("=== CLOSE POSITION REGIME AUDIT ===")
print()


print("=== BY PERIOD ===")

rows = []

for period in ["P1", "P2", "P3", "P4"]:

    part = df[df["period"] == period]

    for direction in ["ALL", "LONG", "SHORT"]:

        if direction == "ALL":
            subset = part
        else:
            subset = part[
                part["direction"] == direction
            ]

        rows.append(
            {
                "period": period,
                "direction": direction,
                "n": len(subset),
                "median": subset["close_position"].median(),
                "mean": subset["close_position"].mean(),
                "q25": subset["close_position"].quantile(0.25),
                "q75": subset["close_position"].quantile(0.75),
                "target_rate": subset["target"].mean(),
            }
        )


summary = pd.DataFrame(rows)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print()
print("=== BULLISH / BEARISH SHARE ===")

direction_share = (
    df.groupby("period", observed=True)["direction"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
)

print(
    direction_share.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


print()
print("=== CLOSE POSITION EXTREMES ===")

for period in ["P1", "P2", "P3", "P4"]:

    part = df[df["period"] == period]

    low_close = (part["close_position"] <= 0.25).mean()
    high_close = (part["close_position"] >= 0.75).mean()

    print(
        f"{period}: "
        f"close<=0.25={low_close:.4f} "
        f"close>=0.75={high_close:.4f}"
    )


print()
print("=== CHECKS ===")

if not df["close_position"].between(0, 1).all():
    raise SystemExit(
        "close_position outside expected [0, 1] range."
    )

print("Checks passed.")
