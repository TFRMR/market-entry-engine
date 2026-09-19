"""Audit momentum +5 outcomes by period, direction, and EMA alignment."""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("data/processed/momentum_training_1r_10.csv")

df = pd.read_csv(INPUT, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

required = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "barrier_entry",
    "barrier_risk",
    "is_momentum_candle",
    "is_bullish",
    "is_bearish",
    "ema_5",
    "ema_20",
}

missing = sorted(required - set(df.columns))
if missing:
    raise SystemExit(f"Missing columns: {missing}")

events = df[df["is_momentum_candle"]].copy()
events["direction"] = np.where(
    events["is_bullish"],
    "LONG",
    "SHORT",
)

# EMA alignment at the event candle.
events["ema_aligned"] = np.where(
    events["direction"] == "LONG",
    events["ema_5"] > events["ema_20"],
    events["ema_5"] < events["ema_20"],
)

# Chronological periods, matching the previous event audits.
events["period"] = pd.qcut(
    events["timestamp"].rank(method="first"),
    4,
    labels=["P1", "P2", "P3", "P4"],
)


def outcome(row, horizon=5):
    idx = int(row.name)
    future = df.iloc[idx + 1:idx + 1 + horizon]

    if len(future) < horizon:
        return np.nan

    entry = float(row["barrier_entry"])
    risk = float(row["barrier_risk"])

    if row["direction"] == "LONG":
        return (
            future.iloc[-1]["close"] - entry
        ) / risk

    return (
        entry - future.iloc[-1]["close"]
    ) / risk


# Important: use original dataframe index for forward lookup.
events["_df_index"] = events.index

values = []

for _, row in events.iterrows():

    original_index = int(row["_df_index"])

    future = df.iloc[
        original_index + 1:
        original_index + 6
    ]

    if len(future) < 5:
        continue

    entry = float(row["barrier_entry"])
    risk = float(row["barrier_risk"])

    if row["direction"] == "LONG":
        close_r = (
            future.iloc[-1]["close"] - entry
        ) / risk
    else:
        close_r = (
            entry - future.iloc[-1]["close"]
        ) / risk

    values.append(
        {
            "period": row["period"],
            "direction": row["direction"],
            "ema_aligned": bool(row["ema_aligned"]),
            "close_r_5": close_r,
        }
    )


result = pd.DataFrame(values)


print("=== P1-P4 × DIRECTION × EMA: +5 ===")

summary = (
    result
    .groupby(
        ["period", "direction", "ema_aligned"],
        observed=True,
    )
    .agg(
        n=("close_r_5", "size"),
        mean_R=("close_r_5", "mean"),
        median_R=("close_r_5", "median"),
        positive_pct=("close_r_5", lambda x: (x > 0).mean()),
        ge_0_5R=("close_r_5", lambda x: (x >= 0.5).mean()),
        ge_1R=("close_r_5", lambda x: (x >= 1.0).mean()),
        le_neg_0_5R=("close_r_5", lambda x: (x <= -0.5).mean()),
        le_neg_1R=("close_r_5", lambda x: (x <= -1.0).mean()),
    )
    .reset_index()
)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print()
print("=== EMA ALIGNMENT SHARE ===")

share = (
    result
    .groupby(
        ["period", "direction"],
        observed=True,
    )["ema_aligned"]
    .mean()
    .reset_index(name="ema_aligned_share")
)

print(
    share.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print()
print("=== CHECKS ===")

if result.empty:
    raise SystemExit("No outcome rows.")

if not np.isfinite(result["close_r_5"]).all():
    raise SystemExit("Non-finite outcome found.")

if not result["direction"].isin(["LONG", "SHORT"]).all():
    raise SystemExit("Invalid direction.")

print("Checks passed.")
