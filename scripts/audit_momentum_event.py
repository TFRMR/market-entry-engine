"""Audit momentum candles as standalone event triggers."""

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
    "is_bullish",
    "is_bearish",
    "is_momentum_candle",
}

missing = sorted(required - set(df.columns))
if missing:
    raise SystemExit(f"Missing columns: {missing}")

if not df["timestamp"].is_monotonic_increasing:
    raise SystemExit("Timestamp order is not chronological.")

if df["timestamp"].duplicated().any():
    raise SystemExit("Duplicate timestamps found.")

if (df["is_bullish"] == df["is_bearish"]).any():
    raise SystemExit("Invalid direction flags.")

events = df[df["is_momentum_candle"]].copy().reset_index()

events["direction"] = np.where(
    events["is_bullish"],
    "LONG",
    "SHORT",
)


# ---------------------------------------------------------------------------
# Chronological periods
# ---------------------------------------------------------------------------

periods = pd.qcut(
    events["timestamp"].rank(method="first"),
    4,
    labels=["P1", "P2", "P3", "P4"],
)

events["period"] = periods


# ---------------------------------------------------------------------------
# Forward outcome calculations
# ---------------------------------------------------------------------------

def calculate_event(row, horizon):

    idx = int(row["index"])
    direction = row["direction"]
    entry = float(row["barrier_entry"])
    risk = float(row["barrier_risk"])

    future = df.iloc[idx + 1: idx + 1 + horizon]

    if len(future) == 0:
        return None

    if direction == "LONG":

        favorable = (
            future["high"].max() - entry
        ) / risk

        adverse = (
            entry - future["low"].min()
        ) / risk

        close_move = (
            future.iloc[-1]["close"] - entry
        ) / risk

    else:

        favorable = (
            entry - future["low"].min()
        ) / risk

        adverse = (
            future["high"].max() - entry
        ) / risk

        close_move = (
            entry - future.iloc[-1]["close"]
        ) / risk

    return {
        "mfe": favorable,
        "mae": adverse,
        "close_move": close_move,
    }


records = []

for _, event in events.iterrows():

    record = {
        "timestamp": event["timestamp"],
        "direction": event["direction"],
        "period": event["period"],
    }

    for horizon in [1, 3, 5, 10]:

        result = calculate_event(
            event,
            horizon,
        )

        if result is None:
            continue

        for key, value in result.items():
            record[f"{key}_{horizon}"] = value

    records.append(record)


outcomes = pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def summarize(frame, horizon):

    close = frame[f"close_move_{horizon}"]
    mfe = frame[f"mfe_{horizon}"]
    mae = frame[f"mae_{horizon}"]

    return {
        "n": len(frame),
        "mean_close_R": close.mean(),
        "median_close_R": close.median(),
        "positive_pct": (close > 0).mean(),
        "ge_0_5R": (close >= 0.5).mean(),
        "ge_1R": (close >= 1.0).mean(),
        "le_neg_0_5R": (close <= -0.5).mean(),
        "le_neg_1R": (close <= -1.0).mean(),
        "mean_MFE": mfe.mean(),
        "median_MFE": mfe.median(),
        "ge_1R_MFE": (mfe >= 1.0).mean(),
        "ge_2R_MFE": (mfe >= 2.0).mean(),
        "mean_MAE": mae.mean(),
        "median_MAE": mae.median(),
        "ge_1R_MAE": (mae >= 1.0).mean(),
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

print("=== MOMENTUM EVENT AUDIT ===")

rows = []

for horizon in [1, 3, 5, 10]:

    for group_name, group in [
        ("ALL", outcomes),
        ("LONG", outcomes[outcomes["direction"] == "LONG"]),
        ("SHORT", outcomes[outcomes["direction"] == "SHORT"]),
    ]:

        result = summarize(
            group,
            horizon,
        )

        result.update(
            {
                "horizon": horizon,
                "group": group_name,
            }
        )

        rows.append(result)


summary = pd.DataFrame(rows)

print()
print("=== AGGREGATE OUTCOMES ===")

print(
    summary[
        [
            "horizon",
            "group",
            "n",
            "mean_close_R",
            "median_close_R",
            "positive_pct",
            "ge_0_5R",
            "ge_1R",
            "mean_MFE",
            "median_MFE",
            "ge_1R_MFE",
            "ge_2R_MFE",
            "mean_MAE",
            "median_MAE",
            "ge_1R_MAE",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ---------------------------------------------------------------------------
# Chronological stability
# ---------------------------------------------------------------------------

print()
print("=== CHRONOLOGICAL CLOSE-MOVE OUTCOMES ===")

rows = []

for period in ["P1", "P2", "P3", "P4"]:

    for direction in ["ALL", "LONG", "SHORT"]:

        if direction == "ALL":
            group = outcomes[
                outcomes["period"] == period
            ]
        else:
            group = outcomes[
                (outcomes["period"] == period)
                & (outcomes["direction"] == direction)
            ]

        for horizon in [1, 3, 5, 10]:

            result = summarize(
                group,
                horizon,
            )

            rows.append(
                {
                    "period": period,
                    "direction": direction,
                    "horizon": horizon,
                    "n": result["n"],
                    "mean_close_R": result["mean_close_R"],
                    "median_close_R": result["median_close_R"],
                    "positive_pct": result["positive_pct"],
                    "ge_1R": result["ge_1R"],
                }
            )


stability = pd.DataFrame(rows)

print(
    stability.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ---------------------------------------------------------------------------
# Direction composition
# ---------------------------------------------------------------------------

print()
print("=== EVENT DIRECTION SHARE ===")

direction_share = pd.crosstab(
    events["period"],
    events["direction"],
    normalize="index",
)

print(
    direction_share.to_string(
        float_format=lambda x: f"{x:.4f}",
    )
)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

print()
print("=== CHECKS ===")

if len(events) != 2231:
    raise SystemExit(
        f"Expected 2231 momentum events, got {len(events)}"
    )

if not outcomes["direction"].isin(
    ["LONG", "SHORT"]
).all():
    raise SystemExit("Invalid event direction.")

for column in [
    "mfe_1",
    "mae_1",
    "close_move_1",
]:
    if not np.isfinite(
        outcomes[column].dropna()
    ).all():
        raise SystemExit(
            f"Non-finite values in {column}"
        )

print("Checks passed.")
