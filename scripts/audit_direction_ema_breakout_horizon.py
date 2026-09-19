"""Audit direction x EMA 5/20 and breakout context across chronological periods."""

from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path("data/processed/momentum_training_1r_10.csv")
HORIZON = 5


def summarize(frame: pd.DataFrame) -> dict:
    x = frame["favorable_r_5"].dropna()
    return {
        "n": len(x),
        "mean_R": x.mean(),
        "median_R": x.median(),
        "positive_pct": (x > 0).mean() * 100,
        "ge_0_5R": (x >= 0.5).mean() * 100,
        "ge_1R": (x >= 1.0).mean() * 100,
    }


def print_group(frame: pd.DataFrame, name: str) -> None:
    s = summarize(frame)
    print(
        f"{name:24s} n={s['n']:4d} "
        f"mean={s['mean_R']:+.4f} "
        f"median={s['median_R']:+.4f} "
        f"positive={s['positive_pct']:6.2f}% "
        f">=0.5R={s['ge_0_5R']:6.2f}% "
        f">=1R={s['ge_1R']:6.2f}%"
    )


df = pd.read_csv(INPUT, parse_dates=["timestamp"])

required = [
    "timestamp",
    "is_bullish",
    "ema_5",
    "ema_20",
    "barrier_risk",
    "close",
    "high",
    "low",
    "previous_high_20",
    "previous_low_20",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")

# Exploratory +5 close-to-close movement normalized by barrier_risk.
# This is NOT executable trade P&L.
df["future_close_5"] = df["close"].shift(-HORIZON)

df["favorable_r_5"] = np.where(
    df["is_bullish"],
    (df["future_close_5"] - df["close"]) / df["barrier_risk"],
    (df["close"] - df["future_close_5"]) / df["barrier_risk"],
)

df["direction"] = np.where(df["is_bullish"], "LONG", "SHORT")

# EMA 5/20 only.
df["ema_bull"] = df["ema_5"] > df["ema_20"]
df["ema_bear"] = df["ema_5"] < df["ema_20"]

# Direction-specific breakout.
df["breakout20_high"] = df["high"] > df["previous_high_20"]
df["breakout20_low"] = df["low"] < df["previous_low_20"]

df = df.dropna(subset=["favorable_r_5"]).copy()

# Same chronological 4-period methodology used in the previous audits.
df["period"] = pd.qcut(
    np.arange(len(df)),
    4,
    labels=["P1", "P2", "P3", "P4"],
)

print("=== Direction x EMA 5/20 x +5 Horizon ===")
print(
    "EMA alignment: EMA5 > EMA20 for bullish; "
    "EMA5 < EMA20 for bearish."
)
print(
    "+5 metric = exploratory favorable close-to-close movement / "
    "barrier_risk, not trade P&L.\n"
)

for period, part in df.groupby("period", observed=True):
    print(
        f"--- {period}: "
        f"{part.timestamp.min()} -> {part.timestamp.max()} ---"
    )

    groups = {
        "ALL": part,
        "LONG": part[part.direction == "LONG"],
        "SHORT": part[part.direction == "SHORT"],

        "LONG_EMA_BULL": part[
            (part.direction == "LONG") & part.ema_bull
        ],
        "LONG_EMA_NOT_BULL": part[
            (part.direction == "LONG") & ~part.ema_bull
        ],

        "SHORT_EMA_BEAR": part[
            (part.direction == "SHORT") & part.ema_bear
        ],
        "SHORT_EMA_NOT_BEAR": part[
            (part.direction == "SHORT") & ~part.ema_bear
        ],

        "LONG_BREAKOUT20": part[
            (part.direction == "LONG") & part.breakout20_high
        ],
        "LONG_NO_BREAKOUT20": part[
            (part.direction == "LONG") & ~part.breakout20_high
        ],

        "SHORT_BREAKOUT20": part[
            (part.direction == "SHORT") & part.breakout20_low
        ],
        "SHORT_NO_BREAKOUT20": part[
            (part.direction == "SHORT") & ~part.breakout20_low
        ],
    }

    for name, subset in groups.items():
        print_group(subset, name)

    def mean(mask):
        x = part.loc[mask, "favorable_r_5"].dropna()
        return x.mean() if len(x) else np.nan

    long_ema_delta = (
        mean((part.direction == "LONG") & part.ema_bull)
        - mean((part.direction == "LONG") & ~part.ema_bull)
    )

    short_ema_delta = (
        mean((part.direction == "SHORT") & part.ema_bear)
        - mean((part.direction == "SHORT") & ~part.ema_bear)
    )

    long_break_delta = (
        mean((part.direction == "LONG") & part.breakout20_high)
        - mean((part.direction == "LONG") & ~part.breakout20_high)
    )

    short_break_delta = (
        mean((part.direction == "SHORT") & part.breakout20_low)
        - mean((part.direction == "SHORT") & ~part.breakout20_low)
    )

    print(
        f"EFFECT EMA:   "
        f"LONG aligned - not = {long_ema_delta:+.4f}R | "
        f"SHORT aligned - not = {short_ema_delta:+.4f}R"
    )

    print(
        f"EFFECT BREAK: "
        f"LONG breakout - no = {long_break_delta:+.4f}R | "
        f"SHORT breakout - no = {short_break_delta:+.4f}R"
    )

    print()

# Audit checks.
assert df.timestamp.is_monotonic_increasing
assert df["direction"].isin(["LONG", "SHORT"]).all()
assert ((df.direction == "LONG") == df.is_bullish).all()
assert (df.ema_bull ^ df.ema_bear).all()
assert set(df.period.astype(str)) == {"P1", "P2", "P3", "P4"}

print("Checks passed.")
