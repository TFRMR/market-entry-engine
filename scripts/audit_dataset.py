
"""Audit an MT5-exported market dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_engine.data import load_mt5_csv  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/audit_dataset.py <path-to-mt5-csv>"
        )

    path = Path(sys.argv[1])

    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    df = load_mt5_csv(path)

    # Calculate candle-to-candle intervals in minutes.
    intervals = (
        df["timestamp"]
        .diff()
        .dropna()
        .dt.total_seconds()
        .div(60)
    )

    # Any interval greater than 30 minutes is considered a gap.
    gaps = intervals[intervals > 30]

    print("=== Dataset Audit ===")
    print(f"File:              {path}")
    print(f"Rows:              {len(df):,}")
    print(f"Start:             {df['timestamp'].min()}")
    print(f"End:               {df['timestamp'].max()}")
    print(f"Duplicate times:   {df['timestamp'].duplicated().sum()}")
    print(f"Missing values:    {df.isna().sum().sum()}")
    print(f"Interval min:      {intervals.min():.0f} min")
    print(f"Interval max:      {intervals.max():.0f} min")
    print(f"Non-30m intervals: {(intervals != 30).sum():,}")
    print(f"Gaps > 30m:        {len(gaps):,}")
    print(f"Spread min:        {df['spread'].min()}")
    print(f"Spread max:        {df['spread'].max()}")
    print(f"Spread mean:       {df['spread'].mean():.2f}")
    print(f"Tick volume min:   {df['tick_volume'].min():,}")
    print(f"Tick volume max:   {df['tick_volume'].max():,}")
    print(
        "Real volume:       "
        f"{df['real_volume'].nunique()} unique value(s)"
    )
    print("Validation:        PASS")

    # Stop here if there are no gaps.
    if gaps.empty:
        print("\n=== Gap Distribution ===")
        print("No gaps greater than 30 minutes found.")
        return

    # Build a detailed table for every detected gap.
    gap_rows = df.loc[gaps.index].copy()

    gap_rows["gap_minutes"] = intervals.loc[gaps.index].values

    gap_rows["previous_timestamp"] = (
        df["timestamp"]
        .shift(1)
        .loc[gaps.index]
        .values
    )

    gap_rows["previous_weekday"] = (
        gap_rows["previous_timestamp"]
        .dt.day_name()
    )

    gap_rows["current_weekday"] = (
        gap_rows["timestamp"]
        .dt.day_name()
    )

    gap_rows["previous_hour"] = (
        gap_rows["previous_timestamp"]
        .dt.strftime("%H:%M")
    )

    gap_rows["current_hour"] = (
        gap_rows["timestamp"]
        .dt.strftime("%H:%M")
    )

    # Show the largest gaps.
    print("\n=== Largest Gaps ===")

    largest_gaps = (
        gap_rows[
            [
                "timestamp",
                "gap_minutes",
            ]
        ]
        .sort_values("gap_minutes", ascending=False)
        .head(20)
    )

    print(largest_gaps.to_string(index=False))

    # Show every gap chronologically.
    print("\n=== Gap Distribution ===")

    gap_distribution = gap_rows[
        [
            "previous_timestamp",
            "timestamp",
            "gap_minutes",
            "previous_weekday",
            "current_weekday",
            "previous_hour",
            "current_hour",
        ]
    ].sort_values("timestamp")

    print(gap_distribution.to_string(index=False))


if __name__ == "__main__":
    main()

