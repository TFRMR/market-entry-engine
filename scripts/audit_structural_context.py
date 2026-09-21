"""Audit structural-context heterogeneity against realized R.

This is a descriptive development-partition audit. It does not optimize
thresholds or train a model. Context values are taken from deterministic
setup-time facts and structural features already defined by the project.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.data import load_mt5_csv
from market_engine.entry import build_setup_candidates
from market_engine.features import build_structural_features
from market_engine.holdout import (
    HISTORICAL_AUDIT_CUTOFF,
    split_historical_boundary,
)
from market_engine.realized import build_realized_r_frame, cluster_bootstrap_mean
from market_engine.structure import (
    build_structural_sequence,
    process_structural_candles,
)


NUMERIC_CONTEXT = (
    "setup_broken_swing_age",
    "setup_entry_beyond_broken_swing_r",
    "setup_invalidation_swing_age",
    "pre_structure_distance_to_high",
    "pre_structure_distance_to_low",
    "pre_structure_bars_since_last_swing",
    "pre_structure_bars_since_last_bos",
)
CATEGORICAL_CONTEXT = (
    "setup_bos_external",
    "pre_structure_direction",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit structural-context heterogeneity against realized R."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--spread-price", type=float, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    return parser.parse_args()


def _quantile_summary(frame: pd.DataFrame, column: str) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = frame.loc[values.notna(), ["realized_r"]].copy()
    valid[column] = values[values.notna()]
    if len(valid) < 20 or valid[column].nunique() < 2:
        print(f"  {column}: insufficient variation (n={len(valid)})")
        return

    try:
        valid["bucket"] = pd.qcut(
            valid[column],
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        )
    except ValueError:
        print(f"  {column}: unable to form quartiles")
        return

    grouped = valid.groupby("bucket", observed=True)["realized_r"].agg(
        ["count", "mean", "median"]
    )
    print(f"  {column}:")
    for bucket, row in grouped.iterrows():
        print(
            f"    {bucket}: n={int(row['count']):4d} "
            f"mean={row['mean']:+.4f} median={row['median']:+.4f}"
        )


def _categorical_summary(frame: pd.DataFrame, column: str) -> None:
    print(f"  {column}:")
    grouped = frame.groupby(column, dropna=False)["realized_r"].agg(
        ["count", "mean", "median"]
    )
    for value, row in grouped.iterrows():
        print(
            f"    {str(value):>10s}: n={int(row['count']):4d} "
            f"mean={row['mean']:+.4f} median={row['median']:+.4f}"
        )


def main() -> None:
    args = _parse_args()
    if args.horizon <= 0:
        raise ValueError("horizon must be greater than zero.")
    if args.spread_price < 0:
        raise ValueError("spread-price must not be negative.")

    frame = load_mt5_csv(args.input_path)
    candidates = build_setup_candidates(frame)
    structural = build_structural_sequence(frame)
    swings, events = process_structural_candles(structural)

    realized = build_realized_r_frame(
        candidates=candidates,
        swings=swings,
        events=events,
        frame=frame,
        spread_price=args.spread_price,
        horizon=args.horizon,
    )
    if realized.empty:
        raise ValueError("No evaluable realized-R setups were found.")

    development, _, _ = split_historical_boundary(
        realized, frame, args.horizon, HISTORICAL_AUDIT_CUTOFF
    )

    structural_features = build_structural_features(frame)
    rows: list[dict[str, object]] = []
    pre_columns = (
        "structure_direction",
        "structure_distance_to_high",
        "structure_distance_to_low",
        "structure_bars_since_last_swing",
        "structure_bars_since_last_bos",
    )
    for _, row in development.iterrows():
        setup_index = int(row["setup_index"])
        feature_index = setup_index - 1
        context = {}
        if feature_index >= 0:
            for column in pre_columns:
                context[f"pre_{column}"] = structural_features.iloc[feature_index][column]
        rows.append(context)

    context_frame = pd.DataFrame(rows)
    audit = pd.concat(
        [development.reset_index(drop=True), context_frame.reset_index(drop=True)],
        axis=1,
    )

    print("=== Structural-Context Heterogeneity Audit ===")
    print()
    print("Partition:")
    print(f"  Historical-audit boundary: {HISTORICAL_AUDIT_CUTOFF}")
    print(f"  Development setups:        {len(audit):,}")
    print(f"  Spread (price units):      {args.spread_price}")
    print(f"  Horizon (candles):         {args.horizon}")
    print()
    print("Baseline:")
    mean, low, high = cluster_bootstrap_mean(
        audit["realized_r"].to_numpy(float),
        pd.to_datetime(audit["setup_timestamp"]).dt.date.to_numpy(),
        draws=2000,
        seed=7,
    )
    print(
        f"  mean R: {mean:+.4f}  95% CI (bootstrap by day): "
        f"[{low:+.4f}, {high:+.4f}]"
    )
    print()
    print("Categorical structural context (descriptive):")
    for column in CATEGORICAL_CONTEXT:
        _categorical_summary(audit, column)
    print()
    print("Numeric structural context, quartiles (descriptive):")
    for column in NUMERIC_CONTEXT:
        _quantile_summary(audit, column)
    print()
    print("Context coverage:")
    for column in (*CATEGORICAL_CONTEXT, *NUMERIC_CONTEXT):
        print(f"  {column:40s}: missing={int(audit[column].isna().sum()):4d}")
    print()
    print("Interpretation guardrails:")
    print("  - Quartiles are descriptive slices, not selected thresholds.")
    print("  - No feature is optimized against realized R in this audit.")
    print("  - Differences can reflect correlated contexts and setup overlap.")
    print("  - Results are development-partition evidence, not pristine OOS.")
    print("  - A useful next step requires stable heterogeneity, not a single best bucket.")
    print()
    print("=== Audit complete ===")


if __name__ == "__main__":
    main()
