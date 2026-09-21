"""Audit stability of locked structural context across time and spread.

This is a predefined development-partition audit. It uses the same structural
context contract as the heterogeneity audit and does not search for thresholds,
new features, or a best-performing bucket.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.data import load_mt5_csv
from market_engine.entry import build_setup_candidates
from market_engine.features import build_structural_features
from market_engine.holdout import HISTORICAL_AUDIT_CUTOFF, split_historical_boundary
from market_engine.realized import build_realized_r_frame
from market_engine.setup_facts import SetupFactContext
from market_engine.structure import build_structural_sequence, process_structural_candles


SPREADS = (0.00, 0.16, 0.30, 0.50)
PERIODS = (
    ("P1", "2024-09-01", "2025-03-01"),
    ("P2", "2025-03-01", "2025-09-01"),
    ("P3", "2025-09-01", "2026-03-18"),
)
CATEGORICAL_CONTEXT = ("setup_bos_external", "pre_structure_direction")
NUMERIC_CONTEXT = (
    "setup_broken_swing_age",
    "setup_entry_beyond_broken_swing_r",
    "setup_invalidation_swing_age",
    "pre_structure_distance_to_high",
    "pre_structure_distance_to_low",
    "pre_structure_bars_since_last_swing",
    "pre_structure_bars_since_last_bos",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--horizon", type=int, default=10)
    return parser.parse_args()


def _context_frame(
    development: pd.DataFrame,
    candidates,
    swings,
    events,
    structural_features: pd.DataFrame,
) -> pd.DataFrame:
    fact_context = SetupFactContext.build(swings, events)
    candidate_by_index = {candidate.setup_index: candidate for candidate in candidates}
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
        candidate = candidate_by_index.get(setup_index)
        context = fact_context.facts(candidate) if candidate is not None else {}
        context.pop("setup_bos_external", None)
        feature_index = setup_index - 1
        if feature_index >= 0:
            for column in pre_columns:
                context[f"pre_{column}"] = structural_features.iloc[feature_index][column]
        rows.append(context)
    return pd.DataFrame(rows)


def _mean(frame: pd.DataFrame) -> float:
    return float(frame["realized_r"].mean()) if not frame.empty else float("nan")


def _direction_summary(frame: pd.DataFrame, column: str) -> None:
    for value in (-1.0, 1.0):
        part = frame[frame[column] == value]
        print(
            f"    {value:+.0f}: n={len(part):4d} mean={_mean(part):+.4f}"
            if not part.empty
            else f"    {value:+.0f}: n=   0 mean=nan"
        )


def _quartile_gap(frame: pd.DataFrame, column: str) -> None:
    valid = frame[[column, "realized_r"]].dropna().copy()
    if len(valid) < 40 or valid[column].nunique() < 2:
        print(f"    {column}: insufficient data (n={len(valid)})")
        return
    valid["bucket"] = pd.qcut(
        valid[column], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    )
    grouped = valid.groupby("bucket", observed=True)["realized_r"].mean()
    values = {str(k): float(v) for k, v in grouped.items()}
    present = [values[k] for k in ("Q1", "Q2", "Q3", "Q4") if k in values]
    gap = max(present) - min(present) if present else float("nan")
    print(
        f"    {column}: "
        + " ".join(f"{k}={values.get(k, float('nan')):+.4f}" for k in ("Q1", "Q2", "Q3", "Q4"))
        + f" | spread={gap:.4f}"
    )


def main() -> None:
    args = _parse_args()
    if args.horizon <= 0:
        raise ValueError("horizon must be greater than zero.")

    frame = load_mt5_csv(args.input_path)
    candidates = build_setup_candidates(frame)
    structural = build_structural_sequence(frame)
    swings, events = process_structural_candles(structural)
    structural_features = build_structural_features(frame)
    realized_by_spread = {}

    for spread in SPREADS:
        realized = build_realized_r_frame(
            candidates, swings, events, frame, spread, args.horizon
        )
        development, _, _ = split_historical_boundary(
            realized, frame, args.horizon, HISTORICAL_AUDIT_CUTOFF
        )
        context = _context_frame(
            development, candidates, swings, events, structural_features
        )
        audit = pd.concat(
            [development.reset_index(drop=True), context.reset_index(drop=True)], axis=1
        )
        realized_by_spread[spread] = audit

    print("=== Structural-Context Stability Audit ===")
    print()
    print("Predefined design:")
    print(f"  Development boundary: {HISTORICAL_AUDIT_CUTOFF}")
    print("  Periods: P1 2024-09..2025-03, P2 2025-03..2025-09, P3 2025-09..2026-03-18")
    print("  Spreads: 0.00, 0.16, 0.30, 0.50 price units")
    print("  Horizon:", args.horizon)
    print("  No threshold search, feature invention, or best-bucket selection.")
    print()

    print("Baseline mean R by period and spread:")
    for period, start, end in PERIODS:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        print(f"  {period} ({start} to {end}):")
        for spread in SPREADS:
            part = realized_by_spread[spread]
            ts = pd.to_datetime(part["setup_timestamp"])
            part = part[(ts >= start_ts) & (ts < end_ts)]
            print(f"    spread={spread:.2f}: n={len(part):4d} mean={_mean(part):+.4f}")
    print()

    for period, start, end in PERIODS:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        print(f"Context stability — {period} ({start} to {end}):")
        for spread in SPREADS:
            base = realized_by_spread[spread]
            ts = pd.to_datetime(base["setup_timestamp"])
            audit = base[(ts >= start_ts) & (ts < end_ts)]
            print(f"  spread={spread:.2f}, n={len(audit):4d}")
            print("    pre_structure_direction:")
            _direction_summary(audit, "pre_structure_direction")
            print("    numeric quartile means:")
            for column in NUMERIC_CONTEXT:
                _quartile_gap(audit, column)
        print()

    print("Interpretation guardrails:")
    print("  - Time slices and spreads were fixed before inspection.")
    print("  - Quartiles remain descriptive within each slice; they are not thresholds.")
    print("  - Stability means the direction of a context contrast persists across periods/costs.")
    print("  - A changing sign or isolated slice is evidence of instability, not a tuning target.")
    print("  - Results are development-partition evidence, not pristine OOS.")
    print()
    print("=== Audit complete ===")


if __name__ == "__main__":
    main()
