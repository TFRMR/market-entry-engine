"""Realized-R audit of structural setups (development period by default)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from market_engine.data import load_mt5_csv
from market_engine.entry import build_setup_candidates
from market_engine.holdout import (
    HISTORICAL_AUDIT_CUTOFF,
    historical_boundary_index,
    split_historical_boundary,
)
from market_engine.labels import STRUCTURAL_LABEL_HORIZON
from market_engine.realized import (
    build_realized_r_frame,
    cluster_bootstrap_mean,
    placebo_mean_r,
)
from market_engine.structure import build_structural_sequence, process_structural_candles


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--spread-price", type=float, required=True)
    parser.add_argument("--horizon", type=int, default=STRUCTURAL_LABEL_HORIZON)
    parser.add_argument("--draws", type=int, default=300, help="Placebo draws.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--include-historical-audit",
        action="store_true",
        help="Also include the historical-audit partition (not pristine OOS).",
    )
    return parser.parse_args()


def _summary(name: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        print(f"  {name:22s}: n=0")
        return
    print(
        f"  {name:22s}: n={len(frame):5d}  mean R={frame['realized_r'].mean():8.4f}  "
        f"win rate={(frame['realized_r'] > 0).mean() * 100:6.2f}%"
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
        candidates, swings, events, frame, args.spread_price, args.horizon
    )

    development, historical_audit, purged = split_historical_boundary(
        realized, frame, args.horizon, HISTORICAL_AUDIT_CUTOFF
    )
    boundary = historical_boundary_index(frame, HISTORICAL_AUDIT_CUTOFF)
    if args.include_historical_audit:
        selected = pd.concat([development, historical_audit], ignore_index=True)
        entry_high = len(frame) - args.horizon
    else:
        selected = development
        entry_high = boundary - args.horizon

    print("=== Realized-R Audit (after spread) ===")
    print()
    if args.include_historical_audit:
        print("!!! Includes the historical-audit partition (NOT pristine OOS). !!!")
        print()
    print("Partition:")
    print(f"  Boundary:                  {HISTORICAL_AUDIT_CUTOFF}")
    print(f"  Development rows:          {len(development):,}")
    print(f"  Historical-audit rows:     {len(historical_audit):,}")
    print(f"  Purged boundary rows:      {len(purged):,}")
    print(f"  Rows analysed:             {len(selected):,}")
    print(f"  Spread (price units):      {args.spread_price}")
    print(f"  Horizon (candles):         {args.horizon}")
    print()

    if selected.empty:
        print("No rows to analyse.")
        return

    print("Exit kinds:")
    kinds = selected["exit_kind"].value_counts()
    for kind in ("TARGET", "STOP", "TIME_STOP"):
        count = int(kinds.get(kind, 0))
        print(f"  {kind:10s}: {count:5d} ({count / len(selected) * 100:6.2f}%)")
    print()

    days = pd.to_datetime(selected["setup_timestamp"]).dt.date.to_numpy()
    mean, low, high = cluster_bootstrap_mean(
        selected["realized_r"].to_numpy(), days, seed=args.seed
    )
    print("Realized R:")
    print(f"  mean:   {mean:8.4f}   95% CI (bootstrap by day): [{low:.4f}, {high:.4f}]")
    print(f"  median: {selected['realized_r'].median():8.4f}")
    print(f"  total:  {selected['realized_r'].sum():8.2f} R over {len(selected):,} setups")
    print(f"  win rate (R > 0): {(selected['realized_r'] > 0).mean() * 100:.2f}%")
    print()

    targets = selected[selected["exit_kind"] == "TARGET"]
    inside_spread = int((targets["realized_r"] <= 0).sum())
    print("Target hits with non-positive realized R (target inside spread cost):")
    print(f"  {inside_spread:,} / {len(targets):,}")
    print()

    print("By exit kind:")
    for kind in ("TARGET", "STOP", "TIME_STOP"):
        _summary(kind, selected[selected["exit_kind"] == kind])
    print()
    print("By direction:")
    for direction in ("UP", "DOWN"):
        _summary(direction, selected[selected["direction"] == direction])
    print()
    print("By BOS scope:")
    _summary("EXTERNAL", selected[selected["setup_bos_external"] == 1.0])
    _summary("INTERNAL", selected[selected["setup_bos_external"] == 0.0])
    print()

    print("By reward/risk after spread:")
    bins = [-np.inf, 0.0, 0.25, 0.5, 1.0, np.inf]
    names = ["<= 0", "(0, 0.25]", "(0.25, 0.5]", "(0.5, 1]", "> 1"]
    buckets = pd.cut(selected["reward_r_after_spread"], bins=bins, labels=names)
    for name in names:
        part = selected[buckets == name]
        _summary(name, part)
    print()

    print("Minimum reward/risk after spread filter (EXPLORATORY: do not pick thresholds here):")
    for threshold in (0.0, 0.5, 0.75, 1.0):
        part = selected[selected["reward_r_after_spread"] >= threshold]
        _summary(f">= {threshold}", part)
    print()

    placebo = placebo_mean_r(
        frame=frame,
        is_up=(selected["direction"] == "UP").to_numpy(),
        risk_dist=selected["risk_dist"].to_numpy(float),
        reward_dist=selected["reward_dist"].to_numpy(float),
        spread_price=args.spread_price,
        horizon=args.horizon,
        entry_low=1,
        entry_high=entry_high,
        draws=args.draws,
        seed=args.seed,
    )
    print("Placebo (random entry time, same direction, stop and target distances):")
    print(f"  real mean R:        {mean:8.4f}")
    print(
        f"  placebo mean R:     {placebo.mean():8.4f}   "
        f"[p5 {np.percentile(placebo, 5):.4f}, p95 {np.percentile(placebo, 95):.4f}]"
    )
    share = float((placebo >= mean).mean())
    print(f"  share of placebo draws >= real mean: {share * 100:.1f}%")
    print()
    print("Notes:")
    print("  - realized_r includes entry spread only; slippage/fees and gap fills are not modelled.")
    print("  - reward_r_after_spread is a target-distance ratio, not a full net P&L metric.")
    print("  - Setups overlap in time; the day-clustered CI is still optimistic.")
    print("  - Placebo is a descriptive timing sanity check, not causal proof.")
    print("  - Not pristine OOS unless the true OOS stream is used. Not trading advice.")
    print()
    print("=== Audit complete ===")


if __name__ == "__main__":
    main()
