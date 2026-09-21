"""Audit the structural setup-label dataset on a real MT5 export."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.data import load_mt5_csv
from market_engine.entry import build_setup_candidates
from market_engine.exits import build_exit_areas
from market_engine.features import build_features
from market_engine.labels import (
    STRUCTURAL_LABEL_HORIZON,
    build_setup_label_dataset,
)
from market_engine.structure import build_structural_sequence, process_structural_candles


FEATURE_COLUMNS = (
    "candle_range",
    "candle_body",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_position",
    "is_bullish",
    "is_bearish",
    "true_range",
    "atr_14",
    "range_to_atr",
    "ema_5",
    "ema_20",
    "ema_50",
    "price_vs_ema_5",
    "price_vs_ema_20",
    "price_vs_ema_50",
    "ema_5_vs_20",
    "ema_20_vs_50",
    "ema_alignment",
    "previous_high_20",
    "previous_low_20",
    "previous_high_50",
    "previous_low_50",
    "distance_to_high_20",
    "distance_to_low_20",
    "distance_to_high_50",
    "distance_to_low_50",
    "breakout_above_20",
    "breakout_below_20",
    "breakout_above_50",
    "breakout_below_50",
    "return_3",
    "return_6",
    "return_12",
    "volume_ratio_20",
    "volume_change_1",
    "high_vs_previous_high_1",
    "high_vs_previous_high_2",
    "high_vs_previous_high_3",
    "low_vs_previous_low_1",
    "low_vs_previous_low_2",
    "low_vs_previous_low_3",
    "range_vs_avg_3",
    "range_vs_avg_5",
    "structure_bullish_bos",
    "structure_bearish_bos",
    "structure_swing_high_valid",
    "structure_swing_low_valid",
    "structure_internal_bos",
    "structure_external_bos",
    "structure_internal_swing",
    "structure_external_swing",
    "structure_last_valid_high",
    "structure_last_valid_low",
    "structure_hh",
    "structure_hl",
    "structure_lh",
    "structure_ll",
    "structure_direction",
    "structure_distance_to_high",
    "structure_distance_to_low",
    "structure_bars_since_last_swing",
    "structure_bars_since_last_bos",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit structural setup labels on an MT5 CSV export."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument(
        "--spread-price",
        type=float,
        required=True,
        help="Execution spread in price units, not raw MT5 <SPREAD> points.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=STRUCTURAL_LABEL_HORIZON,
    )
    parser.add_argument(
        "--with-features",
        action="store_true",
        help="Also build the full feature frame and audit feature missingness.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    frame = load_mt5_csv(args.input_path)
    if args.horizon <= 0:
        raise ValueError("horizon must be greater than zero.")
    if args.spread_price < 0:
        raise ValueError("spread-price must not be negative.")

    candidates = build_setup_candidates(frame)
    structural = build_structural_sequence(frame)
    swings, _ = process_structural_candles(structural)
    feature_frame = build_features(frame) if args.with_features else None

    labeled = build_setup_label_dataset(
        candidates=candidates,
        swings=swings,
        frame=frame,
        feature_frame=feature_frame,
        spread_price=args.spread_price,
        feature_columns=FEATURE_COLUMNS if args.with_features else (),
        horizon=args.horizon,
    )

    candidate_indices = {candidate.setup_index for candidate in candidates}
    incomplete = {
        candidate.setup_index
        for candidate in candidates
        if candidate.setup_index + args.horizon >= len(frame)
    }
    no_exit_area = {
        candidate.setup_index
        for candidate in candidates
        if candidate.setup_index not in incomplete
        and not build_exit_areas(candidate, swings, frame)
    }
    labeled_indices = set(labeled["setup_index"].astype(int))

    excluded_other = candidate_indices - incomplete - no_exit_area - labeled_indices

    print("=== Structural Setup Label Audit ===")
    print()
    print("Candles:")
    print(f"  {len(frame):,}")
    print()
    print("Setup candidates:")
    print(f"  {len(candidates):,}")
    print()
    print("Label dataset:")
    print(f"  {len(labeled):,}")
    print(f"  Incomplete horizon: {len(incomplete):,}")
    print(f"  No valid exit area: {len(no_exit_area):,}")
    if excluded_other:
        print(f"  Other excluded:    {len(excluded_other):,}")
    print()
    print("Labels:")
    label_counts = labeled["label"].value_counts()
    for label in ("TP_FIRST", "SL_FIRST", "UNRESOLVED"):
        count = int(label_counts.get(label, 0))
        percentage = count / len(labeled) * 100 if len(labeled) else 0.0
        print(f"  {label:10s}: {count:5d} ({percentage:6.2f}%)")
    print()
    print("Direction:")
    direction_counts = labeled["direction"].value_counts()
    for direction in ("UP", "DOWN"):
        print(f"  {direction:10s}: {int(direction_counts.get(direction, 0)):5d}")
    print()
    print("Reward/Risk:")
    if labeled.empty:
        print("  No labeled setups.")
    else:
        rr = labeled["reward_risk"].astype(float)
        print(f"  median: {rr.median():.4f}")
        print(f"  p25:    {rr.quantile(0.25):.4f}")
        print(f"  p75:    {rr.quantile(0.75):.4f}")
    print()
    print("Ambiguous barrier:")
    ambiguous = int(labeled["ambiguous_barrier"].sum())
    percentage = ambiguous / len(labeled) * 100 if len(labeled) else 0.0
    print(f"  {ambiguous:,} / {len(labeled):,} ({percentage:.2f}%)")
    print()
    print("Chronology:")
    if labeled.empty:
        print("  No labeled setups.")
    else:
        print(f"  Start: {labeled['setup_timestamp'].min()}")
        print(f"  End:   {labeled['setup_timestamp'].max()}")
        print(
            f"  Duplicate setup candles: "
            f"{labeled['setup_index'].duplicated().sum():,}"
        )
    print()
    if args.with_features:
        print("Features:")
        missing_features = int(
            labeled[list(FEATURE_COLUMNS)].isna().any(axis=1).sum()
        )
        print(f"  Rows with missing feature values: {missing_features:,}")
        print()
    else:
        print("Features:")
        print("  Skipped (use --with-features for the full feature audit).")
        print()
    print("=== Audit complete ===")


if __name__ == "__main__":
    main()
