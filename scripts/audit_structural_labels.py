"""Audit the structural setup-label dataset on a real MT5 export."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.data import load_mt5_csv
from market_engine.entry import build_setup_candidates
from market_engine.features import build_features, build_structural_features
from market_engine.labels import (
    STRUCTURAL_LABEL_HORIZON,
    build_setup_label_dataset,
)
from market_engine.setup_facts import SETUP_FACT_COLUMNS
from market_engine.structure import (
    build_structural_sequence,
    process_structural_candles,
    process_structural_candles_with_context,
)


ALL_FEATURE_COLUMNS = (
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
    "structure_historical_distance_to_high",
    "structure_historical_distance_to_low",
    "structure_bars_since_last_swing",
    "structure_bars_since_last_bos",
)

STRUCTURAL_FEATURE_COLUMNS = tuple(
    column for column in ALL_FEATURE_COLUMNS if column.startswith("structure_")
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
    parser.add_argument(
        "--legacy-features",
        action="store_true",
        help="With --with-features, audit the legacy momentum/EMA set as well (ablation only).",
    )
    parser.add_argument(
        "--inspect-structural-missing",
        action="store_true",
        help="Print concrete labeled setups whose active structural features are missing.",
    )
    parser.add_argument(
        "--inspect-count",
        type=int,
        default=10,
        help="Number of structural-missing setups to inspect (default: 10).",
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
    swings, events = process_structural_candles(structural)
    _, _, snapshots = process_structural_candles_with_context(structural)
    snapshots_by_index = {snapshot.index: snapshot for snapshot in snapshots}
    feature_frame = None
    feature_columns: tuple[str, ...] = ()
    if args.with_features:
        if args.legacy_features:
            feature_frame = build_features(frame)
            feature_columns = ALL_FEATURE_COLUMNS
        else:
            feature_frame = build_structural_features(frame)
            feature_columns = STRUCTURAL_FEATURE_COLUMNS

    labeled = build_setup_label_dataset(
        candidates=candidates,
        swings=swings,
        frame=frame,
        feature_frame=feature_frame,
        spread_price=args.spread_price,
        feature_columns=feature_columns,
        horizon=args.horizon,
        events=events,
    )

    candidate_indices = {candidate.setup_index for candidate in candidates}
    incomplete = {
        candidate.setup_index
        for candidate in candidates
        if candidate.setup_index + args.horizon >= len(frame)
    }
    eligible_indices = candidate_indices - incomplete
    labeled_indices = set(labeled["setup_index"].astype(int))
    no_exit_area = eligible_indices - labeled_indices
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
    if not labeled.empty:
        print("Setup facts (BOS event + swings confirmed before setup):")
        for column in SETUP_FACT_COLUMNS:
            values = labeled[column].astype(float)
            print(
                f"  {column:34s}: median {values.median():9.4f}  "
                f"missing {int(values.isna().sum()):5d}"
            )
        print("Setup facts × label:")
        fact_summary = labeled[["label", *SETUP_FACT_COLUMNS]].copy()
        for column in SETUP_FACT_COLUMNS:
            print(f"  {column}:")
            grouped = fact_summary.groupby("label")[column].agg(["count", "median", "mean"])
            for label in ("TP_FIRST", "SL_FIRST", "UNRESOLVED"):
                if label not in grouped.index:
                    continue
                row = grouped.loc[label]
                print(
                    f"    {label:10s}: n={int(row['count']):4d} "
                    f"median={row['median']:.4f} mean={row['mean']:.4f}"
                )
        print()
        external = labeled["setup_bos_external"].astype(float)
        print("BOS scope × label:")
        scope_table = pd.crosstab(
            external.map({1.0: "EXTERNAL", 0.0: "INTERNAL"}),
            labeled["label"],
        )
        for scope in ("EXTERNAL", "INTERNAL"):
            row = scope_table.loc[scope] if scope in scope_table.index else pd.Series(dtype=int)
            total = int(row.sum()) if not row.empty else 0
            parts = []
            for label in ("TP_FIRST", "SL_FIRST", "UNRESOLVED"):
                count = int(row.get(label, 0))
                pct = count / total * 100 if total else 0.0
                parts.append(f"{label}={count} ({pct:.2f}%)")
            print(f"  {scope:8s}: n={total:4d}  " + "  ".join(parts))
        print()
    if args.with_features:
        print("Features:")
        feature_values = labeled[list(feature_columns)]
        missing_mask = feature_values.isna().any(axis=1)
        missing_features = int(missing_mask.sum())
        print(f"  Rows with missing feature values: {missing_features:,}")
        if missing_features:
            print("  Missing by feature:")
            missing_by_feature = feature_values.isna().sum()
            for column, count in missing_by_feature[missing_by_feature.gt(0)].sort_values(
                ascending=False
            ).items():
                print(f"    {column:34s}: {int(count):5d}")

            structural_missing_columns = (
                "structure_distance_to_high",
                "structure_distance_to_low",
                "structure_bars_since_last_swing",
            )
            structural_missing = feature_values[list(
                structural_missing_columns
            )].isna()
            print("  Structural missingness profile:")
            print(
                "    distance_to_high missing: "
                f"{int(structural_missing['structure_distance_to_high'].sum()):,}"
            )
            print(
                "    distance_to_low missing:  "
                f"{int(structural_missing['structure_distance_to_low'].sum()):,}"
            )
            print(
                "    bars_since_swing missing: "
                f"{int(structural_missing['structure_bars_since_last_swing'].sum()):,}"
            )
            print(
                "    both distances missing:   "
                f"{int(structural_missing[['structure_distance_to_high', 'structure_distance_to_low']].all(axis=1).sum()):,}"
            )
            print(
                "    any structural missing:    "
                f"{int(structural_missing.any(axis=1).sum()):,}"
            )

            active_missing_mask = active_missing.any(axis=1)
            print("  Active-structure availability by label:")
            availability = pd.DataFrame(
                {
                    "label": labeled["label"],
                    "active_missing": active_missing_mask,
                }
            )
            for label in ("TP_FIRST", "SL_FIRST", "UNRESOLVED"):
                label_rows = availability[availability["label"] == label]
                if label_rows.empty:
                    continue
                missing_count = int(label_rows["active_missing"].sum())
                total_count = len(label_rows)
                print(
                    f"    {label:10s}: "
                    f"{missing_count:,}/{total_count:,} missing "
                    f"({missing_count / total_count * 100:.2f}%)"
                )

            missing_setup_indices = labeled.loc[
                missing_mask, "setup_index"
            ].astype(int)
            complete_mask = ~missing_mask
            if complete_mask.any():
                first_complete = int(
                    labeled.loc[complete_mask, "setup_index"].min()
                )
                last_complete = int(
                    labeled.loc[complete_mask, "setup_index"].max()
                )
                print(
                    "  Complete-feature setup range: "
                    f"{first_complete:,}..{last_complete:,}"
                )
                interior_missing = missing_setup_indices[
                    missing_setup_indices > first_complete
                ]
                print(
                    "  Missing rows after first complete setup: "
                    f"{len(interior_missing):,}"
                )
            print(
                "  Missing setup range: "
                f"{missing_setup_indices.min():,}..{missing_setup_indices.max():,}"
            )

            if args.inspect_structural_missing:
                structural_columns = (
                    "structure_distance_to_high",
                    "structure_distance_to_low",
                    "structure_bars_since_last_swing",
                )
                structural_missing_mask = feature_values[list(
                    structural_columns
                )].isna().any(axis=1)
                inspect_rows = labeled.loc[structural_missing_mask].head(
                    args.inspect_count
                )
                print()
                print("  Structural missing examples:")
                if inspect_rows.empty:
                    print("    None.")
                for _, row in inspect_rows.iterrows():
                    setup_index = int(row["setup_index"])
                    snapshot = snapshots_by_index.get(setup_index)
                    print(
                        f"    setup={setup_index:,} "
                        f"time={row['setup_timestamp']} "
                        f"direction={row['direction']} "
                        f"label={row['label']}"
                    )
                    if snapshot is None:
                        print("      snapshot: none")
                        continue
                    high = snapshot.last_high
                    low = snapshot.last_low
                    print(
                        "      snapshot: "
                        f"direction={snapshot.direction.value if snapshot.direction else None} "
                        f"last_bos={snapshot.last_bos_index} "
                        f"last_swing_confirmation={snapshot.last_swing_confirmation_index}"
                    )
                    print(
                        "      last_high: "
                        f"{high.price if high else None} "
                        f"(index={high.index if high else None}, "
                        f"confirmation={high.confirmation_index if high else None}, "
                        f"scope={high.scope.value if high else None})"
                    )
                    print(
                        "      last_low:  "
                        f"{low.price if low else None} "
                        f"(index={low.index if low else None}, "
                        f"confirmation={low.confirmation_index if low else None}, "
                        f"scope={low.scope.value if low else None})"
                    )
        print()
    else:
        print("Features:")
        print("  Skipped (use --with-features for the full feature audit).")
        print()
    print("=== Audit complete ===")


if __name__ == "__main__":
    main()
