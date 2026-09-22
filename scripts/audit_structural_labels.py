"""Audit the structural setup-label dataset on a real MT5 export."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.data import load_mt5_csv
from market_engine.entry import build_setup_candidates
from market_engine.features import build_structural_features
from market_engine.holdout import (
    HISTORICAL_AUDIT_CUTOFF,
    historical_boundary_index,
    split_historical_boundary,
)
from market_engine.labels import (
    PRE_SETUP_FEATURE_COLUMNS,
    STRUCTURAL_LABEL_HORIZON,
    build_setup_label_dataset,
)
from market_engine.setup_facts import SETUP_FACT_COLUMNS
from market_engine.structure import (
    Direction,
    SwingType,
    build_structural_sequence,
    process_structural_candles,
    process_structural_candles_with_context,
)


STRUCTURAL_FEATURE_COLUMNS = (
    # Structure event flags
    "structure_bullish_bos",
    "structure_bearish_bos",
    "structure_bullish_choch",
    "structure_bearish_choch",
    "structure_swing_high_valid",
    "structure_swing_low_valid",
    "structure_internal_bos",
    "structure_external_bos",
    "structure_internal_swing",
    "structure_external_swing",

    # Confirmed structure state
    "structure_last_valid_high",
    "structure_last_valid_low",
    "structure_hh",
    "structure_hl",
    "structure_lh",
    "structure_ll",

    # Event sequence
    "structure_event_count",
    "structure_last_event_index",
    "structure_last_event_age",
    "structure_last_event_type",
    "structure_last_event_direction",
    "structure_last_event_scope",
    "structure_previous_event_index",
    "structure_previous_event_age",
    "structure_previous_event_type",
    "structure_previous_event_direction",
    "structure_previous_event_scope",
    "structure_events_since_last_bos",
    "structure_events_since_last_swing",
    "structure_bars_since_last_bos",
    "structure_bars_since_last_swing",

    # Active structure
    "structure_direction",
    "structure_distance_to_high",
    "structure_distance_to_low",
    "structure_historical_distance_to_high",
    "structure_historical_distance_to_low",

    # Trend / range
    "trend_regime",
    "trend_transition",
    "trend_transition_direction",
    "range_state",
    "range_high",
    "range_low",
    "range_width",
    "range_position",
    "range_position_zone",

    # Liquidity
    "liquidity_high",
    "liquidity_low",
    "liquidity_high_present",
    "liquidity_low_present",
    "distance_to_liquidity_high",
    "distance_to_liquidity_low",
    "liquidity_high_sweep",
    "liquidity_low_sweep",
    "liquidity_sweep",
    "liquidity_sweep_direction",
    "liquidity_sweep_size",

    # Structural S/R + lifecycle
    "local_sr_support", "local_sr_resistance",
    "local_sr_support_present", "local_sr_resistance_present",
    "local_sr_support_age", "local_sr_resistance_age",
    "local_sr_support_swing_type", "local_sr_resistance_swing_type",
    "local_sr_support_label", "local_sr_resistance_label",
    "local_sr_support_scope", "local_sr_resistance_scope",
    "support_level", "resistance_level",
    "distance_to_support", "distance_to_resistance",
    "distance_to_next_structure_level", "leg_position",
    "local_sr_support_state", "local_sr_support_event",
    "local_sr_resistance_state", "local_sr_resistance_event",
    "d1_structure_direction", "d1_sr_support", "d1_sr_resistance",
    "d1_sr_support_present", "d1_sr_resistance_present",
    "d1_sr_support_age", "d1_sr_resistance_age",
    "d1_sr_support_label", "d1_sr_resistance_label",
    "d1_sr_support_scope", "d1_sr_resistance_scope",
    "distance_to_d1_support", "distance_to_d1_resistance",
    "d1_sr_support_state", "d1_sr_support_event",
    "d1_sr_resistance_state", "d1_sr_resistance_event",

    # Order block
    "ob_bullish_present",
    "ob_bullish_size",
    "ob_bullish_age_bars",
    "ob_bullish_distance",
    "ob_bullish_contains_price",
    "ob_bullish_relative_position",
    "ob_bearish_present",
    "ob_bearish_size",
    "ob_bearish_age_bars",
    "ob_bearish_distance",
    "ob_bearish_contains_price",
    "ob_bearish_relative_position",
)


def _audit_sr_point_in_time(frame: pd.DataFrame) -> None:
    """Independently verify S/R lifecycle and completed-D1 point-in-time mapping."""
    violations = 0

    for prefix in ("local", "d1"):
        for side in ("support", "resistance"):
            level = frame[f"{prefix}_sr_{side}"]
            state = frame[f"{prefix}_sr_{side}_state"]
            event = frame[f"{prefix}_sr_{side}_event"]
            violations += int((state.notna() & level.isna()).sum())
            violations += int((event.ne("NONE") & level.isna()).sum())

    ts = pd.to_datetime(frame["timestamp"])
    daily = (
        frame.assign(_date=ts.dt.floor("D"))
        .groupby("_date", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )
        .reset_index()
    )

    if not daily.empty:
        structural = build_structural_sequence(daily)
        swings, _ = process_structural_candles(structural)
        _, _, snapshots = process_structural_candles_with_context(structural)
        by_confirmation: dict[int, list] = {}
        for swing in swings:
            by_confirmation.setdefault(swing.confirmation_index, []).append(swing)

        highs = []
        lows = []
        expected = []
        snapshot_by_index = {s.index: s for s in snapshots}

        for position, row in daily.iterrows():
            for swing in by_confirmation.get(position, []):
                (highs if swing.swing_type is SwingType.HIGH else lows).append(swing)

            snapshot = snapshot_by_index.get(position)
            direction = 0
            if snapshot is not None and snapshot.direction is not None:
                direction = 1 if snapshot.direction is Direction.UP else -1

            close = float(row["close"])
            supports = [s for s in lows if s.price <= close]
            resistances = [s for s in highs if s.price >= close]
            support = max(supports, key=lambda s: (s.price, s.confirmation_index)) if supports else None
            resistance = min(resistances, key=lambda s: (s.price, -s.confirmation_index)) if resistances else None

            expected.append({
                "_date": row["_date"],
                "direction": direction,
                "support": support.price if support else float("nan"),
                "resistance": resistance.price if resistance else float("nan"),
            })

        expected = pd.DataFrame(expected)
        expected["available_date"] = expected["_date"].shift(-1)
        expected = expected.dropna(subset=["available_date"])

        actual = frame.copy()
        actual["_date"] = ts.dt.floor("D")
        actual = actual.merge(expected, left_on="_date", right_on="available_date", how="left")

        for actual_col, expected_col in (
            ("d1_structure_direction", "direction"),
            ("d1_sr_support", "support"),
            ("d1_sr_resistance", "resistance"),
        ):
            a = actual[actual_col].to_numpy(dtype=float)
            e = actual[expected_col].to_numpy(dtype=float)
            mismatch = ~(
                (pd.isna(a) & pd.isna(e))
                | (pd.notna(a) & pd.notna(e) & (abs(a - e) <= 1e-12))
            )
            violations += int(mismatch.sum())

        first_date = actual["_date"].min()
        if first_date in set(expected["available_date"]):
            pass
        else:
            first_rows = actual["_date"].eq(first_date)
            if actual.loc[first_rows, "d1_sr_support"].notna().any() or actual.loc[first_rows, "d1_sr_resistance"].notna().any():
                violations += 1

    print("S/R point-in-time + lifecycle:")
    print(f"  Violations: {violations:,}")
    if violations:
        raise AssertionError("S/R point-in-time/lifecycle audit failed.")
    print("  Validation: PASS")

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
        "--include-historical-audit",
        action="store_true",
        help="Also include the historical-audit partition; it is not pristine OOS.",
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

    # Point-in-time structural chronology audit.
    chronology_violations: list[tuple[int, str, int, int]] = []
    for snapshot in snapshots:
        setup_index = snapshot.index

        if (
            snapshot.last_bos_index is not None
            and snapshot.last_bos_index > setup_index
        ):
            chronology_violations.append(
                (
                    setup_index,
                    "last_bos_index",
                    snapshot.last_bos_index,
                    setup_index,
                )
            )

        for name, swing in (
            ("last_high", snapshot.last_high),
            ("last_low", snapshot.last_low),
        ):
            if swing is None:
                continue
            if swing.confirmation_index > setup_index:
                chronology_violations.append(
                    (
                        setup_index,
                        f"{name}.confirmation_index",
                        swing.confirmation_index,
                        setup_index,
                    )
                )

    print("Point-in-time structural chronology:")
    print(f"  Snapshot chronology violations: {len(chronology_violations):,}")
    if chronology_violations:
        for violation in chronology_violations[:10]:
            setup_index, field, value, current_index = violation
            print(
                f"    setup={setup_index:,} "
                f"{field}={value:,} > current={current_index:,}"
            )
        raise AssertionError(
            "Structural snapshot contains future-confirmed state."
        )
    print("  Validation: PASS")

    feature_chronology_violations: list[tuple[int, str, int, int]] = []

    for setup_index in sorted(snapshots_by_index):
        snapshot = snapshots_by_index[setup_index]

        checks = (
            ("last_bos_index", snapshot.last_bos_index),
            (
                "last_swing_confirmation_index",
                snapshot.last_swing_confirmation_index,
            ),
            (
                "last_high.confirmation_index",
                snapshot.last_high.confirmation_index
                if snapshot.last_high is not None
                else None,
            ),
            (
                "last_low.confirmation_index",
                snapshot.last_low.confirmation_index
                if snapshot.last_low is not None
                else None,
            ),
            (
                "historical_last_high.confirmation_index",
                snapshot.historical_last_high.confirmation_index
                if snapshot.historical_last_high is not None
                else None,
            ),
            (
                "historical_last_low.confirmation_index",
                snapshot.historical_last_low.confirmation_index
                if snapshot.historical_last_low is not None
                else None,
            ),
        )

        for field, value in checks:
            if value is not None and value > setup_index:
                feature_chronology_violations.append(
                    (setup_index, field, value, setup_index)
                )

    print("Feature source chronology:")
    print(
        "  Structural feature chronology violations: "
        f"{len(feature_chronology_violations):,}"
    )
    if feature_chronology_violations:
        for violation in feature_chronology_violations[:10]:
            setup_index, field, value, current_index = violation
            print(
                f"    setup={setup_index:,} "
                f"{field}={value:,} > current={current_index:,}"
            )
        raise AssertionError(
            "Feature source contains future-confirmed structural state."
        )
    print("  Validation: PASS")

    feature_frame = None
    feature_columns: tuple[str, ...] = ()
    if args.with_features:
        feature_frame = build_structural_features(frame)
        feature_columns = STRUCTURAL_FEATURE_COLUMNS
        _audit_sr_point_in_time(feature_frame)

    labeled = build_setup_label_dataset(
        candidates=candidates,
        swings=swings,
        frame=frame,
        feature_frame=feature_frame,
        spread_price=args.spread_price,
        feature_columns=feature_columns,
        horizon=args.horizon,
        events=events,
        pre_feature_columns=PRE_SETUP_FEATURE_COLUMNS if args.with_features else (),
    )


    development, historical_audit, purged_boundary = split_historical_boundary(
        labeled, frame, args.horizon, HISTORICAL_AUDIT_CUTOFF
    )
    if args.include_historical_audit:
        labeled = pd.concat([development, historical_audit], ignore_index=True)
    else:
        labeled = development

    boundary_index = historical_boundary_index(frame, HISTORICAL_AUDIT_CUTOFF)
    if args.include_historical_audit:
        selected_candidates = [
            candidate
            for candidate in candidates
            if (
                candidate.setup_index + args.horizon < boundary_index
                or candidate.setup_index >= boundary_index
            )
        ]
    else:
        selected_candidates = [
            candidate
            for candidate in candidates
            if candidate.setup_index + args.horizon < boundary_index
        ]

    candidate_indices = {candidate.setup_index for candidate in selected_candidates}
    incomplete = {
        candidate.setup_index
        for candidate in selected_candidates
        if candidate.setup_index + args.horizon >= len(frame)
    }
    labeled_indices = set(labeled["setup_index"].astype(int))
    no_exit_area = candidate_indices - incomplete - labeled_indices
    excluded_other = candidate_indices - incomplete - no_exit_area - labeled_indices

    print("=== Structural Setup Label Audit ===")
    print()
    print("Chronological boundary policy:")
    print(f"  Historical audit boundary: {HISTORICAL_AUDIT_CUTOFF}")
    print("  Historical-audit rows are NOT pristine OOS; the full dataset was previously inspected.")
    print(f"  Development rows:          {len(development):,}")
    print(f"  Historical-audit rows:     {len(historical_audit):,}")
    print(f"  Purged boundary rows:      {len(purged_boundary):,}")
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

            active_missing_mask = structural_missing.any(axis=1)
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
        print("  Skipped (use --with-features to audit structural features).")
        print()
    print("Missingness semantics:")

    high_distance_missing = labeled["structure_distance_to_high"].isna()
    low_distance_missing = labeled["structure_distance_to_low"].isna()

    no_confirmed_swing = high_distance_missing & low_distance_missing
    print(
        "  Both structure distances missing with no confirmed swing: "
        f"{int(no_confirmed_swing.sum()):,}"
    )

    partial_structure_distance_missing = (
        high_distance_missing ^ low_distance_missing
    )
    print(
        "  Partial structure-distance missingness: "
        f"{int(partial_structure_distance_missing.sum()):,}"
    )

    sweep_missing = labeled["liquidity_sweep_size"].isna()
    no_sweep = labeled["liquidity_sweep"].fillna(False) == False
    unexpected_sweep_missing = sweep_missing & ~no_sweep
    print(
        "  liquidity_sweep_size missing without no-sweep state: "
        f"{int(unexpected_sweep_missing.sum()):,}"
    )

    transition_missing = labeled["trend_transition_direction"].isna()
    no_transition = labeled["trend_transition"].fillna(False) == False
    unexpected_transition_missing = transition_missing & ~no_transition
    print(
        "  trend_transition_direction missing without no-transition state: "
        f"{int(unexpected_transition_missing.sum()):,}"
    )

    bars_since_swing_missing = labeled["structure_bars_since_last_swing"].isna()
    print(
        "  bars_since_last_swing missing: "
        f"{int(bars_since_swing_missing.sum()):,}"
    )
    print(
        "  bars_since_last_swing missing while either structural distance exists: "
        f"{int((bars_since_swing_missing & ~no_confirmed_swing).sum()):,}"
    )

    for direction in ("bullish", "bearish"):
        present = labeled[f"ob_{direction}_present"].fillna(False)
        attributes = labeled[
            [
                f"ob_{direction}_size",
                f"ob_{direction}_age_bars",
                f"ob_{direction}_distance",
                f"ob_{direction}_contains_price",
                f"ob_{direction}_relative_position",
            ]
        ]
        attribute_missing_without_presence = (
            attributes.isna().any(axis=1) & present
        )
        print(
            f"  {direction} OB attributes missing while present: "
            f"{int(attribute_missing_without_presence.sum()):,}"
        )

    unexpected = (
        int(unexpected_sweep_missing.sum())
        + int(unexpected_transition_missing.sum())
        + int((bars_since_swing_missing & ~no_confirmed_swing).sum())
    )

    if unexpected:
        raise AssertionError(
            f"Unexpected feature missingness detected: {unexpected}"
        )

    print("  Validation: PASS")
    print()

    if args.with_features:
        print("Feature dataset schema audit:")

        feature_frame = labeled[list(STRUCTURAL_FEATURE_COLUMNS)]

        # 1. Dtype inventory
        dtype_counts = feature_frame.dtypes.astype(str).value_counts()
        print("  Dtypes:")
        for dtype, count in dtype_counts.items():
            print(f"    {dtype:<12}: {int(count):,}")

        # 2. Feature availability
        availability = 1.0 - feature_frame.isna().mean()
        print(
            "  Feature availability:"
            f" min={availability.min():.2%}"
            f" median={availability.median():.2%}"
            f" max={availability.max():.2%}"
        )

        # 3. Constant features
        constant_features = [
            column
            for column in feature_frame.columns
            if feature_frame[column].nunique(dropna=False) <= 1
        ]

        print(
            "  Constant features: "
            f"{len(constant_features):,}"
        )
        for column in constant_features:
            print(f"    {column}")

        # 4. Near-constant features
        near_constant_features: list[tuple[str, float, object]] = []

        for column in feature_frame.columns:
            counts = feature_frame[column].value_counts(dropna=False)
            if counts.empty:
                continue

            dominant_value = counts.index[0]
            dominant_share = counts.iloc[0] / len(feature_frame)

            if (
                dominant_share >= 0.995
                and feature_frame[column].nunique(dropna=False) > 1
            ):
                near_constant_features.append(
                    (column, dominant_share, dominant_value)
                )

        print(
            "  Near-constant features (>=99.5% one value): "
            f"{len(near_constant_features):,}"
        )
        for column, share, value in near_constant_features:
            print(
                f"    {column}: "
                f"{share:.2%} dominant={value!r}"
            )

        # 5. Exact duplicate feature columns
        duplicate_feature_pairs: list[tuple[str, str]] = []

        columns = list(feature_frame.columns)
        for i, left in enumerate(columns):
            for right in columns[i + 1:]:
                if feature_frame[left].equals(feature_frame[right]):
                    duplicate_feature_pairs.append((left, right))

        print(
            "  Exact duplicate feature pairs: "
            f"{len(duplicate_feature_pairs):,}"
        )
        for left, right in duplicate_feature_pairs:
            print(f"    {left} == {right}")

        # 6. Semantic dtype groups
        categorical_features = [
            column
            for column in feature_frame.columns
            if feature_frame[column].dtype == "object"
        ]

        boolean_features = [
            column
            for column in feature_frame.columns
            if str(feature_frame[column].dtype) == "bool"
        ]

        numeric_features = [
            column
            for column in feature_frame.columns
            if column not in categorical_features
            and column not in boolean_features
        ]

        print(
            "  Feature groups:"
            f" categorical={len(categorical_features):,}"
            f" boolean={len(boolean_features):,}"
            f" numeric={len(numeric_features):,}"
        )

        print("  Categorical features:")
        for column in categorical_features:
            print(f"    {column}")

        print("  Boolean features:")
        for column in boolean_features:
            print(f"    {column}")

        print("  Validation: PASS")
        print()

    print("=== Audit complete ===")


if __name__ == "__main__":
    main()
