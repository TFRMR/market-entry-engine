"""Deterministic point-in-time market context snapshots.

A context snapshot is a factual representation of what the engine knew at a
setup candle. It does not rank setups, emit a signal, or assign probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from market_engine.entry import SetupCandidate
from market_engine.features import build_structural_features

CONTEXT_COLUMNS: Final[tuple[str, ...]] = (
    "structure_direction",
    "structure_last_event_type",
    "structure_last_event_direction",
    "structure_last_event_scope",
    "structure_previous_event_type",
    "structure_previous_event_direction",
    "structure_previous_event_scope",
    "trend_regime",
    "trend_transition",
    "trend_transition_direction",
    "range_state",
    "range_position_zone",
    "liquidity_sweep",
    "liquidity_sweep_direction",
    "pullback_direction",
    "pullback_active",
    "pullback_depth",
    "pullback_bars",
    "pullback_retracement_ratio",
    "local_sr_support",
    "local_sr_resistance",
    "local_sr_support_state",
    "local_sr_resistance_state",
    "d1_structure_direction",
    "d1_sr_support",
    "d1_sr_resistance",
    "d1_sr_support_state",
    "d1_sr_resistance_state",
    "poi_fvg_present",
    "poi_fvg_age_bars",
    "poi_fvg_distance",
    "poi_fvg_interaction",
    "poi_fvg_lifecycle",
    "poi_fvg_direction",
    "poi_ob_present",
    "poi_ob_age_bars",
    "poi_ob_distance",
    "poi_ob_interaction",
    "poi_ob_lifecycle",
    "poi_ob_direction",
    "poi_obim_present",
    "poi_obim_age_bars",
    "poi_obim_distance",
    "poi_obim_interaction",
    "poi_obim_lifecycle",
    "poi_obim_direction",
    "poi_liquidity_present",
    "poi_liquidity_age_bars",
    "poi_liquidity_distance",
    "poi_liquidity_interaction",
    "poi_liquidity_lifecycle",
    "poi_liquidity_direction",
    "poi_sr_present",
    "poi_sr_age_bars",
    "poi_sr_distance",
    "poi_sr_interaction",
    "poi_sr_lifecycle",
    "poi_sr_direction",
)


@dataclass(frozen=True)
class ContextSnapshot:
    """Facts available at one setup candle."""

    setup_index: int
    setup_timestamp: object
    direction: str
    facts: dict[str, object]


def build_context_snapshots(
    frame: pd.DataFrame,
    candidates: list[SetupCandidate],
    feature_frame: pd.DataFrame | None = None,
) -> list[ContextSnapshot]:
    """Build one point-in-time context snapshot for each setup candidate."""
    features = feature_frame if feature_frame is not None else build_structural_features(frame)

    if len(features) != len(frame):
        raise ValueError("feature_frame must have the same length as frame.")

    missing = sorted(set(CONTEXT_COLUMNS) - set(features.columns))
    if missing:
        raise ValueError(
            "feature_frame is missing context columns: " + ", ".join(missing)
        )

    snapshots: list[ContextSnapshot] = []
    for candidate in candidates:
        row = features.iloc[candidate.setup_index]
        facts = {column: row[column] for column in CONTEXT_COLUMNS}
        snapshots.append(
            ContextSnapshot(
                setup_index=candidate.setup_index,
                setup_timestamp=candidate.setup_timestamp,
                direction=candidate.direction.value,
                facts=facts,
            )
        )

    return snapshots


def build_context_dataset(
    frame: pd.DataFrame,
    candidates: list[SetupCandidate],
    feature_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a flat context dataset without outcome labels or scoring."""
    snapshots = build_context_snapshots(frame, candidates, feature_frame)
    rows = []

    for snapshot in snapshots:
        row = {
            "setup_index": snapshot.setup_index,
            "setup_timestamp": snapshot.setup_timestamp,
            "direction": snapshot.direction,
        }
        row.update(snapshot.facts)
        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=["setup_index", "setup_timestamp", "direction", *CONTEXT_COLUMNS],
    )
