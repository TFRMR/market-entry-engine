"""Canonical labels and structural setup-label generation."""

from __future__ import annotations

from typing import Final

import pandas as pd

from market_engine.entry import SetupCandidate
from market_engine.execution import execute_entry
from market_engine.exits import ExitAreaIndex, build_exit_areas
from market_engine.outcome import TradeOutcome, evaluate_trade
from market_engine.setup_facts import SETUP_FACT_COLUMNS, SetupFactContext
from market_engine.structure import StructureEvent, ValidSwing

TP_FIRST: Final[str] = "TP_FIRST"
SL_FIRST: Final[str] = "SL_FIRST"
BOTH_SAME_CANDLE: Final[str] = "BOTH_SAME_CANDLE"
UNRESOLVED: Final[str] = "UNRESOLVED"

MODEL_LABELS: Final[frozenset[str]] = frozenset({TP_FIRST, SL_FIRST})
ALL_BARRIER_LABELS: Final[frozenset[str]] = frozenset(
    {TP_FIRST, SL_FIRST, BOTH_SAME_CANDLE, UNRESOLVED}
)

STRUCTURAL_LABEL_HORIZON: Final[int] = 10


def is_model_label(label: object) -> bool:
    """Return whether a barrier outcome is eligible as a binary model label."""
    return label in MODEL_LABELS


def label_from_trade_outcome(outcome: TradeOutcome) -> str:
    """Map a deterministic structural trade outcome to the canonical label."""
    if outcome.status == "TARGET":
        return TP_FIRST
    if outcome.status == "STOP":
        return SL_FIRST
    if outcome.status == "OPEN":
        return UNRESOLVED
    raise ValueError(f"Unsupported trade outcome status: {outcome.status!r}")


def build_setup_label_dataset(
    candidates: list[SetupCandidate],
    swings: list[ValidSwing],
    frame: pd.DataFrame,
    feature_frame: pd.DataFrame | None,
    spread_price: float,
    feature_columns: tuple[str, ...],
    horizon: int = STRUCTURAL_LABEL_HORIZON,
    events: list[StructureEvent] | None = None,
) -> pd.DataFrame:
    """Build setup-time features with labels from structural exit areas.

    Features are read from the setup candle only. Labels use the nearest valid
    structural exit area and deterministic execution/outcome semantics over a
    fixed forward horizon.
    """
    if horizon <= 0:
        raise ValueError("horizon must be greater than zero.")

    if feature_columns and feature_frame is None:
        raise ValueError("feature_frame is required when feature_columns are requested.")

    if feature_frame is not None:
        if len(feature_frame) != len(frame):
            raise ValueError("feature_frame must have the same length as frame.")

        missing = sorted(set(feature_columns) - set(feature_frame.columns))
        if missing:
            raise ValueError(
                f"feature_frame is missing required columns: {', '.join(missing)}"
            )

    rows: list[dict[str, object]] = []
    exit_index = ExitAreaIndex.build(frame)
    fact_context = (
        SetupFactContext.build(swings, events) if events is not None else None
    )

    for candidate in candidates:
        if candidate.setup_index + horizon >= len(frame):
            continue

        exit_areas = build_exit_areas(candidate, swings, frame, exit_index)
        if not exit_areas:
            continue

        execution = execute_entry(
            direction=candidate.direction,
            quoted_price=float(candidate.entry_price),
            spread_price=spread_price,
            invalidation_price=float(candidate.invalidation_price),
        )
        outcome = evaluate_trade(
            candidate=candidate,
            execution=execution,
            target=exit_areas[0],
            frame=frame,
            horizon=horizon,
        )

        row: dict[str, object] = {
            "setup_index": candidate.setup_index,
            "setup_timestamp": candidate.setup_timestamp,
            "direction": candidate.direction.value,
            "entry_index": candidate.entry_index,
            "entry_timestamp": candidate.entry_timestamp,
            "entry_price": execution.entry_price,
            "invalidation_price": execution.invalidation_price,
            "risk": execution.risk,
            "target_price": exit_areas[0].price,
            "reward_risk": (
                abs(float(exit_areas[0].price) - execution.entry_price) / execution.risk
            ),
            "ambiguous_barrier": outcome.both_hit,
            "label": label_from_trade_outcome(outcome),
        }

        if fact_context is not None:
            row.update(fact_context.facts(candidate))

        if feature_frame is not None:
            for column in feature_columns:
                row[column] = feature_frame.iloc[candidate.setup_index][column]

        rows.append(row)

    columns = [
        "setup_index",
        "setup_timestamp",
        "direction",
        "entry_index",
        "entry_timestamp",
        "entry_price",
        "invalidation_price",
        "risk",
        "target_price",
        "reward_risk",
        "ambiguous_barrier",
        "label",
        *(SETUP_FACT_COLUMNS if events is not None else ()),
        *feature_columns,
    ]
    return pd.DataFrame(rows, columns=columns)
