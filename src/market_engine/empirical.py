"""Empirical outcome summaries for market-entry research.

This layer describes observed outcome distributions for explicitly supplied
context groups. It does not rank contexts, score setups, or predict future
outcomes.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


OUTCOME_COLUMNS = ("TP_FIRST", "SL_FIRST", "UNRESOLVED", "BOTH_SAME_CANDLE")


def summarize_outcomes(
    dataset: pd.DataFrame,
    group_by: Iterable[str],
    *,
    label_column: str = "label",
) -> pd.DataFrame:
    """Compute observed outcome counts and rates for explicit context groups."""
    group_columns = tuple(group_by)
    required = {*group_columns, label_column}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError("dataset is missing columns: " + ", ".join(missing))

    if not group_columns:
        raise ValueError("group_by must contain at least one column.")

    frame = dataset.loc[dataset[label_column].notna(), [*group_columns, label_column]].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                *group_columns,
                "sample_count",
                "tp_first_count",
                "sl_first_count",
                "unresolved_count",
                "both_same_candle_count",
                "tp_first_rate",
                "sl_first_rate",
                "unresolved_rate",
                "both_same_candle_rate",
            ]
        )

    grouped = frame.groupby(list(group_columns), dropna=False, sort=False)[label_column]
    rows: list[dict[str, object]] = []

    for key, labels in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        counts = labels.value_counts()
        sample_count = int(len(labels))
        row = dict(zip(group_columns, key))
        row.update(
            {
                "sample_count": sample_count,
                "tp_first_count": int(counts.get("TP_FIRST", 0)),
                "sl_first_count": int(counts.get("SL_FIRST", 0)),
                "unresolved_count": int(counts.get("UNRESOLVED", 0)),
                "both_same_candle_count": int(counts.get("BOTH_SAME_CANDLE", 0)),
            }
        )

        for name in ("tp_first", "sl_first", "unresolved", "both_same_candle"):
            row[f"{name}_rate"] = row[f"{name}_count"] / sample_count

        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=[
            *group_columns,
            "sample_count",
            "tp_first_count",
            "sl_first_count",
            "unresolved_count",
            "both_same_candle_count",
            "tp_first_rate",
            "sl_first_rate",
            "unresolved_rate",
            "both_same_candle_rate",
        ],
    )
