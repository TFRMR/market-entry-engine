"""Validate context-conditioned Market Journey probabilities across chronological OOS folds.

Historical OOS candidates are split into five chronological folds inside the
OOS period. Context dimensions are BOS scope and FVG relation at entry.
This is descriptive stability research only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_market_journey_context_research import build_context
from market_engine.data import load_mt5_csv

HORIZON = 40
FOLDS = 5
STATES = ("ENTRY", "SWING_UPDATE")


def assign_oos_folds(context: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    oos = context[context["period"] == "historical_oos"].copy()
    order = (
        oos[["candidate_id", "confirmation_index"]]
        .drop_duplicates()
        .sort_values(["confirmation_index", "candidate_id"])
    )
    ids = order["candidate_id"].tolist()
    if len(ids) < n_folds:
        raise ValueError(f"Need at least {n_folds} OOS candidates, got {len(ids)}")

    fold_map = {
        candidate_id: min((i * n_folds) // len(ids), n_folds - 1)
        for i, candidate_id in enumerate(ids)
    }
    out = context.copy()
    out["oos_fold"] = out["candidate_id"].map(fold_map)
    return out


def build_validation(
    episodes: pd.DataFrame,
    context: pd.DataFrame,
    n_folds: int,
) -> pd.DataFrame:
    context = assign_oos_folds(context, n_folds)

    episode = episodes[
        (episodes["horizon"] == HORIZON)
        & episodes["next_state"].notna()
        & episodes["state"].isin(STATES)
    ].copy()

    episode = episode.merge(
        context[
            [
                "candidate_id",
                "period",
                "direction",
                "bos_scope",
                "fvg_relation_at_entry",
                "oos_fold",
            ]
        ],
        on=["candidate_id", "period", "direction"],
        how="inner",
        suffixes=("", "_context"),
    )

    rows = []
    for direction in ("UP", "DOWN"):
        for dimension in ("bos_scope", "fvg_relation_at_entry"):
            for context_value in sorted(
                episode[dimension].dropna().unique()
            ):
                for fold in range(n_folds):
                    subset = episode[
                        (episode["period"] == "historical_oos")
                        & (episode["direction"] == direction)
                        & (episode[dimension] == context_value)
                        & (episode["oos_fold"] == fold)
                    ]
                    if subset.empty:
                        continue

                    for state in STATES:
                        current = subset[subset["state"] == state]
                        if current.empty:
                            continue
                        denominator = len(current)
                        for outcome, group in current.groupby("next_state"):
                            rows.append({
                                "direction": direction,
                                "dimension": dimension,
                                "context": context_value,
                                "fold": fold + 1,
                                "state": state,
                                "outcome": outcome,
                                "n": len(group),
                                "denominator": denominator,
                                "probability": len(group) / denominator,
                            })

    return pd.DataFrame(rows)


def add_summary(validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if validation.empty:
        return pd.DataFrame()

    for (direction, dimension, context, state, outcome), group in validation.groupby(
        ["direction", "dimension", "context", "state", "outcome"]
    ):
        rows.append({
            "direction": direction,
            "dimension": dimension,
            "context": context,
            "state": state,
            "outcome": outcome,
            "folds_observed": len(group),
            "n_total": group["n"].sum(),
            "probability_mean": group["probability"].mean(),
            "probability_std": group["probability"].std(ddof=0),
            "probability_min": group["probability"].min(),
            "probability_max": group["probability"].max(),
            "probability_range": (
                group["probability"].max() - group["probability"].min()
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path("data/research/xauusd_m30_market_journey_episode.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/research/xauusd_m30_market_journey_context_time_series_validation.csv"
        ),
    )
    parser.add_argument("--folds", type=int, default=FOLDS)
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    episodes = pd.read_csv(args.episodes)
    context = build_context(frame)

    boundary = pd.Timestamp("2026-03-18")
    context["confirmation_timestamp"] = context["confirmation_index"].map(
        lambda i: frame.iloc[int(i)]["timestamp"]
    )
    context["period"] = context["confirmation_timestamp"].map(
        lambda ts: "development"
        if pd.Timestamp(ts) < boundary
        else "historical_oos"
    )

    validation = build_validation(episodes, context, args.folds)
    summary = add_summary(validation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(args.output, index=False)

    print("=== Context-Conditioned Market Journey Stability Validation ===")
    print("Historical OOS only; H40; 5 chronological folds within OOS")
    print("Contexts: BOS scope + FVG relation at entry")
    print()

    if validation.empty:
        print("No validation rows.")
    else:
        print("--- fold results ---")
        print(validation.to_string(index=False))
        print()
        print("--- stability summary ---")
        print(summary.to_string(index=False))

    print()
    print("Context candidates:", len(context))
    print(
        "Unique OOS candidates:",
        context.loc[context["period"] == "historical_oos", "candidate_id"].nunique(),
    )
    print("Episode rows:", len(episodes))
    print("Validation rows:", len(validation))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
