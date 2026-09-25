"""Validate Market Journey empirical probabilities across chronological folds.

This is a descriptive time-series stability check. It does not train a model,
rank states, or produce trading signals. Candidate IDs are treated as
chronological order because the episode dataset preserves source order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HORIZON = 40
STATES = ("ENTRY", "SWING_UPDATE", "CONTINUATION", "TRANSITION", "INVALIDATED")
FOLDS = 5


def assign_folds(frame: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    ids = sorted(frame["candidate_id"].dropna().unique())
    if len(ids) < n_folds:
        raise ValueError(f"Need at least {n_folds} candidates, got {len(ids)}")

    fold_map = {}
    for i, candidate_id in enumerate(ids):
        fold_map[candidate_id] = min((i * n_folds) // len(ids), n_folds - 1)

    out = frame.copy()
    out["fold"] = out["candidate_id"].map(fold_map).astype(int) + 1
    return out


def transition_rows(subset: pd.DataFrame) -> list[dict]:
    observed = subset[subset["next_state"].notna()].copy()
    rows: list[dict] = []

    for state in ("ENTRY", "SWING_UPDATE", "CONTINUATION", "TRANSITION"):
        state_rows = observed[observed["state"] == state]
        denominator = len(state_rows)
        if denominator == 0:
            continue

        for outcome, group in state_rows.groupby("next_state"):
            rows.append({
                "section": "NEXT_STATE",
                "state": state,
                "outcome": outcome,
                "n": len(group),
                "denominator": denominator,
                "probability": len(group) / denominator,
                "duration_median": group["bars_to_next_state"].median(),
            })

    return rows


def state_rows(subset: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for state, group in subset.groupby("state"):
        rows.append({
            "section": "STATE_OUTCOME",
            "state": state,
            "outcome": None,
            "n": len(group),
            "p_hit_1r": group["hit_1r_during_episode"].mean(),
            "p_hit_2r": group["hit_2r_during_episode"].mean(),
            "duration_median": group["duration_bars"].median(),
        })
    return rows


def build_validation(frame: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    rows: list[dict] = []
    frame = assign_folds(frame, n_folds)

    for period in ("historical_oos",):
        for direction in ("UP", "DOWN"):
            for fold in range(1, n_folds + 1):
                subset = frame[
                    (frame["period"] == period)
                    & (frame["direction"] == direction)
                    & (frame["horizon"] == HORIZON)
                    & (frame["fold"] == fold)
                ]
                if subset.empty:
                    continue

                candidate_count = subset["candidate_id"].nunique()
                base = {
                    "period": period,
                    "direction": direction,
                    "horizon": HORIZON,
                    "fold": fold,
                    "candidate_count": candidate_count,
                }

                for row in transition_rows(subset):
                    rows.append({**base, **row})

                for row in state_rows(subset):
                    rows.append({**base, **row})

    return pd.DataFrame(rows)


def add_stability_summary(validation: pd.DataFrame) -> pd.DataFrame:
    transitions = validation[
        (validation["section"] == "NEXT_STATE")
        & validation["state"].isin(["ENTRY", "SWING_UPDATE"])
    ].copy()

    if transitions.empty:
        return pd.DataFrame()

    rows = []
    for (direction, state, outcome), group in transitions.groupby(
        ["direction", "state", "outcome"]
    ):
        rows.append({
            "direction": direction,
            "state": state,
            "outcome": outcome,
            "folds_observed": len(group),
            "probability_mean": group["probability"].mean(),
            "probability_std": group["probability"].std(ddof=0),
            "probability_min": group["probability"].min(),
            "probability_max": group["probability"].max(),
            "probability_range": group["probability"].max() - group["probability"].min(),
            "n_total": group["n"].sum(),
        })

    return pd.DataFrame(rows)


def print_validation(validation: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("=== Market Journey Chronological Stability Validation ===")
    print("Historical OOS only; H40; 5 chronological folds")
    print()

    for direction in ("UP", "DOWN"):
        print(f"--- {direction} ---")
        subset = validation[
            (validation["direction"] == direction)
            & (validation["section"] == "NEXT_STATE")
            & validation["state"].isin(["ENTRY", "SWING_UPDATE"])
        ]
        if subset.empty:
            continue
        cols = ["fold", "state", "outcome", "n", "denominator", "probability"]
        print(subset[cols].to_string(index=False))
        print()

    if not summary.empty:
        print("--- stability summary ---")
        print(
            summary[
                [
                    "direction", "state", "outcome", "folds_observed",
                    "probability_mean", "probability_std",
                    "probability_min", "probability_max",
                    "probability_range", "n_total",
                ]
            ].to_string(index=False)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "episodes_csv",
        type=Path,
        help="Episode CSV from run_market_journey_episode.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/research/xauusd_m30_market_journey_time_series_validation.csv"
        ),
    )
    parser.add_argument("--folds", type=int, default=FOLDS)
    args = parser.parse_args()

    frame = pd.read_csv(args.episodes_csv)
    required = {
        "candidate_id",
        "direction",
        "period",
        "horizon",
        "state",
        "next_state",
        "bars_to_next_state",
        "duration_bars",
        "hit_1r_during_episode",
        "hit_2r_during_episode",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    validation = build_validation(frame, args.folds)
    summary = add_stability_summary(validation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(args.output, index=False)

    print_validation(validation, summary)
    print()
    print("Episode rows:", len(frame))
    print("Validation rows:", len(validation))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
