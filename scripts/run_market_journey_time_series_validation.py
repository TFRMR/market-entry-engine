"""Validate Market Journey empirical probabilities across chronological OOS folds.

Historical OOS candidates are split into five chronological sub-folds inside
the OOS period itself. This is a descriptive stability check only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HORIZON = 40
FOLDS = 5
STATES = ("ENTRY", "SWING_UPDATE", "CONTINUATION", "TRANSITION")


def assign_oos_folds(frame: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    oos = frame[frame["period"] == "historical_oos"]
    ids = sorted(oos["candidate_id"].dropna().unique())
    if len(ids) < n_folds:
        raise ValueError(f"Need at least {n_folds} OOS candidates, got {len(ids)}")

    fold_map = {
        candidate_id: min((i * n_folds) // len(ids), n_folds - 1)
        for i, candidate_id in enumerate(ids)
    }
    out = frame.copy()
    out["oos_fold"] = out["candidate_id"].map(fold_map)
    return out


def transition_rows(subset: pd.DataFrame) -> list[dict]:
    observed = subset[subset["next_state"].notna()]
    rows = []
    for state in STATES:
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
            })
    return rows


def build_validation(frame: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    frame = assign_oos_folds(frame, n_folds)
    rows = []
    for direction in ("UP", "DOWN"):
        for fold in range(n_folds):
            subset = frame[
                (frame["period"] == "historical_oos")
                & (frame["direction"] == direction)
                & (frame["horizon"] == HORIZON)
                & (frame["oos_fold"] == fold)
            ]
            if subset.empty:
                continue
            base = {
                "period": "historical_oos",
                "direction": direction,
                "horizon": HORIZON,
                "fold": fold + 1,
                "candidate_count": subset["candidate_id"].nunique(),
            }
            rows.extend({**base, **row} for row in transition_rows(subset))
    return pd.DataFrame(rows)


def add_summary(validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (direction, state, outcome), group in validation.groupby(
        ["direction", "state", "outcome"]
    ):
        rows.append({
            "direction": direction,
            "state": state,
            "outcome": outcome,
            "folds_observed": len(group),
            "n_total": group["n"].sum(),
            "probability_mean": group["probability"].mean(),
            "probability_std": group["probability"].std(ddof=0),
            "probability_min": group["probability"].min(),
            "probability_max": group["probability"].max(),
            "probability_range": group["probability"].max() - group["probability"].min(),
        })
    return pd.DataFrame(rows)


def print_report(validation: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("=== Market Journey Chronological Stability Validation ===")
    print("Historical OOS only; H40; 5 chronological folds within OOS")
    print()
    for direction in ("UP", "DOWN"):
        print(f"--- {direction} ---")
        subset = validation[validation["direction"] == direction]
        if not subset.empty:
            print(subset[
                ["fold", "candidate_count", "state", "outcome",
                 "n", "denominator", "probability"]
            ].to_string(index=False))
        print()
    print("--- stability summary ---")
    if not summary.empty:
        print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes_csv", type=Path)
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
        "candidate_id", "direction", "period", "horizon", "state",
        "next_state", "bars_to_next_state",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    validation = build_validation(frame, args.folds)
    summary = add_summary(validation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(args.output, index=False)

    print_report(validation, summary)
    print()
    print("Episode rows:", len(frame))
    print("Validation rows:", len(validation))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
