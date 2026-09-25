"""Time-series validation for the empirical Market Journey probability baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.data import load_mt5_csv
from run_market_journey_context_research import build_context

HORIZON = 40
STATES = ("ENTRY", "SWING_UPDATE", "CONTINUATION", "TRANSITION")
N_FOLDS = 5


def add_period(frame: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    boundary = pd.Timestamp("2026-03-18")
    context = context.copy()
    context["confirmation_timestamp"] = context["confirmation_index"].map(
        lambda i: frame.iloc[int(i)]["timestamp"]
    )
    context["period"] = context["confirmation_timestamp"].map(
        lambda ts: "development" if pd.Timestamp(ts) < boundary else "historical_oos"
    )
    return context


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
            "data/research/xauusd_m30_market_journey_probability_time_series_validation.csv"
        ),
    )
    parser.add_argument("--min-n", type=int, default=20)
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    context = add_period(frame, build_context(frame))
    episodes = pd.read_csv(args.episodes)

    episode = episodes[
        (episodes["horizon"] == HORIZON)
        & episodes["state"].isin(STATES)
        & episodes["next_state"].notna()
    ].copy()
    # The episode artifact already carries its own period label. Keep the
    # context-derived period under a distinct name to avoid pandas suffixes.
    episode = episode.merge(
        context[["candidate_id", "bos_scope", "period"]].rename(
            columns={"period": "context_period"}
        ),
        on="candidate_id",
        how="inner",
    )
    episode = episode[episode["context_period"] == "historical_oos"].copy()

    # Chronological candidate folds; all episodes from a candidate stay together.
    candidates = (
        episode[["candidate_id"]]
        .drop_duplicates()
        .sort_values("candidate_id")
        .reset_index(drop=True)
    )
    candidates["fold"] = pd.qcut(
        candidates.index,
        q=N_FOLDS,
        labels=False,
        duplicates="drop",
    ) + 1
    episode = episode.merge(candidates, on="candidate_id", how="left")

    rows: list[dict] = []
    group_cols = ["fold", "direction", "bos_scope", "state"]

    for keys, group in episode.groupby(group_cols):
        fold, direction, bos_scope, state = keys
        denominator = len(group)
        if denominator < args.min_n:
            continue

        counts = group["next_state"].value_counts()
        for next_state in STATES + ("INVALIDATED",):
            n = int(counts.get(next_state, 0))
            rows.append(
                {
                    "fold": int(fold),
                    "direction": direction,
                    "bos_scope": bos_scope,
                    "state": state,
                    "next_state": next_state,
                    "n": n,
                    "denominator": denominator,
                    "probability": n / denominator,
                }
            )

    report = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)

    print("=== Market Journey Probability Time-Series Validation ===")
    print("Historical OOS only | H40 | chronological 5 folds")
    print(f"Minimum fold sample: {args.min_n}")
    print(f"Candidates: {len(candidates)}")
    print(f"Episode rows: {len(episode)}")
    print(f"Validation rows: {len(report)}")
    print(f"Artifact: {args.output}")
    print()

    if not report.empty:
        summary = (
            report.groupby(["direction", "bos_scope", "state", "next_state"])
            .agg(
                folds=("fold", "nunique"),
                n_total=("n", "sum"),
                probability_mean=("probability", "mean"),
                probability_std=("probability", "std"),
            )
            .reset_index()
        )
        summary = summary[summary["folds"] >= 3]
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
