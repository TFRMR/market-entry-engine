"""Market Journey empirical probability baseline.

Descriptive baseline: transition probabilities conditioned on direction,
BOS scope, and current state, plus observed two-step chain probabilities.
No ranking, score, or trading signal is produced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.data import load_mt5_csv
from run_market_journey_context_research import build_context

HORIZON = 40
STATES = ("ENTRY", "SWING_UPDATE", "CONTINUATION", "TRANSITION")


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
        default=Path("data/research/xauusd_m30_market_journey_probability_baseline.csv"),
    )
    parser.add_argument("--min-n", type=int, default=20)
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    context = build_context(frame)
    episodes = pd.read_csv(args.episodes)

    episode = episodes[
        (episodes["horizon"] == HORIZON)
        & episodes["state"].isin(STATES)
    ].copy()
    # Context is candidate-level; period is derived independently here from
    # the same confirmation boundary used by the journey dataset.
    boundary = pd.Timestamp("2026-03-18")
    context = context.copy()
    context["confirmation_timestamp"] = context["confirmation_index"].map(
        lambda i: frame.iloc[int(i)]["timestamp"]
    )
    context["period"] = context["confirmation_timestamp"].map(
        lambda ts: "development" if pd.Timestamp(ts) < boundary else "historical_oos"
    )
    episode = episode.merge(
        context[["candidate_id", "bos_scope"]],
        on="candidate_id",
        how="inner",
    )

    rows: list[dict] = []
    group_cols = ["period", "direction", "bos_scope", "state"]

    for keys, group in episode.groupby(group_cols):
        period, direction, bos_scope, state = keys
        denominator = group["next_state"].notna().sum()
        if denominator < args.min_n:
            continue

        observed = group[group["next_state"].notna()]
        counts = observed["next_state"].value_counts()

        for next_state in STATES + ("INVALIDATED",):
            n = int(counts.get(next_state, 0))
            rows.append(
                {
                    "kind": "next_state",
                    "period": period,
                    "direction": direction,
                    "bos_scope": bos_scope,
                    "horizon": HORIZON,
                    "state": state,
                    "next_state": next_state,
                    "chain": "",
                    "n": n,
                    "denominator": int(denominator),
                    "probability": n / denominator,
                }
            )

    # Two-step empirical chain probability, conditioned on the current state.
    key = ["candidate_id", "period", "episode_index"]
    lookup = episode[key + ["state"]].rename(columns={"state": "next2_state"})
    lookup["episode_index"] -= 2
    chain = episode.merge(lookup, on=key, how="left")
    chain = chain[chain["next_state"].notna() & chain["next2_state"].notna()].copy()
    chain["chain"] = (
        chain["state"].astype(str)
        + " -> "
        + chain["next_state"].astype(str)
        + " -> "
        + chain["next2_state"].astype(str)
    )

    for keys, group in chain.groupby(
        ["period", "direction", "bos_scope", "state", "chain"]
    ):
        period, direction, bos_scope, state, chain_name = keys
        denominator = episode[
            (episode["period"] == period)
            & (episode["direction"] == direction)
            & (episode["bos_scope"] == bos_scope)
            & (episode["state"] == state)
            & episode["next_state"].notna()
        ].shape[0]
        n = len(group)
        if denominator < args.min_n or n < args.min_n:
            continue
        rows.append(
            {
                "kind": "two_step_chain",
                "period": period,
                "direction": direction,
                "bos_scope": bos_scope,
                "horizon": HORIZON,
                "state": state,
                "next_state": group["next_state"].iloc[0],
                "chain": chain_name,
                "n": n,
                "denominator": denominator,
                "probability": n / denominator,
            }
        )

    report = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)

    print("=== Market Journey Empirical Probability Baseline ===")
    print("H40 | direction + BOS scope + state")
    print(f"Minimum group sample: {args.min_n}")
    print()
    for keys, group in report.groupby(
        ["period", "direction", "bos_scope"]
    ):
        print(*keys)
        print(group.to_string(index=False))
        print()

    print("Context candidates:", len(context))
    print("Episode rows used:", len(episode))
    print("Report rows:", len(report))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
