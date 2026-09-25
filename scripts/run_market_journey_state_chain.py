"""Empirical Market Journey state-chain research.

Descriptive research only. No signal, ranking, score, or winner is produced.
Builds one-step and two-step conditional state transitions from H40 episodes.
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
        default=Path("data/research/xauusd_m30_market_journey_state_chain.csv"),
    )
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    episodes = pd.read_csv(args.episodes)
    context = build_context(frame)

    boundary = pd.Timestamp("2026-03-18")
    context["period"] = context["confirmation_index"].map(
        lambda i: "development"
        if pd.Timestamp(frame.iloc[int(i)]["timestamp"]) < boundary
        else "historical_oos"
    )

    episode = episodes[
        (episodes["horizon"] == HORIZON)
        & episodes["state"].isin(STATES)
    ].copy()

    episode = episode.merge(
        context[["candidate_id", "period", "bos_scope"]],
        on=["candidate_id", "period"],
        how="inner",
    )

    episode["episode_index"] = episode["episode_index"].astype(int)

    key = ["candidate_id", "period", "episode_index"]
    next_key = ["candidate_id", "period", "episode_index"]
    lookup = episode[
        key + ["state"]
    ].rename(columns={"state": "next2_state"})
    # Current episode i -> following episode i+1 is already represented by
    # next_state. For the second transition, join state(i+2).
    lookup["episode_index"] = lookup["episode_index"] - 2

    chain = episode.merge(lookup, on=next_key, how="left")
    chain = chain[chain["next_state"].notna()].copy()

    rows: list[dict] = []

    for (period, direction, bos_scope, state), group in chain.groupby(
        ["period", "direction", "bos_scope", "state"]
    ):
        denominator = len(group)

        for next_state, target in group.groupby("next_state"):
            rows.append({
                "period": period,
                "direction": direction,
                "bos_scope": bos_scope,
                "horizon": HORIZON,
                "current_state": state,
                "next_state": next_state,
                "step": 1,
                "n": len(target),
                "denominator": denominator,
                "probability": len(target) / denominator,
            })

            second = target[target["next2_state"].notna()]
            if not second.empty:
                for next2_state, target2 in second.groupby("next2_state"):
                    rows.append({
                        "period": period,
                        "direction": direction,
                        "bos_scope": bos_scope,
                        "horizon": HORIZON,
                        "current_state": f"{state} -> {next_state}",
                        "next_state": next2_state,
                        "step": 2,
                        "n": len(target2),
                        "denominator": len(second),
                        "probability": len(target2) / len(second),
                    })

    report = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)

    print("=== Market Journey Empirical State Chain ===")
    print("H40 | conditioning: direction + BOS scope")
    print("Step 1: current state -> next state")
    print("Step 2: current state -> next state -> following state")
    print()

    if report.empty:
        print("No report rows.")
    else:
        for (period, direction, bos_scope), group in report.groupby(
            ["period", "direction", "bos_scope"]
        ):
            print(f"{period} {direction} {bos_scope}")
            print(group.to_string(index=False))
            print()

    print("Context candidates:", len(context))
    print("Episode rows used:", len(chain))
    print("Report rows:", len(report))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
