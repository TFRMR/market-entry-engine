"""Market Journey chain outcome research.

Descriptive research only. No signal, ranking, score, or winner is produced.
Summarizes empirical outcomes for observed two-step state chains.
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
        default=Path(
            "data/research/xauusd_m30_market_journey_chain_outcome.csv"
        ),
    )
    parser.add_argument("--min-n", type=int, default=20)
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
    lookup = episode[key + ["state"]].rename(columns={"state": "next2_state"})
    lookup["episode_index"] -= 2

    chain = episode.merge(lookup, on=key, how="left")
    chain = chain[chain["next_state"].notna()].copy()

    # Only chains with an observed following state are evaluated as a full
    # two-step chain. This avoids pretending that censored chains are outcomes.
    chain = chain[chain["next2_state"].notna()].copy()
    chain["chain"] = (
        chain["state"].astype(str)
        + " -> "
        + chain["next_state"].astype(str)
        + " -> "
        + chain["next2_state"].astype(str)
    )

    rows: list[dict] = []
    group_cols = [
        "period",
        "direction",
        "bos_scope",
        "state",
        "next_state",
        "next2_state",
    ]

    for keys, group in chain.groupby(group_cols):
        (
            period,
            direction,
            bos_scope,
            state,
            next_state,
            next2_state,
        ) = keys
        if len(group) < args.min_n:
            continue

        rows.append(
            {
                "period": period,
                "direction": direction,
                "bos_scope": bos_scope,
                "horizon": HORIZON,
                "chain": (
                    f"{state} -> {next_state} -> {next2_state}"
                ),
                "n": len(group),
                "duration_median": group["duration_bars"].median(),
                "duration_p25": group["duration_bars"].quantile(0.25),
                "duration_p75": group["duration_bars"].quantile(0.75),
                "mfe_r_median": group["mfe_r_during_episode"].median(),
                "mfe_r_p75": group["mfe_r_during_episode"].quantile(0.75),
                "mae_r_median": group["mae_r_during_episode"].median(),
                "mae_r_p25": group["mae_r_during_episode"].quantile(0.25),
                "p_hit_1r": group["hit_1r_during_episode"].mean(),
                "p_hit_2r": group["hit_2r_during_episode"].mean(),
            }
        )

    report = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)

    print("=== Market Journey Chain Outcome Research ===")
    print("H40 | observed two-step chains | direction + BOS scope")
    print(f"Minimum chain sample: {args.min_n}")
    print()
    if report.empty:
        print("No report rows.")
    else:
        for keys, group in report.groupby(
            ["period", "direction", "bos_scope"]
        ):
            print(*keys)
            print(group.to_string(index=False))
            print()

    print("Context candidates:", len(context))
    print("Observed two-step chain rows:", len(chain))
    print("Report rows:", len(report))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
