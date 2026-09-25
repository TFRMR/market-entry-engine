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
from run_market_journey_episode import build_candidates, excursion

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
    candidates, _events = build_candidates(frame)

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
    lookup = episode[key + ["state", "end_bar"]].rename(
        columns={"state": "next2_state", "end_bar": "next2_end_bar"}
    )
    lookup["episode_index"] -= 2

    chain = episode.merge(lookup, on=key, how="left")
    chain = chain[chain["next_state"].notna()].copy()

    # Only chains with an observed following state are evaluated as a full
    # two-step chain. This avoids pretending that censored chains are outcomes.
    chain = chain[chain["next2_state"].notna()].copy()
    candidate_lookup = candidates.reset_index().rename(columns={"index": "candidate_id"})[
        ["candidate_id", "entry_index", "entry", "risk", "direction"]
    ]
    chain = chain.merge(candidate_lookup, on=["candidate_id", "direction"], how="inner")

    chain["chain_start_index"] = (
        chain["entry_index"] + chain["start_bar"].astype(int)
    )
    chain["chain_end_index"] = (
        chain["entry_index"] + chain["next2_end_bar"].astype(int)
    )

    cumulative = []
    for row in chain.itertuples(index=False):
        mfe_r, mae_r = excursion(
            frame,
            int(row.chain_start_index),
            int(row.chain_end_index),
            row.direction,
            float(row.entry),
            float(row.risk),
        )
        cumulative.append((mfe_r, mae_r))

    chain["chain_mfe_r"] = [item[0] for item in cumulative]
    chain["chain_mae_r"] = [item[1] for item in cumulative]
    chain["chain_duration_bars"] = (
        chain["chain_end_index"] - chain["chain_start_index"]
    )
    chain["chain_hit_1r"] = (chain["chain_mfe_r"] >= 1.0).astype(float)
    chain["chain_hit_2r"] = (chain["chain_mfe_r"] >= 2.0).astype(float)

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
                "duration_median": group["chain_duration_bars"].median(),
                "duration_p25": group["chain_duration_bars"].quantile(0.25),
                "duration_p75": group["chain_duration_bars"].quantile(0.75),
                "mfe_r_median": group["chain_mfe_r"].median(),
                "mfe_r_p75": group["chain_mfe_r"].quantile(0.75),
                "mae_r_median": group["chain_mae_r"].median(),
                "mae_r_p25": group["chain_mae_r"].quantile(0.25),
                "p_hit_1r": group["chain_hit_1r"].mean(),
                "p_hit_2r": group["chain_hit_2r"].mean(),
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
