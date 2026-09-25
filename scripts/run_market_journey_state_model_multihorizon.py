"""Empirical Market Journey state model across multiple horizons.

Descriptive research only. No signal, ranking, score, or winner is produced.
Conditions on direction + BOS scope and summarizes state transitions for H10/H20/H40/H80.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.data import load_mt5_csv
from run_market_journey_context_research import build_context

HORIZONS = (10, 20, 40, 80)
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
        default=Path("data/research/xauusd_m30_market_journey_state_model_multihorizon.csv"),
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
        episodes["horizon"].isin(HORIZONS)
        & episodes["state"].isin(STATES)
        & episodes["next_state"].notna()
    ].copy()

    episode = episode.merge(
        context[["candidate_id", "period", "bos_scope"]],
        on=["candidate_id", "period"],
        how="inner",
    )

    rows: list[dict] = []

    for (period, direction, bos_scope, horizon, state), group in episode.groupby(
        ["period", "direction", "bos_scope", "horizon", "state"]
    ):
        denominator = len(group)
        if denominator == 0:
            continue

        for next_state, target in group.groupby("next_state"):
            rows.append(
                {
                    "period": period,
                    "direction": direction,
                    "bos_scope": bos_scope,
                    "horizon": int(horizon),
                    "state": state,
                    "next_state": next_state,
                    "n": len(target),
                    "denominator": denominator,
                    "probability": len(target) / denominator,
                    "duration_median": target["bars_to_next_state"].median(),
                    "duration_p25": target["bars_to_next_state"].quantile(0.25),
                    "duration_p75": target["bars_to_next_state"].quantile(0.75),
                    "mfe_r_median": target["mfe_r_during_episode"].median(),
                    "mae_r_median": target["mae_r_during_episode"].median(),
                    "p_hit_1r": target["hit_1r_during_episode"].mean(),
                    "p_hit_2r": target["hit_2r_during_episode"].mean(),
                }
            )

    report = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)

    print("=== Market Journey Empirical State Model ===")
    print("Horizons:", ", ".join(map(str, HORIZONS)))
    print("Conditioning: direction + BOS scope")
    print("States:", ", ".join(STATES))
    print()

    if report.empty:
        print("No report rows.")
    else:
        for (period, direction, bos_scope, horizon, state), group in report.groupby(
            ["period", "direction", "bos_scope", "horizon", "state"]
        ):
            print(f"{period} {direction} {bos_scope} H{horizon} {state}")
            print(
                group[
                    [
                        "next_state",
                        "n",
                        "denominator",
                        "probability",
                        "duration_median",
                        "mfe_r_median",
                        "mae_r_median",
                        "p_hit_1r",
                        "p_hit_2r",
                    ]
                ].to_string(index=False)
            )
            print()

    print("Context candidates:", len(context))
    print("Episode rows used:", len(episode))
    print("Report rows:", len(report))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
