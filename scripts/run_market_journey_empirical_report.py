"""Build a compact empirical Market Journey probability report.

Consumes the episode dataset produced by run_market_journey_episode.py and
summarizes state transitions, terminal outcomes, excursion distributions, and
time-to-next-state for development vs historical OOS.

This is descriptive empirical research. No signal, ranking, or winner is
produced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HORIZONS = (10, 20, 40, 80)
STATES = ("ENTRY", "SWING_UPDATE", "CONTINUATION", "TRANSITION", "INVALIDATED")


def summarize_transition_rows(subset: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    observed = subset[subset["next_state"].notna()].copy()
    if observed.empty:
        return rows

    for (state, next_state), group in observed.groupby(["state", "next_state"]):
        denominator = observed.loc[
            observed["state"] == state, "next_state"
        ].notna().sum()
        rows.append({
            "section": "NEXT_STATE",
            "state": state,
            "outcome": next_state,
            "n": len(group),
            "denominator": int(denominator),
            "probability": len(group) / denominator if denominator else None,
            "duration_median": group["bars_to_next_state"].median(),
            "duration_p25": group["bars_to_next_state"].quantile(0.25),
            "duration_p75": group["bars_to_next_state"].quantile(0.75),
            "mfe_r_median": group["mfe_r_during_episode"].median(),
            "mae_r_median": group["mae_r_during_episode"].median(),
        })
    return rows


def summarize_terminal_rows(subset: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    terminal = subset[subset["next_state"].isna()].copy()
    if terminal.empty:
        return rows

    for (state, reason), group in terminal.groupby(["state", "terminal_reason"]):
        denominator = terminal.loc[
            terminal["state"] == state, "terminal_reason"
        ].notna().sum()
        rows.append({
            "section": "TERMINAL",
            "state": state,
            "outcome": reason,
            "n": len(group),
            "denominator": int(denominator),
            "probability": len(group) / denominator if denominator else None,
            "duration_median": group["duration_bars"].median(),
            "duration_p25": group["duration_bars"].quantile(0.25),
            "duration_p75": group["duration_bars"].quantile(0.75),
            "mfe_r_median": group["mfe_r_during_episode"].median(),
            "mae_r_median": group["mae_r_during_episode"].median(),
        })
    return rows


def summarize_state_outcomes(subset: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for state, group in subset.groupby("state"):
        rows.append({
            "section": "STATE_OUTCOME",
            "state": state,
            "outcome": None,
            "n": len(group),
            "denominator": len(group),
            "probability": None,
            "duration_median": group["duration_bars"].median(),
            "duration_p25": group["duration_bars"].quantile(0.25),
            "duration_p75": group["duration_bars"].quantile(0.75),
            "mfe_r_median": group["mfe_r_during_episode"].median(),
            "mae_r_median": group["mae_r_during_episode"].median(),
            "p_hit_1r": group["hit_1r_during_episode"].mean(),
            "p_hit_2r": group["hit_2r_during_episode"].mean(),
        })
    return rows


def build_report(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for period in ("development", "historical_oos"):
        for direction in ("UP", "DOWN"):
            for horizon in HORIZONS:
                subset = frame[
                    (frame["period"] == period)
                    & (frame["direction"] == direction)
                    & (frame["horizon"] == horizon)
                ]
                if subset.empty:
                    continue

                base = {
                    "period": period,
                    "direction": direction,
                    "horizon": horizon,
                }

                for row in summarize_state_outcomes(subset):
                    rows.append({**base, **row})

                for row in summarize_transition_rows(subset):
                    rows.append({**base, **row})

                for row in summarize_terminal_rows(subset):
                    rows.append({**base, **row})

    return pd.DataFrame(rows)


def print_report(report: pd.DataFrame) -> None:
    print("=== Market Journey Empirical Probability Report ===")
    print("Development vs historical OOS; state transitions and outcome distributions")
    print()

    for period in ("development", "historical_oos"):
        for direction in ("UP", "DOWN"):
            for horizon in HORIZONS:
                subset = report[
                    (report["period"] == period)
                    & (report["direction"] == direction)
                    & (report["horizon"] == horizon)
                ]
                if subset.empty:
                    continue

                print(f"--- {period} {direction} H{horizon} ---")

                state = subset[subset["section"] == "STATE_OUTCOME"][
                    [
                        "state", "n", "duration_median", "mfe_r_median",
                        "mae_r_median", "p_hit_1r", "p_hit_2r",
                    ]
                ]
                print(state.to_string(index=False))

                transitions = subset[subset["section"] == "NEXT_STATE"][
                    [
                        "state", "outcome", "n", "denominator",
                        "probability", "duration_median",
                    ]
                ]
                if not transitions.empty:
                    print("next_state:")
                    print(transitions.to_string(index=False))

                terminal = subset[subset["section"] == "TERMINAL"][
                    ["state", "outcome", "n", "denominator", "probability"]
                ]
                if not terminal.empty:
                    print("terminal:")
                    print(terminal.to_string(index=False))
                print()


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
            "data/research/xauusd_m30_market_journey_empirical_report.csv"
        ),
    )
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
        "mfe_r_during_episode",
        "mae_r_during_episode",
        "hit_1r_during_episode",
        "hit_2r_during_episode",
        "terminal_reason",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    report = build_report(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)

    print_report(report)
    print("Episode rows:", len(frame))
    print("Report rows:", len(report))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
