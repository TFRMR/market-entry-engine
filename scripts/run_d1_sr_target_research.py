"""Research D1 structural S/R as natural target and invalidation.

This is a research-layer experiment. It does not change the deterministic setup
contract and does not emit a trading signal.

For each confirmed-BOS setup:
- entry = next-candle open;
- D1 resistance above entry is the LONG target;
- D1 support below entry is the LONG invalidation;
- D1 support below entry is the SHORT target;
- D1 resistance above entry is the SHORT invalidation;
- levels are frozen from the setup-time feature snapshot;
- the first future touch determines the observed path;
- same-candle target/invalidation is retained as ambiguous.

No fixed TP/SL, ATR multiple, score, or probability threshold is introduced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.entry import build_setup_candidates
from market_engine.features import build_structural_features
from market_engine.structure import Direction


def classify_path(
    frame: pd.DataFrame,
    *,
    entry_index: int,
    direction: Direction,
    target: float,
    invalidation: float,
) -> dict[str, object]:
    for index in range(entry_index, len(frame)):
        candle = frame.iloc[index]

        if direction is Direction.UP:
            hit_target = float(candle["high"]) >= target
            hit_invalidation = float(candle["low"]) <= invalidation
        else:
            hit_target = float(candle["low"]) <= target
            hit_invalidation = float(candle["high"]) >= invalidation

        if hit_target and hit_invalidation:
            return {
                "outcome": "BOTH_SAME_CANDLE",
                "exit_index": index,
                "bars_to_outcome": index - entry_index,
            }
        if hit_target:
            return {
                "outcome": "TARGET_FIRST",
                "exit_index": index,
                "bars_to_outcome": index - entry_index,
            }
        if hit_invalidation:
            return {
                "outcome": "INVALIDATION_FIRST",
                "exit_index": index,
                "bars_to_outcome": index - entry_index,
            }

    return {
        "outcome": "UNRESOLVED",
        "exit_index": None,
        "bars_to_outcome": None,
    }


def run(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    features = build_structural_features(frame)
    candidates = build_setup_candidates(frame)

    rows: list[dict[str, object]] = []

    for candidate_id, candidate in enumerate(candidates):
        setup_row = features.iloc[candidate.setup_index]
        entry_index = candidate.entry_index
        entry_price = float(frame.iloc[entry_index]["open"])

        d1_support = setup_row.get("d1_sr_support")
        d1_resistance = setup_row.get("d1_sr_resistance")

        if pd.isna(d1_support) or pd.isna(d1_resistance):
            continue

        d1_support = float(d1_support)
        d1_resistance = float(d1_resistance)

        if candidate.direction is Direction.UP:
            target = d1_resistance
            invalidation = d1_support
            direction = "UP"
        else:
            target = d1_support
            invalidation = d1_resistance
            direction = "DOWN"

        if not (target > entry_price and invalidation < entry_price):
            continue

        risk = abs(entry_price - invalidation)
        reward = abs(target - entry_price)
        if risk <= 0:
            continue

        path = classify_path(
            frame,
            entry_index=entry_index,
            direction=candidate.direction,
            target=target,
            invalidation=invalidation,
        )

        rows.append(
            {
                "candidate_id": candidate_id,
                "setup_index": candidate.setup_index,
                "setup_timestamp": candidate.setup_timestamp,
                "entry_index": entry_index,
                "entry_timestamp": candidate.entry_timestamp,
                "direction": direction,
                "entry_price": entry_price,
                "d1_target": target,
                "d1_invalidation": invalidation,
                "risk": risk,
                "reward": reward,
                "reward_risk": reward / risk,
                "d1_structure_direction": setup_row.get("d1_structure_direction"),
                "d1_target_age": (
                    setup_row.get("d1_sr_resistance_age")
                    if direction == "UP"
                    else setup_row.get("d1_sr_support_age")
                ),
                "d1_invalidation_age": (
                    setup_row.get("d1_sr_support_age")
                    if direction == "UP"
                    else setup_row.get("d1_sr_resistance_age")
                ),
                **path,
            }
        )

    return pd.DataFrame(rows)


def summarize(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()

    rows = []
    for (period, direction), group in dataset.groupby(["period", "direction"], sort=False):
        n = len(group)
        counts = group["outcome"].value_counts()
        target_rate = counts.get("TARGET_FIRST", 0) / n
        invalidation_rate = counts.get("INVALIDATION_FIRST", 0) / n
        ambiguous_rate = counts.get("BOTH_SAME_CANDLE", 0) / n
        unresolved_rate = counts.get("UNRESOLVED", 0) / n

        resolved = group[group["outcome"].isin(["TARGET_FIRST", "INVALIDATION_FIRST"])]
        mean_realized_r = None
        if not resolved.empty:
            realized = resolved.apply(
                lambda row: (
                    row["reward_risk"]
                    if row["outcome"] == "TARGET_FIRST"
                    else -1.0
                ),
                axis=1,
            )
            mean_realized_r = float(realized.mean())

        rows.append(
            {
                "period": period,
                "direction": direction,
                "n": n,
                "target_first": int(counts.get("TARGET_FIRST", 0)),
                "invalidation_first": int(counts.get("INVALIDATION_FIRST", 0)),
                "both_same_candle": int(counts.get("BOTH_SAME_CANDLE", 0)),
                "unresolved": int(counts.get("UNRESOLVED", 0)),
                "target_first_rate": target_rate,
                "invalidation_first_rate": invalidation_rate,
                "both_same_candle_rate": ambiguous_rate,
                "unresolved_rate": unresolved_rate,
                "mean_realized_r_resolved": mean_realized_r,
                "median_reward_risk": float(group["reward_risk"].median()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--historical-start",
        default="2026-03-18",
        help="First timestamp included in the historical/OOS period.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/research/xauusd_m30_d1_sr_target_research.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/research/xauusd_m30_d1_sr_target_summary.csv"),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    dataset = run(frame)

    historical_start = pd.Timestamp(args.historical_start)
    dataset["period"] = dataset["setup_timestamp"].map(
        lambda value: "historical_oos" if pd.Timestamp(value) >= historical_start else "development"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(args.output, index=False)
    summarize(dataset).to_csv(args.summary_output, index=False)

    print("=== D1 Structural S/R Target Research ===")
    print(f"Candidates with usable D1 target/invalidation: {len(dataset)}")
    if dataset.empty:
        return
    print()
    print(summarize(dataset).to_string(index=False))


if __name__ == "__main__":
    main()
