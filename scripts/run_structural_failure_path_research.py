"""Map continuation vs structural failure after a structure-first entry.

Scenario:
BOS -> next same-direction valid swing -> 50% retracement entry.

Track the post-entry path until:
- +1R / +2R / +3R / +4R is reached,
- the confirmed swing invalidation is breached,
- or the observation horizon expires.

This is descriptive research, not a trading rule.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from market_engine.data import load_mt5_csv
from market_engine.structure import (
    Direction,
    SwingType,
    build_structural_sequence,
    process_structural_candles,
)

HORIZONS = (10, 20, 40, 80)
RETRACE = 0.50
MFE_LEVELS = (1.0, 2.0, 3.0, 4.0)


def build_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    structural = build_structural_sequence(frame)
    swings, events = process_structural_candles(structural)
    rows = []
    used = set()

    for event in events:
        if event.event not in {"BULLISH_BOS", "BEARISH_BOS"}:
            continue

        direction = Direction.UP if event.event == "BULLISH_BOS" else Direction.DOWN
        future = [
            swing for swing in swings
            if swing.confirmation_index > event.index
            and (
                (direction is Direction.UP and swing.swing_type is SwingType.LOW)
                or (direction is Direction.DOWN and swing.swing_type is SwingType.HIGH)
            )
        ]
        if not future:
            continue

        swing = min(future, key=lambda item: item.confirmation_index)
        cidx = swing.confirmation_index
        key = (event.index, cidx)
        if key in used:
            continue
        used.add(key)

        confirmation = float(frame.iloc[cidx]["close"])
        invalidation = float(swing.price)
        leg = (
            confirmation - invalidation
            if direction is Direction.UP
            else invalidation - confirmation
        )
        if leg <= 0:
            continue

        entry = (
            confirmation - RETRACE * leg
            if direction is Direction.UP
            else confirmation + RETRACE * leg
        )
        entry_index = None
        for index in range(cidx + 1, len(frame)):
            candle = frame.iloc[index]
            if direction is Direction.UP and float(candle["low"]) <= entry:
                entry_index = index
                break
            if direction is Direction.DOWN and float(candle["high"]) >= entry:
                entry_index = index
                break

        if entry_index is None:
            continue

        risk = abs(entry - invalidation)
        if risk <= 0:
            continue

        rows.append({
            "bos_index": event.index,
            "bos_timestamp": frame.iloc[event.index]["timestamp"],
            "bos_event": event.event,
            "direction": direction.value,
            "confirmation_index": cidx,
            "confirmation_timestamp": frame.iloc[cidx]["timestamp"],
            "entry_index": entry_index,
            "entry_timestamp": frame.iloc[entry_index]["timestamp"],
            "entry": entry,
            "invalidation": invalidation,
            "risk": risk,
            "entry_wait_bars": entry_index - cidx,
        })

    return pd.DataFrame(rows)


def trace(row: pd.Series, frame: pd.DataFrame, horizon: int) -> dict:
    entry_index = int(row["entry_index"])
    entry = float(row["entry"])
    invalidation = float(row["invalidation"])
    risk = float(row["risk"])
    end = min(entry_index + horizon, len(frame) - 1)

    highs = frame.iloc[entry_index + 1:end + 1]["high"].astype(float).to_numpy()
    lows = frame.iloc[entry_index + 1:end + 1]["low"].astype(float).to_numpy()
    if len(highs) == 0:
        return {}

    if row["direction"] == "UP":
        favorable = highs - entry
        adverse = entry - lows
        failure_hit = lows <= invalidation
    else:
        favorable = entry - lows
        adverse = highs - entry
        failure_hit = highs >= invalidation

    failure_idx = np.flatnonzero(failure_hit)
    failure_bar = int(failure_idx[0] + 1) if len(failure_idx) else np.nan

    out = {
        "mfe_r": float(np.max(favorable) / risk),
        "mae_r": float(np.max(adverse) / risk),
        "failure": bool(len(failure_idx)),
        "failure_bar": failure_bar,
    }

    for level in MFE_LEVELS:
        hit = np.flatnonzero(favorable >= level * risk)
        out[f"hit_{level:g}R"] = bool(len(hit))
        out[f"bars_to_{level:g}R"] = int(hit[0] + 1) if len(hit) else np.nan

    if len(failure_idx):
        fi = int(failure_idx[0])
        out["mfe_before_failure_r"] = float(np.max(favorable[:fi + 1]) / risk)
        out["mae_before_failure_r"] = float(np.max(adverse[:fi + 1]) / risk)
    else:
        out["mfe_before_failure_r"] = np.nan
        out["mae_before_failure_r"] = np.nan

    reached_1 = bool(np.any(favorable >= risk))
    reached_2 = bool(np.any(favorable >= 2 * risk))

    if reached_1 and len(failure_idx):
        first_1 = int(np.flatnonzero(favorable >= risk)[0])
        out["failure_after_1R"] = bool(failure_idx[0] > first_1)
    else:
        out["failure_after_1R"] = False

    if reached_2 and len(failure_idx):
        first_2 = int(np.flatnonzero(favorable >= 2 * risk)[0])
        out["failure_after_2R"] = bool(failure_idx[0] > first_2)
    else:
        out["failure_after_2R"] = False

    if len(failure_idx):
        out["path_state"] = "FAILURE"
    elif reached_2:
        out["path_state"] = "CONTINUATION_2R"
    elif reached_1:
        out["path_state"] = "CONTINUATION_1R"
    else:
        out["path_state"] = "UNRESOLVED"

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/research/xauusd_m30_structural_failure_path.csv"),
    )
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    candidates = build_candidates(frame)
    if candidates.empty:
        print("No candidates.")
        return

    boundary = pd.Timestamp("2026-03-18")
    rows = []

    for _, candidate in candidates.iterrows():
        period = (
            "development"
            if pd.Timestamp(candidate["confirmation_timestamp"]) < boundary
            else "historical_oos"
        )
        for horizon in HORIZONS:
            metrics = trace(candidate, frame, horizon)
            if not metrics:
                continue
            rows.append({
                **candidate.to_dict(),
                "period": period,
                "horizon": horizon,
                **metrics,
            })

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    summary = []
    for period in ("development", "historical_oos"):
        for direction in ("UP", "DOWN"):
            for horizon in HORIZONS:
                subset = result[
                    (result["period"] == period)
                    & (result["direction"] == direction)
                    & (result["horizon"] == horizon)
                ]
                if subset.empty:
                    continue

                item = {
                    "period": period,
                    "direction": direction,
                    "horizon": horizon,
                    "n": len(subset),
                    "failure_rate": subset["failure"].mean(),
                    "hit_1R": subset["hit_1R"].mean(),
                    "hit_2R": subset["hit_2R"].mean(),
                    "hit_3R": subset["hit_3R"].mean(),
                    "hit_4R": subset["hit_4R"].mean(),
                    "failure_after_1R": subset["failure_after_1R"].mean(),
                    "failure_after_2R": subset["failure_after_2R"].mean(),
                    "failure_bar_median": subset["failure_bar"].median(),
                    "mfe_before_failure_r_median": subset["mfe_before_failure_r"].median(),
                    "mae_before_failure_r_median": subset["mae_before_failure_r"].median(),
                }
                for level in MFE_LEVELS:
                    item[f"bars_to_{level:g}R_median"] = subset[f"bars_to_{level:g}R"].median()
                summary.append(item)

    print("=== Structural Failure Path Research ===")
    print("BOS -> valid swing -> 50% retracement entry")
    print("Failure = confirmed swing invalidation")
    print()
    print(pd.DataFrame(summary).to_string(index=False))
    print()
    print("Candidates:", len(candidates))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
