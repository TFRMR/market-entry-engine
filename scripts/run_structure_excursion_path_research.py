"""Map the price path after a structure-first 50% pullback entry.

Scenario:
BOS -> next same-direction valid swing -> 50% retracement entry.

Measure the journey in absolute price and in R using the confirmed swing
as the structural reference. The output is descriptive; it does not define
a trading rule.

For each horizon, report:
- MFE/MAE distributions
- probability of reaching +1R/+2R/+3R/+4R
- probability of staying within 0.5R/1R/2R adverse excursion
- median bars to first reach +1R/+2R
- median maximum adverse excursion before first +1R
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
MFE_LEVELS = (1.0, 2.0, 3.0, 4.0)
ADVERSE_LEVELS = (0.5, 1.0, 2.0)
RETRACE = 0.50


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
        swing_price = float(swing.price)
        leg = (
            confirmation - swing_price
            if direction is Direction.UP
            else swing_price - confirmation
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

        risk = abs(entry - swing_price)
        if risk <= 0:
            continue

        rows.append({
            "bos_index": event.index,
            "bos_timestamp": frame.iloc[event.index]["timestamp"],
            "bos_event": event.event,
            "bos_scope": event.scope.value if event.scope else None,
            "direction": direction.value,
            "confirmation_index": cidx,
            "confirmation_timestamp": frame.iloc[cidx]["timestamp"],
            "entry_index": entry_index,
            "entry_timestamp": frame.iloc[entry_index]["timestamp"],
            "entry": entry,
            "invalidation": swing_price,
            "risk": risk,
            "entry_wait_bars": entry_index - cidx,
        })

    return pd.DataFrame(rows)


def path_metrics(row: pd.Series, frame: pd.DataFrame, horizon: int) -> dict:
    entry_index = int(row["entry_index"])
    entry = float(row["entry"])
    risk = float(row["risk"])
    end = min(entry_index + horizon, len(frame) - 1)

    highs = frame.iloc[entry_index + 1:end + 1]["high"].astype(float).to_numpy()
    lows = frame.iloc[entry_index + 1:end + 1]["low"].astype(float).to_numpy()

    if len(highs) == 0:
        return {}

    if row["direction"] == "UP":
        favorable = highs - entry
        adverse = entry - lows
    else:
        favorable = entry - lows
        adverse = highs - entry

    mfe_r_path = np.maximum.accumulate(favorable / risk)
    mae_r_path = np.maximum.accumulate(adverse / risk)

    result = {
        "mfe_price": float(np.max(favorable)),
        "mae_price": float(np.max(adverse)),
        "mfe_r": float(np.max(favorable) / risk),
        "mae_r": float(np.max(adverse) / risk),
    }

    for level in MFE_LEVELS:
        hits = np.flatnonzero(favorable >= level * risk)
        result[f"hit_{level:g}R"] = bool(len(hits))
        result[f"bars_to_{level:g}R"] = (
            int(hits[0] + 1) if len(hits) else np.nan
        )

    for level in ADVERSE_LEVELS:
        result[f"mae_le_{level:g}R"] = bool(np.max(adverse) <= level * risk)

    first_1r = np.flatnonzero(favorable >= risk)
    if len(first_1r):
        stop = int(first_1r[0])
        result["mae_before_1R"] = float(np.max(adverse[: stop + 1]) / risk)
    else:
        result["mae_before_1R"] = np.nan

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/research/xauusd_m30_structure_excursion_path.csv"),
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
            metrics = path_metrics(candidate, frame, horizon)
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
                    "mfe_price_median": subset["mfe_price"].median(),
                    "mae_price_median": subset["mae_price"].median(),
                    "mfe_r_median": subset["mfe_r"].median(),
                    "mae_r_median": subset["mae_r"].median(),
                    "mae_before_1R_median": subset["mae_before_1R"].median(),
                }

                for level in MFE_LEVELS:
                    item[f"p_hit_{level:g}R"] = subset[f"hit_{level:g}R"].mean()
                    item[f"bars_to_{level:g}R_median"] = subset[f"bars_to_{level:g}R"].median()

                for level in ADVERSE_LEVELS:
                    item[f"p_mae_le_{level:g}R"] = subset[f"mae_le_{level:g}R"].mean()

                summary.append(item)

    print("=== Structure Excursion Path Research ===")
    print("BOS -> valid swing -> 50% retracement entry")
    print("Confirmed swing used as structural reference")
    print()
    print(pd.DataFrame(summary).to_string(index=False))
    print()
    print("Candidates:", len(candidates))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
