"""Empirical one-candle-ahead distribution experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from market_engine.data import load_mt5_csv
from run_market_journey_context_research import build_context

BOUNDARY = pd.Timestamp("2026-03-18")
N_FOLDS = 5
OUTCOMES = ("DOJI", "DOWN", "UP")\nLAPLACE_ALPHA = 1.0


def add_period(frame: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    context = context.copy()
    context["confirmation_timestamp"] = context["confirmation_index"].map(
        lambda i: frame.iloc[int(i)]["timestamp"]
    )
    context["period"] = context["confirmation_timestamp"].map(
        lambda ts: "development" if pd.Timestamp(ts) < BOUNDARY else "historical_oos"
    )
    return context


def candle_outcome(row: pd.Series) -> str:
    o, c = float(row.open), float(row.close)
    if c > o:
        return "UP"
    if c < o:
        return "DOWN"
    return "DOJI"


def make_folds(candidates: pd.DataFrame) -> pd.DataFrame:
    candidates = (
        candidates.sort_values("confirmation_index")
        .drop_duplicates("candidate_id")
        .reset_index(drop=True)
    )
    candidates["fold"] = (
        pd.qcut(candidates.index, q=N_FOLDS, labels=False, duplicates="drop") + 1
    )
    return candidates


def probability_table(train: pd.DataFrame) -> pd.DataFrame:
    return (
        train.groupby(["direction", "bos_scope", "state", "next_candle"])
        .size()
        .rename("n")
        .reset_index()
    )


def predict(row: pd.Series, table: pd.DataFrame) -> dict[str, float]:
    exact = table[
        (table.direction == row.direction)
        & (table.bos_scope == row.bos_scope)
        & (table.state == row.state)
    ]
    if exact.empty:
        exact = table[
            (table.direction == row.direction)
            & (table.state == row.state)
        ]
    if exact.empty:
        return {x: 1 / len(OUTCOMES) for x in OUTCOMES}

    probs = exact.set_index("next_candle")["n"]
    probs = probs / probs.sum()
    return {x: float(probs.get(x, 0.0)) for x in OUTCOMES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/research/xauusd_m30_next_candle_probability_calibration.csv"
        ),
    )
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    context = add_period(frame, build_context(frame))
    context = context[context.period == "historical_oos"].copy()

    rows = []
    for _, r in context.iterrows():
        i = int(r.confirmation_index)
        if i + 1 >= len(frame):
            continue
        rows.append(
            {
                "candidate_id": r.candidate_id,
                "confirmation_index": i,
                "direction": r.direction.value if hasattr(r.direction, "value") else r.direction,
                "bos_scope": r.bos_scope.value if hasattr(r.bos_scope, "value") else r.bos_scope,
                "state": "ENTRY",
                "next_candle": candle_outcome(frame.iloc[i + 1]),
            }
        )

    data = pd.DataFrame(rows)
    candidates = make_folds(data[["candidate_id", "confirmation_index"]])
    data = data.merge(
        candidates, on=["candidate_id", "confirmation_index"], how="left"
    )

    predictions = []
    for fold in range(2, N_FOLDS + 1):
        train = data[data.fold < fold]
        test = data[data.fold == fold]
        table = probability_table(train)
        for _, row in test.iterrows():
            probs = predict(row, table)
            predictions.append(
                {
                    "fold": int(fold),
                    "candidate_id": row.candidate_id,
                    "direction": row.direction,
                    "bos_scope": row.bos_scope,
                    "state": row.state,
                    "actual": row.next_candle,
                    **{f"p_{x.lower()}": probs[x] for x in OUTCOMES},
                }
            )

    pred = pd.DataFrame(predictions)
    summary = []
    labels = list(OUTCOMES)
    idx = {x: i for i, x in enumerate(labels)}
    for fold, g in pred.groupby("fold"):
        p = g[[f"p_{x.lower()}" for x in OUTCOMES]].to_numpy()
        y = g.actual.to_numpy()
        yi = np.array([idx[x] for x in y])
        brier = np.mean(np.sum((p - np.eye(len(labels))[yi]) ** 2, axis=1))
        ll = log_loss(y, p, labels=labels)
        accuracy = np.mean(p.argmax(axis=1) == yi)
        summary.append(
            {
                "fold": int(fold),
                "n": len(g),
                "brier_score": float(brier),
                "log_loss": float(ll),
                "argmax_accuracy": float(accuracy),
            }
        )

    result = pd.DataFrame(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print("=== Next Candle Empirical Probability Experiment ===")
    print("Historical OOS | H+1 candle | expanding chronological validation")
    print("Context: direction + BOS scope + current state (ENTRY)")
    print()
    print(result.to_string(index=False))
    print()
    print("Mean metrics:")
    print(result[["brier_score", "log_loss", "argmax_accuracy"]].mean().to_string())
    print()
    print("Observed next-candle distribution:")
    print(
        data.next_candle.value_counts(normalize=True)
        .reindex(OUTCOMES, fill_value=0)
        .to_string()
    )
    print("Candidates:", len(data))
    print("Predictions:", len(pred))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
