"""Forward time-series calibration for the empirical Market Journey model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from market_engine.data import load_mt5_csv
from run_market_journey_context_research import build_context

HORIZON = 40
STATES = ("ENTRY", "SWING_UPDATE", "CONTINUATION", "TRANSITION")
NEXT_STATES = STATES + ("INVALIDATED",)
N_FOLDS = 5


def add_period(frame: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    boundary = pd.Timestamp("2026-03-18")
    context = context.copy()
    context["confirmation_timestamp"] = context["confirmation_index"].map(
        lambda i: frame.iloc[int(i)]["timestamp"]
    )
    context["period"] = context["confirmation_timestamp"].map(
        lambda ts: "development" if pd.Timestamp(ts) < boundary else "historical_oos"
    )
    return context


def make_folds(episode: pd.DataFrame) -> pd.DataFrame:
    candidates = (
        episode[["candidate_id", "confirmation_index"]]
        .drop_duplicates("candidate_id")
        .sort_values("confirmation_index")
        .reset_index(drop=True)
    )
    candidates["fold"] = (
        pd.qcut(candidates.index, q=N_FOLDS, labels=False, duplicates="drop") + 1
    )
    return episode.merge(candidates[["candidate_id", "fold"]], on="candidate_id", how="left")


def probability_table(train: pd.DataFrame) -> pd.DataFrame:
    observed = train[train["next_state"].notna()].copy()
    counts = (
        observed.groupby(["direction", "bos_scope", "state", "next_state"])
        .size()
        .rename("n")
        .reset_index()
    )
    denominators = (
        observed.groupby(["direction", "bos_scope", "state"])
        .size()
        .rename("denominator")
        .reset_index()
    )
    table = counts.merge(
        denominators, on=["direction", "bos_scope", "state"], how="left"
    )
    table["probability"] = table["n"] / table["denominator"]
    return table


def predict(group: pd.Series, table: pd.DataFrame) -> tuple[dict[str, float], str]:
    key = (
        (table["direction"] == group["direction"])
        & (table["bos_scope"] == group["bos_scope"])
        & (table["state"] == group["state"])
    )
    exact = table[key]
    if not exact.empty:
        probs = exact.set_index("next_state")["probability"].to_dict()
        source = "direction+bos_scope+state"
    else:
        # Keep the baseline descriptive: if an exact state context has not
        # appeared in prior data, fall back to the direction-level state model.
        key = (
            (table["direction"] == group["direction"])
            & (table["state"] == group["state"])
        )
        fallback = table[key]
        if fallback.empty:
            return {s: 1.0 / len(NEXT_STATES) for s in NEXT_STATES}, "uniform"
        denom = fallback.groupby(level=0).size() if False else None
        counts = fallback.groupby("next_state")["n"].sum()
        probs = (counts / counts.sum()).to_dict()
        source = "direction+state"
    return {s: float(probs.get(s, 0.0)) for s in NEXT_STATES}, source


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
            "data/research/xauusd_m30_market_journey_probability_calibration.csv"
        ),
    )
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    context = add_period(frame, build_context(frame))
    episodes = pd.read_csv(args.episodes)

    episode = episodes[
        (episodes["horizon"] == HORIZON)
        & episodes["state"].isin(STATES)
        & episodes["next_state"].notna()
    ].copy()

    episode = episode.merge(
        context[["candidate_id", "bos_scope", "period"]].rename(
            columns={"period": "context_period"}
        ),
        on="candidate_id",
        how="inner",
    )
    episode = episode[episode["context_period"] == "historical_oos"].copy()
    episode = make_folds(episode)

    predictions: list[dict] = []

    # Strict forward validation: fold k is predicted only from folds < k.
    for fold in range(2, N_FOLDS + 1):
        train = episode[episode["fold"] < fold]
        test = episode[episode["fold"] == fold]
        table = probability_table(train)

        for _, row in test.iterrows():
            probs, source = predict(row, table)
            for next_state in NEXT_STATES:
                predictions.append(
                    {
                        "fold": fold,
                        "candidate_id": row["candidate_id"],
                        "direction": row["direction"],
                        "bos_scope": row["bos_scope"],
                        "state": row["state"],
                        "actual_next_state": row["next_state"],
                        "predicted_next_state": next_state,
                        "probability": probs[next_state],
                        "source": source,
                    }
                )

    pred = pd.DataFrame(predictions)

    summary_rows: list[dict] = []
    for fold, group in pred.groupby("fold"):
        wide = group.pivot_table(
            index=["candidate_id", "direction", "bos_scope", "state"],
            columns="predicted_next_state",
            values="probability",
            aggfunc="first",
        ).fillna(0.0)
        actual = (
            group.drop_duplicates(
                ["candidate_id", "direction", "bos_scope", "state"]
            )
            .set_index(["candidate_id", "direction", "bos_scope", "state"])[
                "actual_next_state"
            ]
            .loc[wide.index]
        )
        probs = wide.reindex(columns=NEXT_STATES, fill_value=0.0).to_numpy()
        y = actual.to_numpy()
        labels = {s: i for i, s in enumerate(NEXT_STATES)}
        y_idx = np.array([labels[v] for v in y])

        brier = np.mean(np.sum((probs - np.eye(len(NEXT_STATES))[y_idx]) ** 2, axis=1))
        ll = log_loss(y, probs, labels=list(NEXT_STATES))
        accuracy = np.mean(
            wide[ list(NEXT_STATES) ].to_numpy().argmax(axis=1)
            == y_idx
        )
        summary_rows.append(
            {
                "fold": int(fold),
                "n": len(y),
                "brier_score": float(brier),
                "log_loss": float(ll),
                "argmax_accuracy": float(accuracy),
            }
        )

    summary = pd.DataFrame(summary_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)

    print("=== Market Journey Probability Forward Calibration ===")
    print("Historical OOS only | H40 | expanding chronological validation")
    print("Fold 1 is warm-up; fold k is predicted from folds < k.")
    print()
    print(summary.to_string(index=False))
    print()
    print("Mean metrics:")
    print(
        summary[["brier_score", "log_loss", "argmax_accuracy"]]
        .mean()
        .to_string()
    )
    print("Predicted rows:", len(pred))
    print("Artifact:", args.output)


if __name__ == "__main__":
    main()
