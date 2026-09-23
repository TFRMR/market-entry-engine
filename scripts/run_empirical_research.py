"""Run a first empirical context/outcome research pass on an MT5 export."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_engine.context import build_context_dataset
from market_engine.data import load_mt5_csv
from market_engine.entry import build_setup_candidates
from market_engine.features import build_structural_features
from market_engine.holdout import HISTORICAL_AUDIT_CUTOFF
from market_engine.labels import STRUCTURAL_LABEL_HORIZON, build_setup_label_dataset
from market_engine.empirical import summarize_continuous_context, summarize_outcomes
from market_engine.structure import build_structural_sequence, process_structural_candles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--spread-price", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("data/research"))
    args = parser.parse_args()

    frame = load_mt5_csv(args.csv)
    candidates = build_setup_candidates(frame)
    features = build_structural_features(frame)
    structural = build_structural_sequence(frame)
    swings, events = process_structural_candles(structural)

    labels = build_setup_label_dataset(
        candidates=candidates,
        swings=swings,
        frame=frame,
        feature_frame=None,
        spread_price=args.spread_price,
        feature_columns=(),
        horizon=STRUCTURAL_LABEL_HORIZON,
        events=events,
    )

    context = build_context_dataset(
        frame,
        candidates,
        features,
        swings=swings,
        events=events,
    )

    # Labels already contain setup metadata; retain only outcome fields here
    # before joining so the context side remains the canonical setup snapshot.
    outcome_columns = [
        "setup_index",
        "label",
        "reward_risk",
        "ambiguous_barrier",
        "target_price",
        "entry_price",
        "invalidation_price",
        "risk",
    ]
    outcomes = labels[[column for column in outcome_columns if column in labels.columns]]

    dataset = context.merge(
        outcomes,
        on="setup_index",
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output_dir / "xauusd_m30_context_outcomes.csv", index=False)

    dev = dataset[dataset["setup_timestamp"] < HISTORICAL_AUDIT_CUTOFF].copy()
    historical = dataset[dataset["setup_timestamp"] >= HISTORICAL_AUDIT_CUTOFF].copy()

    categorical = summarize_outcomes(
        dev,
        ["direction", "structure_direction", "trend_regime", "range_position_zone"],
    )
    categorical.to_csv(args.output_dir / "xauusd_m30_empirical_categorical.csv", index=False)

    continuous = summarize_continuous_context(
        dev,
        "pullback_retracement_ratio",
    )
    continuous.to_csv(args.output_dir / "xauusd_m30_empirical_retracement.csv", index=False)

    print("=== Empirical Research ===")
    print(f"Setup candidates: {len(candidates)}")
    print(f"Labeled context rows: {len(dataset)}")
    print(f"Development rows: {len(dev)}")
    print(f"Historical rows: {len(historical)}")
    print()
    print("Categorical context distribution:")
    print(categorical.to_string(index=False))
    print()
    print("Pullback retracement distribution:")
    print(continuous.to_string(index=False))


if __name__ == "__main__":
    main()
