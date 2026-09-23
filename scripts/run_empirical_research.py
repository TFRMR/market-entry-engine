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



BASE_CONTEXT_COLUMNS = [
    "direction", "structure_direction", "trend_regime", "range_position_zone"
]

POI_INTERACTION_COLUMNS = {
    "fvg": "poi_fvg_interaction",
    "ob": "poi_ob_interaction",
    "obim": "poi_obim_interaction",
    "liquidity": "poi_liquidity_interaction",
    "sr": "poi_sr_interaction",
}


def summarize_pairwise_contexts(dataset: pd.DataFrame) -> pd.DataFrame:
    """Describe base context plus one POI interaction dimension."""
    parts = []
    for name, poi_column in POI_INTERACTION_COLUMNS.items():
        summary = summarize_outcomes(dataset, [*BASE_CONTEXT_COLUMNS, poi_column])
        summary.insert(0, "poi_dimension", name)
        parts.append(summary)
    return pd.concat(parts, ignore_index=True)


def summarize_poi_interaction_pairs(dataset: pd.DataFrame) -> pd.DataFrame:
    """Describe observed joint states for pairs of POI interactions."""
    parts = []
    names = list(POI_INTERACTION_COLUMNS)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            summary = summarize_outcomes(
                dataset,
                [POI_INTERACTION_COLUMNS[left_name], POI_INTERACTION_COLUMNS[right_name]],
            )
            summary.insert(0, "poi_pair", f"{left_name}+{right_name}")
            parts.append(summary)
    return pd.concat(parts, ignore_index=True)


def summarize_base_poi_pairs(dataset: pd.DataFrame) -> pd.DataFrame:
    """Describe base context plus two POI interaction dimensions."""
    parts = []
    names = list(POI_INTERACTION_COLUMNS)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            summary = summarize_outcomes(
                dataset,
                [
                    *BASE_CONTEXT_COLUMNS,
                    POI_INTERACTION_COLUMNS[left_name],
                    POI_INTERACTION_COLUMNS[right_name],
                ],
            )
            summary.insert(0, "poi_pair", f"{left_name}+{right_name}")
            parts.append(summary)
    return pd.concat(parts, ignore_index=True)


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

    outcome_columns = [
        "setup_index",
        "direction",
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
        on=["setup_index", "direction"],
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

    retracement_values = pd.to_numeric(dev["pullback_retracement_ratio"], errors="coerce").dropna()
    _, retracement_edges = pd.qcut(
        retracement_values,
        q=min(4, int(retracement_values.nunique())),
        duplicates="drop",
        retbins=True,
    )
    historical_categorical = summarize_outcomes(
        historical,
        ["direction", "structure_direction", "trend_regime", "range_position_zone"],
    )
    historical_categorical.to_csv(
        args.output_dir / "xauusd_m30_empirical_historical_categorical.csv",
        index=False,
    )
    historical_continuous = summarize_continuous_context(
        historical,
        "pullback_retracement_ratio",
        bin_edges=retracement_edges.tolist(),
    )
    historical_continuous.to_csv(
        args.output_dir / "xauusd_m30_empirical_historical_retracement.csv",
        index=False,
    )

    pairwise = summarize_pairwise_contexts(dev)
    pairwise.to_csv(
        args.output_dir / "xauusd_m30_empirical_pairwise.csv",
        index=False,
    )
    historical_pairwise = summarize_pairwise_contexts(historical)
    historical_pairwise.to_csv(
        args.output_dir / "xauusd_m30_empirical_historical_pairwise.csv",
        index=False,
    )

    poi_pairs = summarize_poi_interaction_pairs(dev)
    poi_pairs.to_csv(
        args.output_dir / "xauusd_m30_empirical_poi_pairs.csv",
        index=False,
    )
    historical_poi_pairs = summarize_poi_interaction_pairs(historical)
    historical_poi_pairs.to_csv(
        args.output_dir / "xauusd_m30_empirical_historical_poi_pairs.csv",
        index=False,
    )

    base_poi_pairs = summarize_base_poi_pairs(dev)
    base_poi_pairs.to_csv(
        args.output_dir / "xauusd_m30_empirical_base_poi_pairs.csv",
        index=False,
    )
    historical_base_poi_pairs = summarize_base_poi_pairs(historical)
    historical_base_poi_pairs.to_csv(
        args.output_dir / "xauusd_m30_empirical_historical_base_poi_pairs.csv",
        index=False,
    )

    print("=== Empirical Research ===")
    print(f"Setup candidates: {len(candidates)}")
    print(f"Labeled context rows: {len(dataset)}")
    print(f"Development rows: {len(dev)}")
    print(f"Historical rows: {len(historical)}")
    print()
    print("Categorical context distribution:")
    print(categorical.to_string(index=False))
    print()
    print("Pullback retracement distribution (development bins):")
    print(continuous.to_string(index=False))
    print()
    print("Historical/OOS categorical distribution:")
    print(historical_categorical.to_string(index=False))
    print()
    print("Historical/OOS pullback retracement distribution (same development bins):")
    print(historical_continuous.to_string(index=False))
    print()
    print("Pairwise base-context + POI interaction (development):")
    print(pairwise.to_string(index=False))
    print()
    print("Pairwise base-context + POI interaction (historical/OOS):")
    print(historical_pairwise.to_string(index=False))
    print()
    print("POI interaction pairs (development):")
    print(poi_pairs.to_string(index=False))
    print()
    print("POI interaction pairs (historical/OOS):")
    print(historical_poi_pairs.to_string(index=False))
    print()
    print("Base context + two POI interactions (development):")
    print(base_poi_pairs.to_string(index=False))
    print()
    print("Base context + two POI interactions (historical/OOS):")
    print(historical_base_poi_pairs.to_string(index=False)


if __name__ == "__main__":
    main()
