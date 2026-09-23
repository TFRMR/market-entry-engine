"""Run empirical context/outcome research and chronological stability analysis."""

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
    "direction",
    "structure_direction",
    "trend_regime",
    "range_position_zone",
]

POI_INTERACTION_COLUMNS = {
    "fvg": "poi_fvg_interaction",
    "ob": "poi_ob_interaction",
    "obim": "poi_obim_interaction",
    "liquidity": "poi_liquidity_interaction",
    "sr": "poi_sr_interaction",
}

STABILITY_MIN_DEVELOPMENT = 20
STABILITY_MIN_HISTORICAL = 10


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
        for right_name in names[left_index + 1 :]:
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
        for right_name in names[left_index + 1 :]:
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


def build_stability_summary(
    development: pd.DataFrame,
    historical: pd.DataFrame,
    group_by: list[str],
    family: str,
) -> pd.DataFrame:
    """Compare identical context definitions across development and later data."""
    dev_summary = summarize_outcomes(development, group_by).rename(
        columns={
            "sample_count": "development_count",
            "tp_first_count": "development_tp_first_count",
            "sl_first_count": "development_sl_first_count",
            "unresolved_count": "development_unresolved_count",
            "both_same_candle_count": "development_both_same_candle_count",
            "tp_first_rate": "development_tp_first_rate",
            "sl_first_rate": "development_sl_first_rate",
            "unresolved_rate": "development_unresolved_rate",
            "both_same_candle_rate": "development_both_same_candle_rate",
        }
    )
    hist_summary = summarize_outcomes(historical, group_by).rename(
        columns={
            "sample_count": "historical_count",
            "tp_first_count": "historical_tp_first_count",
            "sl_first_count": "historical_sl_first_count",
            "unresolved_count": "historical_unresolved_count",
            "both_same_candle_count": "historical_both_same_candle_count",
            "tp_first_rate": "historical_tp_first_rate",
            "sl_first_rate": "historical_sl_first_rate",
            "unresolved_rate": "historical_unresolved_rate",
            "both_same_candle_rate": "historical_both_same_candle_rate",
        }
    )

    merged = dev_summary.merge(hist_summary, on=group_by, how="inner", validate="one_to_one")
    merged = merged[
        (merged["development_count"] >= STABILITY_MIN_DEVELOPMENT)
        & (merged["historical_count"] >= STABILITY_MIN_HISTORICAL)
    ].copy()
    merged.insert(0, "context_family", family)
    merged["tp_first_rate_delta"] = (
        merged["historical_tp_first_rate"] - merged["development_tp_first_rate"]
    )
    merged["sl_first_rate_delta"] = (
        merged["historical_sl_first_rate"] - merged["development_sl_first_rate"]
    )
    merged["unresolved_rate_delta"] = (
        merged["historical_unresolved_rate"] - merged["development_unresolved_rate"]
    )
    merged["tp_first_rate_abs_delta"] = merged["tp_first_rate_delta"].abs()
    return merged


def build_stability_dataset(
    development: pd.DataFrame,
    historical: pd.DataFrame,
) -> pd.DataFrame:
    """Build chronological stability views without ranking or scoring contexts."""
    views = [
        (
            "base_context",
            BASE_CONTEXT_COLUMNS,
        ),
        (
            "base_plus_one_poi",
            [*BASE_CONTEXT_COLUMNS, "poi_fvg_interaction"],
        ),
        (
            "base_plus_one_poi",
            [*BASE_CONTEXT_COLUMNS, "poi_ob_interaction"],
        ),
        (
            "base_plus_one_poi",
            [*BASE_CONTEXT_COLUMNS, "poi_obim_interaction"],
        ),
        (
            "base_plus_one_poi",
            [*BASE_CONTEXT_COLUMNS, "poi_liquidity_interaction"],
        ),
        (
            "base_plus_one_poi",
            [*BASE_CONTEXT_COLUMNS, "poi_sr_interaction"],
        ),
        (
            "poi_pair",
            ["poi_fvg_interaction", "poi_ob_interaction"],
        ),
        (
            "poi_pair",
            ["poi_fvg_interaction", "poi_obim_interaction"],
        ),
        (
            "poi_pair",
            ["poi_fvg_interaction", "poi_liquidity_interaction"],
        ),
        (
            "poi_pair",
            ["poi_fvg_interaction", "poi_sr_interaction"],
        ),
        (
            "poi_pair",
            ["poi_ob_interaction", "poi_obim_interaction"],
        ),
        (
            "poi_pair",
            ["poi_ob_interaction", "poi_liquidity_interaction"],
        ),
        (
            "poi_pair",
            ["poi_ob_interaction", "poi_sr_interaction"],
        ),
        (
            "poi_pair",
            ["poi_obim_interaction", "poi_liquidity_interaction"],
        ),
        (
            "poi_pair",
            ["poi_obim_interaction", "poi_sr_interaction"],
        ),
        (
            "poi_pair",
            ["poi_liquidity_interaction", "poi_sr_interaction"],
        ),
    ]

    parts = [
        build_stability_summary(development, historical, group_by, family)
        for family, group_by in views
    ]
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

    categorical = summarize_outcomes(dev, BASE_CONTEXT_COLUMNS)
    categorical.to_csv(args.output_dir / "xauusd_m30_empirical_categorical.csv", index=False)

    continuous = summarize_continuous_context(dev, "pullback_retracement_ratio")
    continuous.to_csv(args.output_dir / "xauusd_m30_empirical_retracement.csv", index=False)

    retracement_values = pd.to_numeric(
        dev["pullback_retracement_ratio"], errors="coerce"
    ).dropna()
    _, retracement_edges = pd.qcut(
        retracement_values,
        q=min(4, int(retracement_values.nunique())),
        duplicates="drop",
        retbins=True,
    )
    historical_categorical = summarize_outcomes(historical, BASE_CONTEXT_COLUMNS)
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
    pairwise.to_csv(args.output_dir / "xauusd_m30_empirical_pairwise.csv", index=False)
    historical_pairwise = summarize_pairwise_contexts(historical)
    historical_pairwise.to_csv(
        args.output_dir / "xauusd_m30_empirical_historical_pairwise.csv",
        index=False,
    )

    poi_pairs = summarize_poi_interaction_pairs(dev)
    poi_pairs.to_csv(args.output_dir / "xauusd_m30_empirical_poi_pairs.csv", index=False)
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

    stability = build_stability_dataset(dev, historical)
    stability.to_csv(
        args.output_dir / "xauusd_m30_empirical_stability.csv",
        index=False,
    )

    print("=== Empirical Research ===")
    print(f"Setup candidates: {len(candidates)}")
    print(f"Labeled context rows: {len(dataset)}")
    print(f"Development rows: {len(dev)}")
    print(f"Historical rows: {len(historical)}")
    print()
    print("Stability analysis:")
    print(f"Minimum development sample: {STABILITY_MIN_DEVELOPMENT}")
    print(f"Minimum historical sample: {STABILITY_MIN_HISTORICAL}")
    print(f"Matched context groups: {len(stability)}")
    if not stability.empty:
        print(
            stability[
                [
                    "context_family",
                    "development_count",
                    "historical_count",
                    "development_tp_first_rate",
                    "historical_tp_first_rate",
                    "tp_first_rate_delta",
                    "development_unresolved_rate",
                    "historical_unresolved_rate",
                    "unresolved_rate_delta",
                ]
            ].to_string(index=False)
        )
    print()
    print(
        "Detailed empirical tables remain in data/research/*.csv; "
        "terminal output is intentionally limited to stability groups."
    )


if __name__ == "__main__":
    main()
