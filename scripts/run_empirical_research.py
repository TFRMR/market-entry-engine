"""Run empirical context/outcome research and chronological stability analysis."""

from __future__ import annotations

import argparse
import math
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

CONTEXT_IDENTITY_COLUMNS = [
    *BASE_CONTEXT_COLUMNS,
    *POI_INTERACTION_COLUMNS.values(),
]

STABILITY_MIN_DEVELOPMENT = 20
STABILITY_MIN_HISTORICAL = 10
STABILITY_CONFIDENCE_Z = 1.96


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


def wilson_interval(successes: pd.Series, sample_count: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return a 95% Wilson interval for a binomial rate."""
    proportion = successes / sample_count
    z = STABILITY_CONFIDENCE_Z
    denominator = 1.0 + (z * z) / sample_count
    center = (proportion + (z * z) / (2.0 * sample_count)) / denominator
    margin = (
        z
        * (
            proportion * (1.0 - proportion) / sample_count
            + (z * z) / (4.0 * sample_count * sample_count)
        ).pow(0.5)
        / denominator
    )
    return center - margin, center + margin


def normal_two_sided_p_value(z_score: pd.Series) -> pd.Series:
    """Approximate two-sided normal p-values without an extra dependency."""
    return z_score.abs().map(lambda value: math.erfc(value / math.sqrt(2.0)))


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

    merged["development_tp_first_ci_low"], merged["development_tp_first_ci_high"] = wilson_interval(
        merged["development_tp_first_count"], merged["development_count"]
    )
    merged["historical_tp_first_ci_low"], merged["historical_tp_first_ci_high"] = wilson_interval(
        merged["historical_tp_first_count"], merged["historical_count"]
    )
    merged["development_unresolved_ci_low"], merged["development_unresolved_ci_high"] = wilson_interval(
        merged["development_unresolved_count"], merged["development_count"]
    )
    merged["historical_unresolved_ci_low"], merged["historical_unresolved_ci_high"] = wilson_interval(
        merged["historical_unresolved_count"], merged["historical_count"]
    )

    tp_se = (
        merged["development_tp_first_rate"] * (1.0 - merged["development_tp_first_rate"])
        / merged["development_count"]
        + merged["historical_tp_first_rate"] * (1.0 - merged["historical_tp_first_rate"])
        / merged["historical_count"]
    ).pow(0.5)
    merged["tp_first_rate_delta_ci_low"] = merged["tp_first_rate_delta"] - STABILITY_CONFIDENCE_Z * tp_se
    merged["tp_first_rate_delta_ci_high"] = merged["tp_first_rate_delta"] + STABILITY_CONFIDENCE_Z * tp_se
    merged["tp_first_drift_z"] = merged["tp_first_rate_delta"] / tp_se.replace(0.0, float("nan"))
    merged["tp_first_drift_p_value"] = normal_two_sided_p_value(merged["tp_first_drift_z"].fillna(0.0))

    unresolved_se = (
        merged["development_unresolved_rate"] * (1.0 - merged["development_unresolved_rate"])
        / merged["development_count"]
        + merged["historical_unresolved_rate"] * (1.0 - merged["historical_unresolved_rate"])
        / merged["historical_count"]
    ).pow(0.5)
    merged["unresolved_rate_delta_ci_low"] = (
        merged["unresolved_rate_delta"] - STABILITY_CONFIDENCE_Z * unresolved_se
    )
    merged["unresolved_rate_delta_ci_high"] = (
        merged["unresolved_rate_delta"] + STABILITY_CONFIDENCE_Z * unresolved_se
    )
    merged["unresolved_drift_z"] = (
        merged["unresolved_rate_delta"] / unresolved_se.replace(0.0, float("nan"))
    )
    merged["unresolved_drift_p_value"] = normal_two_sided_p_value(
        merged["unresolved_drift_z"].fillna(0.0)
    )
    return merged


def build_research_hypotheses(stability: pd.DataFrame) -> pd.DataFrame:
    """Extract descriptive, testable research hypotheses from stability results."""
    rows: list[dict[str, object]] = []
    for _, row in stability.iterrows():
        dev_n = int(row["development_count"])
        hist_n = int(row["historical_count"])
        min_n = min(dev_n, hist_n)
        delta = float(row["tp_first_rate_delta"])
        ci_low = float(row["tp_first_rate_delta_ci_low"])
        ci_high = float(row["tp_first_rate_delta_ci_high"])
        abs_delta = abs(delta)
        if min_n >= 100 and abs_delta <= 0.05:
            status = "STABLE_BASELINE"
            rationale = "large matched sample with small observed TP_FIRST drift"
        elif min_n >= 50 and abs_delta <= 0.10:
            status = "STABLE_CANDIDATE"
            rationale = "moderate matched sample with limited observed drift"
        elif min_n < 50:
            status = "LOW_SAMPLE"
            rationale = "historical sample is still small for a strong stability conclusion"
        else:
            status = "DRIFT_REQUIRES_RETEST"
            rationale = "observed drift is material enough to require additional chronological validation"
        uncertainty = "CI_CROSSES_ZERO" if ci_low <= 0.0 <= ci_high else "CI_EXCLUDES_ZERO"

        hypothesis = {
            "context_family": row["context_family"],
            "sample_band": (
                "10-19" if min_n < 20 else
                "20-49" if min_n < 50 else
                "50-99" if min_n < 100 else
                "100-249" if min_n < 250 else "250+"
            ),
            "development_count": dev_n,
            "historical_count": hist_n,
            "tp_first_development_rate": float(row["development_tp_first_rate"]),
            "tp_first_historical_rate": float(row["historical_tp_first_rate"]),
            "tp_first_rate_delta": delta,
            "tp_first_abs_delta": abs_delta,
            "tp_first_delta_ci_low": ci_low,
            "tp_first_delta_ci_high": ci_high,
            "tp_first_drift_p_value": float(row["tp_first_drift_p_value"]),
            "stability_status": status,
            "uncertainty_status": uncertainty,
            "hypothesis": "TP_FIRST outcome distribution remains temporally similar for this context definition",
            "rationale": rationale,
        }
        for column in CONTEXT_IDENTITY_COLUMNS:
            hypothesis[column] = row[column] if column in row.index else pd.NA
        rows.append(hypothesis)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["stability_status", "context_family", "historical_count", *CONTEXT_IDENTITY_COLUMNS],
        ascending=[True, True, False, *([True] * len(CONTEXT_IDENTITY_COLUMNS))],
        na_position="last",
    ).reset_index(drop=True)



def _hypothesis_context_columns(row: pd.Series) -> list[str]:
    """Return the context identity columns that define one hypothesis row."""
    return [
        column
        for column in CONTEXT_IDENTITY_COLUMNS
        if pd.notna(row.get(column, pd.NA))
    ]


def _chronological_fold_bounds(dataset: pd.DataFrame, fold_count: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split observations into equal-sized chronological test periods."""
    ordered = dataset.sort_values("setup_timestamp").reset_index(drop=True)
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if len(ordered) < fold_count:
        raise ValueError("not enough observations for chronological folds")
    timestamps = pd.to_datetime(ordered["setup_timestamp"])
    bounds = []
    for fold_index in range(1, fold_count):
        start = timestamps.iloc[(len(ordered) * fold_index) // fold_count]
        end = timestamps.iloc[(len(ordered) * (fold_index + 1)) // fold_count] if fold_index + 1 < fold_count else timestamps.iloc[-1]
        bounds.append((start, end))
    return bounds


def evaluate_research_hypotheses_chronologically(
    dataset: pd.DataFrame,
    hypotheses: pd.DataFrame,
    fold_count: int = 3,
) -> pd.DataFrame:
    """Evaluate the extracted context hypotheses on expanding chronological folds."""
    ordered = dataset.sort_values("setup_timestamp").reset_index(drop=True)
    timestamps = pd.to_datetime(ordered["setup_timestamp"])
    fold_positions = [
        min(len(ordered) - 1, (len(ordered) * index) // fold_count)
        for index in range(fold_count + 1)
    ]
    fold_edges = [timestamps.iloc[position] for position in fold_positions]
    rows: list[dict[str, object]] = []

    for hypothesis_index, hypothesis in hypotheses.iterrows():
        context_columns = _hypothesis_context_columns(hypothesis)
        family = hypothesis["context_family"]
        mask = pd.Series(True, index=ordered.index)
        for column in context_columns:
            mask &= ordered[column].eq(hypothesis[column])
        matching = ordered[mask].copy()
        if matching.empty:
            continue

        for fold_index in range(1, fold_count + 1):
            test_start = fold_edges[fold_index]
            test_end = fold_edges[fold_index + 1] if fold_index < fold_count else timestamps.iloc[-1]
            if fold_index == fold_count:
                test_mask = timestamps.ge(test_start) & timestamps.le(test_end)
            else:
                test_mask = timestamps.ge(test_start) & timestamps.lt(test_end)
            test = matching[test_mask.loc[matching.index]]
            train = matching[timestamps.loc[matching.index] < test_start]

            if train.empty or test.empty:
                continue

            train_labeled = train[train["label"].notna()]
            test_labeled = test[test["label"].notna()]
            if train_labeled.empty or test_labeled.empty:
                continue

            train_rate = (train_labeled["label"] == "TP_FIRST").mean()
            test_rate = (test_labeled["label"] == "TP_FIRST").mean()
            rows.append(
                {
                    "hypothesis_index": int(hypothesis_index),
                    "context_family": family,
                    "fold": fold_index,
                    "train_end": test_start,
                    "test_start": test_start,
                    "test_end": test_end,
                    "development_train_count": len(train_labeled),
                    "chronological_test_count": len(test_labeled),
                    "train_tp_first_rate": float(train_rate),
                    "test_tp_first_rate": float(test_rate),
                    "tp_first_rate_delta": float(test_rate - train_rate),
                    "hypothesis_status": hypothesis["stability_status"],
                    "uncertainty_status": hypothesis["uncertainty_status"],
                    **{column: hypothesis[column] for column in CONTEXT_IDENTITY_COLUMNS},
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["hypothesis_index", "fold"]
    ).reset_index(drop=True)


def summarize_chronological_hypothesis_evaluation(
    evaluation: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize fold-level temporal drift without ranking hypotheses."""
    if evaluation.empty:
        return evaluation
    return (
        evaluation.groupby(["hypothesis_index", "context_family"], as_index=False)
        .agg(
            folds_evaluated=("fold", "count"),
            median_train_count=("development_train_count", "median"),
            median_test_count=("chronological_test_count", "median"),
            median_abs_tp_first_delta=("tp_first_rate_delta", lambda values: values.abs().median()),
            max_abs_tp_first_delta=("tp_first_rate_delta", lambda values: values.abs().max()),
        )
    )




def build_development_hypothesis_candidates(development: pd.DataFrame) -> pd.DataFrame:
    """Enumerate context definitions using development data only."""
    views = [
        ("base_context", BASE_CONTEXT_COLUMNS),
        *[("base_plus_one_poi", [*BASE_CONTEXT_COLUMNS, column]) for column in POI_INTERACTION_COLUMNS.values()],
        *[
            ("poi_pair", [left, right])
            for index, left in enumerate(POI_INTERACTION_COLUMNS.values())
            for right in list(POI_INTERACTION_COLUMNS.values())[index + 1 :]
        ],
    ]
    rows: list[dict[str, object]] = []
    for family, columns in views:
        counts = development.groupby(columns, dropna=False, observed=False).size().reset_index(name="development_count")
        counts = counts[counts["development_count"] >= STABILITY_MIN_DEVELOPMENT]
        for _, row in counts.iterrows():
            candidate = {
                "context_family": family,
                "development_count": int(row["development_count"]),
                "historical_count": pd.NA,
                "stability_status": "DEVELOPMENT_CANDIDATE",
                "uncertainty_status": "NOT_USED",
            }
            for column in CONTEXT_IDENTITY_COLUMNS:
                candidate[column] = row[column] if column in columns else pd.NA
            rows.append(candidate)
    return pd.DataFrame(rows)

def select_locked_hypotheses(
    chronological_summary: pd.DataFrame,
    hypotheses: pd.DataFrame,
    *,
    min_folds: int = 3,
    min_test_count: int = 20,
    max_median_abs_delta: float = 0.10,
    max_fold_abs_delta: float | None = None,
) -> pd.DataFrame:
    """Lock development hypotheses using chronological development evidence only."""
    if chronological_summary.empty or hypotheses.empty:
        return hypotheses.iloc[0:0].copy()
    eligible = chronological_summary[
        (chronological_summary["folds_evaluated"] >= min_folds)
        & (chronological_summary["median_test_count"] >= min_test_count)
        & (chronological_summary["median_abs_tp_first_delta"] <= max_median_abs_delta)
    ][["hypothesis_index"]]
    locked = hypotheses[hypotheses.index.isin(eligible["hypothesis_index"])].copy()
    locked.insert(0, "locked_hypothesis_index", locked.index.astype(int))
    locked["selection_rule"] = (
        f"development_only: folds>={min_folds}, median_test_count>={min_test_count}, "
        f"median_abs_delta<={max_median_abs_delta:.2f}"
    )
    return locked.reset_index(drop=True)

def evaluate_locked_hypotheses_on_historical(
    historical: pd.DataFrame,
    locked_hypotheses: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate locked context definitions once on untouched historical data."""
    rows: list[dict[str, object]] = []
    labeled = historical[historical["label"].notna()].copy()
    for _, hypothesis in locked_hypotheses.iterrows():
        context_columns = _hypothesis_context_columns(hypothesis)
        mask = pd.Series(True, index=labeled.index)
        for column in context_columns:
            mask &= labeled[column].eq(hypothesis[column])
        matched = labeled[mask]
        if matched.empty:
            continue
        rows.append({
            "locked_hypothesis_index": int(hypothesis["locked_hypothesis_index"]),
            "context_family": hypothesis["context_family"],
            "historical_count": len(matched),
            "historical_tp_first_count": int((matched["label"] == "TP_FIRST").sum()),
            "historical_sl_first_count": int((matched["label"] == "SL_FIRST").sum()),
            "historical_unresolved_count": int((matched["label"] == "UNRESOLVED").sum()),
            "historical_tp_first_rate": float((matched["label"] == "TP_FIRST").mean()),
            "development_count": hypothesis["development_count"],
            **{column: hypothesis[column] for column in CONTEXT_IDENTITY_COLUMNS},
        })
    return pd.DataFrame(rows)


def evaluate_locked_hypotheses_economic(
    dataset: pd.DataFrame,
    locked_hypotheses: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    """Evaluate deterministic R-multiple outcomes for locked contexts."""
    rows: list[dict[str, object]] = []
    labeled = dataset[dataset["label"].notna()].copy()

    for _, hypothesis in locked_hypotheses.iterrows():
        context_columns = _hypothesis_context_columns(hypothesis)
        mask = pd.Series(True, index=labeled.index)
        for column in context_columns:
            mask &= labeled[column].eq(hypothesis[column])
        matched = labeled[mask].copy()
        if matched.empty:
            continue

        label = matched["label"].astype(str)
        tp = matched[label == "TP_FIRST"]
        sl = matched[label == "SL_FIRST"]
        unresolved = matched[label == "UNRESOLVED"]
        ambiguous = matched[label == "BOTH_SAME_CANDLE"]

        tp_rr = pd.to_numeric(tp["reward_risk"], errors="coerce")
        all_rr = pd.to_numeric(matched["reward_risk"], errors="coerce")
        resolved = len(tp) + len(sl)
        resolved_net_r = float(tp_rr.sum()) - float(len(sl))

        rows.append({
            "locked_hypothesis_index": int(hypothesis["locked_hypothesis_index"]),
            "period": period,
            "context_family": hypothesis["context_family"],
            "development_count": int(hypothesis["development_count"]),
            "sample_count": len(matched),
            "economic_sample_count": len(tp) + len(sl) + len(unresolved),
            "tp_first_count": len(tp),
            "sl_first_count": len(sl),
            "unresolved_count": len(unresolved),
            "both_same_candle_count": len(ambiguous),
            "tp_first_rate": float(len(tp) / len(matched)),
            "sl_first_rate": float(len(sl) / len(matched)),
            "unresolved_rate": float(len(unresolved) / len(matched)),
            "mean_reward_risk": float(all_rr.mean()),
            "median_reward_risk": float(all_rr.median()),
            "tp_mean_reward_risk": float(tp_rr.mean()) if len(tp) else float("nan"),
            "tp_median_reward_risk": float(tp_rr.median()) if len(tp) else float("nan"),
            "resolved_net_r": resolved_net_r,
            "expected_r_per_economic_sample": float(
                resolved_net_r / (len(tp) + len(sl) + len(unresolved))
            ),
            "expected_r_per_resolved_trade": float(
                resolved_net_r / resolved
            ) if resolved else float("nan"),
            **{column: hypothesis[column] for column in CONTEXT_IDENTITY_COLUMNS},
        })

    return pd.DataFrame(rows)


def build_stability_report(stability: pd.DataFrame) -> pd.DataFrame:
    """Summarize stability uncertainty by context family and sample-size band."""
    report = stability.copy()
    report["sample_band"] = pd.cut(
        report[["development_count", "historical_count"]].min(axis=1),
        bins=[0, 19, 49, 99, 249, float("inf")],
        labels=["10-19", "20-49", "50-99", "100-249", "250+"],
        include_lowest=True,
    )
    report["tp_first_ci_width"] = (
        report["tp_first_rate_delta_ci_high"] - report["tp_first_rate_delta_ci_low"]
    )
    report["tp_first_ci_excludes_zero"] = (
        (report["tp_first_rate_delta_ci_low"] > 0)
        | (report["tp_first_rate_delta_ci_high"] < 0)
    )
    grouped = (
        report.groupby(["context_family", "sample_band"], observed=False)
        .agg(
            groups=("context_family", "size"),
            development_count_median=("development_count", "median"),
            historical_count_median=("historical_count", "median"),
            median_abs_tp_first_delta=("tp_first_rate_abs_delta", "median"),
            median_tp_first_ci_width=("tp_first_ci_width", "median"),
            ci_excludes_zero_count=("tp_first_ci_excludes_zero", "sum"),
            ci_excludes_zero_rate=("tp_first_ci_excludes_zero", "mean"),
            median_tp_first_p_value=("tp_first_drift_p_value", "median"),
        )
        .reset_index()
    )
    return grouped


def build_stability_dataset(
    development: pd.DataFrame,
    historical: pd.DataFrame,
) -> pd.DataFrame:
    """Build chronological stability views without ranking or scoring contexts."""
    views = [
        ("base_context", BASE_CONTEXT_COLUMNS),
        ("base_plus_one_poi", [*BASE_CONTEXT_COLUMNS, "poi_fvg_interaction"]),
        ("base_plus_one_poi", [*BASE_CONTEXT_COLUMNS, "poi_ob_interaction"]),
        ("base_plus_one_poi", [*BASE_CONTEXT_COLUMNS, "poi_obim_interaction"]),
        ("base_plus_one_poi", [*BASE_CONTEXT_COLUMNS, "poi_liquidity_interaction"]),
        ("base_plus_one_poi", [*BASE_CONTEXT_COLUMNS, "poi_sr_interaction"]),
        ("poi_pair", ["poi_fvg_interaction", "poi_ob_interaction"]),
        ("poi_pair", ["poi_fvg_interaction", "poi_obim_interaction"]),
        ("poi_pair", ["poi_fvg_interaction", "poi_liquidity_interaction"]),
        ("poi_pair", ["poi_fvg_interaction", "poi_sr_interaction"]),
        ("poi_pair", ["poi_ob_interaction", "poi_obim_interaction"]),
        ("poi_pair", ["poi_ob_interaction", "poi_liquidity_interaction"]),
        ("poi_pair", ["poi_ob_interaction", "poi_sr_interaction"]),
        ("poi_pair", ["poi_obim_interaction", "poi_liquidity_interaction"]),
        ("poi_pair", ["poi_obim_interaction", "poi_sr_interaction"]),
        ("poi_pair", ["poi_liquidity_interaction", "poi_sr_interaction"]),
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
    stability_report = build_stability_report(stability)
    stability_report.to_csv(
        args.output_dir / "xauusd_m30_empirical_stability_report.csv",
        index=False,
    )
    research_hypotheses = build_research_hypotheses(stability)
    research_hypotheses.to_csv(
        args.output_dir / "xauusd_m30_empirical_research_hypotheses.csv",
        index=False,
    )

    development_candidates = build_development_hypothesis_candidates(dev)
    development_candidates.to_csv(
        args.output_dir / "xauusd_m30_empirical_development_hypothesis_candidates.csv",
        index=False,
    )
    chronological_hypotheses = evaluate_research_hypotheses_chronologically(
        dev,
        development_candidates,
        fold_count=3,
    )
    chronological_hypotheses.to_csv(
        args.output_dir / "xauusd_m30_empirical_chronological_hypotheses.csv",
        index=False,
    )
    chronological_summary = summarize_chronological_hypothesis_evaluation(
        chronological_hypotheses
    )
    chronological_summary.to_csv(
        args.output_dir / "xauusd_m30_empirical_chronological_hypothesis_summary.csv",
        index=False,
    )
    locked_hypotheses = select_locked_hypotheses(
        chronological_summary,
        development_candidates,
    )
    locked_hypotheses.to_csv(
        args.output_dir / "xauusd_m30_empirical_locked_hypotheses.csv",
        index=False,
    )
    historical_locked = evaluate_locked_hypotheses_on_historical(
        historical,
        locked_hypotheses,
    )
    historical_locked.to_csv(
        args.output_dir / "xauusd_m30_empirical_locked_hypotheses_oos.csv",
        index=False,
    )
    economic_development = evaluate_locked_hypotheses_economic(
        dev,
        locked_hypotheses,
        period="development",
    )
    economic_historical = evaluate_locked_hypotheses_economic(
        historical,
        locked_hypotheses,
        period="historical_oos",
    )
    economic = pd.concat(
        [economic_development, economic_historical],
        ignore_index=True,
    )
    economic.to_csv(
        args.output_dir / "xauusd_m30_empirical_locked_hypotheses_economic.csv",
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
                    "tp_first_rate_delta_ci_low",
                    "tp_first_rate_delta_ci_high",
                    "tp_first_drift_z",
                    "tp_first_drift_p_value",
                ]
            ].to_string(index=False)
        )
    print()
    print("Stability report:")
    print(stability_report.to_string(index=False))
    print()
    print()
    print("Research hypotheses:")
    hypothesis_columns = [
        "context_family",
        *CONTEXT_IDENTITY_COLUMNS,
        "development_count",
        "historical_count",
        "tp_first_rate_delta",
        "tp_first_delta_ci_low",
        "tp_first_delta_ci_high",
        "stability_status",
        "uncertainty_status",
    ]
    print(research_hypotheses[hypothesis_columns].to_string(index=False))
    print()
    print()
    print("Development-only hypothesis selection:")
    print(
        chronological_summary.to_string(index=False)
        if not chronological_summary.empty
        else "No chronological hypothesis evaluations available."
    )
    print()
    print(f"Locked hypotheses: {len(locked_hypotheses)}")
    print(
        locked_hypotheses[
            ["locked_hypothesis_index", "context_family", *CONTEXT_IDENTITY_COLUMNS, "development_count"]
        ].to_string(index=False)
        if not locked_hypotheses.empty
        else "No hypotheses passed the development-only lock criteria."
    )
    print()
    print("Pristine historical OOS for locked hypotheses:")
    print(
        historical_locked.to_string(index=False)
        if not historical_locked.empty
        else "No locked hypotheses matched the historical OOS sample."
    )
    print()
    print("Economic outcome evaluation for locked hypotheses:")
    print(
        economic[
            [
                "locked_hypothesis_index",
                "period",
                "sample_count",
                "tp_first_rate",
                "sl_first_rate",
                "unresolved_rate",
                "mean_reward_risk",
                "median_reward_risk",
                "tp_mean_reward_risk",
                "resolved_net_r",
                "expected_r_per_economic_sample",
                "expected_r_per_resolved_trade",
            ]
        ].to_string(index=False)
        if not economic.empty
        else "No economic evaluations available."
    )
    print()
    print(
        "Research artifacts: "
        "development candidates: data/research/xauusd_m30_empirical_development_hypothesis_candidates.csv; "
        "chronological folds: data/research/xauusd_m30_empirical_chronological_hypotheses.csv; "
        "locked hypotheses: data/research/xauusd_m30_empirical_locked_hypotheses.csv; "
        "pristine historical OOS: data/research/xauusd_m30_empirical_locked_hypotheses_oos.csv."
    )
    print()
    print(
        "Detailed empirical tables remain in data/research/*.csv; "
        "stability report: data/research/xauusd_m30_empirical_stability_report.csv; "
        "research hypotheses: data/research/xauusd_m30_empirical_research_hypotheses.csv."
    )


if __name__ == "__main__":
    main()
