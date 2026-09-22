"""Train a chronological LightGBM baseline on canonical structural setup labels."""

from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from market_engine.data import load_mt5_csv
from market_engine.entry import build_setup_candidates
from market_engine.features import build_structural_features
from market_engine.holdout import HISTORICAL_AUDIT_CUTOFF, split_historical_boundary
from market_engine.labels import build_setup_label_dataset
from market_engine.structure import build_structural_sequence, process_structural_candles


INPUT_PATH = Path(
    "data/raw/XAUUSDc_M30_202409012200_202609182030.csv"
)

SPREAD_PRICE = 0.0
HORIZON = 10

FEATURE_COLUMNS = (
    "structure_bullish_bos",
    "structure_bearish_bos",
    "structure_swing_high_valid",
    "structure_swing_low_valid",
    "structure_internal_bos",
    "structure_external_bos",
    "structure_internal_swing",
    "structure_external_swing",
    "structure_last_valid_high",
    "structure_last_valid_low",
    "structure_hh",
    "structure_hl",
    "structure_lh",
    "structure_ll",
    "structure_event_count",
    "structure_last_event_index",
    "structure_last_event_age",
    "structure_last_event_type",
    "structure_last_event_direction",
    "structure_last_event_scope",
    "structure_previous_event_index",
    "structure_previous_event_age",
    "structure_previous_event_type",
    "structure_previous_event_direction",
    "structure_previous_event_scope",
    "structure_events_since_last_bos",
    "structure_events_since_last_swing",
    "structure_bars_since_last_bos",
    "structure_bars_since_last_swing",
    "structure_direction",
    "structure_distance_to_high",
    "structure_distance_to_low",
    "structure_historical_distance_to_high",
    "structure_historical_distance_to_low",
    "trend_regime",
    "trend_transition",
    "trend_transition_direction",
    "range_state",
    "range_high",
    "range_low",
    "range_width",
    "range_position",
    "range_position_zone",
    "liquidity_high",
    "liquidity_low",
    "liquidity_high_present",
    "liquidity_low_present",
    "distance_to_liquidity_high",
    "distance_to_liquidity_low",
    "liquidity_high_sweep",
    "liquidity_low_sweep",
    "liquidity_sweep",
    "liquidity_sweep_direction",
    "liquidity_sweep_size",
    "ob_bullish_present",
    "ob_bullish_size",
    "ob_bullish_age_bars",
    "ob_bullish_distance",
    "ob_bullish_contains_price",
    "ob_bullish_relative_position",
    "ob_bearish_present",
    "ob_bearish_size",
    "ob_bearish_age_bars",
    "ob_bearish_distance",
    "ob_bearish_contains_price",
    "ob_bearish_relative_position",
)


CATEGORICAL_COLUMNS = (
    "structure_last_event_type",
    "structure_last_event_direction",
    "structure_last_event_scope",
    "structure_previous_event_type",
    "structure_previous_event_direction",
    "structure_previous_event_scope",
    "structure_direction",
    "trend_regime",
    "trend_transition_direction",
    "range_state",
    "range_position_zone",
    "liquidity_sweep_direction",
)


def select_model_features(
    train_frame: pd.DataFrame,
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Select non-constant, non-duplicate features using training data only."""

    constant_features = [
        column
        for column in train_frame.columns
        if train_frame[column].nunique(dropna=False) <= 1
    ]

    selected = [
        column
        for column in train_frame.columns
        if column not in constant_features
    ]

    duplicate_feature_pairs: list[tuple[str, str]] = []
    unique_columns: list[str] = []

    for column in selected:
        duplicate_of = next(
            (
                other
                for other in unique_columns
                if train_frame[column].equals(train_frame[other])
            ),
            None,
        )
        if duplicate_of is not None:
            duplicate_feature_pairs.append((column, duplicate_of))
        else:
            unique_columns.append(column)

    return unique_columns, constant_features, duplicate_feature_pairs


def make_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        random_state=42,
        n_jobs=2,
        verbosity=-1,
    )


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[list(FEATURE_COLUMNS)].copy()

    for column in CATEGORICAL_COLUMNS:
        result[column] = result[column].astype("category")

    return result


def print_temporal_distribution(dataset: pd.DataFrame, bucket_count: int = 4) -> None:
    """Describe outcome distribution across chronological buckets."""

    print()
    print("Temporal outcome distribution:")
    buckets = pd.qcut(
        dataset["setup_timestamp"].rank(method="first"),
        q=bucket_count,
        labels=False,
    )
    for bucket, group in dataset.groupby(buckets, sort=True):
        tp_rate = float(group["target"].mean())
        median_rr = float(group["reward_risk"].median())
        print(
            f"  Bucket {int(bucket) + 1}: "
            f"rows={len(group):,} "
            f"period={group['setup_timestamp'].min()} -> {group['setup_timestamp'].max()} "
            f"TP_FIRST={tp_rate:.4f} "
            f"median_RR={median_rr:.4f}"
        )


def print_feature_drift(
    dataset: pd.DataFrame,
    model_columns: list[str],
    bucket_count: int = 4,
    top_n: int = 15,
) -> None:
    """Describe simple early-vs-late feature distribution drift."""

    buckets = pd.qcut(
        dataset["setup_timestamp"].rank(method="first"),
        q=bucket_count,
        labels=False,
    )
    grouped = {
        int(bucket): group
        for bucket, group in dataset.groupby(buckets, sort=True)
    }
    early = grouped[0]
    late = grouped[bucket_count - 1]

    rows: list[dict[str, object]] = []

    for column in model_columns:
        early_values = early[column]
        late_values = late[column]
        missing_delta = abs(
            float(early_values.isna().mean())
            - float(late_values.isna().mean())
        )

        if column in CATEGORICAL_COLUMNS:
            early_dist = early_values.value_counts(normalize=True, dropna=False)
            late_dist = late_values.value_counts(normalize=True, dropna=False)
            categories = early_dist.index.union(late_dist.index)
            early_dist = early_dist.reindex(categories, fill_value=0.0)
            late_dist = late_dist.reindex(categories, fill_value=0.0)
            drift = float((early_dist - late_dist).abs().max())
            metric = "max_category_delta"
        else:
            early_numeric = pd.to_numeric(early_values, errors="coerce").dropna()
            late_numeric = pd.to_numeric(late_values, errors="coerce").dropna()
            if early_numeric.empty or late_numeric.empty:
                drift = 0.0
            else:
                pooled_iqr = float(
                    pd.concat([early_numeric, late_numeric]).quantile(0.75)
                    - pd.concat([early_numeric, late_numeric]).quantile(0.25)
                )
                median_shift = abs(
                    float(early_numeric.median()) - float(late_numeric.median())
                )
                drift = (
                    median_shift / pooled_iqr
                    if pooled_iqr > 0
                    else median_shift
                )
            metric = "median_shift_over_pooled_iqr"

        rows.append(
            {
                "feature": column,
                "type": "categorical" if column in CATEGORICAL_COLUMNS else "numeric",
                "drift": drift,
                "missing_delta": missing_delta,
                "metric": metric,
            }
        )

    report = pd.DataFrame(rows).sort_values(
        ["drift", "missing_delta"],
        ascending=False,
    )

    print()
    print("Feature distribution drift (earliest vs latest bucket):")
    print(
        f"  Early period: {early['setup_timestamp'].min()} -> "
        f"{early['setup_timestamp'].max()}"
    )
    print(
        f"  Late period:  {late['setup_timestamp'].min()} -> "
        f"{late['setup_timestamp'].max()}"
    )
    print(
        report.head(top_n).to_string(
            index=False,
            formatters={
                "drift": "{:.4f}".format,
                "missing_delta": "{:.4f}".format,
            },
        )
    )


def evaluate_chronological_folds(
    dataset: pd.DataFrame,
    fold_count: int = 3,
) -> None:
    """Evaluate fixed baseline across chronological expanding folds."""

    print()
    print("Chronological fold evaluation:")
    total = len(dataset)
    for fold in range(1, fold_count + 1):
        train_end = int(total * (0.50 + 0.10 * (fold - 1)))
        test_end = int(total * (0.70 + 0.10 * (fold - 1)))
        train = dataset.iloc[:train_end]
        test = dataset.iloc[train_end:test_end]

        X_train_full = prepare_features(train)
        X_test_full = prepare_features(test)
        model_columns, _, _ = select_model_features(X_train_full)
        X_train = X_train_full[model_columns]
        X_test = X_test_full.loc[:, model_columns].copy()

        for column in CATEGORICAL_COLUMNS:
            if column in model_columns:
                X_test.loc[:, column] = X_test[column].cat.set_categories(
                    X_train[column].cat.categories
                )

        model = make_model()
        model.fit(
            X_train,
            train["target"],
            categorical_feature=[
                column for column in CATEGORICAL_COLUMNS
                if column in model_columns
            ],
        )
        probability = model.predict_proba(X_test)[:, 1]
        base_rate = float(train["target"].mean())
        baseline_probability = [base_rate] * len(test)

        print(
            f"  Fold {fold}: "
            f"train={len(train):,} test={len(test):,} "
            f"test_period={test['setup_timestamp'].min()} -> {test['setup_timestamp'].max()} "
            f"ROC-AUC={roc_auc_score(test['target'], probability):.4f} "
            f"log_loss={log_loss(test['target'], probability):.4f} "
            f"baseline={log_loss(test['target'], baseline_probability):.4f}"
        )


def main() -> None:
    print("=== Canonical Structural LightGBM Baseline ===")

    frame = load_mt5_csv(INPUT_PATH)
    candidates = build_setup_candidates(frame)
    structural = build_structural_sequence(frame)
    swings, events = process_structural_candles(structural)
    feature_frame = build_structural_features(frame)

    labeled = build_setup_label_dataset(
        candidates=candidates,
        swings=swings,
        frame=frame,
        feature_frame=feature_frame,
        spread_price=SPREAD_PRICE,
        feature_columns=FEATURE_COLUMNS,
        horizon=HORIZON,
        events=events,
    )

    development, historical_audit, purged = split_historical_boundary(
        labeled,
        frame,
        HORIZON,
        HISTORICAL_AUDIT_CUTOFF,
    )

    dataset = development.loc[
        development["label"].isin(["TP_FIRST", "SL_FIRST"])
    ].copy()

    dataset["target"] = (dataset["label"] == "TP_FIRST").astype("int8")

    X = prepare_features(dataset)
    y = dataset["target"]

    split_index = int(len(dataset) * 0.80)

    train = dataset.iloc[:split_index]
    test = dataset.iloc[split_index:]

    X_train_full = X.iloc[:split_index]
    X_test_full = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model_columns, constant_features, duplicate_feature_pairs = (
        select_model_features(X_train_full)
    )
    X_train = X_train_full[model_columns]
    X_test = X_test_full[model_columns]

    historical_binary = historical_audit.loc[
        historical_audit["label"].isin(["TP_FIRST", "SL_FIRST"])
    ].copy()
    historical_binary["target"] = (
        historical_binary["label"] == "TP_FIRST"
    ).astype("int8")
    X_historical = prepare_features(historical_binary)[model_columns]

    for column in CATEGORICAL_COLUMNS:
        if column in model_columns:
            X_historical[column] = X_historical[column].cat.set_categories(
                X_train[column].cat.categories
            )
    y_historical = historical_binary["target"]

    print()
    print("Dataset:")
    print(f"  Raw candles:          {len(frame):,}")
    print(f"  Setup candidates:     {len(candidates):,}")
    print(f"  Development labels:   {len(development):,}")
    print(f"  Historical audit:     {len(historical_audit):,}")
    print(f"  Purged boundary:      {len(purged):,}")
    print(f"  Binary model rows:    {len(dataset):,}")
    print(f"  Raw features:         {len(FEATURE_COLUMNS):,}")
    print(f"  Model features:       {len(model_columns):,}")
    print(f"  Constant dropped:     {len(constant_features):,}")
    print(f"  Duplicate dropped:    {len(duplicate_feature_pairs):,}")

    print()
    print("Model split:")
    print(f"  Train: {len(train):,}")
    print(f"  Test:  {len(test):,}")
    print(f"  Train period: {train['setup_timestamp'].min()} -> {train['setup_timestamp'].max()}")
    print(f"  Test period:  {test['setup_timestamp'].min()} -> {test['setup_timestamp'].max()}")

    model = make_model()
    model.fit(
        X_train,
        y_train,
        categorical_feature=[column for column in CATEGORICAL_COLUMNS if column in model_columns],
    )

    train_probability = model.predict_proba(X_train)[:, 1]
    test_probability = model.predict_proba(X_test)[:, 1]

    train_prediction = (train_probability >= 0.50).astype("int8")
    test_prediction = (test_probability >= 0.50).astype("int8")

    train_base_rate = float(y_train.mean())
    baseline_probability = [train_base_rate] * len(y_test)

    print()
    print("Metrics:")
    print(f"  Train ROC-AUC:    {roc_auc_score(y_train, train_probability):.4f}")
    print(f"  Test ROC-AUC:     {roc_auc_score(y_test, test_probability):.4f}")
    print(f"  Train log loss:   {log_loss(y_train, train_probability):.4f}")
    print(f"  Test log loss:    {log_loss(y_test, test_probability):.4f}")
    print(f"  Baseline log loss: {log_loss(y_test, baseline_probability):.4f}")
    print(f"  Train accuracy:   {accuracy_score(y_train, train_prediction):.4f}")
    print(f"  Test accuracy:    {accuracy_score(y_test, test_prediction):.4f}")

    historical_probability = model.predict_proba(X_historical)[:, 1]
    historical_base_rate = float(y_historical.mean())
    historical_baseline_probability = [
        historical_base_rate
    ] * len(y_historical)

    print()
    print("Historical-boundary evaluation (not pristine OOS):")
    print(f"  Rows:              {len(historical_binary):,}")
    print(f"  Period:            {historical_binary['setup_timestamp'].min()} -> {historical_binary['setup_timestamp'].max()}")
    print(f"  ROC-AUC:            {roc_auc_score(y_historical, historical_probability):.4f}")
    print(f"  Log loss:           {log_loss(y_historical, historical_probability):.4f}")
    print(f"  Baseline log loss:  {log_loss(y_historical, historical_baseline_probability):.4f}")

    print_temporal_distribution(dataset)
    print_feature_drift(dataset, model_columns)

    evaluate_chronological_folds(dataset)

    print()
    print("Test labels:")
    print(y_test.value_counts().sort_index().to_string())

    print()
    print("Top feature importance:")
    importance = pd.Series(
        model.feature_importances_,
        index=model_columns,
    ).sort_values(ascending=False)

    print(importance.head(20).to_string())

    if constant_features:
        print()
        print("Constant features dropped:")
        for column in constant_features:
            print(f"  {column}")

    if duplicate_feature_pairs:
        print()
        print("Duplicate feature aliases dropped:")
        for duplicate, kept in duplicate_feature_pairs:
            print(f"  {duplicate} == {kept} (kept {kept})")

    print()
    print("=== Structural LightGBM baseline complete ===")


if __name__ == "__main__":
    main()
