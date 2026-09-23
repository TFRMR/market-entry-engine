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
    "structure_bullish_bos", "structure_bearish_bos", "structure_bullish_choch",
    "structure_bearish_choch", "structure_swing_high_valid", "structure_swing_low_valid",
    "structure_internal_bos", "structure_external_bos", "structure_internal_swing",
    "structure_external_swing", "structure_last_valid_high", "structure_last_valid_low",
    "structure_hh", "structure_hl", "structure_lh", "structure_ll", "structure_event_count",
    "structure_last_event_index", "structure_last_event_age", "structure_last_event_type",
    "structure_last_event_direction", "structure_last_event_scope", "structure_previous_event_index",
    "structure_previous_event_age", "structure_previous_event_type", "structure_previous_event_direction",
    "structure_previous_event_scope", "structure_events_since_last_bos", "structure_events_since_last_swing",
    "structure_bars_since_last_bos", "structure_bars_since_last_swing", "structure_direction",
    "structure_distance_to_high", "structure_distance_to_low", "structure_historical_distance_to_high",
    "structure_historical_distance_to_low", "trend_regime", "trend_transition", "trend_transition_direction",
    "range_state", "range_high", "range_low", "range_width", "range_position", "range_position_zone",
    "liquidity_high", "liquidity_low", "liquidity_high_present", "liquidity_low_present",
    "distance_to_liquidity_high", "distance_to_liquidity_low", "liquidity_high_sweep", "liquidity_low_sweep",
    "liquidity_sweep", "liquidity_sweep_direction", "liquidity_sweep_size", "ob_bullish_present",
    "ob_bullish_size", "ob_bullish_age_bars", "ob_bullish_distance", "ob_bullish_contains_price",
    "ob_bullish_relative_position", "ob_bearish_present", "ob_bearish_size", "ob_bearish_age_bars",
    "ob_bearish_distance", "ob_bearish_contains_price", "ob_bearish_relative_position",
)

CATEGORICAL_COLUMNS = (
    "structure_last_event_type", "structure_last_event_direction", "structure_last_event_scope",
    "structure_previous_event_type", "structure_previous_event_direction", "structure_previous_event_scope",
    "structure_direction", "trend_regime", "trend_transition_direction", "range_state",
    "range_position_zone", "liquidity_sweep_direction",
)


def select_model_features(train_frame: pd.DataFrame) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    constant_features = [column for column in train_frame.columns if train_frame[column].nunique(dropna=False) <= 1]
    selected = [column for column in train_frame.columns if column not in constant_features]
    duplicate_feature_pairs: list[tuple[str, str]] = []
    unique_columns: list[str] = []
    for column in selected:
        duplicate_of = next((other for other in unique_columns if train_frame[column].equals(train_frame[other])), None)
        if duplicate_of is not None:
            duplicate_feature_pairs.append((column, duplicate_of))
        else:
            unique_columns.append(column)
    return unique_columns, constant_features, duplicate_feature_pairs


def make_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(objective="binary", n_estimators=100, learning_rate=0.05, num_leaves=15, min_child_samples=20, random_state=42, n_jobs=2, verbosity=-1)


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[list(FEATURE_COLUMNS)].copy()
    for column in CATEGORICAL_COLUMNS:
        result[column] = result[column].astype("category")
    return result


def print_temporal_distribution(dataset: pd.DataFrame, bucket_count: int = 4) -> None:
    print()
    print("Temporal outcome distribution:")
    buckets = pd.qcut(dataset["setup_timestamp"].rank(method="first"), q=bucket_count, labels=False)
    for bucket, group in dataset.groupby(buckets, sort=True):
        print(f"  Bucket {int(bucket) + 1}: rows={len(group):,} period={group['setup_timestamp'].min()} -> {group['setup_timestamp'].max()} TP_FIRST={float(group['target'].mean()):.4f} median_RR={float(group['reward_risk'].median()):.4f}")


def print_feature_drift(dataset: pd.DataFrame, model_columns: list[str], bucket_count: int = 4, top_n: int = 15) -> None:
    buckets = pd.qcut(dataset["setup_timestamp"].rank(method="first"), q=bucket_count, labels=False)
    grouped = {int(bucket): group for bucket, group in dataset.groupby(buckets, sort=True)}
    early, late = grouped[0], grouped[bucket_count - 1]
    rows = []
    for column in model_columns:
        early_values, late_values = early[column], late[column]
        missing_delta = abs(float(early_values.isna().mean()) - float(late_values.isna().mean()))
        if column in CATEGORICAL_COLUMNS:
            early_dist = early_values.value_counts(normalize=True, dropna=False)
            late_dist = late_values.value_counts(normalize=True, dropna=False)
            categories = early_dist.index.union(late_dist.index)
            drift = float((early_dist.reindex(categories, fill_value=0.0) - late_dist.reindex(categories, fill_value=0.0)).abs().max())
            metric = "max_category_delta"
        else:
            early_numeric = pd.to_numeric(early_values, errors="coerce").dropna()
            late_numeric = pd.to_numeric(late_values, errors="coerce").dropna()
            if early_numeric.empty or late_numeric.empty:
                drift = 0.0
            else:
                combined = pd.concat([early_numeric, late_numeric])
                pooled_iqr = float(combined.quantile(0.75) - combined.quantile(0.25))
                median_shift = abs(float(early_numeric.median()) - float(late_numeric.median()))
                drift = median_shift / pooled_iqr if pooled_iqr > 0 else median_shift
            metric = "median_shift_over_pooled_iqr"
        rows.append({"feature": column, "type": "categorical" if column in CATEGORICAL_COLUMNS else "numeric", "drift": drift, "missing_delta": missing_delta, "metric": metric})
    report = pd.DataFrame(rows).sort_values(["drift", "missing_delta"], ascending=False)
    print()
    print("Feature distribution drift (earliest vs latest bucket):")
    print(f"  Early period: {early['setup_timestamp'].min()} -> {early['setup_timestamp'].max()}")
    print(f"  Late period:  {late['setup_timestamp'].min()} -> {late['setup_timestamp'].max()}")
    print(report.head(top_n).to_string(index=False, formatters={"drift": "{:.4f}".format, "missing_delta": "{:.4f}".format}))


def print_feature_outcome_drift(dataset: pd.DataFrame, model_columns: list[str], bucket_count: int = 4, top_n: int = 15) -> None:
    buckets = pd.qcut(dataset["setup_timestamp"].rank(method="first"), q=bucket_count, labels=False)
    grouped = {int(bucket): group for bucket, group in dataset.groupby(buckets, sort=True)}
    early, late = grouped[0], grouped[bucket_count - 1]
    rows = []
    for column in model_columns:
        early_values, late_values = early[column], late[column]
        if column in CATEGORICAL_COLUMNS:
            early_frame = pd.DataFrame({"value": early_values.astype("object"), "target": early["target"]})
            late_frame = pd.DataFrame({"value": late_values.astype("object"), "target": late["target"]})
            early_rates = early_frame.groupby("value", dropna=False)["target"].mean()
            late_rates = late_frame.groupby("value", dropna=False)["target"].mean()
            categories = early_rates.index.union(late_rates.index)
            relationship_drift = float((early_rates.reindex(categories) - late_rates.reindex(categories)).abs().max())
            metric = "max_category_target_rate_delta"
        else:
            early_numeric, late_numeric = pd.to_numeric(early_values, errors="coerce"), pd.to_numeric(late_values, errors="coerce")
            early_mask, late_mask = early_numeric.notna(), late_numeric.notna()
            if early_mask.sum() < 2 or late_mask.sum() < 2 or early.loc[early_mask, "target"].nunique() < 2 or late.loc[late_mask, "target"].nunique() < 2:
                relationship_drift = 0.0
            else:
                relationship_drift = abs(float(roc_auc_score(early.loc[early_mask, "target"], early_numeric.loc[early_mask])) - float(roc_auc_score(late.loc[late_mask, "target"], late_numeric.loc[late_mask])))
            metric = "absolute_auc_delta"
        rows.append({"feature": column, "type": "categorical" if column in CATEGORICAL_COLUMNS else "numeric", "relationship_drift": relationship_drift, "metric": metric})
    report = pd.DataFrame(rows).sort_values("relationship_drift", ascending=False)
    print()
    print("Feature -> outcome relationship drift (earliest vs latest bucket):")
    print(f"  Early period: {early['setup_timestamp'].min()} -> {early['setup_timestamp'].max()}")
    print(f"  Late period:  {late['setup_timestamp'].min()} -> {late['setup_timestamp'].max()}")
    print(report.head(top_n).to_string(index=False, formatters={"relationship_drift": "{:.4f}".format}))


def evaluate_chronological_folds(dataset: pd.DataFrame, fold_count: int = 3) -> None:
    print()
    print("Chronological fold evaluation:")
    total = len(dataset)
    for fold in range(1, fold_count + 1):
        train_end, test_end = int(total * (0.50 + 0.10 * (fold - 1))), int(total * (0.70 + 0.10 * (fold - 1)))
        train, test = dataset.iloc[:train_end], dataset.iloc[train_end:test_end]
        X_train_full, X_test_full = prepare_features(train), prepare_features(test)
        model_columns, _, _ = select_model_features(X_train_full)
        X_train, X_test = X_train_full[model_columns], X_test_full.loc[:, model_columns].copy()
        for column in CATEGORICAL_COLUMNS:
            if column in model_columns:
                X_test.loc[:, column] = X_test[column].cat.set_categories(X_train[column].cat.categories)
        model = make_model()
        model.fit(X_train, train["target"], categorical_feature=[column for column in CATEGORICAL_COLUMNS if column in model_columns])
        probability = model.predict_proba(X_test)[:, 1]
        base_rate = float(train["target"].mean())
        print(f"  Fold {fold}: train={len(train):,} test={len(test):,} test_period={test['setup_timestamp'].min()} -> {test['setup_timestamp'].max()} ROC-AUC={roc_auc_score(test['target'], probability):.4f} log_loss={log_loss(test['target'], probability):.4f} baseline={log_loss(test['target'], [base_rate] * len(test)):.4f}")


def evaluate_pristine_oos(development: pd.DataFrame, historical_audit: pd.DataFrame) -> None:
    development_binary = development.loc[development["label"].isin(["TP_FIRST", "SL_FIRST"])].copy()
    oos = historical_audit.loc[historical_audit["label"].isin(["TP_FIRST", "SL_FIRST"])].copy()
    development_binary["target"] = (development_binary["label"] == "TP_FIRST").astype("int8")
    oos["target"] = (oos["label"] == "TP_FIRST").astype("int8")
    X_development_full, X_oos_full = prepare_features(development_binary), prepare_features(oos)
    model_columns, _, _ = select_model_features(X_development_full)
    X_development, X_oos = X_development_full[model_columns], X_oos_full[model_columns].copy()
    for column in CATEGORICAL_COLUMNS:
        if column in model_columns:
            X_oos.loc[:, column] = X_oos[column].cat.set_categories(X_development[column].cat.categories)
    model = make_model()
    model.fit(X_development, development_binary["target"], categorical_feature=[column for column in CATEGORICAL_COLUMNS if column in model_columns])
    probability = model.predict_proba(X_oos)[:, 1]
    base_rate = float(development_binary["target"].mean())
    print()
    print("Pristine OOS evaluation (locked baseline):")
    print("  Training: full development set only")
    print(f"  Development period: {development_binary['setup_timestamp'].min()} -> {development_binary['setup_timestamp'].max()}")
    print(f"  OOS period:         {oos['setup_timestamp'].min()} -> {oos['setup_timestamp'].max()}")
    print(f"  Development rows:   {len(development_binary):,}")
    print(f"  OOS rows:            {len(oos):,}")
    print(f"  Model features:      {len(model_columns):,}")
    print(f"  OOS ROC-AUC:         {roc_auc_score(oos['target'], probability):.4f}")
    print(f"  OOS log loss:        {log_loss(oos['target'], probability):.4f}")
    print(f"  Baseline log loss:   {log_loss(oos['target'], [base_rate] * len(oos)):.4f}")



FEATURE_GROUPS = {
    "structure": [column for column in FEATURE_COLUMNS if column.startswith("structure_") or column.startswith("trend_")],
    "range": [column for column in FEATURE_COLUMNS if column.startswith("range_")],
    "liquidity": [column for column in FEATURE_COLUMNS if column.startswith("liquidity_") or column.startswith("distance_to_liquidity_")],
    "order_block": [column for column in FEATURE_COLUMNS if column.startswith("ob_")],
}


def evaluate_oos_feature_ablation(development: pd.DataFrame, historical_audit: pd.DataFrame) -> None:
    development_binary = development.loc[development["label"].isin(["TP_FIRST", "SL_FIRST"])].copy()
    oos = historical_audit.loc[historical_audit["label"].isin(["TP_FIRST", "SL_FIRST"])].copy()
    development_binary["target"] = (development_binary["label"] == "TP_FIRST").astype("int8")
    oos["target"] = (oos["label"] == "TP_FIRST").astype("int8")
    X_development_full = prepare_features(development_binary)
    X_oos_full = prepare_features(oos)
    model_columns, _, _ = select_model_features(X_development_full)
    base_rate = float(development_binary["target"].mean())
    baseline_loss = log_loss(oos["target"], [base_rate] * len(oos))
    print()
    print("Pristine OOS feature-group ablation (locked baseline):")
    print("  Remove one feature family, retrain on full development, evaluate on the same OOS.")
    for removed_group, removed_columns in FEATURE_GROUPS.items():
        columns = [column for column in model_columns if column not in removed_columns]
        X_development = X_development_full[columns]
        X_oos = X_oos_full[columns].copy()
        for column in CATEGORICAL_COLUMNS:
            if column in columns:
                X_oos.loc[:, column] = X_oos[column].cat.set_categories(X_development[column].cat.categories)
        model = make_model()
        model.fit(X_development, development_binary["target"], categorical_feature=[column for column in CATEGORICAL_COLUMNS if column in columns])
        probability = model.predict_proba(X_oos)[:, 1]
        oos_loss = log_loss(oos["target"], probability)
        print(f"  Remove {removed_group:<10} features={len(columns):>2} ROC-AUC={roc_auc_score(oos['target'], probability):.4f} log_loss={oos_loss:.4f} vs_baseline={oos_loss - baseline_loss:+.4f}")


def evaluate_oos_individual_ablation(development: pd.DataFrame, historical_audit: pd.DataFrame) -> None:
    development_binary = development.loc[development["label"].isin(["TP_FIRST", "SL_FIRST"])].copy()
    oos = historical_audit.loc[historical_audit["label"].isin(["TP_FIRST", "SL_FIRST"])].copy()
    development_binary["target"] = (development_binary["label"] == "TP_FIRST").astype("int8")
    oos["target"] = (oos["label"] == "TP_FIRST").astype("int8")
    X_development_full = prepare_features(development_binary)
    X_oos_full = prepare_features(oos)
    model_columns, _, _ = select_model_features(X_development_full)
    base_rate = float(development_binary["target"].mean())
    baseline_loss = log_loss(oos["target"], [base_rate] * len(oos))
    groups = {
        "structure": FEATURE_GROUPS["structure"],
        "liquidity": FEATURE_GROUPS["liquidity"],
    }
    print()
    print("Pristine OOS individual-feature ablation (structure + liquidity):")
    print("  Remove one feature at a time, retrain on full development, evaluate on the same OOS.")
    for group_name, group_columns in groups.items():
        print(f"  {group_name}:")
        for removed_column in group_columns:
            columns = [column for column in model_columns if column != removed_column]
            X_development = X_development_full[columns]
            X_oos = X_oos_full[columns].copy()
            for column in CATEGORICAL_COLUMNS:
                if column in columns:
                    X_oos.loc[:, column] = X_oos[column].cat.set_categories(X_development[column].cat.categories)
            model = make_model()
            model.fit(
                X_development,
                development_binary["target"],
                categorical_feature=[column for column in CATEGORICAL_COLUMNS if column in columns],
            )
            probability = model.predict_proba(X_oos)[:, 1]
            oos_auc = roc_auc_score(oos["target"], probability)
            oos_loss = log_loss(oos["target"], probability)
            print(f"    Remove {removed_column:<40} ROC-AUC={oos_auc:.4f} log_loss={oos_loss:.4f} vs_baseline={oos_loss - baseline_loss:+.4f}")


def main() -> None:
    print("=== Canonical Structural LightGBM Baseline ===")
    frame = load_mt5_csv(INPUT_PATH)
    candidates = build_setup_candidates(frame)
    structural = build_structural_sequence(frame)
    swings, events = process_structural_candles(structural)
    feature_frame = build_structural_features(frame)
    labeled = build_setup_label_dataset(candidates=candidates, swings=swings, frame=frame, feature_frame=feature_frame, spread_price=SPREAD_PRICE, feature_columns=FEATURE_COLUMNS, horizon=HORIZON, events=events)
    development, historical_audit, purged = split_historical_boundary(labeled, frame, HORIZON, HISTORICAL_AUDIT_CUTOFF)
    dataset = development.loc[development["label"].isin(["TP_FIRST", "SL_FIRST"])].copy()
    dataset["target"] = (dataset["label"] == "TP_FIRST").astype("int8")
    X, y = prepare_features(dataset), dataset["target"]
    split_index = int(len(dataset) * 0.80)
    train, test = dataset.iloc[:split_index], dataset.iloc[split_index:]
    X_train_full, X_test_full, y_train, y_test = X.iloc[:split_index], X.iloc[split_index:], y.iloc[:split_index], y.iloc[split_index:]
    model_columns, constant_features, duplicate_feature_pairs = select_model_features(X_train_full)
    X_train, X_test = X_train_full[model_columns], X_test_full[model_columns]
    historical_binary = historical_audit.loc[historical_audit["label"].isin(["TP_FIRST", "SL_FIRST"])].copy()
    historical_binary["target"] = (historical_binary["label"] == "TP_FIRST").astype("int8")
    X_historical = prepare_features(historical_binary)[model_columns]
    for column in CATEGORICAL_COLUMNS:
        if column in model_columns:
            X_historical[column] = X_historical[column].cat.set_categories(X_train[column].cat.categories)
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
    model.fit(X_train, y_train, categorical_feature=[column for column in CATEGORICAL_COLUMNS if column in model_columns])
    train_probability, test_probability = model.predict_proba(X_train)[:, 1], model.predict_proba(X_test)[:, 1]
    train_prediction, test_prediction = (train_probability >= 0.50).astype("int8"), (test_probability >= 0.50).astype("int8")
    train_base_rate = float(y_train.mean())
    print()
    print("Metrics:")
    print(f"  Train ROC-AUC:    {roc_auc_score(y_train, train_probability):.4f}")
    print(f"  Test ROC-AUC:     {roc_auc_score(y_test, test_probability):.4f}")
    print(f"  Train log loss:   {log_loss(y_train, train_probability):.4f}")
    print(f"  Test log loss:    {log_loss(y_test, test_probability):.4f}")
    print(f"  Baseline log loss: {log_loss(y_test, [train_base_rate] * len(y_test)):.4f}")
    print(f"  Train accuracy:   {accuracy_score(y_train, train_prediction):.4f}")
    print(f"  Test accuracy:    {accuracy_score(y_test, test_prediction):.4f}")
    historical_probability = model.predict_proba(X_historical)[:, 1]
    historical_base_rate = float(y_historical.mean())
    print()
    print("Historical-boundary evaluation (not pristine OOS):")
    print(f"  Rows:              {len(historical_binary):,}")
    print(f"  Period:            {historical_binary['setup_timestamp'].min()} -> {historical_binary['setup_timestamp'].max()}")
    print(f"  ROC-AUC:            {roc_auc_score(y_historical, historical_probability):.4f}")
    print(f"  Log loss:           {log_loss(y_historical, historical_probability):.4f}")
    print(f"  Baseline log loss:  {log_loss(y_historical, [historical_base_rate] * len(y_historical)):.4f}")
    print_temporal_distribution(dataset)
    print_feature_drift(dataset, model_columns)
    print_feature_outcome_drift(dataset, model_columns)
    evaluate_chronological_folds(dataset)
    evaluate_pristine_oos(development, historical_audit)
    evaluate_oos_feature_ablation(development, historical_audit)
    print()
    print("Test labels:")
    print(y_test.value_counts().sort_index().to_string())
    print()
    print("Top feature importance:")
    print(pd.Series(model.feature_importances_, index=model_columns).sort_values(ascending=False).head(20).to_string())
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
