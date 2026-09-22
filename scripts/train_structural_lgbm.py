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
    "structure_bullish_choch",
    "structure_bearish_choch",
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
        categorical_feature=list(CATEGORICAL_COLUMNS),
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
