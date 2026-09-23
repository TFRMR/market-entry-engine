
def evaluate_pristine_oos(
    development: pd.DataFrame,
    historical_audit: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Fit the locked baseline on all development rows and evaluate once on pristine OOS."""
    development_binary = development.loc[
        development["label"].isin(["TP_FIRST", "SL_FIRST"])
    ].copy()
    historical_binary = historical_audit.loc[
        historical_audit["label"].isin(["TP_FIRST", "SL_FIRST"])
    ].copy()

    development_binary["target"] = (
        development_binary["label"] == "TP_FIRST"
    ).astype("int8")
    historical_binary["target"] = (
        historical_binary["label"] == "TP_FIRST"
    ).astype("int8")

    X_development_full = prepare_features(development_binary)
    X_oos_full = prepare_features(historical_binary)

    model_columns, constant_features, duplicate_feature_pairs = (
        select_model_features(X_development_full)
    )
    X_development = X_development_full[model_columns]
    X_oos = X_oos_full[model_columns].copy()

    for column in CATEGORICAL_COLUMNS:
        if column in model_columns:
            X_oos.loc[:, column] = X_oos[column].cat.set_categories(
                X_development[column].cat.categories
            )

    model = make_model()
    model.fit(
        X_development,
        development_binary["target"],
        categorical_feature=[
            column
            for column in CATEGORICAL_COLUMNS
            if column in model_columns
        ],
    )

    oos_probability = model.predict_proba(X_oos)[:, 1]
    oos_target = historical_binary["target"]
    development_base_rate = float(development_binary["target"].mean())
    baseline_probability = [development_base_rate] * len(oos_target)

    print()
    print("Pristine OOS evaluation (locked baseline):")
    print("  Training: full development set only")
    print(
        f"  Development period: "
        f"{development_binary['setup_timestamp'].min()} -> "
        f"{development_binary['setup_timestamp'].max()}"
    )
    print(
        f"  OOS period:          "
        f"{historical_binary['setup_timestamp'].min()} -> "
        f"{historical_binary['setup_timestamp'].max()}"
    )
    print(f"  Development rows:    {len(development_binary):,}")
    print(f"  OOS rows:             {len(historical_binary):,}")
    print(f"  Model features:      {len(model_columns):,}")
    print(f"  OOS ROC-AUC:         {roc_auc_score(oos_target, oos_probability):.4f}")
    print(f"  OOS log loss:        {log_loss(oos_target, oos_probability):.4f}")
    print(
        f"  Baseline log loss:   "
        f"{log_loss(oos_target, baseline_probability):.4f}"
    )

    if constant_features:
        print()
        print("OOS-fit constant features dropped:")
        for column in constant_features:
            print(f"  {column}")

    if duplicate_feature_pairs:
        print()
        print("OOS-fit duplicate feature aliases dropped:")
        for duplicate, kept in duplicate_feature_pairs:
            print(f"  {duplicate} == {kept} (kept {kept})")


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
        categorical_feature=[
            column for column in CATEGORICAL_COLUMNS if column in model_columns
        ],
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

    print_temporal_distribution(dataset)
    print_feature_drift(dataset, model_columns)
    print_feature_outcome_drift(dataset, model_columns)

    evaluate_chronological_folds(dataset)
    evaluate_pristine_oos(development, historical_audit, list(FEATURE_COLUMNS))

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
