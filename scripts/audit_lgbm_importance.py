"""Audit feature importance from the baseline LightGBM model."""

from pathlib import Path

import lightgbm as lgb
import pandas as pd

INPUT_PATH = Path(
    "data/processed/momentum_model_1r_10_binary.csv"
)

FEATURE_COLUMNS = [
    "candle_range",
    "candle_body",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_position",
    "is_bullish",
    "is_bearish",
    "true_range",
    "atr_14",
    "range_to_atr",
    "ema_5",
    "ema_20",
    "ema_50",
    "price_vs_ema_5",
    "price_vs_ema_20",
    "price_vs_ema_50",
    "ema_5_vs_20",
    "ema_20_vs_50",
    "ema_alignment",
    "previous_high_20",
    "previous_low_20",
    "previous_high_50",
    "previous_low_50",
    "distance_to_high_20",
    "distance_to_low_20",
    "distance_to_high_50",
    "distance_to_low_50",
    "breakout_above_20",
    "breakout_below_20",
    "breakout_above_50",
    "breakout_below_50",
    "return_3",
    "return_6",
    "return_12",
    "volume_ratio_20",
    "volume_change_1",
]


def main() -> None:
    print("=== LightGBM Feature Importance Audit ===")

    frame = pd.read_csv(INPUT_PATH)

    split_index = int(len(frame) * 0.80)

    train = frame.iloc[:split_index]

    X_train = train[FEATURE_COLUMNS]
    y_train = train["target"]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=20,
        random_state=42,
        n_jobs=2,
        verbosity=-1,
    )

    model.fit(X_train, y_train)

    importance_gain = model.booster_.feature_importance(
        importance_type="gain"
    )

    importance_split = model.booster_.feature_importance(
        importance_type="split"
    )

    result = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "gain": importance_gain,
            "split": importance_split,
        }
    )

    result["gain_pct"] = (
        result["gain"]
        / result["gain"].sum()
        * 100
    )

    result = result.sort_values(
        "gain",
        ascending=False,
    )

    print()
    print("Feature importance by gain:")
    print(
        result[
            ["feature", "gain", "gain_pct", "split"]
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print("Top 10 features by gain:")

    for _, row in result.head(10).iterrows():
        print(
            f"  {row['feature']:24s} "
            f"gain={row['gain_pct']:6.2f}% "
            f"splits={int(row['split'])}"
        )

    print()
    print("=== Importance audit complete ===")


if __name__ == "__main__":
    main()
