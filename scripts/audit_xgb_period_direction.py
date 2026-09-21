"""Audit XGBoost OOS performance by chronological period and direction."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

INPUT = Path("data/processed/momentum_model_1r_10_binary.csv")

FEATURES = [
    "candle_range",
    "candle_body",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_position",
    "true_range",
    "atr_14",
    "range_to_atr",
]

TARGET = "target"


df = pd.read_csv(INPUT, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

required = {
    "timestamp",
    "is_bullish",
    "is_bearish",
    TARGET,
    *FEATURES,
}

missing = sorted(required - set(df.columns))

if missing:
    raise SystemExit(f"Missing columns: {missing}")

if not df["timestamp"].is_monotonic_increasing:
    raise SystemExit("Timestamp order is not chronological.")

if df["timestamp"].duplicated().any():
    raise SystemExit("Duplicate timestamps found.")

if (df["is_bullish"] == df["is_bearish"]).any():
    raise SystemExit("Invalid bullish/bearish direction.")

df["direction"] = np.where(
    df["is_bullish"],
    "LONG",
    "SHORT",
)


# ---------------------------------------------------------------------------
# Four chronological test periods
# ---------------------------------------------------------------------------

n = len(df)

df["period"] = pd.qcut(
    np.arange(n),
    q=4,
    labels=["P1", "P2", "P3", "P4"],
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_model():
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=20,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        n_jobs=2,
        eval_metric="logloss",
    )


# ---------------------------------------------------------------------------
# Expanding walk-forward
# ---------------------------------------------------------------------------

test_size = n // 5

# Match the existing 4-fold audit:
# train 390 -> test 390
# train 780 -> test 390
# train 1170 -> test 390
# train 1560 -> test 390
#
# Since n=1950, this gives four equal chronological test blocks.

results = []

for fold in range(4):

    train_end = (fold + 1) * test_size
    test_start = train_end
    test_end = test_start + test_size

    train = df.iloc[:train_end]
    test = df.iloc[test_start:test_end]

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    model = make_model()
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    period = test["period"].iloc[0]

    for direction in ["ALL", "LONG", "SHORT"]:

        if direction == "ALL":
            mask = np.ones(len(test), dtype=bool)
        else:
            mask = (
                test["direction"].to_numpy() == direction
            )

        if mask.sum() == 0:
            continue

        y_true = y_test.to_numpy()[mask]
        y_prob = probabilities[mask]
        y_pred = predictions[mask]

        # AUC requires both classes.
        if len(np.unique(y_true)) == 2:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = np.nan

        ll = log_loss(
            y_true,
            np.clip(y_prob, 1e-6, 1 - 1e-6),
            labels=[0, 1],
        )

        acc = accuracy_score(y_true, y_pred)

        baseline_prob = np.full(
            len(y_true),
            y_true.mean(),
            dtype=float,
        )

        baseline_ll = log_loss(
            y_true,
            baseline_prob,
            labels=[0, 1],
        )

        results.append(
            {
                "fold": fold + 1,
                "period": period,
                "direction": direction,
                "n": len(y_true),
                "positive_rate": y_true.mean(),
                "auc": auc,
                "logloss": ll,
                "baseline_logloss": baseline_ll,
                "logloss_delta": ll - baseline_ll,
                "accuracy": acc,
            }
        )


results = pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

print("=== XGBOOST OOS: PERIOD × DIRECTION ===")
print()

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print()
print("=== AUC MATRIX ===")

auc_matrix = results.pivot(
    index="period",
    columns="direction",
    values="auc",
)

print(
    auc_matrix.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


print()
print("=== LOGLOSS DELTA MATRIX ===")
print("(negative = better than fold baseline)")

ll_matrix = results.pivot(
    index="period",
    columns="direction",
    values="logloss_delta",
)

print(
    ll_matrix.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


print()
print("=== DIRECTION SUMMARY ===")

for direction in ["ALL", "LONG", "SHORT"]:

    subset = results[
        results["direction"] == direction
    ]

    print(
        f"{direction}: "
        f"mean_auc={subset['auc'].mean():.4f} "
        f"mean_logloss={subset['logloss'].mean():.4f} "
        f"mean_baseline={subset['baseline_logloss'].mean():.4f} "
        f"mean_delta={subset['logloss_delta'].mean():+.4f} "
        f"mean_accuracy={subset['accuracy'].mean():.4f}"
    )


print()
print("=== PERIOD SUMMARY ===")

for period in ["P1", "P2", "P3", "P4"]:

    subset = results[
        results["period"] == period
    ]

    print(
        f"{period}: "
        f"ALL_AUC={subset.loc[subset.direction == 'ALL', 'auc'].iloc[0]:.4f} "
        f"LONG_AUC={subset.loc[subset.direction == 'LONG', 'auc'].iloc[0]:.4f} "
        f"SHORT_AUC={subset.loc[subset.direction == 'SHORT', 'auc'].iloc[0]:.4f}"
    )


print()
print("=== CHECKS ===")

if len(results) != 12:
    raise SystemExit(
        f"Expected 12 result rows, got {len(results)}"
    )

if not results["n"].gt(0).all():
    raise SystemExit("Empty result group found.")

print("Checks passed.")
