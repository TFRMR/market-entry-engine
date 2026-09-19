"""Audit CORE feature/target relationship across chronological periods."""

from pathlib import Path

import numpy as np
import pandas as pd


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

if df[TARGET].isna().any():
    raise SystemExit("Missing target value found.")


# ---------------------------------------------------------------------------
# Four chronological periods
# ---------------------------------------------------------------------------

df["period"] = pd.qcut(
    np.arange(len(df)),
    q=4,
    labels=["P1", "P2", "P3", "P4"],
)


# ---------------------------------------------------------------------------
# Feature stability
# ---------------------------------------------------------------------------

rows = []

for period in ["P1", "P2", "P3", "P4"]:

    part = df[df["period"] == period]

    for feature in FEATURES:

        x = part[feature]
        y = part[TARGET]

        valid = x.notna() & y.notna()

        x = x[valid]
        y = y[valid]

        # Spearman correlation is rank-based and less sensitive to scale.
        correlation = x.corr(y, method="spearman")

        rows.append(
            {
                "period": period,
                "feature": feature,
                "n": len(x),
                "target_rate": y.mean(),
                "spearman": correlation,
            }
        )


result = pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output correlation matrix
# ---------------------------------------------------------------------------

print("=== CORE FEATURE / TARGET SPEARMAN BY PERIOD ===")
print()

matrix = result.pivot(
    index="feature",
    columns="period",
    values="spearman",
)

print(
    matrix.to_string(
        float_format=lambda x: f"{x:+.4f}"
    )
)


# ---------------------------------------------------------------------------
# Stability summary
# ---------------------------------------------------------------------------

print()
print("=== FEATURE SIGNAL STABILITY ===")

for feature in FEATURES:

    values = result.loc[
        result["feature"] == feature,
        "spearman",
    ].dropna()

    signs = np.sign(values)

    positive_periods = int((signs > 0).sum())
    negative_periods = int((signs < 0).sum())

    print(
        f"{feature}: "
        f"mean={values.mean():+.4f} "
        f"min={values.min():+.4f} "
        f"max={values.max():+.4f} "
        f"spread={values.max() - values.min():.4f} "
        f"positive={positive_periods}/4 "
        f"negative={negative_periods}/4"
    )


# ---------------------------------------------------------------------------
# Target rate by period
# ---------------------------------------------------------------------------

print()
print("=== TARGET RATE BY PERIOD ===")

target_rates = (
    result
    .drop_duplicates("period")
    .set_index("period")["target_rate"]
)

print(
    target_rates.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ---------------------------------------------------------------------------
# Feature distribution shift
# ---------------------------------------------------------------------------

print()
print("=== FEATURE MEDIAN BY PERIOD ===")

median_table = df.groupby(
    "period",
    observed=True,
)[FEATURES].median().T

print(
    median_table.to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

print()
print("=== CHECKS ===")

expected_rows = 4 * len(FEATURES)

if len(result) != expected_rows:
    raise SystemExit(
        f"Expected {expected_rows} rows, got {len(result)}"
    )

if result["n"].le(0).any():
    raise SystemExit("Empty feature/period group found.")

if not np.isfinite(
    result["spearman"].dropna().to_numpy()
).all():
    raise SystemExit("Non-finite correlation found.")

print("Checks passed.")
