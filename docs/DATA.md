# Market Data

## 1. Purpose

This document defines the market-data contract and records the characteristics of the real dataset used during the initial development of the Market Entry Engine.

The project currently uses an MT5-exported OHLCV dataset for:

* Instrument: `XAUUSDc`
* Timeframe: `M30`
* Source: manual MetaTrader 5 export
* Data period: January 2026 to September 2026

The dataset is broker-specific. The symbol name, trading-session behavior, spread representation, and available volume fields must not be assumed to be identical to other XAUUSD feeds.

---

## 2. Source Dataset

The initial real dataset is an MT5 tab-separated export with the following columns:

| MT5 column  | Normalized column | Meaning                        |
| ----------- | ----------------- | ------------------------------ |
| `<DATE>`    | `date`            | Candle date                    |
| `<TIME>`    | `time`            | Candle time                    |
| `<OPEN>`    | `open`            | Opening price                  |
| `<HIGH>`    | `high`            | Highest price                  |
| `<LOW>`     | `low`             | Lowest price                   |
| `<CLOSE>`   | `close`           | Closing price                  |
| `<TICKVOL>` | `tick_volume`     | Tick volume                    |
| `<VOL>`     | `real_volume`     | Real volume supplied by broker |
| `<SPREAD>`  | `spread`          | Exported spread value          |

The loader combines `<DATE>` and `<TIME>` into a single `timestamp` column.

The normalized schema currently contains:

```text
timestamp
open
high
low
close
tick_volume
real_volume
spread
```

---

## 3. Data Contract

The normalized market dataset must satisfy the following conditions.

### Timestamp

* `timestamp` must exist.
* Timestamps must be unique.
* Timestamps must be in ascending order.
* Missing timestamps are not automatically considered data errors because market data can legitimately contain trading-session closures.

### Price

The following fields are required:

```text
open
high
low
close
```

Prices must:

* be numeric;
* be finite;
* be greater than zero.

Candle structure must satisfy:

```text
high >= max(open, close)
low  <= min(open, close)
high >= low
```

### Volume

The following fields are currently retained:

```text
tick_volume
real_volume
```

Volume values must:

* be numeric;
* be finite;
* not be negative.

For the current `XAUUSDc` dataset, `tick_volume` is the meaningful volume field. `real_volume` is constant at zero in the exported data and therefore does not currently provide useful variation for feature engineering.

### Spread

`spread` must:

* be numeric;
* be finite;
* not be negative.

The raw spread value is retained exactly as exported by MT5. Its unit interpretation is broker/feed-specific and must not be assumed to be a universal price unit.

---

## 4. Real Dataset Audit

The first audit was performed against the real MT5 export.

Dataset:

```text
XAUUSDc_M30_202601012300_202609181730.csv
```

Audit result:

| Property                     |              Result |
| ---------------------------- | ------------------: |
| Rows                         |               8,468 |
| Start                        | 2026-01-01 23:00:00 |
| End                          | 2026-09-18 17:30:00 |
| Duplicate timestamps         |                   0 |
| Missing values               |                   0 |
| Minimum interval             |          30 minutes |
| Maximum interval             |       4,410 minutes |
| Non-30-minute intervals      |                 184 |
| Gaps greater than 30 minutes |                 184 |
| Spread minimum               |                 160 |
| Spread maximum               |                 500 |
| Spread mean                  |              248.76 |
| Real-volume unique values    |                   1 |
| Validation                   |                PASS |

The audit output is retained as development evidence outside the source-code repository.

---

## 5. Interpretation of Time Gaps

The dataset contains 184 intervals that are longer than the nominal M30 interval.

These intervals are not automatically classified as missing candles.

Inspection of the gap distribution shows recurring and systematic patterns, including:

* daily session breaks;
* Friday-to-Sunday weekend closures;
* longer closures around specific dates;
* Monday session-start delays;
* changes in session timing during the dataset period consistent with a schedule shift;
* several special longer closures.

Examples include recurring approximately 90-minute session breaks, weekend gaps, and longer holiday-style closures.

Therefore:

> A gap between two candles does not by itself indicate corrupted or missing market data.

The current data pipeline **does not interpolate, forward-fill, or synthesize candles across these gaps**.

This is intentional.

Synthetic candles could introduce artificial price movement, volatility, volume, and market-structure information that did not exist in the source feed.

---

## 6. Session Gaps vs. Data Errors

The project distinguishes between:

### Expected discontinuities

Examples:

* market/session closure;
* weekend;
* broker schedule changes;
* holidays or special trading hours.

These should remain represented by the absence of candles.

### Potential data-quality problems

Examples:

* duplicate timestamps;
* timestamps going backward;
* malformed timestamps;
* missing OHLC values;
* non-finite numeric values;
* invalid OHLC relationships;
* negative volume;
* negative spread;
* impossible price values.

These should cause validation to fail.

The loader and validator currently enforce the second category but do not attempt to reconstruct the first category.

---

## 7. Timezone and Broker Schedule

The current dataset records timestamps exactly as supplied by the MT5 export.

No universal timezone assumption is currently encoded into the normalized dataset.

Session behavior observed during the audit is considered **broker/feed-specific**.

Future work may explicitly model:

* broker timezone;
* UTC conversion;
* daylight-saving schedule changes;
* trading-session boundaries;
* holidays;
* early closes.

Until those rules are explicitly defined and verified, the original exported timestamps should be preserved.

---

## 8. Volume Policy

The current dataset provides both tick volume and real volume.

Observed behavior:

```text
tick_volume  -> variable
real_volume  -> constant zero
```

Therefore:

* `tick_volume` may be used as a market-activity feature;
* `real_volume` should not currently be used as a predictive feature;
* the raw `real_volume` column is retained for traceability and future compatibility.

This policy may change if a future data source provides meaningful real volume.

---

## 9. Spread Policy

The exported `spread` field is retained because transaction cost is relevant to entry decisions and backtesting.

Observed values in the initial dataset:

```text
minimum: 160
maximum: 500
mean:    248.76
```

These values are treated as raw broker-exported spread values.

Before using spread directly in price calculations, the project must establish the broker's point/digit convention for `XAUUSDc`.

For backtesting, transaction-cost modeling should eventually account for:

* spread;
* slippage;
* fees or commissions where applicable.

---

## 10. Data Handling Rules

The current project follows these rules:

1. Preserve the original candle data.
2. Do not interpolate missing market candles.
3. Do not fill session gaps with synthetic candles.
4. Validate structural OHLC constraints.
5. Preserve broker-provided spread information.
6. Prefer tick volume over the current zero-valued real volume.
7. Preserve the original timestamp representation until timezone/session rules are explicitly defined.
8. Keep raw market data outside Git when it is covered by `.gitignore`.
9. Make preprocessing deterministic and reproducible.
10. Document assumptions before using the data for model training or backtesting.

---

## 11. Current Limitations

The initial dataset is sufficient for beginning feature-engineering work, but several limitations remain.

### Broker-specific data

The data represents `XAUUSDc` from one broker/feed.

Results should not automatically be generalized to every XAUUSD feed.

### Limited historical period

The current dataset covers approximately nine months of M30 data.

This is useful for development but limited for establishing long-term robustness across many market regimes.

### Real volume unavailable

The exported real-volume field is constant zero.

Tick volume is therefore the available activity proxy.

### Session rules not yet formalized

The audit identifies systematic session gaps, but the project has not yet encoded a complete broker trading-calendar model.

### Spread unit not yet normalized

The raw spread values are retained, but their conversion into price units has not yet been formalized.

---

## 12. Audit Tool

The dataset audit can be reproduced with:

```bash
python scripts/audit_dataset.py <path-to-mt5-csv>
```

For example:

```bash
python scripts/audit_dataset.py \
  /home/tofarmer/Desktop/XAUUSDc_M30_202601012300_202609181730.csv
```

The audit script performs validation through the project's normalized MT5 loader and reports:

* dataset size;
* time range;
* duplicate timestamps;
* missing values;
* candle intervals;
* non-standard intervals;
* large gaps;
* spread statistics;
* volume characteristics;
* validation status.

The audit tool is part of the repository so that future datasets can be checked using the same procedure.

---

## 13. Status

M1 Data Foundation is complete enough to proceed to feature engineering.

Completed:

* MT5 CSV loading;
* normalized candle schema;
* OHLCV validation;
* automated tests;
* real-dataset audit;
* session-gap inspection;
* decision not to synthesize candles across market closures.

Next phase:

> **M2 — Feature Engineering**

The feature pipeline should consume the validated normalized dataset and preserve chronological ordering.
