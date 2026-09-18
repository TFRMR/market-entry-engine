# Data Contract

## Purpose

This document defines the input data contract for the Market Entry Engine.

The initial version of the engine is designed specifically for **XAUUSD** market data.

The data layer is responsible for loading, validating, and describing OHLCV candle data before the data is used by feature engineering, labeling, machine learning, or backtesting.

The goal is to establish a consistent, validated, and reproducible representation of market data.

---

## Instrument

The initial version supports one instrument:

```text
XAUUSD
```

The data pipeline must reject datasets containing a different instrument.

Instrument identity should be stored as dataset metadata rather than inferred only from the filename.

---

## Required Fields

Each candle must contain the following fields:

| Field       | Type     | Description                              |
| ----------- | -------- | ---------------------------------------- |
| `timestamp` | datetime | Candle opening time                      |
| `open`      | float    | Opening price                            |
| `high`      | float    | Highest price during the candle          |
| `low`       | float    | Lowest price during the candle           |
| `close`     | float    | Closing price                            |
| `volume`    | float    | Volume value provided by the data source |

---

## Timestamp

`timestamp` represents the opening time of each candle.

Requirements:

* Must be parseable as a datetime.
* Candles must be ordered chronologically.
* Duplicate timestamps are not allowed.
* The dataset must use one consistent timezone.
* The timezone must be explicitly known before the data enters the modeling pipeline.

The initial implementation should preserve the original timestamp information rather than silently converting it to another timezone.

---

## OHLC Constraints

For every candle:

```text
high >= open
high >= close
low <= open
low <= close
high >= low
```

Therefore:

```text
low <= min(open, close)
max(open, close) <= high
```

Any candle violating these constraints must be considered invalid.

---

## Numeric Values

The following fields must contain finite numeric values:

```text
open
high
low
close
volume
```

The following values are invalid:

* `NaN`
* positive infinity
* negative infinity

Prices must be greater than zero.

Volume must not be negative.

A zero volume value is allowed at the data-contract level because its meaning depends on the data source.

---

## Volume

`volume` represents the volume field provided by the data source.

The engine must **not** assume that this represents centralized traded volume.

Depending on the data source, the volume field may represent:

* tick volume
* broker-provided volume
* another volume definition provided by the source

The type and meaning of the volume field should be recorded as dataset metadata whenever the information is available.

The initial data layer must preserve the supplied volume rather than attempting to reinterpret it.

---

## Missing Values

Required fields must not contain missing values after validation.

Missing or invalid rows must not be silently discarded.

The validator should report:

* number of invalid rows
* affected columns
* validation reason

Whether invalid rows are removed, repaired, or cause the dataset to be rejected will be determined by the data-processing policy.

---

## Ordering

Rows must be sorted by ascending `timestamp`.

The validator must detect:

* out-of-order timestamps
* duplicate timestamps

The initial policy is to reject invalid ordering rather than silently reorder data.

This helps expose problems in the original dataset.

---

## Timeframe

The initial data layer does not assume a particular timeframe.

Possible timeframes may include:

```text
1m
5m
15m
30m
1h
4h
1d
```

The actual timeframe used by the project will be selected separately based on the research objective and available XAUUSD data.

One dataset must represent **one timeframe only**.

Mixed timeframes within the same dataset are not allowed.

The timeframe should be recorded as dataset metadata.

---

## Market Data Source

The source of the XAUUSD data must be recorded whenever possible.

Examples of source metadata include:

```text
source
symbol
timeframe
timezone
volume_type
```

The engine should not silently mix data from different sources.

If datasets from different sources are combined in the future, the methodology must explicitly document how they are aligned and validated.

---

## CSV Input

The initial input format is CSV.

Expected minimum structure:

```csv
timestamp,open,high,low,close,volume
2026-01-01 09:00:00,2650.10,2652.40,2648.90,2651.80,1200
2026-01-01 10:00:00,2651.80,2655.20,2650.70,2654.60,1500
```

The exact price values above are examples only.

Column names should be normalized by the loader before validation.

The loader must not silently reinterpret the meaning of columns.

---

## Validation Policy

The data pipeline should distinguish between three major categories of errors.

### 1. Structural Errors

Examples:

* file cannot be read
* required column is missing
* unsupported data structure
* invalid data type

### 2. Value Errors

Examples:

* invalid OHLC relationship
* non-positive price
* negative volume
* non-finite numeric value
* missing required value

### 3. Temporal Errors

Examples:

* duplicate timestamp
* out-of-order timestamp
* mixed timeframe
* inconsistent timezone information

Validation results must be explicit and reproducible.

---

## Data Integrity Principles

The data layer follows these principles:

* Validate data before feature engineering.
* Do not silently repair suspicious market data.
* Do not silently remove invalid rows.
* Preserve source information.
* Make validation failures explainable.
* Keep data validation independent from machine-learning models.
* Do not assume a particular broker's interpretation of market data.
* Avoid look-ahead information entering the dataset.
* Keep the original raw dataset separate from processed data.

---

## Raw vs Processed Data

Raw data must remain unchanged.

The expected directory structure is:

```text
data/
├── raw/
└── processed/
```

### `data/raw/`

Contains original source data.

Raw files should not be modified by the processing pipeline.

### `data/processed/`

Contains validated or transformed datasets produced by the pipeline.

Processing steps must be reproducible from the raw data and documented configuration.

---

## Dataset Metadata

A dataset should eventually have associated metadata describing at least:

```text
instrument
source
timeframe
timezone
volume_type
first_timestamp
last_timestamp
row_count
```

Additional metadata may be added as the project develops.

---

## Out of Scope

This document does not yet define:

* feature engineering
* machine-learning labels
* trading signals
* model training
* model selection
* backtesting
* position sizing
* live trading
* broker execution

These concerns belong to later stages of the project.

---

## Current Scope

The current project scope is intentionally narrow:

```text
Instrument : XAUUSD
Input      : OHLCV
Format     : CSV
Models     : LightGBM + XGBoost
Execution  : Offline analysis / recommendation
```

The timeframe, data source, labeling parameters, and trading assumptions will be defined in later stages.
