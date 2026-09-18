# Data Contract

## Purpose

This document defines the input data contract for the Market Entry Engine.

The initial dataset is an XAUUSD broker feed exported from MetaTrader 5 (MT5). The actual symbol in the supplied export is XAUUSDc, and the supplied timeframe is M30.

The data layer is responsible for loading, validating, and describing candle data before feature engineering, labeling, machine learning, or backtesting.

## Source Format

The first real dataset was exported manually from MT5.

Observed source columns:

    <DATE>  <TIME>  <OPEN>  <HIGH>  <LOW>  <CLOSE>  <TICKVOL>  <VOL>  <SPREAD>

The supplied file is tab-separated. The loader normalizes these source columns into the project's internal schema without changing the raw file.

## Instrument

The supplied instrument is XAUUSDc.

XAUUSDc is the broker's symbol identifier and must be treated as metadata. The engine must not silently rename it to XAUUSD.

## Timeframe

The supplied dataset uses M30.

One dataset must represent one timeframe only. Mixed timeframes within the same dataset are not allowed.

## Normalized Schema

| Field | Type | Description |
| --- | --- | --- |
| timestamp | datetime | Candle opening time |
| open | float | Opening price |
| high | float | Highest price |
| low | float | Lowest price |
| close | float | Closing price |
| tick_volume | integer | MT5 tick volume |
| real_volume | integer | MT5 real volume field |
| spread | integer | MT5 spread field |

For the supplied dataset, real_volume is zero. We preserve it rather than replacing it. For modeling, tick_volume is the volume field currently available from this source.

## Timestamp

MT5 exports date and time as separate fields. The loader combines DATE and TIME into timestamp.

The source timestamps are broker/MT5 timestamps and do not contain timezone information in the CSV itself. The actual timezone must therefore be recorded separately when the dataset is promoted into a modeling dataset.

Requirements:
* timestamps must be parseable;
* duplicate timestamps are not allowed;
* timestamps must be in ascending order;
* timezone assumptions must be explicit;
* the raw timestamp must not be silently shifted.

## OHLC Constraints

For every candle:

    high >= open
    high >= close
    low <= open
    low <= close
    high >= low

Any candle violating these constraints is invalid.

## Numeric Values

Prices must be finite and greater than zero.

Volume values must not be negative.

Spread must not be negative.

Missing or non-finite required values are invalid.

## Missing and Invalid Data

Missing or invalid rows must not be silently discarded.

The validator rejects invalid data rather than repairing or removing rows.

## Ordering and Duplicates

Rows must be chronologically ordered.

The validator detects duplicate and out-of-order timestamps and rejects them rather than silently reordering or deduplicating the source.

## Raw vs Processed

Raw data must remain unchanged.

    data/
    ├── raw/
    └── processed/

data/raw contains the original MT5 export. data/processed contains validated/transformed data generated reproducibly from the raw source.

The supplied raw dataset should not be committed to Git because market-data files are intentionally excluded by the repository's .gitignore.

## Dataset Metadata

    instrument : XAUUSDc
    source     : MT5 export
    timeframe  : M30
    timezone   : to be explicitly determined
    volume_type: tick_volume

Also record first_timestamp, last_timestamp, and row_count.

## Data Integrity Principles

* Validate before feature engineering.
* Never silently repair suspicious market data.
* Never silently remove invalid rows.
* Preserve source information.
* Keep validation independent from machine-learning models.
* Avoid look-ahead information.
* Keep raw data separate from processed data.
* Make transformations reproducible.

## Out of Scope

This document does not yet define feature engineering, machine-learning labels, model training, model selection, trading signals, backtesting, position sizing, live trading, or broker execution.

## Current Scope

    Instrument : XAUUSDc
    Source     : MT5 manual export
    Timeframe  : M30
    Format     : MT5 tab-separated CSV
    Models     : LightGBM + XGBoost
    Execution  : Offline analysis / recommendation
