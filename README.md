# Market Entry Engine

Offline-first market analysis and entry recommendation engine built around
lightweight tabular machine learning.

The initial models are:

- LightGBM
- XGBoost

The project is designed for CPU-based systems with limited memory.

## Goal

Build a reproducible pipeline:

OHLCV Data
-> Data Validation
-> Feature Engineering
-> Label Generation
-> Time-Series Validation
-> LightGBM / XGBoost
-> Out-of-Sample Evaluation
-> Entry Decision Engine
-> Backtesting
-> Paper / Shadow Trading

The eventual signal format is:

- LONG
- SHORT
- NO TRADE

with supporting information such as:

- entry zone
- stop loss
- take profit
- risk/reward
- model probability
- market context

## Project Status

Early development.

### Current engineering position

The deterministic Market Structure layer is being built before context,
machine learning, and entry optimization.

Completed and locked:

- structural candle classification
- inside / outside candle handling
- directional legs
- pullback candidates and validation
- valid swing generation
- HH / HL / LH / LL labels from valid swings
- BOS from confirmed valid swings
- swing confirmation-time handling for no-look-ahead processing
- first BOS structural anchoring
- StructureCheckpoint and forward continuation

Current next step:

- internal / external structure
- external-boundary break and structure rebuild

The project is deliberately keeping market structure deterministic before
adding contextual features or ML models.

## Design Principles

1. Offline-first
2. CPU-friendly
3. Reproducible
4. No look-ahead bias
5. NO TRADE is a valid decision
6. Simple before complex

## Roadmap

### Deterministic market structure

- [x] Data foundation and reproducible raw dataset
- [x] Structural candle classification
- [x] Inside / outside semantics
- [x] Pullback validation
- [x] Valid swing confirmation
- [x] HH / HL / LH / LL
- [x] BOS from confirmed valid swings
- [x] First BOS warm-up / structural anchor
- [x] StructureCheckpoint / forward continuation
- [ ] Internal / external structure
- [ ] External boundary break / rebuild
- [ ] Close Market Structure v1 specification

### Context and entry engine

- [ ] Market context features
- [ ] Setup definition
- [ ] Entry / invalidation / target logic
- [ ] Execution cost model
- [ ] Deterministic baseline

### Statistical / ML evaluation

- [ ] Label generation
- [ ] Time-series validation
- [ ] LightGBM baseline
- [ ] XGBoost baseline
- [ ] Out-of-sample evaluation
- [ ] Backtesting
- [ ] Paper / Shadow trading

## Disclaimer

This project is for research and experimentation.

Model predictions are uncertain. Historical or backtested performance
does not establish future results. This project is not financial advice.
