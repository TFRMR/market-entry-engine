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

The project does not currently claim profitability or suitability
for live trading.

## Design Principles

1. Offline-first
2. CPU-friendly
3. Reproducible
4. No look-ahead bias
5. NO TRADE is a valid decision
6. Simple before complex

## Roadmap

- [ ] M0 - Project foundation
- [ ] M1 - Data pipeline
- [ ] M2 - Feature engineering
- [ ] M3 - Labeling
- [ ] M4 - LightGBM baseline
- [ ] M5 - XGBoost baseline
- [ ] M6 - Validation
- [ ] M7 - Backtesting
- [ ] M8 - Entry engine
- [ ] M9 - Paper / Shadow trading

## Disclaimer

This project is for research and experimentation.

Model predictions are uncertain. Historical or backtested performance
does not establish future results. This project is not financial advice.
