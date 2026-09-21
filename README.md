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

The deterministic Market Structure v1 layer is locked before context,
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
- internal / external structure scope
- external-boundary break and structure rebuild
- deterministic simultaneous-event ordering
- confirmation-time structure feature contract
- active structural context features
- deterministic entry semantics v1

Current next step:

- implement time-series validation
- establish leakage-safe train / validation / test splits
- keep validation chronology explicit

The project deliberately keeps deterministic structure and feature contracts
ahead of statistical modeling and entry optimization.

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
- [x] Internal / external structure
- [x] External boundary break / rebuild
- [x] Close Market Structure v1 specification

### Context and entry engine

- [x] Market context features
- [x] Setup definition
- [x] Setup / entry / invalidation semantics
- [x] Target / exit-area model
  - Exit areas use only valid swings confirmed before setup.
  - A swing is excluded if price breaches its level after confirmation and before setup.
  - Confirmation and setup candles are excluded from the breach window.
- [x] Execution cost model
  - Execution is separated from structural setup generation.
  - `SetupCandidate.entry_price` remains the quoted/setup price.
  - Execution applies `spread_price` expressed in price units.
  - Long entry pays spread above the quoted price.
  - Short entry pays spread below the quoted price.
  - Execution risk is recalculated from the executed entry price to invalidation.
  - Raw MT5 `<SPREAD>` is not used directly; conversion to price units belongs to the data/source adapter.
- [x] Deterministic execution baseline
  - Trade outcome is evaluated from candles after setup.
  - Execution price and risk include the modeled spread.
  - Target and invalidation are evaluated as deterministic price barriers.
  - If stop and target are both touched in one candle, stop is resolved first.
  - Outcomes are `TARGET`, `STOP`, or `OPEN`.
  - PnL and R-multiple use the executed entry price.
- [x] Deterministic backtest orchestration
  - Setup candidates are evaluated through execution and exit-area selection.
  - The nearest valid exit area is used as the deterministic target.
  - Candidates without a valid exit area are skipped.
  - Portfolio sizing and additional transaction costs remain outside this baseline.
- [x] Backtest result aggregation
  - Summarizes total trades and counts for `TARGET`, `STOP`, and `OPEN`.
  - Sums PnL and R-multiple for closed outcomes.
  - Does not assume position sizing, portfolio allocation, or additional costs.

### Statistical / ML evaluation

- [x] Label generation
  - Barrier labels are deterministic and generated only from candles after the setup candle.
  - `TP_FIRST` and `SL_FIRST` are the only binary modeling labels.
  - `BOTH_SAME_CANDLE` is explicitly retained but excluded from binary modeling.
  - `UNRESOLVED` is retained for setups whose horizon expires without a barrier hit.
  - Canonical label definitions live in `market_engine.labels`.
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


### Checkpoint equivalence

The first-BOS checkpoint is defined as a post-candle state. If the first BOS
also closes an external boundary, the boundary rebuild is applied before the
checkpoint is emitted. Forward continuation from that checkpoint must produce
the same post-anchor events and confirmed swings as the uninterrupted full run.

### Quality gate

The current development quality gate is:

- `ruff check .`
- `pytest -q`

Ruff covers the engine and test suite. Exploratory audit scripts under
`scripts/audit_*.py` are intentionally excluded from the production lint scope.

Current checkpoint: **label contract added; local test gate pending.**
