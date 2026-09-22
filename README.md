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

The deterministic engine is intentionally separated from statistical evaluation. Primitive layers describe observable price action, market structure, and market context; they do not emit BUY/SELL signals or probability scores.

## Project Status

Early development.

### Current engineering position

The deterministic Market Structure v1 and Context v1 feature stack are the current
research foundation. The feature design has been simplified to price action and
market structure/context rather than a broad technical-indicator stack.

Current deterministic layers include:

- raw candle price action / displacement geometry
- directional Market Structure with INTERNAL / EXTERNAL scope
- valid swings, HH / HL / LH / LL
- BOS / CHoCH and chronological event sequence
- active and historical structural context
- trend / regime and range
- liquidity and sweeps
- FVG and historical FVG reference routing
- structural Order Block candidates and directional historical projection
- structural S/R and Location
- pullback state

Removed from the current primitive feature stack:

- EMA 5 / 20 / 50 features
- ATR as a feature family and ATR-dependent primitive definitions
- standalone momentum-candle classification as the strategy foundation
- recent-return / recent-movement features
- standalone volume features
- legacy rolling support/resistance as the canonical structural definition
- indicator-derived signal/ranking logic

ATR may still appear in historical/compatibility documentation or in future
normalization experiments, but it is not the foundation of the current feature
contract.

The purpose of the deterministic stack is to expose point-in-time facts that can
later be evaluated empirically for POI research. POI qualification is separate
from primitive feature construction, and claims about outcome quality must be
validated with time-series/OOS evaluation.

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
- [x] CHoCH transition event semantics
- [x] First BOS warm-up / structural anchor
- [x] StructureCheckpoint / forward continuation
- [x] Internal / external structure
- [x] External boundary break / rebuild
- [x] Preserve historical confirmed structure across active-context rebuilds
- [x] Expose historical structure context separately from active structure
- [x] Close Market Structure v1 specification

### Context and entry engine

- [x] Market context feature contract
- [x] Setup definition
- [x] Setup / entry / invalidation semantics
- [x] Target / exit-area model
  - Exit areas use only valid swings confirmed before setup.
  - A swing is excluded if price breaches its level after confirmation and before setup.
  - Confirmation and setup candles are excluded from the breach window.
- [x] Execution cost model
  - Execution is separated from structural setup generation.
  - SetupCandidate.entry_price remains the quoted/setup price.
  - Execution applies spread_price expressed in price units.
  - Long entry pays spread above the quoted price.
  - Short entry pays spread below the quoted price.
  - Execution risk is recalculated from the executed entry price to invalidation.
  - Raw MT5 SPREAD is not used directly; conversion to price units belongs to the data/source adapter.
- [x] Deterministic execution baseline
  - Trade outcome is evaluated from candles after setup.
  - Execution price and risk include the modeled spread.
  - Target and invalidation are evaluated as deterministic price barriers.
  - If stop and target are both touched in one candle, stop is resolved first.
  - Outcomes are TARGET, STOP, or OPEN.
  - PnL and R-multiple use the executed entry price.
- [x] Deterministic backtest orchestration
  - Setup candidates are evaluated through execution and exit-area selection.
  - The nearest valid exit area is used as the deterministic target.
  - Candidates without a valid exit area are skipped.
  - Portfolio sizing and additional transaction costs remain outside this baseline.
- [x] Backtest result aggregation
  - Summarizes total trades and counts for TARGET, STOP, and OPEN.
  - Sums PnL and R-multiple for closed outcomes.
  - Does not assume position sizing, portfolio allocation, or additional costs.

### Statistical / ML evaluation

- [x] Label contract
  - Canonical labels are TP_FIRST, SL_FIRST, BOTH_SAME_CANDLE, and UNRESOLVED.
  - TP_FIRST and SL_FIRST are the only binary modeling labels.
  - BOTH_SAME_CANDLE is retained for the generic barrier-outcome vocabulary.
  - Structural setup labeling uses the nearest valid structural exit area, not fixed R-multiple targets.
- [x] Structural setup dataset builder
  - Builds one row per setup candidate with features read from the setup candle.
  - Uses the same structural exit-area and execution semantics as backtesting.
  - Uses a fixed forward label horizon of 10 candles.
  - Maps TARGET -> TP_FIRST, STOP -> SL_FIRST, and horizon-expired OPEN -> UNRESOLVED.
  - Exposes reward_risk explicitly and flags same-candle stop/target ambiguity.
  - Setups without a complete forward horizon are excluded rather than mislabeled at end-of-data.
  - Candidates without a valid structural exit area are excluded.
- [x] Time-series validation
  - Chronological train / validation / test split with expanding training history.
  - Purge gap defaults to the structural label horizon (10 candles) to prevent forward-label overlap.
  - Explicit embargo gap is supported between validation and test.
  - No shuffling or random split is used.
- [x] Realized-R audit layer
  - Realized R uses the existing deterministic setup, execution, and target semantics.
  - Unresolved setups receive a deterministic TIME_STOP at the horizon close.
  - reward_r_after_spread is explicitly a target-distance ratio, not full net P&L.
  - Random-timing placebo is descriptive only.
  - Day-clustered bootstrap is an uncertainty diagnostic, not a guarantee of independence.
- [x] Deterministic displacement / price action
- [x] Pullback state
- [x] Event Sequence
- [x] Full context leakage audit
- [x] Feature dataset schema / ownership audit
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

- ruff check .
- pytest -q

Ruff covers the engine and test suite. Exploratory audit scripts under
scripts/audit_*.py are intentionally excluded from the production lint scope.

Current checkpoint: Market Structure v1 is locked, including deterministic CHoCH semantics; Context v1 trend/regime, range, liquidity, FVG, Order Block, canonical structural S/R + Location, and FVG Transition reference routing are implemented and quality-gated; next step is deterministic displacement / price action.
