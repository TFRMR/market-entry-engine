# Market Entry Engine

Offline-first market analysis and entry research engine built around deterministic price action, market structure, point-in-time context, and empirical outcome analysis.

The initial statistical models are:

- LightGBM
- XGBoost

The project is designed for CPU-based systems with limited memory.

## Goal

Build a reproducible research pipeline:

OHLCV Data
-> Data Validation
-> Price Action
-> Market Structure
-> Market Context / POI
-> Setup Candidate
-> Outcome
-> Empirical Probability / Distribution
-> Time-Series / OOS Validation
-> ML Evaluation
-> Backtesting
-> Paper / Shadow Trading

The deterministic engine is intentionally separated from statistical evaluation. Primitive layers describe observable price action, market structure, and market context; they do not emit BUY/SELL signals, rankings, or probability scores.

## Project Status

Early development.

### Current engineering position

The deterministic Market Structure v1 and Context v1 feature stack are the current research foundation. The feature design has been simplified to price action and market structure/context rather than a broad technical-indicator stack.

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
- OBIM
- structural S/R and Location
- local + D1 structural S/R mapping
- S/R interaction lifecycle: CREATED / TESTED / SWEPT / BROKEN / FLIPPED
- inducement events
- POI lifecycle and price interaction
- pullback state and continuous retracement measurement
- deterministic setup facts
- point-in-time ContextSnapshot

Removed from the current primitive feature stack:

- EMA 5 / 20 / 50 features
- ATR as a feature family and ATR-dependent primitive definitions
- standalone momentum-candle classification as the strategy foundation
- recent-return / recent-movement features
- standalone volume features
- legacy rolling support/resistance as the canonical structural definition
- indicator-derived signal/ranking logic

ATR may still appear in historical/compatibility documentation or in future normalization experiments, but it is not the foundation of the current feature contract.

The purpose of the deterministic stack is to expose point-in-time facts that can later be evaluated empirically. POI qualification is separate from primitive feature construction, and claims about outcome quality must be validated with time-series/OOS evaluation.

## Design Principles

1. Offline-first
2. CPU-friendly
3. Reproducible
4. No look-ahead bias
5. NO TRADE is a valid decision
6. Simple before complex
7. Deterministic facts before statistical interpretation
8. Empirical distributions before predictive thresholds

## Research Blueprint

The current research path is:

OHLCV
-> VALID CANDLE
-> PRICE ACTION
-> PULLBACK
-> VALID SWING
-> HH / HL / LH / LL
-> INTERNAL + EXTERNAL MARKET STRUCTURE
-> BOS / CHOCH / INDUCEMENT
-> MARKET JOURNEY
   - Liquidity
   - FVG
   - Order Block
   - OBIM
   - Structural S/R
-> POI
-> PRICE INTERACTION
-> CONTEXT
-> SETUP CANDIDATE
-> OUTCOME
-> EMPIRICAL DISTRIBUTION / PROBABILITY
-> OOS VALIDATION

This sequence is a research model, not a trading-rule prescription.

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

### Context and setup research

- [x] Market context feature contract
- [x] Setup definition
- [x] Setup / entry / invalidation semantics
- [x] Target / exit-area model
- [x] Execution cost model
- [x] Deterministic execution baseline
- [x] Deterministic setup label dataset
- [x] Setup facts
- [x] POI lifecycle and price interaction features
- [x] Inducement context
- [x] Point-in-time POI chronology audit
- [x] Point-in-time structural chronology audit
- [x] Point-in-time S/R lifecycle + D1 mapping audit
- [x] ContextSnapshot
- [x] Context + outcome dataset join
- [x] Empirical categorical outcome summaries
- [x] Empirical continuous-context summaries
- [x] First empirical context research pass
- [x] Historical/OOS validation of discovered empirical contexts
- [x] First multi-dimensional empirical context combinations
- [x] Broader pairwise POI/context empirical research runner
- [x] Broader multi-dimensional empirical context analysis
- [x] Chronological stability analysis with minimum-sample visibility
- [ ] Interpret stability findings and select research hypotheses for ML/backtest evaluation

### Statistical / ML evaluation

- [x] LightGBM baseline
- [x] Pristine chronological OOS evaluation
- [x] Feature-group ablation
- [x] Pullback / retracement OOS features
- [ ] XGBoost baseline
- [ ] Model refinement after empirical research

### Execution / deployment

- [ ] Backtesting
- [ ] Paper / Shadow trading

## Current Research Checkpoint

The deterministic context stack is substantially complete.

The immediate research task is no longer to add primitive indicators or repeatedly tune the baseline model. The next step is to measure observed outcome distributions for explicitly defined context dimensions on historical setup data.

The first empirical runner is:

`scripts/run_empirical_research.py`

Example:

```bash
.venv/bin/python scripts/run_empirical_research.py \
  data/raw/XAUUSDc_M30_202409012200_202609182030.csv \
  --spread-price 0.0
```

The runner produces:

- `data/research/xauusd_m30_context_outcomes.csv`
- `data/research/xauusd_m30_empirical_categorical.csv`
- `data/research/xauusd_m30_empirical_retracement.csv`
- `data/research/xauusd_m30_empirical_historical_categorical.csv`
- `data/research/xauusd_m30_empirical_historical_retracement.csv`
- `data/research/xauusd_m30_empirical_combinations.csv`
- `data/research/xauusd_m30_empirical_historical_combinations.csv`

The empirical layer is descriptive. It reports sample counts and observed outcome rates for supplied context groups and quantile bins. It does not rank contexts, assign a score, or define a predictive threshold.

The stability output is `data/research/xauusd_m30_empirical_stability.csv`. It matches identical context definitions across development and later historical/OOS data, keeping groups with at least 20 development observations and 10 historical observations. It reports observed outcome-rate deltas without ranking or scoring contexts.

Stability statistics now include 95% Wilson confidence intervals for development and historical TP_FIRST / UNRESOLVED rates, approximate 95% confidence intervals for development-to-historical rate deltas, and standardized drift statistics (z and two-sided normal p-value) for TP_FIRST and UNRESOLVED. These statistics describe sampling uncertainty and distribution drift; they are not used as a ranking score or trading threshold.

The first runner now validates the same categorical context definitions on the later historical/OOS sample. Continuous retracement bins are fitted on development data and reused unchanged on the historical sample, so the OOS pass does not relearn thresholds from the validation period.

Current M30 empirical checkpoint:

- 1,772 setup candidates
- 1,706 labeled context rows
- 1,279 development rows
- 427 later historical/OOS rows
- categorical context combinations are evaluated on both development and historical/OOS data using the same context definitions
- pullback retracement bins are fitted on development and reused unchanged for OOS
- POI interaction combinations now cover FVG, Order Block, OBIM, liquidity, and structural S/R

The current result is still descriptive research. Differences between development and historical distributions are treated as validation evidence, not as a ranking or trading rule. The runner now expands this pass into pairwise base-context + POI interaction groups, POI interaction pairs, and base-context + two POI interaction dimensions, with separate development and historical/OOS outputs. It also produces a chronological stability table using the same context definitions in both periods, with minimum-sample visibility.

Development and later historical observations remain distinguishable. A context that appears interesting in development data is a research hypothesis until its observed distribution is checked chronologically on later data.

## Structural / S/R Semantics

Structural S/R is contextual, not directional authority. Market direction comes from Market Structure; S/R provides location and interaction context.

D1 S/R is reconstructed from completed daily candles and mapped only to later intraday candles, so the active daily candle cannot leak its unfinished high/low/close into M30 context.

S/R lifecycle is deterministic and descriptive. It does not produce a trading signal or probability.

## Statistical Validation Principles

- All features must be available as of the setup candle.
- No future candle may define a setup-time context fact.
- Chronological splits are preferred over random splits.
- Empirical bins/thresholds are learned only from the supplied research sample.
- Any empirical pattern discovered in development data is a hypothesis until evaluated on later observations.
- Sample size must remain visible beside outcome rates.
- UNRESOLVED and BOTH_SAME_CANDLE outcomes are retained in descriptive summaries.
- Probability belongs to the empirical/model layer, not primitive feature construction.

## Disclaimer

This project is for research and experimentation.

Model predictions are uncertain. Historical or backtested performance does not establish future results. This project is not financial advice.

### Checkpoint equivalence

The first-BOS checkpoint is defined as a post-candle state. If the first BOS also closes an external boundary, the boundary rebuild is applied before the checkpoint is emitted. Forward continuation from that checkpoint must produce the same post-anchor events and confirmed swings as the uninterrupted full run.

### Quality gate

The current development quality gate is:

- ruff check .
- pytest -q

Ruff covers the engine and test suite. Exploratory audit scripts under `scripts/audit_*.py` are intentionally excluded from the production lint scope.

The current research checkpoint is deterministic Context v1 through POI/S/R lifecycle and point-in-time audits, followed by empirical context/outcome research.