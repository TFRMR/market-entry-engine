# Feature Engineering

## 1. Purpose

The current feature stack is deliberately centered on **price action and Market
Structure**. Technical indicators are not used as the foundation of the
primitive feature contract.

The deterministic engine describes what is observable and knowable at each
candle. Statistical modeling is responsible for testing which combinations of
those facts are associated with outcomes.

The intended research flow is:

```
OHLCV
  ↓
Price Action
  ↓
Market Structure
  ├─ INTERNAL
  └─ EXTERNAL
  ↓
Structure Events / Event Sequence
  ↓
Market Context
  ├─ Trend / Range
  ├─ Liquidity
  ├─ FVG
  ├─ Order Block
  ├─ S/R + Location
  └─ Pullback
  ↓
POI candidate/context research
  ↓
ML / OOS evaluation
```

The primitive layers do not assign probability, rank setups, or emit BUY/SELL
signals.

## 2. Removed indicator-oriented feature stack

The following were removed from the current primitive implementation:

- EMA 5 / 20 / 50 context
- standalone ATR / volatility feature family
- momentum-candle strategy classification
- recent movement / return features
- standalone volume features
- ATR-dependent Order Block / FVG / S/R primitive fields
- indicator-derived signal or ranking logic

This cleanup makes the current feature contract primarily raw price-action,
structural, and contextual.

ATR can still be used later as an explicitly scoped normalization experiment,
but it is not required by the current primitive definitions.

## 3. Price Action / Displacement

Current deterministic candle-geometry features:

- `candle_range`
- `candle_body`
- `body_ratio`
- `upper_wick_ratio`
- `lower_wick_ratio`
- `close_position`
- `close_position_in_range`
- `upper_wick_to_body`
- `lower_wick_to_body`

These describe candle geometry only. No fixed momentum threshold is part of
the current primitive contract.

## 4. Market Structure

Market Structure is the primary directional/context foundation.

It contains:

- directional legs
- valid swing HIGH / LOW confirmation
- HH / HL / LH / LL
- BOS
- CHOCH
- INTERNAL / EXTERNAL scope
- active vs historical structure
- confirmation-time availability

A structural fact becomes available only when it is knowable. A future-confirmed
swing, BOS, or other structural event must never be projected backward.

## 5. Structure Events and Event Sequence

Sparse event features expose BOS, CHOCH, swing-validation, HH/HL/LH/LL and
INTERNAL/EXTERNAL event facts.

Event Sequence projects chronological event history into point-in-time fields,
including:

- event count
- last / previous event index, age, type, direction, and scope
- events since last BOS
- events since last swing
- bars since last BOS
- bars since last swing

Event history is descriptive. It does not infer a signal or score.

## 6. Trend and Range

Trend / regime and range features provide structural context around the active
market state:

- trend regime
- trend transition
- directional transition
- range high / low / width
- range position
- range position zone
- range state

These are deterministic context fields, not trading rules.

## 7. Liquidity

Liquidity is derived from confirmed structural highs/lows.

Current fields include:

- liquidity high / low
- presence flags
- distance to liquidity
- high / low sweep
- sweep state
- sweep direction
- sweep size

A clean structural break remains distinct from a liquidity sweep.

## 8. Fair Value Gap

FVG context exposes the latest known historical FVG and its raw spatial
properties.

The separate historical FVG reference layer preserves every valid FVG with
creation-time availability. FVG Transition v1 routes spatial references such
as origin, target, and next-after-target without asserting transition outcome.

FVG is contextual information, not a deterministic signal.

## 9. Order Block

An Order Block is a structural candidate, not automatically a POI.

Current projection fields for each direction are:

- `present`
- `size`
- `age_bars`
- `distance`
- `contains_price`
- `relative_position`

The candidate is derived from confirmed structural swings associated with BOS
or CHOCH events. The zone is wick-to-wick.

The primitive layer does not rank candidates by freshness, mitigation,
imbalance, higher-timeframe alignment, or other heuristic quality criteria.

## 10. S/R and Location

Canonical S/R uses confirmed structural swings:

- confirmed LOW → support
- confirmed HIGH → resistance

Current fields include:

- `support_level`
- `resistance_level`
- `distance_to_support`
- `distance_to_resistance`
- `distance_to_next_structure_level`
- `leg_position`

Location remains structural and point-in-time safe.

## 11. Pullback State

Pullback has a project-specific structural meaning:

> A candidate/reference state that appears while a directional leg is active,
> before a reversal can validate the active extreme as a swing.

A pullback candidate may continue to exist without producing a valid swing.

Current fields include:

- `pullback_active`
- `pullback_direction`
- `pullback_price`
- `pullback_start_index`
- `pullback_extreme_index`
- `pullback_bars`
- `pullback_depth`

## 12. POI Research Boundary

The deterministic stack is intended to provide the facts needed to study POI
quality empirically.

A POI is not defined merely because an Order Block, FVG, liquidity level, or
structural level exists. Candidate qualification may later combine structural
scope, directional context, location, event sequence, displacement, pullback
state, and other explicitly defined facts.

Any claim that a POI is "high probability" must come from measured outcomes
under a time-series/OOS evaluation protocol. The primitive feature layer must
not encode that conclusion in advance.

## 13. No-look-ahead contract

At candle t, features must not use:

- a swing whose confirmation occurs after t
- a BOS / CHOCH that occurs after t
- future FVG / OB / liquidity references
- future target or exit information
- future candle information not explicitly permitted by the feature definition

Feature values are point-in-time facts and may be carried forward only after
their source state becomes available.

## 14. Feature ownership

Each current feature column has one producing builder. The ownership audit found
no cross-builder output collisions.

Event Sequence owns `structure_bars_since_last_bos`; Active Structure owns
direction, structural distances, historical distances, and bars since the last
confirmed swing.

Semantic aliases such as `structure_last_valid_high` and `range_high` are
retained until their public contract is deliberately consolidated.

## 15. Validation status

The current deterministic stack has passed:

- structure chronology audit
- feature source chronology audit
- missingness semantics audit
- feature dataset schema audit
- feature ownership audit
- deterministic event-stream / double-BOS audit
- full automated test suite

The next stage is empirical LightGBM / XGBoost baseline evaluation with
chronological OOS validation.
