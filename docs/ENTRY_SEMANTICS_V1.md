# Entry Semantics v1

Status: **CONTRACT BASELINE**

This document defines the deterministic bridge from confirmed Market Structure events
to a testable trade candidate. It does not claim that the resulting setup is
profitable; it defines when and how a candidate exists.

## 1. Purpose

The entry engine must separate:
- structural fact;
- setup qualification;
- entry timing;
- invalidation;
- target;
- execution cost.

A model may later learn whether a candidate is worth taking, but the baseline
must first be deterministic and reproducible.

## 2. Setup trigger

Version 1 uses a confirmed BOS as the setup trigger.

A candidate is created only when:
- structure_bullish_bos == 1,
- or structure_bearish_bos == 1.

The BOS must already be confirmed by the deterministic structure engine.

A setup is not created from:
- a raw high/low breakout;
- an unconfirmed swing;
- a pullback break;
- a future-confirmed event;
- an event inferred from a later candle.

## 3. Direction

- bullish BOS -> LONG candidate;
- bearish BOS -> SHORT candidate.

If both BOS events occur on the same candle, both candidates are structurally
valid and must be represented independently. The entry engine must not silently
collapse the two directions.

## 4. Entry timing

Version 1 uses **next-candle open** as the executable entry price.

For a setup confirmed on candle t:
- setup information becomes known at t;
- entry is attempted at candle t + 1 open;
- candle t itself cannot be used as the execution price.

This removes intrabar ambiguity from the deterministic baseline.

## 5. Invalidation

For a LONG candidate, the baseline invalidation level is the latest confirmed
valid swing low available at setup confirmation.

For a SHORT candidate, the baseline invalidation level is the latest confirmed
valid swing high available at setup confirmation.

The level must exist at the BOS confirmation candle. Otherwise the candidate is
**structurally incomplete** and is not executable by the baseline.

The invalidation level is fixed when the setup is created. Later swings do not
retroactively move the original stop.

## 6. Risk unit

For LONG:
risk = entry_price - invalidation_price

For SHORT:
risk = invalidation_price - entry_price

A candidate is executable only when risk > 0.

A non-positive risk distance is rejected rather than repaired by moving the stop.

## 7. Target

Version 1 does not hard-code a fixed price or fixed-R target into the setup
candidate. Target construction belongs to the target/exit layer and must use
only levels observable at the decision time.

The candidate therefore carries entry and invalidation semantics first. A later
risk/exit layer may attach one or more deterministic exit areas and calculate
R:R from the proposed entry and invalidation.

A fixed-R target may be used as an explicit experimental policy, but it is not
part of the structural setup contract.

## 8. Execution costs

The baseline cost model has three explicit components:
- spread;
- slippage;
- trading fee.

Costs must be supplied as configuration or data inputs rather than hidden inside
the signal logic.

For a LONG:
- entry executes at the configured next-open price plus entry-side execution cost;
- exit executes at the configured stop/target price minus exit-side execution cost.

For a SHORT, the directions are reversed.

The exact monetary conversion is deferred to the execution-cost implementation;
the contract requires costs to be explicit and applied consistently.

## 9. Candidate lifecycle

A deterministic candidate has these conceptual stages:
SETUP_CONFIRMED -> ENTRY_PENDING -> OPEN -> CLOSED

Possible terminal outcomes include:
- TARGET;
- INVALIDATION;
- END_OF_DATA.

A setup that lacks a valid next candle or valid positive risk does not enter the
open-trade lifecycle.

## 10. No-look-ahead

At setup candle t, the engine may use only information known by t.

At entry candle t + 1, execution uses the next candle's open.

A future candle may determine whether target or invalidation is reached, but it
must never modify:
- the original setup timestamp;
- the entry timestamp;
- the original invalidation level;
- the original target level.

## 11. Same-candle target/invalidation ambiguity

If a future candle touches both target and invalidation and OHLC data cannot
establish the intrabar order, version 1 must not invent an order.

The baseline outcome is AMBIGUOUS unless a higher-resolution execution source
is available.

## 12. Separation of concerns

The architecture is:
Market Structure -> Setup Candidate -> Execution Simulation -> Outcome

Market Structure determines what happened.

The setup layer determines whether a deterministic candidate exists.

The execution layer determines what price and outcome are realizable under the
configured cost assumptions.

Statistical models may later score candidates, but they do not redefine the
underlying structural facts.

## 13. Future extensions

Possible later policies include:
- structure-derived targets;
- trailing invalidation;
- multi-target exits;
- time-based invalidation;
- volatility-normalized targets;
- spread/slippage from historical data;
- position sizing and portfolio constraints.

Each extension requires its own explicit contract and regression tests.