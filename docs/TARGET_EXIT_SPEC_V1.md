# Target and Exit Specification v1

Status: **CONTRACT BASELINE**
Scope: historical targets for direction, exit areas, and invalidation

## 1. Separation of responsibilities

The model forecasts outcomes from historical states.

The deterministic engine defines market facts and candidate structural levels.

The risk engine converts predicted outcome areas into entry/TP/SL calculations.

The model must not hard-code rules such as "TP = 3 ATR" or "SL = 1.3 ATR".

## 2. Direction target

For each decision candle, define a forward observation window.

The dataset records which directional outcome occurs first under the selected target construction:

- UP outcome
- DOWN outcome
- unresolved within horizon

The exact horizon and excursion rules must be locked during label validation and must not depend on future-derived features.

## 3. Exit Area candidates

Exit areas are historical outcome targets derived from levels that are observable from the decision-time context.

Version 1 implements one deterministic source: **confirmed valid swings**.

For a LONG candidate, eligible exit areas are confirmed valid swing highs above
entry. For a SHORT candidate, eligible exit areas are confirmed valid swing lows
below entry.

Levels confirmed after the setup candle are excluded. This keeps exit-area
construction causal at setup time.

Later versions may add structure boundaries, liquidity, FVG, OB, or
support/resistance as separate source types.

A candidate target must have:

- source type;
- price/zone;
- creation/observation timestamp;
- distance from decision price;
- distance normalized by ATR.

## 4. Exit Area 1 / Area 2

Version 1 orders eligible swing levels by absolute distance from the candidate
entry and exposes at most two nearest areas. The source metadata and original
confirmation timestamp are retained.

The model may forecast reachability for two ordered target areas:

- EXIT_AREA_1: nearer actionable target;
- EXIT_AREA_2: farther target.

They are not required to be fixed ATR multiples.

For each area, historical labeling should record:

- reached before invalidation: yes/no;
- time-to-reach;
- maximum favorable excursion before invalidation;
- whether the area was touched or crossed;
- outcome status at horizon end.

## 5. Invalidation

Invalidation is tied to the structural/context condition that would invalidate the directional thesis.

Candidate sources:

- opposite valid swing;
- structure boundary;
- relevant pullback-derived invalidation level;
- other deterministic context level when explicitly defined.

The model may estimate outcome/risk context, but the final distance and R:R are calculated deterministically by the risk engine.

## 6. Probability outputs

The model output may contain:

- P(direction);
- P(reach EXIT_AREA_1 before invalidation);
- P(reach EXIT_AREA_2 before invalidation);
- optional P(invalidation before target).

Probabilities must be calibrated on out-of-sample data before being presented as user-facing percentages.

## 7. Entry

Entry is a separate decision from direction.

The system may define an entry area from current price + deterministic context, then evaluate whether expected target reachability and invalidation distance produce an acceptable risk profile.

No single probability threshold is hard-coded in this specification.

## 8. Risk calculation

For a proposed entry E:

- reward_1 = distance(E, EXIT_AREA_1)
- reward_2 = distance(E, EXIT_AREA_2)
- risk = distance(E, INVALIDATION)

The engine calculates R:R for each target.

A high directional probability does not automatically make a trade eligible.

## 9. Anti-lookahead

Future candles are used only to determine historical labels/outcomes.

At inference time, target probabilities must be based only on features and candidate levels known at the decision timestamp.

Historical swing visualization may point to a past extreme, but its confirmed availability begins at its confirmation timestamp.

## 10. Validation requirements

Before model training:

- verify target ordering;
- verify no target candidate uses future information;
- measure class balance;
- measure unresolved cases;
- inspect target reachability by structure state;
- verify calibration;
- verify that target construction is stable across time splits.
