# Market Structure Specification v1

Status: LOCKED — brainstorming baseline  \
Scope: deterministic market-structure engine for the Market Entry Engine

## 1. Core hierarchy

```
CANDLE VALID
    ↓
PULLBACK / SWING CANDIDATE
    ↓
SWING VALID
    ↓
BREAK OF VALID SWING
    ↓
STRUCTURE VALID
```

The engine must distinguish structural state from candle color and direction.

## 2. Candle validity

- A structural candle is valid when it breaks the high or low of another structural/reference candle.
- Break of high = UP movement.
- Break of low = DOWN movement.
- A candle can break both sides; this is an OUTSIDE case.
- Bullish/bearish is independent of structural direction:
  - close > open = bullish
  - close < open = bearish
- Wick break is sufficient for structural validation. Body break is NOT required.
- Body-break rules, if later useful, belong to setup/execution logic, not core structure validation.

## 3. Inside candle

A candle is INSIDE when:

```
high_current <= high_reference
low_current  >= low_reference
```

Inside candles do not count as separate structural candles.

The structural engine therefore operates on a reduced sequence of structural candles rather than blindly on every raw candle.

## 4. Outside candle

A candle is OUTSIDE when:

```
high_current > high_reference
low_current  < low_reference
```

An outside candle breaks both structural sides and represents/replaces candles contained within its range for structural counting.

Nested inside/outside reference behavior still requires explicit edge-case specification before implementation.

## 5. Pullback / swing candidate

- Every visible turning corner is initially only a PULLBACK / SWING CANDIDATE.
- A candidate is not automatically a swing.
- A failed/unconfirmed pullback must not become a canonical structure node.
- Multiple pullbacks may occur inside one directional leg.
- Pullbacks are **not canonical market-structure nodes**.
- A pullback's role is to act as a **validator for a future swing** and as an **inducement/reference point** for reversal validation.
- Creating, extending, or failing a pullback does not by itself create a new HH/HL/LH/LL node or a new canonical zig-zag in market structure.

Conceptually:

```
P1 → P2 → P3 → P4
```

can all be pullbacks while the canonical market structure remains a straight directional leg.

## 6. Active pullback validation

Within one directional leg, the latest active pullback is the candidate used for validation.

If the active pullback is broken in the opposing direction:
- the pullback is confirmed;
- its associated extreme becomes a SWING VALID.

If the active pullback is broken in the same direction:
- that candidate fails as a validator for the current extreme;
- the directional leg continues;
- a newer pullback candidate may become active;
- the canonical market structure remains the same directional leg.

Only ONE relevant pullback needs to be broken to validate the associated swing. Earlier pullbacks do not all have to be broken.

## 7. Valid swing and associated extreme

A swing becomes valid only after the corresponding pullback has been broken by a reversal.

The **associated extreme** is the extreme point of the directional leg that produced that pullback.

Therefore:

### Swing high

```
directional leg UP
       ↓
  EXTREME HIGH
       ↓
   PULLBACK LOW
       ↓
reversal breaks PULLBACK LOW
       ↓
EXTREME HIGH = SWING HIGH VALID
```

### Swing low

```
directional leg DOWN
       ↓
   EXTREME LOW
       ↓
  PULLBACK HIGH
       ↓
reversal breaks PULLBACK HIGH
       ↓
EXTREME LOW = SWING LOW VALID
```

The break point itself is NOT the swing point. The extreme that produced the confirmed pullback is the swing point.

Important:
- The swing is considered known at the time the confirming break occurs.
- Historical visualization may place the resulting swing marker at its associated extreme.
- Training/event data must preserve the actual confirmation time to avoid look-ahead bias.

## 8. Valid structure

A valid structure does NOT require two or more swings.

A single already-valid swing can be sufficient when that valid swing is subsequently broken.

```
SWING VALID
     ↓
SWING VALID IS BROKEN
     ↓
STRUCTURE VALID EVENT
```

Therefore structure is an event/state consequence of a valid swing being broken, not merely a visual zig-zag.

## 9. HH / HL / LH / LL

HH/HL/LH/LL labels are assigned only to VALID SWINGS.

They must never be assigned to an unconfirmed pullback/candidate.

Examples:
- next valid swing high above previous valid swing high → HH
- next valid swing low above previous valid swing low → HL
- next valid swing high below previous valid swing high → LH
- next valid swing low below previous valid swing low → LL

These labels are derived from valid swing relationships, not rolling highs/lows.

## 10. Internal and external structure

Internal structure can form inside an existing larger external boundary.

Example concept:

```
External: HH ───────────────┐
                            │
                 internal   │
                 swings     │
                            │
External: HL ───────────────┘
```

Valid swings and structure events that occur inside the external HH/HL boundary are internal structure.

When the major external boundary is broken, the previous structural context must be re-evaluated/rebuilt.

No arbitrary price-distance threshold is required to define internal structure; the existing structural boundary defines its scope.

## 11. Canonical principle

The engine must not create zig-zag structure merely because a chart visually contains corners.

Pullbacks are auxiliary validation/reference objects, not canonical structure nodes.

Until a pullback is confirmed:

```
NO VALID SWING
→ NO HH/HL/LH/LL NODE
→ NO CANONICAL STRUCTURE NODE
```

Even when a directional leg contains many pullbacks:

```
EXTREME → P1 → P2 → P3 → P4
```

the canonical market structure remains a straight leg until an active pullback is broken in the opposing direction and the associated extreme becomes a valid swing.

## 12. Explicit unresolved items before coding

The following must be formalized and tested before implementation:

1. Exact recursive reference behavior for nested INSIDE/OUTSIDE sequences.
2. Ordering when one candle breaks multiple pullbacks or structural levels simultaneously.
3. Exact reset/rebuild rules after a valid swing is broken.
4. Minimum information needed to initialize the first external structure.
5. Exact event ordering for simultaneous UP/DOWN breaks.
6. As-of-time representation so no future-confirmed swing leaks into historical model features.

## 13. Planned validation sequence

Before production code:

```
Candle validity
→ Inside
→ Outside
→ Pullback
→ Failed pullback
→ Pullback → valid swing
→ Multiple pullbacks
→ Valid swing → structure break
→ Two-sided break
→ Internal structure
→ External boundary break / rebuild
```

This document is the current locked conceptual baseline, not yet the final executable specification.
