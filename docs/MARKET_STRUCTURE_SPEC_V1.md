# Market Structure Specification v1

Status: IMPLEMENTATION BASELINE — internal/external scope and boundary rebuild implemented  \
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

When an OUTSIDE candle extends the active directional extreme while also crossing the active pullback, the extreme extension takes precedence: the old pullback does not validate the prior extreme on that candle. The directional leg continues from the new extreme.

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

## 10. BOS — Break of Structure

BOS is a break of an already confirmed valid swing.

### Bullish BOS

```
VALID SWING HIGH
       ↓
price breaks swing high
       ↓
BULLISH_BOS
```

### Bearish BOS

```
VALID SWING LOW
       ↓
price breaks swing low
       ↓
BEARISH_BOS
```

Rules:

- A pullback break is NOT a BOS. It validates the associated swing.
- A swing must already be confirmed before it can become a BOS target.
- A swing confirmed on the current candle is not eligible as a BOS target on that same candle.
- Wick breaks are sufficient.
- An OUTSIDE candle may legitimately produce both bullish and bearish BOS events when it breaks two previously confirmed valid swings.
- BOS targets confirmed structure, not merely candle color or current leg direction.

Historical swing location is distinct from information availability:

```
swing historical location
        ≠
swing confirmation time
```

The engine must use confirmation time for event/features so future-confirmed structure cannot leak into historical processing.

## 11. First BOS and structure initialization

Historical candles are used as warm-up/context to establish the first usable structural anchor.

The engine must not define structure using an arbitrary fixed candle count.

```
RAW HISTORY
    ↓
STRUCTURAL CANDLES
    ↓
VALID SWINGS
    ↓
FIRST CONFIRMED BOS
    ↓
STRUCTURE CHECKPOINT
    ↓
FORWARD PROCESSING
```

`StructureCheckpoint` represents the engine state immediately after the BOS candle has been processed.

It preserves:

```
StructureCheckpoint
├── index
├── direction
├── extreme
├── pullback
├── last_swing
├── previous_swing
├── scope
├── last_high
├── last_low
├── broken_high_index
└── broken_low_index
```

This enables:

```
PASS 1
historical warm-up
→ first BOS
→ checkpoint

PASS 2
candle after BOS
→ continue structure
```

The checkpoint is an engine state boundary, not a new market-structure concept.

## 12. Internal and external structure

Internal structure can form inside an existing larger external boundary.

```
External: HH ───────────────┐
                            │
                 internal   │
                 swings     │
                            │
External: HL ───────────────┘
```

Valid swings and structure events inside the existing external boundary are internal structure.

When the major external boundary is broken, the previous structural context is closed and rebuilt forward from the boundary-break candle.

No arbitrary price-distance threshold is required to define internal structure; the existing structural boundary defines its scope.

Scope is assigned from the active external boundary, not from a fixed distance threshold. A valid swing that remains inside the active external high/low boundary is INTERNAL. A break of an active external boundary is EXTERNAL and terminates the previous external context.

### Scope transition

- The active external boundary is the latest confirmed external swing high and external swing low that define the current outer range.
- A newly confirmed valid swing remains INTERNAL when its price is inside that outer range.
- An internal swing does not replace the corresponding external boundary.
- When price breaks an active external boundary, the break is emitted as the applicable BOS event and the previous external context is considered closed.
- The engine rebuilds state from the boundary-break candle using only information available at and after that point; prior internal swings are discarded as targets and are never retroactively promoted to external status.
- After rebuild, the first newly established structural range becomes the new external context.

This scope model is structural: no arbitrary price-distance, candle-count, or volatility threshold is used.

## 13. Explicit unresolved items

The following remain before Market Structure v1 is considered complete:

1. Exact recursive reference behavior for nested INSIDE/OUTSIDE sequences.
2. Exact event ordering for simultaneous structural transitions beyond the currently locked OUTSIDE/extreme-extension rule.

Already resolved in the current implementation:

- first usable BOS can establish the structural anchor;
- required state can be captured in `StructureCheckpoint`;
- processing can continue from the checkpoint;
- swing confirmation time is distinct from historical swing location;
- future-confirmed swings are unavailable before confirmation;
- BOS targets only previously confirmed valid swings;
- internal/external scope is determined by the active structural boundary;
- valid swings inside an established external high/low range are classified INTERNAL;
- external swing candidates update the corresponding external boundary;
- external-boundary break/rebuild behavior is implemented and validated by the focused regression test.

## 14. Planned validation sequence

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
→ First BOS / checkpoint
→ Internal structure
→ External boundary break / rebuild
→ scope transition validation
```

This document is the implementation baseline for the deterministic Market Structure layer. Core rules marked above are locked. Internal/external scope and external-boundary rebuild are now implemented against the locked contract; only the remaining section 13 items must be resolved before Market Structure v1 is closed.
