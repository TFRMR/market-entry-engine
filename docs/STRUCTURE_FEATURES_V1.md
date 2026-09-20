# Structure Feature Contract v1

Status: **IMPLEMENTATION BASELINE**

This document defines the contract between the deterministic Market Structure
engine and downstream feature engineering.

## 1. Purpose

Market Structure is deterministic and event-time aware. Feature engineering
must preserve that property when converting structure output into tabular
features for later statistical or ML models.

The central rule is:

> A structure fact becomes a feature only at the candle where that fact is knowable.

A historical swing location is not available before its confirmation candle.

## 2. Source

The feature layer consumes the deterministic structural sequence and event processor.

The structural candle index is the original positional candle index in the input frame.

## 3. Event features

The first structure feature set exposes these sparse event flags:

| Feature | Meaning |
|---|---|
| `structure_bullish_bos` | bullish BOS confirmed on this candle |
| `structure_bearish_bos` | bearish BOS confirmed on this candle |
| `structure_swing_high_valid` | valid swing high confirmed on this candle |
| `structure_swing_low_valid` | valid swing low confirmed on this candle |
| `structure_internal_bos` | at least one internal BOS on this candle |
| `structure_external_bos` | at least one external BOS on this candle |
| `structure_internal_swing` | at least one internal swing validation on this candle |
| `structure_external_swing` | at least one external swing validation on this candle |
| `structure_hh` | HH confirmed on this candle |
| `structure_hl` | HL confirmed on this candle |
| `structure_lh` | LH confirmed on this candle |
| `structure_ll` | LL confirmed on this candle |

Event flags are 0/1 and are not forward-filled.

A candle may legitimately contain multiple event flags. In particular, an
OUTSIDE candle can produce both bullish and bearish BOS events.

## 4. Confirmed swing context

The feature layer also exposes:

- `structure_last_valid_high`
- `structure_last_valid_low`

These are the latest confirmed valid swing prices, forward-filled only after
their confirmation candle.

They must remain unavailable before confirmation.

Example:

```text
swing location:       candle 100
pullback:             candle 102
confirmation:         candle 105

structure_last_valid_high
candle 100-104:       unavailable
candle 105 onward:    available
```

This preserves the distinction between historical location and information availability.

## 5. No-look-ahead contract

The feature builder may process the complete historical frame, but it must
never expose a future-confirmed structure fact to an earlier row.

Therefore:

- no swing feature is written at the swing location;
- swing features are written at confirmation time;
- BOS features are written at the BOS candle;
- last confirmed swing prices become available at confirmation and are then carried forward;
- future events must not alter prior feature rows.

## 6. Event ordering

The feature representation inherits the Market Structure v1 ordering:

1. BOS against previously confirmed valid swings;
2. bullish BOS before bearish BOS when both occur;
3. current-candle structural processing;
4. swing validation;
5. external-boundary rebuild.

The feature table records the resulting facts at the candle where they become
known; it does not reinterpret or reorder the underlying structure events.

## 7. Scope

`INTERNAL` and `EXTERNAL` are retained as separate event flags because scope
has structural meaning and should not be collapsed into a single generic BOS
or swing feature.

No distance threshold is used to infer scope.

## 8. Separation from raw/context features

Structure features are separate from:

- candle geometry;
- volatility;
- EMA context;
- support/resistance windows;
- recent returns;
- volume features.

Those feature families may later be combined into the model matrix, but their
semantics remain independent.

## 9. Active structural context

The first active-context feature set now exposes:

| Feature | Meaning |
|---|---|
| `structure_direction` | active structural leg: `1` = UP, `-1` = DOWN, `0` = not established |
| `structure_distance_to_high` | latest confirmed valid high minus current close |
| `structure_distance_to_low` | current close minus latest confirmed valid low |
| `structure_bars_since_last_swing` | raw candle-index distance from the latest confirmed swing |
| `structure_bars_since_last_bos` | raw candle-index distance from the latest BOS |

Distance features remain unavailable until the corresponding confirmed swing
exists. When an external-boundary rebuild closes the active context, the old
high/low levels are cleared and remain unavailable until a new swing is
confirmed in the rebuilt context. Age features likewise reflect the active
context and are not carried across a rebuild.

The context is sampled after each accepted structural candle and then carried
forward across raw candles that are INSIDE the current structural reference.
This preserves the structural engine's inside-candle semantics.

## 10. No-look-ahead extension

Active context follows the same information-availability rule as event flags:

- current direction reflects state after the current candle;
- a swing becomes available at confirmation time;
- BOS age starts at zero on the BOS candle;
- future structure cannot alter an earlier row;
- external-boundary rebuilds clear the closed context rather than forward-filling its levels;
- historical swing location is never substituted for confirmation time.

## 11. Future extensions

The next structure-feature extensions should be added only with an explicit
contract and regression test, especially for:

- normalized structure distances using historical volatility;
- setup-state features derived from multiple structure events;
- execution-oriented state that depends on explicitly defined entry timing.

These extensions must preserve the same confirmation-time/no-look-ahead rule.
