# Structure Feature Contract v1

Status: **IMPLEMENTATION BASELINE**

## 1. Purpose

Market Structure is the primary deterministic foundation for downstream
features. It is directional, event-time aware, and explicitly separates
INTERNAL and EXTERNAL structure.

The central rule is:

> A structure fact becomes a feature only at the candle where that fact is knowable.

## 2. Structure event features

The feature layer exposes sparse event flags for:

- bullish / bearish BOS
- bullish / bearish CHOCH
- valid swing HIGH / LOW
- INTERNAL / EXTERNAL BOS
- INTERNAL / EXTERNAL swing validation
- HH / HL / LH / LL

Event flags are not forward-filled.

A candle may contain multiple events. In particular, an OUTSIDE candle may
produce both bullish and bearish BOS events. Their ordering is deterministic.

## 3. Confirmed swing context

Current confirmed swing fields:

- `structure_last_valid_high`
- `structure_last_valid_low`

These become available at confirmation and are then carried forward.

The swing's historical location is not treated as its information-availability
time.

## 4. Active structural context

Active Structure owns:

| Feature | Meaning |
|---|---|
| `structure_direction` | active structural leg: 1 = UP, -1 = DOWN, 0 = not established |
| `structure_distance_to_high` | latest confirmed valid high minus current close |
| `structure_distance_to_low` | current close minus latest confirmed valid low |
| `structure_historical_distance_to_high` | historical confirmed high distance |
| `structure_historical_distance_to_low` | historical confirmed low distance |
| `structure_bars_since_last_swing` | positional distance from latest confirmed swing |

BOS age is **not** owned by Active Structure.

## 5. Event Sequence

Event Sequence owns chronological structural-event history:

- `structure_event_count`
- `structure_last_event_index`
- `structure_last_event_age`
- `structure_last_event_type`
- `structure_last_event_direction`
- `structure_last_event_scope`
- `structure_previous_event_index`
- `structure_previous_event_age`
- `structure_previous_event_type`
- `structure_previous_event_direction`
- `structure_previous_event_scope`
- `structure_events_since_last_bos`
- `structure_events_since_last_swing`
- `structure_bars_since_last_bos`
- `structure_bars_since_last_swing`

This keeps event chronology in the event layer rather than duplicating BOS age
inside Active Structure.

## 6. INTERNAL / EXTERNAL scope

INTERNAL and EXTERNAL are separate structural scopes and remain explicit in the
feature contract.

Scope is determined by the deterministic structure engine. It is not inferred
from an arbitrary distance threshold or technical indicator.

The purpose is to allow downstream research to distinguish local structural
events from broader external structural context.

## 7. No-look-ahead contract

- swing features become available at confirmation
- BOS / CHOCH features are emitted on the event candle
- event history contains only events known by the current candle
- future-confirmed structure cannot alter an earlier row
- active-context rebuilds clear closed active levels where required
- historical confirmed structure remains separately available when defined by
  the contract

## 8. Relationship to POI research

Market Structure does not itself define a POI ranking.

Instead, it provides structural facts that can later be combined with:

- directional context
- INTERNAL / EXTERNAL scope
- BOS / CHOCH sequence
- liquidity
- FVG
- Order Block
- S/R and location
- displacement / price action
- pullback state

Whether a combination corresponds to a useful POI must be determined by
empirical, chronological, and out-of-sample evaluation.

## 9. Validation

Structure feature development is quality-gated by the automated test suite and
point-in-time chronology audits. The current implementation has passed the
full feature/event chronology checks and deterministic event-stream audits.
