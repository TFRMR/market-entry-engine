# Feature Specification v1

Status: DRAFT — methodology baseline
Scope: deterministic structure + market context features for historical inference

## 1. Principles

- Structure features come only from the deterministic market-structure engine.
- No rolling-high/rolling-low feature may be used as a substitute for valid swings.
- Every historical feature must be available at its as-of candle; confirmed future information must not leak backward.
- Continuous measurements should be preserved where possible; fixed trading thresholds belong to evaluation/risk logic, not feature construction.
- AI receives facts/context, not a pre-computed heuristic score.

## 2. Structure

| Feature | Type | Meaning |
|---|---|---|
| structure_direction | categorical | Current canonical directional leg |
| structure_scope | categorical | INTERNAL / EXTERNAL |
| structure_age_bars | numeric | Bars since current structure context began |
| active_extreme_type | categorical | HIGH / LOW |
| active_extreme_price | numeric | Current directional-leg extreme |
| active_extreme_distance_atr | numeric | Price distance from extreme normalized by ATR |
| last_valid_swing_type | categorical | HIGH / LOW |
| last_valid_swing_price | numeric | Last confirmed swing |
| last_valid_swing_age_bars | numeric | Bars since confirmation |
| last_swing_label | categorical | HH / HL / LH / LL |
| distance_to_last_swing_atr | numeric | Current price distance to last valid swing |
| bos_event | categorical/binary | Whether a valid-swing break occurred at this candle |
| bos_direction | categorical | UP / DOWN when BOS occurs |
| choch_event | categorical/binary | Whether a valid-swing break changes the canonical structure direction at this candle |
| choch_direction | categorical | UP / DOWN when CHoCH occurs |
| internal_boundary_distance_atr | numeric | Distance to relevant internal boundary |
| external_boundary_distance_atr | numeric | Distance to relevant external boundary |

## 3. Trend / regime and range

| Feature | Type | Meaning |
|---|---|---|
| trend_regime | categorical | BULLISH / BEARISH / NEUTRAL from deterministic structure direction |
| trend_transition | binary | CHoCH occurred at the current candle |
| trend_transition_direction | categorical | UP / DOWN when CHoCH occurs |
| range_state | categorical | DEFINED when both latest confirmed structural high/low form a positive-width range |
| range_high | numeric | Latest confirmed structural high available as of the candle |
| range_low | numeric | Latest confirmed structural low available as of the candle |
| range_width | numeric | Structural range high minus range low |
| range_position | numeric | Close location within the structural range; not clipped outside 0..1 |
| range_position_zone | categorical | LOW / MID / HIGH; UNDEFINED when no valid range exists |

Rules:

- Trend/regime is derived from deterministic structure direction; it does not use EMA or other indicator alignment.
- CHoCH is the transition event. A transition does not by itself establish a completed reversal.
- The range is a structural context range built from confirmed valid swing levels already available at the current candle.
- No fixed price-distance, candle-count, or volatility threshold is used to define the range.
- Range position may exceed 0..1 when price has moved outside the previously confirmed structural range; values are preserved rather than clipped because the excursion is information.
- Undefined structural history remains missing/undefined rather than being imputed.

## 5. FVG

- fvg_present
- fvg_direction
- fvg_size
- fvg_size_atr
- fvg_age_bars
- fvg_distance
- fvg_distance_atr
- fvg_position
- fvg_creation_timestamp

FVG v1 uses a deterministic three-candle imbalance:

- Bullish FVG: `low[t] > high[t-2]`.
- Bearish FVG: `high[t] < low[t-2]`.
- Bullish zone: `[high[t-2], low[t]]`.
- Bearish zone: `[high[t], low[t-2]]`.

Temporal semantics:

- An FVG is created and becomes available on candle `t`, never before `t`.
- The feature context uses the latest FVG created at or before the current candle.
- Historical FVGs are retained as context; v1 does not define a fill or invalidation rule.
- `fvg_age_bars` is based on positional candle index in the raw feature frame, so INSIDE candles still advance age.
- `fvg_creation_timestamp` records the source candle timestamp when available.

Measurement semantics:

- `fvg_size` is the positive zone width in price units.
- `fvg_size_atr` is `fvg_size / ATR` using ATR available on the creation candle. If creation ATR is unavailable or non-positive, the value is undefined.
- `fvg_distance` is the absolute price distance from close to the zone: zero while price is inside the zone, otherwise the distance to the nearest zone boundary.
- `fvg_distance_atr` is `fvg_distance / current ATR`. If current ATR is unavailable or non-positive, the value is undefined.
- `fvg_position` is `(close - zone_low) / zone_width`. Values below 0 indicate price below the zone, 0..1 indicate a close inside the zone, and values above 1 indicate price above the zone. The value is not clipped.

FVG is contextual information, not a deterministic trade signal.

## 6. Order Block

- ob_present
- ob_direction
- ob_size
- ob_size_atr
- ob_age_bars
- ob_distance
- ob_distance_atr
- ob_contains_price
- ob_relative_position

OB is a contextual hypothesis, not a deterministic trade signal.

## 4. Liquidity

| Feature | Type | Meaning |
|---|---|---|
| liquidity_high | numeric | Latest confirmed structural high available before the current candle |
| liquidity_low | numeric | Latest confirmed structural low available before the current candle |
| liquidity_high_present | binary | Whether a confirmed high is available before the current candle |
| liquidity_low_present | binary | Whether a confirmed low is available before the current candle |
| distance_to_liquidity_high | numeric | High liquidity level minus current close |
| distance_to_liquidity_low | numeric | Current close minus low liquidity level |
| liquidity_high_sweep | binary | Current high breaches the prior confirmed high and current close reclaims at or below it |
| liquidity_low_sweep | binary | Current low breaches the prior confirmed low and current close reclaims at or above it |
| liquidity_sweep | binary | Either liquidity sweep occurred on the current candle |
| liquidity_sweep_direction | categorical | BEARISH for high sweep, BULLISH for low sweep, BOTH when both occur, otherwise NONE |
| liquidity_sweep_size | numeric | Price distance beyond the swept liquidity level |

Rules:

- Liquidity is derived from confirmed structural swing highs/lows; rolling highs/lows are not used as liquidity substitutes.
- The liquidity level for candle t comes from the latest confirmed structural level available before candle t.
- A high sweep requires `high > liquidity_high` and `close <= liquidity_high`.
- A low sweep requires `low < liquidity_low` and `close >= liquidity_low`.
- A clean close through a liquidity level is not classified as a sweep; structural BOS remains the authoritative break event.
- No equal-high/equal-low tolerance is introduced in this first deterministic layer. Equal-level liquidity pools can be added later only with an explicit, data-safe definition.
- Sweep direction describes the rejection implied by the reclaim: high sweep is BEARISH, low sweep is BULLISH.


## 7. Price action / displacement

- candle_range
- candle_body
- body_ratio
- range_atr
- body_atr
- upper_wick_ratio
- lower_wick_ratio
- upper_wick_to_body
- lower_wick_to_body
- close_position_in_range

## 8. Volatility

- atr
- atr_change
- atr_percentile

ATR is normalization/context, not a fixed TP/SL rule.

## 9. Location

- distance_to_support_atr
- distance_to_resistance_atr
- leg_position
- distance_to_next_structure_level_atr
- distance_to_nearest_fvg_atr
- distance_to_nearest_ob_atr
- distance_to_nearest_liquidity_atr

## 10. MTF

Reserved for a later specification. MTF features must follow the same as-of-time rule.

## 11. Forbidden leakage

At candle t, features must not use:

- a swing whose confirmation occurs after t;
- a BOS that occurs after t;
- future FVG/OB/liquidity levels;
- future target/exit information;
- future candle highs/lows except where the feature definition itself is explicitly based on information available at t.

## 12. Output contract

The feature engine should produce:

1. feature values;
2. feature timestamp;
3. source/event timestamp for structural objects where relevant;
4. validity/as-of timestamp;
5. enough metadata to audit why a structural feature has its current value.

## 13. CHoCH semantics

CHoCH is a deterministic structure-transition event derived from BOS semantics:

- BULLISH_CHOCH occurs when a bullish BOS breaks a previously confirmed valid swing while the canonical structure direction immediately before the candle is DOWN.
- BEARISH_CHOCH occurs when a bearish BOS breaks a previously confirmed valid swing while the canonical structure direction immediately before the candle is UP.
- The initial BOS is not CHoCH because there is no prior canonical direction.
- CHoCH is an event/fact, not an entry signal by itself.
- A CHoCH does not by itself prove a completed reversal; subsequent structure events remain authoritative.
- If one candle produces both directional BOS events, CHoCH classification uses the pre-candle canonical direction and preserves deterministic event ordering.
- CHoCH must use the same confirmation-time and no-look-ahead rules as BOS.
