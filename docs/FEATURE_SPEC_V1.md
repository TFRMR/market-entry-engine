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

## 3. FVG

- fvg_present
- fvg_direction
- fvg_size
- fvg_size_atr
- fvg_age_bars
- fvg_distance
- fvg_distance_atr
- fvg_position

FVG must be detected from past/current candles only and represented as a zone with explicit creation time.

## 4. Order Block

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

## 5. Liquidity

- liquidity_level_present
- liquidity_direction
- distance_to_liquidity
- distance_to_liquidity_atr
- liquidity_sweep
- sweep_direction
- sweep_size_atr

Liquidity levels must have an explicit historical creation/observation rule.

## 6. Price action / displacement

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

## 7. Volatility

- atr
- atr_change
- atr_percentile

ATR is normalization/context, not a fixed TP/SL rule.

## 8. Location

- distance_to_support_atr
- distance_to_resistance_atr
- leg_position
- distance_to_next_structure_level_atr
- distance_to_nearest_fvg_atr
- distance_to_nearest_ob_atr
- distance_to_nearest_liquidity_atr

## 9. MTF

Reserved for a later specification. MTF features must follow the same as-of-time rule.

## 10. Forbidden leakage

At candle t, features must not use:

- a swing whose confirmation occurs after t;
- a BOS that occurs after t;
- future FVG/OB/liquidity levels;
- future target/exit information;
- future candle highs/lows except where the feature definition itself is explicitly based on information available at t.

## 11. Output contract

The feature engine should produce:

1. feature values;
2. feature timestamp;
3. source/event timestamp for structural objects where relevant;
4. validity/as-of timestamp;
5. enough metadata to audit why a structural feature has its current value.

## 12. CHoCH semantics

CHoCH is a deterministic structure-transition event derived from BOS semantics:

- BULLISH_CHOCH occurs when a bullish BOS breaks a previously confirmed valid swing while the canonical structure direction immediately before the candle is DOWN.
- BEARISH_CHOCH occurs when a bearish BOS breaks a previously confirmed valid swing while the canonical structure direction immediately before the candle is UP.
- The initial BOS is not CHoCH because there is no prior canonical direction.
- CHoCH is an event/fact, not an entry signal by itself.
- A CHoCH does not by itself prove a completed reversal; subsequent structure events remain authoritative.
- If one candle produces both directional BOS events, CHoCH classification uses the pre-candle canonical direction and preserves deterministic event ordering.
- CHoCH must use the same confirmation-time and no-look-ahead rules as BOS.
