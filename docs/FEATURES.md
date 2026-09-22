# Feature Engineering

## 1. Purpose

The Market Entry Engine is designed as a **momentum-candle specialist**.

The primary market event of interest is a candle whose body represents at least 80% of its total high-low range.

The purpose of feature engineering is therefore not to build a large collection of technical indicators.

Instead, the feature pipeline should describe the context surrounding a momentum candle and allow later analysis to determine which contexts are associated with favorable outcomes.

The project will use empirical analysis to determine which patterns have useful predictive value.

---

## 2. Core Hypothesis

The initial hypothesis is:

> A candle with a body-to-range ratio of at least 80% may contain useful information about short-term directional momentum.

The basic measurement is:

```text
range = high - low
body  = abs(close - open)

body_ratio = body / range
```

A candle qualifies as a momentum-candle candidate when:

```text
body_ratio >= 0.80
```

Direction:

```text
LONG  candidate: close > open
SHORT candidate: close < open
```

If `high == low`, the candle does not qualify as a momentum-candle candidate.

The 80% threshold is part of the initial strategy hypothesis. It should not be optimized against the available dataset before an out-of-sample validation methodology has been established.

---

## 3. Strategy Philosophy

The system should follow this structure:

```text
Validated OHLCV
       |
       v
Momentum Candle Detection
body / range >= 0.80
       |
       v
Context Analysis
       |
       +-- EMA 5 / 20 / 50
       |
       +-- Support / Resistance
       |
       +-- Breakout / Structure
       |
       +-- Candle Size / ATR
       |
       +-- Recent Price Behavior
       |
       v
Outcome Analysis
       |
       v
Probability / Model
       |
       +-- LONG
       +-- SHORT
       +-- NO TRADE
```

The momentum candle is the **event trigger**.

The remaining features describe the environment in which that event occurs.

---

# 4. Momentum-Candle Features

## 4.1 body_ratio

```text
body_ratio = abs(close - open) / (high - low)
```

This is the primary feature.

The initial trigger is:

```text
body_ratio >= 0.80
```

---

## 4.2 candle_direction

A categorical/binary representation of candle direction:

```text
bullish = close > open
bearish = close < open
```

Doji-like candles where:

```text
close == open
```

are not considered directional momentum candidates.

---

## 4.3 candle_body

```text
candle_body = abs(close - open)
```

This represents the absolute size of the candle body.

---

## 4.4 candle_range

```text
candle_range = high - low
```

This represents the complete candle movement.

---

## 4.5 upper_wick_ratio

```text
upper_wick = high - max(open, close)

upper_wick_ratio =
    upper_wick / candle_range
```

---

## 4.6 lower_wick_ratio

```text
lower_wick = min(open, close) - low

lower_wick_ratio =
    lower_wick / candle_range
```

---

## 4.7 close_position

```text
close_position =
    (close - low) / candle_range
```

This describes where the candle closes inside its total range.

Values near 1 indicate a close near the high.

Values near 0 indicate a close near the low.

---

# 5. Candle Size Relative to Volatility

A momentum candle should not be evaluated only by its body percentage.

A 10-point candle and a 100-point candle can both have a body ratio of 90%.

Therefore the initial analysis should also measure candle size relative to recent volatility.

## 5.1 ATR

The first volatility reference is ATR over 14 candles.

True Range:

```text
TR[t] = max(
    high[t] - low[t],
    abs(high[t] - close[t-1]),
    abs(low[t] - close[t-1])
)
```

Initial ATR:

```text
ATR14 = rolling_mean(TR, 14)
```

---

## 5.2 range_to_atr

```text
range_to_atr =
    candle_range / ATR14
```

This allows the analysis to distinguish relatively small momentum candles from volatility-expansion candles.

No specific threshold is assumed yet.

Possible categories can be evaluated later from the data.

---

# 6. EMA Context

The initial trend/context framework uses:

```text
EMA5
EMA20
EMA50
```

The purpose is not to generate an independent trading signal.

The purpose is to describe the trend context surrounding the momentum candle.

---

## 6.1 Price relative to EMA

Candidate features:

```text
close_vs_ema5
close_vs_ema20
close_vs_ema50
```

Defined as:

```text
close_vs_emaN = close / emaN - 1
```

---

## 6.2 EMA relationship

Candidate features:

```text
ema5_vs_ema20
ema20_vs_ema50
```

Defined as:

```text
ema5_vs_ema20 = ema5 / ema20 - 1
ema20_vs_ema50 = ema20 / ema50 - 1
```

---

## 6.3 EMA alignment

For analysis, we should identify structural states such as:

### Bullish alignment

```text
EMA5 > EMA20 > EMA50
```

### Bearish alignment

```text
EMA5 < EMA20 < EMA50
```

### Mixed alignment

All other configurations.

These states are descriptive features, not predetermined entry rules.

The data must determine whether EMA alignment actually changes the outcome of momentum candles.

---

# 7. Support and Resistance Context

Support/resistance is included as a **research context feature group**, not as a predefined trading signal.

The initial objective is to determine whether the location of a momentum candle relative to recent price structure affects continuation probability.

The canonical v1 S/R definition uses confirmed structural swings.

A swing confirmed on candle `t` is available to the location layer on candle `t`. This confirmation-time boundary is intentionally different from the stricter post-confirmation boundary used when structural BOS consumes swings. `HIGH` swings are resistance candidates and `LOW` swings are support candidates. Only swings with `confirmation_index <= current candle index` are available. Support is the nearest confirmed `LOW <= close`; resistance is the nearest confirmed `HIGH >= close`. If no qualifying level exists, the value is `NaN`.


Legacy rolling references remain available for research and ablation:

```text
previous_high_20
previous_low_20
previous_high_50
previous_low_50
```

The implementation must avoid using the current candle's high/low when calculating a **pre-existing** resistance/support level.

For example, a previous-high reference should be based on candles before the momentum candle:

```text
previous_high_20 =
    highest high over the previous 20 candles
```

not:

```text
highest high including the current candle
```

This distinction is important for avoiding look-ahead contamination in structure features.

Canonical location fields:

```text
support_level
resistance_level
distance_to_support
distance_to_resistance
distance_to_support_atr
distance_to_resistance_atr
distance_to_next_structure_level
distance_to_next_structure_level_atr
leg_position
```

ATR only normalizes structural distance; it does not define S/R. In `UP`, next structure is the nearest confirmed `HIGH` strictly above close. In `DOWN`, it is the nearest confirmed `LOW` strictly below close. Nearest FVG, order block, and liquidity location features remain deferred.

# 8. FVG Reference Layer

FVG transition analysis requires all historical FVG references, not only the latest FVG exposed by the existing feature context.

`FVGReference` is an immutable historical object containing:

```text
creation_index
direction
lower
upper
size
creation_timestamp
```

The reference is created on candle `t` using only candles `t-2` and `t`:

```text
bullish: low[t] > high[t-2]
bearish: high[t] < low[t-2]
```

Both comparisons are strict, so equal boundaries do not create a gap.

The reference becomes available on its creation candle. The builder preserves all valid historical FVGs in chronological creation order.

This layer is deliberately separate from `add_fvg_features()`, whose existing contract is to expose only the latest historical FVG context.

`FVGReference` does not contain target, previous/next, interaction, rejection, or transition-outcome fields. Those concepts belong to the later FVG Transition layer and require their own temporal definitions.

---

# 8.1 FVG Transition Reference Routing

The FVG Transition v1 routing layer maps the current price to spatial FVG references without producing a trade signal or transition outcome.

## Temporal availability

Only FVG references satisfying `creation_index <= current_index` are available at the current candle.

## Origin

The origin is the FVG containing the current close:

`lower <= close <= upper`

Exactly one containing FVG is required.

- zero containing FVGs -> `origin = None`
- multiple containing FVGs -> `origin = None` because the spatial relationship is ambiguous

Creation order is not used to resolve overlapping origin candidates.

## Target

For `UP`, target candidates satisfy `lower > close`. The candidate with the smallest `lower` is selected.

For `DOWN`, target candidates satisfy `upper < close`. The candidate with the largest `upper` is selected.

Target routing therefore follows the nearest fully separated FVG in the direction of travel.

## Next-after-target

For `UP`, candidates must satisfy `candidate.lower > target.upper`. The candidate with the smallest `lower` is selected.

For `DOWN`, candidates must satisfy `candidate.upper < target.lower`. The candidate with the largest `upper` is selected.

Overlapping FVGs are not forced into a spatial ordering.

## Scope

FVG Transition v1 currently provides:

- origin reference
- target reference
- next-after-target reference
- temporal availability
- directional spatial routing
- explicit overlap ambiguity

It does not determine rejection, acceptance, break, previous/next reach, transition outcome, probability, or BUY/SELL signals.

# 8.1 FVG Transition Reference Routing

FVG Transition v1 routes the current price through historical FVG references without producing a trade signal or transition outcome.

## Temporal availability

Only references satisfying `creation_index <= current_index` are available.

## Origin

Origin is the FVG containing current close: `lower <= close <= upper`. Exactly one containing FVG is required. Zero or multiple containing FVGs produce `origin = None`. Creation order is not used to resolve overlap.

## Target

For `UP`, candidates satisfy `lower > close`; select the smallest `lower`. For `DOWN`, candidates satisfy `upper < close`; select the largest `upper`.

## Next-after-target

For `UP`, candidates satisfy `candidate.lower > target.upper`; select the smallest `lower`. For `DOWN`, candidates satisfy `candidate.upper < target.lower`; select the largest `upper`.

Overlapping FVGs are not forced into a spatial ordering.

## Scope

This layer provides origin, target, next-after-target, temporal availability, directional spatial routing, and explicit overlap ambiguity. It does not determine rejection, acceptance, break, reach, transition outcome, probability, or BUY/SELL signals.

# 9. Distance to Structure

Legacy distance features:

```text
distance_to_previous_high_20
distance_to_previous_low_20
distance_to_previous_high_50
distance_to_previous_low_50
```

A normalized representation may use current price:

```text
distance_to_high =
    (previous_high - close) / close

distance_to_low =
    (close - previous_low) / close
```

The exact representation will be frozen when the feature implementation is written.

The purpose is to answer questions such as:

* Does momentum continuation behave differently near a recent high?
* Does a bullish momentum candle behave differently after breaking a recent high?
* Does a bearish momentum candle behave differently after breaking a recent low?
* Are momentum candles in the middle of a range less informative?

These are hypotheses to test, not assumptions.

---

# 10. Breakout Context

A momentum candle may occur:

1. inside an existing range;
2. near a structural boundary;
3. through a previous high;
4. through a previous low.

Therefore the initial feature analysis should include breakout state.

Examples:

```text
break_previous_high_20
break_previous_low_20

break_previous_high_50
break_previous_low_50
```

A bullish breakout candidate could be represented by:

```text
close > previous_high_20
```

A bearish breakout candidate:

```text
close < previous_low_20
```

The analysis should distinguish breakout momentum from momentum occurring without a structural break.

---

# 11. Recent Price Context

The system should know what happened immediately before the momentum candle.

Initial candidates:

```text
return_3
return_6
return_12
```

where:

```text
return_N = close[t] / close[t-N] - 1
```

These features allow us to investigate questions such as:

* Does momentum work better after consolidation?
* Does momentum work better when price was already moving in the same direction?
* Does an extremely extended move reduce continuation probability?

Again, these relationships should be discovered empirically.

---

# 12. Volume Context

The current XAUUSDc dataset provides meaningful tick volume but zero-valued real volume.

Therefore the initial volume analysis uses:

```text
tick_volume
```

Potential features:

```text
volume_ratio_20
volume_change_1
```

where:

```text
volume_ratio_20 =
    tick_volume / rolling_mean(tick_volume, 20)
```

and:

```text
volume_change_1 =
    tick_volume[t] / tick_volume[t-1] - 1
```

Volume should initially be treated as contextual information rather than a mandatory entry filter.

---

# 13. Research Questions

The first feature-analysis stage should answer empirical questions rather than optimize a trading rule.

### Momentum quality

* How often does `body_ratio >= 0.80` occur?
* What percentage are bullish?
* What percentage are bearish?
* What is the distribution of candle size relative to ATR?

### Directional continuation

After a momentum candle:

* What happens after 1 candle?
* What happens after 3 candles?
* What happens after 5 candles?
* What happens after 10 candles?

### EMA context

Compare momentum-candle outcomes across:

```text
bullish alignment
bearish alignment
mixed alignment
```

and relative price position to EMA5/20/50.

### Structural context

Compare:

```text
breakout
near resistance/support
inside range
```

### Volatility context

Compare different ranges of:

```text
range_to_atr
```

without assuming the optimal threshold beforehand.

### Volume context

Compare outcomes under different tick-volume conditions.

---

# 14. Outcome Must Be Separate From Features

Feature engineering must not encode the future outcome.

For example:

```text
feature at t
```

may use:

```text
OHLCV <= t
```

but the outcome may use:

```text
future candles t+1 ... t+N
```

The future information belongs exclusively to the labeling/outcome-analysis stage.

This separation is essential for avoiding look-ahead bias.

---

# 15. NO TRADE

The system explicitly supports:

```text
LONG
SHORT
NO TRADE
```

A momentum candle does not automatically become a trade.

The purpose of contextual analysis and later modeling is to determine whether a particular momentum event has sufficiently favorable characteristics.

Low-quality candidates should remain:

```text
NO TRADE
```

rather than being forced into LONG or SHORT.

---

# 16. Initial Feature Set

The first implementation should remain compact.

### Momentum

```text
body_ratio
candle_body
candle_range
upper_wick_ratio
lower_wick_ratio
close_position
```

### Volatility

```text
atr_14
range_to_atr
```

### EMA

```text
close_vs_ema5
close_vs_ema20
close_vs_ema50

ema5_vs_ema20
ema20_vs_ema50
```

### Structure

```text
distance_to_previous_high_20
distance_to_previous_low_20
distance_to_previous_high_50
distance_to_previous_low_50

break_previous_high_20
break_previous_low_20
break_previous_high_50
break_previous_low_50
```

### Recent movement

```text
return_3
return_6
return_12
```

### Volume

```text
volume_ratio_20
volume_change_1
```

This gives us a relatively small research feature set.

Not all of these features are guaranteed to survive into the final model.

---

# 17. Feature Selection Principle

A feature is retained because it provides useful information demonstrated through proper time-series validation.

A feature should not be retained merely because:

* it is a popular indicator;
* it improves in-sample performance;
* it sounds technically sophisticated;
* it increases model complexity.

The project prioritizes:

```text
simple hypothesis
        ↓
clean measurement
        ↓
empirical analysis
        ↓
out-of-sample validation
        ↓
feature retention
```

---

# 18. Future Expansion

Only after the initial feature set has been evaluated should additional concepts be considered.

Potential future research areas:

* consolidation/compression before momentum;
* consecutive directional candles;
* momentum candle after pullback;
* distance from moving averages;
* swing structure;
* higher-timeframe structure;
* session/time-of-day context;
* spread conditions;
* volatility regimes;
* retest behavior after breakout.

These are research candidates, not current strategy rules.

---

# 19. Design Principle

The project is intentionally specialized.

It does not attempt to predict every market movement.

The research target is:

> **Identify high-quality momentum-candle events and determine the market contexts in which their subsequent movement has favorable probability characteristics.**

The 80% body-to-range condition is the initial event definition.

EMA 5/20/50, support/resistance, breakout state, volatility, recent movement, and volume are contextual variables whose usefulness must be demonstrated by data.

No feature should be considered predictive merely because it appears logically plausible.
