# Realized-R Audit

## Status

Step 3 methodology baseline. This layer measures the economic outcome of the
already-defined deterministic structural setup. It does not add new features,
change BOS semantics, or train an ML model.

## Realized R

For each evaluable setup:

1. The setup is generated from confirmed BOS.
2. Entry is the next candle open.
3. Execution applies the configured spread.
4. The nearest eligible confirmed valid swing remains the target.
5. The existing deterministic stop/target semantics are reused.
6. An unresolved trade is closed at the close of the last candle in the
   forward horizon and classified as TIME_STOP.

realized_r is therefore a realized outcome metric after entry spread.

reward_r_after_spread describes target distance relative to execution risk.
It is not a complete net-P&L metric because slippage, fees, and gap-fill
semantics are not modelled.

## Placebo timing

The placebo keeps each selected setup's direction, stop distance, and target
distance, then samples random entry candles from the permitted partition.

This is a descriptive timing sanity check. It is not a causal test and must
not be treated as proof that the structural setup caused the observed result.

## Uncertainty

The audit reports a 95% bootstrap interval for mean realized R by resampling
whole calendar-day clusters.

Setups can overlap in time, so this clustered interval remains optimistic.
It is an uncertainty diagnostic, not a guarantee of independent observations.

## Thresholds

Minimum reward/risk filters are reported only as exploratory diagnostics.
Thresholds must not be selected from this audit and then presented as
validated strategy parameters.

## Historical boundary and OOS

The default audit uses the development partition. The optional historical
partition is explicitly not pristine OOS because the dataset was inspected
before the boundary policy was frozen.

A true OOS evaluation begins only on newly collected data after methodology,
feature definitions, label semantics, model configuration, and selection rules
are frozen.

## Scope

Step 3 answers one question:

Does the existing deterministic structural setup have a measurable realized R
distribution worth carrying forward into later research?

It does not answer model selection, future profitability, or live-trading
performance.
