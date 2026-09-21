# Historical Boundary and True OOS Policy

## Status

The March 18, 2026 boundary is a **historical audit boundary**, not a pristine holdout.

Before this boundary was formalized, the full available dataset was inspected with label-conditional structural audits. Therefore the period from the boundary through the last currently available candle must not be presented as untouched out-of-sample evidence.

## Historical partition

- Boundary: `2026-03-18 00:00:00`
- Development: setup plus the full forward label horizon ends before the boundary.
- Historical audit: setup starts at or after the boundary.
- Purged boundary: setup before the boundary whose forward label horizon crosses it.

The historical-audit partition is still useful for reproducible diagnostics and robustness checks, but it is not valid as the final model-selection holdout.

## True OOS

The true OOS clock starts **after methodology, structural features, label semantics, model configuration, and selection rules are frozen**.

For the current dataset, the next newly collected candles after the existing data end are the clean evaluation stream. Their outcomes must not be used to tune the model before the OOS evaluation is reported.

Do not retroactively call March–September 2026 pristine OOS.

## Pre-setup feature contract

Setup-time structural features may be read from the candle immediately before the setup candle (`pre_*`). This preserves the structural context that existed before a BOS setup candle can trigger an external-boundary rebuild.

Pre-setup features must remain structural facts only; they must not contain label, target, execution, or future-outcome information.

## Honest research log

- Full-data structural label audits: completed before this policy was frozen.
- Historical boundary: retained for reproducible partitioning, not claimed as pristine OOS.
- True OOS: begins with newly collected data after the methodology freeze.
