# Setup Facts v1

Setup facts are deterministic, setup-time context columns added to the
structural label dataset. They describe the BOS that created the setup, using
only the BOS event on the setup candle and swings confirmed strictly before it.

| Column | Meaning |
|---|---|
| `setup_bos_external` | 1.0 if the broken swing was EXTERNAL (external break, rebuild), else 0.0 |
| `setup_broken_swing_age` | Candles from the broken swing's confirmation to the setup candle |
| `setup_entry_beyond_broken_swing_r` | Signed distance from entry beyond the broken swing, in R |
| `setup_invalidation_swing_age` | Candles from the invalidation swing's confirmation to the setup |

Rules:

- Facts never use candles after the setup candle.
- If the BOS event or a required swing cannot be found, every fact is NaN.
- These facts complement, and do not replace, the active/historical structure
  features, which describe the context after the setup candle's own rebuild.

## Feature entry points

- `build_structural_features(frame)`: structure-only features (blueprint default).
- `build_features(frame)`: legacy momentum/EMA/rolling set plus structure. Kept as
  an ablation baseline only; it is not part of the structural blueprint.
