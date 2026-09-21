# Experiment Log

Every run that informs a design decision is logged here, including null results.
Development partition only unless stated. Before running a new variant, add a
row with its hypothesis first, and count it: more variants tried on the same
data means more chance of a coincidental pattern.

Experiments logged so far: 1

| # | Date | Change / hypothesis | Data | Result | Decision |
|---|---|---|---|---|---|
| 1 | 2026-09-21 | Baseline: BOS setup, nearest-swing target, horizon 10, spread 0.30 | development, 1,279 setups | mean R -0.0545, 95% CI [-0.0863, -0.0236] (day-clustered); placebo mean -0.0535, 52% of placebo draws >= real | Baseline raw setup shows no positive realized-R edge; carry forward as a null baseline for structural-context analysis |

## Notes

- Placebo parity means the BOS timing shows no measurable difference from random
  entry timing at this geometry after spread. It is descriptive, not causal.
- Spread sensitivity (0.0 / 0.16 / 0.50) is still to be run to separate cost
  drag from edge.
