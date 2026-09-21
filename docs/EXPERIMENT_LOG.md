# Experiment Log

Every run that informs a design decision is logged here, including null results.
Development partition only unless stated. Before running a new variant, add a
row with its hypothesis first, and count it: more variants tried on the same
data means more chance of a coincidental pattern.

Experiments logged so far: 3

| # | Date | Change / hypothesis | Data | Result | Decision |
|---|---|---|---|---|---|
| 1 | 2026-09-21 | Baseline: BOS setup, nearest-swing target, horizon 10, spread 0.30 | development, 1,279 setups | mean R -0.0545, 95% CI [-0.0863, -0.0236] (day-clustered); placebo mean -0.0535, 52% of placebo draws >= real | Baseline raw setup shows no positive realized-R edge; carry forward as a null baseline for structural-context analysis |
| 2 | 2026-09-21 | Descriptive structural-context heterogeneity audit using locked setup facts and pre-setup structural features | development, 1,279 setups | Categorical split: pre-structure direction -1 mean -0.1136 vs +1 mean +0.0016. Numeric quartiles show variation in broken-swing age, invalidation-swing age, distances, and bars-since-swing; no bucket is treated as a selected threshold. Missingness remains structural for pre-setup distance features. | Heterogeneity is visible descriptively, but this is not evidence of a stable edge. Next run must test stability across time and spread before any ML feature selection. |
| 3 | 2026-09-21 | Predefined structural-context stability audit across chronological development slices and execution spread | development, 1,279 setups; fixed periods P1/P2/P3 and spreads 0.00/0.16/0.30/0.50 | Pending run; script added with fixed time/cost grid and the locked context contract. | Run once and assess persistence of context contrast direction across time and spread; do not tune thresholds or select a best bucket. |

## Notes

- Placebo parity means the BOS timing shows no measurable difference from random
  entry timing at this geometry after spread. It is descriptive, not causal.
- Spread sensitivity (0.0 / 0.16 / 0.50) is still to be run to separate cost
  drag from edge.
- The context audit is descriptive: differences across direction, structural age,
  distance, and recency are not threshold recommendations and are not yet stable
  enough to justify ML feature selection.
