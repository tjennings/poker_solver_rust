# Oracle Boundary Root Update Trace - 2026-05-04

## Question

The aligned `exact_oracle` run still showed large root policy mass movement
even when exploitability was close. This diagnostic checks whether exact and
subgame diverge before the root regret update or during regret accumulation.

## Diagnostic

`compare-solve` now has a hidden flag for iteration-aligned oracle runs:

```text
--root-update-trace-iters <csv>
```

For each listed zero-based iteration it prints:

- max per-hand root action CFV gap between exact and subgame before the update;
- max per-hand root regret-update gap after discounting previous regrets;
- reach-weighted mean root action CFV by action.

The probe snapshots and restores subgame boundary reach and clears probe-filled
boundary CFVs so the diagnostic does not perturb the solve.

## Canonical Results

Spot:

```text
sb:2bb,bb:10bb,sb:22bb,bb:call|JhTh9h|bb:15bb,sb:call|7d
```

Command shape:

```text
compare-solve --river-boundary exact_oracle \
  --oracle-iteration-aligned --root-update-trace-iters 0,999
```

At iteration 0:

- `exact_pre_vs_sub_pre`: 0.000000 chips
- `max_abs_regret_update_gap`: 0.000000 chips
- `exact_post_vs_sub_pre`: 3.510944 chips at `55bb 7h8h`

At iteration 999:

- `exact_pre_vs_sub_pre`: 0.171595 chips / 85.80 mbb at `24bb QhAs`
- `max_abs_regret_update_gap`: 0.170557 chips / 85.28 mbb at `24bb QhAs`
- reach-weighted action mean gaps were small: `24bb` was -0.001878 chips, all
  other actions were within 0.000389 chips.

Corrected final 1000/1000 result:

- exact exploitability: 2.29 mbb/hand
- subgame exploitability: 3.92 mbb/hand
- delta: +1.63 mbb/hand
- mean root mass moved: 0.041
- max root mass moved: 1.000 at `3sTs`

## Interpretation

The initial root update is identical between exact and subgame. By iteration
999, the largest root regret-update gap is about 0.17 chips on a specific hand
and action, while weighted action means are nearly identical. The large root
policy mass movement is therefore concentrated in individual hands rather than
coming from a broad root value shift.

The diagnostic also found the bigger measurement issue: evaluator-backed
boundary CFVs are reach-dependent and must be cleared before finalization.
Without that, finalization can reuse last-iteration cached boundary values and
overstate subgame exploitability. Clearing before finalization changes the
aligned 1000/1000 subgame exploitability from 94.81 mbb/hand to 3.92 mbb/hand
without changing the root strategy diff.

## Next Target

The remaining question is why a few hands flip between near-pure `Check` and
near-pure `24bb` despite small weighted action-value gaps. The next diagnostic
should inspect root regrets and strategy sums for the top moved hands
(`3sTs`, `2sTs`, `4sTs`, `QhAs`) across iterations to distinguish true value
divergence from near-indifference/regret-threshold effects.
