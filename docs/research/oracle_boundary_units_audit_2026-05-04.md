# Oracle Boundary Units Audit (2026-05-04)

Bean: `poker_solver_rust-24bk`

## Purpose

Test whether the `exact_oracle` divergence is explained by a simple scalar unit or normalization mismatch at the depth-boundary CFV handoff.

## Unit Map

- `range_solver::utility::cfvalues_after_history_with_reach` returns internal per-combination counterfactual values. This is the same representation written by terminal evaluation and consumed by regret updates.
- `exact_oracle` uses `BoundaryEvaluator::compute_raw_cfvs_both`, then `evaluate_boundary_raw`. This path writes the returned per-combination values directly into the recursive CFV result and bypasses the legacy `bcfv * half_pot / N * cfreach_adj` formula.
- Legacy boundary evaluators use `compute_cfvs_both` in pot-normalized `bcfv` units where `1.0` means one half-pot. Those values are converted back to internal per-combination CFVs by `evaluate_boundary_single`.
- `SubtreeExactEvaluator` intentionally returns `bcfv` for the legacy path. It solves a downstream subtree, computes raw per-combination CFVs with the actual boundary reach, then divides out `half_pot / N * cfreach_adj`.
- `cfvnet` and gadget opt-outs use the same pot-normalized `bcfv` convention, including the `(ev_chips - half_pot) / half_pot` target convention.

Conclusion from the code map: `exact_oracle` should not be converted to `bcfv`; it is already on the raw per-combination path.

## Scale Sweep

Command shape:

```bash
./target/release/poker-solver-trainer compare-solve \
  --bundle ./local_data/blueprints/1k_100bb_brdcfr_v2 \
  --snapshot snapshot_0013 \
  --spot 'sb:2bb,bb:10bb,sb:22bb,bb:call|JhTh9h|bb:15bb,sb:call|7d' \
  --river-boundary exact_oracle \
  --oracle-scale <scale> \
  --iters 200 \
  --tolerance 1000000
```

Spot:

- Board: `JhTh9h7d`
- Pot: `73`
- Effective stack: `63`
- Position: `BB`
- Boundaries: `11`
- Exact exploitability: `23.06` mbb/hand

| Scale | Subgame exp | Delta | Mean mass | Max mass | Bias summary |
| ---: | ---: | ---: | ---: | ---: | --- |
| `0.0100` | 3.81 | -19.25 | 0.507 | 0.999 | AllIn +0.117, Bet/Raise -0.070, Check -0.046 |
| `0.0274` | 4.68 | -18.38 | 0.766 | 1.000 | Check -0.451, AllIn +0.157, Bet/Raise +0.294 |
| `0.1000` | 1.30 | -21.76 | 0.719 | 1.000 | Bet/Raise +0.239, AllIn +0.234, Check -0.473 |
| `0.2000` | 1.17 | -21.90 | 0.752 | 1.000 | AllIn +0.281, Check -0.521, Bet/Raise +0.240 |
| `0.5000` | 244.18 | +221.12 | 0.564 | 1.000 | Check -0.425, Bet/Raise +0.161, AllIn +0.264 |
| `1.0000` | 2860.16 | +2837.10 | 0.184 | 0.982 | Bet/Raise -0.059, Check +0.115, AllIn -0.055 |
| `2.0000` | 175.53 | +152.47 | 0.628 | 1.000 | Bet/Raise -0.007, Check +0.188, AllIn -0.181 |
| `10.0000` | 219.49 | +196.43 | 0.440 | 1.000 | Check +0.422, Bet/Raise -0.241, AllIn -0.181 |
| `36.3000` | 81.45 | +58.39 | 0.436 | 1.000 | AllIn -0.181, Check +0.427, Bet/Raise -0.246 |

All exploitability values are mbb/hand.

## Interpretation

No scalar conversion recovers the exact root strategy. Some downscales make the subgame's own exploitability look very low, but they do so by producing a strategy that is almost maximally different from the full-depth exact strategy (`mean_mass` around `0.7`, `max_mass=1.0`). That is not a valid boundary-unit fix; it is a different policy, not the exact policy.

The default scale `1.0` has the best root agreement among the tested scales, even though its subgame exploitability is poor. That matches the code map: the raw oracle values are already in the solver's internal per-combination units.

## Follow-Up

Units/normalization is unlikely to be the root cause. The next audit should focus on reach semantics (`poker_solver_rust-qcgs`), especially when and how the first boundary visit caches raw CFVs for both players before both players' boundary reach vectors are available for the current iteration.
