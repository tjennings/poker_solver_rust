# Oracle Boundary Injection Audit - 2026-05-04

## Question

After the raw boundary reach-cache fix, `exact_oracle` improved substantially at
200 iterations but still diverged from the full exact solve on the canonical
turn spot:

```text
sb:2bb,bb:10bb,sb:22bb,bb:call|JhTh9h|bb:15bb,sb:call|7d
```

This audit asks whether the remaining gap is caused by how raw boundary CFVs are
injected into regret updates, or by a higher-level mismatch between the exact
continuation policy and the depth-limited trunk policy.

## Code Path Review

The solver path is shared after evaluation:

- Full-depth terminals call `game.evaluate(result, node, player, cfreach)`.
- Depth-boundary terminals also call `game.evaluate(result, node, player,
  cfreach)`.
- From there, both paths feed the same `cfv_buf -> result -> regret_delta`
  update in `range-solver/src/solver.rs`.
- The only special raw-boundary behavior is inside `evaluate_internal`, where
  raw evaluators write per-combination CFVs directly instead of applying the
  normalized legacy `bcfv * payoff_scale * cfreach_adj` path.

The existing one-boundary trainer contract test exercises this handoff with an
oracle continuation and passes:

```text
cargo test -p poker-solver-trainer oracle_boundary_one_boundary_contract_matches_exact
```

That makes a local sign/scale/player injection bug less likely.

## Diagnostic Added

`compare-solve` now has hidden iteration overrides:

- `--exact-iters <N>` controls the full exact solve used as the ground truth and
  as the oracle continuation source.
- `--subgame-iters <N>` controls the depth-limited subgame solve.
- Both default to `--iters`, preserving existing CLI behavior.

This lets us separate exact-continuation quality from trunk convergence.

## Sweep Results

All runs used:

```text
./target/release/poker-solver-trainer compare-solve \
  --bundle ./local_data/blueprints/1k_100bb_brdcfr_v2 \
  --snapshot snapshot_0013 \
  --spot 'sb:2bb,bb:10bb,sb:22bb,bb:call|JhTh9h|bb:15bb,sb:call|7d' \
  --river-boundary exact_oracle \
  --tolerance 1.0
```

| exact iters | subgame iters | exact exp | subgame exp | delta | mean mass | max mass |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 200 | 200 | 23.06 | 42.85 | +19.79 | 0.080 | 0.893 |
| 1000 | 200 | 2.29 | 137.70 | +135.41 | 0.157 | 1.000 |
| 200 | 1000 | 23.06 | 145.06 | +122.00 | 0.173 | 0.973 |
| 1000 | 1000 | 2.29 | 223.65 | +221.36 | 0.122 | 0.995 |

Units are mbb/hand for exploitability.

## Interpretation

The 200/200 run is the best of the sweep. Increasing either side worsens the
root agreement on this multi-boundary turn spot. That pattern does not look like
a simple raw CFV orientation, unit, or per-player injection error:

- If exact-continuation quality were the only problem, `exact=1000,
  subgame=200` should improve, but it worsens.
- If trunk convergence were the only problem, `exact=200, subgame=1000` should
  improve, but it also worsens.
- The one-boundary contract test shows the raw terminal handoff can reproduce an
  equivalent exact solve in a minimal setting.

The more likely explanation is a coupled-policy mismatch: `exact_oracle` freezes
the finalized average exact continuation and lets the depth-limited trunk solve
against that static continuation. Full exact CFR, however, learns trunk and
continuation regrets together across iterations. On this spot, the final
averaged continuation appears to induce a different depth-limited trunk optimum
than the full exact solver's averaged root policy, and the disagreement becomes
more visible as either side is pushed longer.

## Next Diagnostic

The next useful test is to compare *iteration-aligned* oracle values rather than
only the finalized average continuation:

1. During exact solve, snapshot the continuation strategy or boundary CFVs at the
   same iterations used by the subgame solve.
2. Feed the subgame boundary with the matching per-iteration exact continuation.
3. Compare root regrets/strategies against the full exact solve at the same
   iteration.

If iteration-aligned values collapse the gap, the injection path is fine and
the problem is final-average decoupling. If the gap remains, the next target is
multi-boundary reach semantics or missing state in `cfvalues_after_history_with_reach`.
