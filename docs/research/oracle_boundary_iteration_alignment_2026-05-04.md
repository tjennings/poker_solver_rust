# Oracle Boundary Iteration Alignment - 2026-05-04

## Question

The prior injection audit showed that `exact_oracle` gets worse when exact and
subgame iteration counts are varied independently. One plausible explanation
was final-average decoupling: the subgame solves against the finalized exact
average continuation, while full exact CFR learns trunk and continuation
together.

This diagnostic tests that theory by running exact and subgame in lockstep.
At subgame iteration `t`, each oracle boundary evaluates against the exact game
after exact iteration `t`.

## Diagnostic

`compare-solve` now has a hidden flag:

```text
--oracle-iteration-aligned
```

It requires an `exact_oracle` street boundary and matching `--exact-iters` /
`--subgame-iters`. The implementation does not store full game snapshots.
Instead, it runs exact and subgame interleaved and lets the oracle boundary
evaluator read the live exact game through a shared lock.

## Canonical Spot

```text
sb:2bb,bb:10bb,sb:22bb,bb:call|JhTh9h|bb:15bb,sb:call|7d
```

All runs used `--river-boundary exact_oracle --tolerance 1.0`.

## Results

Note: the original version of this note recorded stale-cache exploitability for
iteration-aligned subgame finalization. The root-update diagnostic found that
the evaluator-backed boundary CFV cache must be cleared before finalization so
final average-strategy EVs are recomputed with the final reaches. The table
below reflects the corrected behavior.

| mode | exact iters | subgame iters | exact exp | subgame exp | delta | mean mass | max mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| finalized average | 200 | 200 | 23.06 | 42.85 | +19.79 | 0.080 | 0.893 |
| iteration aligned | 200 | 200 | 23.06 | 18.95 | -4.12 | 0.091 | 0.915 |
| finalized average | 1000 | 1000 | 2.29 | 223.65 | +221.36 | 0.122 | 0.995 |
| iteration aligned | 1000 | 1000 | 2.29 | 3.92 | +1.63 | 0.041 | 1.000 |

Units are mbb/hand for exploitability.

## Interpretation

Iteration alignment largely recovers exploitability once finalization clears
stale evaluator-backed boundary CFVs. The root policy can still be very
different on individual hands, but the corrected 1000/1000 run is close in
exploitability: +1.63 mbb/hand over the exact solve.

The remaining root-policy gap is more likely a multi-boundary consistency or
near-indifference issue:

- The one-boundary oracle contract still passes.
- The depth-limited trunk has multiple river-boundary terminals competing for
  regret updates at the turn root.
- Each boundary is evaluated independently from the exact continuation under
  the subgame's live boundary reach, but the full exact solve learns root,
  turn, river, and chance continuation regrets as one coupled object.

## Next Target

The next diagnostic should compare exact and subgame regret deltas at the root
for one iteration and one high-divergence hand/action:

1. Run a single exact iteration and capture root action CFVs before the regret
   update.
2. Run the aligned subgame iteration with the same exact continuation and
   capture root action CFVs before the regret update.
3. Compare the per-action CFV vector for the same private hand.

If root action CFVs already differ before regret updating, the issue is in
boundary value composition across multiple terminals. If action CFVs match but
regrets diverge, the issue is in regret/strategy accumulation semantics.
