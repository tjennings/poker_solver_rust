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

| mode | exact iters | subgame iters | exact exp | subgame exp | delta | mean mass | max mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| finalized average | 200 | 200 | 23.06 | 42.85 | +19.79 | 0.080 | 0.893 |
| iteration aligned | 200 | 200 | 23.06 | 88.83 | +65.76 | 0.091 | 0.915 |
| finalized average | 1000 | 1000 | 2.29 | 223.65 | +221.36 | 0.122 | 0.995 |
| iteration aligned | 1000 | 1000 | 2.29 | 94.81 | +92.52 | 0.041 | 1.000 |

Units are mbb/hand for exploitability.

## Interpretation

Iteration alignment does not recover exact. It worsens the 200/200 control and
improves, but does not fix, the 1000/1000 run. That means final-average
decoupling is a contributor at high iteration counts, but not the whole story.

The remaining gap is more likely a multi-boundary consistency issue:

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
