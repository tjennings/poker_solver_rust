# Oracle Boundary Orientation Audit (2026-05-04)

Bean: `poker_solver_rust-hthh`

## Purpose

Test whether the `exact_oracle` divergence is explained by a simple raw-CFV orientation mismatch at depth boundaries: OOP/IP swapped, sign-flipped, or both.

## Setup

Command shape:

```bash
./target/release/poker-solver-trainer compare-solve \
  --bundle ./local_data/blueprints/1k_100bb_brdcfr_v2 \
  --snapshot snapshot_0013 \
  --spot 'sb:2bb,bb:10bb,sb:22bb,bb:call|JhTh9h|bb:15bb,sb:call|7d' \
  --river-boundary exact_oracle \
  --oracle-orientation <mode> \
  --iters 200 \
  --tolerance 0.0
```

Spot:

- Board: `JhTh9h7d`
- Pot: `73`
- Effective stack: `63`
- Position: `BB`
- Boundary: `depth=0, exact_oracle`
- Boundaries: `11`

## Results

| Orientation | Exact exp | Subgame exp | Delta | Mean mass | Max mass | Bias summary |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `current` | 23.06 | 2860.16 | +2837.10 | 0.184 | 0.982 | Check +0.115, Bet/Raise -0.059, AllIn -0.055 |
| `swap` | 23.06 | 18548.70 | +18525.64 | 0.676 | 1.000 | Check -0.393, Bet/Raise +0.238, AllIn +0.155 |
| `sign-flip` | 23.06 | 635.06 | +611.99 | 0.677 | 1.000 | Check -0.127, Bet/Raise -0.201, AllIn +0.328 |
| `swap-sign-flip` | 23.06 | 34631.04 | +34607.98 | 0.822 | 1.000 | Check -0.525, Bet/Raise -0.087, AllIn +0.612 |

All exploitability values are mbb/hand.

## Interpretation

No orientation transform recovers the exact root strategy. `sign-flip` improves subgame exploitability relative to `current`, but it makes root strategy agreement much worse (`mean_mass=0.677`, `max_mass=1.000`). The simple orientation hypotheses therefore do not explain the oracle-boundary failure.

Combined with the one-boundary contract test from step 2, the likely fault is narrower: a multi-boundary, reach, or boundary-injection effect rather than a universal OOP/IP or sign convention bug.

## Follow-Up

Proceed to unit/normalization audit (`poker_solver_rust-24bk`) and reach semantics audit (`poker_solver_rust-qcgs`). The `sign-flip` exploitability improvement is worth keeping in mind, but it is not a viable fix because it destroys root strategy agreement.
