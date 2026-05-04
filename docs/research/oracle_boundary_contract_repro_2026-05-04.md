# Oracle Boundary Contract Repro - 2026-05-04

## Context

Branch: `codex/oracle-boundary-compare-solve`

Commit: `b9633a81`

Command shape:

```bash
./target/release/poker-solver-trainer compare-solve \
  --bundle ./local_data/blueprints/1k_100bb_brdcfr_v2 \
  --snapshot snapshot_0013 \
  --spot 'sb:2bb,bb:10bb,sb:22bb,bb:call|JhTh9h|bb:15bb,sb:call|7d' \
  --river-boundary <mode> \
  --iters 200 \
  --tolerance 1.0
```

Spot summary:

- Board: `JhTh9h7d`
- Pot: `73`
- Effective stack: `63`
- Position: `BB`
- Blueprint snapshot: `snapshot_0013`

## Summary

| Boundary mode | Boundaries | Exact exp | Hybrid exp | Delta | Mean mass moved | Max mass moved | Worst cell delta | Tolerance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `exact` | 0 | 23.06 | 23.06 | +0.00 | 0.000 | 0.000 | 0.0000 | PASS |
| `exact_oracle` | 11 | 23.06 | 2860.16 | +2837.10 | 0.184 | 0.982 | 0.6840 | PASS |
| `exact_subtree` | 11 | 23.06 | 1063.89 | +1040.83 | 0.438 | 1.000 | 0.9676 | PASS |

All exploitability numbers are `mbb/hand`.

Interpretation:

- The all-exact control is clean: exact and hybrid strategies match exactly at root.
- `exact_oracle` still diverges severely, despite feeding boundaries from the solved exact game.
- `exact_subtree` has lower exploitability than `exact_oracle` on this run, but higher average root mass movement.
- The tolerance check is not sufficient as a correctness criterion here: it passes while exploitability is catastrophically worse.

## Per-Mode Notes

### `exact`

- `final_exp`: `23.06`
- `hybrid_exp`: `23.06`
- `mean mass moved`: `0.000`
- `max mass moved`: `0.000`
- Worst cell: `22 @ Check`, `|delta|=0.0000`

### `exact_oracle`

- Boundaries: `11`
- `final_exp`: `23.06`
- `hybrid_exp`: `2860.16`
- `exploitability delta`: `+2837.10`
- `mean mass moved`: `0.184`
- `max mass moved`: `0.982` at `8hAd`
- Bias:
  - `Check`: `+0.115`
  - `Bet/Raise`: `-0.059`
  - `AllIn`: `-0.055`
- Worst cell: `Q9s @ 55bb`, exact `0.8184`, subgame `0.1344`, `|delta|=0.6840`

Largest root strategy movement examples:

| Hand | Exact | Oracle-boundary subgame |
| --- | --- | --- |
| `8hAd` | `[X:0.01 B24:0.98 B55:0.01 A:0.00]` | `[X:0.99 B24:0.01 B55:0.00 A:0.00]` |
| `8h9d` | `[X:0.00 B24:0.85 B55:0.10 A:0.05]` | `[X:0.96 B24:0.02 B55:0.01 A:0.01]` |
| `8dAs` | `[X:0.00 B24:0.00 B55:0.00 A:1.00]` | `[X:0.88 B24:0.00 B55:0.06 A:0.06]` |

### `exact_subtree`

- Boundaries: `11`
- `final_exp`: `23.06`
- `hybrid_exp`: `1063.89`
- `exploitability delta`: `+1040.83`
- `mean mass moved`: `0.438`
- `max mass moved`: `1.000` at `QhAs`
- Bias:
  - `Check`: `+0.191`
  - `Bet/Raise`: `-0.148`
  - `AllIn`: `-0.043`
- Worst cell: `99 @ Check`, exact `0.9988`, subgame `0.0312`, `|delta|=0.9676`

Largest root strategy movement examples:

| Hand | Exact | Exact-subtree subgame |
| --- | --- | --- |
| `QhAs` | `[X:0.00 B24:0.00 B55:0.84 A:0.16]` | `[X:1.00 B24:0.00 B55:0.00 A:0.00]` |
| `7hTd` | `[X:0.00 B24:0.00 B55:0.44 A:0.56]` | `[X:1.00 B24:0.00 B55:0.00 A:0.00]` |
| `7hKs` | `[X:0.00 B24:0.00 B55:0.60 A:0.40]` | `[X:1.00 B24:0.00 B55:0.00 A:0.00]` |

## Boundary Ordinals From `exact_oracle` Trace

Trace command added:

```bash
./target/release/poker-solver-trainer compare-solve \
  --bundle ./local_data/blueprints/1k_100bb_brdcfr_v2 \
  --snapshot snapshot_0013 \
  --spot 'sb:2bb,bb:10bb,sb:22bb,bb:call|JhTh9h|bb:15bb,sb:call|7d' \
  --river-boundary exact_oracle \
  --iters 200 \
  --tolerance 1.0 \
  --trace-boundaries all \
  --trace-iters last \
  --trace-dir /tmp/oracle_boundary_step1_traces
```

The trace files were written to `/tmp/oracle_boundary_step1_traces`. They are intentionally not committed because they include large per-hand range and CFV dumps.

| Ordinal | Pot | Stack | SPR | Boundary spot suffix |
| ---: | ---: | ---: | ---: | --- |
| 0 | 146 | 53 | 0.36 | `bb:check,sb:check` |
| 1 | 242 | 5 | 0.02 | `bb:check,sb:24bb,bb:call` |
| 2 | 398 | 0 | 0.00 | `bb:check,sb:24bb,bb:all-in,sb:call` |
| 3 | 366 | 0 | 0.00 | `bb:check,sb:55bb,bb:call` |
| 4 | 398 | 0 | 0.00 | `bb:check,sb:55bb,bb:all-in,sb:call` |
| 5 | 398 | 0 | 0.00 | `bb:check,sb:all-in,bb:call` |
| 6 | 242 | 5 | 0.02 | `bb:24bb,sb:call` |
| 7 | 398 | 0 | 0.00 | `bb:24bb,sb:all-in,bb:call` |
| 8 | 366 | 0 | 0.00 | `bb:55bb,sb:call` |
| 9 | 398 | 0 | 0.00 | `bb:55bb,sb:all-in,bb:call` |
| 10 | 398 | 0 | 0.00 | `bb:all-in,sb:call` |

Trace detail from ordinal 0 confirms the trace layer reports:

- board/pot/stack/SPR
- full assembled spot string
- OOP/IP ranges
- OOP/IP CFVs in chips
- strategy at the preceding decision

## Immediate Next Questions

The locked repro points away from exact-subtree re-solving noise as the primary cause. The next discriminator should be a boundary-contract test that can answer these independently:

- Does `compute_raw_cfvs_both` return values in the player orientation expected by range-solver boundary evaluation?
- Are oracle CFVs raw chip CFVs, while another layer expects normalized BCFVs?
- Are boundary reaches conditionalized or chance-weighted differently between the exact game and depth-limited game?
- Does depth-boundary injection consume the returned OOP/IP arrays with the correct sign for the traverser?
