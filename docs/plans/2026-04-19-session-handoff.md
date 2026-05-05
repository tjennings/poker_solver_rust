# Session Handoff — 2026-04-19

## What just shipped (unpushed, on main)

Batched turn datagen: 3 commits that batch many turn games per GPU kernel launch using one canonical topology. Eliminates per-game kernel compilation. Plan: `docs/plans/2026-04-11-batched-turn-datagen.md`. Bean: `poker_solver_rust-alpn` (completed).

| Commit | What |
|-|-|
| `e0f811a` | Per-batch fold payoffs — `SubgameSpec` carries `fold_payoffs_p0/p1`, kernel indexes `[bid * num_folds + fi]` |
| `e99fc38` | Per-batch leaf CFVs — kernel indexes `[bid * num_leaves * H + li * H + h]`, `update_leaf_cfvs` takes batched buffers |
| `22971ba` | Canonical topology + batched orchestrator — `build_canonical_turn_tree` (SPR=100, 1326 universal hands), `run_gpu_turn` rewritten with batch loop, diagnostic timing removed, aggregate throughput logging added |

Tests: 94 gpu-range-solver + 221 cfvnet all pass. Only tested at `batch_size=4`.

## Blocker: GPU memory at batch_size=256

Showdown outcomes buffer allocates `[B × num_showdowns × 1326 × 1326 × 4B]` per player. For turn datagen these outcomes are **all zeros** (leaf injection provides the real values), so the allocation is pure waste. At `batch_size=256` this is likely 10s-100s of GB depending on tree size — will OOM on any real GPU.

Options to fix:
- **(A)** Skip showdown outcome allocation when all zeros (add a flag or detect)
- **(B)** Default `gpu_batch_size` to a safe value like 32 (still ~8x speedup)
- **(C)** Both — skip allocation AND allow large batches

The config field `gpu_batch_size: Option<usize>` already exists in `crates/cfvnet/src/config.rs:197`, defaults to 256 via `unwrap_or(256)` in the orchestrator.

## Dirty working tree (pre-existing, intentional)

These were dirty before the batched datagen work and were deliberately left alone:

- `Cargo.lock` — adds `filetime` transitive dep (70 lines)
- `sample_configurations/turn_gpu_datagen.yaml` — `per_file: 100000` added, `bet_size_fuzz: 0.15 → 0.20`
- `docs/plans/2026-04-08-gpu-turn-datagen-brief.md` — untracked plan doc

Decide: commit these as housekeeping, or revert if unwanted.

## 27 unpushed commits on main

Everything from `feat/gpu-turn-datagen` merge through batched datagen. Not pushed to origin. Review before pushing.

## Pre-existing test failures (unrelated)

- `mp_tui_scenarios::tests::resolve_empty_returns_root` — flaky 10s timeout under parallel load, passes in isolation
- `tests::mp_6player_tui_section_parses` — config has `tui.enabled: true` but test asserts `!enabled`

CLAUDE.md requires a green suite. These should be fixed before future feature work.

## Stale in-progress beans to triage

These have been in-progress for weeks with no recent commits. Decide: still active, or close/scrap?

| Bean | Title | Priority | Age |
|-|-|-|-|
| `nrz3` | Fix regret pruning: skip updates for pruned actions | critical | Mar 2 |
| `4j0p` | GPU solver batched ops optimization | high | Apr 3 |
| `nv3f` | Boundary CFV magnitude calibration | normal | Mar 26 |
| `c6e6` | Fix cfvnet compare: skip zero effective stack spots | normal | Mar 12 |
| `i261` | fix(rebel): pot-relative CFVs | normal | — |
| `60h7` | Blueprint V2: Pluribus-style full-game solver (epic) | normal | Mar 6 |
| `elst` | Convergence Validation Harness (epic) | normal | Mar 24 |

## Ready backlog (high priority)

| Bean | Title |
|-|-|
| `b5q3` | Implement per-flop blueprint training pipeline |
| `n854` | Compute average starting ranges from blueprint strategy |
| `zry2` | Lazy storage allocation for blueprint_mp |

## Suggested next actions (pick one)

1. **Fix the OOM blocker** — option (C) above, then run a real `batch_size=256` test
2. **Push to remote** — after verifying batch_size works or capping default to 32
3. **Start production turn datagen** — using the safe batch size
4. **Triage stale beans** — especially `nrz3` (critical bug from March)
5. **Fix pre-existing test failures** — green suite before more feature work
