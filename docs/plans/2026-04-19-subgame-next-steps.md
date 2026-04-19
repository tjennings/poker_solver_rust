# Subgame Solver — Next Steps Handoff

**Date:** 2026-04-19
**Context:** End of multi-session debugging of the `izod` bug (Subgame produces catastrophically wrong strategies vs Exact).

## Current state of main

Branch: `main`, HEAD `de40cdd4`, pushed to origin.

### What landed this session (23 commits)

**Rollout perf (already shipped, depth=0 path):**
- Depth-gated MCCFR sampling (`SAMPLE_AFTER_DECISION_DEPTH`, default 2): ~164× speedup on rollout call latency in isolation
- Per-call RNG seed rotation (`call_counter: Arc<AtomicU64>`) fixing deterministic-stream bug
- Configurable `rollout_enumerate_depth` and `rollout_opponent_samples` in Tauri Settings
- Removed obsolete `flop_combo_threshold` / `turn_combo_threshold` auto-dispatch (user selects Subgame vs Exact explicitly)

**Exact-mode fixes:**
- Matrix snapshot fix (`leafEvalInterval=0` special case was breaking snapshots; renamed to `matrix_snapshot_interval` for clarity)
- Periodic exploitability computation during exact-mode solve

**Diagnostic infrastructure:**
- `bench-rollout` trainer subcommand — per-call latency + hands/sec measurement
- `validate-rollout` trainer subcommand — compare sampled vs exhaustive boundary CFVs
- `compare-solve` trainer subcommand — run a spot through both Subgame and Exact, diff strategies, report mass-moved and action-class bias
- `--dump-boundary-cfvs` flag on compare-solve — inspect stored CFVs per-boundary per-player per-bias

**izod fixes (incremental correctness improvements):**
- Fix #1: `RolloutLeafEvaluator` in K-continuation precompute now uses per-boundary pot instead of root pot
- Fix #2: `TreeAction::Bet` chip values now correctly scaled to unit-game units before clamping
- Fix #3: Per-boundary opponent reach propagated via blueprint-strategy tree walk (was using unfiltered root range)
- Fix #4: `subgame_depth_limit: Option<u8>` knob exposed end-to-end (backend + CLI + Settings UI); **default still 0**

## The open problem

**Subgame mode at depth=0 is architecturally broken** for the repro scenario and likely similar high-SPR spots.

### Repro scenario

```bash
./target/release/poker-solver-trainer compare-solve \
  --bundle local_data/blueprints/1k_100bb_brdcfr_v2 \
  --snapshot snapshot_0013 \
  --spot "sb:2bb,bb:10bb,sb:22bb,bb:call|Jd9d7d" \
  --iters 200 \
  --verbose
```

Bundle: `1k_100bb_brdcfr_v2` (UI display name: "1000k no limp Apr 5").
Spot: 3-bet pot flop, BB facing SB's 22bb 3-bet-call, board `Jd9d7d`, eff stack ~80bb, SPR ~1.8.

### Measurements

| Depth | Exploitability | Memory | Wall | Boundaries | Status |
|-------|----------------|--------|------|------------|--------|
| 0 | 11,354 mbb/hand | 15 MB | 13s | 15 | **shipped, broken** |
| 1 | — | 50 MB | >600s (aborted) | 1,495 | **blocked by precompute cost** |
| 2 | 38.61 mbb/hand | 2.4 GB | 58s | 0 | equivalent to exact |
| exact | 38.61 mbb/hand | 2.4 GB | 59s | 0 | reference baseline |

### Why depth=0 is broken

At depth=0, boundaries sit immediately after flop actions. ALL turn+river play is approximated by blueprint rollout at each boundary. Even with perfect K=1 equity at boundaries, the ceiling is ~7,000 mbb/hand on this spot — structural limit from the shallow cut.

Per-action-class bias analysis confirms it (after all 4 fixes, Exact–Subgame):
```
Check      -0.486  (subgame under-checks)
Bet/Raise  +0.138  (subgame over-bets)
AllIn      +0.347  (subgame over-jams — stuck across all fixes)
```

At depth=2 (=exact), all biases are 0.0 — confirming the residual bias IS the shallow-depth cut, not any remaining bug.

### Why depth=1 is impractical with current architecture

Modicum K-continuation precompute = 1,495 boundaries × 4 biases × 2 players = **11,960 sequential rollout calls**. At ~0.1-0.2s each, precompute takes 20+ minutes before DCFR iter 1.

## Recommended next steps (priority order)

### 1. Parallelize the K-continuation precompute (bean `jpwu`, filed 2026-04-19)

**Why first:** unlocks depth=1 as a practical option. Current loop in `crates/tauri-app/src/game_session.rs:1921-1974` runs serially over `(ordinal × bias × player)` tuples. Each call is independent (writes to distinct `set_boundary_cfvs_multi(ordinal, player, ki, ...)` slots). Should be 4-8× speedup on a multicore machine.

**Expected impact:** 20-min precompute → 3-5 min. Makes depth=1 usable. Whether that's enough to close exploitability to Exact levels is the open question — measure with compare-solve at `--subgame-depth-limit 1` post-fix.

### 2. Replace rollout precompute with cfvnet neural boundary evaluator (new bean — needs to be filed)

**Why:** the project already has cfvnet infrastructure (`crates/cfvnet/`) — a neural net trained to output CFVs given (board, ranges, pot, stacks). Using it instead of rollout precompute would eliminate the Modicum K-continuation bottleneck entirely: one forward pass per boundary × player, microseconds vs ~0.1s per rollout.

**Blockers to investigate:**
- Is there a cfvnet model trained for this spot's abstraction (1k buckets, 100bb)? Check `local_data/` for model files.
- How does `NeuralBoundaryEvaluator` integrate with `SolveBoundaryEvaluator` (see `postflop.rs:1557-1560` for the fallback wiring)?
- Is cfvnet outputting single-continuation CFVs or multi-continuation (K=4)? Modicum expects K=4.

### 3. Lazy on-demand boundary CFV computation

**Why:** current precompute evaluates EVERY boundary up front. DCFR traversals may only visit a fraction of them during the solve. Lazy evaluation (compute on first visit, cache) could dramatically reduce precompute cost — especially at deeper depths where most boundaries are unreached.

**Complexity:** requires threading a cache through `SolveBoundaryEvaluator` and being careful about thread-safety under rayon. Much bigger refactor than #1 or #2.

### 4. Hide or mark Subgame as experimental in UI (stopgap)

While the real fix lands, Subgame at depth=0 is actively harmful — users clicking it get worse-than-blueprint strategies. Gate it off (or rename "Subgame [experimental]") to funnel traffic to Exact. Low-effort, one-commit change. I did not land this but recommend it.

## Open beans relevant to this work

- **`poker_solver_rust-izod`** (critical, status: todo, has full investigation notes) — Subgame converges worse than blueprint
- **`poker_solver_rust-jpwu`** (high, status: todo) — Parallelize Modicum K-continuation precompute
- **`poker_solver_rust-lpok`** (normal, status: todo) — Target Exploitability not ending exact solve early
- **`poker_solver_rust-bxip`** (normal, status: todo) — `build_subgame_solver` has 16+ params, needs options struct refactor
- **`poker_solver_rust-v55b`** (high, status: todo) — Test suite runtime + GPU test cleanup

## Useful context for whoever picks this up

### Architecture primer

Subgame mode is a **Modicum-style K-continuation depth-limited solver** (Brown/Sandholm/Amos NeurIPS 2018):
- Solve the "near" portion of the game tree with DCFR
- At depth boundary leaves, use precomputed CFVs from rollouts of K=4 biased continuations (Unbiased, Fold-biased, Call-biased, Raise-biased)
- The K continuations provide robustness against opponent deviation from the blueprint

### Key files

- `crates/tauri-app/src/game_session.rs` — solve loops, precompute, boundary evaluator integration
  - `game_solve_core` at ~1677: top-level solve orchestration
  - `build_solve_game` at ~1099: tree construction, accepts `depth_limit_override`
  - `SolveBoundaryEvaluator::compute_cfvs` at ~1416: per-call boundary CFV path (live during DCFR)
  - K-continuation precompute loop at ~1921-1974
- `crates/tauri-app/src/postflop.rs` — rollout evaluator, subgame Tauri commands
  - `RolloutLeafEvaluator` at ~357: the rollout evaluator
  - `rollout_chip_values_with_state` at ~411: the hot path (sampled, depth-gated)
  - `compute_boundary_reach` at ~<wherever Fix #3 landed it>: tree walk producing per-boundary opp reach
- `crates/core/src/blueprint_v2/continuation.rs` — rollout_inner, sample_action_index, apply_action (with chip↔unit scaling after Fix #2)
- `crates/trainer/src/compare_solve.rs` — the debug harness (mirrors the precompute + solve loop standalone)

### How to reproduce the bug

```bash
cd /Users/coreco/code/poker_solver_rust
cargo build --release -p poker-solver-trainer
./target/release/poker-solver-trainer compare-solve \
  --bundle local_data/blueprints/1k_100bb_brdcfr_v2 \
  --snapshot snapshot_0013 \
  --spot "sb:2bb,bb:10bb,sb:22bb,bb:call|Jd9d7d" \
  --iters 200 \
  --verbose
```

Expected: Subgame ~11,354 mbb/hand, Exact ~38.61 mbb/hand. Mass-moved ~0.58 at root.

### Session memory saved

- `feedback_diagnose_config_first.md` — check user config values before dispatching diagnostic agents (localStorage caused a wild goose chase earlier in the session)

## Suggested opening prompt for next session

> I'm continuing work on the Subgame solver. Read `docs/plans/2026-04-19-subgame-next-steps.md` for full context. The top priority is bean `jpwu` (parallelize K-continuation precompute) — that's the blocker for making depth=1 usable. Use `compare-solve` against the izod repro spot to verify any change improves exploitability.
