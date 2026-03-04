# Core Crate Function Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Refactor 16 production functions > 60 LOC in `crates/core/src/` to be under 60 LOC each, improving readability without behavioral changes.

**Architecture:** Independent refactors per module. Extract phase helpers for iteration loops, apply TraverseCtx pattern for recursive traversals, reduce argument counts with context structs. No cross-module unification.

**Tech Stack:** Rust, rayon (parallelism), existing CFR/equity infrastructure.

---

## Workstream A: `postflop_exhaustive.rs` (5 functions)

### Task A1: Refactor `exhaustive_solve_one_flop` (148 LOC -> ~50)

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs:802-949`

**Step 1: Read the function and identify phase boundaries**

The iteration loop body at lines 858-949 has 5 distinct phases:
1. Pruning check + traversal (lines 866-887)
2. Counter snapshotting + accumulation (lines 890-893)
3. DCFR discounting + delta merge + regret floor (lines 896-915)
4. Extremal regrets computation (lines 920-924)
5. Exploitability check + progress reporting (lines 926-948)

**Step 2: Create a `SolveLoopCtx` struct to bundle immutable args**

Bundle: `tree`, `layout`, `equity_table`, `dcfr`, `config`, `on_progress`, `counters`, `flop_name`. Keep `bufs` as `&mut` param.

```rust
struct SolveLoopCtx<'a, F: Fn(BuildPhase) + Sync> {
    tree: &'a PostflopTree,
    layout: &'a PostflopLayout,
    equity_table: &'a [f64],
    dcfr: &'a DcfrParams,
    config: &'a PostflopModelConfig,
    on_progress: &'a F,
    counters: Option<&'a SolverCounters>,
    flop_name: String,
    pairs: Vec<(u16, u16)>,
}
```

**Step 3: Extract `apply_dcfr_and_merge_deltas` helper**

```rust
/// Apply DCFR discounting, merge traversal deltas, and clamp regrets.
#[inline]
fn apply_dcfr_and_merge_deltas(&self, bufs: &mut FlopBuffers, iteration: u64) {
    if self.dcfr.should_discount(iteration) {
        self.dcfr.discount_regrets(&mut bufs.regret_sum, iteration);
        self.dcfr.discount_strategy_sums(&mut bufs.strategy_sum, iteration);
    }
    add_into(&mut bufs.regret_sum, &bufs.delta.0);
    add_into(&mut bufs.strategy_sum, &bufs.delta.1);
    if self.dcfr.should_floor_regrets() {
        self.dcfr.floor_regrets(&mut bufs.regret_sum);
    }
    if self.config.regret_floor > 0.0 && self.config.prune_warmup > 0 {
        let floor = -self.config.regret_floor;
        for v in bufs.regret_sum.iter_mut() {
            *v = (*v).max(floor);
        }
    }
}
```

**Step 4: Extract `check_exploitability` helper**

```rust
fn check_exploitability(&self, bufs: &FlopBuffers, iter: usize, num_iterations: usize) -> Option<f64> {
    let freq = self.config.exploitability_freq.max(1);
    if iter >= 1 && (iter % freq == freq - 1 || iter == num_iterations - 1) {
        Some(compute_exploitability(self.tree, self.layout, &bufs.strategy_sum, self.equity_table))
    } else {
        None
    }
}
```

**Step 5: Extract `report_iteration_progress` helper**

```rust
fn report_iteration_progress(&self, iteration: usize, max_iterations: usize, exploitability: f64,
    total_action_slots: u64, pruned_action_slots: u64, max_pos: f64, min_neg: f64) {
    (self.on_progress)(BuildPhase::FlopProgress {
        flop_name: self.flop_name.clone(),
        stage: FlopStage::Solving { iteration, max_iterations,
            delta: exploitability, metric_label: "mBB/h".into(),
            total_action_slots, pruned_action_slots,
            max_positive_regret: max_pos, min_negative_regret: min_neg },
    });
}
```

**Step 6: Extract `run_traversal` helper**

```rust
fn run_traversal(&self, bufs: &mut FlopBuffers, iter: usize) -> bool {
    let prune_active = self.config.prune_warmup > 0
        && iter >= self.config.prune_warmup
        && (self.config.prune_explore_freq == 0 || iter % self.config.prune_explore_freq != 0);
    let ctx = PostflopCfrCtx { tree: self.tree, layout: self.layout,
        equity_table: self.equity_table, snapshot: &bufs.regret_sum,
        iteration: iter as u64 + 1, dcfr: self.dcfr, prune_active,
        prune_regret_threshold: self.config.prune_regret_threshold,
        counters: self.counters };
    parallel_traverse_pooled(&ctx, &self.pairs, std::slice::from_mut(&mut bufs.delta));
    prune_active
}
```

**Step 7: Rewrite `exhaustive_solve_one_flop` using helpers**

The main function constructs `SolveLoopCtx`, emits initial progress, then has a tight loop calling the 4 helpers.

**Step 8: Run tests and clippy**

```bash
cargo test -p poker-solver-core -- postflop_exhaustive
cargo clippy -p poker-solver-core
```

**Step 9: Commit**

```bash
git add crates/core/src/preflop/postflop_exhaustive.rs
git commit -m "refactor(core): decompose exhaustive_solve_one_flop into phase helpers"
```

---

### Task A2: Refactor `compute_equity_table` (128 LOC -> ~40)

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs:212-339`

**Step 1: Extract `build_combo_index` helper**

Lines 226-237 build `combo_cards`, `combo_mask`, `canonical_range` from `combo_map`. Extract into:
```rust
fn build_combo_index(combo_map: &[Vec<(Card, Card)>], n: usize) -> (Vec<(Card, Card)>, Vec<u64>, Vec<Range<usize>>)
```

**Step 2: Extract `enumerate_non_flop_boards` helper**

Lines 254-265 enumerate (turn, river) pairs that don't conflict with the flop. Extract into:
```rust
fn enumerate_non_flop_boards(deck_bits: &[u8; 52], flop_mask: u64) -> Vec<(usize, usize)>
```

**Step 3: Simplify the main function**

`compute_equity_table` becomes: build combo index -> enumerate boards -> parallel fold/reduce over boards -> convert to equity fractions.

**Step 4: Run tests and clippy, commit**

```bash
cargo test -p poker-solver-core -- postflop_exhaustive
cargo clippy -p poker-solver-core
git commit -m "refactor(core): extract helpers from compute_equity_table"
```

---

### Task A3: Refactor `best_response_ev` + `eval_with_avg_strategy` (86 + 66 LOC)

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs:571-656` and `999-1064`

**Step 1: Create a shared `EvalCtx` struct**

Both functions share `tree`, `layout`, `equity_table`, and one of `strategy_sum`. Bundle into:
```rust
struct EvalCtx<'a> {
    tree: &'a PostflopTree,
    layout: &'a PostflopLayout,
    strategy_sum: &'a [f64],
    equity_table: &'a [f64],
}
```

**Step 2: Implement `best_response_ev` as method on `EvalCtx`**

Extract hero best-response (argmax) into `br_hero_decision` helper. Extract opponent avg-strategy play into `avg_strategy_decision` helper (shared with `eval_with_avg_strategy`).

**Step 3: Implement `eval_with_avg_strategy` as method on `EvalCtx`**

Both decision branches use avg strategy — simpler than best_response_ev.

**Step 4: Run tests and clippy, commit**

```bash
cargo test -p poker-solver-core -- postflop_exhaustive
cargo clippy -p poker-solver-core
git commit -m "refactor(core): extract EvalCtx for best_response_ev and eval_with_avg_strategy"
```

---

### Task A4: Refactor `build_exhaustive` (93 LOC -> ~50)

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs:1072-1164`

**Step 1: Extract `solve_and_extract_flop` helper**

The per-flop closure body (lines 1106-1155) handles equity computation, buffer management, solving, value extraction, and progress reporting. Extract into a standalone function.

**Step 2: Simplify `build_exhaustive`**

Main function becomes: setup DCFR params -> par_iter over flops calling helper -> flatten results.

**Step 3: Run tests and clippy, commit**

```bash
cargo test -p poker-solver-core -- postflop_exhaustive
cargo clippy -p poker-solver-core
git commit -m "refactor(core): extract solve_and_extract_flop from build_exhaustive"
```

---

## Workstream B: `postflop_mccfr.rs` (5 functions)

### Task B1: Refactor `mccfr_traverse` (143 LOC -> ~55)

**Files:**
- Modify: `crates/core/src/preflop/postflop_mccfr.rs:272-414`

**Step 1: Create `MccfrTraverseCtx` struct**

Bundle the 13 parameters into immutable context + per-call varying args:

```rust
struct MccfrTraverseCtx<'a> {
    tree: &'a PostflopTree,
    layout: &'a PostflopLayout,
    snapshot: &'a [f64],
    board: [Card; 5],
    iteration: u64,
    dcfr: &'a DcfrParams,
}

struct MccfrTraverseArgs {
    node_idx: u32,
    hero_bucket: u16,
    opp_bucket: u16,
    hero_pos: u8,
    hero_hand: [Card; 2],
    opp_hand: [Card; 2],
    reach_hero: f64,
    reach_opp: f64,
}
```

**Step 2: Extract `mccfr_traverse_hero` and `mccfr_traverse_opponent` helpers**

Mirror the exhaustive pattern: hero builds action values + updates regret/strategy deltas, opponent computes strategy-weighted sum.

**Step 3: Extract `mccfr_terminal_payoff` helper**

Lines 290-311 handle fold + showdown payoff with concrete hand ranking. Extract as free function.

**Step 4: Rewrite `mccfr_traverse` as dispatcher on `MccfrTraverseCtx`**

Pattern-match Terminal/Chance/Decision and delegate.

**Step 5: Update call sites**

Update `MccfrCfrCtx::traverse_pair` to construct `MccfrTraverseCtx` + `MccfrTraverseArgs`.

**Step 6: Run tests and clippy, commit**

```bash
cargo test -p poker-solver-core -- postflop_mccfr
cargo clippy -p poker-solver-core
git commit -m "refactor(core): decompose mccfr_traverse with context struct and helpers"
```

---

### Task B2: Refactor `mccfr_solve_one_flop` (90 LOC -> ~45)

**Files:**
- Modify: `crates/core/src/preflop/postflop_mccfr.rs:116-205`

**Step 1: Extract phase helpers**

Mirror the exhaustive pattern:
- `sample_and_traverse` — deal sampling + parallel traversal
- `merge_mccfr_deltas` — LCFR weighting + delta merge

**Step 2: Simplify the main loop, run tests, commit**

```bash
cargo test -p poker-solver-core -- postflop_mccfr
git commit -m "refactor(core): extract phase helpers from mccfr_solve_one_flop"
```

---

### Task B3: Refactor `mccfr_eval_with_avg_strategy` (90 LOC -> ~50)

**Files:**
- Modify: `crates/core/src/preflop/postflop_mccfr.rs:562-651`

**Step 1: Create `MccfrEvalCtx` struct + extract decision helpers**

Same pattern as `EvalCtx` in exhaustive. Bundle immutable args, extract terminal payoff handling.

**Step 2: Run tests and clippy, commit**

```bash
cargo test -p poker-solver-core -- postflop_mccfr
git commit -m "refactor(core): extract MccfrEvalCtx for mccfr_eval_with_avg_strategy"
```

---

### Task B4: Refactor `mccfr_extract_values` (84 LOC -> ~45)

**Files:**
- Modify: `crates/core/src/preflop/postflop_mccfr.rs:435-518`

**Step 1: Extract `evaluate_hand_pair` helper**

The inner loop body (deal sampling + tree evaluation per pair) is the extraction target.

**Step 2: Run tests and clippy, commit**

```bash
cargo test -p poker-solver-core -- postflop_mccfr
git commit -m "refactor(core): extract evaluate_hand_pair from mccfr_extract_values"
```

---

### Task B5: Refactor `build_mccfr` (63 LOC -> ~40)

**Files:**
- Modify: `crates/core/src/preflop/postflop_mccfr.rs:41-103`

**Step 1: Extract `solve_and_extract_mccfr_flop` helper**

Mirror the `build_exhaustive` pattern — extract per-flop closure body.

**Step 2: Run tests and clippy, commit**

```bash
cargo test -p poker-solver-core -- postflop_mccfr
git commit -m "refactor(core): extract solve_and_extract_mccfr_flop from build_mccfr"
```

---

## Workstream C: Other Modules (6 functions)

### Task C1: Refactor `tree.rs:build_recursive` (99 LOC -> ~50)

**Files:**
- Modify: `crates/core/src/preflop/tree.rs:116-214`

**Step 1: Extract `generate_available_actions` helper**

The action generation logic (fold/call/raises/all-in) spans ~40 lines. Extract into a function returning `Vec<(Action, BuildState)>`.

**Step 2: Extract `is_terminal_state` helper**

Terminal detection (all folded, round complete after flop) is ~15 lines.

**Step 3: Simplify `build_recursive`, run tests, commit**

```bash
cargo test -p poker-solver-core -- preflop::tree
git commit -m "refactor(core): extract helpers from build_recursive"
```

---

### Task C2: Refactor `rank_array_cache.rs:derive_equity_table` (91 LOC -> ~45)

**Files:**
- Modify: `crates/core/src/preflop/rank_array_cache.rs:229-319`

**Step 1: Extract `build_rank_combo_index` helper**

Lines that build combo_cards, combo_mask, canonical_range from rank data.

**Step 2: Extract `accumulate_board_equity` helper**

Per-board rank comparison and equity accumulation.

**Step 3: Run tests and clippy, commit**

```bash
cargo test -p poker-solver-core -- rank_array_cache
git commit -m "refactor(core): extract helpers from derive_equity_table"
```

---

### Task C3: Refactor `rank_array_cache.rs:compute_rank_arrays` (62 LOC -> ~40)

**Files:**
- Modify: `crates/core/src/preflop/rank_array_cache.rs:160-221`

**Step 1: Extract `build_combo_list` and `evaluate_combo_ranks` helpers**

**Step 2: Run tests, commit**

```bash
cargo test -p poker-solver-core -- rank_array_cache
git commit -m "refactor(core): extract helpers from compute_rank_arrays"
```

---

### Task C4: Refactor `simulation.rs:run_simulation` (74 LOC -> ~45)

**Files:**
- Modify: `crates/core/src/simulation.rs:359-432`

**Step 1: Extract `setup_game` and `play_one_hand` helpers**

Game setup (deck builder, agent generators, dealer rotation) and per-hand game loop.

**Step 2: Run tests, commit**

```bash
cargo test -p poker-solver-core -- simulation
git commit -m "refactor(core): extract helpers from run_simulation"
```

---

### Task C5: Refactor `solver.rs:cfr_traverse` (73 LOC -> ~55)

**Files:**
- Modify: `crates/core/src/preflop/solver.rs:483-555`

**Step 1: Extract `terminal_showdown_value` helper**

The showdown branch (lines 499-517) has conditional postflop vs raw equity logic. Extract into a helper.

**Step 2: Run tests, commit**

```bash
cargo test -p poker-solver-core -- preflop::solver
git commit -m "refactor(core): extract terminal_showdown_value from cfr_traverse"
```

---

### Task C6: Final verification

**Step 1: Run full test suite**

```bash
cargo test -p poker-solver-core
```

**Step 2: Run clippy**

```bash
cargo clippy -p poker-solver-core
```

**Step 3: Verify no function exceeds 60 LOC**

Re-run the audit to confirm all targets are under 60 LOC.

**Step 4: Commit any remaining fixes**

---

## Agent Team & Execution Order

| Agent | Tasks | Isolation |
|-|-|-|
| `rust-developer` #1 | A1, A2, A3, A4 (exhaustive) | worktree |
| `rust-developer` #2 | B1, B2, B3, B4, B5 (mccfr) | worktree |
| `rust-developer` #3 | C1, C2, C3, C4, C5 (other) | worktree |

**Parallel execution**: All 3 workstreams run simultaneously. Each agent works in its own worktree.

**Review (after each workstream merges)**:
- `idiomatic-rust-enforcer` — naming, patterns, clippy
- `rust-perf-reviewer` — inlining, allocations, hot-path impact

**Final**: C6 verification after all 3 workstreams merge to main.
