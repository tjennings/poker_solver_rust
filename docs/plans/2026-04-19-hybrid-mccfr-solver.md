# Hybrid MCCFR Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace the broken Modicum K=4 rollout-precompute "subgame" solver with a hybrid DCFR solver that evaluates depth-limited boundaries via live MCCFR sampling, with three config knobs (`depth_limit`, `boundary_refresh_interval`, `samples_per_refresh`) and telemetry for MCCFR performance.

**Architecture:** Near-tree DCFR (identical to current "exact") plus a new `HybridBoundaryEvaluator` that caches per-boundary CFVs and refreshes them periodically via single-pass MCCFR rollouts through the blueprint (unbiased). One sampling pass produces CFVs for *both* OOP and IP from one trajectory set. Trait changes collapse the K=4 multi-continuation storage back to single-K.

**Tech stack:** Rust (workspace crates: `range-solver`, `tauri-app`, `trainer`, `core`), rayon for per-boundary parallelism, React/TypeScript for Tauri frontend, existing `RolloutLeafEvaluator` for the sampler.

**References:**
- Design: `docs/plans/2026-04-19-hybrid-mccfr-solver-design.md`
- Research: `docs/research/2026-04-19-subgame-solving-literature.md`
- Bean: `poker_solver_rust-mx1j`
- Repro: `./target/release/poker-solver-trainer compare-solve --bundle local_data/blueprints/1k_100bb_brdcfr_v2 --snapshot snapshot_0013 --spot "sb:2bb,bb:10bb,sb:22bb,bb:call|Jd9d7d" --iters 200 --verbose`

**Success criteria (run at Phase 10):**
1. Depth=1 Hybrid exploitability on izod repro ≤ 500 mbb/hand of Exact (38.6 mbb). Stretch: ≤ 100 mbb.
2. Depth=1 Hybrid wall-time ≤ 2× Exact (58s). Stretch: ≤ 1×.
3. First DCFR iteration begins within 2 s of solve start.
4. When `depth_limit` ≥ tree_depth, Hybrid produces byte-identical strategies to Exact.
5. Full `cargo test` suite passes in < 60 s.

**Parallelization guidance (for subagent dispatch):**
- Phase 1 (sampler primitive) and Phase 3 (trait extension) are independent and can run in parallel.
- Phases 4-6 depend on 1+2+3 complete.
- Phases 7, 8, 9 depend on 4 complete but are independent of each other.

---

## Phase 0 — Baseline verification

### Task 0.1: Confirm clean baseline

**Files:** none (read-only)

**Step 1:** Verify working tree.
```bash
git status
```
Expected: clean or only the existing plan/research/bean files from commit `f2f918aa`.

**Step 2:** Run test suite and confirm < 60 s.
```bash
time cargo test --workspace --quiet
```
Expected: all pass, total time under 60 s. If not, pause plan and fix (`feedback_iterative_pipeline`).

**Step 3:** Build release binaries used by the izod repro.
```bash
cargo build --release -p poker-solver-trainer
```

**Step 4:** Capture baseline numbers by running compare-solve on izod spot (broken current behavior).
```bash
./target/release/poker-solver-trainer compare-solve \
  --bundle local_data/blueprints/1k_100bb_brdcfr_v2 \
  --snapshot snapshot_0013 \
  --spot "sb:2bb,bb:10bb,sb:22bb,bb:call|Jd9d7d" \
  --iters 200 --verbose 2>&1 | tee /tmp/baseline-izod.log
```
Expected (from prior session): Subgame ≈ 11,354 mbb/hand, Exact ≈ 38.6 mbb/hand. Save log for Phase 10 comparison.

**Step 5:** No commit — this is pre-work.

---

## Phase 1 — Domain: MCCFR boundary sampler primitive

### Task 1.1: Write failing test for `sample_boundary_cfvs`

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs` (add to existing `#[cfg(test)] mod tests` block near the bottom)
- Also inspect: `crates/tauri-app/src/postflop.rs:371-429` (RolloutLeafEvaluator shape)

**Context:** We need a new public method on `RolloutLeafEvaluator` that takes OOP/IP ranges and returns `BoundaryCfvs { oop_cfvs, ip_cfvs }` from `n_samples` single-trajectory rollouts.

**Step 1:** Add a `BoundaryCfvs` struct near top of `postflop.rs` (alongside `RolloutLeafEvaluator`):
```rust
#[derive(Clone, Debug)]
pub struct BoundaryCfvs {
    pub oop_cfvs: Vec<f64>,   // f64 matches rollout_chip_values_with_state output; avoids lossy cast
    pub ip_cfvs:  Vec<f64>,
}
```

> **Plan amendment (2026-04-19, ratified post-Phase-1 spec-compliance review):** `BoundaryCfvs` uses `Vec<f64>` not `Vec<f32>`. Downstream consumers in Phase 2/3 must accept f64 (cast to f32 at the trait boundary if needed).

**Step 2:** Append a failing test to the `#[cfg(test)] mod tests` block at the bottom of `postflop.rs`:
```rust
#[test]
fn sample_boundary_cfvs_returns_both_sides() {
    // Minimal smoke: if the function is callable with uniform ranges and
    // zero samples, both output vectors should be len == num_hands of the
    // respective side and filled with 0.0.
    // Uses the existing postflop test fixtures — mirror pattern in
    // test_rollout_chip_values_with_state if present, or skip if no fixture
    // exists for plain RolloutLeafEvaluator (use ignore! pattern otherwise).
    // ... see Task 1.2 for concrete fixture once code added.
}
```

**Step 3:** Run to confirm it fails for the right reason (`sample_boundary_cfvs` undefined):
```bash
cargo test -p poker-solver-tauri sample_boundary_cfvs_returns_both_sides --no-run 2>&1 | head -20
```
Expected: compile error on the test (method not found).

**Step 4:** Commit the failing test.
```bash
git add crates/tauri-app/src/postflop.rs
git commit -m "test(hybrid): failing test for sample_boundary_cfvs primitive"
```

### Task 1.2: Implement `sample_boundary_cfvs` on `RolloutLeafEvaluator`

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs` (add method alongside `rollout_chip_values_with_state` at ~445-454)

**Step 1:** Implement the method using the existing `rollout_chip_values_with_state` as the per-trajectory worker. Note: `rollout_chip_values_with_state` already takes both oop_range and ip_range and returns per-combo chip values for a given traverser. We call it twice per sample batch (once per traverser=0 and traverser=1) and accumulate into `BoundaryCfvs`.

```rust
impl RolloutLeafEvaluator {
    /// Live MCCFR sampling: draws n_samples trajectories through the
    /// blueprint and returns per-hand CFVs for both players at the
    /// boundary from one integrated sampling pass.
    pub fn sample_boundary_cfvs(
        &self,
        combos: &[[RsPokerCard; 2]],
        board: &[RsPokerCard],
        oop_range: &[f64],
        ip_range:  &[f64],
        boundary_pot: f64,
        boundary_invested: [f64; 2],
        n_samples: u32,
    ) -> BoundaryCfvs {
        assert!(n_samples > 0, "n_samples must be > 0");
        let num_oop = oop_range.len();
        let num_ip  = ip_range.len();

        // Use a temporary per-invocation copy of self with num_rollouts set.
        let mut worker = self.clone_with_num_rollouts(n_samples);

        let oop_chips = worker.rollout_chip_values_with_state(
            combos, board, oop_range, ip_range, 0, boundary_pot, boundary_invested,
        );
        let ip_chips = worker.rollout_chip_values_with_state(
            combos, board, oop_range, ip_range, 1, boundary_pot, boundary_invested,
        );

        // Chip-value outputs are already per-combo; convert to per-hand f32.
        let oop_cfvs: Vec<f32> = oop_chips.into_iter().take(num_oop).map(|v| v as f32).collect();
        let ip_cfvs:  Vec<f32> = ip_chips.into_iter().take(num_ip).map(|v| v as f32).collect();
        BoundaryCfvs { oop_cfvs, ip_cfvs }
    }

    fn clone_with_num_rollouts(&self, n: u32) -> Self {
        let mut clone = Self {
            strategy: self.strategy.clone(),
            abstract_tree: self.abstract_tree.clone(),
            all_buckets: self.all_buckets.clone(),
            decision_idx_map: self.decision_idx_map.clone(),
            abstract_start_node: self.abstract_start_node,
            bias: BiasType::Unbiased,         // Hybrid is always unbiased
            bias_factor: 1.0,
            num_rollouts: n,
            num_opponent_samples: self.num_opponent_samples,
            starting_stack: self.starting_stack,
            root_spr: self.root_spr,
            hand_counter: self.hand_counter.clone(),
            call_counter: self.call_counter.clone(),
            enumerate_decision_depth: self.enumerate_decision_depth,
        };
        clone.num_rollouts = n;
        clone
    }
}
```

**Note:** if `Clone` isn't already derived on `RolloutLeafEvaluator`, add `#[derive(Clone)]` to its struct definition (~line 371). All field types in the listed struct are `Arc`, `Vec`, or `Copy` — clone is cheap and correct.

**Step 2:** Fill in the test body from Task 1.1 with a real fixture. Create a helper in the test module that builds a minimal `RolloutLeafEvaluator` from a synthetic blueprint, or use `crate::postflop::tests::...` helpers if present. If none exist, create a focused fixture:

```rust
#[test]
fn sample_boundary_cfvs_returns_both_sides() {
    use crate::postflop::test_util::minimal_rollout_evaluator;
    let eval = minimal_rollout_evaluator();
    let combos: Vec<[RsPokerCard; 2]> = vec![];  // populate from helper
    let board: Vec<RsPokerCard> = vec![];
    let oop_range = vec![1.0_f64; 10];
    let ip_range  = vec![1.0_f64; 10];
    let result = eval.sample_boundary_cfvs(
        &combos, &board, &oop_range, &ip_range, 44.0, [22.0, 22.0], 1,
    );
    assert_eq!(result.oop_cfvs.len(), 10);
    assert_eq!(result.ip_cfvs.len(),  10);
}
```

If no minimal fixture exists, write the helper function in the test module using the pattern in `test_sample_weighted_basic` (test file already imports `SmallRng`). Keep helper scope module-local.

**Step 3:** Run test to confirm it passes.
```bash
cargo test -p poker-solver-tauri sample_boundary_cfvs_returns_both_sides --quiet
```
Expected: PASS.

**Step 4:** Full crate test to confirm no regression.
```bash
cargo test -p poker-solver-tauri --quiet
```

**Step 5:** Commit.
```bash
git add crates/tauri-app/src/postflop.rs
git commit -m "feat(hybrid): add sample_boundary_cfvs on RolloutLeafEvaluator"
```

### Task 1.3: Convergence test — sampled CFV approaches analytic equity

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs` (add test)

**Step 1:** Add test that runs N=2048 samples against a degenerate blueprint (always-check) on a trivial board/range pair where showdown equity is analytically known. Assert mean |cfv - analytic| < 5% of pot.

```rust
#[test]
fn sample_boundary_cfvs_converges_to_analytic() {
    // Use a 2-hand vs 2-hand setup on a runout board where equity can be
    // computed by brute-force showdown. Blueprint = always-check, so CFV
    // at boundary = pot × (P(win) - 0.5) for heads-up.
    // Set n_samples = 2048; variance should shrink to < 5% of pot.
    // ... (implementation uses minimal_rollout_evaluator with check-only strategy)
}
```

**Step 2:** Run test, confirm pass.
```bash
cargo test -p poker-solver-tauri sample_boundary_cfvs_converges --quiet -- --nocapture
```

**Step 3:** Commit.
```bash
git add crates/tauri-app/src/postflop.rs
git commit -m "test(hybrid): sample_boundary_cfvs converges to analytic equity"
```

---

## Phase 2 — Domain: `HybridBoundaryEvaluator`

### Task 2.1: Failing test — cache + refresh logic

**Files:**
- Create: `crates/tauri-app/src/hybrid_evaluator.rs`

**Step 1:** Create the new module file with the type signature and a failing test:

```rust
//! Hybrid MCCFR boundary evaluator — see docs/plans/2026-04-19-hybrid-mccfr-solver-design.md
use crate::postflop::{BoundaryCfvs, RolloutLeafEvaluator};
use std::collections::HashMap;
use std::sync::{RwLock, atomic::{AtomicU32, Ordering}};

pub struct HybridBoundaryEvaluator {
    sampler: RolloutLeafEvaluator,
    refresh_interval: u32,
    samples_per_refresh: u32,
    cached: RwLock<HashMap<u64, CachedEntry>>,
    current_iter: AtomicU32,
}

struct CachedEntry {
    cfvs: BoundaryCfvs,
    refreshed_at_iter: u32,
}

impl HybridBoundaryEvaluator {
    pub fn new(
        sampler: RolloutLeafEvaluator,
        refresh_interval: u32,
        samples_per_refresh: u32,
    ) -> Self {
        assert!(samples_per_refresh > 0, "samples_per_refresh must be > 0");
        let ri = refresh_interval.max(1);
        Self {
            sampler,
            refresh_interval: ri,
            samples_per_refresh,
            cached: RwLock::new(HashMap::new()),
            current_iter: AtomicU32::new(0),
        }
    }

    pub fn begin_iteration(&self, iter: u32) {
        self.current_iter.store(iter, Ordering::SeqCst);
    }

    pub fn should_refresh(&self, iter: u32, last_refresh: u32) -> bool {
        iter == 0 || (iter - last_refresh) >= self.refresh_interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_refresh_triggers_at_interval() {
        let eval = HybridBoundaryEvaluator {
            sampler: /* dummy — use test helper */ todo!(),
            refresh_interval: 10,
            samples_per_refresh: 100,
            cached: RwLock::new(HashMap::new()),
            current_iter: AtomicU32::new(0),
        };
        assert!(eval.should_refresh(0, 0));   // first iter
        assert!(!eval.should_refresh(5, 0));  // within interval
        assert!(eval.should_refresh(10, 0));  // at interval
    }

    #[test]
    fn samples_zero_panics_in_new() {
        // Use std::panic::catch_unwind with a dummy sampler.
        // Assert that HybridBoundaryEvaluator::new(.., .., 0) panics.
    }
}
```

**Step 2:** Register module in `crates/tauri-app/src/lib.rs`:
```rust
pub mod hybrid_evaluator;
```

**Step 3:** Run — expect compile failure on `todo!()` in test.
```bash
cargo test -p poker-solver-tauri hybrid_evaluator --no-run 2>&1 | head -20
```

**Step 4:** Don't commit yet — next task replaces `todo!()`.

### Task 2.2: Wire refresh logic — `compute_cfvs` method

**Files:**
- Modify: `crates/tauri-app/src/hybrid_evaluator.rs`
- Modify: `crates/tauri-app/src/postflop.rs` (export `BoundaryCfvs` if not already `pub`)

**Step 1:** Implement the main entry point:

```rust
impl HybridBoundaryEvaluator {
    pub fn compute_cfvs(
        &self,
        boundary_id: u64,
        combos: &[RsPokerCard; 2],   // align with sampler's combos type
        board: &[RsPokerCard],
        oop_range: &[f64],
        ip_range:  &[f64],
        boundary_pot: f64,
        boundary_invested: [f64; 2],
    ) -> BoundaryCfvs {
        let iter = self.current_iter.load(Ordering::SeqCst);

        // Fast path: read-lock, return cached if fresh.
        {
            let guard = self.cached.read().unwrap();
            if let Some(entry) = guard.get(&boundary_id) {
                if !self.should_refresh(iter, entry.refreshed_at_iter) {
                    return entry.cfvs.clone();
                }
            }
        }

        // Slow path: re-sample, write-lock, update cache.
        let fresh = self.sampler.sample_boundary_cfvs(
            combos, board, oop_range, ip_range,
            boundary_pot, boundary_invested, self.samples_per_refresh,
        );
        let mut guard = self.cached.write().unwrap();
        guard.insert(boundary_id, CachedEntry { cfvs: fresh.clone(), refreshed_at_iter: iter });
        fresh
    }
}
```

**Step 2:** Fill in `todo!()` in `should_refresh_triggers_at_interval` test — create a helper `dummy_evaluator()` that returns a `HybridBoundaryEvaluator` with a mock/sentinel sampler that never gets called (refresh logic test doesn't touch `sampler`). Simplest path: construct the struct fields directly in the test rather than via `new()`, skipping the sampler requirement. If compiler complains about the field visibility from the test module, mark internal fields `pub(crate)`.

**Step 3:** Run tests.
```bash
cargo test -p poker-solver-tauri hybrid_evaluator --quiet
```
Expected: both tests pass.

**Step 4:** Commit.
```bash
git add crates/tauri-app/src/hybrid_evaluator.rs crates/tauri-app/src/lib.rs crates/tauri-app/src/postflop.rs
git commit -m "feat(hybrid): HybridBoundaryEvaluator with cache + refresh gating"
```

### Task 2.3: Implement `BoundaryEvaluator` trait for `HybridBoundaryEvaluator`

**Files:**
- Modify: `crates/tauri-app/src/hybrid_evaluator.rs`
- Reference: `crates/range-solver/src/game/mod.rs:18-43` (trait definition)

**Context:** Current trait signature is single-player (returns `Vec<f32>` for one player at a time) with a `continuation_index` parameter for K=4 dispatch. Since we're dropping K=4, `continuation_index` is always 0, but we keep the current trait for minimal churn in near-tree code. We internally call our `compute_cfvs(..)` and return only the requested player's side.

**Step 1:** Add trait impl:
```rust
impl range_solver::game::BoundaryEvaluator for HybridBoundaryEvaluator {
    fn num_continuations(&self) -> usize { 1 }

    fn compute_cfvs(
        &self,
        player: usize,
        pot: i32,
        _remaining_stack: f64,
        opponent_reach: &[f32],
        num_hands: usize,
        _continuation_index: usize,
    ) -> Vec<f32> {
        // This adapter calls our richer compute_cfvs. The trait signature
        // only gives us opponent_reach; we need both sides. For hybrid we
        // store the "other side's" range on a thread-local or require
        // callers to set it via a separate `set_ranges_for_boundary`.
        //
        // Simplest wiring for this task: expose two methods — the rich
        // compute_cfvs (our own, called by the near-tree bridge) and this
        // trait stub which forwards to the cache assuming ranges were
        // prepared. In Phase 4 we replace the trait call site with the
        // richer API.
        unimplemented!("use HybridBoundaryEvaluator::compute_cfvs directly — trait stub")
    }
}
```

**Decision:** the trait signature is structurally incompatible with hybrid's both-ranges requirement. Rather than contort, in Phase 3 we extend the trait. This stub ships as a placeholder with a clear panic message; Phase 3 replaces it.

**Step 2:** Commit the placeholder so Phase 3 can diff against it clearly.
```bash
git add crates/tauri-app/src/hybrid_evaluator.rs
git commit -m "feat(hybrid): stub BoundaryEvaluator trait — replaced in Phase 3"
```

---

## Phase 3 — Trait extension (runs in parallel with Phase 1/2)

### Task 3.1: Extend `BoundaryEvaluator` trait with both-ranges method

**Files:**
- Modify: `crates/range-solver/src/game/mod.rs:18-43`
- Modify: `crates/tauri-app/src/game_session.rs:1365-1491` (existing `SolveBoundaryEvaluator::compute_cfvs`)

**Context:** The current trait returns single-side CFVs. We add a new default method `compute_cfvs_both` that by default calls the single-side method twice (for backwards compat) but evaluators can override it for amortization.

**Step 1:** Failing test (in `range-solver` crate, add to existing test module):
```rust
#[test]
fn boundary_evaluator_default_compute_cfvs_both_delegates_to_single_side() {
    // Minimal mock BoundaryEvaluator that returns `player as f32`
    // replicated num_hands times. Verify compute_cfvs_both returns
    // {oop_cfvs: [0.0; n], ip_cfvs: [1.0; n]}.
}
```

**Step 2:** Extend trait:
```rust
pub trait BoundaryEvaluator: Send + Sync {
    fn num_continuations(&self) -> usize { 1 }

    fn compute_cfvs(
        &self,
        player: usize,
        pot: i32,
        remaining_stack: f64,
        opponent_reach: &[f32],
        num_hands: usize,
        continuation_index: usize,
    ) -> Vec<f32>;

    /// Amortized computation: returns CFVs for both players in one pass.
    /// Default implementation calls compute_cfvs twice (backwards compat).
    /// Implementations with internal amortization (e.g., HybridBoundaryEvaluator)
    /// should override.
    fn compute_cfvs_both(
        &self,
        pot: i32,
        remaining_stack: f64,
        oop_reach: &[f32],
        ip_reach:  &[f32],
        num_oop: usize,
        num_ip:  usize,
        continuation_index: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let oop = self.compute_cfvs(0, pot, remaining_stack, ip_reach,  num_oop, continuation_index);
        let ip  = self.compute_cfvs(1, pot, remaining_stack, oop_reach, num_ip,  continuation_index);
        (oop, ip)
    }
}
```

**Step 3:** Run trait test.
```bash
cargo test -p range-solver boundary_evaluator --quiet
```

**Step 4:** Workspace build to confirm existing impls still compile (they use the default).
```bash
cargo build --workspace --all-targets
```

**Step 5:** Commit.
```bash
git add crates/range-solver/src/game/mod.rs
git commit -m "feat(range-solver): add compute_cfvs_both to BoundaryEvaluator trait"
```

### Task 3.2: Override `compute_cfvs_both` on `HybridBoundaryEvaluator`

**Files:**
- Modify: `crates/tauri-app/src/hybrid_evaluator.rs`

**Step 1:** Replace the stub trait impl (from Task 2.3) with the real override:
```rust
impl range_solver::game::BoundaryEvaluator for HybridBoundaryEvaluator {
    fn num_continuations(&self) -> usize { 1 }

    fn compute_cfvs(
        &self, _player: usize, _pot: i32, _remaining_stack: f64,
        _opponent_reach: &[f32], _num_hands: usize, _continuation_index: usize,
    ) -> Vec<f32> {
        panic!("HybridBoundaryEvaluator uses compute_cfvs_both; call that method instead.");
    }

    fn compute_cfvs_both(
        &self,
        pot: i32,
        _remaining_stack: f64,
        oop_reach: &[f32],
        ip_reach:  &[f32],
        _num_oop: usize,
        _num_ip:  usize,
        _continuation_index: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        // Convert reach f32 → f64 for sampler API; hand off to cache.
        let oop_range: Vec<f64> = oop_reach.iter().map(|&v| v as f64).collect();
        let ip_range:  Vec<f64> = ip_reach.iter().map(|&v| v as f64).collect();
        // combos + board + boundary_pot + boundary_invested are NOT in the
        // trait signature. The near-tree caller must prepare them via a
        // wrapper (see Phase 4 Task 4.1). For the trait path alone, this
        // panics; the real dispatch goes through HybridBoundaryEvaluator's
        // own public `compute_cfvs` with full context.
        panic!("HybridBoundaryEvaluator.compute_cfvs_both requires pre-bound context (see Phase 4).");
    }
}
```

**Note:** This looks awkward because the trait doesn't carry enough context. Phase 4 adds a thin wrapper type (`HybridBoundaryEvaluatorForBoundary`) that bakes in per-boundary combos/board/pot and implements the trait cleanly.

**Step 2:** Confirm build.
```bash
cargo build --workspace --all-targets
```

**Step 3:** Commit.
```bash
git add crates/tauri-app/src/hybrid_evaluator.rs
git commit -m "feat(hybrid): BoundaryEvaluator override stubs for HybridBoundaryEvaluator"
```

---

## Phase 4 — Coordination: replace K=4 precompute with hybrid dispatch

### Task 4.1: Build `HybridBoundaryAdapter` — per-boundary trait adapter

**Files:**
- Modify: `crates/tauri-app/src/hybrid_evaluator.rs`

**Context:** Near-tree DCFR calls `BoundaryEvaluator::compute_cfvs(...)` with only `opponent_reach` and `player`. We need to pre-bind `(combos, board, boundary_pot, boundary_invested, boundary_id)` per boundary so the trait can dispatch through the cache.

**Step 1:** Add adapter type:
```rust
/// Per-boundary adapter so HybridBoundaryEvaluator can implement the
/// BoundaryEvaluator trait while carrying boundary-specific context.
/// Constructed fresh for each boundary in the near tree during solve.
pub struct HybridBoundaryAdapter<'a> {
    pub evaluator: &'a HybridBoundaryEvaluator,
    pub boundary_id: u64,
    pub combos: Vec<[RsPokerCard; 2]>,
    pub board: Vec<RsPokerCard>,
    pub boundary_pot: f64,
    pub boundary_invested: [f64; 2],
    pub num_oop: usize,
    pub num_ip:  usize,
}

impl<'a> range_solver::game::BoundaryEvaluator for HybridBoundaryAdapter<'a> {
    fn num_continuations(&self) -> usize { 1 }

    fn compute_cfvs(
        &self,
        player: usize,
        _pot: i32,
        _remaining_stack: f64,
        _opponent_reach: &[f32],
        _num_hands: usize,
        _continuation_index: usize,
    ) -> Vec<f32> {
        // Hybrid always wants both sides — forward to both-call, return requested.
        panic!("HybridBoundaryAdapter: use compute_cfvs_both");
    }

    fn compute_cfvs_both(
        &self,
        _pot: i32,
        _remaining_stack: f64,
        oop_reach: &[f32],
        ip_reach:  &[f32],
        _num_oop: usize,
        _num_ip:  usize,
        _continuation_index: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let oop_range: Vec<f64> = oop_reach.iter().map(|&v| v as f64).collect();
        let ip_range:  Vec<f64> = ip_reach.iter().map(|&v| v as f64).collect();
        let cfvs = self.evaluator.compute_cfvs(
            self.boundary_id,
            &self.combos,
            &self.board,
            &oop_range,
            &ip_range,
            self.boundary_pot,
            self.boundary_invested,
        );
        // BoundaryCfvs stores f64; trait returns f32.
        let oop = cfvs.oop_cfvs.into_iter().map(|v| v as f32).collect();
        let ip  = cfvs.ip_cfvs.into_iter().map(|v| v as f32).collect();
        (oop, ip)
    }
}
```

**Step 2:** Add a unit test asserting adapter forwards correctly (use a recording sampler mock).

**Step 3:** Build + test.
```bash
cargo test -p poker-solver-tauri hybrid_evaluator --quiet
```

**Step 4:** Commit.
```bash
git add crates/tauri-app/src/hybrid_evaluator.rs
git commit -m "feat(hybrid): HybridBoundaryAdapter binds per-boundary context"
```

### Task 4.2: Update near-tree solve caller to use `compute_cfvs_both`

**Files:**
- Modify: `crates/range-solver/src/game/interpreter.rs` or wherever DCFR calls `BoundaryEvaluator::compute_cfvs` (grep for it).

**Step 1:** Locate the DCFR call site(s):
```bash
rg "compute_cfvs" crates/range-solver --type rust
```

**Step 2:** At each call site, rewrite to call `compute_cfvs_both` once per boundary and distribute the results to both players' CFV slots. Keep the old `compute_cfvs` trait method available (for any single-side callers that survive) — default impl still works.

**Step 3:** Build workspace.
```bash
cargo build --workspace --all-targets
```

**Step 4:** Run range-solver tests.
```bash
cargo test -p range-solver --quiet
```

**Step 5:** Commit.
```bash
git add crates/range-solver/src/game/interpreter.rs
git commit -m "refactor(range-solver): DCFR calls compute_cfvs_both at boundaries"
```

### Task 4.3: Replace K=4 precompute loop with hybrid dispatch

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs:1927-2000` (delete precompute loop)
- Modify: `crates/tauri-app/src/game_session.rs` wherever `game.boundary_evaluator` is set (grep)

**Step 1:** Failing integration test at `crates/tauri-app/tests/hybrid_integration.rs` (new file):
```rust
//! Integration test: hybrid mode must produce near-identical exploitability
//! to exact mode when depth_limit exceeds tree depth.
#[test]
fn hybrid_equals_exact_when_depth_exceeds_tree() {
    // Load fixture bundle, run a spot in hybrid mode with depth_limit=10
    // (exceeds any reachable tree depth), compare resulting strategies
    // byte-for-byte against exact mode output.
    // Fixture: smallest bundle + smallest possible spot to keep test < 5s.
}
```

Initially marked `#[ignore]` if fixture infrastructure isn't wired; remove `#[ignore]` once real asserts in place.

**Step 2:** Delete lines 1927–2000 (the K=4 precompute loop). Replace with a hybrid-evaluator constructor + attach to the game:

```rust
// REPLACES the K=4 precompute block at game_session.rs:1927-2000
let hybrid_cfg = HybridSolveConfig {
    depth_limit: subgame_depth_limit.unwrap_or(1),
    boundary_refresh_interval: hybrid_refresh_interval.unwrap_or(10),
    samples_per_refresh: hybrid_samples_per_refresh.unwrap_or(100),
};

if n_boundaries > 0 && cbv_ctx.is_some() {
    let ctx = cbv_ctx.as_ref().unwrap();
    let sampler = RolloutLeafEvaluator {
        bias: BiasType::Unbiased,                       // hybrid is always unbiased
        bias_factor: 1.0,
        num_rollouts: hybrid_cfg.samples_per_refresh,
        /* ... populate other fields same as existing code path ... */
    };
    let hybrid_eval = Arc::new(HybridBoundaryEvaluator::new(
        sampler,
        hybrid_cfg.boundary_refresh_interval,
        hybrid_cfg.samples_per_refresh,
    ));
    game.set_hybrid_evaluator(hybrid_eval);

    // Compute initial reaches (iter 0) and prime cache with first refresh.
    // The near-tree solver will drive further refreshes via begin_iteration.
}
```

**Step 3:** Add `set_hybrid_evaluator` method on the game struct (`range-solver::game::PostFlopGame` or wherever `boundary_evaluator` is stored). This replaces the current `game.boundary_evaluator = Some(Box::new(SolveBoundaryEvaluator { ... }))` pattern. The game holds `Arc<HybridBoundaryEvaluator>`; at each boundary visit, near-tree code constructs a short-lived `HybridBoundaryAdapter<'_>` carrying the boundary-specific context.

**Step 4:** Add iteration hook — in the DCFR iteration loop (in `range-solver`), call `hybrid_evaluator.begin_iteration(iter)` at the start of each iter so `should_refresh` sees the current counter. If iter counter isn't already plumbed, grep for the DCFR main loop and add one line at the top.

**Step 5:** Run integration test (unignored).
```bash
cargo test -p poker-solver-tauri hybrid_equals_exact --quiet
```
Expected: pass.

**Step 6:** Run izod repro smoke in a background step — can be sampled during validation phase.
```bash
./target/release/poker-solver-trainer compare-solve \
  --bundle local_data/blueprints/1k_100bb_brdcfr_v2 \
  --snapshot snapshot_0013 \
  --spot "sb:2bb,bb:10bb,sb:22bb,bb:call|Jd9d7d" \
  --iters 50 --verbose
```
Expected smoke: hybrid completes without precompute stall, reports some exploitability (quality checked in Phase 10).

**Step 7:** Commit.
```bash
git add crates/tauri-app/src/game_session.rs crates/tauri-app/tests/hybrid_integration.rs crates/range-solver/src/game/*.rs
git commit -m "feat(hybrid): replace K=4 precompute with hybrid evaluator dispatch"
```

### Task 4.4: Plumb `HybridSolveConfig` params through `game_solve` command

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs:2193-2219` (`game_solve` command signature)

**Step 1:** Extend the command signature:
```rust
#[tauri::command]
pub fn game_solve(
    session_state: tauri::State<'_, GameSessionState>,
    mode: Option<String>,
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
    matrix_snapshot_interval: Option<u32>,
    rollout_bias_factor: Option<f64>,    // kept for validate-rollout only
    rollout_num_samples: Option<u32>,
    rollout_opponent_samples: Option<u32>,
    rollout_enumerate_depth: Option<u8>,
    range_clamp_threshold: Option<f64>,
    subgame_depth_limit: Option<u8>,     // kept as alias for hybrid_depth_limit
    hybrid_depth_limit: Option<u8>,
    hybrid_refresh_interval: Option<u32>,
    hybrid_samples_per_refresh: Option<u32>,
) -> Result<(), String>
```

**Step 2:** In the function body, resolve the effective depth limit (`hybrid_depth_limit.or(subgame_depth_limit).unwrap_or(1)`) and wire into `HybridSolveConfig` construction from Task 4.3.

**Step 3:** Rename mode string acceptance: accept both `"subgame"` (legacy) and `"hybrid"` (new canonical). Map both to the new hybrid path.

**Step 4:** Build.
```bash
cargo build -p poker-solver-tauri
```

**Step 5:** Commit.
```bash
git add crates/tauri-app/src/game_session.rs
git commit -m "feat(hybrid): plumb HybridSolveConfig through game_solve command"
```

---

## Phase 5 — Deletions (K=4 removal)

### Task 5.1: Remove `set_boundary_cfvs_multi` and multi-K storage

**Files:**
- Modify: `crates/range-solver/src/game/interpreter.rs:343-348, 421-441, 447-459`

**Step 1:** Grep for all remaining callers of `set_boundary_cfvs_multi`, `get_boundary_cfvs_multi`, `init_multi_continuation`, `num_continuations` (on PostFlopGame, not on the trait).
```bash
rg "set_boundary_cfvs_multi|get_boundary_cfvs_multi|init_multi_continuation" --type rust
```

**Step 2:** Delete each method and its storage field on the game struct. Verify only hybrid path touches the CFV storage.

**Step 3:** Collapse `boundary_cfvs: Vec<Mutex<Vec<f32>>>` back to `ordinal * 2 + player` indexing (single-K layout). Keep the mutex/vec layout unchanged — just smaller.

**Step 4:** Build workspace.
```bash
cargo build --workspace --all-targets
```

**Step 5:** Commit.
```bash
git add crates/range-solver/src/game/interpreter.rs
git commit -m "chore(range-solver): remove K-continuation storage (superseded by hybrid)"
```

### Task 5.2: Delete `SolveBoundaryEvaluator` rollout precompute path

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs:1365-1491` (`SolveBoundaryEvaluator`)

**Step 1:** Since `HybridBoundaryEvaluator` is now the active path, decide fate of `SolveBoundaryEvaluator`:
- **Option A:** Delete it entirely — it's only used for the old subgame dispatch, which no longer exists.
- **Option B:** Keep for pure-exact-mode dispatch if exact mode still uses this struct (check).

**Step 2:** Grep references:
```bash
rg "SolveBoundaryEvaluator" --type rust
```
If only referenced from the now-deleted precompute loop, delete the struct. If exact mode still wires it, leave alone and add a comment explaining why.

**Step 3:** Build + test.
```bash
cargo build --workspace --all-targets
cargo test --workspace --quiet
```

**Step 4:** Commit.
```bash
git add crates/tauri-app/src/game_session.rs
git commit -m "chore(hybrid): remove unused SolveBoundaryEvaluator paths"
```

### Task 5.3: Update `compare-solve` dump for single-K output

**Files:**
- Modify: `crates/trainer/src/compare_solve.rs:650-730` (boundary CFV dump)

**Step 1:** The current dump iterates K=4 biases; collapse to single-K. New dump prints boundary_id, OOP CFV stats (min/max/mean/stddev), IP CFV stats.

**Step 2:** Build.
```bash
cargo build -p poker-solver-trainer --release
```

**Step 3:** Smoke test dump.
```bash
./target/release/poker-solver-trainer compare-solve \
  --bundle local_data/blueprints/1k_100bb_brdcfr_v2 \
  --snapshot snapshot_0013 \
  --spot "sb:2bb,bb:10bb,sb:22bb,bb:call|Jd9d7d" \
  --iters 10 --dump-boundary-cfvs 2>&1 | head -40
```

**Step 4:** Commit.
```bash
git add crates/trainer/src/compare_solve.rs
git commit -m "refactor(compare-solve): single-K boundary CFV dump"
```

---

## Phase 6 — Metrics

### Task 6.1: Define `HybridRefreshMetrics` + `HybridSolveMetrics`

**Files:**
- Modify: `crates/tauri-app/src/hybrid_evaluator.rs`

**Step 1:** Define the structs with serde support:
```rust
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HybridRefreshMetrics {
    pub iteration: u32,
    pub refresh_wall_ms: f32,
    pub boundaries_refreshed: u32,
    pub total_samples_drawn: u64,
    pub mean_cfv_variance: f32,
    pub mean_cfv_drift:    f32,
    pub samples_per_sec:   f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HybridSolveMetrics {
    pub refresh_count: u32,
    pub total_refresh_ms: f32,
    pub total_dcfr_ms: f32,
    pub refresh_overhead_pct: f32,
    pub final_mean_cfv_variance: f32,
    pub cfv_drift_trajectory: Vec<f32>,
}
```

**Step 2:** Add fields to `HybridBoundaryEvaluator`:
```rust
pub struct HybridBoundaryEvaluator {
    // ... existing ...
    refresh_metrics: Mutex<Vec<HybridRefreshMetrics>>,
    prev_cfvs_snapshot: RwLock<HashMap<u64, BoundaryCfvs>>,
}
```

**Step 3:** Build.
```bash
cargo build -p poker-solver-tauri
```

**Step 4:** Commit.
```bash
git add crates/tauri-app/src/hybrid_evaluator.rs
git commit -m "feat(hybrid): metric types + storage"
```

### Task 6.2: Collect metrics inside `compute_cfvs` on refresh

**Files:**
- Modify: `crates/tauri-app/src/hybrid_evaluator.rs`

**Step 1:** On refresh path in `compute_cfvs` (Task 2.2 slow-path), record:
- Refresh start `Instant`; elapsed → `refresh_wall_ms`
- Compare new CFVs against `prev_cfvs_snapshot` entry to compute `mean_cfv_drift` (L2 on concat(oop, ip) vectors, averaged per boundary)
- Variance within the sample batch (track from sampler if feasible; v1: mean |diff of first-half vs second-half samples|)
- Append to `refresh_metrics`

**Step 2:** Add a method `drain_metrics(&self) -> HybridSolveMetrics` that aggregates the refresh metrics into the per-solve summary.

**Step 3:** Test for metric aggregation (small unit test with 3 synthetic refreshes).

**Step 4:** Build + test.
```bash
cargo test -p poker-solver-tauri hybrid_evaluator --quiet
```

**Step 5:** Commit.
```bash
git add crates/tauri-app/src/hybrid_evaluator.rs
git commit -m "feat(hybrid): collect per-refresh metrics"
```

### Task 6.3: Wire metrics into progress event channel

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs:41-46` (event types) and ~2960 (emit site)
- Modify: `crates/tauri-app/src/game_session.rs` (emit after each solve iter)

**Step 1:** Extend `SubgameProgressEvent` or add a new event:
```rust
#[derive(Clone, Serialize)]
pub struct HybridProgressEvent {
    pub iteration: u32,
    pub max_iterations: u32,
    pub exploitability_mbb: Option<f32>,
    pub refresh: Option<HybridRefreshMetrics>,    // Some on refresh iters only
}
```

**Step 2:** In the solve loop, after each DCFR iteration, emit this event via the Tauri channel. On final iter, also include `HybridSolveMetrics` (either as a separate finalize event or as the terminating payload).

**Step 3:** Build.
```bash
cargo build -p poker-solver-tauri
```

**Step 4:** Commit.
```bash
git add crates/tauri-app/src/exploration.rs crates/tauri-app/src/game_session.rs
git commit -m "feat(hybrid): stream HybridProgressEvent with refresh metrics"
```

---

## Phase 7 — CLI adapters

### Task 7.1: Rename + add `compare-solve` flags

**Files:**
- Modify: `crates/trainer/src/main.rs:346-384` (CompareSolve enum variant)
- Modify: `crates/trainer/src/compare_solve.rs:830-840` (`run` fn)

**Step 1:** Add flags, preserve legacy alias:
```rust
#[command(name = "compare-solve")]
CompareSolve {
    // ... existing ...
    #[arg(long, alias = "subgame-depth-limit")]
    hybrid_depth_limit: Option<u8>,

    #[arg(long, default_value_t = 10)]
    hybrid_refresh_interval: u32,

    #[arg(long, default_value_t = 100)]
    hybrid_samples_per_refresh: u32,

    #[arg(long)]
    metrics_json: Option<PathBuf>,
}
```

**Step 2:** Thread into `compare_solve::run` signature and dispatch into the solve call.

**Step 3:** When `--metrics-json` is set, after solve completes, serialize `HybridSolveMetrics` to JSON and write to the path. Use `serde_json::to_writer_pretty`.

**Step 4:** Build.
```bash
cargo build -p poker-solver-trainer --release
```

**Step 5:** Smoke test.
```bash
./target/release/poker-solver-trainer compare-solve --help | grep -E "hybrid|metrics"
```

**Step 6:** Commit.
```bash
git add crates/trainer/src/main.rs crates/trainer/src/compare_solve.rs
git commit -m "feat(compare-solve): hybrid_* flags + --metrics-json dump"
```

### Task 7.2: Add `--verbose` aggregate metrics print

**Files:**
- Modify: `crates/trainer/src/compare_solve.rs`

**Step 1:** When `--verbose`, after solve, print `HybridSolveMetrics` fields one per line (refresh_count, total_refresh_ms, refresh_overhead_pct, final_mean_cfv_variance, cfv_drift_trajectory summary: first, last, min, max).

**Step 2:** Smoke test.
```bash
./target/release/poker-solver-trainer compare-solve \
  --bundle local_data/blueprints/1k_100bb_brdcfr_v2 \
  --snapshot snapshot_0013 \
  --spot "sb:2bb,bb:10bb,sb:22bb,bb:call|Jd9d7d" \
  --iters 20 --verbose 2>&1 | tail -30
```
Expected: metrics section visible.

**Step 3:** Commit.
```bash
git add crates/trainer/src/compare_solve.rs
git commit -m "feat(compare-solve): verbose prints hybrid solve metrics"
```

---

## Phase 8 — Tauri mode string canonicalization

### Task 8.1: Rename mode strings + legacy acceptance

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs:205-209` (`solve_for`)
- Modify: `crates/tauri-app/src/game_session.rs:1709` (match site)

**Step 1:** Accept both canonical `"hybrid"` and legacy `"subgame"`:
```rust
pub fn solve_for(&self, mode: &Option<String>) -> &Arc<SolveState> {
    match mode.as_deref() {
        Some("exact") => &self.exact_solve,
        Some("hybrid") | Some("subgame") | None | Some(_) => &self.subgame_solve,
    }
}
```
Rename `SolveState` field `subgame_solve` → `hybrid_solve` (or leave field name — internal only; check grep for uses).

**Step 2:** Build + test.
```bash
cargo build --workspace
cargo test --workspace --quiet
```

**Step 3:** Commit.
```bash
git add crates/tauri-app/src/game_session.rs
git commit -m "refactor(tauri): canonicalize mode string to 'hybrid' with 'subgame' alias"
```

### Task 8.2: Devserver mirror check

**Files:**
- Modify: `crates/devserver/src/main.rs` (the dev server that mirrors Tauri commands)

**Step 1:** Grep the devserver for any command signatures that must gain the new `hybrid_*` params.
```bash
rg "game_solve|subgame_depth_limit" crates/devserver --type rust
```

**Step 2:** Update devserver's `game_solve` handler signature to match Tauri's. The dev server accepts POST JSON and forwards — make sure the new params deserialize.

**Step 3:** Build.
```bash
cargo build -p poker-solver-devserver
```

**Step 4:** Smoke test.
```bash
cargo run -p poker-solver-devserver &
sleep 1
curl -X POST http://localhost:3001/api/is_bundle_loaded -H 'Content-Type: application/json' -d '{}'
kill %1
```
Expected: 200-ish response, no panic.

**Step 5:** Commit.
```bash
git add crates/devserver/src/main.rs
git commit -m "feat(devserver): mirror hybrid_* params in game_solve"
```

---

## Phase 9 — Frontend adapters

### Task 9.1: Add Hybrid Solver settings group

**Files:**
- Modify: `frontend/src/Settings.tsx:320-343` (subgame_depth_limit input)

**Step 1:** Wrap the existing "Subgame Depth Limit" input with a grouped container titled "Hybrid Solver" and rename the label to "Depth limit". Add two new number inputs underneath: "Refresh interval", "Samples per refresh". Wire each to `setConfig({ hybrid_depth_limit })` / `hybrid_refresh_interval` / `hybrid_samples_per_refresh` using the same `useGlobalConfig()` pattern.

**Step 2:** Add three preset buttons (Fast / Balanced / Accurate) that populate all three at once. Use simple button group styling from the existing design system.

**Step 3:** In `useGlobalConfig`, add the three new keys with defaults `depth_limit=1`, `refresh_interval=10`, `samples_per_refresh=100`. Read legacy key `subgame_depth_limit` on boot and migrate it to `hybrid_depth_limit` if set (one-release compat).

**Step 4:** Pass the three new params in the `game_solve` invoke call (in the solve button's onClick handler).

**Step 5:** Run dev server + frontend.
```bash
cargo run -p poker-solver-devserver &
cd frontend && npm run dev &
# open http://localhost:5173 in a browser — verify the three inputs render and the solve button sends the new params (check network tab)
```

**Step 6:** Commit.
```bash
git add frontend/src/Settings.tsx frontend/src/useGlobalConfig.ts  # or wherever config keys live
git commit -m "feat(ui): Hybrid Solver settings group with presets"
```

### Task 9.2: Add Solver Telemetry panel

**Files:**
- Create: `frontend/src/SolverTelemetry.tsx`
- Modify: parent component that owns the solve button (grep for `game_solve`)

**Step 1:** Build collapsible panel (default collapsed). Subscribe to Tauri event (and devserver polling mirror) for the new `HybridProgressEvent`. Display:
- Current iteration / max
- Latest exploitability (if present)
- Refresh overhead % (computed from latest `HybridRefreshMetrics`)
- Sparkline: `cfv_drift` trajectory
- Sparkline: `mean_cfv_variance` trajectory
- End-of-solve: full `HybridSolveMetrics` summary

Use minimal charting — a simple `<svg>` polyline with ~50-point window is enough; no new npm deps.

**Step 2:** Event wiring: if Tauri, use `listen()`; if browser/devserver, fall back to polling `/api/get_solve_progress` (add this endpoint on devserver too if missing).

**Step 3:** Smoke test in browser: click solve, see telemetry update.

**Step 4:** Commit.
```bash
git add frontend/src/SolverTelemetry.tsx frontend/src/*.tsx
git commit -m "feat(ui): Solver Telemetry panel (collapsible, with sparklines)"
```

---

## Phase 10 — Validation

### Task 10.1: Run izod repro and verify success criteria

**Files:** none (measurement only)

**Step 1:** Run at depth=0 (baseline — should improve over broken-Subgame 11,354 mbb).
```bash
./target/release/poker-solver-trainer compare-solve \
  --bundle local_data/blueprints/1k_100bb_brdcfr_v2 \
  --snapshot snapshot_0013 \
  --spot "sb:2bb,bb:10bb,sb:22bb,bb:call|Jd9d7d" \
  --iters 200 --verbose \
  --hybrid-depth-limit 0 \
  --hybrid-refresh-interval 10 \
  --hybrid-samples-per-refresh 100 2>&1 | tee /tmp/hybrid-d0.log
```
Success: hybrid exploitability < 11,000 mbb/hand (strictly better than broken depth=0).

**Step 2:** Run at depth=1 (main target).
```bash
./target/release/poker-solver-trainer compare-solve \
  --bundle local_data/blueprints/1k_100bb_brdcfr_v2 \
  --snapshot snapshot_0013 \
  --spot "sb:2bb,bb:10bb,sb:22bb,bb:call|Jd9d7d" \
  --iters 200 --verbose \
  --hybrid-depth-limit 1 \
  --hybrid-refresh-interval 10 \
  --hybrid-samples-per-refresh 100 2>&1 | tee /tmp/hybrid-d1.log
```
Success criteria:
- Exploitability ≤ 539 mbb (Exact 38.6 + 500 margin). Stretch: ≤ 139.
- Wall ≤ 116 s (2× Exact). Stretch: ≤ 58 s.
- First iter begins within 2 s (check log line stamps).

**Step 3:** Run equivalence check at `--hybrid-depth-limit 10` (exceeds tree depth).
```bash
./target/release/poker-solver-trainer compare-solve \
  ... --hybrid-depth-limit 10 --iters 200 2>&1 | tee /tmp/hybrid-equiv.log
```
Success: hybrid exploitability == exact exploitability (byte-level strategy check).

**Step 4:** Report results in the bean.
```bash
beans update poker_solver_rust-mx1j --body-append "## Phase 10 measurements
- depth=0 hybrid: {actual} mbb (target <11,000)
- depth=1 hybrid: {actual} mbb (target <539, stretch <139), wall {actual}s
- depth=10 equivalence: {pass/fail}"
```

### Task 10.2: Full test suite under 60 s

**Files:** none

**Step 1:**
```bash
time cargo test --workspace --quiet
```
Success: all pass, total < 60 s (per CLAUDE.md requirement).

**Step 2:** If regression, fix before closing.

### Task 10.3: Documentation updates

**Files:**
- Modify: `docs/architecture.md` — document hybrid solver architecture, replace Subgame section
- Modify: `docs/training.md` — update CLI flag docs for `compare-solve`
- Modify: `docs/explorer.md` — document new Settings group + Telemetry panel

**Step 1:** For each doc, find the Subgame / K=4 section and rewrite with the hybrid design. Reference `docs/plans/2026-04-19-hybrid-mccfr-solver-design.md` for depth.

**Step 2:** Commit.
```bash
git add docs/architecture.md docs/training.md docs/explorer.md
git commit -m "docs: hybrid MCCFR solver replaces Subgame (arch, training, explorer)"
```

### Task 10.4: Close bean `mx1j`, note `izod` status

**Files:** none (bean updates)

**Step 1:** If Phase 10.1 success criteria met:
```bash
beans update poker_solver_rust-mx1j -s completed \
  --body-append "## Summary of Changes
Replaced Modicum K=4 rollout precompute with live-sampled hybrid MCCFR at
boundaries. Three config knobs (depth, refresh interval, samples per refresh).
Telemetry streamed to UI. Success criteria met: see Phase 10 log. izod bean
(critical) now resolves — see bean for exploitability numbers."
```

**Step 2:** Update `izod`:
```bash
beans update poker_solver_rust-izod -s completed \
  --body-append "## Resolved by hybrid MCCFR implementation (mx1j)
Phase 10 measurements confirm hybrid at depth=1 achieves exploitability
within spec of exact. See docs/plans/2026-04-19-hybrid-mccfr-solver.md Phase 10."
```

**Step 3:** Close `jpwu` (parallelize K=4 precompute) as scrapped — no longer needed since K=4 precompute is deleted.
```bash
beans update poker_solver_rust-jpwu -s scrapped \
  --body-append "## Reasons for Scrapping
K=4 rollout precompute path was deleted in hybrid MCCFR refactor (mx1j).
Parallelization is moot — the inner loop no longer exists."
```

**Step 4:** Final commit (bean files).
```bash
git add .beans/
git commit -m "chore(beans): close mx1j/izod as resolved, scrap jpwu"
```

---

## Appendix: File change summary

**New files:**
- `crates/tauri-app/src/hybrid_evaluator.rs`
- `crates/tauri-app/tests/hybrid_integration.rs`
- `frontend/src/SolverTelemetry.tsx`

**Modified files (major):**
- `crates/range-solver/src/game/mod.rs` — extend trait
- `crates/range-solver/src/game/interpreter.rs` — remove multi-K storage, call `compute_cfvs_both`
- `crates/tauri-app/src/game_session.rs` — replace precompute loop, thread config, rename mode strings
- `crates/tauri-app/src/postflop.rs` — add `sample_boundary_cfvs`, `BoundaryCfvs`
- `crates/tauri-app/src/exploration.rs` — new event type
- `crates/tauri-app/src/lib.rs` — export new module
- `crates/trainer/src/main.rs` — new CLI flags
- `crates/trainer/src/compare_solve.rs` — flags wiring, dump refactor, metrics JSON
- `crates/devserver/src/main.rs` — mirror hybrid params
- `frontend/src/Settings.tsx` — Hybrid Solver group + presets
- `frontend/src/useGlobalConfig.ts` — new config keys
- `docs/architecture.md`, `docs/training.md`, `docs/explorer.md`

**Deleted (via Phase 5):**
- K-continuation precompute loop body at `game_session.rs:1927-2000`
- `set_boundary_cfvs_multi`, `get_boundary_cfvs_multi`, `init_multi_continuation` on PostFlopGame
- K=4 loop in `compare_solve.rs` boundary CFV dump

**Bean side-effects:**
- `mx1j` → completed
- `izod` → completed
- `jpwu` → scrapped
