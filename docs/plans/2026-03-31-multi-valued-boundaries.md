# Multi-Valued Boundary States Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace single-valued, dynamically-refreshed boundary evaluators with K=4 fixed continuation strategies at each boundary node, following Brown & Sandholm NeurIPS 2018.

**Architecture:** The range solver's boundary terminal nodes become opponent decision nodes with 4 children (unbiased + fold/call/raise biased). All 4×boundaries×players×hands CFVs are precomputed once via rollouts before solving begins. CFR runs on the augmented fixed game — no boundary cache flushes, no feedback loop.

**Tech Stack:** Rust, range-solver crate (`PostFlopGame`, `BoundaryEvaluator`), `RolloutLeafEvaluator` with `BiasType` variants.

**Design doc:** `docs/plans/2026-03-31-multi-valued-boundaries-design.md`

---

### Task 1: Extend BoundaryEvaluator trait with continuation_index

**Files:**
- Modify: `crates/range-solver/src/game/mod.rs:18-36`

**Step 1: Write the failing test**

Add in `crates/range-solver/src/solver.rs` (in the test module):

```rust
#[test]
fn boundary_evaluator_supports_continuation_index() {
    use crate::game::BoundaryEvaluator;

    struct MultiBoundary;
    impl BoundaryEvaluator for MultiBoundary {
        fn num_continuations(&self) -> usize { 4 }
        fn compute_cfvs(
            &self, player: usize, _pot: i32, _remaining: f64,
            _opp_reach: &[f32], _num_hands: usize,
            continuation_index: usize,
        ) -> Vec<f32> {
            // Return different values for each continuation
            vec![(continuation_index as f32 + 1.0) * (player as f32 + 1.0)]
        }
    }

    let eval = MultiBoundary;
    assert_eq!(eval.num_continuations(), 4);
    let v0 = eval.compute_cfvs(0, 100, 50.0, &[1.0], 1, 0);
    let v1 = eval.compute_cfvs(0, 100, 50.0, &[1.0], 1, 1);
    assert_ne!(v0[0], v1[0]); // different continuations give different values
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p range-solver -- boundary_evaluator_supports_continuation`
Expected: compilation error — wrong number of parameters

**Step 3: Implement**

Modify the `BoundaryEvaluator` trait in `crates/range-solver/src/game/mod.rs`:

```rust
pub trait BoundaryEvaluator: Send + Sync {
    /// Number of continuation strategies at each boundary. Default: 1.
    fn num_continuations(&self) -> usize { 1 }

    /// Compute per-hand CFVs at a depth boundary for a specific continuation.
    ///
    /// `continuation_index` selects which of the K continuation strategies
    /// to evaluate (0 = unbiased/blueprint, 1..K-1 = biased variants).
    fn compute_cfvs(
        &self,
        player: usize,
        pot: i32,
        remaining_stack: f64,
        opponent_reach: &[f32],
        num_hands: usize,
        continuation_index: usize,
    ) -> Vec<f32>;
}
```

Update the call site in `evaluation.rs` (line ~92) to pass `continuation_index: 0` (single continuation, legacy behavior):

```rust
let cfvs = evaluator.compute_cfvs(player, pot, remaining, &opp_reach_ref, num_hands, 0);
```

Update all `BoundaryEvaluator` implementations to accept the new parameter:
- `SolveBoundaryEvaluator` in `crates/tauri-app/src/game_session.rs` — add `_continuation_index: usize` parameter
- `NetBoundaryEvaluator` in `crates/cfvnet/src/eval/compare_turn.rs` — add `_continuation_index: usize` parameter
- Any test implementations in `crates/range-solver/src/solver.rs`

**Step 4: Run tests**

Run: `cargo test -p range-solver && cargo build`
Expected: all pass, full workspace compiles

**Step 5: Commit**

```bash
git add crates/range-solver/src/game/mod.rs crates/range-solver/src/game/evaluation.rs crates/tauri-app/src/game_session.rs crates/cfvnet/src/eval/compare_turn.rs
git commit -m "feat: extend BoundaryEvaluator trait with continuation_index and num_continuations"
```

---

### Task 2: Replace single boundary terminal with K-action opponent choice node

This is the core range-solver change. When `num_continuations() > 1`, boundary terminals become opponent decision nodes with K children.

**Files:**
- Modify: `crates/range-solver/src/game/evaluation.rs:61-153` (depth boundary evaluation path)
- Modify: `crates/range-solver/src/game/interpreter.rs` (boundary node allocation — `boundary_cfvs` sizing)

**Step 1: Write the failing test**

Add in `crates/range-solver/src/solver.rs`:

```rust
#[test]
fn multi_continuation_boundary_produces_different_strategies() {
    // Build a small depth-limited game and attach a multi-continuation evaluator.
    // Verify the solver converges and the strategy differs from single-continuation.
    // (Detailed implementation depends on existing test helpers — use build_river_game
    // or similar pattern, adapted for a flop game with depth_limit=Some(0).)
}
```

Note: the exact test setup depends on the existing test infrastructure. The implementer should read existing tests in `solver.rs` and `query.rs` for patterns to build a depth-limited game.

**Step 2: Implement**

The key change is in `evaluation.rs`. Currently the depth boundary case (line 61-153) stores and reads a single set of CFVs per (boundary, player). With K continuations, it needs to:

1. **At tree construction time** (`interpreter.rs`): Allocate `K × boundary_count × 2` CFV slots instead of `boundary_count × 2`. Add `num_continuations` field to `PostFlopGame`.

2. **At evaluation time** (`evaluation.rs`): When K > 1, the boundary node acts as an opponent decision node. Instead of directly returning cached CFVs, compute the strategy-weighted combination across K continuations (regret matching over the K continuation actions) OR — better — restructure so the solver's recursive traversal handles the K actions naturally.

**Approach A (recommended): Internal opponent choice in evaluate_internal.**

At the depth boundary, instead of writing `bcfv * payoff_scale * cfreach` directly to `result`, model K opponent actions:

```rust
// For each opponent continuation k=0..K-1:
//   Compute result_k using bcfvs[k]
// Then combine using opponent's current strategy (regret matching) over K actions
```

This requires the boundary node to have its own regret/strategy storage for the K continuation actions. The simplest way: allocate small buffers (K entries per boundary × per player) on the `PostFlopGame` and update them during the traversal.

**However**, this is complex. A simpler alternative:

**Approach B: Precompute a single blended CFV using fixed equal weights.**

This loses the adaptive opponent property but gives stable values. Each boundary's CFV = average of the 4 continuation CFVs. Simple and fast, but doesn't capture the full multi-valued-state benefit.

**Approach C (recommended for correctness with minimal refactor): Pre-set boundary CFVs to the "worst-case" continuation.**

For each hero hand at each boundary, set the boundary CFV to `min over K continuations` (the value under the opponent's best response). This is conservative — P1 must be robust to the worst continuation. It's a one-line change to the precomputation step and requires no range-solver internal changes.

The implementer should decide between A and C based on complexity. **Approach C is simpler and captures most of the benefit:**

In the precomputation step (Task 3), instead of storing 4 separate CFV sets, compute:

```rust
for each boundary, player, hand:
    cfv = min(cfv_unbiased[hand], cfv_fold_biased[hand], cfv_call_biased[hand], cfv_raise_biased[hand])
```

Then set this single min-CFV via the existing `set_boundary_cfvs()`. The range solver needs NO internal changes.

**The implementer MUST discuss with the lead which approach to take before implementing.** Approach A is theoretically correct but requires significant range-solver refactoring. Approach C is 90% of the benefit with 10% of the work.

**Step 3: Commit**

Commit message depends on chosen approach.

---

### Task 3: Precompute 4 rollouts per boundary at solver construction

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs` (the `SolveBoundaryEvaluator` construction and the solve thread)

**Step 1: Write the failing test**

```rust
#[test]
fn solve_precomputes_four_continuation_cfvs() {
    // Create a mock evaluator that tracks how many times compute_cfvs is called
    // with each continuation_index. Verify 4 different indices are called per boundary.
}
```

**Step 2: Implement**

In `game_session.rs`, in the solve thread (around line 1773-1840), after the `SolveBoundaryEvaluator` is constructed and before the solve loop:

1. For each boundary (0..num_boundaries):
   a. For each player (0, 1):
      b. For each continuation (0..4):
         - Call the boundary evaluator with the blueprint's reach probabilities and continuation_index
         - Store the result

2. If using Approach C from Task 2, compute the per-hand minimum across the 4 continuations and call `game.set_boundary_cfvs(ordinal, player, min_cfvs)`.

3. If using Approach A, store all 4 sets and wire them into the range solver's multi-continuation storage.

The `SolveBoundaryEvaluator` needs to support `continuation_index` by running the rollout with different `BiasType`:
- continuation 0: `BiasType::Unbiased`
- continuation 1: `BiasType::Fold`
- continuation 2: `BiasType::Call`
- continuation 3: `BiasType::Raise`

Update `SolveBoundaryEvaluator::compute_cfvs` to use `continuation_index`:

```rust
fn compute_cfvs(&self, player: usize, pot: i32, remaining_stack: f64,
                opponent_reach: &[f32], num_hands: usize, continuation_index: usize) -> Vec<f32> {
    // SPR=0: use equity (same for all continuations)
    if remaining_stack <= 0.0 {
        return self.compute_equity_cfvs(player, opponent_reach, num_hands);
    }

    // SPR>0: rollout with appropriate bias
    let bias = match continuation_index {
        0 => BiasType::Unbiased,
        1 => BiasType::Fold,
        2 => BiasType::Call,
        3 => BiasType::Raise,
        _ => BiasType::Unbiased,
    };

    self.compute_rollout_cfvs(player, pot, remaining_stack, opponent_reach, num_hands, bias)
}
```

**Step 3: Commit**

```bash
git commit -m "feat: precompute 4 continuation rollouts per boundary at solver construction"
```

---

### Task 4: Remove flush_boundary_caches from solve loop

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs:1733-1736`

**Step 1: Implement**

Remove or comment out the `flush_boundary_caches` call:

```rust
// BEFORE:
if t > 0 && t.is_multiple_of(eval_interval) {
    game.flush_boundary_caches();
}

// AFTER:
// Boundary values are fixed at construction (multi-valued states).
// No mid-solve refresh — this prevents the destabilizing feedback loop.
```

Also remove `flush_boundary_caches` from the postflop.rs solve loop if present.

**Step 2: Run integration test**

Run a solve and verify it converges stably without boundary refreshes.

**Step 3: Commit**

```bash
git commit -m "fix: remove flush_boundary_caches from solve loop for stable convergence"
```

---

### Task 5: Integration test and manual validation

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs` (test section)

**Step 1: Write integration test**

```rust
#[test]
fn multi_continuation_solve_converges() {
    // Build a session, navigate to a flop spot, run solve with 1000 iterations.
    // Verify:
    // 1. Exploitability decreases from initial to final
    // 2. Strategy is not uniform (differentiated across hands)
    // 3. Strong hands have higher bet frequency than weak hands
}
```

**Step 2: Manual validation**

Build release and test in the Tauri UI:
- Navigate to the same QsTs7h flop spot from the earlier screenshot
- Click solve
- Verify the strategy matrix shows differentiation matching the blueprint's pattern
- Verify solve completes without the strategy collapsing to uniform

**Step 3: Commit**

```bash
git commit -m "test: add multi-continuation boundary solve integration test"
```

---

### Implementation Note for the Implementer

**CRITICAL: Before implementing Task 2, discuss with the lead whether to use Approach A (full opponent choice nodes in range-solver) or Approach C (min-over-continuations, no range-solver changes).** Approach C is recommended for the first iteration — it requires only changes in Task 3 (precomputation) and Task 4 (remove flush), with zero range-solver modifications. Approach A is the full paper implementation but requires significant range-solver refactoring.

The recommended path for fastest results:
1. Task 1 (trait change) — straightforward
2. Skip Task 2 initially — use Approach C
3. Task 3 (precompute 4 rollouts, take min) — the core work
4. Task 4 (remove flush) — one line
5. Task 5 (validate)

If Approach C produces good results, Approach A can be added later as an optimization.
