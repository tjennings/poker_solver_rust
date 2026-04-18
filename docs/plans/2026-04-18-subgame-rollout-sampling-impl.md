# Subgame Rollout — Depth-Gated Sampling (Revised) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Drop per-evaluator-call latency from ~8.2s → 80-200ms (50-100× speedup) on the 1176-combo reference scenario by switching `rollout_inner` from exhaustive expectimax to Modicum-style depth-gated MCCFR sampling.

**Architecture:** Enumerate all actions at the top 0-1 decision levels (high entropy, cheap), sample one action per biased strategy at depth ≥ 2, keep chance-node sampling. Variance at the boundary is absorbed by DCFR's across-iteration averaging — same pattern Libratus/Pluribus/Modicum ship.

**Tech Stack:** Rust, `rand::Rng::random_range`, existing `bias_strategy` output, no new deps.

**Supersedes:** `docs/plans/2026-04-17-subgame-rollout-perf-impl.md` (commits 2-6 dropped based on ml-researcher findings — per-hand alloc fixes yield ~5-10× and become irrelevant once sampling cuts terminal count ~1000×).

**Reference:**
- Research output saved in this session transcript; key citation: Modicum (Brown, Sandholm, Amos, NeurIPS 2018, §5).
- Hot loop: `crates/core/src/blueprint_v2/continuation.rs:165-270` (`rollout_inner`).
- Bench: `cargo run --release -p poker-solver-trainer -- bench-rollout --bundle <path> --duration-secs 10`.
- Baseline measurement to beat: `8243 ms/call, 0.12 calls/s, 44.9M hands/s` on `Ks7h2c` bundle `200_100bb_sapcfr/snap`.

---

## Commit 2 — `perf(continuation): depth-gated MCCFR sampling at decision nodes`

### Task 2.1: Thread a decision-depth counter through `rollout_inner`

**Files:**
- Modify: `crates/core/src/blueprint_v2/continuation.rs:143-270`

**Step 1: Extend `rollout_from_boundary` and `rollout_inner` signatures**

Add parameter `decision_depth: u8` to `rollout_inner`. Entry point `rollout_from_boundary` calls with `0`. The Decision arm increments before recursing (both in the enumerate path for depth<threshold and the sample path for depth≥threshold). Chance arm passes through unchanged — the counter counts *decision* depth, not total tree depth.

**Step 2: Add threshold constant**

At top of file:
```rust
/// Decision-depth at which we stop enumerating children and start sampling.
/// Depth 0 and 1 enumerate (Modicum: first levels enumerate; variance accumulates fast further down).
const SAMPLE_AFTER_DECISION_DEPTH: u8 = 2;
```

**Step 3: Implement sample-branch in the Decision arm**

Current (`continuation.rs:230-243`):
```rust
let mut ev = 0.0;
for (i, &child) in children.iter().enumerate() {
    let (child_pot, child_invested) = apply_action(...);
    let child_ev = rollout_inner(..., child, ...);
    ev += f64::from(biased[i]) * child_ev;
}
ev
```

Becomes:
```rust
let ev = if decision_depth < SAMPLE_AFTER_DECISION_DEPTH {
    // Enumerate (existing behavior).
    let mut sum = 0.0;
    for (i, &child) in children.iter().enumerate() {
        let (child_pot, child_invested) = apply_action(&actions[i], pot, invested, actor, ctx.starting_stack);
        let child_ev = rollout_inner(..., child, rng, Some(buckets), child_pot, child_invested, decision_depth + 1);
        sum += f64::from(biased[i]) * child_ev;
    }
    sum
} else {
    // Sample one action from the biased distribution.
    let chosen = sample_action_index(rng, &biased);
    let (child_pot, child_invested) = apply_action(&actions[chosen], pot, invested, actor, ctx.starting_stack);
    rollout_inner(..., children[chosen], rng, Some(buckets), child_pot, child_invested, decision_depth + 1)
};
ev
```

**Step 4: Add `sample_action_index` helper**

```rust
/// Sample an action index from a (biased) probability distribution.
/// Assumes probs sum to ~1.0; falls through to the last index if the
/// cumulative threshold isn't crossed (handles fp drift).
fn sample_action_index(rng: &mut impl Rng, probs: &[f32]) -> usize {
    let r: f32 = rng.random_range(0.0..1.0);
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return i;
        }
    }
    probs.len() - 1
}
```

**Step 5: Update MCCFR and test callers**

Grep for call sites of `rollout_from_boundary` and `rollout_inner`. Update any internal test helper that calls `rollout_inner` directly to pass a starting depth (test-convenience: pass 0).

---

### Task 2.2: Bump chance `num_rollouts` adaptively

**Files:**
- Modify: `crates/core/src/blueprint_v2/continuation.rs` — Chance arm `:244-268`

Previously at 3 samples per chance node. With decision sampling on, variance compounds — bump first chance transitions.

**Step 1: Thread `chance_depth: u8` parameter the same way as `decision_depth`**

Or reuse `decision_depth` if total-depth semantics are equivalent here; prefer a separate `chance_depth` for clarity.

**Step 2: Chance arm samples more on the first two transitions**

```rust
GameNode::Chance { next_street: _, child } => {
    let chance_boost = if chance_depth < 2 { 3 } else { 1 };
    let n = (ctx.num_rollouts * chance_boost).min(deck_count);
    ...
}
```

Or parameterize via a new field on `RolloutContext` if the research gotcha matters enough (`chance_early_multiplier: u32`, default 3). Prefer the simpler constant unless a test case drives otherwise.

---

### Task 2.3: Verify RNG independence per combo per call

**Files:**
- Read: `crates/tauri-app/src/postflop.rs:444` — already seeds `SmallRng::seed_from_u64(i as u64)` per combo inside `rollout_chip_values_with_state`.

This already satisfies gotcha #2 from research (independent RNG per combo per call). No change required — just verify the seed is reset on each evaluate call (it is, because `rng` is declared inside the `.map()`).

**Add assertion** as a debug comment or test: seeds should NOT be shared across combos. Grep for any place `SmallRng` is constructed with a module-level seed — if found, fix.

---

### Task 2.4: Gate exhaustive-mode via `RolloutContext`

**Purpose:** keep the old enumerate-everywhere behavior selectable for commit 3's validation. Not permanent — delete the gate once commit 3 ships and confirms equivalence.

**Files:**
- Modify: `crates/core/src/blueprint_v2/continuation.rs` — `RolloutContext` struct at `:113-125`

**Step 1: Add field**

```rust
/// When true, enumerate at every decision depth (old exhaustive behavior).
/// When false (default for production), sample at depth ≥ SAMPLE_AFTER_DECISION_DEPTH.
/// Used during validation to compare sampled vs exhaustive CFVs.
pub exhaustive: bool,
```

**Step 2: Decision arm respects the flag**

```rust
if ctx.exhaustive || decision_depth < SAMPLE_AFTER_DECISION_DEPTH {
    // enumerate branch
} else {
    // sample branch
}
```

Existing call sites (MCCFR tests, tauri eval) initialize to `false`. Tests for commit 3 construct with `exhaustive: true` on one side.

---

### Task 2.5: Tests

**Files:**
- Modify: `crates/core/src/blueprint_v2/continuation.rs` test module

**Step 1: `sample_action_index` correctness**

```rust
#[test]
fn sample_action_index_respects_probabilities() {
    let probs = [0.0, 0.8, 0.2];
    let mut rng = SmallRng::seed_from_u64(42);
    let mut counts = [0usize; 3];
    for _ in 0..10_000 {
        counts[sample_action_index(&mut rng, &probs)] += 1;
    }
    assert_eq!(counts[0], 0);
    assert!((counts[1] as f64 / 10_000.0 - 0.8).abs() < 0.02);
    assert!((counts[2] as f64 / 10_000.0 - 0.2).abs() < 0.02);
}
```

**Step 2: `rollout_inner` exhaustive mode matches sampled mode in expectation**

Construct a tiny tree (3 decision nodes, 2-action each, 1 chance → shallow enough for both). Run sampled mode 10_000 times and average; run exhaustive once. Assert averages agree within 1% (tolerance driven by the chosen sample count).

**Step 3: Run test gate**

```bash
cd /Users/coreco/code/poker_solver_rust.rollout-perf
cargo test -p poker-solver-core -p poker-solver-tauri -p poker-solver-trainer --quiet
```

Expected: 957 pass, 1 known pre-existing failure `traverse_updates_strategy_sums`.

---

### Task 2.6: Measure

**Step 1: Rebuild and run the bench**

```bash
cd /Users/coreco/code/poker_solver_rust.rollout-perf
cargo build --release -p poker-solver-trainer
./target/release/poker-solver-trainer bench-rollout \
  --bundle /Users/coreco/code/poker_solver_rust/local_data/blueprints/200_100bb_sapcfr/snap \
  --duration-secs 10
```

**Step 2: Record the new `ms/call` and `calls/s`**

Expected target: 80-200 ms/call. If way off in either direction (< 10 ms/call or > 500 ms/call), pause and investigate before committing.

**Step 3: Commit**

```
perf(continuation): depth-gated MCCFR sampling at decision nodes

Replaces exhaustive expectimax enumeration at deep decision nodes with
single-action sampling from the biased blueprint distribution. Follows
Modicum (Brown/Sandholm/Amos NeurIPS 2018). Enumeration retained at
decision depth 0-1 where entropy is highest and cost is low.

Baseline (from ms/call measurement in PR body):  8243 ms/call
After this commit:                               <new> ms/call
Speedup:                                         <X>×
```

---

## Commit 3 — `test(continuation): exploitability validation — sampled vs exhaustive`

### Task 3.1: Add a validation harness

**Files:**
- Add: `crates/trainer/src/bin/validate_rollout.rs` or extend `bench_rollout.rs`

**Purpose:** compare CFVs produced by sampled vs exhaustive rollouts on the canonical bundle/scenario. Compute per-combo L1 and L∞ norms of the difference. Flag if > 1-2 mbb/hand (in pot-fraction units, ~0.01-0.02 of the pot).

**Step 1: Add a `validate-rollout` trainer subcommand**

Takes same bundle + scenario as `bench-rollout`. Invokes `rollout_chip_values_with_state` twice:
1. With `RolloutContext { exhaustive: true, ... }`
2. With `RolloutContext { exhaustive: false, ... }`

Prints:
```
OOP traverser:
  exhaustive combos with nonzero weight: N
  sampled combos with nonzero weight: N
  max_abs_diff: X pot-fraction (Y mbb/hand)
  mean_abs_diff: X pot-fraction (Y mbb/hand)
  L2 diff: X
IP traverser: (same block)
```

**Step 2: Run it, capture the numbers**

```bash
./target/release/poker-solver-trainer validate-rollout \
  --bundle /Users/coreco/code/poker_solver_rust/local_data/blueprints/200_100bb_sapcfr/snap \
  --duration-secs 10
```

**Step 3: Document the result**

Paste output into the PR body under a "Validation" section. Pass criterion: `max_abs_diff < 0.02` (2 mbb/hand at 100 bb stacks).

### Task 3.2: If validation passes, delete the `exhaustive` gate

Once the numbers confirm sampled ≈ exhaustive within tolerance, drop the `exhaustive: bool` field from `RolloutContext`, delete the branching inside `rollout_inner`'s Decision arm, update the validation binary to stop requiring a flag. Commit 3 becomes:

```
test(continuation): validate sampled rollouts match exhaustive within 2 mbb/hand
```

If validation *fails* (> 2 mbb/hand): pause, investigate (increase outer sample count, revisit depth threshold, bump chance rollouts).

---

## Finalize

### Task F1: Full verification (hex:verification-before-completion)

```bash
cargo test -p poker-solver-core -p poker-solver-tauri -p poker-solver-trainer --quiet
cargo clippy -p poker-solver-core -p poker-solver-tauri -p poker-solver-trainer -- -D warnings
```

The `-D warnings` clippy gate will fail on pre-existing warnings — scope the gate to files touched by this PR if needed, or accept "no new warnings" instead of zero warnings.

### Task F2: Open PR (hex:finishing-a-development-branch)

Push and open PR `perf/subgame-rollout`. Body includes the attribution table:

| Commit | ms/call | calls/s | hands/s | Notes |
|--------|---------|---------|---------|-------|
| baseline (1b) | 8243 | 0.12 | 44.9M | exhaustive, pre-pivot |
| sampling pivot (2) | <fill> | <fill> | <fill> | hybrid depth-gated |
| validation (3) | <fill> | <fill> | <fill> | exhaustive flag removed |

Plus a "Validation" block with the max/mean/L2 diffs from commit 3's binary output.

---

## Execution

**Plan complete and saved to `docs/plans/2026-04-18-subgame-rollout-sampling-impl.md`.**

Continuing the current **subagent-driven** stream (1 implementer + 3-stage review per commit). Two commits left in the PR (2 and 3).
