# Skip All-Zero Showdown Outcomes Allocation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate host-side allocation + H2D upload of the all-zero showdown outcomes buffer in GPU turn datagen, fixing the 127 GB host OOM. River path remains byte-identical.

**Architecture:** `SubgameSpec.showdown_outcomes_p0/p1` becomes `Option<Vec<f32>>`. `None` means "this batch has no showdowns to compute" — `prepare_batch` skips host concat and H2D upload, `run_iterations` passes `num_showdowns=0` to the kernel (loop body becomes zero-iteration no-op). River sets `Some(...)`; turn sets `None`.

**Tech Stack:** Rust, cudarc, custom CUDA kernel (`cfr_solve`).

**Design doc:** [`docs/plans/2026-04-20-skip-zero-showdown-buffer-design.md`](2026-04-20-skip-zero-showdown-buffer-design.md)

**Bean:** `poker_solver_rust-vb8r`

---

## Prerequisites

1. You are running in a dedicated git worktree on a feature branch. If not, stop and ask the coordinator to create one.
2. The Python cfvnet fixes (commits `d91894f`, `c1f5fd5`, `6c0e8a1`, `6812547`) are already on main — those are unrelated.
3. Before starting any task, read the design doc in full.
4. Confirm the current state compiles: `cargo check -p gpu-range-solver -p cfvnet --features gpu-turn-datagen` must succeed before any change.
5. **Kernel inspection (required before Task 3):** read `crates/gpu-range-solver/src/kernels.rs` and find the showdown loop. Confirm the loop is shaped like `for (int i = 0; i < num_showdowns; i++) { ... }` or equivalent, so that `num_showdowns == 0` naturally produces zero iterations. If it does NOT handle 0 gracefully (e.g. unconditional reads, off-by-one), the plan's "skip via num_showdowns=0" strategy must change — stop and flag to the coordinator.

---

## Task 1: Wrap the river path in `Some(...)` (no behavior change)

**Goal:** Change `SubgameSpec` field type to `Option<Vec<f32>>`, keep all call sites producing `Some(...)`. This is a pure type refactor — existing tests must pass byte-identically.

**Files:**
- Modify: `crates/gpu-range-solver/src/batch.rs:15-26` (struct definition)
- Modify: `crates/gpu-range-solver/src/batch.rs:57-64` (`SubgameSpec::from_game`)
- Modify: `crates/gpu-range-solver/src/batch.rs:321-336` (`prepare_batch` — unwrap with `.as_deref().unwrap_or(&[])`)
- Modify: `crates/cfvnet/src/datagen/domain/pipeline.rs:1018-1024` (`build_turn_subgame_spec`)

### Step 1.1: Change the struct

Replace `crates/gpu-range-solver/src/batch.rs:15-26` with:

```rust
#[derive(Clone)]
pub struct SubgameSpec {
    /// Per-player initial weights, length `num_hands` each.
    pub initial_weights: [Vec<f32>; 2],
    /// Pre-scaled showdown outcomes for player 0 traversal: `[num_showdowns * H * H]`.
    /// `None` means this spec has no showdowns — used by turn datagen where leaf
    /// injection from BoundaryNet supplies all CFVs at boundary nodes. When `None`,
    /// `prepare_batch` skips host concat and H2D upload, and the kernel sees
    /// `num_showdowns=0` so its showdown loop is a no-op.
    pub showdown_outcomes_p0: Option<Vec<f32>>,
    /// Pre-scaled showdown outcomes for player 1 traversal: `[num_showdowns * H * H]`.
    /// See `showdown_outcomes_p0` for `None` semantics.
    pub showdown_outcomes_p1: Option<Vec<f32>>,
    /// Per-game fold payoffs for P0 traversal: `[num_folds]`.
    pub fold_payoffs_p0: Vec<f32>,
    /// Per-game fold payoffs for P1 traversal: `[num_folds]`.
    pub fold_payoffs_p1: Vec<f32>,
}
```

### Step 1.2: Update river call site

In `crates/gpu-range-solver/src/batch.rs:57-64` (`SubgameSpec::from_game`), wrap the two outcome fields:

```rust
SubgameSpec {
    initial_weights,
    showdown_outcomes_p0: Some(mtd.showdown_outcomes_p0),
    showdown_outcomes_p1: Some(mtd.showdown_outcomes_p1),
    fold_payoffs_p0: mtd.fold_payoffs_p0,
    fold_payoffs_p1: mtd.fold_payoffs_p1,
}
```

### Step 1.3: Update turn call site (still `Some` — behavior unchanged this task)

In `crates/cfvnet/src/datagen/domain/pipeline.rs:1018-1024` (`build_turn_subgame_spec`):

```rust
gpu_range_solver::SubgameSpec {
    initial_weights: [w_oop, w_ip],
    showdown_outcomes_p0: Some(zero_outcomes.clone()),
    showdown_outcomes_p1: Some(zero_outcomes),
    fold_payoffs_p0,
    fold_payoffs_p1,
}
```

### Step 1.4: Update `prepare_batch` consumers

In `crates/gpu-range-solver/src/batch.rs:326-333`, change:

```rust
for (b, spec) in specs.iter().enumerate() {
    let base = b * sd_per_batch;
    let src_len = spec.showdown_outcomes_p0.len().min(sd_per_batch);
    batched_sd_p0[base..base + src_len]
        .copy_from_slice(&spec.showdown_outcomes_p0[..src_len]);
    let src_len = spec.showdown_outcomes_p1.len().min(sd_per_batch);
    batched_sd_p1[base..base + src_len]
        .copy_from_slice(&spec.showdown_outcomes_p1[..src_len]);
}
```

to:

```rust
for (b, spec) in specs.iter().enumerate() {
    let base = b * sd_per_batch;
    if let Some(sd) = spec.showdown_outcomes_p0.as_deref() {
        let src_len = sd.len().min(sd_per_batch);
        batched_sd_p0[base..base + src_len].copy_from_slice(&sd[..src_len]);
    }
    if let Some(sd) = spec.showdown_outcomes_p1.as_deref() {
        let src_len = sd.len().min(sd_per_batch);
        batched_sd_p1[base..base + src_len].copy_from_slice(&sd[..src_len]);
    }
}
```

### Step 1.5: Verify it compiles and tests pass

Run: `cargo check -p gpu-range-solver -p cfvnet --features gpu-turn-datagen`
Expected: clean compile.

Run: `cargo test -p gpu-range-solver 2>&1 | tail -20`
Expected: all existing tests pass. Behavior is identical — we're still allocating+uploading zeros for turn.

### Step 1.6: Commit

```bash
git add crates/gpu-range-solver/src/batch.rs crates/cfvnet/src/datagen/domain/pipeline.rs
git commit -m "$(cat <<'EOF'
refactor(gpu-range-solver): wrap SubgameSpec showdown outcomes in Option

Prepares for skipping the all-zero allocation in turn datagen.
No behavioral change in this commit — all call sites still produce
Some(...). prepare_batch handles None as a zero-length source.

Bean: poker_solver_rust-vb8r

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `active_num_showdowns` tracking to `GpuBatchSolver`

**Goal:** When a batch has no showdown outcomes, the kernel must see `num_showdowns=0`. Today `run_iterations` reads `self.num_showdowns` (topology-wide). Add a per-batch field that `prepare_batch` sets.

**Files:**
- Modify: `crates/gpu-range-solver/src/batch.rs` (struct definition + `prepare_batch` + `run_iterations`)

### Step 2.1: Locate the `GpuBatchSolver` struct

Read `crates/gpu-range-solver/src/batch.rs` lines ~73-150 (or wherever the `GpuBatchSolver` struct is defined). Identify where `num_showdowns: usize` is stored and where the `active_*` fields are.

### Step 2.2: Add `active_num_showdowns` field

Add a new field next to the other `active_*` fields (near `active_batch_size: usize`):

```rust
/// Number of showdowns active in the current batch. Equal to `self.num_showdowns`
/// when the batch has real outcomes, or `0` when all specs set `showdown_outcomes_*`
/// to `None` (turn datagen with leaf injection). Set by `prepare_batch`, consumed
/// by `run_iterations`.
active_num_showdowns: usize,
```

Initialize it to `0` wherever `active_batch_size: 0` is initialized in `GpuBatchSolver::new` (search for `active_batch_size: 0`).

### Step 2.3: Verify compile

Run: `cargo check -p gpu-range-solver`
Expected: clean compile (field is defined but not yet used beyond initialization).

### Step 2.4: Commit

```bash
git add crates/gpu-range-solver/src/batch.rs
git commit -m "$(cat <<'EOF'
refactor(gpu-range-solver): add active_num_showdowns to GpuBatchSolver

Plumbing for per-batch variable showdown count; used in next commit
to let prepare_batch signal "no showdowns" to run_iterations.

Bean: poker_solver_rust-vb8r

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `prepare_batch` branches on presence; `run_iterations` uses `active_num_showdowns`

**Goal:** When all specs have `None` outcomes, skip the host concat + H2D upload, set `active_num_showdowns=0`, use dummy device buffers. Kernel will see `num_showdowns=0` and loop zero times. Mixed batch panics in debug.

**Files:**
- Modify: `crates/gpu-range-solver/src/batch.rs:321-362` (`prepare_batch`)
- Modify: `crates/gpu-range-solver/src/batch.rs:408-409` and `:380-387` (`run_iterations`)

### Step 3.1: Replace the showdown-outcomes block in `prepare_batch`

Replace the block currently at `batch.rs:321-336` (from `// Build batched showdown outcomes` through the two `upload_or_dummy_f32` lines) with:

```rust
// Build batched showdown outcomes: [B * num_showdowns * H * H] for each player.
// If all specs have `None` outcomes (turn datagen with leaf injection), skip the
// allocation + upload entirely and set active_num_showdowns=0 so the kernel's
// showdown loop is a no-op. Mixed Some/None in one batch is a bug.
let any_some_p0 = specs.iter().any(|s| s.showdown_outcomes_p0.is_some());
let any_some_p1 = specs.iter().any(|s| s.showdown_outcomes_p1.is_some());
let all_some_p0 = specs.iter().all(|s| s.showdown_outcomes_p0.is_some());
let all_some_p1 = specs.iter().all(|s| s.showdown_outcomes_p1.is_some());
debug_assert_eq!(
    any_some_p0, all_some_p0,
    "SubgameSpec.showdown_outcomes_p0 must be uniformly Some or None across a batch"
);
debug_assert_eq!(
    any_some_p1, all_some_p1,
    "SubgameSpec.showdown_outcomes_p1 must be uniformly Some or None across a batch"
);
debug_assert_eq!(
    any_some_p0, any_some_p1,
    "SubgameSpec showdown_outcomes_p0 and _p1 must have the same Some/None state"
);

let (d_showdown_p0, d_showdown_p1, active_num_showdowns) = if all_some_p0 {
    let hh = h * h;
    let sd_per_batch = self.num_showdowns * hh;
    let mut batched_sd_p0 = vec![0.0f32; batch_size * sd_per_batch];
    let mut batched_sd_p1 = vec![0.0f32; batch_size * sd_per_batch];
    for (b, spec) in specs.iter().enumerate() {
        let base = b * sd_per_batch;
        if let Some(sd) = spec.showdown_outcomes_p0.as_deref() {
            let src_len = sd.len().min(sd_per_batch);
            batched_sd_p0[base..base + src_len].copy_from_slice(&sd[..src_len]);
        }
        if let Some(sd) = spec.showdown_outcomes_p1.as_deref() {
            let src_len = sd.len().min(sd_per_batch);
            batched_sd_p1[base..base + src_len].copy_from_slice(&sd[..src_len]);
        }
    }
    (
        upload_or_dummy_f32(&self.stream, &batched_sd_p0)?,
        upload_or_dummy_f32(&self.stream, &batched_sd_p1)?,
        self.num_showdowns,
    )
} else {
    // No showdowns to compute. Dummy device buffers keep the kernel-arg plumbing
    // happy; active_num_showdowns=0 makes the kernel's showdown loop zero-iteration.
    (
        upload_or_dummy_f32(&self.stream, &[])?,
        upload_or_dummy_f32(&self.stream, &[])?,
        0,
    )
};
```

### Step 3.2: Store `active_num_showdowns`

At the end of `prepare_batch` (around line 354-361, where `self.active_state = Some(state);` etc. are), add:

```rust
self.active_num_showdowns = active_num_showdowns;
```

### Step 3.3: Use `active_num_showdowns` in `run_iterations`

Change `crates/gpu-range-solver/src/batch.rs:409`:

```rust
let num_showdowns_i32 = self.num_showdowns as i32;
```

to:

```rust
let num_showdowns_i32 = self.active_num_showdowns as i32;
```

### Step 3.4: Verify compile

Run: `cargo check -p gpu-range-solver -p cfvnet --features gpu-turn-datagen`
Expected: clean compile.

### Step 3.5: Run existing tests

Run: `cargo test -p gpu-range-solver 2>&1 | tail -20`
Expected: all pass (river path still produces `Some`, turn path still produces `Some` for now — no behavior change yet).

### Step 3.6: Commit

```bash
git add crates/gpu-range-solver/src/batch.rs
git commit -m "$(cat <<'EOF'
feat(gpu-range-solver): prepare_batch skips showdown alloc when None

When all specs in a batch have showdown_outcomes_{p0,p1} = None,
prepare_batch no longer allocates the batched_sd_p0/p1 host vectors
nor the per-player device buffer. Dummy empty GPU buffers are used
as kernel args, and active_num_showdowns=0 makes the kernel's
showdown loop a zero-iteration no-op. Mixed Some/None in one batch
debug-panics — not a supported state.

River callers continue to pass Some(...); no behavioral change
to the river path. Turn path still passes Some(...) here — it
switches to None in a later commit.

Bean: poker_solver_rust-vb8r

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add unit test — `None` outcomes produce correct CFVs via leaf injection

**Goal:** Prove that a batch with `None` showdown outcomes and leaf-injected CFVs returns the injected values. This is the test that proves the fix is correct.

**Files:**
- Modify: existing test module in `crates/gpu-range-solver/src/batch.rs` (or `solver.rs` if that's where batch tests live). **Read the file first to find the test module — do not create a new file.**

### Step 4.1: Locate existing batch tests

Run: `grep -n "#\[test\]\|#\[cfg(test)\]" crates/gpu-range-solver/src/batch.rs crates/gpu-range-solver/src/solver.rs crates/gpu-range-solver/tests/*.rs 2>/dev/null`

Find an existing test that exercises `GpuBatchSolver::new` + `prepare_batch` + `run_iterations` end-to-end. Model the new test on its structure — same `#[cfg]` gating, same CUDA context setup, same topology builders.

### Step 4.2: Write the failing test

Append this test to the same test module. Read the surrounding imports and adjust names as needed (the `TreeTopology::trivial_*` helper names are indicative — use whatever the existing tests use for a minimal topology).

```rust
#[test]
#[cfg(feature = "cuda")] // match whatever feature gate the existing batch tests use
fn batch_with_none_showdowns_returns_leaf_injection_values() {
    // A tiny topology with 2 leaf (boundary) nodes and 0 real showdowns.
    // Batch of 2 specs, each with None showdown outcomes. Leaf injection
    // supplies known CFVs. Assert the solver's output matches the injected
    // values after one iteration.
    let topo = /* trivial topology: root → 2 leaves */;
    let term = /* matching TerminalData with 2 showdown_nodes (leaf placeholders),
                   0 fold_nodes */;
    let num_hands = 4;
    let num_iters = 1u32;

    let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA required for this test");
    let mut solver = GpuBatchSolver::new(ctx, &topo, &term, num_hands, /*max_batch=*/ 2)
        .expect("GpuBatchSolver::new");

    // Configure leaf injection with 2 leaf nodes at known depths.
    let leaf_node_ids = vec![1i32, 2i32]; // match your topology
    let leaf_depths = vec![1i32, 1i32];
    solver.set_leaf_injection(&leaf_node_ids, &leaf_depths)
        .expect("set_leaf_injection");

    // Two specs, both with None outcomes.
    let spec = SubgameSpec {
        initial_weights: [vec![1.0; num_hands], vec![1.0; num_hands]],
        showdown_outcomes_p0: None,
        showdown_outcomes_p1: None,
        fold_payoffs_p0: vec![],
        fold_payoffs_p1: vec![],
    };
    let specs = vec![spec.clone(), spec];
    solver.prepare_batch(&specs).expect("prepare_batch");

    // Inject known CFVs: spec 0 → [1.0, 2.0, 3.0, 4.0] / [-1, -2, -3, -4] at each leaf
    //                   spec 1 → [10, 20, 30, 40] / [-10, -20, -30, -40]
    let batch_size = 2;
    let num_leaves = 2;
    let mut leaf_p0 = vec![0.0f32; batch_size * num_leaves * num_hands];
    let mut leaf_p1 = vec![0.0f32; batch_size * num_leaves * num_hands];
    for b in 0..batch_size {
        for li in 0..num_leaves {
            for h in 0..num_hands {
                let v = ((b + 1) as f32) * (h + 1) as f32;
                leaf_p0[b * num_leaves * num_hands + li * num_hands + h] = v;
                leaf_p1[b * num_leaves * num_hands + li * num_hands + h] = -v;
            }
        }
    }
    solver.update_leaf_cfvs(&leaf_p0, &leaf_p1).expect("update_leaf_cfvs");

    solver.run_iterations(0, num_iters).expect("run_iterations");

    // Download reach (or cfv, whichever is the appropriate output for this topology).
    // Assert the values propagated from the leaf injection — exact expected values
    // depend on the trivial topology chosen. Start by asserting:
    //   - No NaN
    //   - Non-zero output
    //   - Expected sign per player
    let reach = solver.download_reach().expect("download_reach");
    assert!(reach.iter().all(|v| v.is_finite()), "reach contains NaN/Inf");
    assert!(reach.iter().any(|v| *v != 0.0), "all reach values are zero");
    // Tighter numeric assertions to be added once the trivial topology behavior
    // is established by reading the existing happy-path test.
}
```

**Implementation note:** the *exact* trivial topology and numeric assertion need to be modeled on whatever minimal existing test is in the file. Do not invent a topology from scratch — copy the structure from the most similar existing test and substitute `None` for the showdown outcomes.

### Step 4.3: Run the test — it should pass

This is a rare case where the test should already pass (Task 3 made `None` work correctly). Run:

```
cargo test -p gpu-range-solver --features cuda batch_with_none_showdowns 2>&1 | tail -20
```

Expected: PASS.

If it fails, the error will reveal the gap. Common causes: `active_num_showdowns` not actually being propagated, or leaf-injection not being the sole source of CFVs (topology still has real showdown nodes). Fix the test setup (the implementation should be correct from Task 3).

### Step 4.4: Commit

```bash
git add crates/gpu-range-solver/src/batch.rs
git commit -m "$(cat <<'EOF'
test(gpu-range-solver): cover batch with None showdown outcomes

Proves that prepare_batch + run_iterations with None outcomes +
leaf injection produces the injected CFVs. Protects the
turn-datagen OOM fix against regression.

Bean: poker_solver_rust-vb8r

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add unit test — mixed batch panics in debug

**Goal:** Verify the debug assertion catches mixed `Some`/`None` in one batch.

### Step 5.1: Write the test

In the same test module:

```rust
#[test]
#[should_panic(expected = "uniformly Some or None")]
#[cfg(feature = "cuda")]
fn prepare_batch_panics_on_mixed_showdown_presence() {
    let topo = /* same trivial topology as above */;
    let term = /* matching */;
    let num_hands = 4;

    let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA required");
    let mut solver = GpuBatchSolver::new(ctx, &topo, &term, num_hands, 2).unwrap();

    let spec_none = SubgameSpec {
        initial_weights: [vec![1.0; num_hands], vec![1.0; num_hands]],
        showdown_outcomes_p0: None,
        showdown_outcomes_p1: None,
        fold_payoffs_p0: vec![],
        fold_payoffs_p1: vec![],
    };
    let spec_some = SubgameSpec {
        initial_weights: [vec![1.0; num_hands], vec![1.0; num_hands]],
        showdown_outcomes_p0: Some(vec![0.0; solver.num_showdowns * num_hands * num_hands]),
        showdown_outcomes_p1: Some(vec![0.0; solver.num_showdowns * num_hands * num_hands]),
        fold_payoffs_p0: vec![],
        fold_payoffs_p1: vec![],
    };

    // Mixed batch must panic in debug.
    solver.prepare_batch(&[spec_none, spec_some]).unwrap();
}
```

*If `num_showdowns` is not a public field, inline the trivial topology's known value.*

### Step 5.2: Run the test

Run: `cargo test -p gpu-range-solver --features cuda prepare_batch_panics_on_mixed 2>&1 | tail -20`
Expected: PASS (debug mode runs `debug_assert_eq!` which panics).

### Step 5.3: Commit

```bash
git add crates/gpu-range-solver/src/batch.rs
git commit -m "$(cat <<'EOF'
test(gpu-range-solver): mixed Some/None showdown batch panics in debug

Enforces the invariant that prepare_batch callers must supply
uniform Some/None across a batch. Mixed states aren't a supported
mode — in the canonical-topology batched-datagen model all specs
in a batch share the same terminal structure.

Bean: poker_solver_rust-vb8r

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Flip the turn path to `None`

**Goal:** This is the actual OOM fix. Remove the zero-outcomes allocation from `build_turn_subgame_spec`.

**Files:**
- Modify: `crates/cfvnet/src/datagen/domain/pipeline.rs:1013-1024`

### Step 6.1: Remove the allocation and set `None`

Replace the block at `pipeline.rs:1013-1024`:

```rust
// Showdown outcomes: zeros. Turn boundary nodes get real CFVs from
// leaf injection in the kernel.
let num_showdowns = topo.showdown_nodes.len();
let zero_outcomes = vec![0.0_f32; num_showdowns * num_hands * num_hands];

gpu_range_solver::SubgameSpec {
    initial_weights: [w_oop, w_ip],
    showdown_outcomes_p0: Some(zero_outcomes.clone()),
    showdown_outcomes_p1: Some(zero_outcomes),
    fold_payoffs_p0,
    fold_payoffs_p1,
}
```

with:

```rust
// Showdown outcomes: None. Turn boundary nodes get real CFVs from leaf
// injection (GpuBoundaryEvaluator → update_leaf_cfvs), so the kernel's
// showdown loop runs zero iterations and the all-zero outcomes buffer
// that used to be allocated here is unnecessary.
gpu_range_solver::SubgameSpec {
    initial_weights: [w_oop, w_ip],
    showdown_outcomes_p0: None,
    showdown_outcomes_p1: None,
    fold_payoffs_p0,
    fold_payoffs_p1,
}
```

Remove the unused `num_showdowns` local if present.

### Step 6.2: Verify compile

Run: `cargo check -p cfvnet --features gpu-turn-datagen 2>&1 | tail -10`
Expected: clean compile. If there's a warning about unused `topo` or similar, adjust.

### Step 6.3: Run tests

Run: `cargo test -p gpu-range-solver -p cfvnet --features gpu-turn-datagen 2>&1 | tail -20`
Expected: all pass. The new `batch_with_none_showdowns...` test passes, river tests unchanged.

### Step 6.4: Commit

```bash
git add crates/cfvnet/src/datagen/domain/pipeline.rs
git commit -m "$(cat <<'EOF'
fix(cfvnet): skip all-zero showdown buffer in turn datagen (OOM fix)

build_turn_subgame_spec no longer allocates the
[num_showdowns * 1326 * 1326 * 4B] zero buffer per player per game.
Leaf injection from BoundaryNet already provides the CFVs at turn
boundary nodes via an independent GPU buffer, so the showdown
outcomes were pure waste. At batch_size=256 this eliminates
~230 GB of host RAM allocation per batch.

Closes the OOM that was killing `cfvnet generate` on
turn_gpu_datagen.yaml even at gpu_batch_size=32.

Bean: poker_solver_rust-vb8r

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Full test suite + lint

**Goal:** Verify the whole project still works.

### Step 7.1: Full build

Run: `cargo build -p gpu-range-solver -p cfvnet --features gpu-turn-datagen --release 2>&1 | tail -10`
Expected: clean build.

### Step 7.2: Full test suite (excluding pre-existing broken mp_tui tests from handoff)

Run: `cargo test --workspace 2>&1 | tail -30`
Expected: all pass except the two pre-existing failures documented in `docs/plans/2026-04-19-session-handoff.md` (`mp_tui_scenarios::tests::resolve_empty_returns_root`, `tests::mp_6player_tui_section_parses`). Report those as known-broken, not a regression.

### Step 7.3: Clippy

Run: `cargo clippy -p gpu-range-solver -p cfvnet --features gpu-turn-datagen -- -D warnings 2>&1 | tail -20`
Expected: clean (allow pre-existing warnings in files you didn't touch, but your changes must not introduce new warnings).

### Step 7.4: Test suite timing

Per `CLAUDE.md`, the test suite must complete in <1 min. Run and time:

```bash
time cargo test --workspace 2>&1 | tail -3
```

Expected: real time <60s. If not, stop and flag to the coordinator.

---

## Task 8: Manual validation — production turn datagen at batch=256

**Goal:** The real acceptance criterion. Run the actual datagen command and confirm it no longer OOMs.

### Step 8.1: Confirm config

Ensure `sample_configurations/turn_gpu_datagen.yaml` has `gpu_batch_size: 256` (the handoff doc noted the default is 256 via `unwrap_or`, so removing the explicit `gpu_batch_size: 32` is enough if that line was added earlier — or set it explicitly to 256 for clarity). Leave the file in a clean state you can commit.

### Step 8.2: Launch datagen with RSS monitoring

In one terminal:

```bash
cargo run -p cfvnet --release --features gpu-turn-datagen -- generate \
  -c sample_configurations/turn_gpu_datagen.yaml \
  -o local_data/cfvnet/turn/gpu_v1
```

In another terminal:

```bash
while true; do ps -o rss= -C cfvnet | awk '{s+=$1} END {print strftime("%T"), s/1024/1024, "GB"}'; sleep 2; done
```

### Step 8.3: Record results

- Peak RSS: _____ GB (goal: <64 GB)
- First file completed: ____ (goal: yes)
- Sample records written are well-formed: spot-check 5 records with a Python one-liner using the `cfvnet.data._decode_single_record` helper added in commit `d91894f`.

### Step 8.4: If validation passes

Continue to Task 9.

### Step 8.5: If validation fails

- If it OOMs at `batch=256` but works at `batch=64`: the allocation fix helped but not enough. File findings in bean `ipg7` (already filed for investigating the 4× discrepancy) and STOP. Do not mark `vb8r` complete; coordinator decides next step.
- If it OOMs at any batch: something is still wrong with the fix. Investigate using `massif` (`valgrind --tool=massif`) or `heaptrack` to find the dominant allocator; report back.
- If it crashes with a CUDA error rather than OOM: the kernel is not handling `num_showdowns=0` correctly. Re-read `kernels.rs` showdown loop; if it has an unconditional dereference, the "skip kernel launch entirely" approach from the design's risk section is now required. Report back to coordinator.

---

## Task 9: Close the bean

### Step 9.1: Update bean with summary

```bash
beans update --json poker_solver_rust-vb8r -s completed --body-append "$(cat <<'EOF'


## Summary of Changes

Skipped the all-zero showdown outcomes buffer in turn datagen. `SubgameSpec.showdown_outcomes_p0/p1` is now `Option<Vec<f32>>`; `None` signals "no showdowns", which `prepare_batch` handles by skipping host concat + H2D upload, and `run_iterations` handles by passing `num_showdowns=0` to the kernel.

**Commits:**
- refactor: wrap SubgameSpec fields in Option (river still Some, turn still Some, no behavior change)
- refactor: add active_num_showdowns field
- feat: prepare_batch branches on None; dummy buffers + num_showdowns=0 when absent
- test: None outcomes + leaf injection produces correct CFVs
- test: mixed Some/None batch panics in debug
- fix: turn path now sets None (the actual OOM fix)

**Validation:**
- Full test suite passes in <60s.
- Production turn datagen at `gpu_batch_size=256` stays under <RSS GB> RSS, completes the first output file, records are well-formed.
EOF
)"
```

Replace `<RSS GB>` with the actual measurement from Task 8.3.

### Step 9.2: Commit the bean update

```bash
git add .beans/poker_solver_rust-vb8r*.md
git commit -m "chore: complete bean vb8r (GPU turn datagen OOM fix)"
```

### Step 9.3: Report to coordinator

Final report should include:
- All commit SHAs
- Peak RSS observed during Task 8
- Whether follow-up bean `ipg7` (batch-size plumbing / num_showdowns investigation) is still needed
- Any items discovered that warrant follow-up beans

---

## Execution guidance

- Work in a dedicated worktree.
- Commit after every task.
- Do NOT squash commits — the granularity documents the reasoning.
- If a test fails unexpectedly, stop and report — do not paper over with `#[ignore]` or scope changes.
- If the kernel inspection in "Prerequisites" reveals that `num_showdowns=0` is not handled gracefully, stop and flag to the coordinator BEFORE writing any code. The plan needs adjustment.
- The 127 GB OOM number is the baseline. Record actual peak RSS from Task 8.3 in the bean summary.
