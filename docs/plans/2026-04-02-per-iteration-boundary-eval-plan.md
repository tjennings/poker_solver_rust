# Per-Iteration Boundary Re-evaluation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace the pre-compute-once GPU pipeline with a lockstep ring buffer where K games cycle through GPU boundary eval + DCFR iteration, re-evaluating boundaries every iteration as ranges evolve.

**Architecture:** An orchestrator runs a lockstep loop: batch GPU eval of all K active games' boundaries → parallel solve_step via thread::scope → graduate finished games to writer, inject new ones from deal buffer. No rayon anywhere.

**Tech Stack:** Rust, burn (wgpu backend), range-solver (`solve_step`, `flush_boundary_caches`), `std::thread::scope`

**Design doc:** `docs/plans/2026-04-02-per-iteration-boundary-eval-design.md`

---

## Context for Implementer

**Key existing functions you'll use:**
- `solve_step(game, iteration)` in `crates/range-solver/src/solver.rs:181` — runs one DCFR iteration
- `flush_boundary_caches()` in `crates/range-solver/src/game/interpreter.rs:368` — clears boundary_reach + boundary_cfvs
- `set_boundary_cfvs(ordinal, player, cfvs)` in `interpreter.rs:343` — sets pre-computed boundary values
- `boundary_reach(ordinal, player)` in `interpreter.rs:379` — reads opponent reaches at boundaries
- `compute_exploitability(game)` in `solver.rs` — expensive, only called once at graduation
- `build_game_inputs()` in `turn_generate.rs` — builds GPU input tensor for all boundaries of a game
- `decode_boundary_cfvs()` in `turn_generate.rs` — decodes GPU output into per-boundary CFVs
- `evaluate_batch()` in `turn_generate.rs` — combines build+forward+scatter (but we'll replace the scatter since it calls `set_boundary_cfvs` which is what we want)
- `solve_and_extract()` in `turn_generate.rs:485` — runs full solve + extracts results (we'll split this)

**Key insight:** `boundary_reach` currently caches on first visit only (line 97 of evaluation.rs: `if guard.is_empty()`). For per-iteration re-eval, it must overwrite every visit. However, `flush_boundary_caches()` already exists to clear both reaches and CFVs — we call it between iterations so `solve_step` lazily repopulates `boundary_reach` during traversal.

**The lockstep loop (pseudo-code):**
```
active_games: Vec<(Situation, PostFlopGame, u32)>  // (sit, game, iteration)

loop:
    // 1. GPU: build inputs using boundary_reach from last iteration
    //    (for iteration 0: use initial_weights)
    for game in &active_games:
        build_game_inputs(game)  // reads boundary_reach or initial_weights
    model.forward(combined_tensor)
    for game in &mut active_games:
        set_boundary_cfvs(game)  // scatter GPU output

    // 2. Solve: one iteration per game, parallel via thread::scope
    thread::scope(|s| {
        for chunk in active_games.chunks_mut(chunk_size) {
            s.spawn(|| {
                set_force_sequential(true);
                for (_, game, iter) in chunk {
                    solve_step(game, *iter);
                    *iter += 1;
                }
            });
        }
    });

    // 3. Graduate + inject
    for slot in &mut active_games:
        if slot.iteration >= solver_iterations:
            extract results → send to writer
            replace with new game from deal buffer
    if all graduated and deal buffer empty:
        break
```

---

### Task 1: Fix boundary_reach to overwrite every visit

**Files:**
- Modify: `crates/range-solver/src/game/evaluation.rs:92-101`

**Step 1: Write the failing test**

Add to the test module at the bottom of `evaluation.rs` (or a nearby test file):

```rust
#[test]
fn boundary_reach_updates_every_visit() {
    // Build a minimal turn game with boundaries, set initial reaches,
    // run solve_step, check that boundary_reach reflects the NEW reaches
    // (not the initial ones). Then run a second solve_step and verify
    // reaches changed again.
    //
    // This test verifies the fix: boundary_reach overwrites on every visit
    // instead of caching on first visit only.
}
```

The exact test body depends on constructing a minimal PostFlopGame with boundaries. Use the existing `evaluate_game_boundaries_sets_finite_cfvs` test pattern from `turn_generate.rs` as a template for building a test game.

**Step 2: Fix the cache-once behavior**

In `evaluation.rs`, lines 92-101, change:

```rust
// Lazily cache opponent reach at this boundary.
let opp = player ^ 1;
let reach_index = ordinal * 2 + opp;
if reach_index < self.boundary_reach.len() {
    let guard = self.boundary_reach[reach_index].lock().unwrap();
    if guard.is_empty() {
        drop(guard);
        *self.boundary_reach[reach_index].lock().unwrap() = cfreach.to_vec();
    }
}
```

To:

```rust
// Update opponent reach at this boundary every visit.
let opp = player ^ 1;
let reach_index = ordinal * 2 + opp;
if reach_index < self.boundary_reach.len() {
    *self.boundary_reach[reach_index].lock().unwrap() = cfreach.to_vec();
}
```

This removes the `is_empty()` guard so reaches are always overwritten with the current iteration's values.

**Step 3: Verify compilation and existing tests**

Run: `cargo test -p range-solver 2>&1 | tail -5`
Expected: all pass. The change is backward-compatible — overwriting with the same data on first visit is identical to caching.

**Step 4: Commit**

```
git commit -m "fix: boundary_reach overwrites every visit instead of caching once"
```

---

### Task 2: Add `active_pool_size` config option

**Files:**
- Modify: `crates/cfvnet/src/config.rs:143-208`

**Step 1: Add the field to DatagenConfig**

After `leaf_eval_interval` (line 167), add:

```rust
    /// Number of games in the active pool for per-iteration boundary re-eval.
    /// Only used in model mode. Default 64.
    #[serde(default = "default_active_pool_size")]
    pub active_pool_size: usize,
```

Add the default function:

```rust
fn default_active_pool_size() -> usize {
    64
}
```

Update `Default for DatagenConfig` to include:

```rust
    active_pool_size: default_active_pool_size(),
```

**Step 2: Verify compilation**

Run: `cargo build -p cfvnet 2>&1 | tail -5`
Expected: compiles (existing tests may need `active_pool_size` added if they construct `DatagenConfig` without `..Default::default()`).

**Step 3: Commit**

```
git commit -m "feat: add active_pool_size config for per-iteration boundary eval"
```

---

### Task 3: Build the GPU batch eval function for the ring buffer

**Files:**
- Modify: `crates/cfvnet/src/datagen/turn_generate.rs`

This function evaluates boundaries for ALL active games in one GPU forward pass, using `boundary_reach` from the previous iteration (or `initial_weights` for iteration 0).

**Step 1: Write the function**

Add at module level, near `evaluate_batch`:

```rust
/// Evaluate boundaries for all active games in a single GPU forward pass.
///
/// For each game, reads `boundary_reach` (opponent reaches from last iteration).
/// For iteration 0, uses `initial_weights` since no reaches exist yet.
/// Builds one combined tensor, runs `model.forward()`, scatters results
/// back via `set_boundary_cfvs`.
fn evaluate_pool_boundaries<B2: burn::tensor::backend::Backend>(
    model: &CfvNet<B2>,
    device: &B2::Device,
    active_games: &mut [(Situation, PostFlopGame, u32)],
) where
    B2::Device: Clone,
{
    use burn::tensor::{Tensor, TensorData};

    let mut all_inputs: Vec<f32> = Vec::new();
    let mut requests: Vec<BoundaryRequest> = Vec::new();
    let mut rows_per: Vec<usize> = Vec::new();

    for (gi, (sit, game, _iter)) in active_games.iter().enumerate() {
        build_game_inputs(gi, game, sit, &mut all_inputs, &mut requests, &mut rows_per);
    }

    if all_inputs.is_empty() {
        return;
    }

    let total_rows = all_inputs.len() / INPUT_SIZE;
    let data = TensorData::new(all_inputs, [total_rows, INPUT_SIZE]);
    let input_tensor = Tensor::<B2, 2>::from_data(data, device);
    let output = model.forward(input_tensor);
    let out_vec: Vec<f32> = output.into_data().to_vec().expect("output tensor conversion");

    for (gi, ordinal, player, cfvs) in decode_boundary_cfvs(&out_vec, &requests, &rows_per) {
        active_games[gi].1.set_boundary_cfvs(ordinal, player, cfvs);
    }
}
```

**IMPORTANT:** The current `build_game_inputs` builds input rows using `sit.ranges` (the initial RSP/blueprint ranges). For per-iteration re-eval, the input to the neural net should use the **current boundary reaches** (which evolve during solving), not the initial ranges. This is a key difference.

However, looking at the neural net input layout (from `river_net_evaluator.rs:105-112`):
```
[0..1326)      — OOP range
[1326..2652)   — IP range
[2652..2704)   — board one-hot
...
```

The ranges in the input are the reach probabilities. For the first iteration, these are the initial ranges. For subsequent iterations, they should be the updated `boundary_reach` values.

You'll need a variant of `build_game_inputs` that reads from `game.boundary_reach(ordinal, player)` instead of `sit.ranges`. OR: update `sit.ranges` before each GPU call is NOT correct because ranges diverge per boundary node.

**The clean approach:** Write a new `build_pool_game_inputs` that, for each boundary node, reads the reach from `game.boundary_reach(ordinal, opp_player)` and the traverser's reach from `game.initial_weights(player)` weighted by the strategy. For the first iteration (boundary_reach is empty), fall back to `sit.ranges`.

This is the most complex part of the implementation. The existing `build_game_inputs` computes per-river-card input rows using the flat 1326-indexed ranges. The per-iteration version needs to do the same but with per-boundary reach values.

**Step 2: Test with a simple assertion**

Add a test that builds 2 games, calls `evaluate_pool_boundaries`, verifies boundary CFVs are set (non-empty) on both games.

**Step 3: Commit**

```
git commit -m "feat: add evaluate_pool_boundaries for ring buffer GPU eval"
```

---

### Task 4: Build the lockstep orchestrator loop

**Files:**
- Modify: `crates/cfvnet/src/datagen/turn_generate.rs`

This is the core change. Add a new function `generate_turn_training_data_iterative` that implements the lockstep ring buffer pipeline.

**Step 1: Write the orchestrator function**

```rust
/// Turn datagen with per-iteration boundary re-evaluation.
///
/// K games cycle through a lockstep loop:
/// 1. GPU batch eval all boundaries (one forward pass)
/// 2. solve_step all K games in parallel (thread::scope, force_sequential)
/// 3. Graduate finished games to writer, inject new ones from deal buffer
fn generate_turn_training_data_iterative(
    config: &CfvnetConfig,
    output_path: &Path,
) -> Result<(), String> {
    // ... setup: load model, create channels, start deal gen + writer threads ...

    let pool_size = config.datagen.active_pool_size;
    let solver_iterations = config.datagen.solver_iterations;
    let threads = config.datagen.threads;

    // Fill initial active pool from deal buffer.
    let mut active: Vec<(Situation, PostFlopGame, u32)> = Vec::with_capacity(pool_size);
    for _ in 0..pool_size {
        match deal_rx.recv() {
            Ok((sit, game)) => active.push((sit, game, 0)),
            Err(_) => break,
        }
    }

    // Lockstep loop.
    while !active.is_empty() {
        // 1. GPU: evaluate boundaries for all active games.
        evaluate_pool_boundaries(&model, &device, &mut active);

        // 2. Solve: one iteration per game, parallel via thread::scope.
        let chunk_size = (active.len() + threads - 1) / threads;
        std::thread::scope(|s| {
            for chunk in active.chunks_mut(chunk_size.max(1)) {
                s.spawn(|| {
                    range_solver::set_force_sequential(true);
                    for (_sit, game, iter) in chunk.iter_mut() {
                        solve_step(game, *iter);
                        *iter += 1;
                    }
                });
            }
        });

        // 3. Flush boundary caches so next GPU eval sees fresh reaches.
        for (_sit, game, _iter) in &active {
            game.flush_boundary_caches();
        }

        // 4. Graduate finished games, inject new ones.
        let mut i = 0;
        while i < active.len() {
            if active[i].2 >= solver_iterations {
                let (sit, mut game, _iter) = active.swap_remove(i);
                // Compute exploitability and extract results.
                let exploit = compute_exploitability(&game);
                let pot = f64::from(sit.pot);
                // ... extract CFVs, send to writer ...

                // Inject replacement from deal buffer.
                if let Ok((new_sit, new_game)) = deal_rx.try_recv() {
                    active.push((new_sit, new_game, 0));
                }
            } else {
                i += 1;
            }
        }
    }

    // ... shutdown writer, report stats ...
    Ok(())
}
```

**Step 2: Wire into `generate_turn_training_data`**

In the main dispatch function, add a check for `mode == "iterative"` or use `leaf_eval_interval > 0` as the trigger:

```rust
// In generate_turn_training_data:
if config.datagen.leaf_eval_interval == 0 && !exact_mode && needs_gpu {
    // Per-iteration boundary re-evaluation mode.
    return generate_turn_training_data_iterative(config, output_path);
}
```

Note: `leaf_eval_interval == 0` means "every iteration" per the existing config comment. So `leaf_eval_interval: 0` (the default) would trigger the new path. To keep backward compatibility, you may want to use a different trigger like `mode: "iterative"` instead.

**Step 3: Add deal gen + writer threads**

Reuse the existing patterns:
- Deal gen: same as current Stage 1 (sequential sampling + rayon tree building)
- Writer: same as current Stage 4 (receives `SolveResult` tuples, serializes, writes)

**Step 4: Add progress bar with pool status**

Show: `pool:[K] iter:[avg] deal→[N]→solve→[N]→write  expl:X.X mbb/h`

**Step 5: Test end-to-end**

Write a test that runs `generate_turn_training_data_iterative` with a tiny untrained model, 5 samples, 10 iterations, pool_size=2. Verify output records exist and have finite values.

**Step 6: Commit**

```
git commit -m "feat: lockstep ring buffer pipeline with per-iteration boundary re-eval"
```

---

### Task 5: Handle the boundary reach input to GPU correctly

**Files:**
- Modify: `crates/cfvnet/src/datagen/turn_generate.rs`

This is the critical correctness task. The GPU input rows need to use the **current boundary reaches** (which evolve during solving), not the initial deal ranges.

**Current `build_game_inputs`** reads ranges from `sit.ranges[0]` and `sit.ranges[1]` — the initial deal-time ranges. These are correct for the first iteration but stale afterward.

**For subsequent iterations**, each boundary node has its own `boundary_reach(ordinal, player)` which reflects the current strategy's reach at that boundary. The GPU input should use these per-boundary reaches.

**Step 1: Write `build_pool_game_inputs`**

A variant of `build_game_inputs` that:
- For iteration 0 (or empty boundary_reach): uses `sit.ranges` (same as current)
- For iteration > 0: uses `game.boundary_reach(ordinal, player)` for the opponent range in the neural net input, and recomputes the traverser range from the game's current strategy

OR simpler: just use `sit.ranges` always. The boundary CFVs will still be more accurate than pre-computing once because the boundary_reach affects the `evaluate_boundary_single` call (line 130 of evaluation.rs), not the neural net input. The neural net input is the "context" (board + pot + ranges at the subgame root), while the boundary_reach affects how the CFVs are applied.

**Decision needed:** Check whether the neural net was trained with root-level ranges or boundary-level reaches as input. If root-level ranges (which is how `build_game_inputs` works), then using `sit.ranges` is correct even for subsequent iterations. The neural net predicts CFVs given the subgame root context, and the solver applies them weighted by the current boundary reaches.

If this is the case, `evaluate_pool_boundaries` with the existing `build_game_inputs` is already correct — no changes needed here.

**Step 2: Verify by comparing outputs**

Run the iterative pipeline and the pre-compute pipeline on the same situations. The iterative pipeline should produce lower exploitability (better solutions) if boundary re-eval is working.

**Step 3: Commit if changes needed**

```
git commit -m "fix: use boundary-level reaches in GPU input for per-iteration eval"
```

---

### Task 6: Full test suite + final verification

**Step 1: Run the entire test suite**

Run: `cargo test 2>&1 | grep "test result:"` — all must pass.

**Step 2: Run clippy**

Run: `cargo clippy -p cfvnet -p range-solver 2>&1 | grep "error\|warning" | head -20`

**Step 3: Benchmark comparison**

Run both pipelines on the same config and compare:
- Throughput (samples/sec)
- Average exploitability
- Output record quality

**Step 4: Final commit**

```
git commit -m "chore: cleanup and verify per-iteration boundary eval pipeline"
```

---

## Important Notes for Implementer

1. **`flush_boundary_caches()` already exists** — don't reimplement it. Call it between iterations so `solve_step` lazily repopulates `boundary_reach` during its traversal.

2. **`solve_step` takes `&T` (not `&mut T`)** — it uses interior mutability via `Mutex<Vec<f32>>` for regrets, strategy, and boundary data. This means multiple games CAN be solved in parallel via `thread::scope` since each game is a separate object.

3. **`compute_exploitability` is expensive** — only call it once per game at graduation, not per iteration.

4. **`force_sequential` is thread-local** — each thread in `thread::scope` must call `range_solver::set_force_sequential(true)` independently.

5. **The neural net input question (Task 5)** is the biggest unknown. The `build_game_inputs` function uses `sit.ranges` (initial ranges at deal time). If the neural net was trained this way, it's correct to keep using initial ranges — the boundary_reach affects how CFVs are applied by the solver, not the net's input. Verify this before changing.

6. **Memory:** K=64 games × ~5 MB = ~320 MB for the active pool. Plus the deal buffer channel (256 capacity × ~5 MB = ~1.3 GB). Total ~1.6 GB. Adjust `active_pool_size` if memory is tight.
