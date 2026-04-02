# Batch wgpu Stage 2 GPU Inference — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Batch all boundary evaluations across multiple games into a single GPU forward pass in the wgpu turn datagen path, eliminating per-game GPU round-trips.

**Architecture:** Extract the CUDA path's `build_game_inputs`/`scatter_gpu_results` to module level, then replace the wgpu Stage 2's per-game `evaluate_game_boundaries()` loop with a single batched forward pass using those extracted functions. One file changes: `turn_generate.rs`.

**Tech Stack:** Rust, burn (wgpu backend), `PostFlopGame` from range-solver

**Design doc:** `docs/plans/2026-04-01-wgpu-batch-inference-design.md`
**Bean:** `poker_solver_rust-a1gv`

---

## Context for Implementer

All work happens in one file: `crates/cfvnet/src/datagen/turn_generate.rs` (~2650 lines).

**Key types/functions to understand:**

- `BoundaryRequest` (currently line ~657, inside `#[cfg(feature = "cuda")]` fn) — tracks which game/ordinal/player a batch of GPU rows belongs to, plus the combo indices and river validity masks needed to scatter results back
- `build_game_inputs()` (line ~669) — builds raw `f32` input rows for ALL boundary nodes of a single game, across all 48 river cards. Appends to shared `all_inputs` buffer and pushes `BoundaryRequest`s
- `scatter_gpu_results()` (line ~736) — reads GPU output, averages over river cards per combo, and calls `game.set_boundary_cfvs()` for each request. Currently takes `&mut [Option<PostFlopGame>]`
- `evaluate_game_boundaries()` (line ~147) — the per-game evaluation function used by wgpu Stage 2. Calls `evaluator.evaluate_boundaries()` which does one GPU forward pass per game
- `OUTPUT_SIZE` — constant (1326), currently imported only inside the CUDA fn at line 576

**The wgpu Stage 2 (lines 1435-1512):**

Each of N_GPU_THREADS (2) threads loads its own model, collects a batch of up to GPU_BATCH_SIZE (32) games from a shared channel, then loops `evaluate_game_boundaries()` per game. This is the hot path — each call triggers a separate ~48-row GPU forward pass.

**The CUDA path (lines 569-897):**

Already batches: collects 128 games' boundary inputs into one flat `Vec<f32>`, sends to a GPU thread, gets back one output `Vec<f32>`, scatters results. We're extracting this pattern.

---

### Task 1: Extract BoundaryRequest and constants to module level

**Files:**
- Modify: `crates/cfvnet/src/datagen/turn_generate.rs`

**Step 1: Add `OUTPUT_SIZE` to module-level imports**

At line 43, change:
```rust
use crate::model::network::{CfvNet, INPUT_SIZE};
```
to:
```rust
use crate::model::network::{CfvNet, INPUT_SIZE, OUTPUT_SIZE};
```

**Step 2: Move `BoundaryRequest` struct to module level**

Cut the struct from inside `generate_turn_training_data_cuda` (line ~657-664) and paste it at module level, after the `SOLVE_BATCH_SIZE` constant (after line 53). Add a doc comment:

```rust
/// Tracks a single boundary evaluation request within a batched GPU forward pass.
///
/// Each request corresponds to one (game, boundary_ordinal, player) triple.
/// The GPU output rows for this request span `rows_per` contiguous rows
/// starting at the cumulative offset.
struct BoundaryRequest {
    game_idx: usize,
    ordinal: usize,
    player: usize,
    combo_indices: Vec<usize>,
    river_valid_masks: Vec<Vec<bool>>,
    num_combos: usize,
}
```

**Step 3: Move `PREFIX_LEN` constant to module level**

Cut from line ~667 and place right after `BoundaryRequest`:

```rust
/// Prefix length for batched GPU input rows:
/// ranges (2 × OUTPUT_SIZE) + board_onehot (52) + rank_presence (13).
const PREFIX_LEN: usize = OUTPUT_SIZE * 2 + 52 + 13;
```

**Step 4: Remove the duplicate definitions from inside the CUDA fn**

Delete the `struct BoundaryRequest { ... }` block and `const PREFIX_LEN` that were inside `generate_turn_training_data_cuda`. Also remove the `use crate::model::network::OUTPUT_SIZE;` import on line 576 (now at module level).

**Step 5: Run `cargo build -p cfvnet 2>&1 | tail -5`**

Expected: compiles (possibly with warnings about unused imports on non-cuda builds). No errors.

**Step 6: Commit**

```
git add crates/cfvnet/src/datagen/turn_generate.rs
git commit -m "refactor: extract BoundaryRequest and PREFIX_LEN to module level"
```

---

### Task 2: Extract `build_game_inputs` to module level

**Files:**
- Modify: `crates/cfvnet/src/datagen/turn_generate.rs`

**Step 1: Move `build_game_inputs` out of the CUDA function**

Cut the function `build_game_inputs` (currently line ~669-734 inside the CUDA fn) and paste it at module level, after the `PREFIX_LEN` constant. The signature stays identical:

```rust
/// Build batched GPU input rows for all boundary nodes of a single game.
///
/// For each boundary node × player × river card, appends one `INPUT_SIZE`-element
/// row to `all_inputs`. Also pushes one `BoundaryRequest` per (boundary, player)
/// pair into `requests`, and the corresponding river count into `rows_per`.
fn build_game_inputs(
    gi: usize,
    game: &PostFlopGame,
    sit: &super::sampler::Situation,
    all_inputs: &mut Vec<f32>,
    requests: &mut Vec<BoundaryRequest>,
    rows_per: &mut Vec<usize>,
) {
    // ... body unchanged ...
}
```

No changes to the body — it already references `OUTPUT_SIZE`, `PREFIX_LEN`, `BoundaryRequest`, `card_pair_to_index` which are all module-level after Task 1.

**Step 2: Verify the CUDA fn still compiles**

The CUDA path's call sites at lines ~828 don't change — they already call `build_game_inputs(gi, game, sit, ...)`.

Run: `cargo build -p cfvnet 2>&1 | tail -5`
Expected: compiles with no errors.

**Step 3: Commit**

```
git add crates/cfvnet/src/datagen/turn_generate.rs
git commit -m "refactor: extract build_game_inputs to module level"
```

---

### Task 3: Extract scatter logic to module level

**Files:**
- Modify: `crates/cfvnet/src/datagen/turn_generate.rs`

The existing `scatter_gpu_results` takes `&mut [Option<PostFlopGame>]` (CUDA path uses `Vec<Option<PostFlopGame>>`). The wgpu path stores games as `Vec<(Situation, PostFlopGame)>`. Rather than generalizing the signature, extract just the averaging logic and let each call site do its own game access.

**Step 1: Create `decode_boundary_cfvs` at module level**

Place after `build_game_inputs`:

```rust
/// Decode batched GPU output into per-request CFVs by averaging over river cards.
///
/// Returns one `(game_idx, ordinal, player, cfvs)` tuple per request.
/// Each `cfvs` is a `Vec<f32>` of length `num_combos`.
fn decode_boundary_cfvs(
    out_vec: &[f32],
    requests: &[BoundaryRequest],
    rows_per: &[usize],
) -> Vec<(usize, usize, usize, Vec<f32>)> {
    let mut results = Vec::with_capacity(requests.len());
    let mut row_offset = 0;
    for (req, &nr) in requests.iter().zip(rows_per.iter()) {
        let mut cfv_sum = vec![0.0_f64; req.num_combos];
        let mut cfv_count = vec![0_u32; req.num_combos];
        for (ri, mask) in req.river_valid_masks.iter().enumerate() {
            let rs = (row_offset + ri) * OUTPUT_SIZE;
            for (i, &idx) in req.combo_indices.iter().enumerate() {
                if mask[i] {
                    cfv_sum[i] += f64::from(out_vec[rs + idx]);
                    cfv_count[i] += 1;
                }
            }
        }
        row_offset += nr;
        let cfvs: Vec<f32> = cfv_sum
            .iter()
            .zip(cfv_count.iter())
            .map(|(&s, &c)| if c > 0 { (s / f64::from(c)) as f32 } else { 0.0 })
            .collect();
        results.push((req.game_idx, req.ordinal, req.player, cfvs));
    }
    results
}
```

**Step 2: Update the CUDA path's `scatter_gpu_results` to use `decode_boundary_cfvs`**

Replace the body of the existing `scatter_gpu_results` inside the CUDA fn (lines ~736-763):

```rust
fn scatter_gpu_results(
    out_vec: &[f32],
    requests: &[BoundaryRequest],
    rows_per: &[usize],
    games: &mut [Option<PostFlopGame>],
) {
    for (gi, ordinal, player, cfvs) in decode_boundary_cfvs(out_vec, requests, rows_per) {
        if let Some(game) = &mut games[gi] {
            game.set_boundary_cfvs(ordinal, player, cfvs);
        }
    }
}
```

**Step 3: Verify compilation**

Run: `cargo build -p cfvnet 2>&1 | tail -5`
Expected: compiles.

**Step 4: Run existing tests to verify no regression**

Run: `cargo test -p cfvnet evaluate_game_boundaries_sets_finite_cfvs 2>&1 | tail -5`
Expected: PASS (this test uses `evaluate_game_boundaries` which is unchanged, confirming the extraction didn't break the CUDA scatter path).

Also: `cargo test -p cfvnet pipeline_matches_monolithic 2>&1 | tail -5`
Expected: PASS

**Step 5: Commit**

```
git add crates/cfvnet/src/datagen/turn_generate.rs
git commit -m "refactor: extract decode_boundary_cfvs for reuse across GPU backends"
```

---

### Task 4: Write test for batched multi-game boundary evaluation

**Files:**
- Modify: `crates/cfvnet/src/datagen/turn_generate.rs` (test section, line ~1625+)

**Step 1: Write the test**

Add this test in the `#[cfg(test)] mod tests` block, after the existing `evaluate_game_boundaries_sets_finite_cfvs` test:

```rust
/// Test that batched GPU evaluation (build_game_inputs + forward + decode)
/// produces the same boundary CFVs as per-game evaluate_game_boundaries.
#[test]
fn batched_evaluation_matches_per_game() {
    use burn::tensor::{Tensor, TensorData};

    let device = <B as burn::tensor::backend::Backend>::Device::default();
    let model = CfvNet::<B>::new(&device, 1, 8, INPUT_SIZE);
    let evaluator = RiverNetEvaluator::new(
        CfvNet::<B>::new(&device, 1, 8, INPUT_SIZE),
        device.clone(),
    );

    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let datagen_config = DatagenConfig {
        num_samples: 1,
        street: "turn".into(),
        solver_iterations: 20,
        target_exploitability: Some(0.05),
        threads: 1,
        seed: Some(99),
        ..Default::default()
    };

    // Build 3 games to ensure multi-game batching is tested.
    let bet_sizes = vec![parse_bet_sizes_depth(&["50%".into(), "a".into()])];
    let mut games_per_game: Vec<(Situation, PostFlopGame)> = Vec::new();
    for _ in 0..10 {
        let sit = sample_situation(&datagen_config, 200, 4, &mut rng);
        if sit.effective_stack <= 0 {
            continue;
        }
        if let Some(game) = build_turn_game(
            sit.board_cards(),
            f64::from(sit.pot),
            f64::from(sit.effective_stack),
            &sit.ranges,
            &bet_sizes,
        ) {
            if game.num_boundary_nodes() > 0 {
                games_per_game.push((sit, game));
            }
        }
        if games_per_game.len() >= 3 {
            break;
        }
    }
    assert!(
        games_per_game.len() >= 2,
        "need at least 2 valid games for batching test"
    );

    // --- Path A: per-game evaluation (existing) ---
    let mut per_game_cfvs: Vec<Vec<(usize, usize, Vec<f32>)>> = Vec::new();
    for (sit, game) in &games_per_game {
        let mut game_clone = game.clone();
        evaluate_game_boundaries(
            &mut game_clone,
            sit.board_cards(),
            f64::from(sit.pot),
            f64::from(sit.effective_stack),
            &sit.ranges,
            &evaluator,
        );
        // Read back boundary CFVs.
        let mut cfvs_for_game = Vec::new();
        for ord in 0..game_clone.num_boundary_nodes() {
            for player in 0..2usize {
                let cfvs = game_clone.boundary_cfvs(ord, player);
                cfvs_for_game.push((ord, player, cfvs.to_vec()));
            }
        }
        per_game_cfvs.push(cfvs_for_game);
    }

    // --- Path B: batched evaluation (new) ---
    let mut all_inputs: Vec<f32> = Vec::new();
    let mut requests: Vec<BoundaryRequest> = Vec::new();
    let mut rows_per: Vec<usize> = Vec::new();
    for (gi, (sit, game)) in games_per_game.iter().enumerate() {
        build_game_inputs(gi, game, sit, &mut all_inputs, &mut requests, &mut rows_per);
    }

    let total_rows = all_inputs.len() / INPUT_SIZE;
    let data = TensorData::new(all_inputs, [total_rows, INPUT_SIZE]);
    let input_tensor = Tensor::<B, 2>::from_data(data, &device);
    let output = model.forward(input_tensor);
    let out_vec: Vec<f32> = output.into_data().to_vec().expect("output tensor conversion");

    let decoded = decode_boundary_cfvs(&out_vec, &requests, &rows_per);

    // --- Compare ---
    for (gi, ordinal, player, batched_cfvs) in &decoded {
        let per_game = &per_game_cfvs[*gi];
        let matching = per_game
            .iter()
            .find(|(o, p, _)| *o == *ordinal && *p == *player)
            .expect("should find matching boundary");
        for (i, (&b, &p)) in batched_cfvs.iter().zip(matching.2.iter()).enumerate() {
            assert!(
                (b - p).abs() < 1e-4,
                "game {gi} ord {ordinal} player {player} combo {i}: batched={b} per_game={p}"
            );
        }
    }
}
```

**IMPORTANT:** This test uses TWO separate `CfvNet` instances (`model` for batched path, evaluator wraps a second copy). Since both are initialized identically (same architecture, same random seed from burn's default init), they should produce the same outputs. If burn's init is non-deterministic across instances, the test will need to share one model. In that case, change to:

```rust
let model = CfvNet::<B>::new(&device, 1, 8, INPUT_SIZE);
let evaluator = RiverNetEvaluator::new(model.clone(), device.clone());
```

(CfvNet implements `Clone` via burn's `Module` derive.)

**Step 2: Check if `boundary_cfvs` getter exists on PostFlopGame**

Run: `grep -n "fn boundary_cfvs\|fn get_boundary_cfvs" crates/range-solver/src/game/*.rs`

If it doesn't exist, the test needs a different comparison strategy. Alternative: compare by solving both games and checking the solve results match (like the existing `pipeline_matches_monolithic_solve` test). Adjust the test accordingly.

**Step 3: Run the test**

Run: `cargo test -p cfvnet batched_evaluation_matches_per_game 2>&1 | tail -15`

If it fails because `boundary_cfvs` doesn't exist, adapt the test to solve both games and compare solve results instead. Replace the "Compare" section with:

```rust
// Scatter batched results onto cloned games and solve both paths.
for (gi, ordinal, player, cfvs) in decoded {
    games_per_game[gi].1.set_boundary_cfvs(ordinal, player, cfvs);
}
// ... solve and compare game values
```

**Step 4: Commit once passing**

```
git add crates/cfvnet/src/datagen/turn_generate.rs
git commit -m "test: add batched_evaluation_matches_per_game"
```

---

### Task 5: Replace wgpu Stage 2 with batched evaluation

**Files:**
- Modify: `crates/cfvnet/src/datagen/turn_generate.rs`

This is the core change. Replace the per-game evaluation loop in each GPU thread with batched evaluation.

**Step 1: Modify the GPU thread body**

In the wgpu Stage 2 (inside `generate_turn_training_data`), the GPU thread currently:
1. Loads model → wraps in `RiverNetEvaluator`
2. Collects batch of games
3. Loops `evaluate_game_boundaries()` per game
4. Sends games downstream

Change it to:
1. Loads model → keeps raw `model` and `device` (no evaluator wrapper)
2. Collects batch of games
3. Calls `build_game_inputs` for all games → one `model.forward()` → `decode_boundary_cfvs` → sets on games
4. Sends games downstream

Replace the thread body (lines ~1446-1511). The new body:

```rust
stage2_handles.push(std::thread::spawn(move || {
    use burn::tensor::{Tensor, TensorData};

    let device = <B as burn::tensor::backend::Backend>::Device::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    let model = CfvNet::<B>::new(
        &device,
        river_hidden_layers,
        river_hidden_size,
        INPUT_SIZE,
    )
    .load_file(&river_model_path_owned, &recorder, &device)
    .unwrap_or_else(|e| panic!("GPU thread {gpu_id}: load river model: {e}"));

    let mut batch: Vec<(Situation, PostFlopGame)> = Vec::with_capacity(GPU_BATCH_SIZE);

    loop {
        // Lock rx1, block on first recv, drain non-blocking up to batch size.
        {
            let rx = rx1_ref.lock().expect("rx1 lock");
            match rx.recv() {
                Ok(item) => batch.push(item),
                Err(_) => break,
            }
            while batch.len() < GPU_BATCH_SIZE {
                match rx.try_recv() {
                    Ok(item) => batch.push(item),
                    Err(_) => break,
                }
            }
        } // Drop lock before GPU work.

        // Batched evaluation: one forward pass for all games in batch.
        evaluate_batch(&model, &device, &mut batch);
        stage2_count_ref.fetch_add(batch.len() as u64, Ordering::Relaxed);

        for item in batch.drain(..) {
            if tx2_ref.send(item).is_err() {
                return;
            }
        }
    }

    // Process remaining.
    if !batch.is_empty() {
        evaluate_batch(&model, &device, &mut batch);
        stage2_count_ref.fetch_add(batch.len() as u64, Ordering::Relaxed);
        for item in batch.drain(..) {
            let _ = tx2_ref.send(item);
        }
    }
}));
```

**Step 2: Add the `evaluate_batch` helper**

Place this at module level (or as a local fn — module level is cleaner for testing):

```rust
/// Evaluate all boundary nodes across a batch of games in a single GPU forward pass.
///
/// Builds one combined input tensor for all games, runs `model.forward()`,
/// then scatters the averaged CFVs back to each game's boundary nodes.
fn evaluate_batch<B2: burn::tensor::backend::Backend>(
    model: &CfvNet<B2>,
    device: &B2::Device,
    batch: &mut [(Situation, PostFlopGame)],
) where
    B2::Device: Clone,
{
    use burn::tensor::{Tensor, TensorData};

    let mut all_inputs: Vec<f32> = Vec::new();
    let mut requests: Vec<BoundaryRequest> = Vec::new();
    let mut rows_per: Vec<usize> = Vec::new();

    for (gi, (sit, game)) in batch.iter().enumerate() {
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
        batch[gi].1.set_boundary_cfvs(ordinal, player, cfvs);
    }
}
```

**Step 3: Clean up unused imports**

The wgpu path no longer uses `evaluate_game_boundaries` or `RiverNetEvaluator` in Stage 2. However, `evaluate_game_boundaries` is still used by tests, and `RiverNetEvaluator` is still imported at module level for tests. Check that removing any `use` statements doesn't break other code. Run `cargo build -p cfvnet 2>&1 | grep "unused"` and fix any warnings.

**Step 4: Run the new test**

Run: `cargo test -p cfvnet batched_evaluation_matches_per_game 2>&1 | tail -10`
Expected: PASS

**Step 5: Run all cfvnet tests**

Run: `cargo test -p cfvnet 2>&1 | tail -5`
Expected: all pass

**Step 6: Commit**

```
git add crates/cfvnet/src/datagen/turn_generate.rs
git commit -m "feat: batch wgpu Stage 2 GPU inference across games

Replace per-game evaluate_game_boundaries() loop with single batched
forward pass using build_game_inputs + model.forward + decode_boundary_cfvs.
Reduces GPU round-trips from ~32 per batch to 1."
```

---

### Task 6: Full test suite + final commit

**Step 1: Run the entire test suite**

Run: `cargo test 2>&1 | grep "test result:" | head -30`
Expected: all lines show `ok`, zero failures.

Run: `cargo test 2>&1 | tail -3`
Expected: clean finish.

**Step 2: Run clippy**

Run: `cargo clippy -p cfvnet 2>&1 | grep "error\|warning" | head -20`
Fix any new warnings introduced by the changes.

**Step 3: Verify full workspace builds**

Run: `cargo build 2>&1 | tail -3`
Expected: clean build (the extracted functions are not `#[cfg(feature = "cuda")]` so they compile on all targets).

**Step 4: Final commit if any clippy fixes were needed**

```
git add -A
git commit -m "chore: fix clippy warnings from wgpu batching refactor"
```

**Step 5: Update the bean**

```
beans update poker_solver_rust-a1gv -s completed --body-append "## Summary of Changes\n\nExtracted CUDA path's batching functions to module level and replaced wgpu Stage 2's per-game GPU calls with single batched forward pass per batch."
```
