# Boundary Eval Batching (Layer A) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate 256× per-batch ORT dispatch overhead in turn datagen by calling `evaluate_boundaries_batched` once per DCFR eval round instead of once per sit.

**Architecture:** The function `evaluate_boundaries_batched` in `crates/cfvnet/src/datagen/gpu_boundary_eval.rs:109` already accepts `&[BoundaryEvalRequest]` and handles multiple requests correctly (one combined ORT `session.run` call, then a per-request reduction). The bug is that `batch_boundary_leaf_cfvs` in `crates/cfvnet/src/datagen/domain/pipeline.rs:746-826` calls it inside a `for sit in sits` loop with `&[single_request]`. The fix: collect all requests first, then call the batched function once. A reference per-sit helper is retained as a test-only function so we can assert numerical equivalence between the two paths.

**Tech Stack:** Rust, ort (ONNX Runtime with TensorRT EP), cudarc, cfvnet.

**Bean:** `poker_solver_rust-p99d` (high — unblocks production turn datagen at batch ≥ 32).

**Design doc:** [`docs/plans/2026-04-20-boundary-eval-batching-design.md`](2026-04-20-boundary-eval-batching-design.md)

---

## Prerequisites

1. You are running in a dedicated git worktree on a feature branch. If not, stop and ask the coordinator.
2. Bean `oox2` (CUDA smem overflow fix) is already merged (commits `bf2a373`..`8946729` on main). Without that the integration test cannot launch and you cannot validate this work.
3. The ONNX BoundaryNet model exists at `local_data/models/cfvnet_river_py_v2/model.onnx`. Confirm with `ls -la local_data/models/cfvnet_river_py_v2/model.onnx`. If absent, stop and flag.
4. **Read the design doc in full** before touching code. Ideally also read `crates/cfvnet/src/datagen/gpu_boundary_eval.rs:109-278` to understand how the batched function groups inputs and reduces outputs.
5. Baseline measurement (optional, informational): before starting, run the `oox2` Task 7 command once and time it:
   ```bash
   mkdir -p local_data/cfvnet/turn/p99d_baseline
   time cargo run -p cfvnet --release --features gpu-turn-datagen -- generate \
     -c sample_configurations/turn_gpu_datagen.yaml \
     -o local_data/cfvnet/turn/p99d_baseline \
     --num-samples 32 --per-file 8 2>&1 | tee local_data/cfvnet/turn/p99d_baseline/_run.log
   ```
   Record the wall clock time. You will compare against this in Task 5. If time constraints make you want to skip this, it's OK — we already have the oox2 baseline recorded (~53 min for 4 files).

6. The yaml currently has `gpu_batch_size: 4`. **Leave this value unchanged** for the Task 5 validation — same config as the `oox2` baseline. Restoring to a larger batch is a separate housekeeping step.

---

## Task 1: Extract a test-only reference helper (per-sit path)

**Goal:** Mirror the current `batch_boundary_leaf_cfvs` implementation into a private, test-gated helper function `per_sit_batch_boundary_leaf_cfvs`. This is the "reference impl" the regression test in Task 2 will compare against after the refactor in Task 3. No behavioral change — the original function still exists and is still called from production code.

**Files:**
- Modify: `crates/cfvnet/src/datagen/domain/pipeline.rs` (add a new `#[cfg(test)]`-gated helper in the same `impl DomainPipeline` block, or as a free fn at module scope — see Step 1.2 for the choice).

### Step 1.1: Locate the current implementation

The function lives at `pipeline.rs:746-826`. Read it carefully so you understand every step. Note the associated-fn signature uses `Self::`. The tests gating elsewhere in this file use `#[cfg(feature = "gpu-turn-datagen")]` for the CUDA-using tests — you need both `#[cfg(test)]` and `#[cfg(feature = "gpu-turn-datagen")]` (the smoke test at `canonical_turn_tree_runs_one_iteration_without_smem_overflow` is a good reference for gating).

### Step 1.2: Add the reference helper

At the end of the `impl DomainPipeline` block (or as a free `#[cfg(test)]` helper in the same file — whichever matches existing code layout better; the private test helper pattern is fine either way), append:

```rust
#[cfg(all(test, feature = "gpu-turn-datagen"))]
#[allow(clippy::too_many_arguments)]
fn per_sit_batch_boundary_leaf_cfvs(
    evaluator: &crate::datagen::gpu_boundary_eval::GpuBoundaryEvaluator,
    topo: &gpu_range_solver::extract::TreeTopology,
    boundary_node_ids: &[usize],
    canonical_hand_cards: &[(u8, u8)],
    num_hands: usize,
    strategy_sums: &[Vec<f32>],
    specs: &[gpu_range_solver::SubgameSpec],
    sits: &[crate::datagen::sampler::Situation],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    use gpu_range_solver::compute_reach_at_nodes;
    use crate::datagen::gpu_boundary_eval::{
        evaluate_boundaries_batched, BoundaryEvalRequest,
    };
    use crate::datagen::range_gen::NUM_COMBOS;

    debug_assert_eq!(strategy_sums.len(), specs.len());
    debug_assert_eq!(strategy_sums.len(), sits.len());

    let num_boundaries = boundary_node_ids.len();
    let slot_len = num_boundaries * num_hands;
    let batch_len = specs.len();
    let mut batched_p0 = vec![0.0_f32; batch_len * slot_len];
    let mut batched_p1 = vec![0.0_f32; batch_len * slot_len];

    for (b, sit) in sits.iter().enumerate() {
        let spec = &specs[b];
        let strategy_sum = &strategy_sums[b];

        let reach = compute_reach_at_nodes(
            topo,
            strategy_sum,
            &spec.initial_weights,
            num_hands,
            boundary_node_ids,
        );

        debug_assert_eq!(num_hands, NUM_COMBOS);
        let mut oop_reach_1326 = vec![0.0_f32; num_boundaries * NUM_COMBOS];
        let mut ip_reach_1326 = vec![0.0_f32; num_boundaries * NUM_COMBOS];
        for bi in 0..num_boundaries {
            let src_base = bi * num_hands;
            let dst_base = bi * NUM_COMBOS;
            ip_reach_1326[dst_base..dst_base + NUM_COMBOS]
                .copy_from_slice(&reach[0][src_base..src_base + num_hands]);
            oop_reach_1326[dst_base..dst_base + NUM_COMBOS]
                .copy_from_slice(&reach[1][src_base..src_base + num_hands]);
        }

        let board_4: [u8; 4] = [sit.board[0], sit.board[1], sit.board[2], sit.board[3]];
        let request = BoundaryEvalRequest {
            board: board_4,
            pot: sit.pot as f32,
            effective_stack: sit.effective_stack as f32,
            oop_reach: oop_reach_1326,
            ip_reach: ip_reach_1326,
            num_boundaries,
        };

        let results =
            evaluate_boundaries_batched(evaluator, &[request], canonical_hand_cards)
                .map_err(|e| format!("boundary eval failed: {e}"))?;
        let slot_start = b * slot_len;
        batched_p0[slot_start..slot_start + slot_len]
            .copy_from_slice(&results[0].leaf_cfv_p0);
        batched_p1[slot_start..slot_start + slot_len]
            .copy_from_slice(&results[0].leaf_cfv_p1);
    }

    Ok((batched_p0, batched_p1))
}
```

This is a byte-for-byte copy of the current `batch_boundary_leaf_cfvs` body (check yourself — the only differences are (a) `#[cfg(all(test, feature = "gpu-turn-datagen"))]` gating and (b) the function name). It exists solely so the regression test can compare against the old behavior after Task 3 refactors the real function.

If your crate's test layout or associated-fn setup makes this awkward as a method, a free function in the same file is fine — the test in Task 2 can call it by path.

### Step 1.3: Verify it compiles

Run:
```bash
cargo check -p cfvnet --features gpu-turn-datagen --tests 2>&1 | tail -10
```
Expected: clean compile. If cfg gating is wrong (e.g. `#[cfg(test)]` outside an `impl` block) you'll see a clear error — fix the gating.

### Step 1.4: Do NOT commit yet

This helper is dead code without Task 2's test. Commit them together at the end of Task 2.

---

## Task 2: Write the failing regression test

**Goal:** A test that exercises both `batch_boundary_leaf_cfvs` (production path) and `per_sit_batch_boundary_leaf_cfvs` (reference) with the same inputs and asserts their leaf-CFV outputs match element-wise within FP tolerance. **Before Task 3, this test trivially passes because both functions share the same code.** After Task 3 refactors the production path, the test becomes meaningful.

**Files:**
- Modify: `crates/cfvnet/src/datagen/domain/pipeline.rs` tests module (find the `mod tests` or `mod gpu_turn_tests` block where `canonical_turn_tree_runs_one_iteration_without_smem_overflow` lives — add the new test there).

### Step 2.1: Read the existing test for setup patterns

Read `canonical_turn_tree_runs_one_iteration_without_smem_overflow` in its entirety. Use the same patterns for:
- Loading the ONNX model into `GpuBoundaryEvaluator`.
- Constructing the canonical topology via `build_canonical_turn_tree`.
- Building `SubgameSpec`s and `Situation`s.

The new test will need a small batch (4 sits is enough — use deterministic seeds).

### Step 2.2: Write the new test

Append to the same test module:

```rust
#[test]
#[cfg(feature = "gpu-turn-datagen")]
fn layer_a_batched_matches_per_sit_within_tolerance() {
    // Regression test for bean p99d: the refactored (batched-once)
    // path of batch_boundary_leaf_cfvs must produce leaf CFVs matching
    // the old (per-sit, batch-of-1 ORT call) path element-wise within
    // tight FP tolerance. Tolerance accounts for TensorRT potentially
    // picking different kernels at different batch sizes.

    let model_path = std::path::Path::new("local_data/models/cfvnet_river_py_v2/model.onnx");
    if !model_path.exists() {
        eprintln!(
            "SKIPPING: BoundaryNet model missing at {} — \
             run cfvnet train-boundary first",
            model_path.display()
        );
        return;
    }
    let evaluator =
        crate::datagen::gpu_boundary_eval::GpuBoundaryEvaluator::load(model_path)
            .expect("load BoundaryNet");

    // Build canonical turn topology (same as the smoke test).
    // Fill in call-site arguments to match build_canonical_turn_tree's real
    // signature — see canonical_turn_tree_runs_one_iteration_without_smem_overflow
    // for the pattern.
    let topo = /* build_canonical_turn_tree(...) */;
    let num_hands = 1326usize;
    let boundary_node_ids = /* topo.showdown_nodes.clone() or equivalent */;

    // Build 4 deterministic sits and matching specs/strategy_sums.
    // Use simple fixed values — the goal is to pin a reproducible input
    // that exercises the batching, not to test a realistic game state.
    let batch_size = 4usize;
    let sits: Vec<crate::datagen::sampler::Situation> = (0..batch_size)
        .map(|i| crate::datagen::sampler::Situation {
            // Use small-perturbation board/pot/stack values so each sit
            // produces distinct inputs (catches per-sit indexing bugs).
            board: [0, 1, 2, 3 + i as u8],
            pot: 100.0 + 10.0 * i as f64,
            effective_stack: 200.0 - 10.0 * i as f64,
            // Fill remaining fields with zeros / defaults. If Situation
            // has required fields not covered here, read its definition
            // at crates/cfvnet/src/datagen/sampler.rs:9 and fill them in.
            ..Default::default()
        })
        .collect();

    // Strategy sums: uniform / zero for determinism.
    let strategy_sums: Vec<Vec<f32>> = (0..batch_size)
        .map(|_| vec![0.0f32; topo.num_edges * num_hands])
        .collect();

    // SubgameSpecs: uniform reach on both players, None showdown outcomes
    // (matches build_turn_subgame_spec post-vb8r).
    let specs: Vec<gpu_range_solver::SubgameSpec> = (0..batch_size)
        .map(|_| gpu_range_solver::SubgameSpec {
            initial_weights: [
                vec![1.0f32 / num_hands as f32; num_hands],
                vec![1.0f32 / num_hands as f32; num_hands],
            ],
            showdown_outcomes_p0: None,
            showdown_outcomes_p1: None,
            // fold_payoffs sized to match the topology's fold_nodes count —
            // look at the smoke test's construction for the pattern.
            fold_payoffs_p0: vec![/* ... */],
            fold_payoffs_p1: vec![/* ... */],
        })
        .collect();

    let hand_cards: Vec<(u8, u8)> = (0..num_hands as u16)
        .map(|i| range_solver::card::index_to_card_pair(i as usize))
        .collect();

    // Call both paths.
    let (batched_p0, batched_p1) = DomainPipeline::batch_boundary_leaf_cfvs(
        &evaluator,
        &topo,
        &boundary_node_ids,
        &hand_cards,
        num_hands,
        &strategy_sums,
        &specs,
        &sits,
    )
    .expect("batched path");

    let (reference_p0, reference_p1) = DomainPipeline::per_sit_batch_boundary_leaf_cfvs(
        &evaluator,
        &topo,
        &boundary_node_ids,
        &hand_cards,
        num_hands,
        &strategy_sums,
        &specs,
        &sits,
    )
    .expect("per-sit reference path");

    // Element-wise tolerance: |a - b| <= 1e-5 + 1e-4 * |b|
    assert_eq!(batched_p0.len(), reference_p0.len());
    assert_eq!(batched_p1.len(), reference_p1.len());

    let check_close = |batched: &[f32], reference: &[f32], label: &str| {
        for (i, (&b, &r)) in batched.iter().zip(reference).enumerate() {
            let abs_tol = 1e-5_f32;
            let rel_tol = 1e-4_f32;
            let diff = (b - r).abs();
            let limit = abs_tol + rel_tol * r.abs();
            assert!(
                diff <= limit,
                "{label}[{i}] batched={b} reference={r} diff={diff} limit={limit}"
            );
            assert!(b.is_finite(), "{label}[{i}] batched value not finite: {b}");
        }
    };

    check_close(&batched_p0, &reference_p0, "p0");
    check_close(&batched_p1, &reference_p1, "p1");
}
```

**Developer notes (fill these in while writing — don't skip):**

- Some fields in `Situation` and `SubgameSpec` may have required values not shown above. Read the struct definitions (`crates/cfvnet/src/datagen/sampler.rs:9` for Situation, `crates/gpu-range-solver/src/batch.rs` for SubgameSpec) and fill them in.
- `fold_payoffs_p0/p1` length must equal `topo.fold_nodes.len()` (or whatever field contains fold count). Use zeros.
- `boundary_node_ids` must match what the smoke test uses; copy that construction verbatim.
- Don't be clever about making the inputs realistic — the test just needs to exercise the batching, and random seeds are fine if the seed is pinned (use a fixed `StdRng::seed_from_u64(...)` if anything random is required).
- If `Situation` doesn't derive `Default`, fill every field explicitly. Don't introduce `Default` just for the test.

### Step 2.3: Run the test — expect it to PASS

Pre-refactor, both functions share the same body, so the tolerance-check is vacuously true (zero diff).

Run:
```bash
cargo test -p cfvnet --features gpu-turn-datagen layer_a_batched_matches_per_sit_within_tolerance 2>&1 | tail -20
```

Expected: PASS. Runtime: likely 30-90 seconds (most of it loading the ONNX model and TensorRT engine initial compile).

If the test fails pre-refactor, something in the test setup is wrong — not a Layer A bug. Fix the test before proceeding.

### Step 2.4: Commit (Task 1 + Task 2 together)

```bash
git add crates/cfvnet/src/datagen/domain/pipeline.rs
git commit -m "$(cat <<'EOF'
test(cfvnet): regression test for batched boundary eval (p99d Layer A)

Adds a test-only per_sit_batch_boundary_leaf_cfvs helper that mirrors
the current production implementation, and a regression test that
asserts batch_boundary_leaf_cfvs produces leaf CFVs matching the
reference within tight FP tolerance (|a-b| <= 1e-5 + 1e-4*|b|).

Pre-refactor, both paths share the same body so the test passes
trivially. After the Layer A refactor (next commit) it becomes a
meaningful cross-implementation regression check.

Bean: poker_solver_rust-p99d

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Refactor `batch_boundary_leaf_cfvs` to batch the ORT call

**Goal:** Replace the `for sit in sits` loop's single-request eval with (a) a loop that builds all requests, then (b) a single `evaluate_boundaries_batched` call over all requests, then (c) a loop that copies results into batched buffers.

**File:** `crates/cfvnet/src/datagen/domain/pipeline.rs:746-826`.

### Step 3.1: Replace the function body

Replace the existing body of `batch_boundary_leaf_cfvs` (from the beginning of the `for (b, sit) in sits.iter().enumerate() {` loop through the `Ok((batched_p0, batched_p1))` return) with:

```rust
    #[cfg(feature = "gpu-turn-datagen")]
    #[allow(clippy::too_many_arguments)]
    fn batch_boundary_leaf_cfvs(
        evaluator: &crate::datagen::gpu_boundary_eval::GpuBoundaryEvaluator,
        topo: &gpu_range_solver::extract::TreeTopology,
        boundary_node_ids: &[usize],
        canonical_hand_cards: &[(u8, u8)],
        num_hands: usize,
        strategy_sums: &[Vec<f32>],
        specs: &[gpu_range_solver::SubgameSpec],
        sits: &[crate::datagen::sampler::Situation],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        use gpu_range_solver::compute_reach_at_nodes;
        use crate::datagen::gpu_boundary_eval::{
            evaluate_boundaries_batched, BoundaryEvalRequest,
        };
        use crate::datagen::range_gen::NUM_COMBOS;

        debug_assert_eq!(strategy_sums.len(), specs.len());
        debug_assert_eq!(strategy_sums.len(), sits.len());

        let num_boundaries = boundary_node_ids.len();
        let slot_len = num_boundaries * num_hands;
        let batch_len = specs.len();
        let mut batched_p0 = vec![0.0_f32; batch_len * slot_len];
        let mut batched_p1 = vec![0.0_f32; batch_len * slot_len];

        // Build all requests first — no ORT call inside this loop.
        let mut requests: Vec<BoundaryEvalRequest> = Vec::with_capacity(sits.len());
        for (b, sit) in sits.iter().enumerate() {
            let spec = &specs[b];
            let strategy_sum = &strategy_sums[b];

            let reach = compute_reach_at_nodes(
                topo,
                strategy_sum,
                &spec.initial_weights,
                num_hands,
                boundary_node_ids,
            );

            debug_assert_eq!(num_hands, NUM_COMBOS);
            let mut oop_reach_1326 = vec![0.0_f32; num_boundaries * NUM_COMBOS];
            let mut ip_reach_1326 = vec![0.0_f32; num_boundaries * NUM_COMBOS];
            for bi in 0..num_boundaries {
                let src_base = bi * num_hands;
                let dst_base = bi * NUM_COMBOS;
                ip_reach_1326[dst_base..dst_base + NUM_COMBOS]
                    .copy_from_slice(&reach[0][src_base..src_base + num_hands]);
                oop_reach_1326[dst_base..dst_base + NUM_COMBOS]
                    .copy_from_slice(&reach[1][src_base..src_base + num_hands]);
            }

            let board_4: [u8; 4] = [sit.board[0], sit.board[1], sit.board[2], sit.board[3]];
            requests.push(BoundaryEvalRequest {
                board: board_4,
                pot: sit.pot as f32,
                effective_stack: sit.effective_stack as f32,
                oop_reach: oop_reach_1326,
                ip_reach: ip_reach_1326,
                num_boundaries,
            });
        }

        // Single batched ORT call for all sits.
        let results = evaluate_boundaries_batched(evaluator, &requests, canonical_hand_cards)
            .map_err(|e| format!("boundary eval failed: {e}"))?;
        debug_assert_eq!(results.len(), sits.len());

        // Copy per-sit results into the batched buffers.
        for (b, result) in results.into_iter().enumerate() {
            let slot_start = b * slot_len;
            batched_p0[slot_start..slot_start + slot_len]
                .copy_from_slice(&result.leaf_cfv_p0);
            batched_p1[slot_start..slot_start + slot_len]
                .copy_from_slice(&result.leaf_cfv_p1);
        }

        Ok((batched_p0, batched_p1))
    }
```

Do NOT delete the `per_sit_batch_boundary_leaf_cfvs` helper from Task 1 — it stays as the regression reference.

### Step 3.2: Verify it compiles

```bash
cargo check -p cfvnet --features gpu-turn-datagen 2>&1 | tail -10
```
Expected: clean.

### Step 3.3: Run the regression test — expect it to PASS

```bash
cargo test -p cfvnet --features gpu-turn-datagen layer_a_batched_matches_per_sit_within_tolerance 2>&1 | tail -20
```

Expected: PASS. Runtime similar to before — the test still runs both paths, so total ORT work is the same.

**If the test FAILS:**

- Read the error. It will print `batched[i]=X reference[i]=Y diff=Z limit=L` for the first mismatch.
- If the diff is O(1e-3) relative or larger, TensorRT is picking a substantively different kernel at batch=256× that changes numerics meaningfully. Options: (a) loosen the tolerance to `1e-3 relative` and document why in a comment, (b) investigate if the per-request reduction in `evaluate_boundaries_batched` has any order-dependence, (c) report back to coordinator and discuss.
- If the diff is in unexpected positions (e.g. only p1 mismatches, not p0), check the copy-back loop: `results.into_iter().enumerate()` must yield in the same order as the input `requests`. This is guaranteed by the current `evaluate_boundaries_batched`, but verify nothing went wrong.
- If you see a panic instead of a diff failure, read the panic message — likely a slice-length mismatch from a wrong `slot_len` or `results.len()` assumption.

### Step 3.4: Run the existing smoke test to confirm no regression

```bash
cargo test -p cfvnet --features gpu-turn-datagen canonical_turn_tree_runs_one_iteration_without_smem_overflow 2>&1 | tail -20
```

Expected: PASS. Runtime should be **materially faster** than before (the smoke test runs one DCFR iteration which includes the boundary eval). Record the before/after time in your final report.

### Step 3.5: Commit

```bash
git add crates/cfvnet/src/datagen/domain/pipeline.rs
git commit -m "$(cat <<'EOF'
feat(cfvnet): batch boundary eval across canonical topology (p99d Layer A)

batch_boundary_leaf_cfvs now builds all per-sit BoundaryEvalRequests
up front and calls evaluate_boundaries_batched once over the full
batch, instead of once per sit. At gpu_batch_size=256 this kills
~256x of per-call ORT/TensorRT dispatch overhead per DCFR eval round.

The per_sit_batch_boundary_leaf_cfvs helper (introduced in the
previous commit) remains as a test-only regression reference — the
regression test now asserts the two paths produce identical leaf CFVs
within tight FP tolerance.

Expected speedup: the dominant cost in turn datagen (observed 1625 s
per sample at batch=4 in bean oox2's Task 7) was 256x-of-1 batch
ORT calls; Layer A eliminates that factor in the critical path.

Bean: poker_solver_rust-p99d

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Full-suite build + clippy

**Goal:** Per `CLAUDE.md`, confirm the whole project still builds and clippy is clean for the crates we touched.

### Step 4.1: Build the whole workspace

```bash
cargo build --workspace 2>&1 | tail -5
```
Expected: clean.

### Step 4.2: Clippy on changed crates

```bash
cargo clippy -p cfvnet --features gpu-turn-datagen -- -D warnings 2>&1 | grep -E "warning:|error:" | grep -E "crates/cfvnet/src/datagen/domain/pipeline.rs" | head -20
```
Expected: no warnings in `pipeline.rs`. Pre-existing warnings in untouched files may appear in the full output — only the `pipeline.rs` ones are in scope.

If a new warning appears in your changes, fix it minimally (don't refactor).

### Step 4.3: Workspace tests

```bash
time cargo test --workspace 2>&1 | tail -10
```

Expected: 201 passed, the same 8 pre-existing `mp_tui_scenarios` failures as documented in bean `oox2`'s Task 8. Report any new failures — those are regressions.

If this takes >5 minutes, note the time but don't block on it (the <60s CLAUDE.md target is a pre-existing violation unrelated to this work).

---

## Task 5: Manual validation — full turn datagen at `gpu_batch_size=4`

**Goal:** Prove the refactor speeds up real datagen. Compare wall-clock against the `oox2` baseline (53 min for 4 files).

### Step 5.1: Run the datagen command

```bash
mkdir -p local_data/cfvnet/turn/p99d_verify
time cargo run -p cfvnet --release --features gpu-turn-datagen -- generate \
  -c sample_configurations/turn_gpu_datagen.yaml \
  -o local_data/cfvnet/turn/p99d_verify \
  --num-samples 32 --per-file 8 2>&1 | tee local_data/cfvnet/turn/p99d_verify/_run.log
```

### Step 5.2: Record results

- Exit code: ____ (goal: 0)
- Number of files written: ____ / 4
- Total wall clock: ____ (goal: materially less than 53 min baseline — ideally <20 min)
- Samples/sec reported in log: ____ (compare to "0.0 samples/sec" from oox2 baseline)
- Peak RSS: ____ GB (sanity check: still under 64 GB)

Spot-check one output file for well-formedness:

```bash
ls -la local_data/cfvnet/turn/p99d_verify_*
```

Files should be ~276 KB each (same size as oox2 baseline) with exactly 16 GPU turn records per file.

### Step 5.3: If results are disappointing

If wall clock is within ~2× of the oox2 baseline (i.e. >25 min for 4 files), Layer A's dispatch-overhead hypothesis was wrong — the real bottleneck is CPU prep (reach computation + reshape + input building). In that case:

- Do NOT abandon Layer A. The code change is correct and harmless — the regression test proves equivalence.
- Commit the current state, close `p99d` as "Layer A done, throughput still blocked".
- Coordinator should file a Layer B bean (rayon-parallelize per-sit prep) as the next priority.

### Step 5.4: If results are successful

Proceed to Task 6.

---

## Task 6: Bean closure

### Step 6.1: Update bean with summary

```bash
beans update --json poker_solver_rust-p99d -s completed --body-append "$(cat <<'EOF'


## Summary of Changes (Layer A)

Hoisted evaluate_boundaries_batched out of the per-sit loop in
batch_boundary_leaf_cfvs. At gpu_batch_size=N, previously ran N batch-of-1
ORT/TensorRT calls per DCFR eval round; now runs 1 batch-of-N call.
evaluate_boundaries_batched was already batch-capable — only the call
site was single-request.

**Plan:** `docs/plans/2026-04-20-boundary-eval-batching.md`

**Commits:**
- test: regression test + per-sit reference helper
- feat: batched call site

**Validation:**
- Regression test: batched path matches per-sit path within 1e-5 abs / 1e-4 rel tolerance.
- Smoke test `canonical_turn_tree_runs_one_iteration_without_smem_overflow` still passes; runtime improvement: <RECORD BEFORE/AFTER>.
- Manual datagen (gpu_batch_size=4, 32 samples, 8/file):
  - oox2 baseline: 53 min for 4 files.
  - Layer A: <RECORD WALL CLOCK> for 4 files.
  - Peak RSS: <RECORD> GB.

**Layer B/C status:** deferred. Filed as follow-up beans only if Layer A's measured speedup doesn't reach production viability.
EOF
)"
```

Replace `<RECORD ...>` placeholders with actual measurements from Task 5.

### Step 6.2: Commit the bean update

```bash
git add .beans/poker_solver_rust-p99d*.md
git commit -m "chore: complete bean p99d (boundary eval Layer A)"
```

### Step 6.3: Report to coordinator

Final report must include:
- All commit SHAs.
- Before/after wall clock from the smoke test (fast way to see the dispatch speedup).
- Before/after wall clock from manual datagen validation.
- Peak RSS from validation.
- Whether follow-up Layer B/C beans should be filed (yes if throughput target not yet met).

---

## Execution guidance

- Work in a dedicated worktree.
- Commit after Tasks 2 and 3. Task 1's helper gets committed with Task 2's test in one commit.
- Do NOT squash.
- Do NOT delete `per_sit_batch_boundary_leaf_cfvs` after Task 3 — it's the regression reference going forward.
- If the regression test fails at Task 3.3, STOP and report — do not paper over with a looser tolerance without discussing.
- Do not attempt Layer B (rayon) in this bean. It's explicitly deferred.
