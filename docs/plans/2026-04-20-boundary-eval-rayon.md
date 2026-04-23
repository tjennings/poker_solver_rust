# Rayon-Parallelize Boundary Eval (Layers B + C) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the main-thread serial CPU bottleneck in turn datagen by rayon-parallelizing the two remaining embarrassingly-parallel loops: the per-sit prep in `batch_boundary_leaf_cfvs` (Layer B) and the per-request f64 reduction in `evaluate_boundaries_batched` (Layer C).

**Architecture:** Two independent rayon migrations. Both parallelize only *across* sits/requests, so per-element arithmetic is unchanged → output is byte-identical to the serial reference. Layer C requires pre-computing per-request row offsets once (replacing the current sequential `row_cursor`). No API changes to public functions; the test-only `per_sit_batch_boundary_leaf_cfvs` reference from bean `p99d` remains as the byte-exact comparator.

**Tech Stack:** Rust, rayon (already a workspace dep), cfvnet crate.

**Bean:** `poker_solver_rust-6c5k` (high — unblocks turn datagen throughput target).

**Design doc:** [`docs/plans/2026-04-20-boundary-eval-rayon-design.md`](2026-04-20-boundary-eval-rayon-design.md)

---

## Prerequisites

1. You are running in a dedicated git worktree on a feature branch. If not, stop and ask the coordinator.
2. Layer A (bean `p99d`, commits `7287ff6` and `1939ef0`) is already merged on main. Verify with `git log --oneline -5` — you should see those SHAs (or equivalents post-any-rebase).
3. `rayon` is already in `crates/cfvnet/Cargo.toml` as `rayon = { workspace = true }`. Do NOT add it again.
4. The ONNX BoundaryNet model exists at `local_data/models/cfvnet_river_py_v2/model.onnx` — required for the regression tests.
5. **Read the design doc in full before touching code.** It explains why byte-identical output is safe to assert (only parallelizing across independent work units).
6. **Read the existing Layer A regression test** at the bottom of `crates/cfvnet/src/datagen/domain/pipeline.rs` tests module (`layer_a_batched_matches_per_sit_within_tolerance`). It is the model for your new Layer B+C regression test.
7. **Read the existing `per_sit_batch_boundary_leaf_cfvs` helper** in the same file. This is the serial reference the new test will compare against after both Layer B and Layer C land. It must NOT be modified.

---

## Task 1: Layer C — rayon the reduction in `evaluate_boundaries_batched`

**Goal:** Parallelize the per-request f64 reduction. Refactor the sequential `row_cursor` into pre-computed per-request offsets, then `into_par_iter().map().collect()` over request indices. Output byte-identical.

**File:** `crates/cfvnet/src/datagen/gpu_boundary_eval.rs:203-272`.

### Step 1.1: Read the current reduction loop

Read `crates/cfvnet/src/datagen/gpu_boundary_eval.rs:203-272`. Key things:
- The sequential `row_cursor` at lines 205, 215-218 threads state through all requests.
- Each request's output (`leaf_cfv_p0`, `leaf_cfv_p1`) is independent.
- `meta` comes from `request_meta[req_idx]` — a `&Vec<BoundaryMeta>` built earlier in the function, still serial at this stage.
- `hand_combo_indices`, `hand_cards`, `all_outputs` are read-only.

### Step 1.2: Replace the reduction block

Replace `crates/cfvnet/src/datagen/gpu_boundary_eval.rs:203-269` (from the `// Reduce:` comment through the closing `}` of the `for (req_idx, req) in requests.iter().enumerate()` loop, inclusive of the `results.push(...)` call at 265-268) with:

```rust
    // Pre-compute per-request row offsets. Each request consumes
    // `num_boundaries * num_rivers * 2` rows from `all_outputs`
    // (two players per boundary). Offsets replace the sequential
    // `row_cursor` state so the reduction can run in parallel.
    let row_offsets: Vec<usize> = request_meta
        .iter()
        .zip(requests.iter())
        .scan(0usize, |acc, (meta, req)| {
            let off = *acc;
            *acc += req.num_boundaries * meta.valid_rivers.len() * 2;
            Some(off)
        })
        .collect();

    // Reduce: average over rivers, weighted by opponent reach. One result
    // per request. Parallelized across requests — per-element arithmetic
    // is unchanged, so output is byte-identical to the serial version.
    use rayon::prelude::*;
    let results: Vec<BoundaryEvalResult> = (0..requests.len())
        .into_par_iter()
        .map(|req_idx| {
            let req = &requests[req_idx];
            let meta = &request_meta[req_idx];
            let num_rivers = meta.valid_rivers.len();
            let row_base = row_offsets[req_idx];

            let mut leaf_cfv_p0: Vec<f32> =
                Vec::with_capacity(req.num_boundaries * num_hands);
            let mut leaf_cfv_p1: Vec<f32> =
                Vec::with_capacity(req.num_boundaries * num_hands);

            for bi in 0..req.num_boundaries {
                let p0_row_start = row_base + bi * num_rivers * 2;
                let p1_row_start = p0_row_start + num_rivers;

                let denorm = f64::from(req.pot + req.effective_stack);

                for (hi, &combo_idx) in hand_combo_indices.iter().enumerate() {
                    let (c0, c1) = hand_cards[hi];

                    // Player 0: opponent is IP.
                    let mut cfv_sum_p0 = 0.0_f64;
                    let mut weight_sum_p0 = 0.0_f64;
                    for (ri, &river) in meta.valid_rivers.iter().enumerate() {
                        if c0 == river || c1 == river {
                            continue;
                        }
                        let row = p0_row_start + ri;
                        let net_val = f64::from(all_outputs[row * NUM_COMBOS + combo_idx]);
                        cfv_sum_p0 += meta.ip_weights[bi][ri] * net_val;
                        weight_sum_p0 += meta.ip_weights[bi][ri];
                    }
                    let cfv_p0 = if weight_sum_p0 > 0.0 {
                        (cfv_sum_p0 / weight_sum_p0) * denorm
                    } else {
                        0.0
                    };
                    leaf_cfv_p0.push(cfv_p0 as f32);

                    // Player 1: opponent is OOP.
                    let mut cfv_sum_p1 = 0.0_f64;
                    let mut weight_sum_p1 = 0.0_f64;
                    for (ri, &river) in meta.valid_rivers.iter().enumerate() {
                        if c0 == river || c1 == river {
                            continue;
                        }
                        let row = p1_row_start + ri;
                        let net_val = f64::from(all_outputs[row * NUM_COMBOS + combo_idx]);
                        cfv_sum_p1 += meta.oop_weights[bi][ri] * net_val;
                        weight_sum_p1 += meta.oop_weights[bi][ri];
                    }
                    let cfv_p1 = if weight_sum_p1 > 0.0 {
                        (cfv_sum_p1 / weight_sum_p1) * denorm
                    } else {
                        0.0
                    };
                    leaf_cfv_p1.push(cfv_p1 as f32);
                }
            }

            BoundaryEvalResult {
                leaf_cfv_p0,
                leaf_cfv_p1,
            }
        })
        .collect();
```

**Critical correctness notes:**
- The inner arithmetic is identical to the serial version — same order of multiplies and adds, same f64 intermediate precision, same `as f32` cast.
- The `row_offsets` scan reproduces the serial `row_cursor` exactly: it starts at 0 and advances `num_boundaries * num_rivers * 2` per request. The two `num_rivers` increments inside the old loop (one for p0, one for p1) collectively move the cursor forward by `2 * num_rivers` per boundary — exactly what the scan computes cumulatively per request.
- For each boundary within a request, `p0_row_start = row_base + bi * num_rivers * 2` and `p1_row_start = p0_row_start + num_rivers`. Check this against the old code: in the old loop, `p0_row_start = row_cursor` before the p0 block increments by `num_rivers`, then `p1_row_start = row_cursor` before the p1 block increments by `num_rivers`. So for boundary 0: `p0=0, p1=num_rivers`. For boundary 1: `p0=2*num_rivers, p1=3*num_rivers`. General: `p0 = bi * 2 * num_rivers`, `p1 = p0 + num_rivers`. The new formula matches.
- The `use rayon::prelude::*;` import goes at the local scope (inside the function). If the file already imports rayon prelude, remove the local `use` — grep first: `grep -n "rayon::prelude" crates/cfvnet/src/datagen/gpu_boundary_eval.rs`. If absent, keep the local `use`.

### Step 1.3: Verify it compiles

Run:
```bash
cargo check -p cfvnet --features gpu-turn-datagen 2>&1 | tail -10
```
Expected: clean. Likely warnings if `use rayon::prelude::*;` is unused (it isn't — `into_par_iter`/`map`/`collect` come from there).

### Step 1.4: Run the existing Layer A regression test

The Layer A test (`layer_a_batched_matches_per_sit_within_tolerance`) compares the production `batch_boundary_leaf_cfvs` (which calls `evaluate_boundaries_batched`) against the per-sit reference. After Task 1 the production reduction runs in parallel, so if our refactor is correct, this test must still pass.

```bash
cargo test -p cfvnet --features gpu-turn-datagen layer_a_batched_matches_per_sit_within_tolerance 2>&1 | tail -10
```

Expected: PASS. Runtime similar to before (still runs both paths).

**If it fails:**
- Read the error — it prints the first mismatched index and values.
- Most likely cause: the `row_offsets` scan has an off-by-one or the new `p0_row_start` / `p1_row_start` formula is wrong. Re-read the "Critical correctness notes" in Step 1.2 and check the arithmetic.
- Do NOT loosen the tolerance. The refactor must produce exactly equivalent output.

### Step 1.5: Commit

```bash
git add crates/cfvnet/src/datagen/gpu_boundary_eval.rs
git commit -m "$(cat <<'EOF'
feat(cfvnet): rayon-parallelize per-request reduction (p99d Layer C, bean 6c5k)

Replaces the serial `for (req_idx, req) in requests.iter().enumerate()`
reduction in evaluate_boundaries_batched with `into_par_iter().map().collect()`
over request indices. The sequential `row_cursor` is pre-computed into a
`row_offsets: Vec<usize>` via a scan so each request's reduction is fully
independent.

Per-element arithmetic is unchanged — each request's inner loops run the
same f64 multiply-adds in the same order — so output is byte-identical
to the serial version. The existing Layer A regression test (which
compares the batched production path against the per-sit serial reference)
continues to pass.

Bean: poker_solver_rust-6c5k

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Layer B — rayon the per-sit prep in `batch_boundary_leaf_cfvs`

**Goal:** Parallelize the outer per-sit loop in `batch_boundary_leaf_cfvs` over the canonical batch. Each sit does `compute_reach_at_nodes` (pure), reshape to 1326 (pure), build `BoundaryEvalRequest` (pure) — no shared mutable state.

**File:** `crates/cfvnet/src/datagen/domain/pipeline.rs` (the `batch_boundary_leaf_cfvs` function, post-Layer-A).

### Step 2.1: Locate the current function

After Layer A the function body has this shape (verify by reading `batch_boundary_leaf_cfvs` starting at the line currently around `:746` — use `grep -n "fn batch_boundary_leaf_cfvs" crates/cfvnet/src/datagen/domain/pipeline.rs` to find the exact line):

```rust
let mut requests: Vec<BoundaryEvalRequest> = Vec::with_capacity(sits.len());
for (b, sit) in sits.iter().enumerate() {
    let spec = &specs[b];
    let strategy_sum = &strategy_sums[b];
    let reach = compute_reach_at_nodes(...);
    // reshape to 1326
    requests.push(BoundaryEvalRequest { ... });
}
let results = evaluate_boundaries_batched(evaluator, &requests, canonical_hand_cards)?;
// copy-back loop
```

### Step 2.2: Replace the per-sit loop

Replace the `for (b, sit) in sits.iter().enumerate() { ... }` block (from the `let mut requests: Vec<BoundaryEvalRequest> = Vec::with_capacity(sits.len());` line through its closing `}` — **DO NOT** touch the `evaluate_boundaries_batched` call or the copy-back loop below it) with:

```rust
        use rayon::prelude::*;

        // Build one BoundaryEvalRequest per sit in parallel. Each sit's work
        // (reach computation, 1326-reshape, request struct) is independent.
        // Per-element arithmetic is unchanged vs the serial loop.
        let requests: Vec<BoundaryEvalRequest> = (0..sits.len())
            .into_par_iter()
            .map(|b| {
                let sit = &sits[b];
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
                BoundaryEvalRequest {
                    board: board_4,
                    pot: sit.pot as f32,
                    effective_stack: sit.effective_stack as f32,
                    oop_reach: oop_reach_1326,
                    ip_reach: ip_reach_1326,
                    num_boundaries,
                }
            })
            .collect();
```

**Key correctness notes:**
- `compute_reach_at_nodes` is pure (takes `&TreeTopology` read-only, allocates its own output). Confirmed in `crates/gpu-range-solver/src/batch.rs:765-790`.
- `(0..sits.len()).into_par_iter()` preserves ordering via `collect::<Vec<_>>()`. `requests[b]` corresponds to `sits[b]`.
- The closure captures `topo`, `num_hands`, `boundary_node_ids`, `num_boundaries` by immutable reference — all `Send + Sync` (topology types contain `Vec<usize>`, primitives, no interior mutability).
- Do NOT change the `debug_assert_eq!(num_hands, NUM_COMBOS)` — it's inside the closure now; rayon will happily let it fire per worker.
- If `grep` shows `rayon::prelude` is already imported at the top of `pipeline.rs` or in the enclosing `impl`, you may omit the local `use rayon::prelude::*;`. Check first.

### Step 2.3: Verify it compiles

```bash
cargo check -p cfvnet --features gpu-turn-datagen 2>&1 | tail -10
```
Expected: clean. If there are `Send`/`Sync` errors, something in the closure captures is not thread-safe — verify `topo: &gpu_range_solver::extract::TreeTopology` is `Sync` (Vec fields → yes; any `Rc`/`RefCell` → no). If blocked, stop and report.

### Step 2.4: Run the existing Layer A regression test

```bash
cargo test -p cfvnet --features gpu-turn-datagen layer_a_batched_matches_per_sit_within_tolerance 2>&1 | tail -10
```

Expected: PASS. If it fails here but passed after Task 1, the Layer B refactor changed a per-sit value somewhere — re-read the diff of your changes against the pre-refactor code to find any unintended edit.

### Step 2.5: Commit

```bash
git add crates/cfvnet/src/datagen/domain/pipeline.rs
git commit -m "$(cat <<'EOF'
feat(cfvnet): rayon-parallelize per-sit prep in batch_boundary_leaf_cfvs (Layer B, bean 6c5k)

Replaces the serial `for (b, sit) in sits.iter().enumerate()` loop that
builds per-sit BoundaryEvalRequests with `into_par_iter().map().collect()`
across sits. Each sit's work (compute_reach_at_nodes + 1326-reshape +
request struct) is pure and independent.

Output ordering preserved — `requests[b]` still corresponds to `sits[b]`.
Per-element arithmetic unchanged. The Layer A regression test continues
to assert numerical equivalence vs the serial reference and still passes.

Bean: poker_solver_rust-6c5k

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Byte-identical regression test for Layer B+C combined

**Goal:** A dedicated test that asserts the post-refactor production path (with both B and C parallelized) produces output **byte-identical** to the serial reference `per_sit_batch_boundary_leaf_cfvs`. Stronger than Layer A's tolerance because we're only parallelizing across independent units — per-element math is unchanged.

**File:** `crates/cfvnet/src/datagen/domain/pipeline.rs` tests module. Add next to the existing `layer_a_batched_matches_per_sit_within_tolerance`.

### Step 3.1: Write the test

Model on `layer_a_batched_matches_per_sit_within_tolerance` (same setup: load ONNX model, build canonical topology, construct 4 deterministic sits/specs/strategy_sums). Append this test in the same `mod gpu_turn_tests` (or whichever module contains the Layer A test):

```rust
#[test]
#[cfg(feature = "gpu-turn-datagen")]
fn layer_bc_parallel_matches_serial_exact() {
    // Regression test for bean 6c5k (Layer B+C). Unlike Layer A's tolerance-
    // based test, this asserts byte-identical output — we only parallelize
    // across independent sits/requests, so per-element arithmetic is unchanged.
    //
    // Uses the same setup as layer_a_batched_matches_per_sit_within_tolerance.
    // If you refactor that test's setup into a helper, reuse it here.

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

    // --- Setup block: copy verbatim from layer_a_batched_matches_per_sit_within_tolerance ---
    // (topology, num_hands, boundary_node_ids, 4 sits, strategy_sums, specs, hand_cards)
    // If that test extracted this into a helper `fn make_regression_inputs(...)`,
    // call it. Otherwise copy the construction here.

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
    .expect("parallel path");

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
    .expect("serial reference path");

    assert_eq!(batched_p0.len(), reference_p0.len());
    assert_eq!(batched_p1.len(), reference_p1.len());

    let check_bitwise_eq = |parallel: &[f32], serial: &[f32], label: &str| {
        for (i, (&p, &s)) in parallel.iter().zip(serial).enumerate() {
            assert!(
                p.to_bits() == s.to_bits(),
                "{label}[{i}] parallel={p} (bits={:#010x}) serial={s} (bits={:#010x})",
                p.to_bits(), s.to_bits()
            );
        }
    };

    check_bitwise_eq(&batched_p0, &reference_p0, "p0");
    check_bitwise_eq(&batched_p1, &reference_p1, "p1");
}
```

**Implementation notes:**
- If the Layer A test didn't factor its setup into a helper, the cleanest thing is to copy-paste it. Don't refactor the Layer A test as part of this bean — keep the blast radius small. If you're careful, a small private `fn make_regression_inputs()` helper is OK, but it's optional.
- `f32::to_bits()` compares NaN-bit-patterns correctly; for regular values this is the strongest possible equality check.
- `check_bitwise_eq` uses `to_bits()` instead of `==` so that if any NaN slips through, the assertion fails visibly rather than silently returning `NaN != NaN`.

### Step 3.2: Run it — expect PASS

```bash
cargo test -p cfvnet --features gpu-turn-datagen layer_bc_parallel_matches_serial_exact 2>&1 | tail -20
```

Expected: PASS. If it fails with bit-mismatches, investigate:
- Is any inner f64 reduction running in a different order? (Shouldn't be — we only parallelized across outer index.)
- Is any closure capturing by value instead of reference and computing differently? (Unlikely — rayon captures are immutable.)
- The test's serial reference calls `per_sit_batch_boundary_leaf_cfvs`, which runs fully serial. If that helper has been accidentally modified, restore it exactly (check against the version introduced in Layer A's commit `7287ff6`).

If the test's bit-identical assertion fails but Layer A's tolerance-based test still passes, you have a real FP drift (small but non-zero). Report back — do NOT weaken the assertion without understanding why.

### Step 3.3: Commit

```bash
git add crates/cfvnet/src/datagen/domain/pipeline.rs
git commit -m "$(cat <<'EOF'
test(cfvnet): byte-identical regression test for Layer B+C (bean 6c5k)

Asserts that the post-refactor `batch_boundary_leaf_cfvs` (parallelized
per-sit prep + parallelized reduction) produces leaf CFVs bit-identical
to the serial reference `per_sit_batch_boundary_leaf_cfvs`. Uses
`f32::to_bits()` equality — stronger than Layer A's tolerance because
rayon parallelism is across independent work units, so per-element
arithmetic is unchanged.

Bean: poker_solver_rust-6c5k

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Build + clippy

### Step 4.1: Full workspace build

```bash
cargo build --workspace 2>&1 | tail -5
```
Expected: clean.

### Step 4.2: Clippy on changed files

```bash
cargo clippy -p cfvnet --features gpu-turn-datagen -- -D warnings 2>&1 | grep -E "warning:|error:" | grep -E "(gpu_boundary_eval\.rs|pipeline\.rs)" | head -20
```

Expected: no warnings in your two modified files (lines you touched). Pre-existing warnings in those files at lines you did NOT modify are acceptable.

If a new warning appears in your changes, fix it minimally (don't refactor). Common candidates after a rayon refactor:
- `clippy::needless_collect` — rayon's collect-before-next-step pattern may be flagged; silence with `#[allow(clippy::needless_collect)]` *only* if rayon actually requires the materialization.
- `clippy::explicit_counter_loop` — not typically introduced by these changes.

### Step 4.3: Workspace tests

```bash
time cargo test --workspace 2>&1 | tail -10
```

Expected: same 8 pre-existing `mp_tui_scenarios` failures as documented in bean `oox2`. Any NEW failure is a regression from your work — investigate.

Runtime >60s is acknowledged pre-existing per CLAUDE.md; not in scope to fix here.

---

## Task 5: Manual datagen validation

**Goal:** Prove Layer B+C actually unlocks throughput. This is the acceptance criterion.

### Step 5.1: Confirm yaml config

The yaml should have `gpu_batch_size: 4` and the full `solver_iterations: 300, leaf_eval_interval: 50` to match the p99d baseline. Check:

```bash
grep -E "gpu_batch_size|solver_iterations|leaf_eval_interval" sample_configurations/turn_gpu_datagen.yaml
```

If `gpu_batch_size` is missing, add `gpu_batch_size: 4` under the `datagen:` block. If `bet_size_fuzz` is `0` (set during oox2 validation), leave it — you only care about throughput, not statistical coverage of the datagen.

### Step 5.2: Launch the run

```bash
mkdir -p local_data/cfvnet/turn/6c5k_verify
time cargo run -p cfvnet --release --features gpu-turn-datagen -- generate \
  -c sample_configurations/turn_gpu_datagen.yaml \
  -o local_data/cfvnet/turn/6c5k_verify \
  --num-samples 32 --per-file 8 2>&1 | tee local_data/cfvnet/turn/6c5k_verify/_run.log
```

### Step 5.3: Monitor (optional, in a second terminal)

```bash
# Watch GPU util and main-thread CPU.
while true; do
  rss=$(ps -o rss= -C cfvnet 2>/dev/null | awk '{s+=$1} END {print s+0}')
  main_pcpu=$(ps -C cfvnet -o pcpu= 2>/dev/null | awk 'NR==1{print $1}')
  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "$(date +%T) cfvnet_cpu=${main_pcpu}% gpu=${gpu}%"
  sleep 5
done
```

Note: this monitor shows **total** process CPU % (sum across threads), not per-thread. `cfvnet_cpu` will likely be ~1200% (12 cores × 100%) during the parallelized phases if rayon is working. GPU util should now spend meaningful time >0%.

### Step 5.4: Record results

- Exit code: _____ (must be 0)
- Wall clock: _____ (oox2 baseline ~53 min for 4 files; Layer A ~44 min; target for Layer B+C: <15 min for 4 files, ideally <10 min)
- Files written: _____/4
- Peak RSS: _____ GB
- Observed GPU utilization peak: _____% (was ~0% persistently pre-Layer-B+C)
- Total process CPU %: _____ (should hit >500% at peaks, meaning rayon is using multiple cores)

### Step 5.5: If throughput IS improved

Record the numbers in Task 6's bean summary. Proceed to Task 6.

### Step 5.6: If throughput IS NOT improved

- If wall clock is within ~20% of the Layer A baseline, rayon isn't doing what we expect. Check:
  - `grep "rayon" crates/cfvnet/src/datagen/gpu_boundary_eval.rs crates/cfvnet/src/datagen/domain/pipeline.rs` — confirm the prelude import landed in both files.
  - Use `RAYON_NUM_THREADS=1 time cargo run ...` to force serial rayon, see if it matches the pre-6c5k Layer A time. If it's faster than single-threaded rayon run, rayon's scheduling is fine; bottleneck is elsewhere.
  - If rayon appears to be running but throughput is identical, the real ceiling is probably in the ORT call itself or in `compute_reach_at_nodes`. File follow-up beans; close 6c5k as "parallelization landed, further optimization required."
- If wall clock is WORSE than Layer A, you've introduced rayon overhead at a scale too small to amortize. Unlikely at batch=4+, but possible. Investigate and report.

---

## Task 6: Bean closure

### Step 6.1: Update bean with summary

```bash
beans update --json poker_solver_rust-6c5k -s completed --body-append "$(cat <<'EOF'


## Summary of Changes (Layers B + C)

Rayon-parallelized the two main-thread serial loops identified in bean `p99d` Task 5 as the throughput bottleneck: the per-sit prep in `batch_boundary_leaf_cfvs` (Layer B) and the per-request f64 reduction in `evaluate_boundaries_batched` (Layer C). Both parallelize only across sits/requests, so per-element arithmetic is unchanged — output is byte-identical to the serial reference.

**Plan:** `docs/plans/2026-04-20-boundary-eval-rayon.md`
**Design:** `docs/plans/2026-04-20-boundary-eval-rayon-design.md`

**Commits:**
- feat(cfvnet): rayon-parallelize reduction (Layer C)
- feat(cfvnet): rayon-parallelize per-sit prep (Layer B)
- test(cfvnet): byte-identical regression test for B+C

**Validation:**
- Layer A's tolerance-based regression test still passes (parallel path matches per-sit serial reference within 1e-5/1e-4).
- New byte-identical regression test (`layer_bc_parallel_matches_serial_exact`) passes: parallel output == serial reference to the last bit.
- Manual datagen validation (gpu_batch_size=4, 32 samples, 8/file):
  - oox2 baseline (pre-any-parallelism): ~15 min/file.
  - Layer A only: ~11 min/file.
  - **Layer B+C: <WALL CLOCK> min/file.**
  - Observed peak GPU utilization: <PEAK>% (was ~0% pre-B+C).
  - Observed total CPU %: <CPU> (rayon using multiple cores).

**Follow-up bean needs (file only if validation shows remaining ceiling):**
- Input-building parallelization in `gpu_boundary_eval.rs:121-194` (also serial main-thread today).
- Cache/transpose optimization of the reduction access pattern.
EOF
)"
```

Replace `<WALL CLOCK>`, `<PEAK>`, `<CPU>` with actual measurements from Task 5.

### Step 6.2: Commit the bean update

```bash
git add .beans/poker_solver_rust-6c5k*.md
git commit -m "chore: complete bean 6c5k (boundary eval Layers B+C)"
```

### Step 6.3: Report to coordinator

Final report must include:
- All commit SHAs.
- Before/after wall clock from the manual datagen.
- Observed peak GPU utilization and total CPU % (proves rayon is actually parallelizing).
- Whether turn datagen throughput is now in a production-viable range, or further work is needed (file bullet points for follow-up beans).

---

## Execution guidance

- Work in a dedicated worktree.
- Commit exactly as prescribed — one commit per task (1, 2, 3 each as a separate commit; Task 4's clippy fixes if any go into whichever commit caused them — no separate commit for clippy-only tweaks).
- Do NOT squash.
- Do NOT modify `per_sit_batch_boundary_leaf_cfvs` — it's the serial reference going forward.
- If the regression test fails at any task, STOP and report — do not paper over with a weaker assertion.
- Do not attempt Layer D (transpose optimization) or Layer "A-precursor" (input-building parallelization) in this bean. They are explicitly out of scope.
