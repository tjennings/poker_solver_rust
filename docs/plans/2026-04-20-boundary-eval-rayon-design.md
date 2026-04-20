# Design — Rayon-parallelize boundary-eval CPU loops (Layers B + C)

**Date:** 2026-04-20
**Bean:** `poker_solver_rust-6c5k`
**Status:** Approved, ready for implementation plan

## Problem

After bean `p99d` (Layer A — single batched ORT call) landed, turn datagen throughput improved by only ~30% at `gpu_batch_size=4`. The user observed main thread pegged at 100% CPU while GPU utilization sat at 0% for most wall-clock. The Layer A change killed the dispatch-overhead in the ORT call but left the surrounding CPU work fully serial on the main thread.

Two serial loops dominate main-thread time:

1. **`batch_boundary_leaf_cfvs`** at `crates/cfvnet/src/datagen/domain/pipeline.rs:772-823`. For each sit: `compute_reach_at_nodes` (CPU tree walk), 1326-combo reshape, `BoundaryEvalRequest` construction. Embarrassingly parallel over sits.

2. **`evaluate_boundaries_batched`** reduction at `crates/cfvnet/src/datagen/gpu_boundary_eval.rs:207-269`. Per request: nested loops computing reach-weighted average of per-river NN outputs, in f64, for `num_boundaries × num_hands × num_rivers × 2 players` elements. Each request writes to its own `leaf_cfv_p0/p1` vectors — embarrassingly parallel over requests.

At `batch=4`, per-batch main-thread work is roughly:
- Reach + reshape + request build: 4 sits × (tree walk + O(num_boundaries × 1326)) = seconds per batch.
- Reduction: 4 requests × ~30 boundaries × 1326 hands × 48 rivers × 2 players ≈ 15M f64 ops per ORT call × 6 eval rounds per batch = ~90M f64 ops.

On 12 cores, rayon parallelism over sits should reduce both to ~O(max / 12).

## Scope & non-goals

**In scope:**

- Parallelize the outer `for (b, sit)` loop in `batch_boundary_leaf_cfvs` with rayon (Layer B).
- Parallelize the outer `for (req_idx, req)` reduction loop in `evaluate_boundaries_batched` with rayon (Layer C).
- Regression test asserting byte-identical output vs pre-refactor serial reference.

**Out of scope:**

- Parallelizing the inner per-hand or per-river loops inside each request's reduction. One level of parallelism over sits is likely enough at batch ≥ 12 (our thread count); if not, file follow-up.
- Rayon-parallelizing the ORT input-building loop at `gpu_boundary_eval.rs:121-194`. That loop is likely bound by `encode_boundary_inference_input` + `zero_conflicting_hands` CPU work — parallelizable but separate from this bean. File follow-up if measurement shows it as the new ceiling.
- Cache/transpose optimizations (Layer D).
- Any DCFR kernel changes.

## Approach

Two independent rayon migrations, both preserving byte-identical output.

### Layer B — `batch_boundary_leaf_cfvs`

**Before** (`pipeline.rs:772-823`, per-sit loop builds one request):

```
Vec<BoundaryEvalRequest> requests
for (b, sit) in sits.iter().enumerate():
    reach = compute_reach_at_nodes(spec[b], strategy_sums[b])
    reshape to 1326
    requests.push(BoundaryEvalRequest { ...sit[b] })
```

**After:**

```
let requests: Vec<BoundaryEvalRequest> = (0..sits.len())
    .into_par_iter()
    .map(|b| {
        let reach = compute_reach_at_nodes(...);
        // reshape
        BoundaryEvalRequest { ... }
    })
    .collect();
```

`compute_reach_at_nodes` is pure, takes `&TreeTopology` (shared read, no interior mutability), allocates its own outputs — safe for concurrent calls. Confirmed by reading `crates/gpu-range-solver/src/batch.rs:765-790`.

### Layer C — `evaluate_boundaries_batched` reduction

**Before** (`gpu_boundary_eval.rs:203-269`, serial `row_cursor` threading through all requests):

```
let mut results: Vec<BoundaryEvalResult> = Vec::with_capacity(requests.len());
let mut row_cursor = 0usize;
for (req_idx, req) in requests.iter().enumerate() {
    for bi in 0..req.num_boundaries {
        let p0_row_start = row_cursor; row_cursor += num_rivers;
        let p1_row_start = row_cursor; row_cursor += num_rivers;
        // reduce → push into leaf_cfv_p0/p1
    }
    results.push(BoundaryEvalResult { leaf_cfv_p0, leaf_cfv_p1 });
}
```

**After:** precompute per-request row offsets once, then `into_par_iter`:

```
// Each request consumes num_boundaries * num_rivers * 2 rows.
let row_offsets: Vec<usize> = requests
    .iter()
    .zip(&request_meta)
    .scan(0usize, |acc, (req, meta)| {
        let off = *acc;
        *acc += req.num_boundaries * meta.valid_rivers.len() * 2;
        Some(off)
    })
    .collect();

let results: Vec<BoundaryEvalResult> = (0..requests.len())
    .into_par_iter()
    .map(|req_idx| {
        let req = &requests[req_idx];
        let meta = &request_meta[req_idx];
        let row_base = row_offsets[req_idx];
        // same inner loops as before, reading all_outputs[row_base + ...]
        // produces owned BoundaryEvalResult
    })
    .collect();
```

Because each request's output is computed independently with identical arithmetic to the serial version, output is byte-identical.

### Files touched

- Modified: `crates/cfvnet/src/datagen/domain/pipeline.rs:772-823` (Layer B refactor; preserve the test-only `per_sit_batch_boundary_leaf_cfvs` reference as-is — it's the serial baseline).
- Modified: `crates/cfvnet/src/datagen/gpu_boundary_eval.rs:203-269` (Layer C refactor).
- Modified: `crates/cfvnet/src/datagen/domain/pipeline.rs` tests module — one new regression test asserting byte-identical output across the parallel and serial paths.
- Possibly modified: `crates/cfvnet/src/datagen/gpu_boundary_eval.rs` tests module — if existing tests assume the serial loop order, verify they still pass.

### Rayon thread pool

Use the default global pool. rayon defaults to all physical cores (12 on this workstation). The ORT intra-op pool (7-8 threads observed) runs during `session.run`, which is a different phase from the rayon-parallelized prep and reduction — no temporal contention. Do NOT configure a custom pool.

## Testing

1. **Byte-identical regression test (new):** `layer_bc_parallel_matches_serial_exact` in `crates/cfvnet/src/datagen/domain/pipeline.rs` tests module, gated `#[cfg(feature = "gpu-turn-datagen")]`.
   - Build canonical turn topology, load real ONNX model.
   - 4 deterministic sits (same setup as Layer A's regression test).
   - Call `batch_boundary_leaf_cfvs` (parallel path, production).
   - Call the existing test-only `per_sit_batch_boundary_leaf_cfvs` (serial reference, from Layer A).
   - Assert each output f32 is bit-equal via `v.to_bits() == r.to_bits()` — stronger than tolerance because we're only parallelizing across independent work, so per-element arithmetic is unchanged.

2. **Layer A's existing `layer_a_batched_matches_per_sit_within_tolerance` still passes.** The serial reference it compares against is untouched; the production path now runs reduction in parallel but produces same values. Pre-refactor and post-refactor behavior is numerically identical.

3. **Smoke test `canonical_turn_tree_runs_one_iteration_without_smem_overflow` still passes.**

## Manual validation (required before closing the bean)

Re-run the p99d Task 5 command (`cargo run -p cfvnet --release --features gpu-turn-datagen -- generate -c sample_configurations/turn_gpu_datagen.yaml -o <out> --num-samples 32 --per-file 8`) at `gpu_batch_size=4`.

**Baselines for comparison:**
- oox2 baseline (pre-Layer A): ~15 min/file.
- p99d Layer A: ~11 min/file.

**Success criteria:**
- Manual datagen completes in materially less than 11 min/file — target <3 min/file at batch=4. At batch=32 should be even better (per-sit work gets more parallelism per rayon batch).
- GPU utilization visibly non-zero for a meaningful fraction of wall time (was ~0% pre-Layer B+C).
- Main thread no longer pegged at 100% for most of the run (should drop to ~50% range as work distributes).

## Risks

- **Rayon dispatch overhead at tiny batches.** ~µs per `par_iter` call; at batch=1 this is pure waste. batch_size=1 isn't a realistic production config, but any test that uses batch=1 may need to opt out. Mitigation: not a concern for production paths.
- **Thread oversubscription during transition moments.** If ORT and rayon ever run simultaneously (they don't today), total threads could exceed 12. Mitigation: the ORT `session.run` is a single call that the rayon-parallel reduction runs *after*, not during. Verified by reading `evaluate_boundaries_batched`'s sequential structure.
- **Allocation churn in `into_par_iter().map().collect()`.** Each rayon worker builds its own `BoundaryEvalResult` with its own `Vec<f32>`s, then rayon collects them in order. One alloc per request — no worse than the current per-request `Vec::with_capacity(...)`. No regression.
- **Byte-identical assertion may be violated by compiler re-ordering across opt levels.** Extremely unlikely — the inner reduction loops do pairwise f64 multiply-adds in a deterministic nested order, and rayon preserves ordering at the `.collect()` level. If the test ever fails on a non-canonical target, revisit tolerance.
- **Layer B and Layer C combined bigger change.** More code churn per commit. Mitigation: split into two atomic commits in the implementation plan (Layer B, then Layer C), regression test after each.

## Follow-ups (not filed preemptively)

1. If measurement shows the ORT input-building loop (`gpu_boundary_eval.rs:121-194`) as the new ceiling, file a bean for Layer D (rayon it). Likely small win but plausible.
2. If `compute_reach_at_nodes` itself becomes the bottleneck per-sit (unlikely but possible on very deep trees), consider a GPU implementation.
3. TensorRT engine cache persistence — unrelated, carried over from the Layer A design doc.
