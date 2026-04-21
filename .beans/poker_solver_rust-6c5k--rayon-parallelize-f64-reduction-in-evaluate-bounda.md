---
# poker_solver_rust-6c5k
title: Rayon-parallelize f64 reduction in evaluate_boundaries_batched (turn datagen Layer C)
status: completed
type: task
priority: high
created_at: 2026-04-20T22:47:35Z
updated_at: 2026-04-21T01:01:18Z
---

Follow-up to bean \`p99d\` Layer A. The Layer A fix (hoist ORT call out of the per-sit loop) shipped but turn-datagen throughput remains CPU-bound at any batch size.

## Evidence (from p99d Task 5 validation, 2026-04-20)

With Layer A landed:
- Main thread pegged at 100%, 7 ORT intra-op threads at ~30% each.
- GPU utilization 0% for most of wall time.
- batch=4 with \`solver_iterations=30\` (smoke config) produced 0 files after 25 min.
- batch=4 with \`solver_iterations=300\` (production config) produced files in ~11 min each — about 30% faster than the pre-Layer-A oox2 baseline (~15 min/file), not the 5-10× we hoped.

## Root cause (surface level)

Main thread is bottlenecked on the per-request f64 reduction in \`evaluate_boundaries_batched\` at \`crates/cfvnet/src/datagen/gpu_boundary_eval.rs:207-269\`:

\`\`\`
for (req_idx, req) in requests.iter().enumerate() {
    for bi in 0..req.num_boundaries {
        for (hi, &combo_idx) in hand_combo_indices.iter().enumerate() {
            // Player 0: inner river loop doing f64 multiply-adds
            // Player 1: same structure
        }
    }
}
\`\`\`

Size per batch=4 call: roughly \`4 requests × ~30 boundaries × 1326 hands × 48 rivers × 2 players\` ≈ 15M f64 multiply-adds — trivial on GPU, dominant on single-threaded CPU. Layer A made this 4× bigger per call (more requests at once) but didn't parallelize it.

The outer \`for req in requests\` loop is embarrassingly parallel — each request's reduction writes to an independent \`leaf_cfv_p0/p1\` vector.

## Work

- [ ] Rayon \`par_iter_mut\` (or \`par_chunks_mut\`) the outer \`for req in requests\` reduction loop. Each request's output is independent — no shared mutable state.
- [ ] Also consider rayon over the \`for (hi, &combo_idx) in hand_combo_indices.iter().enumerate()\` inner loop if 256 requests × 12 threads is still not enough parallelism (probably fine for 256 sits / 12 threads ≈ 21 each).
- [ ] Add a regression test mirroring Layer A's — asserts parallel reduction produces identical output to serial reduction within tight FP tolerance.
- [ ] Re-run the p99d Task 5 validation and record the actual speedup.

## Also consider (separate work items if Layer C alone isn't enough)

- **Layer B**: Rayon-parallelize the per-sit prep (reach + reshape + input build) in \`batch_boundary_leaf_cfvs\` at \`crates/cfvnet/src/datagen/domain/pipeline.rs:772-823\`. Embarrassingly parallel over sits. May be even bigger than Layer C.
- **Layer D**: The reduction reads \`all_outputs\` in an access pattern that might be cache-unfriendly (row=p0_row_start+ri per hand). A transpose or tiled reduction could reduce memory stalls.

## Priority

High — turn datagen is still throughput-blocked. Without Layer C/B, the 50M-in-1h target is unreachable no matter how big \`gpu_batch_size\` is.



## Summary of Changes

Shipped Layers B and C from the design: rayon-parallelized (B) the per-sit prep in \`batch_boundary_leaf_cfvs\` and (C) the per-request f64 reduction in \`evaluate_boundaries_batched\`. Both parallelize only across sits/requests — per-element arithmetic unchanged — so output is byte-identical to the serial reference.

**Plan:** \`docs/plans/2026-04-20-boundary-eval-rayon.md\`
**Design:** \`docs/plans/2026-04-20-boundary-eval-rayon-design.md\`

**Commits on main:**
- \`187d047\` feat(cfvnet): rayon-parallelize per-request reduction (Layer C)
- \`153c510\` feat(cfvnet): rayon-parallelize per-sit prep (Layer B)
- \`bc444f9\` test(cfvnet): byte-identical regression test for Layer B+C

**Correctness validation:**
- Layer A's tolerance regression test (\`layer_a_batched_matches_per_sit_within_tolerance\`): PASS post-refactor.
- New byte-identical regression test (\`layer_bc_parallel_matches_serial_exact\`) using \`f32::to_bits()\` equality: PASS.
- Full workspace test suite: same 2 pre-existing \`poker-solver-core::blueprint_mp\` timer-sensitive failures, unchanged.

**Throughput validation at gpu_batch_size=4** (matches p99d Task 5 baseline):
- Wall clock: 45m35s (32 samples, 4 files × 8 samples).
- Files written: 4/4, 276 KB each.
- CPU %: peaked at 404% (pre-6c5k baseline 313%) — rayon confirmed parallelizing.
- GPU util: peak 100% (pre-6c5k baseline ~0% persistent) — GPU now actually exercising DCFR.
- Peak RSS: 52.75 GB.

**Interpretation:** wall clock unchanged vs Layer A alone. At batch=4 the DCFR GPU kernel iterations dominate the per-batch time, so parallelizing CPU prep/reduction moves the bottleneck but doesn't shrink the critical path. Layer B+C is expected to scale better at larger batch, where CPU work becomes a larger share per batch. Filed \`krhd\` to measure at batch=32 and 256.

**Unblocks:**
- \`vb8r\` end-to-end validation — now numerically and functionally correct with rayon in the hot path.
- \`krhd\` — throughput scaling measurement.
