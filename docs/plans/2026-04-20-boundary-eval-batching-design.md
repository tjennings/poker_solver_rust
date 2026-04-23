# Design — Batch boundary evaluation across the canonical-topology batch (Layer A)

**Date:** 2026-04-20
**Bean:** `poker_solver_rust-p99d`
**Status:** Approved, ready for implementation plan

## Problem

Turn datagen wall-clock is dominated by ORT / TensorRT per-call dispatch overhead. Inside each DCFR eval round, `batch_boundary_leaf_cfvs` runs a `for sit in sits` loop that calls `evaluate_boundaries_batched(evaluator, &[single_request], ...)` once per sit. At `gpu_batch_size=256` this is 256 batch-of-1 ORT calls per round; with `solver_iterations=300, leaf_eval_interval=50` (~6 rounds/batch), that's ~1,536 batch-of-1 ORT calls before the first batch completes.

Symptom observed during bean `oox2`'s Task 7 validation: GPU compute 0% despite 40 GB GPU memory allocated; main thread pegged at 100%, 7 ORT intra-op threads at ~48%; at `batch=4` a single file (8 samples) takes ~13 minutes, so ~1,625 s/sample — nowhere near the 50M-in-1h target.

`evaluate_boundaries_batched` (in `crates/cfvnet/src/datagen/gpu_boundary_eval.rs:109-278`) is **already** capable of batching — it takes `&[BoundaryEvalRequest]`, builds one combined input tensor at lines 121-194, and calls `infer_batch` ONCE at line 198. The call site never exercises that path: it always passes `&[single_request]`.

## Scope & non-goals

**In scope (Layer A only):**

- Rearrange `batch_boundary_leaf_cfvs` in `crates/cfvnet/src/datagen/domain/pipeline.rs:746-826` so the ORT call runs once over all sits' requests, not once per sit.
- Add a regression test verifying the batched-once path produces leaf CFVs matching the per-sit path within tight FP tolerance (`1e-5` absolute, `1e-4` relative).

**Out of scope (deferred):**

- Rayon-parallelizing the per-sit prep (reach computation, 1326-reshape, input building). Only tackle if Layer A alone doesn't hit throughput target — revisit after measuring.
- Rayon-parallelizing the reduction pass in `gpu_boundary_eval.rs:207-269`. Same rationale.
- Any change to `evaluate_boundaries_batched`'s internal structure — the function is already batch-correct.
- Measuring / tuning TensorRT engine cache behavior.
- Unrelated CPU optimizations (`zero_conflicting_hands`, reach reshaping).

## Approach (single option — Layer A)

### Change shape

**Before** (`pipeline.rs:772-823`):

```
for b in 0..batch_size:
  reach = compute_reach_at_nodes(spec[b], strategy_sums[b])
  reshape reach to 1326-combo layout
  req = BoundaryEvalRequest { ...sit[b] }
  results = evaluate_boundaries_batched(eval, &[req], hand_cards)  // batch=1
  copy results[0].leaf_cfv_{p0,p1} into batched_{p0,p1}[slot(b)]
```

**After:**

```
Vec<BoundaryEvalRequest> requests
for b in 0..batch_size:
  reach = compute_reach_at_nodes(spec[b], strategy_sums[b])
  reshape reach to 1326-combo layout
  requests.push(BoundaryEvalRequest { ...sit[b] })

results = evaluate_boundaries_batched(eval, &requests, hand_cards)  // batch=256
for b in 0..batch_size:
  copy results[b].leaf_cfv_{p0,p1} into batched_{p0,p1}[slot(b)]
```

### Files / functions

- **Modified:** `crates/cfvnet/src/datagen/domain/pipeline.rs:746-826` (`batch_boundary_leaf_cfvs`).
- **No change:** `crates/cfvnet/src/datagen/gpu_boundary_eval.rs` — the function already handles N requests; call site is the only bug.
- **Modified:** `crates/cfvnet/src/datagen/domain/pipeline.rs` tests module — new regression test.

### Numerical equivalence

ONNX Runtime inference on a batch of N rows is mathematically identical to N inference calls with 1 row each — output row `i` depends only on input row `i`. However, TensorRT may pick a different CUDA kernel for batch=12288 rows than for batch=48 rows, and different kernels can round differently at the last ULP. Regression test (see Testing) establishes the acceptable drift.

### Memory

Combined input tensor grows from `(2 × 48) × INPUT_SIZE × 4` (≈ one sit's worth) to `(256 × 2 × 48) × INPUT_SIZE × 4` (256 sits' worth). For typical BoundaryNet `INPUT_SIZE` on the order of 1000-2000 floats, that's ~100 MB host-side before H2D, comfortably within the 120 GB host budget. Will be measured in Task 8 of the implementation plan.

## Testing

1. **Numerical regression test (new):** `layer_a_batched_matches_per_sit_within_tolerance` in `crates/cfvnet/src/datagen/domain/pipeline.rs` tests module, gated `#[cfg(feature = "gpu-turn-datagen")]`.
   - Build canonical turn topology + load real ONNX model.
   - Construct 4 sits with deterministic seeds.
   - Compute `leaf_cfv` two ways:
     - A) New path: call `batch_boundary_leaf_cfvs` with all 4 sits.
     - B) Old path: build a private test helper that replicates the pre-Layer-A loop (batch-of-1 calls), call it with the same 4 sits.
   - Assert element-wise: `|a - b| <= 1e-5 + 1e-4 * |b|`.

2. **Existing smoke test untouched:** `canonical_turn_tree_runs_one_iteration_without_smem_overflow` still runs a full DCFR iteration — now exercises the batched path. Continues to pass.

3. **Manual validation:** re-run the `oox2` Task 7 command (`cargo run -p cfvnet --release --features gpu-turn-datagen -- generate -c sample_configurations/turn_gpu_datagen.yaml -o <out> --num-samples 32 --per-file 8`). Record wall clock before/after. Baseline from oox2: 53 min for 4 files. Target post-Layer-A: <20 min for 4 files (rough — Layer A kills 256× dispatch overhead in the critical path).

## Manual validation (required before closing the bean)

Both:
- Numerical test passes (batched output == per-sit output within tolerance).
- Datagen command produces 4 files (32 samples, 64 records) with exit 0, wall time materially improved over the `oox2` baseline.

## Risks

- **TensorRT engine re-compile on first big batch.** Switching from batch=N to batch=256×N will likely trigger a one-time TRT engine rebuild at startup (may add 30-120s). Expected and amortized. Record actual startup time in the bean summary.
- **FP drift exceeds tolerance.** If TRT picks meaningfully different kernels and the regression test trips, we need to either (a) tighten-and-document the tolerance, or (b) investigate if any reduction in `evaluate_boundaries_batched` is order-sensitive. Mitigation: run with a looser tolerance first (`1e-3 relative`) to see the actual drift magnitude, then tighten.
- **Host memory pressure** at large batches. Combined input tensor growth is ~256× but started small, so absolute number should stay under ~1 GB. Not expected to be an issue, but Task 8 measures RSS and confirms.
- **Under-estimated dispatch amortization.** If Layer A speedup is smaller than expected (<2×), the real bottleneck was the CPU prep loop, not ORT dispatch — in which case Layer B becomes priority. Measurement will tell us.

## Follow-ups (separate beans if needed, not filed preemptively)

1. Layer B — parallelize per-sit reach + reshape + request building with rayon.
2. Layer C — parallelize the reduction pass in `evaluate_boundaries_batched`.
3. TensorRT engine cache tuning — make the first-batch warmup cost persistent across runs via ORT's EP options.
