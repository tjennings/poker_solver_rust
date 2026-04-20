# Design — Skip all-zero showdown outcomes allocation in GPU turn datagen

**Date:** 2026-04-20
**Bean:** `poker_solver_rust-vb8r`
**Status:** Approved, ready for implementation plan

## Problem

`cargo run -p cfvnet --release --features gpu-turn-datagen -- generate -c sample_configurations/turn_gpu_datagen.yaml` OOMs even at `gpu_batch_size: 32`:

- Host RAM OOM (global_oom, not GPU, not cgroup-constrained)
- RSS at kill: 127.6 GB (`total-vm: 407 GB, anon-rss: 127584260 kB`)
- Blocks production turn datagen — cannot reach the planned 50M-sample target

## Root cause (confirmed by code-explorer)

Two host-side copies of an all-zero buffer:

1. **`pipeline.rs:1015-1021`** (`build_turn_subgame_spec`): `vec![0.0_f32; num_showdowns * 1326 * 1326]`, cloned twice into `SubgameSpec.showdown_outcomes_p0/p1`. ~225 MB per player per spec at `num_showdowns=24`.
2. **`batch.rs:322-325`** (`prepare_batch`): per-spec vectors concatenated into flat host `Vec<f32>` of size `batch_size × num_showdowns × 1326²`, then `clone_htod`'d to GPU. ~14.4 GB per player per batch at `batch_size=32, num_showdowns=24`.

For turn datagen the values are always zero: `build_turn_subgame_spec` fills them with zeros and leaf injection from BoundaryNet supplies the real CFVs at boundary nodes via an independent GPU buffer (`d_leaf_cfv_p0/p1`, allocated in `batch.rs:set_leaf_injection`). The showdown-payoff kernel still executes over those zeros, producing a zero contribution — wasted GPU work atop the wasted allocation.

River datagen legitimately uses the buffer — `SubgameSpec::from_game` → `build_mega_terminal_data` expands real `outcome_matrix_p0` values that the kernel multiplies through.

## Scope & non-goals

**In scope:**

- Eliminate the allocation, host concat, H2D upload, and GPU compute for the showdown outcomes buffer in the turn-datagen path.
- Keep the river-datagen path byte-identical in behavior.

**Out of scope (follow-ups):**

- 4× discrepancy between expected ~28 GB at `batch=32` and observed 127 GB OOM — separate investigation bean; non-blocking if the fix succeeds at `batch=256`.
- Throughput tuning beyond the OOM fix.
- Refactoring the river path.
- Thread-fanout config.

## Approach (Option B — explicit absence as the "no-showdown" signal)

Three alternatives were considered:

- **A. Surgical gate** — hard-coded `if turn_mode` guards at each allocation site. Fast but makes turn's memory footprint depend on a conventional caller contract. Rejected.
- **B. Explicit absence** — model "no showdown outcomes needed" as `None` (or empty) in the struct. Downstream code branches on presence. **Chosen.**
- **C. Demand-driven refactor** — push allocation entirely into the kernel path, remove the field from `SubgameSpec`. Over-engineered for this fix.

### Data-structure change (`crates/gpu-range-solver/src/batch.rs`)

```rust
pub struct SubgameSpec {
    // ...
    pub showdown_outcomes_p0: Option<Vec<f32>>,  // was Vec<f32>
    pub showdown_outcomes_p1: Option<Vec<f32>>,  // was Vec<f32>
    // ...
}
```

`None` on both fields means "this spec has no showdown payoffs to compute"; the kernel's showdown pass is skipped for the batch containing it.

Mixed `Some`/`None` across a batch is disallowed and panics in debug builds. All specs in a batch share the canonical topology (per the recent batched-datagen work at `22971ba`), so either all specs have real outcomes (river) or none do (turn).

### Call-site changes

- **`pipeline.rs:build_turn_subgame_spec`** — drop `zero_outcomes` and its two clones. Set both fields to `None`.
- **`batch.rs:SubgameSpec::from_game`** (river) — wrap existing `Vec<f32>` in `Some(...)`. Behavior unchanged.
- **`batch.rs:prepare_batch`** — branch on `specs[0].showdown_outcomes_p0.is_some()`:
  - `Some` → existing path (pack into flat `Vec<f32>`, upload to GPU device buffer).
  - `None` → skip host concat, skip H2D upload, set `num_showdowns=0` in the launch parameters. Debug-assert all other specs in the batch also have `None`.

### Kernel change

When `num_showdowns == 0`, `run_iterations` does not launch the showdown-payoff kernel at all (host-side skip, not kernel-side early-return). This removes both the launch overhead and any chance of a div-by-zero / OOB in the kernel. The kernel's source is unchanged.

### Struct/field-level summary

| File | Change |
|-|-|
| `crates/gpu-range-solver/src/batch.rs` | `SubgameSpec.showdown_outcomes_p0/p1`: `Vec<f32>` → `Option<Vec<f32>>`. `prepare_batch` branches on presence. `run_iterations` skips showdown-payoff launch when absent. |
| `crates/gpu-range-solver/src/solver.rs` | `build_mega_terminal_data` still returns real `Vec<f32>`; callers wrap in `Some(...)`. |
| `crates/cfvnet/src/datagen/domain/pipeline.rs` | `build_turn_subgame_spec` sets both fields to `None`; remove `zero_outcomes`. |

## Testing

1. **`gpu-range-solver` unit test (new):** `subgame_spec_without_showdown_outcomes_uses_only_leaf_injection` — batch of 2 specs with `None` showdown outcomes and known leaf-injected CFVs. Asserts the solver output equals the leaf-injected values within fp tolerance.
2. **`gpu-range-solver` unit test (new):** `subgame_spec_mixed_showdown_presence_panics_in_debug` — one `Some`, one `None` in the same batch; debug-mode panic.
3. **`gpu-range-solver` regression (existing):** any existing river-path test must continue to pass and produce numerically identical output (no drift introduced by the `Option` wrapping).
4. **`cfvnet` integration test:** a small turn-datagen run at `gpu_batch_size=4` that exercises `run_gpu_turn` end-to-end and asserts the output is well-formed (non-NaN, correct record count, valid range sums). Keep existing `#[cfg(feature = "gpu-turn-datagen")]` gating.

## Manual validation (required before closing the bean)

Run `cargo run -p cfvnet --release --features gpu-turn-datagen -- generate -c sample_configurations/turn_gpu_datagen.yaml` with `gpu_batch_size: 256` on the user's workstation. Monitor RSS via `top` / `ps`.

**Success criteria:** RSS stays well under host RAM (<64 GB target on a typical 128 GB box), first output file completes, samples written are well-formed.

## Risks

- **Kernel launch parameter threading.** `num_showdowns` is currently passed through several levels (`SubgameSpec` → `TerminalData` → kernel launch). A missed site could result in the kernel reading uninitialized memory. Mitigation: the unit test with `None` outcomes will catch this — if any code path still tries to read the non-existent buffer it will panic on `unwrap()` or read zero-length.
- **River regression from `Option` wrapping.** `Some(vec)` is semantically equivalent but adds one branch per access site. Mitigation: regression test and diff review.
- **Unreconciled 4× OOM factor.** If the observed 127 GB RSS at `batch=32` is not fully explained by the showdown buffer, the fix may not reach `batch=256` in one step. Filed as follow-up bean; first-file smoke run at `batch=256` confirms or refutes.

## Follow-ups (separate beans)

1. Investigate the `gpu_batch_size` plumbing — is the config actually reaching `run_gpu_turn`? Does the observed OOM math require a larger `num_showdowns` than 24?
2. Post-fix throughput characterization — sample/sec at `batch_size=256` vs target (50M samples in 1 hour).
