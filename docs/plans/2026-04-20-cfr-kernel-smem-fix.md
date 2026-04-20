# `cfr_solve` Kernel Shared-Memory Overflow Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Make the `cfr_solve` kernel launch succeed on the canonical turn tree by removing the 79 KB `edge_parent/child/player` arrays from dynamic shared memory and reading them directly from global memory (where they are already resident).

**Architecture:** The canonical turn tree requires ~103 KB of dynamic shared memory per block — exceeds CUDA's 48 KB default and Ada's ~99 KB opt-in cap. Root cause: the kernel redundantly copies `edge_parent/child/player` from global into smem at block start. These three arrays are uniform-broadcast reads (all threads in a block read the same index at each traversal step), so L1 cache on Ampere/Ada serves them from global at near-smem throughput. With edges removed from smem, the per-block request drops to ~26 KB — under the 48 KB default with headroom.

**Tech Stack:** Rust, cudarc 0.19, CUDA (nvrtc runtime compile), RTX 6000 Ada (CC 8.9).

**Bean:** `poker_solver_rust-oox2` (critical — blocks all GPU turn datagen; blocks `vb8r` and `p99d`).

**Design inputs (already gathered):** ml-researcher confirmed the memory-tier strategy; feature-dev:code-architect produced the detailed refactor blueprint. See the conversation context; the essentials are inlined below.

---

## Prerequisites

1. You are running in a dedicated git worktree on a feature branch. If not, stop and ask the coordinator to create one.
2. The canonical-turn-tree commit `22971ba` and the vb8r commits (`42d883a` through `be0131a`) are already on the branch. Do not revert them.
3. Before starting Task 2, **read the kernel source** at `crates/gpu-range-solver/src/kernels.rs:307-653` (the full `HAND_PARALLEL_KERNEL_SOURCE` constant, starting at the `pub const HAND_PARALLEL_KERNEL_SOURCE` line through its closing `"#;`). You need to know every place `s_parent`, `s_child`, `s_player` are referenced so no mutation is missed.
4. Baseline: confirm the failure reproduces before changing code.

   Run:
   ```bash
   mkdir -p local_data/cfvnet/turn/smem_fix_baseline
   cargo run -p cfvnet --release --features gpu-turn-datagen -- generate \
     -c sample_configurations/turn_gpu_datagen.yaml \
     -o local_data/cfvnet/turn/smem_fix_baseline \
     --num-samples 8 --per-file 4 2>&1 | tail -10
   ```
   Expected: fails with `run_iterations failed: DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")` after `[tree] memory per game: 57.9 MB`. Record this in your notes. If it does *not* fail here, stop and flag to the coordinator — the environment has drifted.

5. `sample_configurations/turn_gpu_datagen.yaml` currently has `gpu_batch_size: 4` (set during validation of bean `vb8r`). Leave this value for the baseline and the Task 8 verification run. You may restore it to `32` in a final housekeeping commit after everything passes.

---

## Task 1: Failing unit test — new smem formula for canonical turn tree

**Goal:** Write the unit test that drives the host-side formula change. The test asserts that for canonical-turn-tree dimensions, `compute_hand_parallel_shared_mem` returns a value ≤ 48 KB (CUDA default per-block limit). This test fails today because the formula includes the 79 KB edge arrays. No CUDA required — this is pure arithmetic.

**Files:**
- Modify: `crates/gpu-range-solver/src/gpu.rs` (test module around `hand_parallel_shared_mem_size_computation` at line 513)

### Step 1.1: Add the failing test

Append this test to the `mod tests` block in `crates/gpu-range-solver/src/gpu.rs`, immediately after `hand_parallel_shared_mem_size_computation`:

```rust
#[test]
fn canonical_turn_tree_smem_fits_under_cuda_default_limit() {
    // Regression test for bean oox2: the canonical turn tree
    // (SPR=100, bet sizes [25%, 50%, 100%, a] × [25%, 75%, a]) has
    // num_edges=6590, num_nodes=6591, max_depth=16. With edges stored
    // in dynamic shared memory, the required smem was ~103 KB which
    // exceeds CUDA's 48 KB default per-block limit, causing
    // CUDA_ERROR_INVALID_VALUE at kernel launch.
    //
    // After moving edge_parent/child/player out of smem and reading
    // them directly from global memory, the kernel's dynamic smem
    // must fit under the 48 KB default on any CUDA-capable GPU.
    const CUDA_DEFAULT_SMEM_PER_BLOCK: usize = 48 * 1024;
    let size = compute_hand_parallel_shared_mem(6590, 16, 6591);
    assert!(
        size <= CUDA_DEFAULT_SMEM_PER_BLOCK,
        "canonical turn tree smem {} bytes must fit under CUDA default {} bytes",
        size,
        CUDA_DEFAULT_SMEM_PER_BLOCK
    );
}
```

### Step 1.2: Run the test — expect it to FAIL

Run:
```bash
cargo test -p gpu-range-solver --lib -- gpu::tests::canonical_turn_tree_smem_fits_under_cuda_default_limit 2>&1 | tail -10
```

Expected: FAIL with assertion message like `canonical turn tree smem 105580 bytes must fit under CUDA default 49152 bytes`.

Record the exact "bytes" number — Task 3 will use it to verify the formula change.

### Step 1.3: Commit

```bash
git add crates/gpu-range-solver/src/gpu.rs
git commit -m "$(cat <<'EOF'
test(gpu-range-solver): failing test for canonical turn tree smem budget

Drives the fix in bean oox2 — this test must pass after edge arrays
move out of dynamic shared memory. Today it fails because the formula
returns ~103 KB, exceeding CUDA's 48 KB default per-block limit.

Bean: poker_solver_rust-oox2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Remove edge-array smem copies from the kernel

**Goal:** Edit the CUDA source in `kernels.rs` so the kernel reads `edge_parent/child/player` directly from the global pointers already in its signature. Remove the smem pointer declarations, the cooperative-load loop for those three arrays, and the corresponding bytes from the smem layout. Leave `level_starts`, `level_counts`, and `actions_per_node` in smem unchanged.

**File:** `crates/gpu-range-solver/src/kernels.rs` (the `HAND_PARALLEL_KERNEL_SOURCE` constant, starting at line 307)

**Important:** This is a CUDA source string in a Rust raw string literal. Be careful with indentation and the surrounding `r#"..."#` delimiters. Do NOT introduce new raw-string delimiters.

### Step 2.1: Remove the three smem pointer declarations

At `kernels.rs:384-386`, remove:

```c
    int* s_parent = (int*)shared_raw;
    int* s_child = s_parent + E;
    int* s_player = s_child + E;
```

### Step 2.2: Rebase the remaining smem pointers to start at `shared_raw`

At `kernels.rs:387-389`, change:

```c
    int* s_level_starts = s_player + E;
    int* s_level_counts = s_level_starts + (max_depth + 1);
    float* s_actions = (float*)(s_level_counts + (max_depth + 1));
```

to:

```c
    int* s_level_starts = (int*)shared_raw;
    int* s_level_counts = s_level_starts + (max_depth + 1);
    float* s_actions = (float*)(s_level_counts + (max_depth + 1));
```

### Step 2.3: Remove the cooperative load loop for edge arrays

At `kernels.rs:391-396`, remove this block entirely:

```c
    // Cooperative load of topology into shared memory
    for (int i = tid; i < E; i += blockDim.x) {
        s_parent[i] = edge_parent[i];
        s_child[i] = edge_child[i];
        s_player[i] = edge_player[i];
    }
```

The remaining cooperative-load loops for `level_starts/level_counts` (line 397-400) and `actions_per_node` (line 401-403) stay unchanged. The `__syncthreads()` at line 404 stays — it now guards only the level + actions smem loads, which are still needed.

### Step 2.4: Update the smem layout comment

At `kernels.rs:376-383`, replace:

```c
    // Layout shared memory:
    // [0..E*4] edge_parent as int
    // [E*4..2*E*4] edge_child as int
    // [2*E*4..3*E*4] edge_player as int
    // [3*E*4..3*E*4+(max_depth+1)*4] level_starts as int
    // [then (max_depth+1)*4] level_counts as int
    // [then N*4] actions_per_node as float
    // Then fold eval scratch: card_reach[52] + total_reach
```

with:

```c
    // Layout shared memory:
    // [0..(max_depth+1)*4] level_starts as int
    // [then (max_depth+1)*4] level_counts as int
    // [then N*4] actions_per_node as float
    // Then fold eval scratch: card_reach[52] + total_reach (static __shared__)
    // NOTE: edge_parent/edge_child/edge_player are read directly from global
    // memory (uniform-broadcast pattern — L1 cache serves them at near-smem
    // throughput, and smem no longer fits the canonical turn tree's 6590 edges).
```

### Step 2.5: Replace all `s_parent[e]`, `s_child[e]`, `s_player[e]` accesses

Find every occurrence of `s_parent`, `s_child`, `s_player` in the kernel body and replace with the corresponding global-memory argument:

- `s_parent[X]` → `edge_parent[X]`
- `s_child[X]` → `edge_child[X]`
- `s_player[X]` → `edge_player[X]`

Known sites (based on a pre-scan — re-verify by grep):

- `kernels.rs:464` — `int parent = s_parent[e];` → `int parent = edge_parent[e];`
- `kernels.rs:478` — `int child = s_child[edge];` → `int child = edge_child[edge];`
- `kernels.rs:483` — `if (s_player[edge] == player) {` → `if (edge_player[edge] == player) {`
- `kernels.rs:571` — `int parent = s_parent[e];` → `int parent = edge_parent[e];`
- `kernels.rs:582` — `if (s_player[e] == player) {` → `if (edge_player[e] == player) {`
- `kernels.rs:588` — `cfv[bid * NH + s_child[e + a] * H + h];` → `cfv[bid * NH + edge_child[e + a] * H + h];`
- `kernels.rs:597` — `cfv[bid * NH + s_child[e + a] * H + h];` → `cfv[bid * NH + edge_child[e + a] * H + h];`
- `kernels.rs:606` — `node_cfv += cfv[bid * NH + s_child[e + a] * H + h];` → `node_cfv += cfv[bid * NH + edge_child[e + a] * H + h];`

Confirm with:
```bash
grep -n 's_parent\|s_child\|s_player' crates/gpu-range-solver/src/kernels.rs
```
Expected: only matches in Rust test code (not the CUDA string), or zero matches if the test code didn't reference them. The CUDA string between `HAND_PARALLEL_KERNEL_SOURCE: &str = r#"` and its closing `"#;` must contain zero occurrences of `s_parent`, `s_child`, `s_player`.

### Step 2.6: Verify the kernel still compiles (with NVRTC)

Run:
```bash
cargo test -p gpu-range-solver --lib -- kernels::tests::hand_parallel_kernel_source_valid 2>&1 | tail -10
```

(Or any existing test that calls `HandParallelKernel::compile` — these trigger nvrtc and fail loudly if the CUDA source is malformed.)

Expected: PASS. If NVRTC errors out with "undefined identifier s_parent" or similar, re-run the grep and catch any missed site.

### Step 2.7: Do NOT commit yet

The kernel change and the host-side formula change are inseparable: committing the kernel with the old formula produces a launch with wrong smem budget that would read past the end of the smem allocation. Commit them together after Task 3.

---

## Task 3: Update the host-side shared-memory formula

**Goal:** Match `compute_hand_parallel_shared_mem` to the new smem layout. Drops the `3 * num_edges * 4` term. Retains the `num_edges` parameter in the signature (many callers pass it; removing it would be a broader refactor) but marks it unused.

**File:** `crates/gpu-range-solver/src/gpu.rs:349-360`

### Step 3.1: Replace the function

Replace:

```rust
/// Compute dynamic shared memory size needed for the hand-parallel kernel.
/// Dynamic layout (extern __shared__):
///   edge_parent[E] + edge_child[E] + edge_player[E] (int each)
///   + level_starts[max_depth+1] + level_counts[max_depth+1] (int each)
///   + actions_per_node[N] (float)
/// Note: card_reach[52] and total_reach are static __shared__ (not dynamic).
pub fn compute_hand_parallel_shared_mem(num_edges: usize, max_depth: usize, num_nodes: usize) -> usize {
    let topology = 3 * num_edges * 4; // parent, child, player as int
    let levels = 2 * (max_depth + 1) * 4; // starts + counts as int
    let actions = num_nodes * 4; // actions_per_node as float
    topology + levels + actions
}
```

with:

```rust
/// Compute dynamic shared memory size needed for the hand-parallel kernel.
/// Dynamic layout (extern __shared__):
///   level_starts[max_depth+1] + level_counts[max_depth+1] (int each)
///   + actions_per_node[N] (float)
/// Note: edge_parent/edge_child/edge_player are read directly from global
/// memory (uniform-broadcast access pattern, L1-cached). They used to live
/// in smem but that pushed the canonical turn tree over CUDA's per-block
/// limit (~103 KB vs 48 KB default). See bean poker_solver_rust-oox2.
/// Note: card_reach[52] and total_reach are static __shared__ (not dynamic).
pub fn compute_hand_parallel_shared_mem(
    _num_edges: usize,
    max_depth: usize,
    num_nodes: usize,
) -> usize {
    let levels = 2 * (max_depth + 1) * 4; // starts + counts as int
    let actions = num_nodes * 4; // actions_per_node as float
    levels + actions
}
```

### Step 3.2: Update the existing formula test

At `crates/gpu-range-solver/src/gpu.rs:513-520`, replace:

```rust
    #[test]
    fn hand_parallel_shared_mem_size_computation() {
        // E=10 edges, max_depth=3 (4 levels), N=5 nodes
        let size = compute_hand_parallel_shared_mem(10, 3, 5);
        // Dynamic only: 3*E*4 + 2*(max_depth+1)*4 + N*4
        let expected = 3 * 10 * 4 + 2 * 4 * 4 + 5 * 4;
        assert_eq!(size, expected);
    }
```

with:

```rust
    #[test]
    fn hand_parallel_shared_mem_size_computation() {
        // E=10 edges (unused after oox2 fix), max_depth=3 (4 levels), N=5 nodes
        let size = compute_hand_parallel_shared_mem(10, 3, 5);
        // Dynamic layout: 2*(max_depth+1)*4 + N*4 (edges now in global memory)
        let expected = 2 * 4 * 4 + 5 * 4;
        assert_eq!(size, expected);
    }
```

### Step 3.3: Run both tests — expect PASS

Run:
```bash
cargo test -p gpu-range-solver --lib -- gpu::tests::hand_parallel_shared_mem_size_computation gpu::tests::canonical_turn_tree_smem_fits_under_cuda_default_limit 2>&1 | tail -10
```

Expected: both PASS. The new test from Task 1 now proves the canonical tree fits under 48 KB (~26 KB, well under).

### Step 3.4: Verify nothing else broke

Run the full gpu-range-solver test suite (not needing CUDA features for this layer):
```bash
cargo test -p gpu-range-solver --lib 2>&1 | tail -20
```

Expected: all pass. If a test fails because it asserted against the old formula, fix that specific test to match the new formula (a similar edit to Step 3.2).

### Step 3.5: Commit the kernel + formula together

```bash
git add crates/gpu-range-solver/src/kernels.rs crates/gpu-range-solver/src/gpu.rs
git commit -m "$(cat <<'EOF'
fix(gpu-range-solver): remove edge-array smem copies from cfr_solve kernel

The canonical turn tree (num_edges=6590, num_nodes=6591, max_depth=16)
required ~103 KB of dynamic shared memory per block, exceeding CUDA's
48 KB default and Ada's ~99 KB opt-in cap. CUDA_ERROR_INVALID_VALUE
at launch blocked all GPU turn datagen.

Move edge_parent/edge_child/edge_player out of dynamic shared memory.
They are passed as kernel args (already in global memory) and all
threads in a block read the same index at each traversal step —
uniform-broadcast L1-cache-friendly access, near-smem throughput on
Ampere/Ada. New smem budget is ~26 KB — under the 48 KB default with
headroom for future level/actions growth.

Bean: poker_solver_rust-oox2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Belt-and-suspenders — opt in to Ada's extended smem

**Goal:** Call `set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 99 * 1024)` on the compiled `cfr_solve` function. Unnecessary at the current 26 KB smem budget, but protects against future smem creep across the 48 KB line.

**File:** `crates/gpu-range-solver/src/gpu.rs:300-310` (the `HandParallelKernel::compile` impl)

### Step 4.1: Find the exact cudarc 0.19 API for `set_attribute`

Run:
```bash
grep -rn "set_attribute\|MaxDynamicSharedSize\|MAX_DYNAMIC_SHARED_SIZE\|CUfunction_attribute" \
  ~/.cargo/registry/src/ 2>/dev/null | grep -v test | head -30
```

You will see the path in cudarc's sys module. In cudarc 0.19 the typical call is `func.set_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes)`.

If the API is named differently in this cudarc version, adapt. If you cannot find a set-attribute API exposed on `CudaFunction` in cudarc 0.19, **skip this task entirely** — the kernel already fits under the 48 KB default, so the opt-in is pure defense-in-depth. Note the reason in the commit message or skip the commit.

### Step 4.2: Add the call

Modify `HandParallelKernel::compile` to:

```rust
impl HandParallelKernel {
    pub fn compile(ctx: &Arc<CudaContext>) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            crate::kernels::HAND_PARALLEL_KERNEL_SOURCE,
            crate::kernels::hand_parallel_compile_opts(),
        )?;
        let module = ctx.load_module(ptx)?;
        let cfr_solve = module.load_function("cfr_solve")?;

        // Opt into Ada's extended per-block dynamic smem (99 KB) as a
        // defense-in-depth measure. Current smem budget is ~26 KB — well
        // under the 48 KB default — so this only matters if the topology
        // grows. Failure is not fatal (pre-Ampere GPUs may not support it).
        // See bean poker_solver_rust-oox2.
        let _ = cfr_solve.set_attribute(
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            99 * 1024,
        );

        Ok(Self { cfr_solve })
    }
}
```

Adapt the path to match what Step 4.1 discovered.

### Step 4.3: Verify compile

```bash
cargo check -p gpu-range-solver 2>&1 | tail -10
```

Expected: clean. If there's a type error around the `set_attribute` argument type (expected `u32` vs `i32`), cast accordingly.

### Step 4.4: Run the kernel-compile test

```bash
cargo test -p gpu-range-solver --lib 2>&1 | tail -10
```

Expected: all pass. `set_attribute` fires at compile time, so if it's wrong you'll see an nvrtc / driver error loading the kernel.

### Step 4.5: Commit

```bash
git add crates/gpu-range-solver/src/gpu.rs
git commit -m "$(cat <<'EOF'
feat(gpu-range-solver): opt in to Ada's 99 KB per-block dynamic smem

Defense-in-depth: if cfr_solve's dynamic shared memory ever grows
past the 48 KB default, this attribute lets it up to Ada's 99 KB
ceiling rather than failing CUDA_ERROR_INVALID_VALUE at launch.
Current budget is ~26 KB so this is not load-bearing today.

Bean: poker_solver_rust-oox2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Launch-time debug assertion

**Goal:** In `run_iterations`, debug-assert that `shared_mem_bytes` is at or below the CUDA default per-block limit. If a future change pushes it over 48 KB, fail with a message naming the tree dimensions instead of the opaque `CUDA_ERROR_INVALID_VALUE`.

**File:** `crates/gpu-range-solver/src/batch.rs:436-446` (the launch config region in `run_iterations`)

### Step 5.1: Add the assertion

At `crates/gpu-range-solver/src/batch.rs`, just **before** the `let cfg = LaunchConfig { ... };` block (~line 442), insert:

```rust
        #[cfg(debug_assertions)]
        {
            // CUDA's default per-block dynamic shared-memory limit is 48 KB.
            // Exceeding it produces CUDA_ERROR_INVALID_VALUE at launch with
            // no useful diagnostic. Assert here so the panic names the
            // topology dimensions. See bean poker_solver_rust-oox2.
            const CUDA_DEFAULT_SMEM_PER_BLOCK: u32 = 48 * 1024;
            assert!(
                self.shared_mem_bytes <= CUDA_DEFAULT_SMEM_PER_BLOCK,
                "cfr_solve dynamic smem {} bytes exceeds CUDA 48 KB default. \
                 Tree: num_edges={}, num_nodes={}, max_depth={}. \
                 Either shrink the dynamic smem layout or call \
                 set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES).",
                self.shared_mem_bytes, self.num_edges, self.num_nodes, self.max_depth
            );
        }
```

### Step 5.2: Verify the project still builds

```bash
cargo check -p gpu-range-solver 2>&1 | tail -5
```

Expected: clean. If `self.max_depth` is not a field (look for the actual field name — it may be `max_depth` or stored elsewhere), adapt the message to use whatever is available, but make sure the panic still identifies the offending topology.

### Step 5.3: Commit

```bash
git add crates/gpu-range-solver/src/batch.rs
git commit -m "$(cat <<'EOF'
test(gpu-range-solver): debug-assert cfr_solve smem budget at launch

If the dynamic shared-memory request ever creeps over CUDA's 48 KB
default again, panic with the offending topology dimensions instead
of the opaque CUDA_ERROR_INVALID_VALUE the driver returns.

Bean: poker_solver_rust-oox2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Integration test — canonical turn tree runs end-to-end

**Goal:** Add a test in `cfvnet` that exercises the full canonical turn pipeline on a real topology. This is the test that catches the class of bug `oox2` represents: unit tests pass on the river topology because it's tiny; only the real turn topology exposes the smem overflow.

**File:** `crates/cfvnet/src/datagen/domain/pipeline.rs` (find its test module — if absent, put the test in an existing datagen test module; if still absent, create `crates/cfvnet/tests/canonical_turn_smoke.rs`).

### Step 6.1: Locate an appropriate test module

Run:
```bash
grep -rn "fn build_canonical_turn_tree" crates/cfvnet/src
grep -rn "#\[cfg(feature = \"gpu-turn-datagen\")\]" crates/cfvnet/src | head -5
grep -rn "run_gpu_turn\|build_canonical_turn_tree" crates/cfvnet/src | head -10
```

Find where `build_canonical_turn_tree` is defined and where existing `gpu-turn-datagen` tests live (if any). Use the closest existing test module. If none exists, create `crates/cfvnet/tests/canonical_turn_smoke.rs` as an integration test.

### Step 6.2: Write the smoke test

The test must:
1. Build the canonical turn topology.
2. Construct a `GpuBatchSolver` on it with `max_batch_size=1`.
3. Run at least one iteration of `run_iterations`.
4. Assert no panic, no CUDA error, and that output buffers contain finite numbers.

Model the test on the existing `batch_with_none_showdowns_returns_leaf_injection_values` at `crates/gpu-range-solver/src/batch.rs:1651` — that test already demonstrates the pattern for `GpuBatchSolver::new` + `set_leaf_injection` + `prepare_batch` + `update_leaf_cfvs` + `run_iterations` + `extract_results`. **Read that test first before writing this one.**

Sketch (adapt to the real `build_canonical_turn_tree` signature — you may have to construct a minimal `GameConfig`/`TurnSpec` first):

```rust
#[test]
#[cfg(feature = "gpu-turn-datagen")]
fn canonical_turn_tree_runs_one_iteration_without_smem_overflow() {
    // Regression test for bean oox2: the canonical turn tree previously
    // failed with CUDA_ERROR_INVALID_VALUE at the first run_iterations
    // call because its dynamic shared-memory request (~103 KB) exceeded
    // CUDA's 48 KB default per-block limit. After moving edge arrays out
    // of smem, the kernel launch must succeed.
    //
    // This is a smoke test: one iteration, batch=1, trivial ranges,
    // leaf injection providing zero CFVs. We are asserting the kernel
    // launches and returns finite results — not checking CFR correctness.

    // 1. Build the canonical turn topology exactly as run_gpu_turn does.
    //    (Replace this stub with the real constructor and its args.)
    let topo = /* call build_canonical_turn_tree with production params */;
    let term = /* matching terminal data */;
    let num_hands = 1326usize;

    let ctx = cudarc::driver::CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let mut solver = GpuBatchSolver::new(&topo, &term, 1, num_hands, 1)
        .expect("GpuBatchSolver::new on canonical turn topology");

    // 2. Minimal leaf injection using the topology's own showdown nodes.
    let leaf_node_ids: Vec<i32> =
        topo.showdown_nodes.iter().map(|&n| n as i32).collect();
    let leaf_depths: Vec<i32> = topo
        .showdown_nodes
        .iter()
        .map(|&n| topo.node_depth[n] as i32)
        .collect();
    solver
        .set_leaf_injection(&leaf_node_ids, &leaf_depths)
        .expect("set_leaf_injection");

    // 3. Prepare batch with a single spec, None showdown outcomes
    //    (matches build_turn_subgame_spec post-vb8r).
    let spec = SubgameSpec {
        initial_weights: [vec![1.0f32 / num_hands as f32; num_hands],
                          vec![1.0f32 / num_hands as f32; num_hands]],
        showdown_outcomes_p0: None,
        showdown_outcomes_p1: None,
        fold_payoffs_p0: vec![0.0; /* num_folds for this topo */],
        fold_payoffs_p1: vec![0.0; /* num_folds for this topo */],
    };
    solver.prepare_batch(&[spec]).expect("prepare_batch");

    // 4. Zero leaf CFVs (smoke test — just want the kernel to launch).
    let num_leaves = leaf_node_ids.len();
    let zeros = vec![0.0f32; num_leaves * num_hands];
    solver
        .update_leaf_cfvs(&zeros, &zeros)
        .expect("update_leaf_cfvs");

    // 5. One iteration — this is the call that used to fail.
    solver.run_iterations(0, 1).expect("run_iterations");

    // 6. Assert finiteness of outputs.
    let results = solver.extract_results().expect("extract_results");
    assert_eq!(results.len(), 1);
    for &v in &results[0].strategy_sum {
        assert!(v.is_finite(), "strategy_sum must be finite, got {v}");
    }
}
```

**Developer judgment:**
- The `build_canonical_turn_tree` signature is unknown — inspect it and fill in the params that match how `run_gpu_turn` calls it in `pipeline.rs`. Use the same params production uses.
- `num_folds` needs to come from `term.fold_payoffs.len()` — adapt the `fold_payoffs_p0/p1` vec lengths accordingly.
- If the test helpers in `gpu-range-solver` are not `pub` outside the crate, you may need to either (a) mark the needed items `pub`, (b) put this test inside `gpu-range-solver` instead of `cfvnet`, or (c) build the topology via cfvnet's code path (which is what it exercises in production). Option (c) is cleanest.

### Step 6.3: Run the smoke test

Run:
```bash
cargo test -p cfvnet --features gpu-turn-datagen canonical_turn_tree_runs_one_iteration 2>&1 | tail -20
```

Expected (after Tasks 2+3 landed): PASS. Launch succeeds, results are finite.

If the test fails with `CUDA_ERROR_INVALID_VALUE`, the kernel-source edit missed a smem-pointer reference; re-run Task 2 Step 2.5's grep and fix.

If the test fails with `CUDA_ERROR_ILLEGAL_ADDRESS` at sync time, the kernel is reading from an out-of-range global pointer — most likely the `edge_*` arguments are null/undersized from a missed plumbing path. Re-check that `edge_parent/child/player` are passed into `cfr_solve` in `batch.rs:462-512`.

### Step 6.4: Commit

```bash
git add crates/cfvnet/  # or the tests/ path you created
git commit -m "$(cat <<'EOF'
test(cfvnet): smoke test for canonical turn tree GPU solve

Covers the regression from bean oox2: the canonical turn tree's
dynamic shared memory exceeded CUDA's 48 KB default per-block limit,
causing CUDA_ERROR_INVALID_VALUE at first run_iterations. Unit tests
in gpu-range-solver didn't catch it because the river test topology
needs <1 KB of smem.

Smoke test runs one iteration on the real canonical turn topology
and asserts finite outputs.

Bean: poker_solver_rust-oox2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Manual end-to-end validation

**Goal:** Confirm the fix works in the real pipeline at `gpu_batch_size=4` (the baseline that reproduced the failure in Prerequisites Step 4).

### Step 7.1: Re-run the same command that failed in Prerequisites

```bash
mkdir -p local_data/cfvnet/turn/smem_fix_verify
cargo run -p cfvnet --release --features gpu-turn-datagen -- generate \
  -c sample_configurations/turn_gpu_datagen.yaml \
  -o local_data/cfvnet/turn/smem_fix_verify \
  --num-samples 32 --per-file 8 2>&1 | tee local_data/cfvnet/turn/smem_fix_verify/_run.log
```

Expected: process runs to completion (exit 0). The log no longer contains `DriverError(CUDA_ERROR_INVALID_VALUE, ...)`. At least one file of 8 samples is written.

**Caveat:** the boundary-evaluation throughput bottleneck from bean `p99d` is still present and unrelated to this fix. At batch=4 the run may complete in single-digit minutes per sample. Do not let that mislead you — the acceptance criterion here is "run_iterations does not fail at launch," not "throughput is acceptable."

### Step 7.2: Spot-check the output files

```bash
ls -la local_data/cfvnet/turn/smem_fix_verify/
```

Expected: at least one file with a non-`_run.log` name, non-zero size.

If helpful, use the Python record decoder at `cfvnet.data._decode_single_record` to eyeball a sample:

```bash
python3 -c "
from cfvnet.data import _decode_single_record
import glob
f = [p for p in glob.glob('local_data/cfvnet/turn/smem_fix_verify/*') if '_run.log' not in p][0]
with open(f, 'rb') as fh:
    rec = _decode_single_record(fh)
    print({k: (v.shape if hasattr(v, 'shape') else v) for k, v in rec.items()})
"
```

Expected: a dict with shaped numpy arrays, no NaN in CFV fields.

### Step 7.3: Record in the bean

Note: no commit here — this is validation. Record the observed behaviour in the bean closure (Task 9).

### Step 7.4: If Task 7 FAILS

Possible causes and what to do:

- **Still `CUDA_ERROR_INVALID_VALUE`:** the kernel/source edit was incomplete. Go back to Task 2 Step 2.5 and re-grep.
- **`CUDA_ERROR_ILLEGAL_ADDRESS`:** a kernel global-memory read is out of range. Most likely a missed `s_*` → `edge_*` conversion site, or a bug in argument order. Check `batch.rs:485-494` to confirm the right globals are still being pushed.
- **`extract_results` panics / empty output:** implementation bug unrelated to smem. Stop and flag to coordinator; do NOT paper over.
- **Hangs indefinitely with no output after 10 minutes:** likely the boundary-eval bottleneck from `p99d` is dominating. That is OUT OF SCOPE for `oox2`. Report as "launch succeeded, but throughput too low to complete a file — expected, `p99d` is the fix" and proceed to Task 8.

---

## Task 8: Full test suite + clippy + timing

**Goal:** Per `CLAUDE.md`, confirm the whole project still works.

### Step 8.1: Full workspace test

```bash
time cargo test --workspace 2>&1 | tail -40
```

Expected: all tests pass except the two pre-existing failures documented in `docs/plans/2026-04-19-session-handoff.md` (`mp_tui_scenarios::tests::resolve_empty_returns_root`, `tests::mp_6player_tui_section_parses`). Report those as known-broken, not a regression.

Expected wall time: <60 s per `CLAUDE.md`. If it exceeds, stop and flag to the coordinator — do not try to land the smem fix on top of a slow suite.

### Step 8.2: Clippy

```bash
cargo clippy -p gpu-range-solver -p cfvnet --features gpu-turn-datagen -- -D warnings 2>&1 | tail -20
```

Expected: no new warnings introduced by your changes. Pre-existing warnings in files you did not touch are acceptable.

### Step 8.3: If the suite is slow or breaks

- If a test that was passing before is now failing due to your changes: fix it.
- If the suite now takes >60 s but your changes are small and localized: re-time from a clean `cargo test --no-run` to separate compile from run; if pure test time is still <60 s you may have hit a compile-cache miss. Flag to the coordinator either way.

---

## Task 9: Close the bean

### Step 9.1: Update bean with summary

```bash
beans update --json poker_solver_rust-oox2 -s completed --body-append "$(cat <<'EOF'


## Summary of Changes

Moved `edge_parent/edge_child/edge_player` out of the `cfr_solve` kernel's dynamic shared memory and into direct global-memory reads. The three arrays were already uploaded as GPU globals (`d_edge_parent/child/person`) — the smem copy was redundant. Dynamic smem budget for the canonical turn tree drops from ~103 KB (over the 48 KB CUDA default) to ~26 KB (comfortable headroom).

**Commits (atomic):**
- test: failing unit test for canonical-turn-tree smem budget
- fix: remove edge-array smem copies from cfr_solve kernel (paired with host-side formula update — inseparable)
- feat: opt in to Ada's 99 KB extended dynamic smem (defense-in-depth)
- test: debug-assert cfr_solve smem budget at launch
- test: canonical turn tree smoke test for run_iterations

**Validation:**
- Unit test `canonical_turn_tree_smem_fits_under_cuda_default_limit` passes (no CUDA needed).
- Integration test `canonical_turn_tree_runs_one_iteration_without_smem_overflow` passes on RTX 6000 Ada.
- Production datagen command (`cargo run -p cfvnet ... generate ... --num-samples 32 --per-file 8`) no longer fails at launch. Throughput is still constrained by bean `p99d` (boundary-evaluation serialization) — expected, out of scope for this bean.

**Unblocks:**
- `vb8r` — end-to-end validation of the OOM fix can now proceed.
- `p99d` — boundary-evaluation batching can now be measured, since the kernel launches.
EOF
)"
```

### Step 9.2: Commit the bean update

```bash
git add .beans/poker_solver_rust-oox2*.md
git commit -m "chore: complete bean oox2 (cfr_solve smem overflow fix)"
```

### Step 9.3: Report to coordinator

Final report must include:
- All commit SHAs from Tasks 1-6.
- Peak smem actually requested at launch (print `self.shared_mem_bytes` via `eprintln!` temporarily or from the debug-assert message).
- Exit status of the Task 7 datagen command.
- Test suite timing from Task 8.1.
- Whether `p99d` should now be unblocked (yes, assuming Task 7 passed).
- Any items discovered that warrant follow-up beans (e.g., if `set_attribute` API turned out not to exist in cudarc 0.19 as documented, file a small bean).

---

## Execution guidance

- Work in a dedicated worktree.
- Do NOT squash commits — the granularity is the audit trail.
- If a CUDA error shows up mid-way that isn't `CUDA_ERROR_INVALID_VALUE` (e.g. `ILLEGAL_ADDRESS`), stop and diagnose. Do not paper over by adding guards.
- Do not skip Task 6 even if Task 7 passes — the integration test is the regression protection, without it the bug can recur silently.
- Do not change the CFR algorithm, the tree structure, or the bet-size abstraction. This bean is purely a memory-layout fix.
