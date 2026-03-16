# Bucketed GPU Solver — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild the GPU DCFR+ solver to operate on configurable buckets (500/1000) instead of 1326 concrete combos, matching the Supremus architecture. Phased approach: river → turn → flop, each with correctness validation and performance benchmarks.

**Architecture:** Reuse existing GPU solver kernels (already parameterized by `num_hands` — just pass `num_buckets`). Replace terminal showdown evaluation with bucket-vs-bucket equity matrix multiply. Replace CFV net I/O with bucket-space encoding (2×num_buckets+1 input, num_buckets output). Load bucket assignments from existing `.buckets` files.

**Tech Stack:** Rust, cudarc (existing CUDA kernels), BucketFile (core crate), CudaNetInference (cuBLAS)

**Performance target:** 50M samples in 1 hour (13,889 samples/s)

---

## DCFR+ Algorithm (Supremus-Specific)

The bucketed solver MUST use the Supremus DCFR+ variant, not standard DCFR or the range-solver's power-of-4 scheme. Key differences from our current implementation:

### Regret Discounting (same as standard DCFR)

```
pos_discount = t^1.5 / (t^1.5 + 1)    // alpha=1.5
neg_discount = 0.5                      // beta=0 → t^0/(t^0+1) = 0.5
new_regret = old_regret * discount + instantaneous_regret
```

This part is unchanged from our current GPU solver (matches standard DCFR with alpha=1.5, beta=0).

### Strategy Weighting (DIFFERENT — linear, not multiplicative)

**Current (wrong):** Multiplicative discount with power-of-4 periodic resets:
```
strategy_sum = strategy_sum * strat_discount + strategy
where strat_discount = ((t - nlp4) / (t - nlp4 + 1))^3
```

**Supremus DCFR+ (correct):** Additive linear weighting with delay d=100:
```
weight = max(0, t - d)      // d=100, t is iteration number
strategy_sum += weight * strategy
```

The first 100 iterations get zero weight (strategy exploration only). After that, each iteration's strategy is weighted linearly by `t - 100`. No multiplicative decay, no periodic resets.

**Implementation in update_regrets kernel:**
```cuda
// OLD (wrong):
strategy_sum[idx] = strategy_sum[idx] * strat_discount + strategy[idx];

// NEW (Supremus DCFR+):
float weight = (float)iteration - 100.0f;
if (weight > 0.0f) {
    strategy_sum[idx] += weight * strategy[idx];
}
```

The kernel parameter changes from `strat_discount: f32` to `iteration: u32` (or `strat_weight: f32` precomputed as `max(0, t - 100)`).

### Simultaneous Updates (DIFFERENT for CFVnet leaf eval)

**Current:** Both traversers share a strategy snapshot but traverser 1 sees traverser 0's regret updates.

**Supremus:** True simultaneous — both traversers compute against the exact same strategy and reach probabilities. Regrets from both traversers are accumulated independently using the same pre-iteration state. Neither sees the other's within-iteration updates.

**Implementation:**
1. Compute strategy from regrets ONCE at start of iteration
2. Propagate reach ONCE
3. For each traverser: compute CFVs and accumulate regret deltas into SEPARATE buffers
4. After both traversers: merge regret deltas into the main regret buffer
5. Update strategy sum ONCE

This requires two temporary regret delta buffers (one per traverser) instead of updating regrets in-place.

Alternative (simpler, close enough): keep current structure where regret_match + forward_pass run once, then each traverser does terminal eval + backward CFV + regret update sequentially. The regret updates from traverser 0 ARE visible to traverser 1's regret_match on the NEXT iteration, but NOT within the same iteration (since regret_match runs before the traverser loop). This is already nearly simultaneous — the only leak is that update_regrets for traverser 0 modifies the regret buffer before traverser 1's update_regrets runs, but both use the SAME strategy (computed before the loop). This is acceptable for Phase A; true simultaneous can be added later if needed.

### Summary of Changes for the Bucketed Solver

| Parameter | Current GPU | Supremus DCFR+ | Change needed? |
|-----------|-------------|----------------|---------------|
| pos_discount | `(t-1)^1.5/((t-1)^1.5+1)` | `t^1.5/(t^1.5+1)` | Minor: use `t` not `t-1` |
| neg_discount | 0.5 | 0.5 | No change |
| Strategy weight | Multiplicative power-of-4 | Additive `max(0, t-100)` | **YES — fundamental change** |
| Update order | Near-simultaneous | Simultaneous | Minor (current is close enough) |
| Delay d | Implicit via power-of-4 | d=100 explicit | **YES** |

---

## Phase A: Bucketed River Solver + Training

**Goal:** Solve river subgames in bucket space, train a river CFV net, validate against CPU range-solver. This is the foundation — everything else builds on it.

---

### Task A1: Bucket Equity Table Precomputation

**Files:**
- Create: `crates/gpu-solver/src/bucketed/mod.rs`
- Create: `crates/gpu-solver/src/bucketed/equity.rs`
- Modify: `crates/gpu-solver/src/lib.rs`

Compute the bucket-vs-bucket equity matrix for a given river board.

**Step 1: Define the module**

```rust
// crates/gpu-solver/src/bucketed/mod.rs
pub mod equity;

// crates/gpu-solver/src/lib.rs
pub mod bucketed;
```

**Step 2: Implement bucket equity table**

```rust
// crates/gpu-solver/src/bucketed/equity.rs
use poker_solver_core::blueprint_v2::bucket_file::BucketFile;

/// Compute bucket-vs-bucket equity matrix for a river board.
///
/// Returns `[num_buckets × num_buckets]` where `equity[i * num_buckets + j]`
/// = expected payoff (in half-pots) when bucket i faces bucket j at showdown.
/// Positive = bucket i wins, negative = loses. Card-blocking hands are excluded.
///
/// The matrix is pre-normalized: values are fractions of half_pot, not raw chip amounts.
pub fn compute_bucket_equity_table(
    board: &[u8; 5],           // 5 river cards (range-solver encoding)
    bucket_file: &BucketFile,  // river bucket assignments
    num_buckets: usize,
) -> Vec<f32> {
    // 1. Find the board index in the bucket file
    // 2. For each combo pair (i, j) where cards don't conflict:
    //    a. Evaluate hand strengths
    //    b. Look up bucket_i = bucket_file.get_bucket(board_idx, combo_i)
    //    c. Look up bucket_j = bucket_file.get_bucket(board_idx, combo_j)
    //    d. Accumulate: if hand_i > hand_j: equity[bucket_i][bucket_j] += 1
    //                   if hand_i < hand_j: equity[bucket_i][bucket_j] -= 1
    // 3. Normalize by count of hand pairs per bucket pair
    // Return flat [num_buckets * num_buckets]
}
```

Use `range_solver::card::evaluate_hand_strength()` for hand evaluation.
Use `BucketFile::board_index_map()` to find the board index, or iterate boards to match.

**Step 3: Test**

```rust
#[test]
fn test_bucket_equity_table() {
    let bf = BucketFile::load(Path::new("local_data/clusters_500bkt_v3/river.buckets")).unwrap();
    let board = [0, 5, 10, 15, 20]; // some valid 5-card board
    let eq = compute_bucket_equity_table(&board, &bf, 500);
    assert_eq!(eq.len(), 500 * 500);
    // Equity should be anti-symmetric: E[i][j] ≈ -E[j][i]
    // Diagonal should be ~0 (same bucket ties roughly half the time)
}
```

**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): add bucket-vs-bucket equity table precomputation"
```

---

### Task A2: Bucketed FlatTree Builder

**Files:**
- Create: `crates/gpu-solver/src/bucketed/tree.rs`

Build a FlatTree-like structure but with `num_buckets` instead of `num_hands`. The tree topology (nodes, actions, levels) is identical — only the per-hand dimension changes.

**Step 1: Define BucketedTree**

```rust
pub struct BucketedTree {
    // Same topology as FlatTree:
    pub node_types: Vec<NodeType>,
    pub pots: Vec<f32>,
    pub child_offsets: Vec<u32>,
    pub children: Vec<u32>,
    pub parent_nodes: Vec<u32>,
    pub parent_actions: Vec<u32>,
    pub level_starts: Vec<u32>,
    pub infoset_ids: Vec<u32>,
    pub infoset_num_actions: Vec<u32>,
    pub num_infosets: usize,
    pub terminal_indices: Vec<u32>,
    pub boundary_indices: Vec<u32>,
    pub boundary_pots: Vec<f32>,
    pub boundary_stacks: Vec<f32>,

    // Bucketed dimensions (replaces num_hands):
    pub num_buckets: usize,

    // Per-terminal equity tables (for showdown):
    // equity_tables[showdown_ordinal] = [num_buckets * num_buckets] flat matrix
    pub showdown_equity_tables: Vec<Vec<f32>>,

    // Fold payoffs (simplified — no card blocking):
    // For fold: payoff = ±half_pot (same for all buckets)
    pub fold_half_pots: Vec<f32>,  // [num_fold_terminals] — half_pot at each fold node
    pub fold_players: Vec<u32>,    // [num_fold_terminals] — who folded
}
```

**Step 2: Build from PostFlopGame + BucketFile**

```rust
impl BucketedTree {
    pub fn from_postflop_game(
        game: &mut PostFlopGame,
        bucket_file: &BucketFile,
        num_buckets: usize,
    ) -> Self {
        // BFS walk same as FlatTree::from_postflop_game
        // At showdown terminals: compute bucket equity table
        // At fold terminals: just store half_pot and folded player
        // num_hands → num_buckets throughout
    }
}
```

**Step 3: Test — build and verify structure**

**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): add BucketedTree builder"
```

---

### Task A3: Bucketed Showdown Kernel

**Files:**
- Create: `crates/gpu-solver/kernels/bucketed_showdown.cu`
- Modify: `crates/gpu-solver/src/gpu.rs`

Replace hand strength comparison with equity matrix-vector multiply:

```cuda
extern "C" __global__ void bucketed_showdown_eval(
    float* cfvalues,
    const float* opp_reach,
    const unsigned int* terminal_nodes,
    const float* equity_tables,        // [num_sd_terminals * num_buckets * num_buckets]
    const float* half_pots,            // [num_sd_terminals]
    unsigned int num_sd_terminals,
    unsigned int num_hands,            // = num_buckets * num_spots
    unsigned int num_buckets
) {
    // One thread per (terminal, bucket)
    // cfv[bucket_i] = half_pot * sum_j(equity[i][j] * opp_reach[j])
    // This is a matrix-vector multiply!
}
```

For 500 buckets, each thread does 500 multiply-adds. Total: 500 × num_terminals × num_spots threads. Much cheaper than 1326² hand-vs-hand comparison.

**Step 1: Write the kernel**
**Step 2: Add launch method**
**Step 3: Test against CPU computation**
**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): add bucketed showdown evaluation kernel"
```

---

### Task A4: Bucketed Fold Kernel

**Files:**
- Create: `crates/gpu-solver/kernels/bucketed_fold.cu`
- Modify: `crates/gpu-solver/src/gpu.rs`

Simplified fold — no card blocking:

```cuda
extern "C" __global__ void bucketed_fold_eval(
    float* cfvalues,
    const float* opp_reach,
    const unsigned int* terminal_nodes,
    const float* half_pots,            // [num_fold_terminals]
    const unsigned int* fold_player,
    unsigned int traverser,
    unsigned int num_fold_terminals,
    unsigned int num_hands,            // = num_buckets * num_spots
    unsigned int num_buckets
) {
    // cfv[bucket_i] = ±half_pot * sum(opp_reach[all buckets])
    // No card blocking — all buckets contribute
}
```

**Step 1-4: Implement, test, commit**

```bash
git commit -am "feat(gpu-solver): add bucketed fold evaluation kernel"
```

---

### Task A5: BucketedGpuSolver

**Files:**
- Create: `crates/gpu-solver/src/bucketed/solver.rs`

The DCFR+ solver operating in bucket space. Reuses existing kernels (regret_match, forward_pass, backward_cfv, update_regrets) by passing `num_buckets` as `num_hands`.

**Step 1: Define struct**

```rust
pub struct BucketedGpuSolver<'a> {
    gpu: &'a GpuContext,
    num_buckets: u32,
    // Same fields as GpuSolver but dimensioned by num_buckets
    regrets: CudaSlice<f32>,           // [num_infosets * max_actions * num_buckets]
    strategy_sum: CudaSlice<f32>,
    current_strategy: CudaSlice<f32>,
    reach_oop: CudaSlice<f32>,         // [num_nodes * num_buckets]
    reach_ip: CudaSlice<f32>,
    cfvalues: CudaSlice<f32>,
    // Tree structure + bucketed terminal data
    // ...
}
```

**Step 2: Implement solve loop**

Same DCFR+ loop as GpuSolver but uses bucketed fold/showdown kernels.

**Step 3: Initial reach from ranges + bucket mapping**

The solver needs initial reach in bucket space. For a given board:
```
for combo in 0..1326:
    bucket = bucket_file.get_bucket(board_idx, combo)
    reach[bucket] += range[combo]
```

This mapping runs on CPU (cheap, once per batch) and uploads `[num_buckets]` reach.

**Step 4: Test — solve a river position, verify strategy sums to 1.0**

**Step 5: Correctness validation**

```
gpu-eval-bucketed --model [none] --street river --num-spots 30 --solve-iters 1000
```

Compare bucketed solver strategy against the CPU range-solver (concrete hands). The strategies won't match exactly (bucketing introduces approximation) but dominant actions should agree and exploitability should be similar.

**Step 6: Commit**

```bash
git commit -am "feat(gpu-solver): add BucketedGpuSolver for river"
```

---

### Task A6: Sample Generation — Reuse cfvnet Sampler

**Files:**
- Create: `crates/gpu-solver/src/bucketed/sampler.rs`

Reuse the cfvnet sampler for high-quality situation generation. The cfvnet sampler has extensive work ensuring:
- **Stratified pot sampling** via configurable `pot_intervals` — uniform coverage across pot sizes
- **SPR stratification** via `spr_intervals` with rejection sampling — covers all stack-to-pot ratio regimes
- **R(S,p) ranges** — hand-strength-correlated range generation (DeepStack algorithm), NOT uniform random weights
- **2D pot×SPR uniformity** — verified by integration tests

This runs on CPU (cheap — <1ms per situation). The sample is then converted to bucket space and uploaded to GPU.

**Step 1: Wrap cfvnet sampler**

```rust
use cfvnet::datagen::sampler::{sample_situation, Situation};
use cfvnet::config::DatagenConfig;

pub fn sample_and_bucketize(
    datagen_config: &DatagenConfig,
    bucket_file: &BucketFile,
    num_buckets: usize,
    initial_stack: i32,
    board_size: usize,
    rng: &mut impl Rng,
) -> BucketedSituation {
    // 1. Sample situation using cfvnet sampler (board, pot, stack, R(S,p) ranges)
    let sit = sample_situation(datagen_config, initial_stack, board_size, rng);

    // 2. Find board index in bucket file
    let board_idx = find_board_index(bucket_file, &sit.board[..board_size]);

    // 3. Map 1326-dim ranges to num_buckets-dim reach
    let mut oop_reach = vec![0.0f32; num_buckets];
    let mut ip_reach = vec![0.0f32; num_buckets];
    for combo in 0..1326 {
        let bucket = bucket_file.get_bucket(board_idx, combo as u16) as usize;
        if bucket < num_buckets {
            oop_reach[bucket] += sit.ranges[0][combo];
            ip_reach[bucket] += sit.ranges[1][combo];
        }
    }

    BucketedSituation {
        board: sit.board,
        board_size: sit.board_size,
        board_idx,
        pot: sit.pot,
        effective_stack: sit.effective_stack,
        oop_reach,
        ip_reach,
    }
}
```

**Step 2: Configuration**

The YAML config includes cfvnet-style sampling params:

```yaml
sampling:
  pot_intervals: [[4, 50], [50, 100], [100, 150], [150, 200]]
  spr_intervals: [[0.0, 0.5], [0.5, 1.5], [1.5, 4.0], [4.0, 8.0], [8.0, 50.0]]
  initial_stack: 200
```

**Step 3: Test** — sample 1000 situations, verify pot/SPR distributions are uniform across intervals.

**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): bucketed sampler with cfvnet-quality distributions"
```

---

### Task A7: Combo-to-Bucket Mapping Utilities

**Files:**
- Create: `crates/gpu-solver/src/bucketed/mapping.rs`

Utilities for converting between 1326-combo space and bucket space.

**Step 1: Range → bucket reach**

```rust
/// Convert 1326-dim range weights to num_buckets-dim bucket reach.
pub fn range_to_bucket_reach(
    range: &[f32; 1326],
    bucket_file: &BucketFile,
    board_idx: u32,
    num_buckets: usize,
) -> Vec<f32> {
    let mut reach = vec![0.0f32; num_buckets];
    for combo in 0..1326u16 {
        let bucket = bucket_file.get_bucket(board_idx, combo) as usize;
        if bucket < num_buckets {
            reach[bucket] += range[combo as usize];
        }
    }
    reach
}
```

**Step 2: Bucket strategy → combo strategy (for display)**

```rust
/// Expand bucket-space strategy to 1326-combo strategy for display.
/// Each combo gets its bucket's strategy.
pub fn bucket_strategy_to_combos(
    bucket_strategy: &[f32],  // [num_actions * num_buckets]
    bucket_file: &BucketFile,
    board_idx: u32,
    num_buckets: usize,
    num_actions: usize,
) -> Vec<f32> {  // [num_actions * 1326]
    let mut combo_strat = vec![0.0f32; num_actions * 1326];
    for combo in 0..1326u16 {
        let bucket = bucket_file.get_bucket(board_idx, combo) as usize;
        for a in 0..num_actions {
            combo_strat[a * 1326 + combo as usize] = bucket_strategy[a * num_buckets + bucket];
        }
    }
    combo_strat
}
```

**Step 3: Board → board_idx lookup**

```rust
/// Find the board index in the bucket file for a given set of cards.
/// Canonicalizes the board first, then looks up in the board index map.
pub fn find_board_index(
    bucket_file: &BucketFile,
    board_cards: &[u8],
) -> u32 {
    // Convert u8 cards to Card, canonicalize, pack, look up
    // Uses BucketFile::board_index_map() or binary search on sorted boards
}
```

**Step 4: Test** — round-trip: range → buckets → expand back. Verify expanded strategy is valid.

**Step 5: Commit**

```bash
git commit -am "feat(gpu-solver): combo-to-bucket mapping utilities"
```

---

### Task A8: River Batch Solver + Training Pipeline

**Files:**
- Create: `crates/gpu-solver/src/bucketed/batch.rs`
- Create: `crates/gpu-solver/src/bucketed/training.rs`

Batch-solve N river spots in bucket space and train a river CFV net.

**Step 1: BatchBucketedSolver**

Same pattern as BatchGpuSolver but with `num_buckets` dimensions. All spots share the same tree topology. Per-spot data:
- `initial_reach_oop/ip`: `[num_spots × num_buckets]` — from range_to_bucket_reach()
- `equity_tables`: `[num_showdown_terminals × num_spots × num_buckets × num_buckets]` — per-spot because different boards have different equity. Note: this is large for many spots. With 500 buckets, each table is 500×500×4=1MB. For 1000 spots × 5 showdown terminals = 5GB. May need to share equity tables across spots with the same board, or compute on-the-fly.

**Optimization for equity tables**: Since equity tables are per-board and the sampler generates random boards, each spot has a unique table. For batch_size=1000, this is 5GB — too much. Solutions:
- **Reduce batch_size** for the solver (e.g., 100 spots)
- **Compute equity matrix-vector product on CPU** and upload CFV results (defeats GPU purpose)
- **Use a single shared equity table** based on average equity across boards (lossy)
- **Compute equity on-the-fly in the kernel** — each thread evaluates the few hands in its bucket pair. Viable for 500 buckets but complex.

Best approach for Phase A: **use moderate batch_size (100-500)** where equity tables fit in VRAM. At 500 buckets, 100 spots × 5 terminals × 500 × 500 × 4 = 500MB. Manageable.

**Step 2: Training loop**

```
Per batch (CPU + GPU):
  1. CPU: sample N situations using cfvnet sampler (stratified pot/SPR + R(S,p))
  2. CPU: map ranges to bucket reach, compute equity tables per board
  3. CPU → GPU: upload bucket reach + equity tables
  4. GPU: batch solve (DCFR+ with bucketed kernels) → extract root CFVs
  5. GPU: insert CFVs into reservoir (bucket-space)
  6. GPU: train step (burn-cuda)
```

Steps 1-3 are CPU (sampling + bucket mapping + equity precomputation). These run once per batch — the solve (step 4) at 4000 iterations dominates.

**Step 3: CFV net architecture**

Input: `2 × num_buckets + 1 = 1001` (for 500 buckets)
Output: `num_buckets = 500`
Hidden: 7 × 500 (same as Supremus)

The network is much smaller than the concrete version (1001 input vs 2720).

**Step 4: Reservoir encoding**

Bucket-space reservoir records:
```
input:  [2 × num_buckets + 1] = 1001 floats
target: [num_buckets]          = 500 floats
mask:   [num_buckets]          = 500 floats (all 1.0 for valid buckets)
```

No board one-hot needed — bucket assignments already encode board context.

**Step 5: Performance benchmark**

Target: 50M samples in 1 hour. With 500 buckets (2.65x less compute per kernel), expect proportionally faster throughput.

**Step 6: Commit**

```bash
git commit -am "feat(gpu-solver): bucketed river batch solver and training pipeline"
```

---

### Task A9: Evaluation — gpu-eval-bucketed (cfvnet compare style)

**Files:**
- Create: `crates/gpu-solver/src/bucketed/eval.rs`

A diagnostic command matching `cfvnet compare` output but for bucketed models.

**Step 1: Per-spot evaluation**

For each of N test spots:
1. Sample a situation using cfvnet sampler (quality distribution)
2. Map to bucket space
3. Solve exactly in bucket space (high iterations, 10,000+) → ground-truth bucket CFVs
4. Encode as network input `[2 × num_buckets + 1]`
5. Run model forward pass → predicted bucket CFVs
6. Compare: MAE, max error, per-bucket error histogram

**Step 2: Metrics (matching cfvnet compare)**

```
Per spot:
  MAE (pot-relative): mean absolute error across valid buckets
  Max error: worst single-bucket error
  mBB error: MAE × 1000 (assuming 1 pot unit)

Summary:
  Overall MAE (pot-relative)
  Overall mBB/hand
  Max error across all spots
  Per-bucket error distribution (histogram: how many buckets have error < 0.01, < 0.02, etc.)
  Spots where dominant action disagrees with exact solve
```

**Step 3: Output format**

```
Evaluating river model: models/bucketed_v1/river (7x500, 500 buckets)
Solving 100 spots at 10000 iterations...

  Spot   1/100: MAE=0.0142  mBB= 7.1  max=0.089  (OOP check 62%, solver check 64%)
  Spot   2/100: MAE=0.0098  mBB= 4.9  max=0.045  (OOP bet 78%, solver bet 80%)
  ...

Summary:
  Mean Absolute Error:     0.0134 pot-relative (13.4 mBB/hand)
  Max Error:               0.112
  Dominant action agree:   94/100 (94%)
  Bucket error histogram:
    <0.005: 312 buckets (62.4%)
    <0.01:  108 buckets (21.6%)
    <0.02:   62 buckets (12.4%)
    <0.05:   14 buckets (2.8%)
    >0.05:    4 buckets (0.8%)
```

**Step 4: CLI command**

```bash
gpu-eval-bucketed --model models/river --hidden-layers 7 --hidden-size 500 \
  --buckets local_data/clusters_500bkt_v3/river.buckets --num-buckets 500 \
  --street river --num-spots 100 --solve-iters 10000
```

**Step 5: Commit**

```bash
git commit -am "feat(gpu-solver): add gpu-eval-bucketed diagnostic command"
```

---

### Phase A Validation Checkpoint

Before proceeding to Phase B:
1. ✅ Bucketed river solver produces valid strategies (bucket probs sum to 1.0)
2. ✅ Bucketed solver matches CPU range-solver on dominant actions (>90% agreement after expanding buckets to combos)
3. ✅ River CFV net validation loss decreasing over training
4. ✅ `gpu-eval-bucketed --street river` reports MAE < 0.05 pot-relative (<50 mBB/hand) with sufficient training data
5. ✅ Performance: river training >1000 samples/s (bucket-space is smaller)
6. ✅ Pot/SPR distribution of training samples matches cfvnet quality (verified by histogram)

---

## Phase B: Bucketed Turn Solver + Training

**Goal:** Add neural leaf evaluation for turn subgames using the Phase A river model. Same architecture as concrete Phase 3 but in bucket space.

---

### Task B1: Bucket-Space Leaf Evaluation

**Files:**
- Create: `crates/gpu-solver/src/bucketed/leaf_eval.rs`
- Create: `crates/gpu-solver/kernels/bucketed_encode_leaf.cu`
- Create: `crates/gpu-solver/kernels/bucketed_average_leaf.cu`

At depth boundaries, encode bucket-space reach into network input (2×num_buckets+1), run forward pass, convert output back to DCFR+ cfvalues.

**Key difference from concrete version**: No 2720-dim encoding. Just:
- `input[0..500]` = OOP bucket reach at boundary
- `input[500..1000]` = IP bucket reach at boundary
- `input[1000]` = pot / normalization_constant

The encoding kernel is trivial — just gather reach values and append pot.

No num_combinations issue — the bucket equity tables are pre-normalized. The CFV net output IS the cfvalue (no conversion needed).

**Step 1: Encode kernel** — gather reach at boundary + pot into `[num_inputs × (2*num_buckets+1)]`

**Step 2: Average kernel** — average model outputs across possible next cards (48 for turn→river, 49 for flop→turn)

**Step 3: Test**

**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): bucketed leaf evaluation kernels"
```

---

### Task B2: Bucketed Turn Solver + Training

**Files:**
- Create: `crates/gpu-solver/src/bucketed/turn_solver.rs`
- Modify: `crates/gpu-solver/src/bucketed/training.rs`

Same pattern as concrete turn solver but in bucket space. Uses BucketedGpuSolver + bucket-space leaf eval with river model.

**Step 1: TurnBucketedSolver**

For each of 48 possible river cards at a depth boundary:
1. Look up bucket assignments for that 5-card river board
2. Map boundary reach from turn-bucket-space to river-bucket-space
3. Encode as `[2×num_buckets+1]` input
4. Forward pass through river model → river-bucket-space CFVs
5. Map back to turn-bucket-space
6. Average across 48 river cards

The bucket-space mapping between streets is the key new complexity. For a turn combo at turn bucket `t`, and a specific river card, the same combo might map to river bucket `r`. The reach transfer is:
```
river_reach[r] += turn_reach[t] × fraction_of_combos_in_t_that_map_to_r
```

This is a sparse matrix multiply, precomputed per (turn_board, river_card).

**Step 2: Training pipeline**

Same as river: sample → solve → reservoir → train. But the solve includes leaf eval.

**Step 3: Performance benchmark**

Target for turn: batch_size=1000 with 500 buckets should be significantly faster than 1326 concrete hands.

**Step 4: Diagnostic: eval command**

```bash
gpu-eval-bucketed-model --street turn --model models/turn --leaf-model models/river
```

**Step 5: Commit**

```bash
git commit -am "feat(gpu-solver): bucketed turn solver and training pipeline"
```

---

### Phase B Validation Checkpoint

1. ✅ Turn solver produces valid strategies with neural leaf eval
2. ✅ Turn training loss decreasing
3. ✅ Turn MAE improving with more samples
4. ✅ Performance: turn training >100 samples/s
5. ✅ No unit mismatch — bucket-space values are self-consistent

---

## Phase C: Bucketed Flop + Preflop

**Goal:** Complete the model stack. Same patterns as concrete Phases 4, but in bucket space.

---

### Task C1: Flop Solver + Training

Same as turn but:
- 3-card boards, 49 possible turn cards per boundary
- Uses turn model at leaves
- Bucket mapping: flop-bucket-space ↔ turn-bucket-space

### Task C2: Preflop Auxiliary Training

Same as concrete: inference-only, 1755 canonical flops, weighted averaging. But in bucket space — much smaller tensors (500-dim instead of 1326-dim).

### Task C3: Full Stack CLI

```bash
gpu-train-bucketed-stack -c config.yaml -o models/bucketed_stack_v1
```

Config:
```yaml
buckets:
  path: "local_data/clusters_500bkt_v3"
  num_buckets: 500

river_model:
  train: true  # or path to pre-trained
  num_samples: 50000000

turn:
  num_samples: 20000000
  # ...

flop:
  num_samples: 5000000

preflop:
  num_samples: 10000000
```

### Task C4: Eval + Diagnostics Suite

```bash
# Evaluate each model independently
gpu-eval-bucketed-model --street river --model models/river --num-spots 100
gpu-eval-bucketed-model --street turn --model models/turn --leaf-model models/river
gpu-eval-bucketed-model --street flop --model models/flop --leaf-model models/turn

# Full-stack resolve test
gpu-resolve --model-stack models/bucketed_stack_v1 --board "Qs Jh 2c" --oop-range "QQ+" --ip-range "22+"
```

---

### Phase C Validation Checkpoint

1. ✅ All 4 models trained
2. ✅ Each street's MAE is improving with more data
3. ✅ Full stack resolving works in Explorer
4. ✅ Performance: 50M river samples in <1 hour
5. ✅ Exploitability < 50 mBB/hand on test positions (with enough training)
