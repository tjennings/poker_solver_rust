# Batch GPU Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Build a `BatchGpuSolver` that solves N independent river spots simultaneously on the GPU, sharing one tree topology and batching all hands across spots into a single set of kernel launches.

**Architecture:** All N spots share the same bet-size config → identical tree topology (nodes, infosets, child structure). Only per-hand data varies (ranges, hand strengths, payoffs, card blocking). We set the kernel's `num_hands = N × hands_per_spot`, so existing kernels parallelize across all spots' hands in one launch. A new `BatchGpuSolver` struct handles the merge/split.

**Tech Stack:** Rust, cudarc (existing CUDA wrapper), range-solver (game building), cfvnet sampler (random situation generation)

---

## Task 1: RiverSpot Input Type

**Files:**
- Modify: `crates/gpu-solver/src/lib.rs`
- Create: `crates/gpu-solver/src/batch.rs`

**Step 1: Create batch module with RiverSpot**

In `crates/gpu-solver/src/batch.rs`:

```rust
use range_solver::bet_size::BetSizeOptions;

/// A single river spot to be solved in a batch.
/// All spots in a batch must share the same bet sizes (tree topology).
#[derive(Debug, Clone)]
pub struct RiverSpot {
    /// Flop cards (3 cards, each 0..51)
    pub flop: [u8; 3],
    /// Turn card (0..51)
    pub turn: u8,
    /// River card (0..51)
    pub river: u8,
    /// OOP range weights (1326 combos, 0.0 for blocked/absent)
    pub oop_range: Vec<f32>,
    /// IP range weights (1326 combos)
    pub ip_range: Vec<f32>,
    /// Starting pot size
    pub pot: i32,
    /// Effective stack
    pub effective_stack: i32,
}

/// Shared tree configuration for a batch (same for all spots).
#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub oop_bet_sizes: BetSizeOptions,
    pub ip_bet_sizes: BetSizeOptions,
}
```

**Step 2: Add module to lib.rs**

Add `#[cfg(feature = "cuda")] pub mod batch;` to `crates/gpu-solver/src/lib.rs`.

**Step 3: Verify it compiles**

Run: `cargo check -p poker-solver-gpu --features cuda`

**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): add RiverSpot and BatchConfig types"
```

---

## Task 2: Build PostFlopGame from RiverSpot

**Files:**
- Modify: `crates/gpu-solver/src/batch.rs`
- Test: inline tests

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_game_from_spot() {
        let spot = RiverSpot {
            flop: [36, 25, 0],  // Qs Jh 2c
            turn: [24],          // 8d  — actually just u8: 24
            river: [2],          // 3s  — actually just u8: 2
            oop_range: {
                let r: range_solver::range::Range = "AA,KK".parse().unwrap();
                r.raw_data().to_vec()
            },
            ip_range: {
                let r: range_solver::range::Range = "QQ,JJ".parse().unwrap();
                r.raw_data().to_vec()
            },
            pot: 100,
            effective_stack: 100,
        };
        let config = BatchConfig::default(); // 50%,a bets

        let mut game = build_game_from_spot(&spot, &config).unwrap();
        assert!(game.num_private_hands(0) > 0);
        assert!(game.num_private_hands(1) > 0);
    }
}
```

**Step 2: Implement `build_game_from_spot`**

```rust
use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
use range_solver::range::Range;
use range_solver::{CardConfig, PostFlopGame};

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            oop_bet_sizes: BetSizeOptions::try_from(("50%,a", "")).unwrap(),
            ip_bet_sizes: BetSizeOptions::try_from(("50%,a", "")).unwrap(),
        }
    }
}

pub fn build_game_from_spot(
    spot: &RiverSpot,
    config: &BatchConfig,
) -> Result<PostFlopGame, String> {
    let oop_range = Range::from_raw_data(&spot.oop_range);
    let ip_range = Range::from_raw_data(&spot.ip_range);

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: spot.flop,
        turn: spot.turn,
        river: spot.river,
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: spot.pot,
        effective_stack: spot.effective_stack,
        river_bet_sizes: [config.oop_bet_sizes.clone(), config.ip_bet_sizes.clone()],
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config)?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)?;
    game.allocate_memory(false);
    Ok(game)
}
```

**Step 3: Run test**

Run: `cargo test -p poker-solver-gpu --features cuda test_build_game_from_spot`

**Step 4: Commit**

```bash
git commit -am "feat(gpu-solver): add build_game_from_spot helper"
```

---

## Task 3: Batch Per-Hand Data Assembly

**Files:**
- Modify: `crates/gpu-solver/src/batch.rs`
- Test: inline tests

This is the core of the batch design. We build one FlatTree from the first spot (shared topology), then extract only the per-hand data from each spot and concatenate along the hand dimension.

**Step 1: Write the failing test**

```rust
#[test]
fn test_batch_per_hand_data() {
    let spots = make_test_spots(3); // helper: 3 random river spots
    let config = BatchConfig::default();
    let batch_data = BatchPerHandData::build(&spots, &config).unwrap();

    // Total hands = sum of max(oop, ip) hands per spot
    assert!(batch_data.total_hands > 0);
    // Topology from first spot
    assert!(batch_data.topology.num_nodes() > 0);
    // Reach arrays sized correctly
    assert_eq!(batch_data.initial_reach_oop.len(),
        batch_data.topology.num_nodes() * batch_data.total_hands);
    // hand_strengths sized correctly
    assert_eq!(batch_data.hand_strengths_oop.len(), batch_data.total_hands);
}
```

**Step 2: Implement BatchPerHandData**

```rust
use crate::tree::FlatTree;

/// Per-hand data concatenated across N spots.
/// Hand indices 0..hands[0] belong to spot 0,
/// hands[0]..hands[0]+hands[1] belong to spot 1, etc.
pub struct BatchPerHandData {
    /// Shared tree topology (from first spot).
    pub topology: FlatTree,
    /// Total hands across all spots.
    pub total_hands: usize,
    /// Number of hands per spot (for splitting results).
    pub hands_per_spot: Vec<usize>,
    /// Concatenated initial reach for OOP: [num_nodes * total_hands]
    /// Layout: only root node (0) has non-zero values.
    pub initial_reach_oop: Vec<f32>,
    /// Concatenated initial reach for IP.
    pub initial_reach_ip: Vec<f32>,
    /// Concatenated hand strengths OOP: [total_hands]
    pub hand_strengths_oop: Vec<u32>,
    pub hand_strengths_ip: Vec<u32>,
    /// Concatenated valid matchup matrices: [total_hands * total_hands]
    /// But this is too large! Each spot has its own num_hands × num_hands block.
    /// We need a block-diagonal structure.
    pub valid_matchups_oop: Vec<f32>,
    pub valid_matchups_ip: Vec<f32>,
    /// Per-terminal payoffs concatenated. Fold: [num_fold_terminals * total_hands] layout.
    /// Actually fold payoffs are per-terminal constants, not per-hand. They differ per spot
    /// because pot sizes differ. We need per-spot fold payoffs.
    pub fold_amount_win: Vec<f32>,   // [num_fold_terminals * num_spots]
    pub fold_amount_lose: Vec<f32>,
    pub fold_player: Vec<u32>,       // [num_fold_terminals] (same across spots)
    pub showdown_amount_win: Vec<f32>, // [num_showdown_terminals * num_spots]
    pub showdown_amount_lose: Vec<f32>,
    pub num_spots: usize,
}
```

Wait — there's a subtlety. **Payoffs differ per spot** because pot sizes differ at each node (different starting pots/stacks). The tree topology is the same (same nodes), but the pot at each node depends on the spot's starting pot and stack.

**Revised approach**: Instead of storing per-terminal payoff constants, we need per-(terminal, spot) payoffs. The terminal kernels need to index payoffs by `terminal_idx * num_spots + spot_idx` where `spot_idx = hand / hands_per_spot`.

Actually, simpler: since `num_hands = N × hands_per_spot_max`, and each "hand" belongs to a spot, we can store payoffs as `[num_terminals × total_hands]` where the payoff for hand h at terminal t is the payoff for hand h's spot at that terminal. We precompute this on CPU.

**Even simpler**: The fold kernel currently computes `cfv = payoff * opp_reach_sum`. If payoff varies per spot, we make it `payoff[hand]` instead of `payoff[terminal]`. Then `fold_amount_win` becomes `[num_fold_terminals × total_hands]`.

This is the cleanest approach. Update the fold and showdown kernels to use per-(terminal, hand) payoffs.

**Step 3: Implement the build method**

The `build` method:
1. Build PostFlopGame from first spot → FlatTree (topology)
2. For each spot, build its PostFlopGame → FlatTree (per-hand data)
3. Verify all spots have same topology (same num_nodes, num_infosets, etc.)
4. Concatenate per-hand arrays:
   - `initial_reach_oop/ip`: concat each spot's reach arrays
   - `hand_strengths_oop/ip`: concat
   - `valid_matchups`: build block-diagonal matrix
   - Fold/showdown payoffs: build per-(terminal, hand) arrays

**Step 4: Run test, commit**

```bash
git commit -am "feat(gpu-solver): add BatchPerHandData assembly"
```

---

## Task 4: Update Terminal Kernels for Per-Hand Payoffs

**Files:**
- Modify: `crates/gpu-solver/kernels/terminal_fold_eval.cu`
- Modify: `crates/gpu-solver/kernels/terminal_showdown_eval.cu`
- Modify: `crates/gpu-solver/src/gpu.rs`
- Test: inline tests

The batch solver needs per-(terminal, hand) payoffs because different spots have different pots.

**Step 1: Update fold kernel**

Change `fold_amount_win` from `[num_fold_terminals]` to `[num_fold_terminals * num_hands]`:

```cuda
extern "C" __global__ void terminal_fold_eval(
    float* cfvalues,
    const float* opp_reach,
    const unsigned int* terminal_nodes,
    const float* fold_amount_win,    // NOW [num_fold_terminals * num_hands]
    const float* fold_amount_lose,   // NOW [num_fold_terminals * num_hands]
    const unsigned int* fold_player,
    const float* valid_matchups,
    unsigned int traverser,
    unsigned int num_fold_terminals,
    unsigned int num_hands
) {
    // ... same as before but:
    float win = fold_amount_win[term_idx * num_hands + hand];
    float lose = fold_amount_lose[term_idx * num_hands + hand];
    // ... rest unchanged
}
```

**Step 2: Update showdown kernel similarly**

Change `amount_win/lose` from `[num_showdown_terminals]` to `[num_showdown_terminals * num_hands]`.

**Step 3: Update launch methods in gpu.rs**

The signatures don't change (they already take `CudaSlice<f32>`) — just the buffer sizes change.

**Step 4: Update existing GpuSolver**

In `solver.rs`, when building fold/showdown payoff buffers, broadcast the scalar payoff to all hands:
```rust
// Old: fold_win.push(payoff[0]);
// New: for each hand, push the same payoff
for _ in 0..num_hands {
    fold_win.push(payoff[0]);
}
```

**Step 5: Run all existing tests to verify no regression**

Run: `cargo test -p poker-solver-gpu --features cuda`

**Step 6: Commit**

```bash
git commit -am "feat(gpu-solver): update terminal kernels for per-hand payoffs"
```

---

## Task 5: BatchGpuSolver Implementation

**Files:**
- Modify: `crates/gpu-solver/src/batch.rs`
- Test: inline tests

**Step 1: Write the failing test**

```rust
#[test]
fn test_batch_solver_3_spots() {
    let spots = make_test_spots(3);
    let config = BatchConfig::default();
    let gpu = GpuContext::new(0).unwrap();

    let mut solver = BatchGpuSolver::new(&gpu, &spots, &config).unwrap();
    let result = solver.solve(500).unwrap();

    assert_eq!(result.strategies.len(), 3);
    // Each strategy should have valid probabilities
    for strat in &result.strategies {
        assert!(!strat.is_empty());
    }
}
```

**Step 2: Implement BatchGpuSolver**

The BatchGpuSolver is essentially a GpuSolver that operates on concatenated hands. It:

1. **`new()`**: Build BatchPerHandData, upload to GPU (same pattern as GpuSolver::new but with total_hands)
2. **`solve()`**: Run the same DCFR+ iteration loop as GpuSolver, but with `num_hands = total_hands`
3. **Result extraction**: Slice the output strategy into per-spot chunks

The key insight: **the existing kernels work unchanged** because they parallelize over `(node, hand)` pairs. With `num_hands = total_hands`, each kernel launch processes all spots simultaneously.

The only complication is the valid_matchups matrix. Currently it's `[num_hands × num_hands]` which would be `[total_hands × total_hands]` — potentially huge (300 spots × 300 hands = 90K hands → 8.1 billion entries). This is way too large.

**Solution for card blocking**: Make valid_matchups block-diagonal. Each spot's hands only interact with hands from the same spot. The kernel already iterates `for opp in 0..num_hands`, but we need to restrict it to the current spot's hands.

Simplest fix: pass `hands_per_spot` to the kernel and compute `spot_idx = hand / hands_per_spot`. Then only sum over opponents in the same spot: `opp_start = spot_idx * hands_per_spot`, `opp_end = opp_start + hands_per_spot`.

Update the fold and showdown kernels to accept `hands_per_spot` and restrict the opponent loop.

**Step 3: Implement solve loop**

Reuse the same iteration structure from GpuSolver::solve(). The only differences:
- `num_hands = total_hands` (much larger)
- Terminal kernels use per-hand payoffs and spot-scoped opponent loops
- Result extraction splits strategy by spot

**Step 4: Run test, commit**

```bash
git commit -am "feat(gpu-solver): implement BatchGpuSolver"
```

---

## Task 6: Correctness Validation

**Files:**
- Create: `crates/gpu-solver/tests/bench_batch.rs`

**Step 1: Write the comparison test**

Solve 50 random spots:
- Batch on GPU (BatchGpuSolver, 1000 iterations)
- Sequentially on CPU (range_solver::solve, 1000 iterations)
- Compare per-spot strategies: max diff < 0.01, dominant action agreement > 95%

Use the cfvnet sampler to generate random situations:
```rust
use cfvnet::datagen::sampler::{sample_situation, Situation};
use cfvnet::config::DatagenConfig;
```

Or just generate random boards/ranges manually as in the existing debug_compare tests.

**Step 2: Write the performance benchmark**

Solve 300 spots:
- Batch on GPU with timing
- Sequential on CPU with timing (and optionally rayon-parallel CPU)
- Print: total GPU time, total CPU time, speedup, spots/second

**Step 3: Run tests**

```bash
cargo test -p poker-solver-gpu --features cuda --release --test bench_batch -- --nocapture
```

**Step 4: Commit**

```bash
git commit -am "test(gpu-solver): add batch correctness validation and benchmark"
```

---

## Task 7: Update Beans

**Step 1: Update Phase 2 bean status**

```bash
beans update poker_solver_rust-9v41 -s in-progress
```

**Step 2: Commit**

```bash
git add .beans/ && git commit -m "beans: update Phase 2 status to in-progress"
```
