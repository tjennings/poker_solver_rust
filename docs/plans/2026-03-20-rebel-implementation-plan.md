# ReBeL Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Implement ReBeL (Recursive Belief-based Learning) for HUNL poker, seeded from a trained blueprint V2 — combining depth-limited CFR subgame solving with a learned value network that predicts counterfactual values at any public belief state.

**Architecture:** New `rebel` crate with PBS (Public Belief State) representation, blueprint-seeded data generation, disk-backed reservoir buffer, and integration with existing `range-solver` (DCFR) + `cfvnet` (value network model + training). Bottom-up offline seeding (river → turn → flop → preflop) bootstraps the value net from blueprint play, then a live self-play loop refines it via Algorithm 1 from the paper.

**Tech Stack:** Rust, Burn 0.16 (CUDA backend), range-solver (DCFR with depth boundaries), cfvnet (CfvNet model 2720→1326, TrainingRecord format, training pipeline), serde/YAML config, memmap2 for disk buffer, rayon for parallelism.

---

## Prerequisites

- A trained blueprint V2 available (strategy bundle + bucket files)
- Working test suite completing in < 1 minute: `cargo test`
- Clean working tree

## Key Existing Infrastructure

| Component | Location | Used For |
|-----------|----------|----------|
| `PostFlopGame` | `crates/range-solver/src/game/` | DCFR solver with depth boundary support |
| `set_boundary_cfvs()` | `crates/range-solver/src/game/interpreter.rs:339` | Setting external CFVs at depth boundaries |
| `TreeConfig.depth_limit` | `crates/range-solver/src/action_tree.rs:192` | Limiting tree to N street transitions |
| `PLAYER_DEPTH_BOUNDARY_FLAG` | `crates/range-solver/src/action_tree.rs` | Marks depth boundary terminals (value=40) |
| `CfvNet` model | `crates/cfvnet/src/model/network.rs` | 2720→7×500→1326 MLP with BatchNorm+PReLU |
| `TrainingRecord` | `crates/cfvnet/src/datagen/storage.rs:25` | Binary training data format (~16KB/record) |
| `write_record`/`read_record` | `crates/cfvnet/src/datagen/storage.rs` | TrainingRecord binary I/O |
| `encode_record` | `crates/cfvnet/src/model/dataset.rs:149` | TrainingRecord → CfvItem (2720 input) |
| `train()` | `crates/cfvnet/src/model/training.rs:538` | Full training pipeline with streaming dataloader |
| `LeafEvaluator` trait | `crates/core/src/blueprint_v2/cfv_subgame_solver.rs:30` | Interface for neural leaf evaluation |
| `SharedRiverNetEvaluator` | `crates/cfvnet/src/eval/river_net_evaluator.rs:75` | GPU-batched LeafEvaluator impl |
| `BlueprintV2Strategy` | `crates/core/src/blueprint_v2/bundle.rs:35` | Strategy lookup: `get_action_probs(decision_idx, bucket)` |
| `AllBuckets` | `crates/core/src/blueprint_v2/mccfr.rs:88` | Bucket lookup: `get_bucket(street, hole_cards, board)` |
| `GameTree` | `crates/core/src/blueprint_v2/game_tree.rs` | Blueprint game tree with Decision/Chance/Terminal nodes |
| `solve()` | `crates/range-solver/src/solver.rs:105` | Main DCFR solve entry point |
| `card_pair_to_index` | `crates/range-solver/src/card.rs` | Maps two cards (0-51) to combo index (0-1325) |
| `solve_situation` | `crates/cfvnet/src/datagen/solver.rs:45` | End-to-end situation → SolveResult |
| `CfvSubgameSolver` | `crates/core/src/blueprint_v2/cfv_subgame_solver.rs:149` | Depth-bounded solver with integrated LeafEvaluator |

## Card Encoding

Throughout this plan, cards use the **range-solver encoding**: `card_id = 4 * rank + suit` where rank ∈ [0,12] (2 through A) and suit ∈ [0,3]. Combo indices use `card_pair_to_index(c1, c2)` → [0, 1325] for unordered pairs. The 1326 combos = C(52,2).

---

## Phase 1: Foundation

### Task 1: Create rebel crate scaffold

**Files:**
- Create: `crates/rebel/Cargo.toml`
- Create: `crates/rebel/src/lib.rs`
- Modify: `Cargo.toml` (workspace root, line 3)

**Step 1: Create Cargo.toml**

```toml
[package]
name = "rebel"
version.workspace = true
edition.workspace = true
license.workspace = true
authors.workspace = true

[dependencies]
poker-solver-core = { path = "../core" }
range-solver = { path = "../range-solver" }
cfvnet = { path = "../cfvnet" }

serde.workspace = true
serde_yaml.workspace = true
rayon.workspace = true
rs_poker.workspace = true

rand = "0.9"
rand_chacha = "0.9"
memmap2 = "0.9"
bytemuck = { version = "1", features = ["derive"] }

[dev-dependencies]
tempfile = "3"
```

**Step 2: Create lib.rs**

```rust
pub mod config;
pub mod pbs;
pub mod belief_update;
```

Create empty stub files for each module (just `// TODO` placeholder).

**Step 3: Add to workspace**

In root `Cargo.toml`, add `"crates/rebel"` to the `members` list (line 3).

**Step 4: Verify**

Run: `cargo check -p rebel`
Expected: compiles with no errors

**Step 5: Commit**

```bash
git add crates/rebel/ Cargo.toml
git commit -m "feat(rebel): scaffold rebel crate with dependencies"
```

---

### Task 2: PBS struct

**Files:**
- Create: `crates/rebel/src/pbs.rs`

**Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbs_new_uniform() {
        let board = vec![0u8, 4, 8, 12, 16];
        let pbs = Pbs::new_uniform(board.clone(), 200, 400);

        assert_eq!(pbs.board, board);
        assert_eq!(pbs.pot, 200);
        assert_eq!(pbs.effective_stack, 400);

        // Non-blocked combos should have reach 1.0
        // Combo of cards 20 and 24 (not on board) should be 1.0
        let idx = combo_index(20, 24);
        assert_eq!(pbs.reach_probs[0][idx], 1.0);
        assert_eq!(pbs.reach_probs[1][idx], 1.0);
    }

    #[test]
    fn test_pbs_blocks_board_combos() {
        let board = vec![0u8, 4, 8, 12, 16];
        let pbs = Pbs::new_uniform(board.clone(), 200, 400);

        // Any combo containing a board card should be 0.0
        for &bc in &board {
            for other in 0..52u8 {
                if other != bc && !board.contains(&other) {
                    let idx = combo_index(bc.min(other), bc.max(other));
                    assert_eq!(pbs.reach_probs[0][idx], 0.0,
                        "combo ({bc}, {other}) should be blocked by board");
                }
            }
        }
    }

    #[test]
    fn test_pbs_beliefs_normalize() {
        let board = vec![0u8, 4, 8];
        let pbs = Pbs::new_uniform(board, 100, 300);
        let beliefs = pbs.beliefs(0);
        let sum: f32 = beliefs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "beliefs should sum to 1.0, got {sum}");
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p rebel pbs::tests`
Expected: FAIL — `Pbs` not defined

**Step 3: Implement**

```rust
use range_solver::card::card_pair_to_index;

pub const NUM_COMBOS: usize = 1326;

/// Compute combo index for two cards in range-solver encoding (0-51).
/// Cards can be in any order.
pub fn combo_index(c1: u8, c2: u8) -> usize {
    card_pair_to_index(c1, c2)
}

/// Public Belief State: public game state + reach probability distributions.
///
/// reach_probs[player][combo] represents the probability that `player` holds
/// `combo` AND has reached this point in the hand, given observed actions.
#[derive(Clone)]
pub struct Pbs {
    /// Board cards in range-solver encoding (4*rank + suit). Length 0-5.
    pub board: Vec<u8>,
    /// Pot size in chips.
    pub pot: i32,
    /// Effective stack remaining in chips.
    pub effective_stack: i32,
    /// Reach probabilities: reach_probs[player][combo_index].
    pub reach_probs: Box<[[f32; NUM_COMBOS]; 2]>,
}

impl Pbs {
    /// Create a PBS with uniform reach (1.0 for all non-blocked combos).
    pub fn new_uniform(board: Vec<u8>, pot: i32, effective_stack: i32) -> Self {
        let mut pbs = Self {
            board,
            pot,
            effective_stack,
            reach_probs: Box::new([[1.0; NUM_COMBOS]; 2]),
        };
        pbs.zero_blocked_combos();
        pbs
    }

    /// Zero reach for combos sharing any card with the board.
    pub fn zero_blocked_combos(&mut self) {
        for &board_card in &self.board {
            for other in 0..52u8 {
                if other == board_card { continue; }
                let idx = combo_index(board_card, other);
                self.reach_probs[0][idx] = 0.0;
                self.reach_probs[1][idx] = 0.0;
            }
        }
        // Zero self-pairs (card with itself) — impossible hands
        for c in 0..52u8 {
            // card_pair_to_index requires c1 != c2, so skip
        }
    }

    /// Convert reach probabilities to normalized beliefs (sum to 1.0).
    pub fn beliefs(&self, player: usize) -> Vec<f32> {
        let reach = &self.reach_probs[player];
        let sum: f32 = reach.iter().sum();
        if sum <= 0.0 {
            return vec![0.0; NUM_COMBOS];
        }
        reach.iter().map(|&r| r / sum).collect()
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p rebel pbs::tests`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/rebel/src/pbs.rs
git commit -m "feat(rebel): PBS struct with reach probabilities and card blocking"
```

---

### Task 3: RebelConfig

**Files:**
- Create: `crates/rebel/src/config.rs`
- Create: `sample_configurations/rebel_river_seed.yaml`

**Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_config() {
        let yaml = r#"
blueprint_path: "/data/blueprint"
cluster_dir: "/data/clusters"
output_dir: "/data/rebel"

game:
  initial_stack: 400
  small_blind: 1
  big_blind: 2

seed:
  num_hands: 1000000
  seed: 42
  threads: 16
  solver_iterations: 1024
  target_exploitability: 0.005
  bet_sizes:
    flop: [[0.33, 0.67, 1.0], [0.33, 0.67, 1.0]]
    turn: [[0.5, 0.75, 1.0], [0.5, 0.75, 1.0]]
    river: [[0.5, 0.75, 1.0], [0.5, 0.75, 1.0]]

training:
  hidden_layers: 7
  hidden_size: 500
  batch_size: 4096
  epochs: 200
  learning_rate: 0.0003

buffer:
  max_records: 12000000
  path: "rebel_buffer.bin"
"#;
        let config: RebelConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.seed.num_hands, 1_000_000);
        assert_eq!(config.seed.solver_iterations, 1024);
        assert_eq!(config.training.hidden_layers, 7);
        assert_eq!(config.buffer.max_records, 12_000_000);
        assert_eq!(config.game.initial_stack, 400);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p rebel config::tests`
Expected: FAIL

**Step 3: Implement**

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RebelConfig {
    pub blueprint_path: String,
    pub cluster_dir: String,
    pub output_dir: String,
    pub game: GameConfig,
    pub seed: SeedConfig,
    pub training: TrainingConfig,
    pub buffer: BufferConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GameConfig {
    pub initial_stack: i32,
    #[serde(default = "default_sb")]
    pub small_blind: i32,
    #[serde(default = "default_bb")]
    pub big_blind: i32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SeedConfig {
    pub num_hands: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_threads")]
    pub threads: usize,
    #[serde(default = "default_solver_iters")]
    pub solver_iterations: u32,
    #[serde(default = "default_target_expl")]
    pub target_exploitability: f32,
    pub bet_sizes: BetSizeConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BetSizeConfig {
    pub flop: [Vec<f64>; 2],
    pub turn: [Vec<f64>; 2],
    pub river: [Vec<f64>; 2],
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainingConfig {
    #[serde(default = "default_layers")]
    pub hidden_layers: usize,
    #[serde(default = "default_hidden")]
    pub hidden_size: usize,
    #[serde(default = "default_batch")]
    pub batch_size: usize,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default = "default_lr")]
    pub learning_rate: f64,
    #[serde(default = "default_huber")]
    pub huber_delta: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BufferConfig {
    #[serde(default = "default_max_records")]
    pub max_records: usize,
    pub path: String,
}

fn default_sb() -> i32 { 1 }
fn default_bb() -> i32 { 2 }
fn default_seed() -> u64 { 42 }
fn default_threads() -> usize { 16 }
fn default_solver_iters() -> u32 { 1024 }
fn default_target_expl() -> f32 { 0.005 }
fn default_layers() -> usize { 7 }
fn default_hidden() -> usize { 500 }
fn default_batch() -> usize { 4096 }
fn default_epochs() -> usize { 200 }
fn default_lr() -> f64 { 3e-4 }
fn default_huber() -> f64 { 1.0 }
fn default_max_records() -> usize { 12_000_000 }
```

**Step 4: Run test**

Run: `cargo test -p rebel config::tests`
Expected: PASS

**Step 5: Create sample config file**

Save a complete `sample_configurations/rebel_river_seed.yaml` with realistic defaults for river offline seeding.

**Step 6: Commit**

```bash
git add crates/rebel/src/config.rs sample_configurations/rebel_river_seed.yaml
git commit -m "feat(rebel): RebelConfig with YAML deserialization and sample config"
```

---

### Task 4: Belief updates

**Files:**
- Create: `crates/rebel/src/belief_update.rs`

The belief update implements: after player P takes action A at a decision node, for every combo i, `reach[P][i] *= strategy(bucket(i))[A]`. All combos in the same bucket get the same action probability from the blueprint.

**Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_reach_basic() {
        // 3 combos, 2 buckets, 2 actions
        // combo 0 → bucket 0, combo 1 → bucket 0, combo 2 → bucket 1
        let combo_buckets: Vec<u16> = vec![0, 0, 1];
        // bucket 0: action probs [0.7, 0.3]
        // bucket 1: action probs [0.4, 0.6]
        let action_probs_per_bucket: Vec<Vec<f32>> = vec![
            vec![0.7, 0.3],
            vec![0.4, 0.6],
        ];
        let mut reach = vec![1.0f32; 3];
        let action_taken = 0; // first action

        update_reach(&mut reach, &combo_buckets, &action_probs_per_bucket, action_taken);

        assert!((reach[0] - 0.7).abs() < 1e-6); // bucket 0, action 0 → 0.7
        assert!((reach[1] - 0.7).abs() < 1e-6); // bucket 0, action 0 → 0.7
        assert!((reach[2] - 0.4).abs() < 1e-6); // bucket 1, action 0 → 0.4
    }

    #[test]
    fn test_update_reach_preserves_zero() {
        let combo_buckets: Vec<u16> = vec![0, 0];
        let action_probs: Vec<Vec<f32>> = vec![vec![0.5, 0.5]];
        let mut reach = vec![0.0, 1.0]; // combo 0 is blocked
        update_reach(&mut reach, &combo_buckets, &action_probs, 0);

        assert_eq!(reach[0], 0.0); // stays zero (blocked combo)
        assert!((reach[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_update_reach_sequential() {
        // Two sequential updates simulate two actions in a hand
        let combo_buckets: Vec<u16> = vec![0, 1];
        let probs: Vec<Vec<f32>> = vec![vec![0.8, 0.2], vec![0.3, 0.7]];
        let mut reach = vec![1.0, 1.0];

        update_reach(&mut reach, &combo_buckets, &probs, 0); // action 0
        update_reach(&mut reach, &combo_buckets, &probs, 1); // action 1

        // combo 0: 1.0 * 0.8 * 0.2 = 0.16
        // combo 1: 1.0 * 0.3 * 0.7 = 0.21
        assert!((reach[0] - 0.16).abs() < 1e-6);
        assert!((reach[1] - 0.21).abs() < 1e-6);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p rebel belief_update::tests`
Expected: FAIL

**Step 3: Implement**

```rust
/// Update reach probabilities after an action is taken.
///
/// For each combo i: reach[i] *= action_probs_per_bucket[bucket(i)][action_taken]
///
/// # Arguments
/// - `reach`: mutable reach probabilities (one per combo)
/// - `combo_buckets`: bucket index for each combo
/// - `action_probs_per_bucket`: action_probs_per_bucket[bucket][action] = probability
/// - `action_taken`: index of the action that was taken
pub fn update_reach(
    reach: &mut [f32],
    combo_buckets: &[u16],
    action_probs_per_bucket: &[Vec<f32>],
    action_taken: usize,
) {
    for (i, r) in reach.iter_mut().enumerate() {
        if *r == 0.0 { continue; }
        let bucket = combo_buckets[i] as usize;
        *r *= action_probs_per_bucket[bucket][action_taken];
    }
}

/// Update reach using a flat strategy array from BlueprintV2Strategy.
///
/// `strategy.get_action_probs(decision_idx, bucket)` returns &[f32] of length num_actions.
/// This function calls it per-bucket and applies the update.
pub fn update_reach_from_blueprint(
    reach: &mut [f32; 1326],
    combo_buckets: &[u16; 1326],
    strategy: &poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy,
    decision_idx: usize,
    action_taken: usize,
) {
    // Cache action probs per bucket to avoid repeated lookups
    let mut cache: std::collections::HashMap<u16, f32> = std::collections::HashMap::new();

    for (i, r) in reach.iter_mut().enumerate() {
        if *r == 0.0 { continue; }
        let bucket = combo_buckets[i];
        let prob = *cache.entry(bucket).or_insert_with(|| {
            let probs = strategy.get_action_probs(decision_idx, bucket);
            if action_taken < probs.len() {
                probs[action_taken]
            } else {
                0.0
            }
        });
        *r *= prob;
    }
}
```

**Step 4: Run test**

Run: `cargo test -p rebel belief_update::tests`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/rebel/src/belief_update.rs
git commit -m "feat(rebel): belief update — multiply reach by action probabilities per bucket"
```

---

## Phase 2: Blueprint Sampler + Data Buffer

### Task 5: Blueprint sampler

**Files:**
- Create: `crates/rebel/src/blueprint_sampler.rs`

The sampler loads a trained blueprint, plays hands under blueprint policy, and snapshots PBSs at street boundaries. It is the key data source for offline seeding.

**Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deal_and_compute_buckets() {
        // Test that deal_hand returns valid cards with no duplicates
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let deal = deal_hand(&mut rng);

        // 9 unique cards: 2 hole cards per player + 5 board
        let mut all_cards: Vec<u8> = Vec::new();
        all_cards.extend_from_slice(&deal.hole_cards[0]);
        all_cards.extend_from_slice(&deal.hole_cards[1]);
        all_cards.extend_from_slice(&deal.board);
        all_cards.sort();
        all_cards.dedup();
        assert_eq!(all_cards.len(), 9, "all 9 cards must be unique");

        // All cards in valid range
        for &c in &all_cards {
            assert!(c < 52, "card {c} out of range");
        }
    }

    #[test]
    fn test_play_hand_produces_pbs_snapshots() {
        // This is an integration test that requires a MockStrategy.
        // In production, use a trained blueprint.
        let strategy = MockStrategy::uniform(3); // 3 actions, uniform
        let tree = MockGameTree::simple_preflop_flop();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let deal = deal_hand(&mut rng);
        let snapshots = play_hand(&strategy, &tree, &deal, &mut rng);

        // Should have at least one PBS snapshot (at the first street boundary)
        assert!(!snapshots.is_empty());
        for pbs in &snapshots {
            // Board should have cards
            assert!(!pbs.board.is_empty());
            // Reach probs should have some non-zero values
            let sum: f32 = pbs.reach_probs[0].iter().sum();
            assert!(sum > 0.0, "OOP reach should have non-zero mass");
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p rebel blueprint_sampler::tests`
Expected: FAIL

**Step 3: Implement**

The sampler needs these components:

```rust
use crate::pbs::{Pbs, NUM_COMBOS, combo_index};
use crate::belief_update::update_reach_from_blueprint;
use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::game_tree::{GameTree, GameNode, TreeAction};
use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
use poker_solver_core::blueprint_v2::Street;
use rand::Rng;

/// A dealt hand: hole cards for both players + full 5-card board.
pub struct Deal {
    pub hole_cards: [[u8; 2]; 2], // [OOP, IP] hole cards in range-solver encoding
    pub board: [u8; 5],           // full board (flop + turn + river)
}

/// Deal a random hand: 2 hole cards per player + 5 board cards.
pub fn deal_hand<R: Rng>(rng: &mut R) -> Deal {
    let mut deck: Vec<u8> = (0..52).collect();
    // Fisher-Yates shuffle first 9 positions
    for i in 0..9 {
        let j = rng.random_range(i..52);
        deck.swap(i, j);
    }
    Deal {
        hole_cards: [[deck[0], deck[1]], [deck[2], deck[3]]],
        board: [deck[4], deck[5], deck[6], deck[7], deck[8]],
    }
}

/// Precompute bucket for each combo given a board and street.
///
/// Returns [u16; 1326] where combo_buckets[combo_index] = bucket.
/// Combos blocked by the board get bucket 0 (irrelevant since reach = 0).
pub fn compute_combo_buckets(
    buckets: &AllBuckets,
    street: Street,
    board: &[u8],
) -> [u16; 1326] {
    let mut result = [0u16; 1326];
    for c1 in 0..52u8 {
        if board.contains(&c1) { continue; }
        for c2 in (c1 + 1)..52u8 {
            if board.contains(&c2) { continue; }
            let idx = combo_index(c1, c2);
            // Convert range-solver card encoding to rs_poker Card for bucket lookup
            let cards = cards_from_rs(c1, c2); // helper to convert encoding
            result[idx] = buckets.get_bucket(street, cards, board);
        }
    }
    result
}

/// Play one hand under blueprint policy, returning PBS snapshots at street boundaries.
///
/// Traverses the blueprint game tree, sampling actions from the strategy.
/// At each decision node, updates ALL combos' reach probabilities.
/// At chance nodes (street boundaries), snapshots the PBS.
pub fn play_hand<R: Rng>(
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    buckets: &AllBuckets,
    deal: &Deal,
    initial_stack: i32,
    rng: &mut R,
) -> Vec<Pbs> {
    let mut snapshots = Vec::new();
    let mut reach: [[f32; NUM_COMBOS]; 2] = [[1.0; NUM_COMBOS]; 2];

    // Zero combos blocked by hole cards (each player can't have opponent's cards)
    // Note: we track beliefs from an EXTERNAL observer's perspective,
    // so we do NOT zero opponent's cards. All combos start at 1.0
    // except those blocked by known board cards (handled per-street below).

    let mut decision_idx: usize = 0;
    let mut invested = [0i32; 2]; // track pot contributions

    // Apply blinds
    invested[0] = 1; // SB = 1 chip (OOP posts SB in HUNL)
    invested[1] = 2; // BB = 2 chips (IP posts BB)

    play_node(
        strategy, tree, buckets, deal, initial_stack,
        tree.root as usize, &mut reach, &mut decision_idx,
        &mut invested, &mut snapshots, rng,
    );

    snapshots
}

fn play_node<R: Rng>(
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    buckets: &AllBuckets,
    deal: &Deal,
    initial_stack: i32,
    node_idx: usize,
    reach: &mut [[f32; NUM_COMBOS]; 2],
    decision_idx: &mut usize,
    invested: &mut [i32; 2],
    snapshots: &mut Vec<Pbs>,
    rng: &mut R,
) {
    match &tree.nodes[node_idx] {
        GameNode::Terminal { .. } => {
            // Hand is over, no more snapshots
        }
        GameNode::Chance { next_street, child } => {
            // Street boundary — snapshot PBS
            let board_cards = match next_street {
                Street::Flop => 3,
                Street::Turn => 4,
                Street::River => 5,
                _ => 0,
            };
            let board = deal.board[..board_cards].to_vec();
            let pot = invested[0] + invested[1];
            let eff_stack = initial_stack - invested[0].max(invested[1]);

            // Zero board-blocked combos in reach
            let mut pbs_reach = Box::new(*reach);
            for &bc in &board {
                for other in 0..52u8 {
                    if other == bc { continue; }
                    let idx = combo_index(bc, other);
                    pbs_reach[0][idx] = 0.0;
                    pbs_reach[1][idx] = 0.0;
                }
            }

            snapshots.push(Pbs {
                board,
                pot,
                effective_stack: eff_stack,
                reach_probs: pbs_reach,
            });

            // Continue into next street
            play_node(
                strategy, tree, buckets, deal, initial_stack,
                *child as usize, reach, decision_idx, invested, snapshots, rng,
            );
        }
        GameNode::Decision { player, street, actions, children } => {
            let p = *player as usize;

            // Determine board for current street
            let board_len = match street {
                Street::Preflop => 0,
                Street::Flop => 3,
                Street::Turn => 4,
                Street::River => 5,
            };
            let board = &deal.board[..board_len];

            // Get bucket for the dealt hand
            let actual_cards = deal.hole_cards[p];
            let actual_bucket = buckets.get_bucket(*street, cards_from_rs(actual_cards[0], actual_cards[1]), board);

            // Get action probabilities for the actual hand's bucket
            let action_probs = strategy.get_action_probs(*decision_idx, actual_bucket);
            let num_actions = actions.len();

            // Sample an action
            let r: f32 = rng.random();
            let mut cumulative = 0.0;
            let mut action_idx = num_actions - 1;
            for (a, &prob) in action_probs.iter().enumerate() {
                cumulative += prob;
                if r < cumulative {
                    action_idx = a;
                    break;
                }
            }

            // Update reach for ALL combos of the acting player
            let combo_bkts = compute_combo_buckets(buckets, *street, board);
            update_reach_from_blueprint(
                &mut reach[p], &combo_bkts, strategy, *decision_idx, action_idx,
            );

            // Update investment tracking based on action taken
            update_invested(invested, &actions[action_idx], p, initial_stack);

            // Advance decision index (counted across ALL decision nodes in tree order)
            *decision_idx += 1;

            // Recurse into chosen child
            play_node(
                strategy, tree, buckets, deal, initial_stack,
                children[action_idx] as usize, reach, decision_idx, invested, snapshots, rng,
            );
        }
    }
}

/// Update invested amounts based on the action taken.
fn update_invested(invested: &mut [i32; 2], action: &TreeAction, player: usize, initial_stack: i32) {
    match action {
        TreeAction::Fold | TreeAction::Check => {}
        TreeAction::Call => {
            invested[player] = invested[1 - player]; // match opponent's investment
        }
        TreeAction::Bet(amount) => {
            invested[player] += (*amount as i32);
        }
        TreeAction::Raise(total) => {
            invested[player] = *total as i32;
        }
        TreeAction::AllIn => {
            invested[player] = initial_stack;
        }
    }
}
```

**Important:** The `cards_from_rs(c1, c2)` helper converts range-solver card encoding (u8) to whatever format `AllBuckets::get_bucket` expects. Check the existing conversion utilities in `poker_solver_core::card_utils` and `cfvnet::model::dataset`.

**Note on decision_idx tracking:** The `decision_idx` must match how `BlueprintV2Strategy` indexes decisions. Verify by checking `BlueprintV2Strategy::node_action_counts` — each entry corresponds to one decision node in tree traversal order. If the tree is traversed depth-first (preorder), increment `decision_idx` at each Decision node visited. If the blueprint uses a different traversal order, match that order.

**Step 4: Run test**

Run: `cargo test -p rebel blueprint_sampler::tests`
Expected: PASS (unit tests with mock; integration test may need real blueprint)

**Step 5: Commit**

```bash
git add crates/rebel/src/blueprint_sampler.rs
git commit -m "feat(rebel): blueprint sampler — play hands under blueprint policy, snapshot PBSs"
```

---

### Task 6: Disk-backed reservoir buffer

**Files:**
- Create: `crates/rebel/src/data_buffer.rs`

Fixed-size records enable mmap random access. Uses `cfvnet::datagen::storage::TrainingRecord` format with board padded to 5 cards.

**Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_buffer_append_and_count() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test_buffer.bin");
        let mut buf = DiskBuffer::create(&path, 1000).unwrap();

        assert_eq!(buf.len(), 0);

        let record = make_test_record();
        buf.append(&record).unwrap();
        assert_eq!(buf.len(), 1);

        buf.append(&record).unwrap();
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_buffer_random_sample() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test_buffer.bin");
        let mut buf = DiskBuffer::create(&path, 1000).unwrap();

        // Append 100 records with distinct pot values
        for i in 0..100 {
            let mut rec = make_test_record();
            rec.pot = i as f32;
            buf.append(&rec).unwrap();
        }

        // Sample 10 records
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(99);
        let samples = buf.sample(&mut rng, 10).unwrap();
        assert_eq!(samples.len(), 10);

        // Pot values should be in [0, 100)
        for s in &samples {
            assert!(s.pot >= 0.0 && s.pot < 100.0);
        }
    }

    #[test]
    fn test_buffer_reservoir_replacement() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test_buffer.bin");
        let mut buf = DiskBuffer::create(&path, 10).unwrap(); // max 10

        // Append 20 records
        for i in 0..20 {
            let mut rec = make_test_record();
            rec.pot = i as f32;
            buf.append(&rec).unwrap();
        }

        // Buffer should contain exactly 10 records
        assert_eq!(buf.len(), 10);
    }

    fn make_test_record() -> BufferRecord {
        BufferRecord {
            board: [0, 4, 8, 12, 16],
            board_card_count: 5,
            pot: 100.0,
            effective_stack: 300.0,
            player: 0,
            game_value: 0.0,
            oop_reach: [0.5; 1326],
            ip_reach: [0.5; 1326],
            cfvs: [0.0; 1326],
            valid_mask: [1u8; 1326],
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p rebel data_buffer::tests`
Expected: FAIL

**Step 3: Implement**

```rust
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use memmap2::Mmap;
use rand::Rng;

/// Fixed-size buffer record for mmap random access.
/// Layout is compatible with cfvnet TrainingRecord semantics.
#[derive(Clone)]
#[repr(C)]
pub struct BufferRecord {
    pub board: [u8; 5],
    pub board_card_count: u8,
    pub pot: f32,
    pub effective_stack: f32,
    pub player: u8,
    pub game_value: f32,
    pub oop_reach: [f32; 1326],
    pub ip_reach: [f32; 1326],
    pub cfvs: [f32; 1326],
    pub valid_mask: [u8; 1326],
}

impl BufferRecord {
    /// Byte size of one record on disk.
    pub const SIZE: usize = 5 + 1 + 4 + 4 + 1 + 4 + (1326 * 4) * 3 + 1326;
    // = 5 + 1 + 4 + 4 + 1 + 4 + 15912 + 1326 = 17257

    pub fn serialize(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.board);
        buf.push(self.board_card_count);
        buf.extend_from_slice(&self.pot.to_le_bytes());
        buf.extend_from_slice(&self.effective_stack.to_le_bytes());
        buf.push(self.player);
        buf.extend_from_slice(&self.game_value.to_le_bytes());
        for &v in &self.oop_reach { buf.extend_from_slice(&v.to_le_bytes()); }
        for &v in &self.ip_reach { buf.extend_from_slice(&v.to_le_bytes()); }
        for &v in &self.cfvs { buf.extend_from_slice(&v.to_le_bytes()); }
        buf.extend_from_slice(&self.valid_mask);
    }

    pub fn deserialize(data: &[u8]) -> Self {
        assert!(data.len() >= Self::SIZE);
        let mut pos = 0;
        let mut board = [0u8; 5];
        board.copy_from_slice(&data[pos..pos+5]); pos += 5;
        let board_card_count = data[pos]; pos += 1;
        let pot = f32::from_le_bytes(data[pos..pos+4].try_into().unwrap()); pos += 4;
        let effective_stack = f32::from_le_bytes(data[pos..pos+4].try_into().unwrap()); pos += 4;
        let player = data[pos]; pos += 1;
        let game_value = f32::from_le_bytes(data[pos..pos+4].try_into().unwrap()); pos += 4;
        let read_f32_array = |data: &[u8], pos: &mut usize| -> [f32; 1326] {
            let mut arr = [0.0f32; 1326];
            for v in &mut arr {
                *v = f32::from_le_bytes(data[*pos..*pos+4].try_into().unwrap());
                *pos += 4;
            }
            arr
        };
        let oop_reach = read_f32_array(data, &mut pos);
        let ip_reach = read_f32_array(data, &mut pos);
        let cfvs = read_f32_array(data, &mut pos);
        let mut valid_mask = [0u8; 1326];
        valid_mask.copy_from_slice(&data[pos..pos+1326]);
        Self { board, board_card_count, pot, effective_stack, player, game_value,
               oop_reach, ip_reach, cfvs, valid_mask }
    }
}

/// Disk-backed reservoir buffer with mmap random sampling.
pub struct DiskBuffer {
    path: PathBuf,
    file: File,
    count: usize,
    max_records: usize,
    total_appended: usize, // total ever appended (for reservoir sampling)
}

impl DiskBuffer {
    pub fn create(path: &Path, max_records: usize) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true).write(true).create(true).truncate(true)
            .open(path)?;
        Ok(Self { path: path.to_path_buf(), file, count: 0, max_records, total_appended: 0 })
    }

    pub fn open(path: &Path, max_records: usize) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let file_len = file.metadata()?.len() as usize;
        let count = file_len / BufferRecord::SIZE;
        Ok(Self { path: path.to_path_buf(), file, count, max_records, total_appended: count })
    }

    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }

    /// Append a record. If buffer is full, use reservoir sampling to decide
    /// whether to replace an existing record.
    pub fn append(&mut self, record: &BufferRecord) -> io::Result<()> {
        let mut buf = Vec::with_capacity(BufferRecord::SIZE);
        record.serialize(&mut buf);

        if self.count < self.max_records {
            // Buffer not full — append at end
            self.file.seek(SeekFrom::End(0))?;
            self.file.write_all(&buf)?;
            self.count += 1;
        } else {
            // Reservoir sampling: replace random record with probability max_records / total_appended
            let mut rng = rand::rng();
            let j = rng.random_range(0..self.total_appended + 1);
            if j < self.max_records {
                let offset = (j * BufferRecord::SIZE) as u64;
                self.file.seek(SeekFrom::Start(offset))?;
                self.file.write_all(&buf)?;
            }
        }
        self.total_appended += 1;
        Ok(())
    }

    /// Sample `n` random records from the buffer using mmap.
    pub fn sample<R: Rng>(&self, rng: &mut R, n: usize) -> io::Result<Vec<BufferRecord>> {
        if self.count == 0 { return Ok(Vec::new()); }
        let mmap = unsafe { Mmap::map(&self.file)? };
        let mut records = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = rng.random_range(0..self.count);
            let offset = idx * BufferRecord::SIZE;
            let data = &mmap[offset..offset + BufferRecord::SIZE];
            records.push(BufferRecord::deserialize(data));
        }
        Ok(records)
    }
}
```

**Step 4: Run test**

Run: `cargo test -p rebel data_buffer::tests`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/rebel/src/data_buffer.rs
git commit -m "feat(rebel): disk-backed reservoir buffer with mmap random sampling"
```

---

### Task 7: PBS generation pipeline (end-to-end)

**Files:**
- Create: `crates/rebel/src/generate.rs`

Wires the blueprint sampler to the disk buffer: generate N hands in parallel, snapshot PBSs, write to buffer.

**Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_pbs_count() {
        // With a mock strategy, verify that generating 100 hands produces
        // a reasonable number of PBS snapshots (at least 1 per hand on average).
        // Full integration test requires a real blueprint — see integration tests.
    }
}
```

**Step 2: Implement**

```rust
use crate::blueprint_sampler::{deal_hand, play_hand, Deal};
use crate::config::RebelConfig;
use crate::data_buffer::DiskBuffer;
use crate::pbs::Pbs;
use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::game_tree::GameTree;
use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::sync::Mutex;

/// Generate PBS snapshots from blueprint play and write to buffer.
///
/// Returns the total number of PBSs generated.
pub fn generate_pbs(
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    buckets: &AllBuckets,
    config: &RebelConfig,
    buffer: &Mutex<DiskBuffer>,
) -> usize {
    let num_hands = config.seed.num_hands;
    let initial_stack = config.game.initial_stack;
    let seed = config.seed.seed;

    let total_pbs = std::sync::atomic::AtomicUsize::new(0);

    // Parallel hand generation
    (0..num_hands).into_par_iter().for_each(|hand_idx| {
        let mut rng = ChaCha8Rng::seed_from_u64(seed + hand_idx as u64);
        let deal = deal_hand(&mut rng);
        let snapshots = play_hand(strategy, tree, buckets, &deal, initial_stack, &mut rng);

        // Write snapshots to buffer (lock per batch, not per record)
        if !snapshots.is_empty() {
            let mut buf = buffer.lock().unwrap();
            for pbs in &snapshots {
                let record = pbs_to_buffer_record(pbs, 0); // OOP perspective
                buf.append(&record).ok();
                let record = pbs_to_buffer_record(pbs, 1); // IP perspective
                buf.append(&record).ok();
            }
            total_pbs.fetch_add(snapshots.len(), std::sync::atomic::Ordering::Relaxed);
        }
    });

    total_pbs.load(std::sync::atomic::Ordering::Relaxed)
}

/// Convert a PBS to a BufferRecord.
/// `player` indicates which player's CFVs we'll later fill in (0=OOP, 1=IP).
/// CFVs are zeroed here — they'll be filled by the solver in Phase 3.
fn pbs_to_buffer_record(pbs: &Pbs, player: u8) -> crate::data_buffer::BufferRecord {
    let mut board = [0u8; 5];
    let board_count = pbs.board.len().min(5);
    board[..board_count].copy_from_slice(&pbs.board[..board_count]);

    // Build valid mask: 1 for combos not blocked by board
    let mut valid_mask = [1u8; 1326];
    for &bc in &pbs.board {
        for other in 0..52u8 {
            if other == bc { continue; }
            let idx = crate::pbs::combo_index(bc, other);
            valid_mask[idx] = 0;
        }
    }

    crate::data_buffer::BufferRecord {
        board,
        board_card_count: board_count as u8,
        pot: pbs.pot as f32,
        effective_stack: pbs.effective_stack as f32,
        player,
        game_value: 0.0, // filled after solving
        oop_reach: pbs.reach_probs[0],
        ip_reach: pbs.reach_probs[1],
        cfvs: [0.0; 1326], // filled after solving
        valid_mask,
    }
}
```

**Step 3: Run test**

Run: `cargo test -p rebel generate::tests`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/rebel/src/generate.rs
git commit -m "feat(rebel): PBS generation pipeline — parallel blueprint sampling to disk buffer"
```

---

### Task 8: rebel-seed CLI subcommand

**Files:**
- Modify: `crates/trainer/src/main.rs` (add RebelSeed variant)
- Modify: `crates/trainer/Cargo.toml` (add rebel dependency)

**Step 1: Add rebel dependency to trainer**

In `crates/trainer/Cargo.toml`, add:
```toml
rebel = { path = "../rebel" }
```

**Step 2: Add CLI variant**

In `crates/trainer/src/main.rs`, add to the `Commands` enum:

```rust
/// Generate PBS training data from blueprint play for ReBeL offline seeding
#[command(name = "rebel-seed")]
RebelSeed {
    /// Path to ReBeL YAML configuration file
    #[arg(short, long)]
    config: String,
},
```

**Step 3: Add handler**

In the main `match` block:

```rust
Commands::RebelSeed { config } => {
    let yaml = std::fs::read_to_string(&config)
        .map_err(|e| format!("Failed to read config: {e}"))?;
    let rebel_config: rebel::config::RebelConfig = serde_yaml::from_str(&yaml)
        .map_err(|e| format!("Failed to parse config: {e}"))?;

    // Load blueprint
    eprintln!("Loading blueprint from {}", rebel_config.blueprint_path);
    let strategy = BlueprintV2Strategy::load(&rebel_config.blueprint_path)
        .map_err(|e| format!("Failed to load blueprint: {e}"))?;

    // Load bucket files
    eprintln!("Loading bucket files from {}", rebel_config.cluster_dir);
    let buckets = AllBuckets::load(&rebel_config.cluster_dir)
        .map_err(|e| format!("Failed to load buckets: {e}"))?;

    // Build game tree (reuse blueprint's tree config)
    let tree = GameTree::build(/* blueprint config params */);

    // Create buffer
    let buffer_path = std::path::Path::new(&rebel_config.output_dir)
        .join(&rebel_config.buffer.path);
    let buffer = std::sync::Mutex::new(
        rebel::data_buffer::DiskBuffer::create(&buffer_path, rebel_config.buffer.max_records)
            .map_err(|e| format!("Failed to create buffer: {e}"))?
    );

    // Generate PBSs
    eprintln!("Generating {} hands...", rebel_config.seed.num_hands);
    let pbs_count = rebel::generate::generate_pbs(
        &strategy, &tree, &buckets, &rebel_config, &buffer,
    );
    eprintln!("Generated {pbs_count} PBS snapshots → {}", buffer_path.display());
}
```

**Step 4: Verify**

Run: `cargo build -p poker-solver-trainer`
Expected: compiles

**Step 5: Commit**

```bash
git add crates/trainer/src/main.rs crates/trainer/Cargo.toml
git commit -m "feat(trainer): add rebel-seed CLI subcommand for PBS generation"
```

---

## Phase 3: Offline River Seeding

### Task 9: PBS → range-solver conversion and river solving

**Files:**
- Create: `crates/rebel/src/solver.rs`

Converts a PBS (with reach probabilities as ranges) into a range-solver `PostFlopGame`, solves it, and extracts per-combo CFVs.

**Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_river_pbs() {
        // Create a river PBS with uniform reach on a specific board
        let board = vec![0u8, 4, 8, 12, 16]; // As Ah 3s 4s 5s
        let pbs = Pbs::new_uniform(board, 200, 300);

        let solve_config = SolveConfig {
            bet_sizes: default_river_bet_sizes(),
            solver_iterations: 200,
            target_exploitability: 0.01,
        };

        let result = solve_river_pbs(&pbs, &solve_config).unwrap();

        // Should have 1326 CFVs, non-blocked ones should be finite
        assert_eq!(result.oop_cfvs.len(), 1326);
        assert_eq!(result.ip_cfvs.len(), 1326);

        // Board-blocked combos should have 0 CFV
        let blocked_idx = combo_index(0, 4); // both on board
        assert_eq!(result.oop_cfvs[blocked_idx], 0.0);

        // Non-blocked combos should have non-zero CFVs (at least some)
        let non_blocked_count = result.oop_cfvs.iter()
            .filter(|&&v| v.abs() > 1e-10).count();
        assert!(non_blocked_count > 100, "most combos should have non-zero CFVs");
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p rebel solver::tests`
Expected: FAIL

**Step 3: Implement**

```rust
use crate::pbs::{Pbs, NUM_COMBOS, combo_index};
use range_solver::*;

pub struct SolveConfig {
    pub bet_sizes: Vec<Vec<f64>>,  // [OOP, IP] bet sizes as pot fractions
    pub solver_iterations: u32,
    pub target_exploitability: f32,
}

pub struct SolveResult {
    pub oop_cfvs: [f32; 1326],
    pub ip_cfvs: [f32; 1326],
    pub oop_game_value: f32,
    pub ip_game_value: f32,
    pub exploitability: f32,
}

/// Solve a river PBS using the range-solver DCFR engine.
///
/// Converts PBS reach probabilities to range-solver Range format,
/// builds a river PostFlopGame, solves it, and extracts CFVs.
pub fn solve_river_pbs(pbs: &Pbs, config: &SolveConfig) -> Result<SolveResult, String> {
    assert!(pbs.board.len() == 5, "river PBS must have 5 board cards");

    // Convert reach probabilities to Range format
    // Range expects per-combo weights in range-solver combo ordering
    let oop_range = reach_to_range(&pbs.reach_probs[0], &pbs.board);
    let ip_range = reach_to_range(&pbs.reach_probs[1], &pbs.board);

    // Build card config
    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [card_from_id(pbs.board[0]), card_from_id(pbs.board[1]),
               card_from_id(pbs.board[2])],
        turn: card_from_id(pbs.board[3]),
        river: card_from_id(pbs.board[4]),
    };

    // Build tree config for river
    let bet_sizes_oop = BetSizeOptions::try_from(config.bet_sizes[0].as_slice())?;
    let bet_sizes_ip = BetSizeOptions::try_from(config.bet_sizes[1].as_slice())?;
    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: pbs.pot,
        effective_stack: pbs.effective_stack,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [BetSizeOptions::default(); 2],
        turn_bet_sizes: [BetSizeOptions::default(); 2],
        river_bet_sizes: [bet_sizes_oop, bet_sizes_ip],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.1,
        merging_threshold: 0.0,
        depth_limit: None, // No depth limit for river — solve to showdown
    };

    // Build and solve
    let action_tree = ActionTree::new(tree_config)?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)?;
    game.allocate_memory(false);
    let exploitability = solve(
        &mut game,
        config.solver_iterations,
        config.target_exploitability,
        false, // no progress printing
    );

    // Extract CFVs per combo
    // After solving, use compute_current_ev to get per-combo expected values
    // These are the counterfactual values we need for training
    finalize(&mut game);
    let oop_evs = extract_root_cfvs(&game, 0);
    let ip_evs = extract_root_cfvs(&game, 1);

    // Map from solver combo ordering back to canonical 1326 ordering
    let mut oop_cfvs = [0.0f32; 1326];
    let mut ip_cfvs = [0.0f32; 1326];
    let solver_combos = game.private_cards(0);
    for (solver_idx, &(c1, c2)) in solver_combos.iter().enumerate() {
        let canonical_idx = combo_index(card_to_id(c1), card_to_id(c2));
        oop_cfvs[canonical_idx] = oop_evs[solver_idx] as f32;
    }
    let solver_combos_ip = game.private_cards(1);
    for (solver_idx, &(c1, c2)) in solver_combos_ip.iter().enumerate() {
        let canonical_idx = combo_index(card_to_id(c1), card_to_id(c2));
        ip_cfvs[canonical_idx] = ip_evs[solver_idx] as f32;
    }

    let oop_gv = weighted_game_value(&oop_cfvs, &pbs.reach_probs[0]);
    let ip_gv = weighted_game_value(&ip_cfvs, &pbs.reach_probs[1]);

    Ok(SolveResult { oop_cfvs, ip_cfvs, oop_game_value: oop_gv,
                     ip_game_value: ip_gv, exploitability })
}

/// Extract root counterfactual values for a player from a solved game.
fn extract_root_cfvs(game: &PostFlopGame, player: usize) -> Vec<f64> {
    // Use game.expected_values(player) or compute from cfvalues at root
    // The exact API depends on PostFlopGame — check query.rs
    game.expected_values(player).to_vec()
}

/// Compute range-weighted game value: sum(cfv[i] * reach[i]) / sum(reach[i])
fn weighted_game_value(cfvs: &[f32; 1326], reach: &[f32; 1326]) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..1326 {
        num += cfvs[i] as f64 * reach[i] as f64;
        den += reach[i] as f64;
    }
    if den > 0.0 { (num / den) as f32 } else { 0.0 }
}

/// Convert reach probabilities to range-solver Range format.
fn reach_to_range(reach: &[f32; 1326], board: &[u8]) -> Range {
    // Range::from_weights expects weights indexed by combo
    // Match the exact Range construction used in cfvnet::datagen::solver::solve_situation
    Range::from_weights(reach)
}
```

**Note:** The exact API for `extract_root_cfvs` depends on `PostFlopGame`'s query methods. Check `crates/range-solver/src/game/query.rs` — look for `expected_values()`, `equity()`, or `cfvalues()` methods. Also check how `cfvnet::datagen::solver::solve_situation` extracts EVs (lines 85-100 in `crates/cfvnet/src/datagen/solver.rs`).

**Note:** Card ID ↔ Card conversion helpers (`card_from_id`, `card_to_id`) should follow the pattern in `cfvnet::datagen::solver.rs` or `cfvnet::model::dataset.rs`.

**Step 4: Run test**

Run: `cargo test -p rebel solver::tests -- --release`
Expected: PASS (use `--release` since solving is slow in debug mode)

**Step 5: Commit**

```bash
git add crates/rebel/src/solver.rs
git commit -m "feat(rebel): PBS solver — convert PBS to range-solver, solve river, extract CFVs"
```

---

### Task 10: River solving pipeline

**Files:**
- Modify: `crates/rebel/src/generate.rs` (add solving phase)

After generating PBSs (Task 7), solve each river PBS and fill in the CFVs in the buffer records.

**Step 1: Write failing test**

```rust
#[test]
fn test_solve_and_fill_buffer() {
    // Create a buffer with unsolved records (cfvs = 0)
    // Run solve_buffer_records()
    // Verify CFVs are now non-zero for non-blocked combos
}
```

**Step 2: Implement**

```rust
/// Solve all unsolved PBS records in the buffer and fill in CFVs.
///
/// Reads records from buffer, solves each as a river subgame,
/// writes CFVs back into the record.
pub fn solve_buffer_records(
    buffer: &mut DiskBuffer,
    solve_config: &SolveConfig,
    threads: usize,
) -> usize {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    let total = buffer.len();
    let solved = std::sync::atomic::AtomicUsize::new(0);

    // Read all records, solve in parallel, write back
    // Process in chunks to limit memory
    let chunk_size = 1000;
    for chunk_start in (0..total).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(total);
        let records: Vec<_> = (chunk_start..chunk_end)
            .map(|i| buffer.read_record(i).unwrap())
            .collect();

        let solved_records: Vec<_> = pool.install(|| {
            records.into_par_iter().map(|mut rec| {
                let pbs = buffer_record_to_pbs(&rec);
                if let Ok(result) = solve_river_pbs(&pbs, solve_config) {
                    let cfvs = if rec.player == 0 { result.oop_cfvs } else { result.ip_cfvs };
                    let gv = if rec.player == 0 { result.oop_game_value } else { result.ip_game_value };
                    rec.cfvs = cfvs;
                    rec.game_value = gv;
                    solved.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                rec
            }).collect()
        });

        // Write back
        for (i, rec) in solved_records.into_iter().enumerate() {
            buffer.write_record(chunk_start + i, &rec).unwrap();
        }

        let done = solved.load(std::sync::atomic::Ordering::Relaxed);
        eprintln!("Solved {done}/{total} records");
    }

    solved.load(std::sync::atomic::Ordering::Relaxed)
}
```

**Note:** This requires adding `read_record(index)` and `write_record(index, record)` methods to `DiskBuffer` (seek to `index * RECORD_SIZE`, read/write).

**Step 3: Run test**

Run: `cargo test -p rebel generate::tests -- --release`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/rebel/src/generate.rs crates/rebel/src/data_buffer.rs
git commit -m "feat(rebel): river solving pipeline — solve PBSs in parallel, fill CFVs in buffer"
```

---

### Task 11: Training integration

**Files:**
- Create: `crates/rebel/src/training.rs`

Converts buffer records to cfvnet `TrainingRecord` format and invokes the existing cfvnet training pipeline.

**Step 1: Write failing test**

```rust
#[test]
fn test_buffer_record_to_training_record() {
    let buf_rec = make_test_solved_record();
    let train_rec = to_training_record(&buf_rec);

    assert_eq!(train_rec.board.len(), buf_rec.board_card_count as usize);
    assert_eq!(train_rec.pot, buf_rec.pot);
    assert_eq!(train_rec.player, buf_rec.player);
    assert_eq!(train_rec.cfvs, buf_rec.cfvs);
}
```

**Step 2: Implement**

```rust
use crate::data_buffer::{DiskBuffer, BufferRecord};
use cfvnet::datagen::storage::{TrainingRecord, write_record};
use std::path::Path;
use std::io::BufWriter;
use std::fs::File;

/// Convert buffer records to cfvnet TrainingRecord binary files for training.
///
/// Writes records to `output_path` in cfvnet binary format, compatible with
/// the existing cfvnet training pipeline.
pub fn export_training_data(
    buffer: &DiskBuffer,
    output_path: &Path,
) -> std::io::Result<usize> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    let mut count = 0;

    for i in 0..buffer.len() {
        let buf_rec = buffer.read_record(i)?;
        let train_rec = to_training_record(&buf_rec);
        write_record(&mut writer, &train_rec)?;
        count += 1;
    }

    Ok(count)
}

/// Convert a BufferRecord to a cfvnet TrainingRecord.
pub fn to_training_record(rec: &BufferRecord) -> TrainingRecord {
    let board_count = rec.board_card_count as usize;
    TrainingRecord {
        board: rec.board[..board_count].to_vec(),
        pot: rec.pot,
        effective_stack: rec.effective_stack,
        player: rec.player,
        game_value: rec.game_value,
        oop_range: rec.oop_reach,  // reach probs serve as range weights
        ip_range: rec.ip_reach,
        cfvs: rec.cfvs,
        valid_mask: rec.valid_mask,
    }
}
```

Then invoke cfvnet training via the existing `cfvnet::model::training::train()` function, passing the exported data path and a `TrainConfig` built from `RebelConfig::training`.

**Step 3: Run test**

Run: `cargo test -p rebel training::tests`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/rebel/src/training.rs
git commit -m "feat(rebel): training integration — export buffer to cfvnet format, invoke training"
```

---

## Phase 4: Depth-Limited Multi-Street Seeding

### Task 12: RebelLeafEvaluator

**Files:**
- Create: `crates/rebel/src/leaf_evaluator.rs`

Implements the `LeafEvaluator` trait backed by a trained CfvNet value network. This allows the range-solver to use neural net predictions at depth boundaries (e.g., turn → river boundary).

**Step 1: Write failing test**

```rust
#[test]
fn test_rebel_leaf_evaluator_returns_correct_shape() {
    // Load a trained model (or create a random one for testing)
    // Call evaluate() and verify output shape matches num_combos
}
```

**Step 2: Implement**

Follow the pattern from `crates/cfvnet/src/eval/river_net_evaluator.rs` (`SharedRiverNetEvaluator`).

```rust
use poker_solver_core::blueprint_v2::cfv_subgame_solver::LeafEvaluator;
use cfvnet::model::network::CfvNet;
use cfvnet::model::dataset::encode_situation_for_inference;
use burn::prelude::*;
use std::sync::{Arc, Mutex};

/// LeafEvaluator backed by a CfvNet value network.
///
/// Evaluates depth boundary nodes by querying the neural network.
/// Thread-safe via Arc<Mutex<CfvNet>>.
pub struct RebelLeafEvaluator<B: Backend> {
    model: Arc<Mutex<CfvNet<B>>>,
    device: B::Device,
}

impl<B: Backend> RebelLeafEvaluator<B> {
    pub fn new(model: CfvNet<B>, device: B::Device) -> Self {
        Self {
            model: Arc::new(Mutex::new(model)),
            device,
        }
    }
}

impl<B: Backend> LeafEvaluator for RebelLeafEvaluator<B> {
    fn evaluate(
        &self,
        combos: &[[Card; 2]],
        board: &[Card],
        pot: f64,
        effective_stack: f64,
        oop_range: &[f64],
        ip_range: &[f64],
        traverser: u8,
    ) -> Vec<f64> {
        // Build 2720-element input vector using cfvnet encoding
        let input = build_leaf_input(board, pot, effective_stack, oop_range, ip_range, traverser);

        // Forward pass through model
        let model = self.model.lock().unwrap();
        let input_tensor = Tensor::<B, 2>::from_floats(
            Data::new(input, Shape::new([1, 2720])),
            &self.device,
        );
        let output = model.forward(input_tensor);
        let output_data: Vec<f32> = output.to_data().to_vec().unwrap();

        // Map 1326 outputs back to solver combo ordering
        map_outputs_to_combos(&output_data, combos, board)
    }

    fn evaluate_boundaries(
        &self,
        combos: &[[Card; 2]],
        board: &[Card],
        oop_range: &[f64],
        ip_range: &[f64],
        requests: &[(f64, f64, u8)],
    ) -> Vec<Vec<f64>> {
        // Batch all requests into a single GPU forward pass
        // Follow SharedRiverNetEvaluator::evaluate_boundaries pattern
        let batch_size = requests.len();
        let mut inputs = Vec::with_capacity(batch_size * 2720);

        for &(pot, eff_stack, traverser) in requests {
            let input = build_leaf_input(board, pot, eff_stack, oop_range, ip_range, traverser);
            inputs.extend_from_slice(&input);
        }

        let model = self.model.lock().unwrap();
        let input_tensor = Tensor::<B, 2>::from_floats(
            Data::new(inputs, Shape::new([batch_size, 2720])),
            &self.device,
        );
        let output = model.forward(input_tensor);
        let output_data: Vec<f32> = output.to_data().to_vec().unwrap();

        // Split outputs per request and map to solver combo ordering
        (0..batch_size)
            .map(|i| {
                let slice = &output_data[i * 1326..(i + 1) * 1326];
                map_outputs_to_combos(slice, combos, board)
            })
            .collect()
    }
}
```

**Key difference from `SharedRiverNetEvaluator`:** The existing evaluator averages over 48 possible river cards for turn evaluation. `RebelLeafEvaluator` does NOT do this — it evaluates the PBS directly at the boundary, regardless of which street boundary it is. The value net has learned to predict CFVs for any street.

**Step 3: Run test**

Run: `cargo test -p rebel leaf_evaluator::tests`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/rebel/src/leaf_evaluator.rs
git commit -m "feat(rebel): RebelLeafEvaluator — CfvNet-backed LeafEvaluator for depth boundaries"
```

---

### Task 13: Multi-street offline seeding

**Files:**
- Modify: `crates/rebel/src/solver.rs` (add depth-limited solving)
- Modify: `crates/rebel/src/generate.rs` (add multi-street support)

Extend the solver to handle turn/flop/preflop PBSs using depth-limited solving with the value net at boundaries.

**Step 1: Write failing test**

```rust
#[test]
fn test_solve_turn_pbs_with_leaf_evaluator() {
    // Create a turn PBS (4 board cards)
    // Create a mock LeafEvaluator that returns zeros
    // Solve with depth_limit=0 (turn betting only, river boundary uses evaluator)
    // Verify CFVs are produced
}
```

**Step 2: Implement**

```rust
/// Solve a depth-limited PBS using range-solver with value net at boundaries.
///
/// `depth_limit`: 0 = current street only, 1 = current + next, etc.
/// At depth boundaries, the LeafEvaluator provides CFVs.
pub fn solve_pbs_depth_limited(
    pbs: &Pbs,
    solve_config: &SolveConfig,
    depth_limit: u8,
    evaluator: &dyn LeafEvaluator,
) -> Result<SolveResult, String> {
    let board_count = pbs.board.len();
    let initial_state = match board_count {
        0 => BoardState::Flop,   // preflop PBS, tree starts at flop? Or preflop?
        3 => BoardState::Flop,
        4 => BoardState::Turn,
        5 => BoardState::River,
        _ => return Err(format!("invalid board size: {board_count}")),
    };

    // Build tree config with depth_limit
    let tree_config = TreeConfig {
        initial_state,
        starting_pot: pbs.pot,
        effective_stack: pbs.effective_stack,
        depth_limit: Some(depth_limit),
        // ... bet sizes from config ...
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config)?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)?;
    game.allocate_memory(false);

    // Set boundary CFVs using the LeafEvaluator
    let num_boundaries = game.num_boundary_nodes();
    if num_boundaries > 0 {
        let combos_oop = game.private_cards(0);
        let combos_ip = game.private_cards(1);
        // ... build oop_range, ip_range from PBS reach probs ...

        // Use evaluate_boundaries for batch GPU evaluation
        let requests: Vec<(f64, f64, u8)> = (0..num_boundaries)
            .map(|b| {
                let pot = game.boundary_pot(b) as f64;
                let eff = pbs.effective_stack as f64; // approximate
                (pot, eff, 0u8) // OOP first
            })
            .collect();

        let board_cards = /* convert pbs.board to Card format */;
        let oop_cfvs_batch = evaluator.evaluate_boundaries(
            combos_oop, &board_cards, &oop_range_f64, &ip_range_f64, &requests,
        );

        for (b, cfvs) in oop_cfvs_batch.into_iter().enumerate() {
            game.set_boundary_cfvs(b, 0, cfvs.iter().map(|&v| v as f32).collect());
        }

        // Same for IP
        let ip_requests: Vec<_> = requests.iter().map(|&(p, e, _)| (p, e, 1u8)).collect();
        let ip_cfvs_batch = evaluator.evaluate_boundaries(
            combos_ip, &board_cards, &oop_range_f64, &ip_range_f64, &ip_requests,
        );
        for (b, cfvs) in ip_cfvs_batch.into_iter().enumerate() {
            game.set_boundary_cfvs(b, 1, cfvs.iter().map(|&v| v as f32).collect());
        }
    }

    // Solve
    let exploitability = solve(&mut game, solve_config.solver_iterations,
                                solve_config.target_exploitability, false);
    finalize(&mut game);

    // Extract CFVs (same as river solving)
    // ...
    Ok(SolveResult { /* ... */ })
}
```

**Step 3: Multi-street pipeline**

The offline seeding pipeline runs bottom-up:

```
1. Generate river PBSs → solve (no depth limit) → train river value net
2. Generate turn PBSs → solve with depth_limit=0 + river value net → retrain value net
3. Generate flop PBSs → solve with depth_limit=0 + turn/river value net → retrain value net
4. Generate preflop PBSs → solve with depth_limit=0 + flop/turn/river value net → retrain value net
```

Each iteration retrains the SAME value net on ALL accumulated data (river + turn + flop + preflop).

**Step 4: Run test**

Run: `cargo test -p rebel solver::tests -- --release`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/rebel/src/solver.rs crates/rebel/src/generate.rs
git commit -m "feat(rebel): depth-limited solving with LeafEvaluator at street boundaries"
```

---

### Task 14: Multi-street training orchestration

**Files:**
- Modify: `crates/rebel/src/training.rs`

Add the bottom-up training loop: for each street (river → turn → flop → preflop), generate data, solve, accumulate, retrain.

**Step 1: Implement**

```rust
/// Run the full offline seeding pipeline: river → turn → flop → preflop.
///
/// At each street:
/// 1. Generate PBSs from blueprint play
/// 2. Solve subgames (exact for river, depth-limited for others)
/// 3. Append training data to accumulated buffer
/// 4. Retrain value net on all accumulated data
pub fn offline_seeding_pipeline(config: &RebelConfig) -> Result<(), String> {
    let streets = [Street::River, Street::Turn, Street::Flop, Street::Preflop];

    let mut model: Option<CfvNet<_>> = None;

    for &street in &streets {
        eprintln!("=== Offline seeding: {:?} ===", street);

        // 1. Generate PBSs for this street
        let pbs_count = generate_street_pbs(config, street)?;
        eprintln!("Generated {pbs_count} PBSs for {street:?}");

        // 2. Solve subgames
        let evaluator = model.as_ref().map(|m| {
            Box::new(RebelLeafEvaluator::new(m.clone(), device.clone())) as Box<dyn LeafEvaluator>
        });
        let depth_limit = if street == Street::River { None } else { Some(0u8) };
        solve_street_pbs(config, street, depth_limit, evaluator.as_deref())?;

        // 3. Export to training format (accumulated across all streets so far)
        export_accumulated_training_data(config)?;

        // 4. Retrain value net
        let train_result = train_value_net(config)?;
        eprintln!("{street:?} training MSE: {:.6}", train_result.final_val_loss);

        // Load retrained model for next street's leaf evaluation
        model = Some(load_trained_model(config)?);
    }

    Ok(())
}
```

**Step 2: Run test (integration)**

This is a full pipeline test requiring a trained blueprint. Create a minimal integration test or manual validation:

Run: `cargo run -p poker-solver-trainer --release -- rebel-seed --config sample_configurations/rebel_river_seed.yaml`
Expected: Generates PBSs, solves, trains, reports MSE

**Step 3: Commit**

```bash
git add crates/rebel/src/training.rs
git commit -m "feat(rebel): offline seeding pipeline — bottom-up river→preflop with value net retraining"
```

---

## Phase 5: Live Self-Play

### Task 15: Subgame construction at PBS

**Files:**
- Create: `crates/rebel/src/subgame_solve.rs`

Builds a depth-limited subgame at a PBS and solves it using CFR-D with value net at leaves. This is the core search component of Algorithm 1.

**Step 1: Write failing test**

```rust
#[test]
fn test_subgame_solve_produces_strategy_and_cfvs() {
    // Create a PBS at a turn state
    // Build depth-limited subgame (turn betting only)
    // Solve with mock leaf evaluator
    // Verify: root CFVs are computed, average strategy is available
}
```

**Step 2: Implement**

```rust
/// Result of solving a subgame at a PBS.
pub struct SubgameSolveResult {
    /// Counterfactual values at the root PBS for each player.
    /// root_cfvs[player][combo] = infostate value
    pub root_cfvs: [[f32; 1326]; 2],
    /// Average strategy across all CFR iterations.
    /// Not stored long-term — only used for action sampling.
    pub average_strategy: Vec<Vec<f32>>,
    /// Per-iteration strategies (for random iteration sampling).
    /// iteration_strategies[t] = strategy at iteration t
    /// Only stored if needed for action sampling.
    pub iteration_strategies: Vec<Vec<Vec<f32>>>,
}

/// Solve a subgame at a PBS using CFR-D with value net at depth boundaries.
///
/// This is the search component of Algorithm 1:
/// - Build a depth-limited subgame (current street only)
/// - Run T iterations of CFR-D
/// - At leaf PBSs, query the value net
/// - Return root infostate values and average strategy
pub fn solve_subgame(
    pbs: &Pbs,
    evaluator: &dyn LeafEvaluator,
    bet_sizes: &BetSizeConfig,
    cfr_iterations: u32,
) -> Result<SubgameSolveResult, String> {
    // Build PostFlopGame with depth_limit=0 (current street only)
    // Set boundary CFVs from evaluator
    // Run solve_step() for each iteration, tracking per-iteration strategies
    // Compute average strategy and root CFVs

    let mut game = build_subgame(pbs, bet_sizes)?;

    // Set boundary CFVs
    set_boundary_cfvs_from_evaluator(&mut game, pbs, evaluator)?;

    // Run CFR iterations, recording strategies at each
    let mut iteration_strategies = Vec::with_capacity(cfr_iterations as usize);
    for t in 0..cfr_iterations {
        solve_step(&game, t);
        if t >= cfr_iterations / 2 {
            // Only record strategies from second half (after warm-up)
            iteration_strategies.push(extract_current_strategy(&game));
        }
    }
    finalize(&mut game);

    // Extract root CFVs
    let root_cfvs = extract_root_cfvs_both_players(&game);
    let average_strategy = extract_average_strategy(&game);

    Ok(SubgameSolveResult {
        root_cfvs,
        average_strategy,
        iteration_strategies,
    })
}
```

**Step 3: Run test**

Run: `cargo test -p rebel subgame_solve::tests -- --release`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/rebel/src/subgame_solve.rs
git commit -m "feat(rebel): subgame solving — CFR-D at PBS with value net leaf evaluation"
```

---

### Task 16: Self-play loop (Algorithm 1)

**Files:**
- Create: `crates/rebel/src/self_play.rs`

The core ReBeL training loop: play hands via self-play, solve subgames at each decision, collect training data, train value net.

**Step 1: Write failing test**

```rust
#[test]
fn test_self_play_one_hand_produces_training_examples() {
    // Play one hand with mock value net
    // Verify training examples are produced (at least 1 per street traversed)
}
```

**Step 2: Implement**

```rust
use crate::pbs::{Pbs, NUM_COMBOS};
use crate::belief_update::update_reach;
use crate::subgame_solve::{solve_subgame, SubgameSolveResult};
use crate::data_buffer::{DiskBuffer, BufferRecord};

/// Training example from self-play: a PBS and its computed CFVs.
pub struct TrainingExample {
    pub pbs: Pbs,
    pub cfvs: [[f32; 1326]; 2], // [OOP, IP] counterfactual values
}

/// Play one hand via self-play with subgame solving at each decision.
///
/// This implements Algorithm 1 from the ReBeL paper:
/// 1. Deal random cards, set initial PBS (uniform beliefs)
/// 2. At each decision point:
///    a. Build and solve depth-limited subgame
///    b. Record (PBS, root CFVs) as training example
///    c. Sample a random CFR iteration t
///    d. Use π^t to sample actions for both players
///    e. Update beliefs via Bayes rule
///    f. Advance to next PBS
/// 3. Continue until hand is terminal
pub fn play_self_play_hand<R: Rng>(
    evaluator: &dyn LeafEvaluator,
    bet_sizes: &BetSizeConfig,
    cfr_iterations: u32,
    initial_stack: i32,
    epsilon: f32, // exploration rate
    rng: &mut R,
) -> Vec<TrainingExample> {
    let mut examples = Vec::new();

    // Deal cards
    let deal = deal_hand(rng);
    let board = deal.board;

    // Initial PBS: uniform beliefs, preflop root
    let mut pbs = Pbs::new_uniform(vec![], initial_stack - 3, initial_stack); // pot = SB + BB = 3
    // Note: exact initial pot depends on blind structure

    // Game state tracking
    let mut street = Street::Preflop;
    let mut acting_player = 0u8; // OOP acts first postflop; preflop is special

    loop {
        // Check termination conditions
        if is_terminal(&pbs) { break; }

        // Solve subgame at current PBS
        let result = match solve_subgame(&pbs, evaluator, bet_sizes, cfr_iterations) {
            Ok(r) => r,
            Err(_) => break,
        };

        // Record training example
        examples.push(TrainingExample {
            pbs: pbs.clone(),
            cfvs: result.root_cfvs,
        });

        // Sample random CFR iteration for action selection
        let t = rng.random_range(0..result.iteration_strategies.len());
        let strategy_t = &result.iteration_strategies[t];

        // Exploration: with probability epsilon, one random player takes uniform action
        let exploring = rng.random::<f32>() < epsilon;
        let explore_player = if exploring { rng.random_range(0..2) } else { usize::MAX };

        // Sample actions for both players using strategy from iteration t
        // ... (depends on how strategy is indexed per player per node)

        // Update beliefs based on sampled actions
        // For acting player: reach[p][i] *= π^t(action | hand_i)
        // ... update pbs.reach_probs ...

        // Advance to next PBS (next decision or next street)
        // ... update pbs.board, pbs.pot, pbs.effective_stack ...
    }

    examples
}

/// Run the self-play training loop.
///
/// Alternates between:
/// 1. Playing N hands via self-play (generating training examples)
/// 2. Training the value net on random batch from buffer
pub fn self_play_training_loop(
    config: &RebelConfig,
    evaluator: &mut RebelLeafEvaluator<impl Backend>,
    buffer: &mut DiskBuffer,
) -> Result<(), String> {
    let hands_per_batch = 100; // configurable
    let total_hands = config.seed.num_hands;

    for batch_start in (0..total_hands).step_by(hands_per_batch) {
        // 1. Play hands in parallel, collect examples
        let examples: Vec<TrainingExample> = (0..hands_per_batch)
            .into_par_iter()
            .flat_map(|i| {
                let mut rng = ChaCha8Rng::seed_from_u64(config.seed.seed + batch_start as u64 + i as u64);
                play_self_play_hand(
                    evaluator, &config.seed.bet_sizes,
                    config.seed.solver_iterations,
                    config.game.initial_stack,
                    0.25, // epsilon
                    &mut rng,
                )
            })
            .collect();

        // 2. Append to buffer
        for ex in &examples {
            let rec_oop = training_example_to_record(ex, 0);
            buffer.append(&rec_oop)?;
            let rec_ip = training_example_to_record(ex, 1);
            buffer.append(&rec_ip)?;
        }

        // 3. Train value net on random batch from buffer
        if buffer.len() >= config.training.batch_size {
            train_one_epoch(config, buffer, evaluator)?;
        }

        eprintln!("Batch {}: {} examples, buffer size: {}",
            batch_start / hands_per_batch, examples.len(), buffer.len());
    }

    Ok(())
}
```

**Note:** The self-play loop is the most complex part of ReBeL. The implementation above is a skeleton — the action sampling and belief update logic within the inner loop needs careful implementation matching the paper's Algorithm 1. Key subtleties:
- Random iteration sampling for safe search
- Belief updates use the sampled iteration's strategy, not the average
- Exploration applies to one random player per hand
- Street transitions require updating the board and zeroing new blocked combos

**Step 3: Run test**

Run: `cargo test -p rebel self_play::tests -- --release`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/rebel/src/self_play.rs
git commit -m "feat(rebel): self-play loop — Algorithm 1 with subgame solving and belief updates"
```

---

### Task 17: rebel-train CLI

**Files:**
- Modify: `crates/trainer/src/main.rs`

**Step 1: Add CLI variant**

```rust
/// Run ReBeL live self-play training loop
#[command(name = "rebel-train")]
RebelTrain {
    /// Path to ReBeL YAML configuration file
    #[arg(short, long)]
    config: String,

    /// Path to pre-trained value net (from offline seeding) to start from
    #[arg(long)]
    model: Option<String>,
},
```

**Step 2: Add handler**

Wire to `self_play_training_loop()`:
- Load config
- Load or create value net
- Create/open buffer
- Run training loop
- Save final model

**Step 3: Verify**

Run: `cargo build -p poker-solver-trainer`
Expected: compiles

**Step 4: Commit**

```bash
git add crates/trainer/src/main.rs
git commit -m "feat(trainer): add rebel-train CLI subcommand for live self-play training"
```

---

## Phase 6: Validation & Metrics

### Task 18: Validation metrics

**Files:**
- Create: `crates/rebel/src/validation.rs`

**Step 1: Write failing test**

```rust
#[test]
fn test_mse_computation() {
    let predicted = vec![1.0, 2.0, 3.0];
    let actual = vec![1.1, 2.2, 2.8];
    let mask = vec![1.0, 1.0, 1.0];
    let mse = compute_mse(&predicted, &actual, &mask);
    let expected = ((0.1f32.powi(2) + 0.2f32.powi(2) + 0.2f32.powi(2)) / 3.0);
    assert!((mse - expected).abs() < 1e-6);
}
```

**Step 2: Implement**

```rust
/// Compute masked MSE between predicted and actual CFVs.
pub fn compute_mse(predicted: &[f32], actual: &[f32], mask: &[f32]) -> f32 {
    let mut sum_sq = 0.0f64;
    let mut count = 0.0f64;
    for i in 0..predicted.len() {
        if mask[i] > 0.0 {
            let diff = predicted[i] as f64 - actual[i] as f64;
            sum_sq += diff * diff;
            count += 1.0;
        }
    }
    if count > 0.0 { (sum_sq / count) as f32 } else { 0.0 }
}

/// Validate value net predictions against a held-out set of pre-solved subgames.
///
/// Returns MSE per street.
pub fn validate_value_net(
    model: &CfvNet<impl Backend>,
    validation_records: &[TrainingRecord],
    device: &impl Device,
) -> HashMap<Street, f32> {
    let mut mse_by_street: HashMap<Street, (f64, f64)> = HashMap::new();

    for rec in validation_records {
        let street = street_from_board_size(rec.board.len());
        let input = encode_record(rec);
        // Forward pass
        let predicted = model_predict(model, &input.input, device);
        let mse = compute_mse(&predicted, &input.target, &input.mask);

        let entry = mse_by_street.entry(street).or_insert((0.0, 0.0));
        entry.0 += mse as f64;
        entry.1 += 1.0;
    }

    mse_by_street.into_iter()
        .map(|(street, (sum, count))| (street, (sum / count) as f32))
        .collect()
}

/// Generate held-out validation set by solving random subgames exactly.
pub fn generate_validation_set(
    config: &RebelConfig,
    num_per_street: usize,
) -> Vec<TrainingRecord> {
    // For each street, sample random PBSs and solve exactly (no depth limit)
    // Store as TrainingRecords for later MSE comparison
    let mut records = Vec::new();
    for &street in &[Street::River, Street::Turn, Street::Flop] {
        for i in 0..num_per_street {
            // Sample random PBS
            // Solve exactly
            // Convert to TrainingRecord
            // records.push(...)
        }
    }
    records
}
```

**Step 3: Run test**

Run: `cargo test -p rebel validation::tests`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/rebel/src/validation.rs
git commit -m "feat(rebel): validation metrics — MSE per street, held-out validation set"
```

---

### Task 19: rebel-eval CLI and head-to-head

**Files:**
- Modify: `crates/trainer/src/main.rs`
- Modify: `crates/rebel/src/validation.rs`

**Step 1: Add CLI variant**

```rust
/// Evaluate a trained ReBeL value net: MSE, exploitability, head-to-head
#[command(name = "rebel-eval")]
RebelEval {
    /// Path to ReBeL YAML configuration file
    #[arg(short, long)]
    config: String,

    /// Path to trained value net model
    #[arg(long)]
    model: String,

    /// Evaluation mode: mse, exploit, or h2h (head-to-head vs blueprint)
    #[arg(long, default_value = "mse")]
    mode: String,

    /// Number of hands for head-to-head evaluation
    #[arg(long, default_value_t = 100000)]
    num_hands: usize,
},
```

**Step 2: Implement head-to-head**

```rust
/// Head-to-head: ReBeL agent vs blueprint agent.
///
/// ReBeL agent uses subgame solving at every decision.
/// Blueprint agent uses tabular strategy lookup.
/// Returns win rate in mbb/hand.
pub fn head_to_head(
    rebel_model: &CfvNet<impl Backend>,
    blueprint: &BlueprintV2Strategy,
    buckets: &AllBuckets,
    tree: &GameTree,
    config: &RebelConfig,
    num_hands: usize,
) -> f64 {
    // Play num_hands with ReBeL as P0 and blueprint as P1
    // Then swap and play num_hands more
    // Compute average win rate in mbb/hand

    let mut total_value = 0.0f64;

    for hand_idx in 0..num_hands {
        // Deal cards
        // Play hand: ReBeL solves subgames, blueprint uses table lookup
        // Record payoff
    }

    // Average mbb/hand
    (total_value / num_hands as f64) * 1000.0 // convert to milli-big-blinds
}
```

**Step 3: Wire CLI handler, build, commit**

```bash
git add crates/trainer/src/main.rs crates/rebel/src/validation.rs
git commit -m "feat(trainer): add rebel-eval CLI — MSE validation, exploitability, head-to-head"
```

---

## Module Checklist

After all tasks, `crates/rebel/src/lib.rs` should export:

```rust
pub mod config;
pub mod pbs;
pub mod belief_update;
pub mod blueprint_sampler;
pub mod data_buffer;
pub mod generate;
pub mod solver;
pub mod training;
pub mod leaf_evaluator;
pub mod subgame_solve;
pub mod self_play;
pub mod validation;
```

And `crates/trainer/src/main.rs` should have three new subcommands:
- `rebel-seed` — Generate PBS training data from blueprint play
- `rebel-train` — Run live self-play training loop
- `rebel-eval` — Validate trained value net

## Update Documentation

After implementation, update:
- `docs/architecture.md` — Add ReBeL section describing PBS, self-play, value net
- `docs/training.md` — Add rebel-seed, rebel-train, rebel-eval CLI docs
- `CLAUDE.md` — Add rebel crate to crate map
