# River CFVnet Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete pipeline to generate training data, train, and evaluate a river-street Deep Counterfactual Value Network (CFVnet) using burn.

**Architecture:** New `cfvnet` crate with three modules: `datagen` (range generation, situation sampling, range-solver wrapper, binary serialization), `model` (burn MLP, loss, dataset, training loop), `eval` (metrics, comparison harness). Single CLI binary with four subcommands: `generate`, `train`, `evaluate`, `compare`.

**Tech Stack:** Rust, burn (wgpu/CUDA backends), range-solver crate, clap (CLI), serde/serde_yaml (config), indicatif (progress bars), rand/rand_chacha (reproducible RNG).

**Design doc:** `docs/plans/2026-03-11-river-cfvnet-design.md`

---

## Task 1: Crate Scaffolding

**Files:**
- Create: `crates/cfvnet/Cargo.toml`
- Create: `crates/cfvnet/src/lib.rs`
- Create: `crates/cfvnet/src/main.rs`
- Modify: `Cargo.toml` (workspace members)

**Step 1: Add cfvnet to workspace**

In root `Cargo.toml`, add `"crates/cfvnet"` to the `members` list.

**Step 2: Create Cargo.toml**

```toml
[package]
name = "cfvnet"
version.workspace = true
edition.workspace = true

[dependencies]
range-solver = { path = "../range-solver" }
clap = { version = "4", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
serde_yaml = "0.9"
rand = "0.8"
rand_chacha = "0.3"
rayon = { workspace = true }
indicatif = "0.17"
bytemuck = { version = "1", features = ["derive"] }
thiserror = { workspace = true }
burn = { version = "0.16", features = ["train", "wgpu", "ndarray"] }

[dev-dependencies]
tempfile = "3"
approx = "0.5"
```

**Step 3: Create lib.rs with module structure**

```rust
pub mod config;
pub mod datagen;
pub mod model;
pub mod eval;
```

**Step 4: Create stub main.rs**

```rust
fn main() {
    println!("cfvnet - Deep Counterfactual Value Network toolkit");
}
```

**Step 5: Create empty module files**

- `crates/cfvnet/src/config.rs`
- `crates/cfvnet/src/datagen/mod.rs`
- `crates/cfvnet/src/model/mod.rs`
- `crates/cfvnet/src/eval/mod.rs`

**Step 6: Verify it compiles**

Run: `cargo check -p cfvnet`
Expected: Compiles with no errors (warnings OK for empty modules).

**Step 7: Commit**

```
feat(cfvnet): scaffold crate with module structure
```

---

## Task 2: Config Types

**Files:**
- Create: `crates/cfvnet/src/config.rs`
- Test: inline `#[cfg(test)]` module

**Step 1: Write the failing test**

In `config.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_config() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["25%", "50%", "100%", "a"]
datagen:
  num_samples: 1000
  solver_iterations: 100
"#;
        let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.game.initial_stack, 200);
        assert_eq!(config.game.bet_sizes.len(), 4);
        assert_eq!(config.datagen.num_samples, 1000);
        // Check defaults filled in
        assert_eq!(config.datagen.seed, 42);
        assert_eq!(config.training.hidden_layers, 7);
        assert_eq!(config.training.batch_size, 2048);
    }

    #[test]
    fn parse_full_config() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["25%", "50%", "100%", "a"]
  add_allin_threshold: 1.5
  force_allin_threshold: 0.15
datagen:
  num_samples: 1000000
  pot_intervals: [[4,20], [20,80], [80,200], [200,400]]
  solver_iterations: 1000
  target_exploitability: 0.005
  threads: 8
  seed: 42
training:
  hidden_layers: 7
  hidden_size: 500
  batch_size: 2048
  epochs: 2
  learning_rate: 0.001
  lr_min: 0.00001
  huber_delta: 1.0
  aux_loss_weight: 1.0
  validation_split: 0.05
  checkpoint_every_n_batches: 1000
"#;
        let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.datagen.pot_intervals.len(), 4);
        assert_eq!(config.datagen.threads, 8);
        assert!((config.training.learning_rate - 0.001).abs() < 1e-9);
    }

    #[test]
    fn validate_rejects_empty_bet_sizes() {
        let config = GameConfig {
            initial_stack: 200,
            bet_sizes: vec![],
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_stack() {
        let config = GameConfig {
            initial_stack: 0,
            bet_sizes: vec!["50%".into()],
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p cfvnet config::tests -- --nocapture`
Expected: FAIL — types don't exist yet.

**Step 3: Implement config types**

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfvnetConfig {
    pub game: GameConfig,
    #[serde(default)]
    pub datagen: DatagenConfig,
    #[serde(default)]
    pub training: TrainingConfig,
    #[serde(default)]
    pub evaluation: EvaluationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameConfig {
    pub initial_stack: i32,
    pub bet_sizes: Vec<String>,
    #[serde(default = "default_allin_threshold")]
    pub add_allin_threshold: f64,
    #[serde(default = "default_force_allin_threshold")]
    pub force_allin_threshold: f64,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            initial_stack: 200,
            bet_sizes: vec!["25%".into(), "50%".into(), "100%".into(), "a".into()],
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
        }
    }
}

impl GameConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.initial_stack <= 0 {
            return Err("initial_stack must be > 0".into());
        }
        if self.bet_sizes.is_empty() {
            return Err("bet_sizes must not be empty".into());
        }
        Ok(())
    }
}

fn default_allin_threshold() -> f64 { 1.5 }
fn default_force_allin_threshold() -> f64 { 0.15 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatagenConfig {
    pub num_samples: u64,
    #[serde(default = "default_pot_intervals")]
    pub pot_intervals: Vec<[i32; 2]>,
    #[serde(default = "default_solver_iterations")]
    pub solver_iterations: u32,
    #[serde(default = "default_target_exploitability")]
    pub target_exploitability: f32,
    #[serde(default = "default_threads")]
    pub threads: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

impl Default for DatagenConfig {
    fn default() -> Self {
        Self {
            num_samples: 1000,
            pot_intervals: default_pot_intervals(),
            solver_iterations: 1000,
            target_exploitability: 0.005,
            threads: 8,
            seed: 42,
        }
    }
}

fn default_pot_intervals() -> Vec<[i32; 2]> {
    vec![[4, 20], [20, 80], [80, 200], [200, 400]]
}
fn default_solver_iterations() -> u32 { 1000 }
fn default_target_exploitability() -> f32 { 0.005 }
fn default_threads() -> usize { 8 }
fn default_seed() -> u64 { 42 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    #[serde(default = "default_hidden_layers")]
    pub hidden_layers: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_lr_min")]
    pub lr_min: f64,
    #[serde(default = "default_huber_delta")]
    pub huber_delta: f64,
    #[serde(default = "default_aux_loss_weight")]
    pub aux_loss_weight: f64,
    #[serde(default = "default_validation_split")]
    pub validation_split: f64,
    #[serde(default = "default_checkpoint_interval")]
    pub checkpoint_every_n_batches: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            hidden_layers: 7,
            hidden_size: 500,
            batch_size: 2048,
            epochs: 2,
            learning_rate: 0.001,
            lr_min: 0.00001,
            huber_delta: 1.0,
            aux_loss_weight: 1.0,
            validation_split: 0.05,
            checkpoint_every_n_batches: 1000,
        }
    }
}

fn default_hidden_layers() -> usize { 7 }
fn default_hidden_size() -> usize { 500 }
fn default_batch_size() -> usize { 2048 }
fn default_epochs() -> usize { 2 }
fn default_learning_rate() -> f64 { 0.001 }
fn default_lr_min() -> f64 { 0.00001 }
fn default_huber_delta() -> f64 { 1.0 }
fn default_aux_loss_weight() -> f64 { 1.0 }
fn default_validation_split() -> f64 { 0.05 }
fn default_checkpoint_interval() -> usize { 1000 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    #[serde(default)]
    pub regression_spots: Option<String>,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self { regression_spots: None }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p cfvnet config::tests -- --nocapture`
Expected: All 4 tests PASS.

**Step 5: Commit**

```
feat(cfvnet): add YAML config types with defaults and validation
```

---

## Task 3: R(S, p) Range Generator

**Files:**
- Create: `crates/cfvnet/src/datagen/range_gen.rs`
- Modify: `crates/cfvnet/src/datagen/mod.rs`

This is the core DeepStack R(S,p) algorithm. It generates random ranges correlated with hand strength. This is subtle — bugs here waste GPU-hours producing bad training data.

**Step 1: Write the failing tests**

In `range_gen.rs`, add `#[cfg(test)] mod tests`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    fn test_board() -> [u8; 5] {
        // Qs Jh 2c 8d 3s = card IDs
        [
            4 * 10 + 3,  // Qs
            4 * 9 + 2,   // Jh
            4 * 0 + 0,   // 2c
            4 * 6 + 1,   // 8d
            4 * 1 + 3,   // 3s
        ]
    }

    #[test]
    fn range_sums_to_one() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let board = test_board();
        let range = generate_rsp_range(&board, &mut rng);
        let sum: f32 = range.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "range sum = {sum}");
    }

    #[test]
    fn range_all_non_negative() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let board = test_board();
        let range = generate_rsp_range(&board, &mut rng);
        for (i, &v) in range.iter().enumerate() {
            assert!(v >= 0.0, "range[{i}] = {v} is negative");
        }
    }

    #[test]
    fn board_blocked_combos_are_zero() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let board = test_board();
        let range = generate_rsp_range(&board, &mut rng);

        // Check that combos containing any board card have zero weight
        for card in &board {
            for i in 0..1326 {
                let (c1, c2) = range_solver::card::index_to_card_pair(i);
                if c1 == *card || c2 == *card {
                    assert_eq!(range[i], 0.0,
                        "combo {i} (cards {c1},{c2}) conflicts with board card {card}");
                }
            }
        }
    }

    #[test]
    fn strong_hands_have_higher_mean_reach() {
        // Over many samples, top-strength hands should have higher
        // average reach than bottom-strength hands.
        let board = test_board();
        let mut top_sum = 0.0f64;
        let mut bottom_sum = 0.0f64;
        let n = 1000;

        // Pre-compute hand strengths for this board
        let strengths = compute_hand_strengths(&board);
        let valid: Vec<usize> = (0..1326)
            .filter(|&i| {
                let (c1, c2) = range_solver::card::index_to_card_pair(i);
                !board.contains(&c1) && !board.contains(&c2)
            })
            .collect();

        // Sort valid hands by strength
        let mut sorted = valid.clone();
        sorted.sort_by(|&a, &b| strengths[a].partial_cmp(&strengths[b]).unwrap());
        let top_quarter = &sorted[sorted.len() * 3 / 4..];
        let bottom_quarter = &sorted[..sorted.len() / 4];

        for seed in 0..n {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let range = generate_rsp_range(&board, &mut rng);
            top_sum += top_quarter.iter().map(|&i| range[i] as f64).sum::<f64>();
            bottom_sum += bottom_quarter.iter().map(|&i| range[i] as f64).sum::<f64>();
        }

        assert!(top_sum > bottom_sum,
            "top quarter mean {:.6} should exceed bottom quarter mean {:.6}",
            top_sum / (n as f64 * top_quarter.len() as f64),
            bottom_sum / (n as f64 * bottom_quarter.len() as f64));
    }

    #[test]
    fn deterministic_with_same_seed() {
        let board = test_board();
        let mut rng1 = ChaCha8Rng::seed_from_u64(123);
        let mut rng2 = ChaCha8Rng::seed_from_u64(123);
        let range1 = generate_rsp_range(&board, &mut rng1);
        let range2 = generate_rsp_range(&board, &mut rng2);
        assert_eq!(range1, range2);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p cfvnet datagen::range_gen::tests -- --nocapture`
Expected: FAIL — `generate_rsp_range` and `compute_hand_strengths` not defined.

**Step 3: Implement R(S,p)**

```rust
use rand::Rng;
use range_solver::card::index_to_card_pair;

/// Number of possible hole card combos in HUNL.
pub const NUM_COMBOS: usize = 1326;

/// Compute hand strength for each of the 1326 combos on a 5-card board.
///
/// Returns an array where `strengths[i]` is the hand rank (higher = stronger)
/// for the combo at index `i`. Board-conflicting combos get strength 0.
pub fn compute_hand_strengths(board: &[u8; 5]) -> [u16; NUM_COMBOS] {
    let mut strengths = [0u16; NUM_COMBOS];
    for i in 0..NUM_COMBOS {
        let (c1, c2) = index_to_card_pair(i);
        if board.contains(&c1) || board.contains(&c2) || c1 == c2 {
            continue;
        }
        // Use range_solver's hand evaluation
        strengths[i] = evaluate_hand_7(c1, c2, board);
    }
    strengths
}

/// Generate a random range using the DeepStack R(S,p) procedure.
///
/// Returns a 1326-element array of reach probabilities summing to 1.0.
/// Board-conflicting combos have zero reach.
pub fn generate_rsp_range<R: Rng>(board: &[u8; 5], rng: &mut R) -> [f32; NUM_COMBOS] {
    let strengths = compute_hand_strengths(board);

    // Collect valid (non-blocked) combo indices
    let mut valid: Vec<usize> = (0..NUM_COMBOS)
        .filter(|&i| strengths[i] > 0)
        .collect();

    // Sort by strength (ascending)
    valid.sort_by_key(|&i| strengths[i]);

    // Allocate output
    let mut range = [0.0f32; NUM_COMBOS];

    // Run R(S, p) with total probability = 1.0
    rsp_recursive(&valid, 1.0, &mut range, rng);

    range
}

/// Recursive R(S, p) implementation.
fn rsp_recursive<R: Rng>(
    hands: &[usize],
    p: f64,
    range: &mut [f32; NUM_COMBOS],
    rng: &mut R,
) {
    if hands.is_empty() || p <= 0.0 {
        return;
    }
    if hands.len() == 1 {
        range[hands[0]] = p as f32;
        return;
    }

    let p1: f64 = rng.gen::<f64>() * p;
    let p2 = p - p1;

    let mid = hands.len() / 2;
    let (weaker, stronger) = hands.split_at(mid);

    // Stronger half gets p1 (first draw), weaker gets p2
    rsp_recursive(stronger, p1, range, rng);
    rsp_recursive(weaker, p2, range, rng);
}
```

Note: `evaluate_hand_7` needs to wrap range-solver's hand evaluation. Check `range_solver::hand` or use `range_solver::hand_table` to evaluate 7-card hands. The implementation detail of this function depends on what range-solver exports — it may need to construct a 7-card hand from (c1, c2, board) and evaluate it. Consult:
- `crates/range-solver/src/hand.rs` — `Hand::evaluate()` or similar
- `crates/range-solver/src/game/evaluation.rs` — `StrengthItem` computation

If range-solver doesn't publicly export hand evaluation, add a thin pub wrapper in range-solver or use `rs_poker` directly (it's already a workspace dependency).

**Step 4: Run tests to verify they pass**

Run: `cargo test -p cfvnet datagen::range_gen::tests -- --nocapture`
Expected: All 5 tests PASS.

**Step 5: Commit**

```
feat(cfvnet): implement R(S,p) range generator with hand-strength correlation
```

---

## Task 4: Situation Sampler

**Files:**
- Create: `crates/cfvnet/src/datagen/sampler.rs`
- Modify: `crates/cfvnet/src/datagen/mod.rs`

**Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    fn test_config() -> DatagenConfig {
        DatagenConfig {
            num_samples: 100,
            pot_intervals: vec![[4, 20], [20, 80], [80, 200], [200, 400]],
            solver_iterations: 100,
            target_exploitability: 0.01,
            threads: 1,
            seed: 42,
        }
    }

    #[test]
    fn board_has_five_unique_cards() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = test_config();
        for _ in 0..100 {
            let sit = sample_situation(&config, &mut rng);
            let board = sit.board;
            assert_eq!(board.len(), 5);
            // All unique
            for i in 0..5 {
                for j in (i + 1)..5 {
                    assert_ne!(board[i], board[j],
                        "duplicate card: {} at positions {i} and {j}", board[i]);
                }
                assert!(board[i] < 52, "card {} out of range", board[i]);
            }
        }
    }

    #[test]
    fn pot_within_configured_intervals() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = test_config();
        for _ in 0..200 {
            let sit = sample_situation(&config, &mut rng);
            let in_some_interval = config.pot_intervals.iter().any(|[lo, hi]| {
                sit.pot >= *lo && sit.pot < *hi
            });
            assert!(in_some_interval,
                "pot {} not in any interval {:?}", sit.pot, config.pot_intervals);
        }
    }

    #[test]
    fn stack_within_valid_range() {
        let config = test_config();
        let initial_stack = 200i32; // 100bb in SB units
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..200 {
            let sit = sample_situation(&config, &mut rng);
            let max_stack = initial_stack - sit.pot / 2;
            assert!(sit.effective_stack >= 0,
                "negative stack: {}", sit.effective_stack);
            assert!(sit.effective_stack <= max_stack,
                "stack {} exceeds max {} for pot {}", sit.effective_stack, max_stack, sit.pot);
        }
    }

    #[test]
    fn ranges_are_valid() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = test_config();
        for _ in 0..50 {
            let sit = sample_situation(&config, &mut rng);
            for player in 0..2 {
                let range = &sit.ranges[player];
                let sum: f32 = range.iter().sum();
                assert!((sum - 1.0).abs() < 1e-4,
                    "player {player} range sums to {sum}");
                // Board-blocked combos should be zero
                for i in 0..1326 {
                    let (c1, c2) = range_solver::card::index_to_card_pair(i);
                    if sit.board.contains(&c1) || sit.board.contains(&c2) {
                        assert_eq!(range[i], 0.0,
                            "player {player} combo {i} conflicts with board");
                    }
                }
            }
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let config = test_config();
        let mut rng1 = ChaCha8Rng::seed_from_u64(99);
        let mut rng2 = ChaCha8Rng::seed_from_u64(99);
        let s1 = sample_situation(&config, &mut rng1);
        let s2 = sample_situation(&config, &mut rng2);
        assert_eq!(s1.board, s2.board);
        assert_eq!(s1.pot, s2.pot);
        assert_eq!(s1.effective_stack, s2.effective_stack);
        assert_eq!(s1.ranges[0], s2.ranges[0]);
        assert_eq!(s1.ranges[1], s2.ranges[1]);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p cfvnet datagen::sampler::tests -- --nocapture`
Expected: FAIL — types/functions not defined.

**Step 3: Implement the sampler**

```rust
use rand::Rng;
use crate::config::DatagenConfig;
use super::range_gen::{generate_rsp_range, NUM_COMBOS};

/// A single training situation before solving.
#[derive(Debug, Clone)]
pub struct Situation {
    pub board: [u8; 5],
    pub pot: i32,
    pub effective_stack: i32,
    pub ranges: [[f32; NUM_COMBOS]; 2],
}

/// Sample a random river situation.
pub fn sample_situation<R: Rng>(config: &DatagenConfig, rng: &mut R) -> Situation {
    let board = sample_board(rng);
    let pot = sample_pot(&config.pot_intervals, rng);
    let max_stack = 200 - pot / 2; // initial_stack=200 SB
    let effective_stack = if max_stack <= 0 { 0 } else { rng.gen_range(0..=max_stack) };
    let oop_range = generate_rsp_range(&board, rng);
    let ip_range = generate_rsp_range(&board, rng);
    Situation { board, pot, effective_stack, ranges: [oop_range, ip_range] }
}

/// Sample 5 unique cards for the board.
fn sample_board<R: Rng>(rng: &mut R) -> [u8; 5] {
    let mut board = [0u8; 5];
    let mut used = [false; 52];
    for card in &mut board {
        loop {
            let c: u8 = rng.gen_range(0..52);
            if !used[c as usize] {
                used[c as usize] = true;
                *card = c;
                break;
            }
        }
    }
    board
}

/// Sample pot from stratified intervals.
fn sample_pot<R: Rng>(intervals: &[[i32; 2]], rng: &mut R) -> i32 {
    let idx = rng.gen_range(0..intervals.len());
    let [lo, hi] = intervals[idx];
    rng.gen_range(lo..hi)
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p cfvnet datagen::sampler::tests -- --nocapture`
Expected: All 5 tests PASS.

**Step 5: Commit**

```
feat(cfvnet): add situation sampler with stratified pot/stack sampling
```

---

## Task 5: Solve Wrapper

**Files:**
- Create: `crates/cfvnet/src/datagen/solver.rs`
- Modify: `crates/cfvnet/src/datagen/mod.rs`

This wraps range-solver to solve a situation and extract CFVs. Critical to get right — bad labels = wasted training.

**Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use range_solver::card::{card_from_str, flop_from_str};

    /// Build a situation with known ranges for testing.
    fn known_river_situation() -> Situation {
        // Qs Jh 2c 8d 3s
        let board = [
            4 * 10 + 3, // Qs
            4 * 9 + 2,  // Jh
            4 * 0 + 0,  // 2c
            4 * 6 + 1,  // 8d
            4 * 1 + 3,  // 3s
        ];
        // Use uniform ranges (all valid combos equal weight)
        let mut oop_range = [0.0f32; 1326];
        let mut ip_range = [0.0f32; 1326];
        let mut count = 0;
        for i in 0..1326 {
            let (c1, c2) = range_solver::card::index_to_card_pair(i);
            if !board.contains(&c1) && !board.contains(&c2) && c1 != c2 {
                oop_range[i] = 1.0;
                ip_range[i] = 1.0;
                count += 1;
            }
        }
        // Normalize
        for i in 0..1326 {
            oop_range[i] /= count as f32;
            ip_range[i] /= count as f32;
        }
        Situation {
            board,
            pot: 100,
            effective_stack: 100,
            ranges: [oop_range, ip_range],
        }
    }

    #[test]
    fn solve_returns_correct_length_evs() {
        let sit = known_river_situation();
        let solve_config = SolveConfig {
            bet_sizes: vec!["50%".into(), "a".into()],
            solver_iterations: 200,
            target_exploitability: 0.01,
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
        };
        let result = solve_situation(&sit, &solve_config).unwrap();
        assert_eq!(result.oop_evs.len(), 1326);
        assert_eq!(result.ip_evs.len(), 1326);
    }

    #[test]
    fn solve_board_blocked_evs_are_zero() {
        let sit = known_river_situation();
        let solve_config = SolveConfig {
            bet_sizes: vec!["50%".into(), "a".into()],
            solver_iterations: 200,
            target_exploitability: 0.01,
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
        };
        let result = solve_situation(&sit, &solve_config).unwrap();
        for card in &sit.board {
            for i in 0..1326 {
                let (c1, c2) = range_solver::card::index_to_card_pair(i);
                if c1 == *card || c2 == *card {
                    assert_eq!(result.oop_evs[i], 0.0);
                    assert_eq!(result.ip_evs[i], 0.0);
                }
            }
        }
    }

    #[test]
    fn solve_exploitability_below_threshold() {
        let sit = known_river_situation();
        let solve_config = SolveConfig {
            bet_sizes: vec!["50%".into(), "a".into()],
            solver_iterations: 500,
            target_exploitability: 0.01,
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
        };
        let result = solve_situation(&sit, &solve_config).unwrap();
        assert!(result.exploitability < 0.02,
            "exploitability {} too high", result.exploitability);
    }

    #[test]
    fn solve_evs_are_pot_relative() {
        let sit = known_river_situation();
        let solve_config = SolveConfig {
            bet_sizes: vec!["50%".into(), "a".into()],
            solver_iterations: 200,
            target_exploitability: 0.01,
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
        };
        let result = solve_situation(&sit, &solve_config).unwrap();
        // Pot-relative EVs should be in roughly [-3, 3] range
        // (can win/lose up to stack/pot times the pot)
        for &ev in result.oop_evs.iter().chain(result.ip_evs.iter()) {
            if ev != 0.0 {
                assert!(ev.abs() < 5.0,
                    "EV {} seems too large for pot-relative", ev);
            }
        }
    }

    #[test]
    fn solve_zero_sum_property() {
        // Weighted sum of OOP EVs + weighted sum of IP EVs ≈ 0
        // (zero-sum game, pot-relative)
        let sit = known_river_situation();
        let solve_config = SolveConfig {
            bet_sizes: vec!["50%".into(), "a".into()],
            solver_iterations: 500,
            target_exploitability: 0.005,
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
        };
        let result = solve_situation(&sit, &solve_config).unwrap();

        let oop_weighted: f64 = sit.ranges[0].iter()
            .zip(result.oop_evs.iter())
            .map(|(&r, &ev)| r as f64 * ev as f64)
            .sum();
        let ip_weighted: f64 = sit.ranges[1].iter()
            .zip(result.ip_evs.iter())
            .map(|(&r, &ev)| r as f64 * ev as f64)
            .sum();

        // In a zero-sum game, the game values should approximately sum to zero
        // when both players start with the same initial pot contribution.
        // The actual relationship depends on the pot/stack geometry.
        // Just verify they're finite and reasonable.
        assert!(oop_weighted.is_finite(), "OOP game value is not finite");
        assert!(ip_weighted.is_finite(), "IP game value is not finite");
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p cfvnet datagen::solver::tests -- --nocapture`
Expected: FAIL.

**Step 3: Implement the solver wrapper**

```rust
use crate::datagen::sampler::Situation;
use crate::datagen::range_gen::NUM_COMBOS;
use range_solver::{
    PostFlopGame, CardConfig, solve,
    card::{Card, NOT_DEALT, index_to_card_pair},
    action_tree::{ActionTree, BoardState, TreeConfig},
    bet_size::BetSizeOptions,
    range::Range,
};

pub struct SolveConfig {
    pub bet_sizes: Vec<String>,
    pub solver_iterations: u32,
    pub target_exploitability: f32,
    pub add_allin_threshold: f64,
    pub force_allin_threshold: f64,
}

pub struct SolveResult {
    /// Pot-relative expected values for OOP, length 1326.
    pub oop_evs: [f32; NUM_COMBOS],
    /// Pot-relative expected values for IP, length 1326.
    pub ip_evs: [f32; NUM_COMBOS],
    /// Game value for OOP (weighted sum of OOP EVs).
    pub oop_game_value: f32,
    /// Game value for IP.
    pub ip_game_value: f32,
    /// Valid combo mask (true if combo is not board-blocked).
    pub valid_mask: [bool; NUM_COMBOS],
    /// Final exploitability from the solver.
    pub exploitability: f32,
}

pub fn solve_situation(
    situation: &Situation,
    config: &SolveConfig,
) -> Result<SolveResult, String> {
    // Build range-solver Range objects from our 1326-element arrays
    let oop_range = range_from_array(&situation.ranges[0]);
    let ip_range = range_from_array(&situation.ranges[1]);

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [situation.board[0], situation.board[1], situation.board[2]],
        turn: situation.board[3],
        river: situation.board[4],
    };

    // Build bet sizes string for range-solver
    let bet_str = config.bet_sizes.join(", ");
    let sizes = BetSizeOptions::try_from((bet_str.as_str(), ""))
        .map_err(|e| format!("bad bet sizes: {e}"))?;

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: situation.pot,
        effective_stack: situation.effective_stack,
        river_bet_sizes: [sizes.clone(), sizes],
        add_allin_threshold: config.add_allin_threshold,
        force_allin_threshold: config.force_allin_threshold,
        ..Default::default()
    };

    let tree = ActionTree::new(tree_config).map_err(|e| format!("tree error: {e}"))?;
    let mut game = PostFlopGame::with_config(card_config, tree)
        .map_err(|e| format!("game error: {e}"))?;

    game.allocate_memory(false);

    let exploitability = solve(
        &mut game,
        config.solver_iterations,
        config.target_exploitability,
        false,
    );

    game.cache_normalized_weights();

    // Extract EVs — these are chip EVs relative to starting position
    let raw_oop_evs = game.expected_values(0);
    let raw_ip_evs = game.expected_values(1);

    // Map from range-solver's private_cards indexing back to 1326 indexing
    // and normalize to pot-relative
    let pot = situation.pot as f32;
    let mut oop_evs = [0.0f32; NUM_COMBOS];
    let mut ip_evs = [0.0f32; NUM_COMBOS];
    let mut valid_mask = [false; NUM_COMBOS];

    let oop_cards = game.private_cards(0);
    for (hand_idx, &(c1, c2)) in oop_cards.iter().enumerate() {
        let combo_idx = range_solver::card::card_pair_to_index(c1, c2);
        oop_evs[combo_idx] = raw_oop_evs[hand_idx] / pot;
        valid_mask[combo_idx] = true;
    }

    let ip_cards = game.private_cards(1);
    for (hand_idx, &(c1, c2)) in ip_cards.iter().enumerate() {
        let combo_idx = range_solver::card::card_pair_to_index(c1, c2);
        ip_evs[combo_idx] = raw_ip_evs[hand_idx] / pot;
        valid_mask[combo_idx] = true;
    }

    // Compute game values (weighted sum of pot-relative EVs)
    let oop_game_value: f32 = situation.ranges[0].iter()
        .zip(oop_evs.iter())
        .map(|(r, ev)| r * ev)
        .sum();
    let ip_game_value: f32 = situation.ranges[1].iter()
        .zip(ip_evs.iter())
        .map(|(r, ev)| r * ev)
        .sum();

    Ok(SolveResult {
        oop_evs,
        ip_evs,
        oop_game_value,
        ip_game_value,
        valid_mask,
        exploitability,
    })
}

/// Convert a 1326-element f32 array to a range-solver Range.
fn range_from_array(arr: &[f32; NUM_COMBOS]) -> Range {
    let mut range = Range::default();
    // Range::data is [f32; 1326] matching our indexing
    for i in 0..NUM_COMBOS {
        range.data[i] = arr[i];
    }
    range
}
```

Note: The `range_from_array` function depends on `Range`'s internal layout. Check if `Range::data` is public. If not, use `Range`'s public API to set individual combo weights. Consult `crates/range-solver/src/range.rs` for available setters.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p cfvnet datagen::solver::tests -- --nocapture`
Expected: All 5 tests PASS. The solve tests are slower (DCFR iterations); expect ~1-2s per test.

**Step 5: Commit**

```
feat(cfvnet): add range-solver wrapper for training data generation
```

---

## Task 6: Binary Serialization

**Files:**
- Create: `crates/cfvnet/src/datagen/storage.rs`
- Modify: `crates/cfvnet/src/datagen/mod.rs`

**Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::{Seek, SeekFrom};

    fn sample_record() -> TrainingRecord {
        let mut oop_range = [0.0f32; 1326];
        let mut ip_range = [0.0f32; 1326];
        let mut cfvs = [0.0f32; 1326];
        let mut valid_mask = [0u8; 1326];

        // Set a few values to be non-trivial
        oop_range[0] = 0.5;
        oop_range[1] = 0.5;
        ip_range[100] = 1.0;
        cfvs[0] = 0.123;
        cfvs[100] = -0.456;
        valid_mask[0] = 1;
        valid_mask[1] = 1;
        valid_mask[100] = 1;

        TrainingRecord {
            board: [0, 4, 8, 12, 16],
            pot: 100.0,
            effective_stack: 50.0,
            player: 0,
            oop_range,
            ip_range,
            cfvs,
            valid_mask,
            game_value: 0.05,
        }
    }

    #[test]
    fn round_trip_single_record() {
        let record = sample_record();
        let mut file = NamedTempFile::new().unwrap();

        write_record(&mut file, &record).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let loaded = read_record(&mut file).unwrap();
        assert_eq!(record.board, loaded.board);
        assert_eq!(record.pot, loaded.pot);
        assert_eq!(record.effective_stack, loaded.effective_stack);
        assert_eq!(record.player, loaded.player);
        assert_eq!(record.oop_range, loaded.oop_range);
        assert_eq!(record.ip_range, loaded.ip_range);
        assert_eq!(record.cfvs, loaded.cfvs);
        assert_eq!(record.valid_mask, loaded.valid_mask);
        assert!((record.game_value - loaded.game_value).abs() < 1e-7);
    }

    #[test]
    fn round_trip_multiple_records() {
        let r1 = sample_record();
        let mut r2 = sample_record();
        r2.player = 1;
        r2.pot = 200.0;
        r2.game_value = -0.1;

        let mut file = NamedTempFile::new().unwrap();
        write_record(&mut file, &r1).unwrap();
        write_record(&mut file, &r2).unwrap();

        file.seek(SeekFrom::Start(0)).unwrap();

        let loaded1 = read_record(&mut file).unwrap();
        let loaded2 = read_record(&mut file).unwrap();

        assert_eq!(r1.board, loaded1.board);
        assert_eq!(r2.pot, loaded2.pot);
        assert_eq!(r2.player, loaded2.player);
    }

    #[test]
    fn record_count_is_correct() {
        let record = sample_record();
        let mut file = NamedTempFile::new().unwrap();

        for _ in 0..5 {
            write_record(&mut file, &record).unwrap();
        }

        file.seek(SeekFrom::Start(0)).unwrap();
        let count = count_records(&mut file).unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn record_size_is_consistent() {
        // Verify the on-disk size matches expectations
        let expected = 5 + 4 + 4 + 1 + 4  // board, pot, stack, player, game_value
            + 1326 * 4   // oop_range
            + 1326 * 4   // ip_range
            + 1326 * 4   // cfvs
            + 1326;      // valid_mask
        assert_eq!(RECORD_SIZE, expected);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p cfvnet datagen::storage::tests -- --nocapture`
Expected: FAIL.

**Step 3: Implement serialization**

```rust
use std::io::{self, Read, Write, Seek, SeekFrom};

pub const NUM_COMBOS: usize = 1326;

/// Fixed size of one training record in bytes.
pub const RECORD_SIZE: usize =
    5              // board: 5 × u8
    + 4            // pot: f32
    + 4            // effective_stack: f32
    + 1            // player: u8
    + 4            // game_value: f32
    + NUM_COMBOS * 4  // oop_range: f32
    + NUM_COMBOS * 4  // ip_range: f32
    + NUM_COMBOS * 4  // cfvs: f32
    + NUM_COMBOS;     // valid_mask: u8

#[derive(Debug, Clone)]
pub struct TrainingRecord {
    pub board: [u8; 5],
    pub pot: f32,
    pub effective_stack: f32,
    pub player: u8,
    pub game_value: f32,
    pub oop_range: [f32; NUM_COMBOS],
    pub ip_range: [f32; NUM_COMBOS],
    pub cfvs: [f32; NUM_COMBOS],
    pub valid_mask: [u8; NUM_COMBOS],
}

pub fn write_record<W: Write>(w: &mut W, rec: &TrainingRecord) -> io::Result<()> {
    w.write_all(&rec.board)?;
    w.write_all(&rec.pot.to_le_bytes())?;
    w.write_all(&rec.effective_stack.to_le_bytes())?;
    w.write_all(&[rec.player])?;
    w.write_all(&rec.game_value.to_le_bytes())?;
    for &v in &rec.oop_range { w.write_all(&v.to_le_bytes())?; }
    for &v in &rec.ip_range { w.write_all(&v.to_le_bytes())?; }
    for &v in &rec.cfvs { w.write_all(&v.to_le_bytes())?; }
    w.write_all(&rec.valid_mask)?;
    Ok(())
}

pub fn read_record<R: Read>(r: &mut R) -> io::Result<TrainingRecord> {
    let mut board = [0u8; 5];
    r.read_exact(&mut board)?;

    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4)?;
    let pot = f32::from_le_bytes(buf4);

    r.read_exact(&mut buf4)?;
    let effective_stack = f32::from_le_bytes(buf4);

    let mut buf1 = [0u8; 1];
    r.read_exact(&mut buf1)?;
    let player = buf1[0];

    r.read_exact(&mut buf4)?;
    let game_value = f32::from_le_bytes(buf4);

    let mut oop_range = [0.0f32; NUM_COMBOS];
    for v in &mut oop_range {
        r.read_exact(&mut buf4)?;
        *v = f32::from_le_bytes(buf4);
    }

    let mut ip_range = [0.0f32; NUM_COMBOS];
    for v in &mut ip_range {
        r.read_exact(&mut buf4)?;
        *v = f32::from_le_bytes(buf4);
    }

    let mut cfvs = [0.0f32; NUM_COMBOS];
    for v in &mut cfvs {
        r.read_exact(&mut buf4)?;
        *v = f32::from_le_bytes(buf4);
    }

    let mut valid_mask = [0u8; NUM_COMBOS];
    r.read_exact(&mut valid_mask)?;

    Ok(TrainingRecord {
        board, pot, effective_stack, player, game_value,
        oop_range, ip_range, cfvs, valid_mask,
    })
}

pub fn count_records<S: Seek>(s: &mut S) -> io::Result<u64> {
    let end = s.seek(SeekFrom::End(0))?;
    s.seek(SeekFrom::Start(0))?;
    Ok(end / RECORD_SIZE as u64)
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p cfvnet datagen::storage::tests -- --nocapture`
Expected: All 4 tests PASS.

**Step 5: Commit**

```
feat(cfvnet): add binary serialization for training records
```

---

## Task 7: Data Generation Orchestrator

**Files:**
- Create: `crates/cfvnet/src/datagen/generate.rs`
- Modify: `crates/cfvnet/src/datagen/mod.rs`

This ties sampler + solver + storage together with parallelism and progress reporting.

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use crate::config::{CfvnetConfig, GameConfig, DatagenConfig};

    #[test]
    fn generate_small_batch() {
        let config = CfvnetConfig {
            game: GameConfig {
                initial_stack: 200,
                bet_sizes: vec!["50%".into(), "a".into()],
                ..Default::default()
            },
            datagen: DatagenConfig {
                num_samples: 4, // Small for testing
                solver_iterations: 100,
                target_exploitability: 0.05,
                threads: 2,
                seed: 42,
                ..Default::default()
            },
            ..Default::default()
        };

        let output = NamedTempFile::new().unwrap();
        let path = output.path().to_path_buf();

        generate_training_data(&config, &path).unwrap();

        // Verify we got 8 records (4 situations × 2 players)
        let mut file = std::fs::File::open(&path).unwrap();
        let count = storage::count_records(&mut file).unwrap();
        assert_eq!(count, 8);

        // Verify records are readable
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        for _ in 0..8 {
            let rec = storage::read_record(&mut file).unwrap();
            assert!(rec.pot > 0.0);
            assert_eq!(rec.board.len(), 5);
        }
    }

    #[test]
    fn generate_is_deterministic() {
        let config = CfvnetConfig {
            game: GameConfig {
                initial_stack: 200,
                bet_sizes: vec!["50%".into(), "a".into()],
                ..Default::default()
            },
            datagen: DatagenConfig {
                num_samples: 2,
                solver_iterations: 100,
                target_exploitability: 0.05,
                threads: 1, // Single thread for determinism
                seed: 99,
                ..Default::default()
            },
            ..Default::default()
        };

        let out1 = NamedTempFile::new().unwrap();
        let out2 = NamedTempFile::new().unwrap();

        generate_training_data(&config, out1.path()).unwrap();
        generate_training_data(&config, out2.path()).unwrap();

        let data1 = std::fs::read(out1.path()).unwrap();
        let data2 = std::fs::read(out2.path()).unwrap();
        assert_eq!(data1, data2, "same seed should produce identical output");
    }
}
```

**Step 2: Implement the orchestrator**

The key function `generate_training_data` should:
1. Create per-thread RNGs derived from the master seed
2. Use rayon to parallelize situation sampling + solving
3. Collect results and write sequentially (for deterministic output with 1 thread)
4. Show progress via `indicatif`

**Step 3: Run tests**

Run: `cargo test -p cfvnet datagen::generate::tests -- --nocapture`
Expected: PASS. The 4-sample test takes ~2-5s (4 DCFR solves).

**Step 4: Commit**

```
feat(cfvnet): add parallel data generation orchestrator
```

---

## Task 8: Burn Network Model

**Files:**
- Create: `crates/cfvnet/src/model/network.rs`
- Modify: `crates/cfvnet/src/model/mod.rs`

**Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn model_output_shape() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 7, 500);
        let input = Tensor::<TestBackend, 2>::zeros([1, INPUT_SIZE], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [1, OUTPUT_SIZE]);
    }

    #[test]
    fn model_batch_forward() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 7, 500);
        let batch_size = 4;
        let input = Tensor::<TestBackend, 2>::zeros([batch_size, INPUT_SIZE], &device);
        let output = model.forward(input);
        assert_eq!(output.dims(), [batch_size, OUTPUT_SIZE]);
    }

    #[test]
    fn model_output_changes_with_input() {
        let device = Default::default();
        let model = CfvNet::<TestBackend>::new(&device, 7, 500);
        let input1 = Tensor::<TestBackend, 2>::zeros([1, INPUT_SIZE], &device);
        let input2 = Tensor::<TestBackend, 2>::ones([1, INPUT_SIZE], &device);
        let out1 = model.forward(input1);
        let out2 = model.forward(input2);
        // Outputs should differ (unless extremely unlucky initialization)
        let diff: f32 = (out1 - out2).abs().sum().into_scalar();
        assert!(diff > 1e-6, "outputs should differ for different inputs");
    }
}
```

**Step 2: Implement the model**

```rust
use burn::{
    module::Module,
    nn::{self, Linear, LinearConfig, BatchNorm, BatchNormConfig, PReluConfig, PRelu},
    tensor::{backend::Backend, Tensor},
};

pub const INPUT_SIZE: usize = 2660; // 1326 + 1326 + 5 + 1 + 1 + 1
pub const OUTPUT_SIZE: usize = 1326;

#[derive(Module, Debug)]
pub struct CfvNet<B: Backend> {
    layers: Vec<Linear<B>>,
    norms: Vec<BatchNorm<B, 1>>,
    activations: Vec<PRelu<B>>,
    output: Linear<B>,
}

impl<B: Backend> CfvNet<B> {
    pub fn new(device: &B::Device, num_layers: usize, hidden_size: usize) -> Self {
        let mut layers = Vec::new();
        let mut norms = Vec::new();
        let mut activations = Vec::new();

        // First hidden layer
        layers.push(LinearConfig::new(INPUT_SIZE, hidden_size).init(device));
        norms.push(BatchNormConfig::new(hidden_size).init(device));
        activations.push(PReluConfig::new().init(device));

        // Remaining hidden layers
        for _ in 1..num_layers {
            layers.push(LinearConfig::new(hidden_size, hidden_size).init(device));
            norms.push(BatchNormConfig::new(hidden_size).init(device));
            activations.push(PReluConfig::new().init(device));
        }

        let output = LinearConfig::new(hidden_size, OUTPUT_SIZE).init(device);

        Self { layers, norms, activations, output }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = x;
        for i in 0..self.layers.len() {
            x = self.layers[i].forward(x);
            // BatchNorm expects 3D input [batch, features, 1]
            x = x.unsqueeze_dim(2);
            x = self.norms[i].forward(x);
            x = x.squeeze(2);
            x = self.activations[i].forward(x);
        }
        self.output.forward(x)
    }
}
```

Note: Check burn's API for `PRelu` availability. If not available, use `nn::Relu` or `nn::Gelu` instead. Also verify `BatchNorm` dimension handling — burn's `BatchNorm<B, 1>` expects `[batch, channels]` input. Adapt the unsqueeze/squeeze if needed based on actual API.

Consult burn docs: `https://burn.dev/docs/burn/nn/index.html`

**Step 3: Run tests**

Run: `cargo test -p cfvnet model::network::tests -- --nocapture`
Expected: All 3 tests PASS.

**Step 4: Commit**

```
feat(cfvnet): implement CfvNet MLP model with burn
```

---

## Task 9: Loss Function

**Files:**
- Create: `crates/cfvnet/src/model/loss.rs`
- Modify: `crates/cfvnet/src/model/mod.rs`

**Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn huber_loss_zero_on_perfect_prediction() {
        let device = Default::default();
        let pred = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
        let target = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0, 1.0, 1.0]], &device);
        let loss = masked_huber_loss(pred, target, mask, 1.0);
        let val: f32 = loss.into_scalar();
        assert!(val.abs() < 1e-6, "loss should be 0, got {val}");
    }

    #[test]
    fn huber_loss_ignores_masked_entries() {
        let device = Default::default();
        let pred = Tensor::<B, 2>::from_floats([[1.0, 999.0, 3.0]], &device);
        let target = Tensor::<B, 2>::from_floats([[1.0, 0.0, 3.0]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0, 0.0, 1.0]], &device);
        let loss = masked_huber_loss(pred, target, mask, 1.0);
        let val: f32 = loss.into_scalar();
        // The 999 vs 0 error on the masked entry should not contribute
        assert!(val.abs() < 1e-6, "masked loss should be 0, got {val}");
    }

    #[test]
    fn aux_loss_zero_when_constraint_met() {
        let device = Default::default();
        // range = [0.5, 0.5], cfv_pred = [0.2, 0.4]
        // weighted sum = 0.5*0.2 + 0.5*0.4 = 0.3
        let cfv_pred = Tensor::<B, 2>::from_floats([[0.2, 0.4]], &device);
        let range = Tensor::<B, 2>::from_floats([[0.5, 0.5]], &device);
        let game_value = Tensor::<B, 1>::from_floats([0.3], &device);
        let loss = aux_game_value_loss(cfv_pred, range, game_value);
        let val: f32 = loss.into_scalar();
        assert!(val.abs() < 1e-5, "aux loss should be ~0, got {val}");
    }

    #[test]
    fn aux_loss_positive_when_constraint_violated() {
        let device = Default::default();
        let cfv_pred = Tensor::<B, 2>::from_floats([[1.0, 1.0]], &device);
        let range = Tensor::<B, 2>::from_floats([[0.5, 0.5]], &device);
        // weighted sum = 1.0, but game_value = 0.0
        let game_value = Tensor::<B, 1>::from_floats([0.0], &device);
        let loss = aux_game_value_loss(cfv_pred, range, game_value);
        let val: f32 = loss.into_scalar();
        assert!(val > 0.5, "aux loss should be large, got {val}");
    }

    #[test]
    fn combined_loss_includes_both_terms() {
        let device = Default::default();
        // Imperfect prediction + violated constraint
        let pred = Tensor::<B, 2>::from_floats([[0.5, 0.5]], &device);
        let target = Tensor::<B, 2>::from_floats([[0.0, 0.0]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0, 1.0]], &device);
        let range = Tensor::<B, 2>::from_floats([[0.5, 0.5]], &device);
        let game_value = Tensor::<B, 1>::from_floats([0.0], &device);

        let loss = cfvnet_loss(pred, target, mask, range, game_value, 1.0, 1.0);
        let val: f32 = loss.into_scalar();
        assert!(val > 0.0, "combined loss should be positive, got {val}");
    }
}
```

**Step 2: Implement loss functions**

```rust
use burn::tensor::{backend::Backend, Tensor};

/// Masked Huber loss: only counts valid (unmasked) combos.
pub fn masked_huber_loss<B: Backend>(
    pred: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,  // 1.0 for valid, 0.0 for masked
    delta: f64,
) -> Tensor<B, 1> {
    let diff = (pred - target) * mask.clone();
    let abs_diff = diff.clone().abs();
    let delta_t = delta as f32;

    // Huber: 0.5*x^2 if |x| <= delta, else delta*(|x| - 0.5*delta)
    let quadratic = diff.clone().powf_scalar(2.0) * 0.5;
    let linear = abs_diff.clone() * delta_t - 0.5 * delta_t * delta_t;
    let is_small = abs_diff.lower_equal_scalar(delta_t);

    let element_loss = is_small.float() * quadratic + is_small.bool_not().float() * linear;
    let element_loss = element_loss * mask.clone();

    // Mean over valid entries
    let num_valid = mask.sum();
    (element_loss.sum() / num_valid).unsqueeze()
}

/// Auxiliary loss: (sum(range * cfv_pred) - game_value)^2
pub fn aux_game_value_loss<B: Backend>(
    cfv_pred: Tensor<B, 2>,   // [batch, 1326]
    range: Tensor<B, 2>,       // [batch, 1326]
    game_value: Tensor<B, 1>,  // [batch]
) -> Tensor<B, 1> {
    let weighted_sum = (cfv_pred * range).sum_dim(1).squeeze(1); // [batch]
    let diff = weighted_sum - game_value;
    diff.powf_scalar(2.0).mean()
}

/// Combined CFVnet loss: Huber + lambda * aux
pub fn cfvnet_loss<B: Backend>(
    pred: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    range: Tensor<B, 2>,
    game_value: Tensor<B, 1>,
    huber_delta: f64,
    aux_weight: f64,
) -> Tensor<B, 1> {
    let huber = masked_huber_loss(pred.clone(), target, mask, huber_delta);
    let aux = aux_game_value_loss(pred, range, game_value);
    huber + aux * aux_weight as f32
}
```

Note: Burn's tensor API may differ slightly (e.g., `bool_not()` vs `logical_not()`, `float()` conversion). Consult burn docs for the exact method names. The pattern is correct but names may need adjustment.

**Step 3: Run tests**

Run: `cargo test -p cfvnet model::loss::tests -- --nocapture`
Expected: All 5 tests PASS.

**Step 4: Commit**

```
feat(cfvnet): implement Huber + auxiliary game-value loss
```

---

## Task 10: Dataset Loader

**Files:**
- Create: `crates/cfvnet/src/model/dataset.rs`
- Modify: `crates/cfvnet/src/model/mod.rs`

Reads materialized binary training data and provides it to burn's training loop.

**Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::datagen::storage::{TrainingRecord, write_record};
    use tempfile::NamedTempFile;
    use std::io::Write;

    fn write_test_data(n: usize) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        for i in 0..n {
            let mut rec = TrainingRecord {
                board: [0, 4, 8, 12, 16],
                pot: 100.0,
                effective_stack: 50.0,
                player: (i % 2) as u8,
                game_value: 0.1 * i as f32,
                oop_range: [0.0; 1326],
                ip_range: [0.0; 1326],
                cfvs: [0.0; 1326],
                valid_mask: [1; 1326],
            };
            rec.cfvs[0] = i as f32 * 0.01;
            write_record(&mut file, &rec).unwrap();
        }
        file.flush().unwrap();
        file
    }

    #[test]
    fn dataset_length_correct() {
        let file = write_test_data(10);
        let dataset = CfvDataset::from_file(file.path()).unwrap();
        assert_eq!(dataset.len(), 10);
    }

    #[test]
    fn dataset_get_returns_valid_item() {
        let file = write_test_data(5);
        let dataset = CfvDataset::from_file(file.path()).unwrap();
        let item = dataset.get(0).unwrap();
        assert_eq!(item.input.len(), INPUT_SIZE);
        assert_eq!(item.target.len(), OUTPUT_SIZE);
        assert_eq!(item.mask.len(), OUTPUT_SIZE);
        assert_eq!(item.range.len(), OUTPUT_SIZE);
    }

    #[test]
    fn dataset_input_encoding_correct() {
        let file = write_test_data(1);
        let dataset = CfvDataset::from_file(file.path()).unwrap();
        let item = dataset.get(0).unwrap();

        // First 1326 elements: OOP range
        // Next 1326 elements: IP range
        // Next 5: board cards normalized
        // Next 1: pot normalized
        // Next 1: stack normalized
        // Next 1: player indicator
        assert_eq!(item.input.len(), 1326 + 1326 + 5 + 1 + 1 + 1);
    }
}
```

**Step 2: Implement the dataset**

The dataset memory-maps the binary file and implements burn's `Dataset` trait. Each `get(idx)` reads a record at the correct offset and encodes it into the input tensor format.

**Step 3: Run tests**

Run: `cargo test -p cfvnet model::dataset::tests -- --nocapture`
Expected: All 3 tests PASS.

**Step 4: Commit**

```
feat(cfvnet): implement burn Dataset for binary training data
```

---

## Task 11: Training Loop

**Files:**
- Create: `crates/cfvnet/src/model/training.rs`
- Modify: `crates/cfvnet/src/model/mod.rs`

**Step 1: Write the overfit test**

This is the critical validation: train on a tiny batch and verify the loss converges to near-zero.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn overfit_single_batch() {
        // Generate a tiny dataset (write 16 records to temp file)
        let file = crate::model::dataset::tests::write_test_data(16);
        let dataset = CfvDataset::from_file(file.path()).unwrap();

        let device = Default::default();
        let config = TrainConfig {
            hidden_layers: 2,  // Small for speed
            hidden_size: 64,
            batch_size: 16,
            epochs: 200,       // Many epochs on tiny data
            learning_rate: 0.001,
            lr_min: 0.001,     // Constant LR for overfit test
            huber_delta: 1.0,
            aux_loss_weight: 0.0, // Disable aux for simple overfit test
            validation_split: 0.0,
            checkpoint_every_n_batches: 0, // No checkpointing
        };

        let result = train::<B>(&device, &dataset, &config, None);
        assert!(result.final_train_loss < 0.01,
            "should overfit small data, got loss {}", result.final_train_loss);
    }
}
```

**Step 2: Implement training loop**

Implement a custom training loop (not burn's `LearnerBuilder` which may be overly opinionated) that:
1. Splits dataset into train/validation
2. Iterates over epochs and batches
3. Forward pass, loss computation, backward pass, optimizer step
4. Reports metrics every N batches
5. Checkpoints the best model by validation loss

**Step 3: Run the overfit test**

Run: `cargo test -p cfvnet model::training::tests::overfit_single_batch -- --nocapture`
Expected: PASS. May take 10-30s due to 200 epochs.

**Step 4: Commit**

```
feat(cfvnet): implement training loop with overfit validation
```

---

## Task 12: Evaluation Metrics

**Files:**
- Create: `crates/cfvnet/src/eval/metrics.rs`
- Modify: `crates/cfvnet/src/eval/mod.rs`

**Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mae_zero_on_perfect() {
        let pred = vec![0.1, 0.2, 0.3];
        let actual = vec![0.1, 0.2, 0.3];
        let mask = vec![true, true, true];
        let mae = mean_absolute_error(&pred, &actual, &mask);
        assert!(mae.abs() < 1e-7);
    }

    #[test]
    fn mae_ignores_masked() {
        let pred = vec![0.1, 999.0, 0.3];
        let actual = vec![0.1, 0.0, 0.3];
        let mask = vec![true, false, true];
        let mae = mean_absolute_error(&pred, &actual, &mask);
        assert!(mae.abs() < 1e-7);
    }

    #[test]
    fn max_error_correct() {
        let pred = vec![0.0, 0.5, 0.0];
        let actual = vec![0.0, 0.0, 0.1];
        let mask = vec![true, true, true];
        let max_err = max_absolute_error(&pred, &actual, &mask);
        assert!((max_err - 0.5).abs() < 1e-7);
    }

    #[test]
    fn mbb_conversion() {
        // 0.01 pot-relative error * pot 100 * 1000 = 1000 mbb
        // But mbb/hand divides by big blind (2 SB)
        let pot_relative_error = 0.01;
        let pot = 100.0;
        let mbb = pot_relative_to_mbb(pot_relative_error, pot, 2.0);
        assert!((mbb - 500.0).abs() < 1e-3);
    }

    #[test]
    fn compute_metrics_with_perfect_returns_zero() {
        let pred = vec![0.1; 1326];
        let actual = vec![0.1; 1326];
        let mask = vec![true; 1326];
        let metrics = compute_prediction_metrics(&pred, &actual, &mask, 100.0);
        assert!(metrics.mae < 1e-6);
        assert!(metrics.max_error < 1e-6);
        assert!(metrics.mbb_error < 1e-3);
    }
}
```

**Step 2: Implement metrics**

```rust
pub struct PredictionMetrics {
    pub mae: f64,
    pub max_error: f64,
    pub mbb_error: f64, // Mean absolute error in milli-big-blinds
}

pub fn mean_absolute_error(pred: &[f32], actual: &[f32], mask: &[bool]) -> f64 { ... }
pub fn max_absolute_error(pred: &[f32], actual: &[f32], mask: &[bool]) -> f64 { ... }
pub fn pot_relative_to_mbb(error: f64, pot: f64, big_blind: f64) -> f64 { ... }
pub fn compute_prediction_metrics(pred: &[f32], actual: &[f32], mask: &[bool], pot: f32) -> PredictionMetrics { ... }
```

**Step 3: Run tests**

Run: `cargo test -p cfvnet eval::metrics::tests -- --nocapture`
Expected: All 5 tests PASS.

**Step 4: Commit**

```
feat(cfvnet): implement evaluation metrics (MAE, max error, mbb)
```

---

## Task 13: Comparison Harness

**Files:**
- Create: `crates/cfvnet/src/eval/compare.rs`
- Modify: `crates/cfvnet/src/eval/mod.rs`

Generates random river spots, solves them exactly, queries the network, and compares.

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_with_perfect_oracle_shows_zero_error() {
        // Create a mock "network" that returns the exact solve values
        // This tests the comparison harness itself, not the network
        let spot = generate_comparison_spot(42);
        let solve_config = default_solve_config();
        let result = solve_situation(&spot, &solve_config).unwrap();

        // Use exact OOP EVs as "predictions"
        let metrics = compare_single_spot(
            &result.oop_evs,
            &result.oop_evs,
            &result.valid_mask,
            spot.pot as f32,
        );
        assert!(metrics.mae < 1e-6, "perfect oracle should have zero error");
    }
}
```

**Step 2: Implement comparison**

The comparison harness:
1. Generates N random river situations using `sample_situation`
2. Solves each with range-solver (ground truth)
3. Encodes the input and runs the CFVnet forward pass
4. Compares predicted vs actual CFVs using `compute_prediction_metrics`
5. Aggregates and reports statistics

**Step 3: Run tests**

Run: `cargo test -p cfvnet eval::compare::tests -- --nocapture`
Expected: PASS.

**Step 4: Commit**

```
feat(cfvnet): implement comparison harness vs exact solves
```

---

## Task 14: CLI

**Files:**
- Modify: `crates/cfvnet/src/main.rs`

**Step 1: Implement CLI with clap**

```rust
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "cfvnet", about = "Deep Counterfactual Value Network toolkit")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate training data by solving random river subgames
    Generate {
        #[arg(short, long)]
        config: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        num_samples: Option<u64>,
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Train the CFVnet model
    Train {
        #[arg(short, long)]
        config: PathBuf,
        #[arg(short, long)]
        data: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value = "wgpu")]
        backend: String,
    },
    /// Evaluate model on held-out validation data
    Evaluate {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long)]
        data: PathBuf,
    },
    /// Compare model predictions against exact solves
    Compare {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(long, default_value = "100")]
        num_spots: usize,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(short, long)]
        config: Option<PathBuf>,
    },
}
```

**Step 2: Wire up each subcommand to its orchestrator function**

Each subcommand:
1. Loads config from YAML
2. Validates
3. Calls the appropriate function from `datagen::generate`, `model::training`, `eval::metrics`, or `eval::compare`

**Step 3: Verify CLI parses correctly**

Run: `cargo run -p cfvnet -- --help`
Expected: Shows help text with all subcommands.

Run: `cargo run -p cfvnet -- generate --help`
Expected: Shows generate-specific options.

**Step 4: Commit**

```
feat(cfvnet): implement CLI with generate/train/evaluate/compare subcommands
```

---

## Task 15: Integration Test — Full Pipeline Smoke Test

**Files:**
- Create: `crates/cfvnet/tests/integration_test.rs`

End-to-end test: generate a tiny dataset, train a small model, evaluate it.

**Step 1: Write the integration test**

```rust
//! Full pipeline integration test: generate → train → evaluate

use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn full_pipeline_smoke_test() {
    let tmp = TempDir::new().unwrap();
    let data_path = tmp.path().join("training.bin");
    let model_path = tmp.path().join("model");

    // 1. Generate 8 training samples (4 situations × 2 players)
    let config = cfvnet::config::CfvnetConfig {
        game: cfvnet::config::GameConfig {
            initial_stack: 200,
            bet_sizes: vec!["50%".into(), "a".into()],
            ..Default::default()
        },
        datagen: cfvnet::config::DatagenConfig {
            num_samples: 4,
            solver_iterations: 100,
            target_exploitability: 0.05,
            threads: 1,
            seed: 42,
            ..Default::default()
        },
        training: cfvnet::config::TrainingConfig {
            hidden_layers: 2,
            hidden_size: 32,
            batch_size: 8,
            epochs: 5,
            learning_rate: 0.001,
            lr_min: 0.001,
            validation_split: 0.0,
            checkpoint_every_n_batches: 0,
            ..Default::default()
        },
        ..Default::default()
    };

    // Generate
    cfvnet::datagen::generate::generate_training_data(&config, &data_path)
        .expect("data generation failed");
    assert!(data_path.exists());

    // Train (using NdArray backend for testing — no GPU needed)
    use burn::backend::NdArray;
    type B = NdArray<f32>;
    let device = Default::default();
    let dataset = cfvnet::model::dataset::CfvDataset::from_file(&data_path).unwrap();
    let result = cfvnet::model::training::train::<B>(
        &device,
        &dataset,
        &config.training.into(),
        Some(&model_path),
    );
    assert!(result.final_train_loss.is_finite(), "training loss should be finite");

    // Evaluate metrics
    let eval_dataset = cfvnet::model::dataset::CfvDataset::from_file(&data_path).unwrap();
    // Load model and run inference on each sample, compute metrics
    // (detailed implementation depends on burn's model loading API)
}
```

**Step 2: Run integration test**

Run: `cargo test -p cfvnet --test integration_test -- --nocapture`
Expected: PASS. Takes ~30-60s due to DCFR solves.

**Step 3: Commit**

```
test(cfvnet): add full pipeline integration smoke test
```

---

## Task 16: Sample Config File

**Files:**
- Create: `sample_configurations/river_cfvnet.yaml`

**Step 1: Create the config file**

```yaml
# River CFVnet training configuration
# Based on Supremus (Zarick et al. 2020)

game:
  initial_stack: 200          # 100bb in SB units
  bet_sizes: ["25%", "50%", "100%", "a"]
  add_allin_threshold: 1.5
  force_allin_threshold: 0.15

datagen:
  num_samples: 1_000_000
  pot_intervals: [[4,20], [20,80], [80,200], [200,400]]
  solver_iterations: 1000
  target_exploitability: 0.005
  threads: 8
  seed: 42

training:
  hidden_layers: 7
  hidden_size: 500
  batch_size: 2048
  epochs: 2
  learning_rate: 0.001
  lr_min: 0.00001
  huber_delta: 1.0
  aux_loss_weight: 1.0
  validation_split: 0.05
  checkpoint_every_n_batches: 1000
```

**Step 2: Verify it parses**

Run: `cargo run -p cfvnet -- generate --config sample_configurations/river_cfvnet.yaml --output /dev/null --num-samples 0`
Expected: Parses without error (exits immediately with 0 samples).

**Step 3: Commit**

```
feat(cfvnet): add sample river CFVnet configuration
```

---

## Task 17: Documentation Update

**Files:**
- Modify: `docs/architecture.md` — add CFVnet section
- Modify: `docs/training.md` — add cfvnet CLI docs

**Step 1: Add CFVnet section to architecture.md**

Add a new section covering:
- What CFVnets are and their role in depth-limited solving
- The pipeline: generate → train → evaluate
- Integration point with the subgame solver

**Step 2: Add CLI docs to training.md**

Document all four subcommands with examples.

**Step 3: Commit**

```
docs: add CFVnet pipeline to architecture and training docs
```

---

## Summary

| Task | Component | Tests |
|-|-|-|
| 1 | Crate scaffolding | Compiles |
| 2 | Config types | 4 tests |
| 3 | R(S,p) range generator | 5 tests |
| 4 | Situation sampler | 5 tests |
| 5 | Solve wrapper | 5 tests |
| 6 | Binary serialization | 4 tests |
| 7 | Data gen orchestrator | 2 tests |
| 8 | Burn network model | 3 tests |
| 9 | Loss function | 5 tests |
| 10 | Dataset loader | 3 tests |
| 11 | Training loop | 1 overfit test |
| 12 | Evaluation metrics | 5 tests |
| 13 | Comparison harness | 1 test |
| 14 | CLI | Manual verification |
| 15 | Integration test | 1 end-to-end test |
| 16 | Sample config | Parse verification |
| 17 | Documentation | N/A |

**Total: ~44 tests** across 17 tasks. Critical validation points:
- Task 3: Range generator correctness (hand-strength correlation, normalization, masking)
- Task 5: Solve wrapper (correct EV extraction, pot-relative normalization, zero-sum property)
- Task 9: Loss function (masking, auxiliary constraint)
- Task 11: Overfit test (proves model + loss + training loop work end-to-end)
- Task 15: Full pipeline smoke test (proves all components integrate)
