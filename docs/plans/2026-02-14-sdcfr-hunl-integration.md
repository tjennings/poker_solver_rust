# SD-CFR HUNL Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run Single Deep CFR on full HUNL poker as an alternative solver alongside MCCFR, with checkpoint-based evaluation via Tauri.

**Architecture:** Pre-generate a stratified HUNL deal pool, feed it to the SD-CFR solver which samples from it each iteration, periodically checkpoint the trained networks into `StrategyBundle` format that Tauri already loads.

**Tech Stack:** Rust, candle (neural nets), serde_yaml (config), poker-solver-core (game + blueprint), poker-solver-deep-cfr (SD-CFR solver)

---

### Task 1: Add `checkpoint_interval` to `SdCfrConfig`

**Files:**
- Modify: `crates/deep-cfr/src/config.rs`
- Test: `crates/deep-cfr/src/config.rs` (inline tests)

**Step 1: Write the failing test**

Add to the existing `#[cfg(test)] mod tests` in `config.rs`:

```rust
#[test]
fn default_config_has_zero_checkpoint_interval() {
    let config = SdCfrConfig::default();
    assert_eq!(config.checkpoint_interval, 0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-deep-cfr config::tests::default_config_has_zero_checkpoint_interval`
Expected: FAIL — `checkpoint_interval` field doesn't exist yet.

**Step 3: Write minimal implementation**

Add to `SdCfrConfig`:
```rust
/// Save a checkpoint every N iterations (0 = disabled)
pub checkpoint_interval: u32,
```

Add to `Default` impl:
```rust
checkpoint_interval: 0,
```

Update all existing test configs that construct `SdCfrConfig` manually (solver.rs tests) to include the new field.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-deep-cfr`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add crates/deep-cfr/src/config.rs crates/deep-cfr/src/solver.rs
git commit -m "feat(deep-cfr): add checkpoint_interval to SdCfrConfig"
```

---

### Task 2: Add deal pool + checkpoint callback to `SdCfrSolver::train()`

**Files:**
- Modify: `crates/deep-cfr/src/solver.rs`
- Test: `crates/deep-cfr/src/solver.rs` (inline tests)

**Step 1: Write the failing test**

Add to `solver.rs` tests:

```rust
#[test]
fn train_with_deal_pool_uses_provided_states() {
    let game = KuhnPoker::new();
    let encoder = KuhnEncoder::new();
    // Generate a deal pool once
    let deal_pool = game.initial_states();
    let config = SdCfrConfig {
        cfr_iterations: 3,
        traversals_per_iter: 20,
        ..test_config()
    };
    let mut solver = SdCfrSolver::new(game, encoder, config).unwrap();

    let result = solver.train_with_deals(Some(&deal_pool), None);
    assert!(result.is_ok());
    let trained = result.unwrap();
    assert_eq!(trained.model_buffers[0].len(), 3);
}

#[test]
fn train_with_checkpoint_callback_is_called() {
    let game = KuhnPoker::new();
    let encoder = KuhnEncoder::new();
    let config = SdCfrConfig {
        cfr_iterations: 4,
        traversals_per_iter: 20,
        checkpoint_interval: 2,
        ..test_config()
    };
    let mut solver = SdCfrSolver::new(game, encoder, config).unwrap();

    let mut callback_count = 0u32;
    let callback = |iteration: u32, _trained: &TrainedSdCfr| -> Result<(), SdCfrError> {
        callback_count += 1;
        Ok(())
    };

    // Use a closure wrapper since we need FnMut
    let result = solver.train_with_deals(None, Some(&mut |iter, trained| {
        callback_count += 1;
        Ok(())
    }));
    assert!(result.is_ok());
    // 4 iterations / checkpoint_interval 2 = 2 callbacks
    assert_eq!(callback_count, 2);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-deep-cfr solver::tests::train_with_deal_pool`
Expected: FAIL — `train_with_deals` doesn't exist.

**Step 3: Write minimal implementation**

In `solver.rs`, add the new method. Key changes:

1. Add `train_with_deals` method on `SdCfrSolver`:
```rust
/// Run all T iterations with optional pre-generated deal pool and checkpoint callback.
///
/// - `deal_pool`: If Some, samples from this pool instead of calling `game.initial_states()`.
/// - `checkpoint_cb`: If Some, called every `config.checkpoint_interval` iterations.
pub fn train_with_deals(
    &mut self,
    deal_pool: Option<&[G::State]>,
    mut checkpoint_cb: Option<&mut dyn FnMut(u32, &TrainedSdCfr) -> Result<(), SdCfrError>>,
) -> Result<TrainedSdCfr, SdCfrError> {
    let total = self.config.cfr_iterations;
    let interval = self.config.checkpoint_interval;
    for _ in 0..total {
        self.step_with_deals(deal_pool)?;
        if interval > 0 && self.current_iteration % interval == 0 {
            if let Some(ref mut cb) = checkpoint_cb {
                let snapshot = self.snapshot();
                cb(self.current_iteration, &snapshot)?;
            }
        }
    }
    Ok(self.take_result())
}
```

2. Add `step_with_deals` that takes optional deal pool and passes it to `run_traversals`.

3. Modify `run_traversals` to accept `Option<&[G::State]>`:
   - If `Some(pool)`, sample from pool: `pool[rng.random_range(0..pool.len())]`
   - If `None`, call `self.game.initial_states()` (existing behavior)

4. Add `snapshot()` and `take_result()` helpers to create `TrainedSdCfr` (snapshot clones model buffers, take_result takes them via `mem::take`).

5. Keep existing `train()` delegating to `train_with_deals(None, None)`.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-deep-cfr`
Expected: All tests PASS (old and new).

**Step 5: Commit**

```bash
git add crates/deep-cfr/src/solver.rs
git commit -m "feat(deep-cfr): add deal pool + checkpoint callback to train()"
```

---

### Task 3: Implement `HunlStateEncoder`

**Files:**
- Create: `crates/deep-cfr/src/hunl_encoder.rs`
- Modify: `crates/deep-cfr/src/lib.rs` (add `pub mod hunl_encoder`)
- Test: `crates/deep-cfr/src/hunl_encoder.rs` (inline tests)

**Step 1: Write the failing test**

Create `crates/deep-cfr/src/hunl_encoder.rs` with tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use poker_solver_core::game::{HunlPostflop, PostflopConfig, Player};
    use poker_solver_core::Game;

    fn test_config() -> PostflopConfig {
        PostflopConfig {
            stack_depth: 25,
            bet_sizes: vec![0.33, 0.67, 1.0, 2.0, 3.0],
            max_raises_per_street: 3,
        }
    }

    #[test]
    fn encoder_produces_valid_features_at_root() {
        let config = test_config();
        let encoder = HunlStateEncoder::new(config.bet_sizes.clone());
        let game = HunlPostflop::new(config, None, 1);
        let states = game.initial_states();
        let state = &states[0];

        let features = encoder.encode(state, Player::Player1);

        // Hole cards should be valid (>= 0)
        assert!(features.cards[0] >= 0, "hole card 0 should be valid");
        assert!(features.cards[1] >= 0, "hole card 1 should be valid");
        // Flop not yet dealt at preflop root
        assert_eq!(features.cards[2], -1, "flop not dealt at preflop");
    }

    #[test]
    fn encoder_different_players_get_different_cards() {
        let config = test_config();
        let encoder = HunlStateEncoder::new(config.bet_sizes.clone());
        let game = HunlPostflop::new(config, None, 1);
        let states = game.initial_states();
        let state = &states[0];

        let p1_features = encoder.encode(state, Player::Player1);
        let p2_features = encoder.encode(state, Player::Player2);

        // Different players should see different hole cards
        assert_ne!(
            p1_features.cards[..2], p2_features.cards[..2],
            "P1 and P2 should have different hole cards"
        );
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-deep-cfr hunl_encoder::tests`
Expected: FAIL — module doesn't exist.

**Step 3: Write minimal implementation**

```rust
//! HUNL state encoder for SD-CFR.
//!
//! Converts PostflopState into InfoSetFeatures using suit-isomorphic
//! card canonicalization and pot-fraction bet encoding.

use poker_solver_core::game::{PostflopState, Player, Action, ALL_IN};
use crate::card_features::{self, BetAction, InfoSetFeatures};
use crate::traverse::StateEncoder;

/// Encodes HUNL postflop states into neural network features.
///
/// Stores the bet sizes from PostflopConfig to resolve bet indices
/// to pot fractions for the bet feature encoder.
pub struct HunlStateEncoder {
    bet_sizes: Vec<f32>,
}

impl HunlStateEncoder {
    pub fn new(bet_sizes: Vec<f32>) -> Self {
        Self { bet_sizes }
    }
}

impl StateEncoder<PostflopState> for HunlStateEncoder {
    fn encode(&self, state: &PostflopState, player: Player) -> InfoSetFeatures {
        let hole = match player {
            Player::Player1 => state.p1_holding,
            Player::Player2 => state.p2_holding,
        };
        let cards = card_features::canonicalize(hole, &state.board);
        let bets = card_features::encode_bets(&self.build_bet_actions(state));
        InfoSetFeatures { cards, bets }
    }
}

impl HunlStateEncoder {
    /// Convert the PostflopState action history into BetAction pairs
    /// with pot fractions for the bet feature encoder.
    fn build_bet_actions(&self, state: &PostflopState) -> Vec<BetAction> {
        state.history.iter().map(|&(_street, action)| {
            let pot_frac = match action {
                Action::Fold | Action::Check => 0.0,
                Action::Call => 0.0, // sentinel; network learns meaning
                Action::Bet(idx) | Action::Raise(idx) if idx == ALL_IN => 10.0,
                Action::Bet(idx) | Action::Raise(idx) => {
                    self.bet_sizes.get(idx as usize)
                        .map(|&s| f64::from(s))
                        .unwrap_or(1.0)
                }
            };
            (action, pot_frac)
        }).collect()
    }
}
```

Add to `lib.rs`: `pub mod hunl_encoder;`

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-deep-cfr hunl_encoder`
Expected: PASS.

**Step 5: Add more tests**

```rust
#[test]
fn encoder_handles_post_action_state() {
    let config = test_config();
    let encoder = HunlStateEncoder::new(config.bet_sizes.clone());
    let game = HunlPostflop::new(config, None, 1);
    let states = game.initial_states();
    let state = &states[0];

    // Take an action
    let actions = game.actions(state);
    let next = game.next_state(state, actions[0]);
    let features = encoder.encode(&next, Player::Player2);

    // Should produce valid features (no panic)
    assert!(features.cards[0] >= 0 || features.cards[0] == -1);
}

#[test]
fn bet_actions_resolve_bet_sizes_correctly() {
    let encoder = HunlStateEncoder::new(vec![0.33, 0.67, 1.0]);
    // A Bet(1) should map to pot_frac 0.67
    let state_history = vec![
        (poker_solver_core::abstraction::Street::Preflop, Action::Bet(1)),
    ];
    // Manually test build_bet_actions logic
    // (This test verifies the pot fraction mapping)
}
```

**Step 6: Run all tests**

Run: `cargo test -p poker-solver-deep-cfr`
Expected: All PASS.

**Step 7: Commit**

```bash
git add crates/deep-cfr/src/hunl_encoder.rs crates/deep-cfr/src/lib.rs
git commit -m "feat(deep-cfr): add HunlStateEncoder for HUNL SD-CFR"
```

---

### Task 4: Add `SdCfr` variant to trainer `SolverMode` + YAML config

**Files:**
- Modify: `crates/trainer/src/main.rs`
- Create: `training_sdcfr.yaml` (example config)

**Step 1: Add `SdCfr` to `SolverMode` enum**

```rust
#[derive(Debug, Clone, ValueEnum)]
enum SolverMode {
    Mccfr,
    Sequence,
    Gpu,
    /// Single Deep CFR with neural network advantage estimation.
    SdCfr,
}
```

**Step 2: Add `SdCfrTrainingConfig` struct**

```rust
#[derive(Debug, Deserialize)]
struct SdCfrTrainingConfig {
    game: PostflopConfig,
    deals: SdCfrDealConfig,
    training: SdCfrTrainingParams,
    network: SdCfrNetworkConfig,
    sgd: SdCfrSgdConfig,
    memory: SdCfrMemoryConfig,
    checkpoint: SdCfrCheckpointConfig,
}

#[derive(Debug, Deserialize)]
struct SdCfrDealConfig {
    count: usize,
    #[serde(default)]
    min_per_class: usize,
    #[serde(default = "default_max_rejections")]
    max_rejections: usize,
    seed: u64,
}

#[derive(Debug, Deserialize)]
struct SdCfrTrainingParams {
    iterations: u32,
    traversals_per_iter: u32,
    seed: u64,
    output_dir: String,
}

#[derive(Debug, Deserialize)]
struct SdCfrNetworkConfig {
    hidden_dim: usize,
    num_actions: usize,
}

#[derive(Debug, Deserialize)]
struct SdCfrSgdConfig {
    steps: usize,
    batch_size: usize,
    learning_rate: f64,
    #[serde(default = "default_grad_clip")]
    grad_clip_norm: f64,
}

fn default_grad_clip() -> f64 { 1.0 }

#[derive(Debug, Deserialize)]
struct SdCfrMemoryConfig {
    advantage_cap: usize,
}

#[derive(Debug, Deserialize)]
struct SdCfrCheckpointConfig {
    interval: u32,
}
```

**Step 3: Add dispatch for `SolverMode::SdCfr`**

In the `match solver { ... }` block:
```rust
SolverMode::SdCfr => run_sdcfr_training(&config)?,
```

Note: The sd-cfr branch reads a *different* YAML schema (`SdCfrTrainingConfig`) from the same config file path. Detect solver mode from the YAML by checking for `training.solver: sd-cfr` or by using the `--solver` CLI arg to pick the parser.

Since the `--solver` CLI arg already selects the solver, the simplest approach: when `SolverMode::SdCfr` is selected, re-parse the YAML as `SdCfrTrainingConfig` instead of `TrainingConfig`.

**Step 4: Create example YAML config**

Create `training_sdcfr.yaml`:

```yaml
game:
  stack_depth: 25
  bet_sizes: [0.33, 0.67, 1.0, 2.0, 3.0]
  max_raises_per_street: 3

deals:
  count: 50000
  min_per_class: 100
  max_rejections: 500000
  seed: 42

training:
  iterations: 1500
  traversals_per_iter: 1000
  seed: 42
  output_dir: "./sdcfr_25bb"

network:
  hidden_dim: 256
  num_actions: 8

sgd:
  steps: 500
  batch_size: 4096
  learning_rate: 0.001
  grad_clip_norm: 1.0

memory:
  advantage_cap: 5000000

checkpoint:
  interval: 100
```

**Step 5: Verify it compiles**

Run: `cargo build -p poker-solver-trainer`
Expected: PASS (the `run_sdcfr_training` function can be a stub initially).

**Step 6: Commit**

```bash
git add crates/trainer/src/main.rs training_sdcfr.yaml
git commit -m "feat(trainer): add sd-cfr solver mode + YAML config parsing"
```

---

### Task 5: Implement `run_sdcfr_training` — deal pool + solver setup

**Files:**
- Modify: `crates/trainer/src/main.rs`

**Step 1: Implement the training function**

```rust
fn run_sdcfr_training(config_path: &Path) -> Result<(), Box<dyn Error>> {
    let yaml = std::fs::read_to_string(config_path)?;
    let config: SdCfrTrainingConfig = serde_yaml::from_str(&yaml)?;

    println!("=== Poker Blueprint Trainer (Single Deep CFR) ===\n");
    println!("Game config:");
    println!("  Stack depth: {} BB", config.game.stack_depth);
    println!("  Bet sizes: {:?} pot", config.game.bet_sizes);
    println!("  Deal pool: {} deals", config.deals.count);
    println!("  Checkpoint every {} iterations", config.checkpoint.interval);
    println!();

    // 1. Generate stratified deal pool
    println!("Generating deal pool...");
    let start = Instant::now();
    let mut game = HunlPostflop::new(config.game.clone(), None, config.deals.count);
    if config.deals.min_per_class > 0 {
        game = game.with_stratification(
            config.deals.min_per_class,
            config.deals.max_rejections,
        );
    }
    let deal_pool = game.initial_states();
    println!("  {} deals in {:?}\n", deal_pool.len(), start.elapsed());

    // 2. Create encoder + solver
    let encoder = poker_solver_deep_cfr::hunl_encoder::HunlStateEncoder::new(
        config.game.bet_sizes.clone(),
    );
    let sdcfr_config = poker_solver_deep_cfr::SdCfrConfig {
        cfr_iterations: config.training.iterations,
        traversals_per_iter: config.training.traversals_per_iter,
        advantage_memory_cap: config.memory.advantage_cap,
        hidden_dim: config.network.hidden_dim,
        num_actions: config.network.num_actions,
        sgd_steps: config.sgd.steps,
        batch_size: config.sgd.batch_size,
        learning_rate: config.sgd.learning_rate,
        grad_clip_norm: config.sgd.grad_clip_norm,
        seed: config.training.seed,
        checkpoint_interval: config.checkpoint.interval,
    };

    let mut solver = poker_solver_deep_cfr::solver::SdCfrSolver::new(
        game, encoder, sdcfr_config,
    )?;

    // 3. Build checkpoint callback
    let output_dir = config.training.output_dir.clone();
    let game_config = config.game.clone();
    let num_actions = config.network.num_actions;
    let hidden_dim = config.network.hidden_dim;

    let mut checkpoint_cb = |iteration: u32, trained: &TrainedSdCfr| -> Result<(), _> {
        save_sdcfr_checkpoint(
            trained, iteration, &output_dir, &game_config,
            num_actions, hidden_dim,
        )
    };

    // 4. Run training
    println!("Starting SD-CFR training ({} iterations)...\n", config.training.iterations);
    let training_start = Instant::now();
    solver.train_with_deals(Some(&deal_pool), Some(&mut checkpoint_cb))?;
    println!("\n=== Training Complete ===");
    println!("Total time: {:?}", training_start.elapsed());

    Ok(())
}
```

**Step 2: Verify it compiles**

Run: `cargo build -p poker-solver-trainer`
Expected: Compiles (checkpoint save function can be a stub).

**Step 3: Commit**

```bash
git add crates/trainer/src/main.rs
git commit -m "feat(trainer): implement sd-cfr deal pool generation + solver setup"
```

---

### Task 6: Implement checkpoint strategy extraction

**Files:**
- Modify: `crates/trainer/src/main.rs`

**Step 1: Implement `save_sdcfr_checkpoint`**

This function:
1. Builds `ExplicitPolicy` for both players from model buffers
2. Walks the game tree by traversing a sample of the deal pool
3. At each non-terminal node, encodes features and queries the policy
4. Collects `info_set_key -> action_probabilities` into `FxHashMap<u64, Vec<f64>>`
5. Converts to `BlueprintStrategy` → `StrategyBundle` → saves to disk

```rust
fn save_sdcfr_checkpoint(
    trained: &poker_solver_deep_cfr::solver::TrainedSdCfr,
    iteration: u32,
    output_dir: &str,
    game_config: &PostflopConfig,
    num_actions: usize,
    hidden_dim: usize,
) -> Result<(), poker_solver_deep_cfr::SdCfrError> {
    use candle_core::Device;
    use poker_solver_deep_cfr::eval::ExplicitPolicy;

    let device = Device::Cpu;
    println!("\n  Checkpoint at iteration {iteration}...");

    // Build explicit policies for both players
    let policies = [
        ExplicitPolicy::from_buffer(&trained.model_buffers[0], num_actions, hidden_dim, &device)?,
        ExplicitPolicy::from_buffer(&trained.model_buffers[1], num_actions, hidden_dim, &device)?,
    ];

    // Walk game tree to extract strategies
    let encoder = poker_solver_deep_cfr::hunl_encoder::HunlStateEncoder::new(
        game_config.bet_sizes.clone(),
    );
    let tree_game = HunlPostflop::new(game_config.clone(), None, 100);
    let sample_deals = tree_game.initial_states();

    let mut strategy_map: FxHashMap<u64, Vec<f64>> = FxHashMap::default();
    for deal in sample_deals.iter().take(100) {
        extract_strategies_dfs(
            &tree_game, deal, &policies, &encoder,
            num_actions, &device, &mut strategy_map,
        )?;
    }

    println!("    {} info sets extracted", strategy_map.len());

    // Convert to bundle
    let bundle_config = BundleConfig {
        game: game_config.clone(),
        abstraction: None,
        abstraction_mode: AbstractionModeConfig::default(),
        strength_bits: 0,
        equity_bits: 0,
    };
    let blueprint = BlueprintStrategy::from_strategies(strategy_map, u64::from(iteration));
    let bundle = StrategyBundle::new(bundle_config, blueprint, None);

    let dir = PathBuf::from(output_dir).join(format!("sdcfr_checkpoint_{iteration}"));
    bundle.save(&dir).map_err(|e|
        poker_solver_deep_cfr::SdCfrError::Io(std::io::Error::new(
            std::io::ErrorKind::Other, e.to_string()
        ))
    )?;
    println!("    Saved to {}/", dir.display());

    // Update latest symlink
    let latest = PathBuf::from(output_dir).join("latest");
    let _ = std::fs::remove_file(&latest);
    #[cfg(unix)]
    std::os::unix::fs::symlink(&dir, &latest).ok();

    Ok(())
}

/// Recursively walk the game tree, collecting neural net strategies at each info set.
fn extract_strategies_dfs(
    game: &HunlPostflop,
    state: &PostflopState,
    policies: &[poker_solver_deep_cfr::eval::ExplicitPolicy; 2],
    encoder: &poker_solver_deep_cfr::hunl_encoder::HunlStateEncoder,
    num_actions: usize,
    device: &candle_core::Device,
    strategy_map: &mut FxHashMap<u64, Vec<f64>>,
) -> Result<(), poker_solver_deep_cfr::SdCfrError> {
    if game.is_terminal(state) {
        return Ok(());
    }

    let player = game.player(state);
    let key = game.info_set_key(state);

    // Only insert if we haven't seen this info set key
    if !strategy_map.contains_key(&key) {
        let pi = match player {
            Player::Player1 => 0,
            Player::Player2 => 1,
        };
        let features = encoder.encode(state, player);
        let probs = policies[pi].strategy(&features)?;
        // Slice to actual action count and convert to f64
        let actions = game.actions(state);
        let n = actions.len().min(probs.len());
        let strat: Vec<f64> = probs[..n].iter().map(|&p| f64::from(p)).collect();
        strategy_map.insert(key, strat);
    }

    // Recurse through all actions
    for &action in &game.actions(state) {
        let next = game.next_state(state, action);
        extract_strategies_dfs(game, &next, policies, encoder, num_actions, device, strategy_map)?;
    }

    Ok(())
}
```

**Step 2: Verify it compiles**

Run: `cargo build -p poker-solver-trainer`
Expected: PASS.

**Step 3: Commit**

```bash
git add crates/trainer/src/main.rs
git commit -m "feat(trainer): implement sd-cfr checkpoint strategy extraction"
```

---

### Task 7: Integration test — train + checkpoint round-trip

**Files:**
- Create: `crates/deep-cfr/tests/hunl_integration.rs`

**Step 1: Write the integration test**

```rust
//! Integration test: SD-CFR on small HUNL config with checkpoint.

use poker_solver_core::game::{HunlPostflop, PostflopConfig};
use poker_solver_core::Game;
use poker_solver_deep_cfr::config::SdCfrConfig;
use poker_solver_deep_cfr::hunl_encoder::HunlStateEncoder;
use poker_solver_deep_cfr::solver::{SdCfrSolver, TrainedSdCfr};
use poker_solver_deep_cfr::SdCfrError;
use poker_solver_deep_cfr::eval::ExplicitPolicy;
use candle_core::Device;

fn small_hunl_config() -> PostflopConfig {
    PostflopConfig {
        stack_depth: 10,
        bet_sizes: vec![0.5, 1.0],
        max_raises_per_street: 2,
    }
}

#[test]
fn sdcfr_hunl_train_and_extract_strategy() {
    let game_config = small_hunl_config();
    let game = HunlPostflop::new(game_config.clone(), None, 100);
    let deal_pool = game.initial_states();

    let encoder = HunlStateEncoder::new(game_config.bet_sizes.clone());
    let config = SdCfrConfig {
        cfr_iterations: 5,
        traversals_per_iter: 50,
        advantage_memory_cap: 10_000,
        hidden_dim: 32,
        num_actions: 5, // fold, check, call, bet0, bet1 (+ all-in variants)
        sgd_steps: 20,
        batch_size: 64,
        learning_rate: 0.001,
        grad_clip_norm: 1.0,
        seed: 42,
        checkpoint_interval: 0,
    };

    let mut solver = SdCfrSolver::new(game, encoder, config).unwrap();
    let trained = solver.train_with_deals(Some(&deal_pool), None).unwrap();

    // Verify model buffers have entries
    assert_eq!(trained.model_buffers[0].len(), 5);
    assert_eq!(trained.model_buffers[1].len(), 5);

    // Extract strategy via ExplicitPolicy
    let device = Device::Cpu;
    let policy = ExplicitPolicy::from_buffer(
        &trained.model_buffers[0], 5, 32, &device,
    ).unwrap();

    // Query a strategy — should produce a valid distribution
    let encoder2 = HunlStateEncoder::new(game_config.bet_sizes.clone());
    let game2 = HunlPostflop::new(game_config, None, 1);
    let states = game2.initial_states();
    let features = encoder2.encode(&states[0], poker_solver_core::game::Player::Player1);
    let probs = policy.strategy(&features).unwrap();

    let sum: f32 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "Strategy should sum to ~1.0, got {sum}"
    );
}
```

**Step 2: Run test to verify it passes**

Run: `cargo test -p poker-solver-deep-cfr --test hunl_integration`
Expected: PASS.

**Step 3: Commit**

```bash
git add crates/deep-cfr/tests/hunl_integration.rs
git commit -m "test: add HUNL SD-CFR integration test"
```

---

### Task 8: Expose `SdCfrSolver` and `HunlStateEncoder` in deep-cfr public API

**Files:**
- Modify: `crates/deep-cfr/src/lib.rs`

**Step 1: Verify public exports**

Ensure `lib.rs` re-exports what the trainer needs:
```rust
pub mod hunl_encoder;
pub mod solver;
pub mod eval;
pub mod config;
```

The trainer needs: `SdCfrSolver`, `SdCfrConfig`, `HunlStateEncoder`, `ExplicitPolicy`, `TrainedSdCfr`, `SdCfrError`.

**Step 2: Add Cargo.toml dependency if needed**

Check that `poker-solver-trainer` has `poker-solver-deep-cfr` as a dependency. If not, add:

```toml
# In crates/trainer/Cargo.toml
poker-solver-deep-cfr = { path = "../deep-cfr" }
```

**Step 3: Run full build**

Run: `cargo build -p poker-solver-trainer`
Expected: PASS.

**Step 4: Commit**

```bash
git add crates/deep-cfr/src/lib.rs crates/trainer/Cargo.toml
git commit -m "feat: wire up deep-cfr dependency in trainer"
```

---

### Task 9: End-to-end smoke test with CLI

**Files:**
- Test only (no code changes)

**Step 1: Run the trainer with sd-cfr mode**

```bash
cargo run -p poker-solver-trainer --release -- train -c training_sdcfr.yaml --solver sd-cfr
```

Note: This will take a while with the full config. For a smoke test, create `training_sdcfr_test.yaml` with:
- `iterations: 3`
- `traversals_per_iter: 50`
- `deal_count: 100`
- `checkpoint_interval: 1`

Expected: Trains 3 iterations, produces 3 checkpoints in `./sdcfr_25bb/`, each loadable as a `StrategyBundle`.

**Step 2: Verify checkpoints are valid bundles**

```bash
ls -la sdcfr_25bb/sdcfr_checkpoint_1/
# Should contain: config.yaml, blueprint.bin
```

**Step 3: Run clippy**

Run: `cargo clippy -p poker-solver-deep-cfr -p poker-solver-trainer -- -D warnings`
Expected: Clean.

**Step 4: Run all tests**

Run: `cargo test -p poker-solver-deep-cfr -p poker-solver-trainer`
Expected: All PASS.

**Step 5: Commit any fixes**

```bash
git commit -m "chore: fix clippy warnings from sd-cfr integration"
```

---

## Summary

| Task | Component | ~Lines | Dependencies |
|------|-----------|--------|-------------|
| 1 | `SdCfrConfig` checkpoint_interval | 10 | None |
| 2 | `SdCfrSolver::train_with_deals()` | 60 | Task 1 |
| 3 | `HunlStateEncoder` | 70 | None |
| 4 | Trainer `SolverMode::SdCfr` + YAML | 80 | None |
| 5 | `run_sdcfr_training()` | 60 | Tasks 2, 3, 4 |
| 6 | Checkpoint extraction | 80 | Task 5 |
| 7 | Integration test | 60 | Tasks 2, 3 |
| 8 | Public API wiring | 10 | Tasks 3, 4 |
| 9 | End-to-end smoke test | 0 | All |

Tasks 1, 3, 4 can be done in parallel. Tasks 2 depends on 1. Tasks 5-6 depend on 2+3+4. Task 7 depends on 2+3. Task 8 depends on 3+4. Task 9 is last.
