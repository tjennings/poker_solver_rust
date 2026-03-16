//! YAML-driven config for training the full CFVNet model stack.
//!
//! Parses a single YAML file into [`GpuTrainingStackConfig`] and provides
//! [`train_full_stack`] which orchestrates river → turn → flop → preflop
//! training in sequence.

use serde::Deserialize;

#[cfg(feature = "training")]
use std::path::{Path, PathBuf};

#[cfg(feature = "training")]
use burn::tensor::backend::AutodiffBackend;

#[cfg(feature = "training")]
use crate::training::pipeline::{train_river_cfvnet, RiverTrainingConfig};
#[cfg(feature = "training")]
use crate::training::turn_pipeline::{train_turn_cfvnet_cuda, TurnTrainingConfig};
#[cfg(feature = "training")]
use crate::training::flop_pipeline::{train_flop_cfvnet_cuda, FlopTrainingConfig};
#[cfg(feature = "training")]
use crate::training::preflop_pipeline::{train_preflop_cfvnet_cuda, PreflopTrainingConfig};

/// Top-level stack training configuration parsed from YAML.
#[derive(Debug, Deserialize)]
pub struct GpuTrainingStackConfig {
    pub game: GameConfig,
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub river: StreetConfig,
    pub turn: StreetConfig,
    pub flop: StreetConfig,
    pub preflop: StreetConfig,
}

/// Shared game parameters.
#[derive(Debug, Deserialize)]
pub struct GameConfig {
    pub initial_stack: i32,
    pub bet_sizes: Vec<String>,
    pub ref_pot: i32,
    pub ref_stack: i32,
}

/// Shared model architecture parameters.
#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub hidden_layers: usize,
    pub hidden_size: usize,
}

/// Shared training hyperparameters.
#[derive(Debug, Deserialize)]
pub struct TrainingConfig {
    pub train_batch_size: usize,
    pub train_steps_per_batch: usize,
    pub learning_rate: f64,
    pub huber_delta: f64,
    pub aux_loss_weight: f64,
    pub seed: u64,
}

fn default_solve_iters() -> u32 {
    4000
}
fn default_batch_size() -> usize {
    1000
}
fn default_validation() -> u64 {
    10_000
}
fn default_checkpoint() -> u64 {
    1_000_000
}

/// Per-street overrides for training parameters.
#[derive(Debug, Deserialize)]
pub struct StreetConfig {
    pub num_samples: u64,
    #[serde(default = "default_solve_iters")]
    pub solve_iterations: u32,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    pub reservoir_capacity: usize,
    #[serde(default = "default_validation")]
    pub validation_interval: u64,
    #[serde(default = "default_checkpoint")]
    pub checkpoint_interval: u64,
    /// Override shared train_batch_size for this street.
    pub train_batch_size: Option<usize>,
    /// Override shared train_steps_per_batch for this street.
    pub train_steps_per_batch: Option<usize>,
    /// Override shared learning_rate for this street.
    pub learning_rate: Option<f64>,
}

// ---------------------------------------------------------------------------
// Stack training pipeline
// ---------------------------------------------------------------------------

/// Train the full CFVNet model stack: river → turn → flop → preflop.
///
/// Each phase writes its model into a subdirectory of `output_dir`, and
/// subsequent phases load the previous model as their leaf evaluator.
#[cfg(feature = "training")]
pub fn train_full_stack<B: AutodiffBackend>(
    config: &GpuTrainingStackConfig,
    output_dir: &Path,
    device: &B::Device,
) -> Result<(), String> {
    let river_dir = output_dir.join("river");
    let turn_dir = output_dir.join("turn");
    let flop_dir = output_dir.join("flop");
    let preflop_dir = output_dir.join("preflop");

    // 1. Train river
    eprintln!("=== PHASE 1/4: RIVER ===");
    let river_config = build_river_config(config, &river_dir);
    train_river_cfvnet::<B>(&river_config, device)?;

    // River model path is output_dir/river/model
    let river_model_path = river_dir.join("model");

    // 2. Train turn (uses river model)
    eprintln!("=== PHASE 2/4: TURN ===");
    let turn_config = build_turn_config(config, &river_model_path, &turn_dir);
    train_turn_cfvnet_cuda::<B>(&turn_config, device)?;

    // Turn model path is output_dir/turn/model
    let turn_model_path = turn_dir.join("model");

    // 3. Train flop (uses turn model)
    eprintln!("=== PHASE 3/4: FLOP ===");
    let flop_config = build_flop_config(config, &turn_model_path, &flop_dir);
    train_flop_cfvnet_cuda::<B>(&flop_config, device)?;

    // Flop model path is output_dir/flop/model
    let flop_model_path = flop_dir.join("model");

    // 4. Train preflop (uses flop model)
    eprintln!("=== PHASE 4/4: PREFLOP ===");
    let preflop_config = build_preflop_config(config, &flop_model_path, &preflop_dir);
    train_preflop_cfvnet_cuda::<B>(&preflop_config, device)?;

    eprintln!("=== ALL 4 MODELS TRAINED ===");
    eprintln!("Output: {}", output_dir.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Config builders
// ---------------------------------------------------------------------------

#[cfg(feature = "training")]
fn build_river_config(config: &GpuTrainingStackConfig, output_dir: &Path) -> RiverTrainingConfig {
    let street = &config.river;
    RiverTrainingConfig {
        num_samples: street.num_samples,
        solve_iterations: street.solve_iterations,
        batch_size: street.batch_size,
        reservoir_capacity: street.reservoir_capacity,
        hidden_layers: config.model.hidden_layers,
        hidden_size: config.model.hidden_size,
        train_batch_size: street
            .train_batch_size
            .unwrap_or(config.training.train_batch_size),
        train_steps_per_batch: street
            .train_steps_per_batch
            .unwrap_or(config.training.train_steps_per_batch),
        learning_rate: street
            .learning_rate
            .unwrap_or(config.training.learning_rate),
        huber_delta: config.training.huber_delta,
        aux_loss_weight: config.training.aux_loss_weight,
        validation_interval: street.validation_interval,
        checkpoint_interval: street.checkpoint_interval,
        gt_validation_positions: 0,
        gt_solve_iterations: 0,
        output_dir: output_dir.to_path_buf(),
        seed: config.training.seed,
        ref_pot: config.game.ref_pot,
        ref_stack: config.game.ref_stack,
        use_persistent_kernel: false,
    }
}

#[cfg(feature = "training")]
fn build_turn_config(
    config: &GpuTrainingStackConfig,
    river_model_path: &Path,
    output_dir: &Path,
) -> TurnTrainingConfig {
    let street = &config.turn;
    TurnTrainingConfig {
        river_model_path: river_model_path.to_path_buf(),
        river_hidden_layers: config.model.hidden_layers,
        river_hidden_size: config.model.hidden_size,
        num_samples: street.num_samples,
        solve_iterations: street.solve_iterations,
        batch_size: street.batch_size,
        reservoir_capacity: street.reservoir_capacity,
        hidden_layers: config.model.hidden_layers,
        hidden_size: config.model.hidden_size,
        train_batch_size: street
            .train_batch_size
            .unwrap_or(config.training.train_batch_size),
        train_steps_per_batch: street
            .train_steps_per_batch
            .unwrap_or(config.training.train_steps_per_batch),
        learning_rate: street
            .learning_rate
            .unwrap_or(config.training.learning_rate),
        huber_delta: config.training.huber_delta,
        aux_loss_weight: config.training.aux_loss_weight,
        validation_interval: street.validation_interval,
        checkpoint_interval: street.checkpoint_interval,
        gt_validation_positions: 0,
        gt_solve_iterations: 0,
        output_dir: output_dir.to_path_buf(),
        seed: config.training.seed,
        ref_pot: config.game.ref_pot,
        ref_stack: config.game.ref_stack,
    }
}

#[cfg(feature = "training")]
fn build_flop_config(
    config: &GpuTrainingStackConfig,
    turn_model_path: &Path,
    output_dir: &Path,
) -> FlopTrainingConfig {
    let street = &config.flop;
    FlopTrainingConfig {
        turn_model_path: turn_model_path.to_path_buf(),
        turn_hidden_layers: config.model.hidden_layers,
        turn_hidden_size: config.model.hidden_size,
        num_samples: street.num_samples,
        solve_iterations: street.solve_iterations,
        batch_size: street.batch_size,
        reservoir_capacity: street.reservoir_capacity,
        hidden_layers: config.model.hidden_layers,
        hidden_size: config.model.hidden_size,
        train_batch_size: street
            .train_batch_size
            .unwrap_or(config.training.train_batch_size),
        train_steps_per_batch: street
            .train_steps_per_batch
            .unwrap_or(config.training.train_steps_per_batch),
        learning_rate: street
            .learning_rate
            .unwrap_or(config.training.learning_rate),
        huber_delta: config.training.huber_delta,
        aux_loss_weight: config.training.aux_loss_weight,
        validation_interval: street.validation_interval,
        checkpoint_interval: street.checkpoint_interval,
        output_dir: output_dir.to_path_buf(),
        seed: config.training.seed,
        ref_pot: config.game.ref_pot,
        ref_stack: config.game.ref_stack,
    }
}

#[cfg(feature = "training")]
fn build_preflop_config(
    config: &GpuTrainingStackConfig,
    flop_model_path: &Path,
    output_dir: &Path,
) -> PreflopTrainingConfig {
    let street = &config.preflop;
    PreflopTrainingConfig {
        flop_model_path: flop_model_path.to_path_buf(),
        flop_hidden_layers: config.model.hidden_layers,
        flop_hidden_size: config.model.hidden_size,
        num_samples: street.num_samples,
        reservoir_capacity: street.reservoir_capacity,
        hidden_layers: config.model.hidden_layers,
        hidden_size: config.model.hidden_size,
        train_batch_size: street
            .train_batch_size
            .unwrap_or(config.training.train_batch_size),
        train_steps_per_batch: street
            .train_steps_per_batch
            .unwrap_or(config.training.train_steps_per_batch),
        batch_size: street.batch_size,
        learning_rate: street
            .learning_rate
            .unwrap_or(config.training.learning_rate),
        huber_delta: config.training.huber_delta,
        aux_loss_weight: config.training.aux_loss_weight,
        validation_interval: street.validation_interval,
        checkpoint_interval: street.checkpoint_interval,
        output_dir: output_dir.to_path_buf(),
        seed: config.training.seed,
        ref_pot: config.game.ref_pot,
        ref_stack: config.game.ref_stack,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_stack_config() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "100%", "a"]
  ref_pot: 100
  ref_stack: 100

model:
  hidden_layers: 4
  hidden_size: 256

training:
  train_batch_size: 512
  train_steps_per_batch: 5
  learning_rate: 0.001
  huber_delta: 1.0
  aux_loss_weight: 1.0
  seed: 42

river:
  num_samples: 1000000
  solve_iterations: 2000
  batch_size: 500
  reservoir_capacity: 50000

turn:
  num_samples: 500000
  solve_iterations: 2000
  batch_size: 500
  reservoir_capacity: 50000

flop:
  num_samples: 500000
  solve_iterations: 2000
  batch_size: 200
  reservoir_capacity: 50000

preflop:
  num_samples: 200000
  batch_size: 1
  reservoir_capacity: 50000
"#;
        let config: GpuTrainingStackConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.game.ref_pot, 100);
        assert_eq!(config.model.hidden_layers, 4);
        assert_eq!(config.model.hidden_size, 256);
        assert_eq!(config.training.seed, 42);
        assert_eq!(config.river.num_samples, 1_000_000);
        assert_eq!(config.river.solve_iterations, 2000);
        assert_eq!(config.turn.batch_size, 500);
        assert_eq!(config.flop.reservoir_capacity, 50000);
        assert_eq!(config.preflop.num_samples, 200_000);
        // Defaults
        assert_eq!(config.preflop.solve_iterations, 4000); // default
        assert_eq!(config.preflop.validation_interval, 10_000); // default
        assert_eq!(config.preflop.checkpoint_interval, 1_000_000); // default
    }

    #[test]
    fn test_street_overrides() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%"]
  ref_pot: 100
  ref_stack: 100

model:
  hidden_layers: 7
  hidden_size: 500

training:
  train_batch_size: 1024
  train_steps_per_batch: 10
  learning_rate: 0.001
  huber_delta: 1.0
  aux_loss_weight: 1.0
  seed: 42

river:
  num_samples: 100
  reservoir_capacity: 100
  train_batch_size: 2048
  learning_rate: 0.0005

turn:
  num_samples: 100
  reservoir_capacity: 100

flop:
  num_samples: 100
  reservoir_capacity: 100

preflop:
  num_samples: 100
  reservoir_capacity: 100
"#;
        let config: GpuTrainingStackConfig = serde_yaml::from_str(yaml).unwrap();
        // River overrides
        assert_eq!(config.river.train_batch_size, Some(2048));
        assert_eq!(config.river.learning_rate, Some(0.0005));
        // Turn uses shared defaults
        assert_eq!(config.turn.train_batch_size, None);
        assert_eq!(config.turn.learning_rate, None);
    }
}
