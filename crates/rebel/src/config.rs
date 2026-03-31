// RebelConfig — YAML configuration

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
    #[serde(default)]
    pub inference: InferenceConfig,
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
    /// Add all-in if max bet / pot <= this threshold (range-solver TreeConfig).
    #[serde(default = "default_add_allin")]
    pub add_allin_threshold: f64,
    /// Force all-in if SPR after call <= this threshold (range-solver TreeConfig).
    #[serde(default = "default_force_allin")]
    pub force_allin_threshold: f64,
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

// --- Default functions ---

fn default_sb() -> i32 {
    1
}

fn default_bb() -> i32 {
    2
}

fn default_seed() -> u64 {
    42
}

fn default_threads() -> usize {
    16
}

fn default_solver_iters() -> u32 {
    1024
}

fn default_add_allin() -> f64 {
    1.5
}

fn default_force_allin() -> f64 {
    0.15
}

fn default_target_expl() -> f32 {
    0.005
}

fn default_layers() -> usize {
    7
}

fn default_hidden() -> usize {
    500
}

fn default_batch() -> usize {
    4096
}

fn default_epochs() -> usize {
    200
}

fn default_lr() -> f64 {
    3e-4
}

fn default_huber() -> f64 {
    1.0
}

fn default_max_records() -> usize {
    12_000_000
}

fn default_inf_batch() -> usize {
    256
}

fn default_inf_timeout() -> u64 {
    100
}

fn default_train_every() -> usize {
    50
}

fn default_train_batch() -> usize {
    512
}

fn default_replay_cap() -> usize {
    200_000
}

fn default_inf_lr() -> f64 {
    3e-4
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InferenceConfig {
    #[serde(default = "default_inf_batch")]
    pub batch_size: usize,
    #[serde(default = "default_inf_timeout")]
    pub batch_timeout_us: u64,
    #[serde(default = "default_train_every")]
    pub train_every_n_solves: usize,
    #[serde(default = "default_train_batch")]
    pub train_batch_size: usize,
    #[serde(default = "default_replay_cap")]
    pub replay_capacity: usize,
    #[serde(default = "default_inf_lr")]
    pub learning_rate: f64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            batch_size: default_inf_batch(),
            batch_timeout_us: default_inf_timeout(),
            train_every_n_solves: default_train_every(),
            train_batch_size: default_train_batch(),
            replay_capacity: default_replay_cap(),
            learning_rate: default_inf_lr(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL_CONFIG_YAML: &str = r#"
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

    #[test]
    fn test_parse_full_config() {
        let config: RebelConfig = serde_yaml::from_str(FULL_CONFIG_YAML).unwrap();

        assert_eq!(config.blueprint_path, "/data/blueprint");
        assert_eq!(config.cluster_dir, "/data/clusters");
        assert_eq!(config.output_dir, "/data/rebel");

        assert_eq!(config.game.initial_stack, 400);
        assert_eq!(config.game.small_blind, 1);
        assert_eq!(config.game.big_blind, 2);

        assert_eq!(config.seed.num_hands, 1_000_000);
        assert_eq!(config.seed.seed, 42);
        assert_eq!(config.seed.threads, 16);
        assert_eq!(config.seed.solver_iterations, 1024);
        assert!((config.seed.target_exploitability - 0.005).abs() < 1e-6);

        // Check bet sizes
        assert_eq!(config.seed.bet_sizes.flop[0], vec![0.33, 0.67, 1.0]);
        assert_eq!(config.seed.bet_sizes.flop[1], vec![0.33, 0.67, 1.0]);
        assert_eq!(config.seed.bet_sizes.turn[0], vec![0.5, 0.75, 1.0]);
        assert_eq!(config.seed.bet_sizes.river[1], vec![0.5, 0.75, 1.0]);

        assert_eq!(config.training.hidden_layers, 7);
        assert_eq!(config.training.hidden_size, 500);
        assert_eq!(config.training.batch_size, 4096);
        assert_eq!(config.training.epochs, 200);
        assert!((config.training.learning_rate - 0.0003).abs() < 1e-9);
        assert!((config.training.huber_delta - 1.0).abs() < 1e-9);

        assert_eq!(config.buffer.max_records, 12_000_000);
        assert_eq!(config.buffer.path, "rebel_buffer.bin");

        // Inference defaults (not specified in YAML)
        assert_eq!(config.inference.batch_size, 256);
        assert_eq!(config.inference.batch_timeout_us, 100);
        assert_eq!(config.inference.train_every_n_solves, 50);
        assert_eq!(config.inference.train_batch_size, 512);
        assert_eq!(config.inference.replay_capacity, 200_000);
        assert!((config.inference.learning_rate - 3e-4).abs() < 1e-9);
    }

    #[test]
    fn test_parse_with_defaults() {
        let minimal_yaml = r#"
blueprint_path: "/data/blueprint"
cluster_dir: "/data/clusters"
output_dir: "/data/rebel"
game:
  initial_stack: 400
seed:
  num_hands: 500000
  bet_sizes:
    flop: [[0.5, 1.0], [0.5, 1.0]]
    turn: [[0.75], [0.75]]
    river: [[1.0], [1.0]]
training: {}
buffer:
  path: "buf.bin"
"#;
        let config: RebelConfig = serde_yaml::from_str(minimal_yaml).unwrap();

        // Game defaults
        assert_eq!(config.game.small_blind, 1);
        assert_eq!(config.game.big_blind, 2);

        // Seed defaults
        assert_eq!(config.seed.num_hands, 500_000);
        assert_eq!(config.seed.seed, 42);
        assert_eq!(config.seed.threads, 16);
        assert_eq!(config.seed.solver_iterations, 1024);
        assert!((config.seed.target_exploitability - 0.005).abs() < 1e-6);

        // Training defaults
        assert_eq!(config.training.hidden_layers, 7);
        assert_eq!(config.training.hidden_size, 500);
        assert_eq!(config.training.batch_size, 4096);
        assert_eq!(config.training.epochs, 200);
        assert!((config.training.learning_rate - 3e-4).abs() < 1e-9);
        assert!((config.training.huber_delta - 1.0).abs() < 1e-9);

        // Buffer defaults
        assert_eq!(config.buffer.max_records, 12_000_000);

        // Inference defaults
        assert_eq!(config.inference.batch_size, 256);
        assert_eq!(config.inference.batch_timeout_us, 100);
        assert_eq!(config.inference.train_every_n_solves, 50);
        assert_eq!(config.inference.train_batch_size, 512);
        assert_eq!(config.inference.replay_capacity, 200_000);
        assert!((config.inference.learning_rate - 3e-4).abs() < 1e-9);
    }

    #[test]
    fn test_parse_explicit_inference_config() {
        let yaml = r#"
blueprint_path: "/data/blueprint"
cluster_dir: "/data/clusters"
output_dir: "/data/rebel"
game:
  initial_stack: 400
seed:
  num_hands: 500000
  bet_sizes:
    flop: [[0.5, 1.0], [0.5, 1.0]]
    turn: [[0.75], [0.75]]
    river: [[1.0], [1.0]]
training: {}
buffer:
  path: "buf.bin"
inference:
  batch_size: 128
  batch_timeout_us: 200
  train_every_n_solves: 100
  train_batch_size: 1024
  replay_capacity: 500000
  learning_rate: 0.001
"#;
        let config: RebelConfig = serde_yaml::from_str(yaml).unwrap();

        assert_eq!(config.inference.batch_size, 128);
        assert_eq!(config.inference.batch_timeout_us, 200);
        assert_eq!(config.inference.train_every_n_solves, 100);
        assert_eq!(config.inference.train_batch_size, 1024);
        assert_eq!(config.inference.replay_capacity, 500_000);
        assert!((config.inference.learning_rate - 0.001).abs() < 1e-9);
    }

    #[test]
    fn test_roundtrip() {
        let config: RebelConfig = serde_yaml::from_str(FULL_CONFIG_YAML).unwrap();
        let serialized = serde_yaml::to_string(&config).unwrap();
        let roundtripped: RebelConfig = serde_yaml::from_str(&serialized).unwrap();

        // Verify key fields survive roundtrip
        assert_eq!(config.blueprint_path, roundtripped.blueprint_path);
        assert_eq!(config.cluster_dir, roundtripped.cluster_dir);
        assert_eq!(config.output_dir, roundtripped.output_dir);
        assert_eq!(config.game.initial_stack, roundtripped.game.initial_stack);
        assert_eq!(config.game.small_blind, roundtripped.game.small_blind);
        assert_eq!(config.game.big_blind, roundtripped.game.big_blind);
        assert_eq!(config.seed.num_hands, roundtripped.seed.num_hands);
        assert_eq!(config.seed.seed, roundtripped.seed.seed);
        assert_eq!(config.seed.threads, roundtripped.seed.threads);
        assert_eq!(config.seed.solver_iterations, roundtripped.seed.solver_iterations);
        assert!((config.seed.target_exploitability - roundtripped.seed.target_exploitability).abs() < 1e-6);
        assert_eq!(config.seed.bet_sizes.flop, roundtripped.seed.bet_sizes.flop);
        assert_eq!(config.seed.bet_sizes.turn, roundtripped.seed.bet_sizes.turn);
        assert_eq!(config.seed.bet_sizes.river, roundtripped.seed.bet_sizes.river);
        assert_eq!(config.training.hidden_layers, roundtripped.training.hidden_layers);
        assert_eq!(config.training.hidden_size, roundtripped.training.hidden_size);
        assert_eq!(config.training.batch_size, roundtripped.training.batch_size);
        assert_eq!(config.training.epochs, roundtripped.training.epochs);
        assert!((config.training.learning_rate - roundtripped.training.learning_rate).abs() < 1e-9);
        assert!((config.training.huber_delta - roundtripped.training.huber_delta).abs() < 1e-9);
        assert_eq!(config.buffer.max_records, roundtripped.buffer.max_records);
        assert_eq!(config.buffer.path, roundtripped.buffer.path);
        assert_eq!(config.inference.batch_size, roundtripped.inference.batch_size);
        assert_eq!(config.inference.batch_timeout_us, roundtripped.inference.batch_timeout_us);
        assert_eq!(config.inference.train_every_n_solves, roundtripped.inference.train_every_n_solves);
        assert_eq!(config.inference.train_batch_size, roundtripped.inference.train_batch_size);
        assert_eq!(config.inference.replay_capacity, roundtripped.inference.replay_capacity);
        assert!((config.inference.learning_rate - roundtripped.inference.learning_rate).abs() < 1e-9);
    }
}
