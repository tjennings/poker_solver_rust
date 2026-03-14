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
    #[serde(default = "default_board_size")]
    pub board_size: usize,
    #[serde(default = "default_allin_threshold")]
    pub add_allin_threshold: f64,
    #[serde(default = "default_force_allin_threshold")]
    pub force_allin_threshold: f64,
    /// Path to a trained river CFV model, used as a leaf evaluator for turn solving.
    #[serde(default)]
    pub river_model_path: Option<String>,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            initial_stack: 200,
            bet_sizes: vec!["25%".into(), "50%".into(), "100%".into(), "a".into()],
            board_size: 5,
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            river_model_path: None,
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
        if self.board_size != 4 && self.board_size != 5 {
            return Err(format!("board_size must be 4 or 5, got {}", self.board_size));
        }
        Ok(())
    }
}

fn default_board_size() -> usize {
    5
}
fn default_allin_threshold() -> f64 {
    1.5
}
fn default_force_allin_threshold() -> f64 {
    0.15
}

/// Return the number of board cards for a given street name.
///
/// Panics if `street` is not one of `"river"`, `"turn"`, or `"flop"`.
pub fn board_cards_for_street(street: &str) -> usize {
    match street {
        "river" => 5,
        "turn" => 4,
        "flop" => 3,
        other => panic!("unknown street: {other:?} (expected river, turn, or flop)"),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatagenConfig {
    pub num_samples: u64,
    /// Which street to generate data for: "river" (default), "turn", or "flop".
    #[serde(default = "default_street")]
    pub street: String,
    #[serde(default = "default_pot_intervals")]
    pub pot_intervals: Vec<[i32; 2]>,
    #[serde(default)]
    pub spr_intervals: Option<Vec<[f64; 2]>>,
    #[serde(default = "default_solver_iterations")]
    pub solver_iterations: u32,
    #[serde(default = "default_target_exploitability")]
    pub target_exploitability: f32,
    #[serde(default = "default_threads")]
    pub threads: usize,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Re-evaluate leaf boundaries every N iterations. 0 = every iteration (default).
    /// E.g. 1000 means evaluate at iteration 1, 1000, 2000, ... and the final iteration.
    #[serde(default)]
    pub leaf_eval_interval: u32,
}

impl Default for DatagenConfig {
    fn default() -> Self {
        Self {
            num_samples: 1000,
            street: default_street(),
            pot_intervals: default_pot_intervals(),
            spr_intervals: None,
            solver_iterations: 1000,
            target_exploitability: 0.005,
            threads: 8,
            seed: Some(42),
            leaf_eval_interval: 0,
        }
    }
}

fn default_street() -> String {
    "river".into()
}
fn default_pot_intervals() -> Vec<[i32; 2]> {
    vec![[4, 20], [20, 80], [80, 200], [200, 400]]
}
fn default_solver_iterations() -> u32 {
    1000
}
fn default_target_exploitability() -> f32 {
    0.005
}
fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(8)
}
/// Resolve seed: use provided value or generate a random one.
pub fn resolve_seed(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(|| rand::random())
}

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
    pub checkpoint_every_n_epochs: usize,
    #[serde(default = "default_shuffle_buffer_size")]
    pub shuffle_buffer_size: usize,
    #[serde(default = "default_prefetch_depth")]
    pub prefetch_depth: usize,
    #[serde(default = "default_encoder_threads")]
    pub encoder_threads: usize,
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
            checkpoint_every_n_epochs: 1000,
            shuffle_buffer_size: 262_144,
            prefetch_depth: 4,
            encoder_threads: default_encoder_threads(),
        }
    }
}

fn default_hidden_layers() -> usize {
    7
}
fn default_hidden_size() -> usize {
    500
}
fn default_batch_size() -> usize {
    2048
}
fn default_epochs() -> usize {
    2
}
fn default_learning_rate() -> f64 {
    0.001
}
fn default_lr_min() -> f64 {
    0.00001
}
fn default_huber_delta() -> f64 {
    1.0
}
fn default_aux_loss_weight() -> f64 {
    1.0
}
fn default_validation_split() -> f64 {
    0.05
}
fn default_checkpoint_interval() -> usize {
    1000
}
fn default_shuffle_buffer_size() -> usize {
    262_144
}
fn default_prefetch_depth() -> usize {
    4
}
fn default_encoder_threads() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4)
        .saturating_sub(2) // reserve 1 for reader, 1 for training loop
        .max(1)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvaluationConfig {
    #[serde(default)]
    pub regression_spots: Option<String>,
}

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
        assert_eq!(config.datagen.seed, None);
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
  checkpoint_every_n_epochs: 1000
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

    #[test]
    fn board_cards_for_known_streets() {
        assert_eq!(board_cards_for_street("river"), 5);
        assert_eq!(board_cards_for_street("turn"), 4);
        assert_eq!(board_cards_for_street("flop"), 3);
    }

    #[test]
    #[should_panic(expected = "unknown street")]
    fn board_cards_for_unknown_street_panics() {
        board_cards_for_street("preflop");
    }

    #[test]
    fn datagen_street_defaults_to_river() {
        let config = DatagenConfig::default();
        assert_eq!(config.street, "river");
    }

    #[test]
    fn parse_config_with_street() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  num_samples: 100
  street: "turn"
"#;
        let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.datagen.street, "turn");
    }

    #[test]
    fn parse_config_with_spr_intervals() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  num_samples: 100
  spr_intervals: [[0.0, 0.5], [0.5, 1.5], [1.5, 4.0]]
"#;
        let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
        let spr = config.datagen.spr_intervals.unwrap();
        assert_eq!(spr.len(), 3);
        assert!((spr[0][0] - 0.0).abs() < 1e-9);
        assert!((spr[0][1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn parse_config_without_spr_intervals_is_none() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  num_samples: 100
"#;
        let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.datagen.spr_intervals.is_none());
    }

    #[test]
    fn parse_config_with_river_model_path() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
  river_model_path: "/path/to/river_model"
datagen:
  num_samples: 100
"#;
        let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.game.river_model_path.as_deref(), Some("/path/to/river_model"));
    }

    #[test]
    fn parse_config_with_streaming_params() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  num_samples: 100
training:
  shuffle_buffer_size: 131072
  prefetch_depth: 8
"#;
        let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.training.shuffle_buffer_size, 131_072);
        assert_eq!(config.training.prefetch_depth, 8);
    }

    #[test]
    fn config_defaults_for_streaming() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  num_samples: 100
"#;
        let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.training.shuffle_buffer_size, 262_144);
        assert_eq!(config.training.prefetch_depth, 4);
    }
}
