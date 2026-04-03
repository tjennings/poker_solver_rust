use serde::{Deserialize, Serialize};

/// Bet size configuration supporting both flat and per-depth formats.
///
/// Flat: `["50%", "100%", "a"]` — one raise depth with these sizes.
/// Nested: `[["33%", "75%"], ["100%"], ["a"]]` — one entry per raise depth.
#[derive(Debug, Clone, Serialize)]
pub struct BetSizeConfig(pub Vec<Vec<String>>);

impl<'de> Deserialize<'de> for BetSizeConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Nested(Vec<Vec<String>>),
            Flat(Vec<String>),
        }
        match Raw::deserialize(deserializer)? {
            Raw::Nested(v) => Ok(BetSizeConfig(v)),
            Raw::Flat(v) => Ok(BetSizeConfig(vec![v])),
        }
    }
}

impl Default for BetSizeConfig {
    fn default() -> Self {
        BetSizeConfig(vec![vec!["25%".into(), "50%".into(), "100%".into(), "a".into()]])
    }
}

impl BetSizeConfig {
    /// Number of raise depths.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty() || self.0.iter().all(|d| d.is_empty())
    }

    pub fn depths(&self) -> &[Vec<String>] {
        &self.0
    }

    /// Flatten all depths into a single list (for river datagen which doesn't use per-depth).
    pub fn flat(&self) -> Vec<String> {
        self.0.iter().flat_map(|d| d.iter().cloned()).collect()
    }

    /// Join all sizes into a comma-separated string (for range-solver BetSizeOptions parsing).
    pub fn join_flat(&self, sep: &str) -> String {
        self.flat().join(sep)
    }
}

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
    /// Bet sizes per raise depth. Can be:
    /// - Flat: `["50%", "100%", "a"]` — same sizes for all depths, one depth only
    /// - Nested: `[["33%", "75%"], ["100%", "200%"], ["a"]]` — per-depth sizes
    /// Number of entries in nested form = max raises per round.
    pub bet_sizes: BetSizeConfig,
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
            bet_sizes: BetSizeConfig::default(),
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
        if self.bet_sizes.depths().is_empty() {
            return Err("bet_sizes must have at least one depth".into());
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
    /// Datagen mode: "model" uses river neural net at boundaries, "exact" solves to showdown.
    #[serde(default = "default_datagen_mode")]
    pub mode: String,
    #[serde(default = "default_pot_intervals")]
    pub pot_intervals: Vec<[i32; 2]>,
    #[serde(default)]
    pub spr_intervals: Option<Vec<[f64; 2]>>,
    #[serde(default = "default_solver_iterations")]
    pub solver_iterations: u32,
    /// Early-exit exploitability target (chips per pot). Omit or set null to disable.
    #[serde(default)]
    pub target_exploitability: Option<f32>,
    #[serde(default = "default_threads")]
    pub threads: usize,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Re-evaluate leaf boundaries every N iterations. 0 = every iteration (default).
    /// E.g. 1000 means evaluate at iteration 1, 1000, 2000, ... and the final iteration.
    #[serde(default)]
    pub leaf_eval_interval: u32,
    /// Number of games in the active pool for per-iteration boundary re-eval.
    /// Only used in model mode. Default 64.
    #[serde(default = "default_active_pool_size")]
    pub active_pool_size: usize,
    /// Per-deal bet size perturbation. Each bet size is multiplied by
    /// `1.0 + uniform(-fuzz, +fuzz)`. Default 0.0 (no fuzzing).
    #[serde(default)]
    pub bet_size_fuzz: f64,
    /// Output path for turn training data. CLI `-o` overrides this.
    #[serde(default)]
    pub turn_output: Option<String>,
    /// Output path for river training data (extracted from exact turn+river solves).
    /// When set in exact mode, river records are written alongside turn records.
    #[serde(default)]
    pub river_output: Option<String>,
    /// Max samples per output file. Splits into multiple files if total exceeds this.
    /// Default: no splitting (all samples in one file).
    #[serde(default)]
    pub per_file: Option<u64>,
    /// Path to blueprint bundle for realistic range generation.
    /// When set, datagen uses blueprint-propagated ranges instead of random RSP.
    #[serde(default)]
    pub blueprint_path: Option<String>,
}

impl Default for DatagenConfig {
    fn default() -> Self {
        Self {
            num_samples: 1000,
            street: default_street(),
            mode: default_datagen_mode(),
            pot_intervals: default_pot_intervals(),
            spr_intervals: None,
            solver_iterations: 1000,
            target_exploitability: None,
            threads: 8,
            seed: Some(42),
            leaf_eval_interval: 0,
            active_pool_size: default_active_pool_size(),
            bet_size_fuzz: 0.0,
            turn_output: None,
            river_output: None,
            per_file: None,
            blueprint_path: None,
        }
    }
}

fn default_datagen_mode() -> String {
    "model".into()
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
fn default_active_pool_size() -> usize {
    64
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
        // Flat config wraps as 1 depth with 4 sizes.
        assert_eq!(config.game.bet_sizes.len(), 1);
        assert_eq!(config.game.bet_sizes.depths()[0].len(), 4);
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
  target_exploitability: 0.005  # optional; omit for fixed iterations
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
            bet_sizes: BetSizeConfig(vec![]),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_stack() {
        let config = GameConfig {
            initial_stack: 0,
            bet_sizes: BetSizeConfig(vec![vec!["50%".into()]]),
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

    #[test]
    fn datagen_mode_defaults_to_model() {
        let config = DatagenConfig::default();
        assert_eq!(config.mode, "model");
    }

    #[test]
    fn datagen_bet_size_fuzz_defaults_to_zero() {
        let config = DatagenConfig::default();
        assert!((config.bet_size_fuzz - 0.0).abs() < 1e-12);
    }

    #[test]
    fn parse_config_with_exact_mode() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  num_samples: 100
  mode: "exact"
"#;
        let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.datagen.mode, "exact");
    }

    #[test]
    fn parse_config_with_bet_size_fuzz() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  num_samples: 100
  bet_size_fuzz: 0.10
"#;
        let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
        assert!((config.datagen.bet_size_fuzz - 0.10).abs() < 1e-9);
    }

    #[test]
    fn parse_config_without_mode_defaults_to_model() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  num_samples: 100
"#;
        let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.datagen.mode, "model");
    }

    #[test]
    fn river_output_defaults_to_none() {
        let config = DatagenConfig::default();
        assert!(config.river_output.is_none());
    }

    #[test]
    fn parse_config_with_river_output() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  num_samples: 100
  mode: "exact"
  river_output: "/tmp/river_data.bin"
"#;
        let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.datagen.river_output.as_deref(), Some("/tmp/river_data.bin"));
    }

    #[test]
    fn parse_config_without_river_output_is_none() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  num_samples: 100
  mode: "exact"
"#;
        let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.datagen.river_output.is_none());
    }

    #[test]
    fn blueprint_path_defaults_to_none() {
        let config = DatagenConfig::default();
        assert!(config.blueprint_path.is_none());
    }

    #[test]
    fn parse_config_with_blueprint_path() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  num_samples: 100
  blueprint_path: "/path/to/blueprint_bundle"
"#;
        let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.datagen.blueprint_path.as_deref(), Some("/path/to/blueprint_bundle"));
    }

    #[test]
    fn parse_config_without_blueprint_path_is_none() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  num_samples: 100
"#;
        let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.datagen.blueprint_path.is_none());
    }

    #[test]
    fn active_pool_size_defaults_to_64() {
        let config = DatagenConfig::default();
        assert_eq!(config.active_pool_size, 64);
    }

    #[test]
    fn parse_config_with_active_pool_size() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  num_samples: 100
  active_pool_size: 128
"#;
        let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.datagen.active_pool_size, 128);
    }

    #[test]
    fn parse_config_without_active_pool_size_uses_default() {
        let yaml = r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "a"]
datagen:
  num_samples: 100
"#;
        let config: CfvnetConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.datagen.active_pool_size, 64);
    }
}
