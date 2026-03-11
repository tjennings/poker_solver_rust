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

fn default_allin_threshold() -> f64 {
    1.5
}
fn default_force_allin_threshold() -> f64 {
    0.15
}

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
fn default_solver_iterations() -> u32 {
    1000
}
fn default_target_exploitability() -> f32 {
    0.005
}
fn default_threads() -> usize {
    8
}
fn default_seed() -> u64 {
    42
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    #[serde(default)]
    pub regression_spots: Option<String>,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            regression_spots: None,
        }
    }
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
