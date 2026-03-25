use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceConfig {
    pub game: GameDef,
    pub baseline: BaselineDef,
    pub mccfr: MccfrDef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameDef {
    pub flops: Vec<String>,
    pub starting_pot: i32,
    pub effective_stack: i32,
    pub bet_sizes: String,
    pub raise_sizes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineDef {
    pub max_iterations: u32,
    pub target_exploitability: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MccfrDef {
    pub iterations: u64,
    pub buckets: BucketsDef,
    pub checkpoints: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketsDef {
    pub preflop: u16,
    pub flop: u16,
    pub turn: u16,
    pub river: u16,
}

impl ConvergenceConfig {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&contents)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_convergence_config_from_yaml() {
        let yaml = r#"
game:
  flops: ["QhJdTh", "Ks7d2c"]
  starting_pot: 2
  effective_stack: 20
  bet_sizes: "50%,100%,a"
  raise_sizes: "50%,100%,a"
baseline:
  max_iterations: 1000
  target_exploitability: 0.001
mccfr:
  iterations: 1000000
  buckets:
    preflop: 169
    flop: 169
    turn: 200
    river: 200
  checkpoints: [1000, 10000]
"#;
        let config: ConvergenceConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.game.flops.len(), 2);
        assert_eq!(config.game.flops[0], "QhJdTh");
        assert_eq!(config.game.flops[1], "Ks7d2c");
        assert_eq!(config.game.starting_pot, 2);
        assert_eq!(config.game.effective_stack, 20);
        assert_eq!(config.game.bet_sizes, "50%,100%,a");
        assert_eq!(config.game.raise_sizes, "50%,100%,a");
        assert_eq!(config.baseline.max_iterations, 1000);
        assert!((config.baseline.target_exploitability - 0.001).abs() < 1e-6);
        assert_eq!(config.mccfr.iterations, 1_000_000);
        assert_eq!(config.mccfr.buckets.preflop, 169);
        assert_eq!(config.mccfr.buckets.flop, 169);
        assert_eq!(config.mccfr.buckets.turn, 200);
        assert_eq!(config.mccfr.buckets.river, 200);
        assert_eq!(config.mccfr.checkpoints, vec![1000, 10000]);
    }

    #[test]
    fn test_parse_single_flop_config() {
        let yaml = r#"
game:
  flops: ["8c8d3h"]
  starting_pot: 4
  effective_stack: 10
  bet_sizes: "a"
  raise_sizes: "a"
baseline:
  max_iterations: 500
  target_exploitability: 0.01
mccfr:
  iterations: 50000
  buckets:
    preflop: 169
    flop: 169
    turn: 50
    river: 50
  checkpoints: [5000, 50000]
"#;
        let config: ConvergenceConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.game.flops.len(), 1);
        assert_eq!(config.game.flops[0], "8c8d3h");
        assert_eq!(config.game.effective_stack, 10);
        assert_eq!(config.mccfr.buckets.turn, 50);
    }

    #[test]
    fn test_load_config_from_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test_config.yaml");
        std::fs::write(
            &path,
            r#"
game:
  flops: ["QhJdTh"]
  starting_pot: 2
  effective_stack: 20
  bet_sizes: "67%"
  raise_sizes: "67%"
baseline:
  max_iterations: 100
  target_exploitability: 0.01
mccfr:
  iterations: 10000
  buckets:
    preflop: 169
    flop: 169
    turn: 100
    river: 100
  checkpoints: [5000, 10000]
"#,
        )
        .unwrap();

        let config = ConvergenceConfig::load(&path).unwrap();
        assert_eq!(config.game.flops, vec!["QhJdTh"]);
        assert_eq!(config.mccfr.iterations, 10000);
    }

    #[test]
    fn test_load_config_missing_file_returns_error() {
        let result = ConvergenceConfig::load(Path::new("/nonexistent/config.yaml"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_config_invalid_yaml_returns_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("bad.yaml");
        std::fs::write(&path, "this is not valid yaml config: [[[").unwrap();
        let result = ConvergenceConfig::load(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_three_flops() {
        let yaml = r#"
game:
  flops: ["QhJdTh", "Ks7d2c", "8c8d3h"]
  starting_pot: 2
  effective_stack: 20
  bet_sizes: "50%,100%,a"
  raise_sizes: "50%,100%,a"
baseline:
  max_iterations: 1000
  target_exploitability: 0.001
mccfr:
  iterations: 1000000
  buckets:
    preflop: 169
    flop: 169
    turn: 200
    river: 200
  checkpoints: [1000, 10000, 100000, 500000, 1000000]
"#;
        let config: ConvergenceConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.game.flops.len(), 3);
        assert_eq!(config.mccfr.checkpoints.len(), 5);
    }
}
