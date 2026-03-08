use serde::{Deserialize, Serialize};

/// Top-level config for the full Blueprint V2 pipeline.
///
/// Covers game parameters, card abstraction clustering, action abstraction,
/// MCCFR training schedule, and snapshot output settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlueprintV2Config {
    pub game: GameConfig,
    pub clustering: ClusteringConfig,
    pub action_abstraction: ActionAbstractionConfig,
    pub training: TrainingConfig,
    pub snapshots: SnapshotConfig,
}

/// Core game parameters (stakes and stack depth).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameConfig {
    /// Human-readable name for this blueprint (shown in UI).
    pub name: String,
    pub players: u8,
    /// Stack depth in big blinds.
    pub stack_depth: f64,
    /// Small blind size in big blinds (typically 0.5).
    pub small_blind: f64,
    /// Big blind size in big blinds (typically 1.0).
    pub big_blind: f64,
    /// Rake as a fraction of the pot (0.0 = no rake, 0.05 = 5%).
    #[serde(default)]
    pub rake_rate: f64,
    /// Maximum rake in chips (min bet units). 0.0 = no cap.
    #[serde(default)]
    pub rake_cap: f64,
}

/// Card abstraction clustering configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringConfig {
    pub algorithm: ClusteringAlgorithm,
    pub preflop: StreetClusterConfig,
    pub flop: StreetClusterConfig,
    pub turn: StreetClusterConfig,
    pub river: StreetClusterConfig,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_kmeans_iterations")]
    pub kmeans_iterations: u32,
}

/// Supported clustering algorithms for card abstraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusteringAlgorithm {
    PotentialAwareEmd,
}

/// Per-street bucket count for clustering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreetClusterConfig {
    pub buckets: u16,
}

/// Action abstraction: allowed bet sizes per street and raise cap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionAbstractionConfig {
    /// Preflop sizes as string labels (e.g. `"2.5bb"`, `"3.0x"`).
    /// Outer vec is per raise depth, inner vec is the set of sizes at that depth.
    pub preflop: Vec<Vec<String>>,
    /// Flop bet sizes as pot fractions, indexed by raise depth.
    pub flop: Vec<Vec<f64>>,
    /// Turn bet sizes as pot fractions, indexed by raise depth.
    pub turn: Vec<Vec<f64>>,
    /// River bet sizes as pot fractions, indexed by raise depth.
    pub river: Vec<Vec<f64>>,
}

/// MCCFR training schedule and parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Path to pre-computed clustering data.
    pub cluster_path: String,
    /// Stop after this many iterations (if set).
    #[serde(default)]
    pub iterations: Option<u64>,
    /// Stop after this many minutes (if set).
    #[serde(default)]
    pub time_limit_minutes: Option<u64>,
    /// Iterations of LCFR warmup before switching to linear averaging.
    #[serde(default = "default_lcfr_warmup", alias = "lcfr_warmup_minutes")]
    pub lcfr_warmup_iterations: u64,
    /// Iterations between LCFR discount applications.
    #[serde(default = "default_discount_interval")]
    pub lcfr_discount_interval: u64,
    /// Start pruning negative-regret actions after this many iterations.
    #[serde(default = "default_prune_after", alias = "prune_after_minutes")]
    pub prune_after_iterations: u64,
    /// Regret threshold below which actions are pruned (in BB units;
    /// internally scaled by ×1000 to match stored regret precision).
    #[serde(default = "default_prune_threshold")]
    pub prune_threshold: i32,
    /// Probability of exploring a pruned action anyway.
    #[serde(default = "default_prune_explore")]
    pub prune_explore_pct: f64,
    /// Minutes between progress log lines.
    #[serde(default = "default_print_every")]
    pub print_every_minutes: u64,
    /// Number of iterations per parallel batch.
    #[serde(default = "default_batch_size")]
    pub batch_size: u64,
    /// Stop when mean strategy delta falls below this threshold.
    /// Checked every `print_every_minutes`. `None` = never stop on delta.
    #[serde(default)]
    pub target_strategy_delta: Option<f64>,
    /// Zero action probabilities below this threshold when exporting the
    /// strategy to a bundle (e.g. 0.03 = drop actions under 3%).
    /// Remaining probabilities are renormalized. Default 0.0 (disabled).
    #[serde(default)]
    pub purify_threshold: f64,
}

/// Snapshot (checkpoint) output settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotConfig {
    /// Minutes of training before the first snapshot.
    pub warmup_minutes: u64,
    /// Minutes between successive snapshots.
    pub snapshot_every_minutes: u64,
    /// Directory to write snapshot files.
    pub output_dir: String,
    /// If true, scan `output_dir` for the latest snapshot and resume training
    /// from its regrets. The snapshot with the highest number is used.
    #[serde(default)]
    pub resume: bool,
    /// Maximum number of snapshot directories to keep. When exceeded, the
    /// oldest snapshots are deleted after each new save. `None` = unlimited.
    #[serde(default)]
    pub max_snapshots: Option<u32>,
}

// ── Default value functions ──────────────────────────────────────────

const fn default_seed() -> u64 {
    42
}

const fn default_kmeans_iterations() -> u32 {
    100
}

const fn default_lcfr_warmup() -> u64 {
    5_000_000
}

const fn default_discount_interval() -> u64 {
    500_000
}

const fn default_prune_after() -> u64 {
    5_000_000
}

const fn default_prune_threshold() -> i32 {
    -250
}

const fn default_prune_explore() -> f64 {
    0.05
}

const fn default_print_every() -> u64 {
    10
}

const fn default_batch_size() -> u64 {
    200
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_toy_config() {
        let yaml = r#"
game:
  name: "Test Config"
  players: 6
  stack_depth: 100.0
  small_blind: 0.5
  big_blind: 1.0

clustering:
  algorithm: potential_aware_emd
  preflop:
    buckets: 169
  flop:
    buckets: 200
  turn:
    buckets: 200
  river:
    buckets: 200

action_abstraction:
  preflop:
    - ["2.5bb"]
    - ["3.0x"]
  flop:
    - [0.33, 0.67, 1.0]
    - [0.5, 1.0]
  turn:
    - [0.5, 1.0]
    - [0.67, 1.0]
  river:
    - [0.5, 1.0]
    - [1.0]

training:
  cluster_path: "/tmp/clusters"
  iterations: 10000
  lcfr_warmup_iterations: 5000000

snapshots:
  warmup_minutes: 60
  snapshot_every_minutes: 30
  output_dir: "/tmp/snapshots"
"#;

        let cfg: BlueprintV2Config =
            serde_yaml::from_str(yaml).expect("failed to parse toy config");

        // Game
        assert_eq!(cfg.game.name, "Test Config");
        assert_eq!(cfg.game.players, 6);
        assert!((cfg.game.stack_depth - 100.0).abs() < f64::EPSILON);
        assert!((cfg.game.small_blind - 0.5).abs() < f64::EPSILON);
        assert!((cfg.game.big_blind - 1.0).abs() < f64::EPSILON);
        // rake defaults to 0.0 when omitted
        assert!((cfg.game.rake_rate).abs() < f64::EPSILON);
        assert!((cfg.game.rake_cap).abs() < f64::EPSILON);

        // Clustering
        assert!(matches!(
            cfg.clustering.algorithm,
            ClusteringAlgorithm::PotentialAwareEmd
        ));
        assert_eq!(cfg.clustering.preflop.buckets, 169);
        assert_eq!(cfg.clustering.flop.buckets, 200);
        assert_eq!(cfg.clustering.turn.buckets, 200);
        assert_eq!(cfg.clustering.river.buckets, 200);
        // Defaults
        assert_eq!(cfg.clustering.seed, default_seed());
        assert_eq!(cfg.clustering.kmeans_iterations, default_kmeans_iterations());

        // Action abstraction
        assert_eq!(cfg.action_abstraction.preflop.len(), 2);
        assert_eq!(cfg.action_abstraction.preflop[0], vec!["2.5bb"]);
        assert_eq!(cfg.action_abstraction.flop[0], vec![0.33, 0.67, 1.0]);

        // Training
        assert_eq!(cfg.training.cluster_path, "/tmp/clusters");
        assert_eq!(cfg.training.iterations, Some(10_000));
        assert_eq!(cfg.training.time_limit_minutes, None);
        assert_eq!(cfg.training.lcfr_warmup_iterations, 5_000_000);
        // Defaults
        assert_eq!(cfg.training.lcfr_discount_interval, default_discount_interval());
        assert_eq!(cfg.training.prune_after_iterations, default_prune_after());
        assert_eq!(cfg.training.prune_threshold, default_prune_threshold());
        assert!((cfg.training.prune_explore_pct - default_prune_explore()).abs() < f64::EPSILON);
        assert_eq!(cfg.training.print_every_minutes, default_print_every());
        assert_eq!(cfg.training.batch_size, default_batch_size());

        // Snapshots
        assert_eq!(cfg.snapshots.warmup_minutes, 60);
        assert_eq!(cfg.snapshots.snapshot_every_minutes, 30);
        assert_eq!(cfg.snapshots.output_dir, "/tmp/snapshots");
    }

    #[test]
    fn test_serialize_round_trip() {
        let original = BlueprintV2Config {
            game: GameConfig {
                name: "Round Trip Test".to_string(),
                players: 2,
                stack_depth: 50.0,
                small_blind: 0.5,
                big_blind: 1.0,
                rake_rate: 0.045,
                rake_cap: 3.0,
            },
            clustering: ClusteringConfig {
                algorithm: ClusteringAlgorithm::PotentialAwareEmd,
                preflop: StreetClusterConfig { buckets: 169 },
                flop: StreetClusterConfig { buckets: 500 },
                turn: StreetClusterConfig { buckets: 500 },
                river: StreetClusterConfig { buckets: 500 },
                seed: 123,
                kmeans_iterations: 50,
            },
            action_abstraction: ActionAbstractionConfig {
                preflop: vec![vec!["2.5bb".to_owned()], vec!["3.0x".to_owned()]],
                flop: vec![vec![0.5, 1.0]],
                turn: vec![vec![0.5, 1.0]],
                river: vec![vec![1.0]],
            },
            training: TrainingConfig {
                cluster_path: "/data/clusters".to_owned(),
                iterations: None,
                time_limit_minutes: Some(720),
                lcfr_warmup_iterations: 5_000_000,
                lcfr_discount_interval: 500_000,
                prune_after_iterations: 5_000_000,
                prune_threshold: 0,
                prune_explore_pct: 0.05,
                print_every_minutes: 10,
                batch_size: 200,
                target_strategy_delta: None,
                purify_threshold: 0.0,
            },
            snapshots: SnapshotConfig {
                warmup_minutes: 120,
                snapshot_every_minutes: 60,
                output_dir: "/data/snapshots".to_owned(),
                resume: false,
                max_snapshots: None,
            },
        };

        let yaml = serde_yaml::to_string(&original).expect("failed to serialize");
        let restored: BlueprintV2Config =
            serde_yaml::from_str(&yaml).expect("failed to deserialize round-tripped YAML");

        // Game
        assert_eq!(restored.game.name, "Round Trip Test");
        assert_eq!(restored.game.players, original.game.players);
        assert!((restored.game.stack_depth - original.game.stack_depth).abs() < f64::EPSILON);
        assert!((restored.game.rake_rate - 0.045).abs() < f64::EPSILON);
        assert!((restored.game.rake_cap - 3.0).abs() < f64::EPSILON);

        // Clustering
        assert_eq!(restored.clustering.seed, 123);
        assert_eq!(restored.clustering.kmeans_iterations, 50);
        assert_eq!(restored.clustering.flop.buckets, 500);

        // Action abstraction
        assert_eq!(restored.action_abstraction.preflop, original.action_abstraction.preflop);
        assert_eq!(restored.action_abstraction.river, original.action_abstraction.river);

        // Training
        assert_eq!(restored.training.cluster_path, "/data/clusters");
        assert_eq!(restored.training.iterations, None);
        assert_eq!(restored.training.time_limit_minutes, Some(720));
        assert_eq!(restored.training.prune_threshold, 0);

        // Snapshots
        assert_eq!(restored.snapshots.warmup_minutes, 120);
        assert_eq!(restored.snapshots.output_dir, "/data/snapshots");
    }

    #[test]
    fn test_deserialize_with_rake() {
        let yaml = r#"
game:
  name: "Raked Game"
  players: 2
  stack_depth: 100.0
  small_blind: 0.5
  big_blind: 1.0
  rake_rate: 0.05
  rake_cap: 3.0

clustering:
  algorithm: potential_aware_emd
  preflop:
    buckets: 169
  flop:
    buckets: 200
  turn:
    buckets: 200
  river:
    buckets: 200

action_abstraction:
  preflop:
    - ["2.5bb"]
  flop:
    - [1.0]
  turn:
    - [1.0]
  river:
    - [1.0]

training:
  cluster_path: "/tmp/clusters"
  iterations: 100

snapshots:
  warmup_minutes: 60
  snapshot_every_minutes: 30
  output_dir: "/tmp/snapshots"
"#;

        let cfg: BlueprintV2Config =
            serde_yaml::from_str(yaml).expect("failed to parse raked config");

        assert_eq!(cfg.game.name, "Raked Game");
        assert!((cfg.game.rake_rate - 0.05).abs() < f64::EPSILON);
        assert!((cfg.game.rake_cap - 3.0).abs() < f64::EPSILON);
    }
}
