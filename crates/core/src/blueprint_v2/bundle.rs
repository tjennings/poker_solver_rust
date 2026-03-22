//! Snapshot bundle format — save/load training checkpoints to disk.
//!
//! On-disk layout:
//! ```text
//! output_dir/
//!   config.yaml                    # full BlueprintV2Config
//!   snapshot_NNNN/                 # numbered snapshots
//!     strategy.bin                 # bincode: BlueprintV2Strategy
//!     metadata.json                # { iteration, elapsed_minutes, ... }
//!     regrets.bin                  # bincode: raw storage (for resume)
//!   final/
//!     strategy.bin
//!     metadata.json
//!     regrets.bin
//! ```

// Arena indices are u32; action counts fit in u16. Truncation is safe for
// any practical game tree.
#![allow(clippy::cast_possible_truncation)]

use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::config::BlueprintV2Config;
use super::game_tree::{GameNode, GameTree};
use super::storage::BlueprintStorage;

/// Extracted average strategy, ready for serialization.
///
/// Stores per-decision-node action probabilities as `f32` in the same
/// flat layout used by [`BlueprintStorage`], but with concrete
/// probabilities instead of raw regrets / strategy sums.
#[derive(Debug, Serialize, Deserialize)]
pub struct BlueprintV2Strategy {
    /// Per-decision-node flat `f32` action probabilities.
    ///
    /// Length = sum over all decision nodes of
    /// `bucket_count_for_street * num_actions`.
    pub action_probs: Vec<f32>,
    /// Number of actions at each decision node (for reconstruction).
    pub node_action_counts: Vec<u16>,
    /// Street index (0-3) for each decision node.
    pub node_street_indices: Vec<u8>,
    /// Bucket counts per street `[preflop, flop, turn, river]`.
    pub bucket_counts: [u16; 4],
    /// MCCFR iterations completed when this snapshot was taken.
    pub iterations: u64,
    /// Wall-clock minutes elapsed when this snapshot was taken.
    pub elapsed_minutes: u64,
    /// Pre-computed flat offset for each decision node (not serialized).
    #[serde(skip)]
    node_offsets: Vec<usize>,
}

impl BlueprintV2Strategy {
    /// Extract the average strategy from storage, converting every
    /// (node, bucket) pair's strategy-sum distribution into `f32`
    /// probabilities.
    #[must_use]
    pub fn from_storage(storage: &BlueprintStorage, tree: &GameTree) -> Self {
        Self::from_storage_with_threshold(storage, tree, 0.0)
    }

    /// Build a strategy bundle from storage, zeroing actions below `purify_threshold`.
    ///
    /// Actions with probability below the threshold are set to zero and the
    /// remaining probabilities renormalized. Use 0.0 to disable purification.
    #[must_use]
    pub fn from_storage_with_threshold(
        storage: &BlueprintStorage,
        tree: &GameTree,
        purify_threshold: f64,
    ) -> Self {
        let mut action_probs = Vec::new();
        let mut node_action_counts = Vec::new();
        let mut node_street_indices = Vec::new();

        for (i, node) in tree.nodes.iter().enumerate() {
            if let GameNode::Decision {
                street, actions, ..
            } = node
            {
                let num_actions = actions.len() as u16;
                let street_idx = *street as u8;
                let buckets = storage.bucket_counts[street_idx as usize];

                node_action_counts.push(num_actions);
                node_street_indices.push(street_idx);

                for bucket in 0..buckets {
                    let avg = if purify_threshold > 0.0 {
                        storage.purified_average_strategy(i as u32, bucket, purify_threshold)
                    } else {
                        storage.average_strategy(i as u32, bucket)
                    };
                    for &p in &avg {
                        action_probs.push(p as f32);
                    }
                }
            }
        }

        let node_offsets = compute_node_offsets(
            &node_action_counts,
            &node_street_indices,
            storage.bucket_counts,
        );

        Self {
            action_probs,
            node_action_counts,
            node_street_indices,
            bucket_counts: storage.bucket_counts,
            iterations: 0,
            elapsed_minutes: 0,
            node_offsets,
        }
    }

    /// Rebuild transient fields that are not serialized (e.g.
    /// `node_offsets`).  Must be called after deserialization.
    fn post_deserialize(&mut self) {
        self.node_offsets = compute_node_offsets(
            &self.node_action_counts,
            &self.node_street_indices,
            self.bucket_counts,
        );
    }

    /// Serialize this strategy to a binary file via bincode.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the file cannot be created or written.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, self)
            .map_err(|e| io::Error::other(e.to_string()))
    }

    /// Deserialize a strategy from a binary file.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the file cannot be opened or contains
    /// invalid data.
    pub fn load(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut strat: Self = bincode::deserialize_from(reader)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        strat.post_deserialize();
        Ok(strat)
    }

    /// Get the action probability slice for a given decision node and bucket.
    ///
    /// Returns a slice of `f32` probabilities, one per action, or
    /// an empty slice if the index or bucket is out of range.
    #[must_use]
    pub fn get_action_probs(&self, decision_idx: usize, bucket: u16) -> &[f32] {
        if decision_idx >= self.node_action_counts.len() {
            return &[];
        }
        let num_actions = self.node_action_counts[decision_idx] as usize;
        let street_idx = self.node_street_indices[decision_idx] as usize;
        let buckets = self.bucket_counts[street_idx] as usize;

        if bucket as usize >= buckets {
            return &[];
        }

        let offset = self.node_offsets[decision_idx] + bucket as usize * num_actions;

        if offset + num_actions > self.action_probs.len() {
            return &[];
        }
        &self.action_probs[offset..offset + num_actions]
    }

    /// The number of decision nodes in this strategy.
    #[must_use]
    pub fn num_decision_nodes(&self) -> usize {
        self.node_action_counts.len()
    }

    /// Create an empty strategy with no decision nodes.
    ///
    /// Useful for tests and fallback construction where no trained
    /// blueprint is available.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            action_probs: Vec::new(),
            node_action_counts: Vec::new(),
            node_street_indices: Vec::new(),
            bucket_counts: [0; 4],
            iterations: 0,
            elapsed_minutes: 0,
            node_offsets: Vec::new(),
        }
    }

    /// Create a strategy with explicit action probabilities for testing.
    ///
    /// Computes the internal `node_offsets` table from the provided fields.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn from_parts(
        action_probs: Vec<f32>,
        node_action_counts: Vec<u16>,
        node_street_indices: Vec<u8>,
        bucket_counts: [u16; 4],
    ) -> Self {
        let node_offsets = compute_node_offsets(
            &node_action_counts,
            &node_street_indices,
            bucket_counts,
        );
        Self {
            action_probs,
            node_action_counts,
            node_street_indices,
            bucket_counts,
            iterations: 0,
            elapsed_minutes: 0,
            node_offsets,
        }
    }
}

/// Build a prefix-sum offset table so `get_action_probs` is O(1).
fn compute_node_offsets(
    node_action_counts: &[u16],
    node_street_indices: &[u8],
    bucket_counts: [u16; 4],
) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(node_action_counts.len());
    let mut offset = 0_usize;
    for (i, &n_act) in node_action_counts.iter().enumerate() {
        offsets.push(offset);
        let st = node_street_indices[i] as usize;
        let bk = bucket_counts[st] as usize;
        offset += bk * n_act as usize;
    }
    offsets
}

/// Save a full snapshot directory with strategy, metadata JSON, and
/// raw regrets (for training resume).
///
/// # Errors
///
/// Returns `Err` if any file cannot be created or written.
pub fn save_snapshot(
    dir: &Path,
    strategy: &BlueprintV2Strategy,
    storage: &BlueprintStorage,
    metadata_json: &str,
) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    strategy.save(&dir.join("strategy.bin"))?;
    storage.save_regrets(&dir.join("regrets.bin"))?;
    std::fs::write(dir.join("metadata.json"), metadata_json)?;
    Ok(())
}

/// Save the [`BlueprintV2Config`] as YAML in the output directory.
///
/// # Errors
///
/// Returns `Err` if the directory cannot be created or the file
/// cannot be written.
pub fn save_config(dir: &Path, config: &BlueprintV2Config) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let yaml =
        serde_yaml::to_string(config).map_err(|e| io::Error::other(e.to_string()))?;
    std::fs::write(dir.join("config.yaml"), yaml)
}

/// Load a [`BlueprintV2Config`] from `config.yaml` inside the given
/// directory.
///
/// # Errors
///
/// Returns `Err` if the file cannot be read or contains invalid YAML.
pub fn load_config(dir: &Path) -> io::Result<BlueprintV2Config> {
    let yaml = std::fs::read_to_string(dir.join("config.yaml"))?;
    serde_yaml::from_str(&yaml)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_v2::config::*;
    use crate::blueprint_v2::game_tree::GameTree;
    use crate::blueprint_v2::storage::BlueprintStorage;

    fn toy_tree() -> GameTree {
        GameTree::build(
            10.0,
            0.5,
            1.0,
            &[vec!["2.5bb".into()]],
            &[vec![1.0]],
            &[vec![1.0]],
            &[vec![1.0]],
        )
    }

    #[test]
    fn strategy_from_storage() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

        assert!(!strategy.action_probs.is_empty());
        assert!(!strategy.node_action_counts.is_empty());

        // Fresh storage produces uniform distributions: all probs in [0, 1].
        for &p in &strategy.action_probs {
            assert!(
                (0.0..=1.0).contains(&p),
                "invalid probability: {p}"
            );
        }
    }

    #[test]
    fn strategy_save_load_round_trip() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("strategy.bin");
        strategy.save(&path).expect("save strategy");

        let loaded = BlueprintV2Strategy::load(&path).expect("load strategy");
        assert_eq!(loaded.action_probs.len(), strategy.action_probs.len());
        assert_eq!(loaded.node_action_counts, strategy.node_action_counts);
        assert_eq!(loaded.bucket_counts, strategy.bucket_counts);

        for (a, b) in strategy
            .action_probs
            .iter()
            .zip(loaded.action_probs.iter())
        {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn save_snapshot_creates_files() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

        let dir = tempfile::tempdir().expect("create temp dir");
        let snapshot_dir = dir.path().join("snapshot_0000");

        let metadata = r#"{"iteration": 100, "elapsed_minutes": 5}"#;
        save_snapshot(&snapshot_dir, &strategy, &storage, metadata)
            .expect("save snapshot");

        assert!(snapshot_dir.join("strategy.bin").exists());
        assert!(snapshot_dir.join("regrets.bin").exists());
        assert!(snapshot_dir.join("metadata.json").exists());
    }

    #[test]
    fn config_save_load_round_trip() {
        let config = BlueprintV2Config {
            game: GameConfig {
                name: "Test".to_string(),
                players: 2,
                stack_depth: 10.0,
                small_blind: 0.5,
                big_blind: 1.0,
                rake_rate: 0.0,
                rake_cap: 0.0,
            },
            clustering: ClusteringConfig {
                algorithm: ClusteringAlgorithm::PotentialAwareEmd,
                preflop: StreetClusterConfig { buckets: 50, delta_bins: None, expected_delta: false, sample_boards: None },
                flop: StreetClusterConfig { buckets: 50, delta_bins: None, expected_delta: false, sample_boards: None },
                turn: StreetClusterConfig { buckets: 50, delta_bins: None, expected_delta: false, sample_boards: None },
                river: StreetClusterConfig { buckets: 50, delta_bins: None, expected_delta: false, sample_boards: None },
                seed: 42,
                kmeans_iterations: 50,
                cfvnet_river_data: None,
                per_flop: None,
            },
            action_abstraction: ActionAbstractionConfig {
                preflop: vec![vec!["2.5bb".into()]],
                flop: vec![vec![1.0]],
                turn: vec![vec![1.0]],
                river: vec![vec![1.0]],
            },
            training: TrainingConfig {
                cluster_path: None,
                iterations: Some(100),
                time_limit_minutes: None,
                lcfr_warmup_iterations: 0,
                lcfr_discount_interval: 1,
                prune_after_iterations: 200,
                prune_threshold: 0,
                prune_explore_pct: 0.05,
                print_every_minutes: 10,
                batch_size: 200,
                target_strategy_delta: None,
                purify_threshold: 0.0,
                equity_cache_path: None,
                dcfr_alpha: 1.0,
                dcfr_beta: 1.0,
                dcfr_gamma: 1.0,
            },
            snapshots: SnapshotConfig {
                warmup_minutes: 60,
                snapshot_every_minutes: 30,
                output_dir: "runs/".into(),
                resume: false,
                max_snapshots: None,
            },
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        save_config(dir.path(), &config).expect("save config");
        let loaded = load_config(dir.path()).expect("load config");
        assert!(
            (loaded.game.stack_depth - 10.0).abs() < f64::EPSILON
        );
        assert_eq!(loaded.clustering.river.buckets, 50);
    }
}
