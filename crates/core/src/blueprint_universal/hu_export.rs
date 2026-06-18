//! HU legacy-to-universal exporter (Phase 2).
//!
//! Converts a [`BlueprintV2Strategy`] (from `strategy.bin`) plus its
//! [`GameTree`] and [`BlueprintV2Config`] into the universal dense bundle
//! format.  Probabilities are passed through exactly as stored (f32,
//! no renormalization) so the export is bitwise identical to legacy.
//!
//! ## Row identity
//!
//! Namespace: `hu_arena` (0).  For HU arena rows, `local_bucket ==
//! global_bucket` (both are the per-street bucket index); this is
//! stable within any single bundle.
//!
//! ## Row key fingerprint
//!
//! Stable FNV-1a hash over `(namespace, seat, street,
//! source_node_idx, global_bucket)`.  Version 1 -- inputs documented
//! here and recorded in the manifest via `optional_features`.
//!
//! ## Action amounts
//!
//! `Bet(v)` and `Raise(v)` carry amounts in big blinds in the tree.
//! We convert to chips by multiplying by `config.game.big_blind`.
//! `Call` and `AllIn` resolved amounts require betting-state tracking
//! that the [`GameTree`] nodes do not carry; those are set to 0 chips
//! with `size_key` documenting the limitation.  A future phase may
//! add pot/contribution tracking to the tree.
//!
//! ## `size_key`
//!
//! Derived from the tree action's own parameterization:
//! - `Bet(v)` / `Raise(v)`: the BB amount as a string (e.g. `"5bb"`,
//!   `"10bb"`).
//! - `Fold`, `Check`, `Call`: empty string (no sizing dimension).
//! - `AllIn`: `"allin"`.

// Arena indices are u32; action counts fit in u16. Truncation is safe
// for any practical game tree.
#![allow(clippy::cast_possible_truncation)]

use std::collections::BTreeMap;
use std::path::Path;

use super::bundle::{write_bundle, BundleData};
use super::descriptors::{ActionDescriptor, ActionKind, RowDescriptor};
use super::error::FormatError;
use super::export_common::{
    self, RowEntry, bucket_semantic_fingerprint, flatten_entries,
    row_key_fingerprint,
};
use super::hash::Fnv1aHasher;
use super::manifest::{
    ActionsMetadata, BucketsMetadata, CompatibilityMetadata, GameMetadata,
    LayoutMetadata, Manifest, RakeConfig, SeatDescriptor, StrategyMetadata,
    TrainingMetadata,
};
use crate::blueprint_v2::bundle::{load_config, BlueprintV2Strategy};
use crate::blueprint_v2::config::BlueprintV2Config;
use crate::blueprint_v2::game_tree::{GameNode, GameTree, TreeAction};
use crate::blueprint_v2::storage::action_schema_fingerprint;

use super::export_common::NS_HU_ARENA as HU_ARENA_NS;

// ---------------------------------------------------------------------------
// Action mapping
// ---------------------------------------------------------------------------

/// Map a [`TreeAction`] to an [`ActionKind`] and aggressiveness flag.
fn map_action_kind(action: &TreeAction) -> (ActionKind, bool) {
    match action {
        TreeAction::Fold => (ActionKind::Fold, false),
        TreeAction::Check => (ActionKind::Check, false),
        TreeAction::Call => (ActionKind::Call, false),
        TreeAction::Bet(_) => (ActionKind::Bet, true),
        TreeAction::Raise(_) => (ActionKind::Raise, true),
        // AllIn in the HU tree is always generated in the raise/bet
        // section (aggressive).  The tree does not distinguish
        // call-all-in vs raise-all-in at the TreeAction level.
        TreeAction::AllIn => (ActionKind::AllInBetRaise, true),
    }
}

/// Compute `amount_chips` for a tree action.
///
/// `Bet(v)` and `Raise(v)` are in big blinds; we multiply by
/// `big_blind` to get chips.  Other actions use 0 (see module doc).
#[allow(clippy::cast_sign_loss)]
fn action_amount_chips(action: &TreeAction, big_blind: f64) -> u32 {
    match action {
        TreeAction::Bet(v) | TreeAction::Raise(v) => {
            (v * big_blind).round() as u32
        }
        _ => 0,
    }
}

/// Derive the `size_key` for a tree action.
fn action_size_key(action: &TreeAction) -> String {
    match action {
        TreeAction::Bet(v) | TreeAction::Raise(v) => format!("{v}bb"),
        TreeAction::AllIn => "allin".to_string(),
        _ => String::new(),
    }
}

/// Derive a human-readable label for a tree action.
fn action_label(action: &TreeAction) -> String {
    match action {
        TreeAction::Fold => "Fold".to_string(),
        TreeAction::Check => "Check".to_string(),
        TreeAction::Call => "Call".to_string(),
        TreeAction::Bet(v) => format!("Bet {v}bb"),
        TreeAction::Raise(v) => format!("Raise {v}bb"),
        TreeAction::AllIn => "All-In".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Core in-memory export
// ---------------------------------------------------------------------------

/// Export error for the HU exporter.
///
/// Wraps [`FormatError`] with exporter-specific variants.
#[derive(Debug)]
pub enum ExportError {
    /// A required file is missing from the legacy bundle.
    MissingFile {
        /// Human-readable detail about which file is missing.
        detail: String,
    },
    /// The snapshot name is invalid.
    BadSnapshotName {
        /// The invalid name that was provided.
        name: String,
    },
    /// Format error from the universal bundle writer.
    Format(FormatError),
    /// I/O error.
    Io(std::io::Error),
    /// JSON deserialization error.
    Json(String),
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingFile { detail } => {
                write!(f, "missing file: {detail}")
            }
            Self::BadSnapshotName { name } => {
                write!(f, "bad snapshot name: {name}")
            }
            Self::Format(e) => write!(f, "format error: {e}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Json(e) => write!(f, "JSON error: {e}"),
        }
    }
}

impl std::error::Error for ExportError {}

impl From<FormatError> for ExportError {
    fn from(e: FormatError) -> Self {
        Self::Format(e)
    }
}

impl From<std::io::Error> for ExportError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Training provenance metadata for the export.
pub struct TrainingInfo<'a> {
    /// Backend identifier (e.g. `"hu_dense"`, `"hu_sparse_projected"`).
    pub source_backend: &'a str,
    /// Number of training iterations completed.
    pub iterations: u64,
    /// Wall-clock training time in minutes.
    pub elapsed_minutes: f64,
}

/// Result of in-memory export: rows, actions, probabilities, and manifest.
pub struct ExportOutput {
    /// Row descriptors in spec sort order.
    pub rows: Vec<RowDescriptor>,
    /// Action descriptors (referenced by row offsets).
    pub actions: Vec<ActionDescriptor>,
    /// Probability values (referenced by row offsets).
    pub probs: Vec<f32>,
    /// Bundle manifest.
    pub manifest: Manifest,
}

/// In-memory export of a [`BlueprintV2Strategy`] into universal dense
/// bundle arrays and manifest.
///
/// # Errors
///
/// Returns [`ExportError`] on internal consistency failures.
pub fn export_hu_strategy_to_universal(
    config: &BlueprintV2Config,
    tree: &GameTree,
    strategy: &BlueprintV2Strategy,
    training: &TrainingInfo<'_>,
) -> Result<ExportOutput, ExportError> {
    let bucket_counts = [
        config.clustering.preflop.buckets,
        config.clustering.flop.buckets,
        config.clustering.turn.buckets,
        config.clustering.river.buckets,
    ];

    let mut entries = collect_row_entries(tree, strategy, bucket_counts, config);

    // Sort by spec order: (namespace, seat, street, source_node_idx,
    // global_bucket, row_key_fingerprint).
    entries.sort_by_key(RowEntry::sort_key);

    let (rows, actions, probs) = flatten_entries(&entries);

    let manifest = build_manifest(
        config,
        training,
        &rows,
        &actions,
        &probs,
        bucket_counts,
    );

    Ok(ExportOutput { rows, actions, probs, manifest })
}

// RowEntry and sort_key are provided by export_common.

/// Extract probabilities for a single bucket from the legacy strategy.
#[allow(clippy::cast_precision_loss)]
fn extract_bucket_probs(
    strategy: &BlueprintV2Strategy,
    decision_idx: usize,
    bucket: u16,
    num_actions: usize,
) -> Vec<f32> {
    let legacy_probs = strategy.get_action_probs(decision_idx, bucket);
    if legacy_probs.len() == num_actions {
        legacy_probs.to_vec()
    } else {
        let uniform = 1.0f32 / num_actions as f32;
        vec![uniform; num_actions]
    }
}

/// Collect all `(decision_node, bucket)` rows from the strategy.
fn collect_row_entries(
    tree: &GameTree,
    strategy: &BlueprintV2Strategy,
    bucket_counts: [u16; 4],
    config: &BlueprintV2Config,
) -> Vec<RowEntry> {
    let mut entries = Vec::new();
    let mut decision_idx = 0usize;

    for (node_idx, node) in tree.nodes.iter().enumerate() {
        let (player, street, tree_actions) = match node {
            GameNode::Decision {
                player, street, actions, ..
            } => (*player, *street as u8, actions),
            _ => continue,
        };

        let buckets = bucket_counts[street as usize];
        let schema_fp = action_schema_fingerprint(tree_actions);
        let action_descs =
            build_action_descriptors(tree_actions, config.game.big_blind);

        for bucket in 0..buckets {
            let probs = extract_bucket_probs(
                strategy, decision_idx, bucket, tree_actions.len(),
            );
            let fp = row_key_fingerprint(
                HU_ARENA_NS, player, street,
                node_idx as u32, u32::from(bucket),
            );
            entries.push(RowEntry {
                namespace: HU_ARENA_NS,
                seat: player,
                street,
                local_bucket: bucket,
                source_node_idx: node_idx as u32,
                global_bucket: u32::from(bucket),
                row_key_fp: fp,
                action_schema_fp: schema_fp,
                actions: action_descs.clone(),
                probs,
                semantic_key_kind: 0,
                semantic_key_index: 0,
            });
        }

        decision_idx += 1;
    }

    entries
}

/// Build action descriptors for a node's actions.
fn build_action_descriptors(
    tree_actions: &[TreeAction],
    big_blind: f64,
) -> Vec<ActionDescriptor> {
    tree_actions
        .iter()
        .enumerate()
        .map(|(idx, action)| {
            let (kind, is_aggressive) = map_action_kind(action);
            ActionDescriptor {
                kind,
                amount_chips: action_amount_chips(action, big_blind),
                size_key: action_size_key(action),
                label: action_label(action),
                is_aggressive,
                source_action_index: idx as u16,
            }
        })
        .collect()
}

// flatten_entries is provided by export_common.

// ---------------------------------------------------------------------------
// Manifest construction
// ---------------------------------------------------------------------------

/// Compute a stable game fingerprint from the config.
fn game_fingerprint(config: &BlueprintV2Config) -> u64 {
    let mut h = Fnv1aHasher::new();
    h.mix_u64(u64::from(config.game.players));
    h.mix_u64(config.game.stack_depth.to_bits());
    h.mix_u64(config.game.small_blind.to_bits());
    h.mix_u64(config.game.big_blind.to_bits());
    h.finish()
}

/// Compute an action abstraction fingerprint from the config.
fn action_abstraction_fingerprint(config: &BlueprintV2Config) -> u64 {
    let mut h = Fnv1aHasher::new();
    let repr = format!(
        "{:?}|{:?}|{:?}|{:?}",
        config.action_abstraction.preflop,
        config.action_abstraction.flop,
        config.action_abstraction.turn,
        config.action_abstraction.river,
    );
    h.mix_bytes(repr.as_bytes());
    h.finish()
}

// bucket_semantic_fingerprint is provided by export_common.

/// Build game metadata from config.
fn build_game_metadata(config: &BlueprintV2Config) -> GameMetadata {
    let stack = config.game.stack_depth;
    GameMetadata {
        game_kind: "holdem_no_limit".to_string(),
        num_players: 2,
        seats: vec![
            SeatDescriptor {
                seat_id: 0,
                label: "SB".to_string(),
                blind_ante_role: "small_blind".to_string(),
                starting_stack: stack,
            },
            SeatDescriptor {
                seat_id: 1,
                label: "BB".to_string(),
                blind_ante_role: "big_blind".to_string(),
                starting_stack: stack,
            },
        ],
        button_seat: 0,
        small_blind: config.game.small_blind,
        big_blind: config.game.big_blind,
        antes: vec![],
        straddles: vec![],
        stack_units: "chips".to_string(),
        rake: RakeConfig {
            rate: config.game.rake_rate,
            cap: config.game.rake_cap,
        },
        max_flop_players: None,
    }
}

/// Build bucket metadata from config and bucket counts.
fn build_buckets_metadata(
    config: &BlueprintV2Config,
    bucket_counts: [u16; 4],
) -> BucketsMetadata {
    let mut street_bucket_counts = BTreeMap::new();
    street_bucket_counts.insert(
        "preflop".to_string(),
        u64::from(bucket_counts[0]),
    );
    street_bucket_counts
        .insert("flop".to_string(), u64::from(bucket_counts[1]));
    street_bucket_counts
        .insert("turn".to_string(), u64::from(bucket_counts[2]));
    street_bucket_counts.insert(
        "river".to_string(),
        u64::from(bucket_counts[3]),
    );

    BucketsMetadata {
        bucket_mode: bucket_mode_for_config(config),
        bucket_semantic_fingerprint: bucket_semantic_fingerprint(
            bucket_counts,
        ),
        street_bucket_counts,
        bucket_files: vec![],
        bucket_generator_version: env!("CARGO_PKG_VERSION").to_string(),
        per_flop_bucket_config: None,
    }
}

/// Build training metadata for the manifest.
fn build_training_metadata(info: &TrainingInfo<'_>) -> TrainingMetadata {
    TrainingMetadata {
        source_backend: info.source_backend.to_string(),
        unit_kind: "Iteration".to_string(),
        units_completed: info.iterations,
        elapsed_minutes: info.elapsed_minutes,
        strategy_unit: "average_strategy".to_string(),
        command: None,
        config_path: None,
        config_fingerprint: None,
    }
}

/// Build layout metadata from the flattened arrays.
fn build_layout_metadata(
    rows: &[RowDescriptor],
    actions: &[ActionDescriptor],
    probs: &[f32],
) -> LayoutMetadata {
    LayoutMetadata {
        row_count: rows.len() as u64,
        action_descriptor_count: actions.len() as u64,
        probability_count: probs.len() as u64,
        row_sort_order:
            "namespace_seat_street_node_bucket_fingerprint".to_string(),
        row_namespace: vec!["hu_arena".to_string()],
        missing_row_policy: "reject".to_string(),
    }
}

fn build_manifest(
    config: &BlueprintV2Config,
    training: &TrainingInfo<'_>,
    rows: &[RowDescriptor],
    actions: &[ActionDescriptor],
    probs: &[f32],
    bucket_counts: [u16; 4],
) -> Manifest {
    Manifest {
        format_name: "dense_blueprint".to_string(),
        format_version: 1,
        compat_min_reader: 1,
        created_at: now_rfc3339_approx(),
        producer: format!(
            "poker-solver-core {}",
            env!("CARGO_PKG_VERSION"),
        ),
        producer_git: option_env!("GIT_HASH")
            .unwrap_or("unknown")
            .to_string(),
        required_features: Vec::new(),
        optional_features: vec!["row_key_fingerprint_v1".to_string()],
        game_fingerprint: game_fingerprint(config),
        source_config_fingerprint: None,
        game: build_game_metadata(config),
        training: build_training_metadata(training),
        strategy: StrategyMetadata {
            normalization_tolerance: 1e-4,
        },
        layout: build_layout_metadata(rows, actions, probs),
        actions: ActionsMetadata {
            action_abstraction_fingerprint:
                action_abstraction_fingerprint(config),
        },
        buckets: build_buckets_metadata(config, bucket_counts),
        compatibility: CompatibilityMetadata {
            legacy_fallback: false,
            missing_row_policy: "reject".to_string(),
            resumable: false,
        },
        files: BTreeMap::new(),
    }
}

/// Determine the bucket mode string based on config.
fn bucket_mode_for_config(config: &BlueprintV2Config) -> String {
    if config.training.cluster_path.is_some() {
        "referenced_external".to_string()
    } else {
        "preflop_canonical_classes".to_string()
    }
}

// Timestamp helpers delegate to export_common.
fn now_rfc3339_approx() -> String {
    export_common::now_rfc3339_approx()
}

// ---------------------------------------------------------------------------
// Disk-level export (legacy bundle dir -> universal bundle)
// ---------------------------------------------------------------------------

/// Snapshot metadata from a legacy `metadata.json`.
#[derive(serde::Deserialize)]
struct SnapshotMetadata {
    iteration: Option<u64>,
    elapsed_minutes: Option<f64>,
}

/// Validate a snapshot name: must be `final` or `snapshot_NNNN`.
fn validate_snapshot_name(name: &str) -> Result<(), ExportError> {
    if name == "final" {
        return Ok(());
    }
    if name.starts_with("snapshot_") && name.len() > 9 {
        let suffix = &name[9..];
        if suffix.chars().all(|c| c.is_ascii_digit()) {
            return Ok(());
        }
    }
    Err(ExportError::BadSnapshotName {
        name: name.to_string(),
    })
}

/// Export a legacy HU bundle directory to universal dense format.
///
/// # Arguments
///
/// * `bundle_dir` - Root of the legacy bundle (contains `config.yaml`).
/// * `snapshot` - Snapshot name (`"final"` or `"snapshot_NNNN"`).
/// * `out_dir` - Output directory for the universal bundle.
///
/// # Errors
///
/// Returns [`ExportError`] for missing files, bad snapshot names,
/// deserialization failures, or write errors.
/// Require a file to exist, returning `MissingFile` otherwise.
fn require_file(path: &Path, label: &str) -> Result<(), ExportError> {
    if path.exists() {
        Ok(())
    } else {
        Err(ExportError::MissingFile {
            detail: format!("{label} not found at {}", path.display()),
        })
    }
}

/// Load snapshot metadata from a legacy `metadata.json`.
fn load_snapshot_metadata(
    snapshot_dir: &Path,
) -> Result<SnapshotMetadata, ExportError> {
    let path = snapshot_dir.join("metadata.json");
    require_file(&path, "metadata.json")?;
    let text = std::fs::read_to_string(&path).map_err(ExportError::Io)?;
    serde_json::from_str(&text)
        .map_err(|e| ExportError::Json(e.to_string()))
}

/// Build a [`GameTree`] from a [`BlueprintV2Config`].
fn tree_from_config(config: &BlueprintV2Config) -> GameTree {
    GameTree::build_with_options(
        config.game.stack_depth,
        config.game.small_blind,
        config.game.big_blind,
        &config.action_abstraction.preflop,
        &config.action_abstraction.flop,
        &config.action_abstraction.turn,
        &config.action_abstraction.river,
        config.game.allow_preflop_limp,
    )
}

/// Write a universal dense bundle from in-memory HU training state.
///
/// Called at snapshot time when `format` is `universal` or `both`.
/// The bundle is written to `out_dir`, which should be a `universal/`
/// subdirectory inside the snapshot directory.
///
/// # Errors
///
/// Returns [`ExportError`] on export or I/O failure.
pub fn write_hu_universal_snapshot(
    config: &BlueprintV2Config,
    tree: &GameTree,
    strategy: &BlueprintV2Strategy,
    iterations: u64,
    elapsed_minutes: f64,
    config_yaml_path: &Path,
    out_dir: &Path,
) -> Result<(), ExportError> {
    let source_backend = match config.training.storage_backend.as_str() {
        "sparse" | "lazy" => "hu_sparse_projected",
        _ => "hu_dense",
    };

    let training = TrainingInfo {
        source_backend,
        iterations,
        elapsed_minutes,
    };

    let mut output = export_hu_strategy_to_universal(
        config, tree, strategy, &training,
    )?;

    export_common::retain_config_yaml(
        config_yaml_path,
        out_dir,
        &mut output.manifest,
    )?;

    write_bundle(
        out_dir,
        &output.manifest,
        &BundleData {
            rows: &output.rows,
            actions: &output.actions,
            probs: &output.probs,
        },
    )?;

    Ok(())
}

/// Export a legacy HU bundle directory to universal dense format.
///
/// # Errors
///
/// Returns [`ExportError`] for missing files, bad snapshot names,
/// deserialization failures, or write errors.
pub fn export_hu_bundle(
    bundle_dir: &Path,
    snapshot: &str,
    out_dir: &Path,
) -> Result<(), ExportError> {
    validate_snapshot_name(snapshot)?;

    require_file(&bundle_dir.join("config.yaml"), "config.yaml")?;
    let config = load_config(bundle_dir).map_err(ExportError::Io)?;

    let snapshot_dir = bundle_dir.join(snapshot);
    let strategy_path = snapshot_dir.join("strategy.bin");
    require_file(&strategy_path, "strategy.bin")?;
    let strategy =
        BlueprintV2Strategy::load(&strategy_path).map_err(ExportError::Io)?;

    let metadata = load_snapshot_metadata(&snapshot_dir)?;
    let tree = tree_from_config(&config);

    let source_backend = match config.training.storage_backend.as_str() {
        "sparse" | "lazy" => "hu_sparse_projected",
        _ => "hu_dense",
    };

    let training = TrainingInfo {
        source_backend,
        iterations: metadata.iteration.unwrap_or(0),
        elapsed_minutes: metadata.elapsed_minutes.unwrap_or(0.0),
    };

    let mut output = export_hu_strategy_to_universal(
        &config, &tree, &strategy, &training,
    )?;

    // Retain config.yaml so the bundle is self-contained for the Explorer.
    export_common::retain_config_yaml(
        &bundle_dir.join("config.yaml"),
        out_dir,
        &mut output.manifest,
    )?;

    write_bundle(
        out_dir,
        &output.manifest,
        &BundleData {
            rows: &output.rows,
            actions: &output.actions,
            probs: &output.probs,
        },
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_key_fingerprint_is_deterministic() {
        let fp1 = row_key_fingerprint(0, 0, 1, 42, 7);
        let fp2 = row_key_fingerprint(0, 0, 1, 42, 7);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn row_key_fingerprint_varies_with_inputs() {
        let fp1 = row_key_fingerprint(0, 0, 1, 42, 7);
        let fp2 = row_key_fingerprint(0, 0, 1, 42, 8);
        assert_ne!(fp1, fp2);

        let fp3 = row_key_fingerprint(0, 1, 1, 42, 7);
        assert_ne!(fp1, fp3);
    }

    #[test]
    fn action_size_key_bet() {
        assert_eq!(action_size_key(&TreeAction::Bet(5.0)), "5bb");
    }

    #[test]
    fn action_size_key_allin() {
        assert_eq!(action_size_key(&TreeAction::AllIn), "allin");
    }

    #[test]
    fn action_size_key_fold() {
        assert_eq!(action_size_key(&TreeAction::Fold), "");
    }

    #[test]
    fn action_amount_chips_bet() {
        // Bet 5bb with big_blind=2.0 -> 10 chips.
        assert_eq!(action_amount_chips(&TreeAction::Bet(5.0), 2.0), 10);
    }

    #[test]
    fn action_amount_chips_call() {
        assert_eq!(action_amount_chips(&TreeAction::Call, 2.0), 0);
    }

    #[test]
    fn validate_snapshot_name_final() {
        assert!(validate_snapshot_name("final").is_ok());
    }

    #[test]
    fn validate_snapshot_name_numbered() {
        assert!(validate_snapshot_name("snapshot_0001").is_ok());
    }

    #[test]
    fn validate_snapshot_name_invalid() {
        assert!(validate_snapshot_name("invalid_name!").is_err());
    }

    #[test]
    fn validate_snapshot_name_snapshot_no_digits() {
        assert!(validate_snapshot_name("snapshot_").is_err());
    }

    #[test]
    fn map_action_fold() {
        let (kind, agg) = map_action_kind(&TreeAction::Fold);
        assert_eq!(kind, ActionKind::Fold);
        assert!(!agg);
    }

    #[test]
    fn map_action_allin() {
        let (kind, agg) = map_action_kind(&TreeAction::AllIn);
        assert_eq!(kind, ActionKind::AllInBetRaise);
        assert!(agg);
    }

    /// Regression: refactoring to `Fnv1aHasher` must not change values.
    #[test]
    fn row_key_fingerprint_regression() {
        // Capture a known value computed by the original inline FNV-1a.
        let fp = row_key_fingerprint(0, 0, 1, 42, 7);
        assert_eq!(fp, 0xba11_9b0e_8372_975f);
    }

    #[test]
    fn now_rfc3339_approx_is_valid_rfc3339() {
        let ts = now_rfc3339_approx();
        // RFC 3339 format: YYYY-MM-DDTHH:MM:SSZ (20 chars exactly)
        assert_eq!(ts.len(), 20, "bad length: {ts:?}");
        assert!(ts.ends_with('Z'), "must end with Z: {ts:?}");
        assert_eq!(&ts[4..5], "-", "missing dash after year: {ts:?}");
        assert_eq!(&ts[7..8], "-", "missing dash after month: {ts:?}");
        assert_eq!(&ts[10..11], "T", "missing T separator: {ts:?}");
        assert_eq!(&ts[13..14], ":", "missing colon after hour: {ts:?}");
        assert_eq!(&ts[16..17], ":", "missing colon after min: {ts:?}");
        // Year must be >= 2020 (sanity check for current time)
        let year: u32 = ts[0..4].parse().expect("year is numeric");
        assert!(year >= 2020, "year too small: {year}");
    }

    // epoch_secs_to_rfc3339 tests moved to export_common.

    #[test]
    fn write_hu_universal_snapshot_produces_loadable_bundle() {
        use crate::blueprint_v2::config::*;
        use crate::blueprint_v2::game_tree::GameTree;
        use crate::blueprint_v2::storage::BlueprintStorage;

        let config = BlueprintV2Config {
            game: GameConfig {
                name: "NativeWriteTest".to_string(),
                players: 2,
                stack_depth: 10.0,
                small_blind: 1.0,
                big_blind: 2.0,
                rake_rate: 0.0,
                rake_cap: 0.0,
                allow_preflop_limp: true,
            },
            clustering: ClusteringConfig {
                algorithm: ClusteringAlgorithm::PotentialAwareEmd,
                preflop: StreetClusterConfig {
                    buckets: 3, delta_bins: None, expected_delta: false,
                    sample_boards: None, metric: Default::default(),
                },
                flop: StreetClusterConfig {
                    buckets: 2, delta_bins: None, expected_delta: false,
                    sample_boards: None, metric: Default::default(),
                },
                turn: StreetClusterConfig {
                    buckets: 2, delta_bins: None, expected_delta: false,
                    sample_boards: None, metric: Default::default(),
                },
                river: StreetClusterConfig {
                    buckets: 2, delta_bins: None, expected_delta: false,
                    sample_boards: None, metric: Default::default(),
                },
                seed: 42, kmeans_iterations: 10, cfvnet_river_data: None,
                per_flop: None,
            },
            action_abstraction: ActionAbstractionConfig {
                preflop: vec![vec!["5bb".into()]],
                flop: vec![vec![1.0]],
                turn: vec![vec![1.0]],
                river: vec![vec![1.0]],
            },
            training: TrainingConfig {
                cluster_path: None, iterations: Some(100),
                time_limit_minutes: None, lcfr_warmup_iterations: 0,
                lcfr_discount_interval: 1, prune_after_iterations: 200,
                prune_threshold: 0, prune_explore_pct: 0.05,
                print_every_minutes: 10, batch_size: 200,
                target_strategy_delta: None, purify_threshold: 0.0,
                equity_cache_path: None, dcfr_alpha: 1.0, dcfr_beta: 1.0,
                dcfr_gamma: 1.0, dcfr_epoch_cap: None,
                optimizer: "dcfr".to_string(),
                storage_backend: "dense".to_string(),
                sapcfr_eta: 0.5, brcfr_eta: 0.6,
                brcfr_warmup_iterations: 0, brcfr_interval: 100_000_000,
                use_baselines: false, baseline_alpha: 0.01,
                baseline_validation: Default::default(),
                prune_streets: None, regret_floor: None,
                exploitability_interval_minutes: 0,
                exploitability_samples: 100_000,
            },
            snapshots: SnapshotConfig {
                warmup_minutes: 60, snapshot_every_minutes: 30,
                output_dir: "runs/".into(), resume: false,
                max_snapshots: None, format: SnapshotFormat::Legacy,
            },
        };

        let tree = GameTree::build_with_options(
            config.game.stack_depth, config.game.small_blind,
            config.game.big_blind, &config.action_abstraction.preflop,
            &config.action_abstraction.flop, &config.action_abstraction.turn,
            &config.action_abstraction.river, config.game.allow_preflop_limp,
        );
        let storage = BlueprintStorage::new(&tree, [3, 2, 2, 2]);
        let strategy = crate::blueprint_v2::bundle::BlueprintV2Strategy::from_storage(
            &storage, &tree,
        );

        let dir = tempfile::tempdir().expect("create temp dir");
        let universal_dir = dir.path().join("universal");
        // Save config.yaml for retain_config_yaml
        let config_yaml = serde_yaml::to_string(&config).unwrap();
        let config_path = dir.path().join("config.yaml");
        std::fs::write(&config_path, &config_yaml).unwrap();

        write_hu_universal_snapshot(
            &config, &tree, &strategy, 42, 1.5, &config_path, &universal_dir,
        )
        .expect("native write should succeed");

        // Verify loadable
        assert!(universal_dir.join("blueprint.json").exists());
        assert!(universal_dir.join("strategy.rows.bin").exists());
        assert!(universal_dir.join("strategy.probs.f32.bin").exists());
        assert!(universal_dir.join("config.yaml").exists());

        let reader = crate::blueprint_universal::BundleReader::open(&universal_dir)
            .expect("bundle should load");
        assert!(reader.row_count() > 0);
    }
}
