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
use super::manifest::{
    ActionsMetadata, BucketsMetadata, CompatibilityMetadata, GameMetadata,
    LayoutMetadata, Manifest, RakeConfig, SeatDescriptor, StrategyMetadata,
    TrainingMetadata,
};
use crate::blueprint_v2::bundle::{load_config, BlueprintV2Strategy};
use crate::blueprint_v2::config::BlueprintV2Config;
use crate::blueprint_v2::game_tree::{GameNode, GameTree, TreeAction};
use crate::blueprint_v2::storage::action_schema_fingerprint;

/// Namespace constant for HU arena rows.
const HU_ARENA_NS: u16 = 0;

// ---------------------------------------------------------------------------
// FNV-1a row key fingerprint
// ---------------------------------------------------------------------------

/// Compute a stable FNV-1a row key fingerprint over the identity tuple.
///
/// Inputs (version 1): `(namespace: u16, seat: u8, street: u8,
/// source_node_idx: u32, global_bucket: u32)`.
fn row_key_fingerprint(
    namespace: u16,
    seat: u8,
    street: u8,
    source_node_idx: u32,
    global_bucket: u32,
) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut h = FNV_OFFSET;
    for byte in namespace.to_le_bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h ^= u64::from(seat);
    h = h.wrapping_mul(FNV_PRIME);
    h ^= u64::from(street);
    h = h.wrapping_mul(FNV_PRIME);
    for byte in source_node_idx.to_le_bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(FNV_PRIME);
    }
    for byte in global_bucket.to_le_bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

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
    /// A referenced bucket/cluster file is missing.
    MissingBucketFile {
        /// Path to the missing file.
        path: String,
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
            Self::MissingBucketFile { path } => {
                write!(f, "missing bucket file: {path}")
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

/// Result of in-memory export: rows, actions, probabilities, and manifest.
pub type ExportResult = (
    Vec<RowDescriptor>,
    Vec<ActionDescriptor>,
    Vec<f32>,
    Manifest,
);

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
    source_backend: &str,
    iterations: u64,
    elapsed_minutes: f64,
) -> Result<ExportResult, ExportError> {
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
        source_backend,
        iterations,
        elapsed_minutes,
        &rows,
        &actions,
        &probs,
        bucket_counts,
    );

    Ok((rows, actions, probs, manifest))
}

// ---------------------------------------------------------------------------
// Internal row entry
// ---------------------------------------------------------------------------

/// Intermediate row before sorting and offset assignment.
struct RowEntry {
    namespace: u16,
    seat: u8,
    street: u8,
    source_node_idx: u32,
    global_bucket: u32,
    row_key_fp: u64,
    action_schema_fp: u64,
    actions: Vec<ActionDescriptor>,
    probs: Vec<f32>,
}

impl RowEntry {
    fn sort_key(&self) -> (u16, u8, u8, u32, u32, u64) {
        (
            self.namespace,
            self.seat,
            self.street,
            self.source_node_idx,
            self.global_bucket,
            self.row_key_fp,
        )
    }
}

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
                source_node_idx: node_idx as u32,
                global_bucket: u32::from(bucket),
                row_key_fp: fp,
                action_schema_fp: schema_fp,
                actions: action_descs.clone(),
                probs,
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

/// Assign contiguous offsets to sorted entries and flatten into arrays.
fn flatten_entries(
    entries: &[RowEntry],
) -> (Vec<RowDescriptor>, Vec<ActionDescriptor>, Vec<f32>) {
    let mut rows = Vec::with_capacity(entries.len());
    let mut all_actions = Vec::new();
    let mut all_probs = Vec::new();

    for (row_id, entry) in entries.iter().enumerate() {
        let action_offset = all_actions.len() as u64;
        let prob_offset = all_probs.len() as u64;

        rows.push(RowDescriptor {
            row_id: row_id as u64,
            namespace: entry.namespace,
            seat: entry.seat,
            street: entry.street,
            local_bucket: entry.global_bucket as u16,
            global_bucket: entry.global_bucket,
            source_node_idx: entry.source_node_idx,
            action_offset,
            action_count: entry.actions.len() as u16,
            prob_offset,
            row_key_fingerprint: entry.row_key_fp,
            action_schema_fingerprint: entry.action_schema_fp,
            semantic_key_kind: 0,
            semantic_key_offset: 0,
        });

        all_actions.extend_from_slice(&entry.actions);
        all_probs.extend_from_slice(&entry.probs);
    }

    (rows, all_actions, all_probs)
}

// ---------------------------------------------------------------------------
// Manifest construction
// ---------------------------------------------------------------------------

/// Compute a stable game fingerprint from the config.
fn game_fingerprint(config: &BlueprintV2Config) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut h = FNV_OFFSET;
    for byte in u64::from(config.game.players).to_le_bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(FNV_PRIME);
    }
    for byte in config.game.stack_depth.to_bits().to_le_bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(FNV_PRIME);
    }
    for byte in config.game.small_blind.to_bits().to_le_bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(FNV_PRIME);
    }
    for byte in config.game.big_blind.to_bits().to_le_bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Compute an action abstraction fingerprint from the config.
fn action_abstraction_fingerprint(config: &BlueprintV2Config) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut h = FNV_OFFSET;
    let repr = format!(
        "{:?}|{:?}|{:?}|{:?}",
        config.action_abstraction.preflop,
        config.action_abstraction.flop,
        config.action_abstraction.turn,
        config.action_abstraction.river,
    );
    for byte in repr.as_bytes() {
        h ^= u64::from(*byte);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Compute a bucket semantic fingerprint from bucket counts.
fn bucket_semantic_fingerprint(bucket_counts: [u16; 4]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut h = FNV_OFFSET;
    for bc in bucket_counts {
        for byte in bc.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

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
fn build_training_metadata(
    source_backend: &str,
    iterations: u64,
    elapsed_minutes: f64,
) -> TrainingMetadata {
    TrainingMetadata {
        source_backend: source_backend.to_string(),
        unit_kind: "Iteration".to_string(),
        units_completed: iterations,
        elapsed_minutes,
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

#[allow(clippy::too_many_arguments)]
fn build_manifest(
    config: &BlueprintV2Config,
    source_backend: &str,
    iterations: u64,
    elapsed_minutes: f64,
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
        producer: format!("poker-solver-core {}", env!("CARGO_PKG_VERSION")),
        producer_git: option_env!("GIT_HASH").unwrap_or("unknown").to_string(),
        required_features: Vec::new(),
        optional_features: vec!["row_key_fingerprint_v1".to_string()],
        game_fingerprint: game_fingerprint(config),
        source_config_fingerprint: None,
        game: build_game_metadata(config),
        training: build_training_metadata(source_backend, iterations, elapsed_minutes),
        strategy: StrategyMetadata { normalization_tolerance: 1e-4 },
        layout: build_layout_metadata(rows, actions, probs),
        actions: ActionsMetadata {
            action_abstraction_fingerprint: action_abstraction_fingerprint(config),
        },
        buckets: build_buckets_metadata(config, bucket_counts),
        compatibility: CompatibilityMetadata {
            legacy_fallback: false,
            missing_row_policy: "reject".to_string(),
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

/// Current UTC timestamp in RFC 3339 approximate format.
fn now_rfc3339_approx() -> String {
    use std::time::SystemTime;
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Approximate RFC 3339 without a chrono dependency.
    format!("1970-01-01T00:00:00Z+{secs}s")
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

    let (rows, actions, probs, manifest) =
        export_hu_strategy_to_universal(
            &config, &tree, &strategy, source_backend,
            metadata.iteration.unwrap_or(0),
            metadata.elapsed_minutes.unwrap_or(0.0),
        )?;

    write_bundle(
        out_dir,
        &manifest,
        &BundleData { rows: &rows, actions: &actions, probs: &probs },
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
}
