//! MP eager-to-universal exporter (Phase 3).
//!
//! Converts live [`MpStorage`] strategy sums plus [`MpGameTree`] node
//! metadata and [`BlueprintMpConfig`] into the universal dense bundle
//! format.
//!
//! ## Row identity
//!
//! Namespace: `mp_arena` (1).  `local_bucket == global_bucket`
//! (both are the per-street bucket index in eager mode -- plain u16,
//! mirroring `hu_export`'s convention).
//!
//! ## Action amounts
//!
//! MP [`TreeAction::Lead(v)`] and [`TreeAction::Raise(v)`] store a
//! raise-TO amount in **chips** on ALL streets (verified against
//! `game_tree.rs` lines 460-543: `try_add_sized_action` pushes
//! `variant(raise_to.0)` where `raise_to` is always in chips).
//! This differs from HU `blueprint_v2::TreeAction` where amounts
//! are in big blinds.  The universal format's `amount_chips` field
//! is populated directly (no BB conversion needed).
//!
//! ## Probability conversion
//!
//! `MpStorage::average_strategy` returns `f64`; we convert with
//! `as f32`.  The acceptance test compares through the same conversion
//! (bitwise equality).
//!
//! ## Fingerprint hardening
//!
//! The MP action-abstraction fingerprint hashes typed values
//! field-by-field via [`Fnv1aHasher`] -- it does NOT use Debug
//! formatting (unlike the HU fingerprint, which is left as-is to
//! avoid changing existing HU manifest values).

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::BTreeMap;
use std::path::Path;

use super::bundle::{write_bundle, BundleData};
use super::descriptors::{ActionDescriptor, ActionKind};
use super::error::FormatError;
use super::export_common::{
    RowEntry, bucket_semantic_fingerprint, flatten_entries, now_rfc3339_approx,
    row_key_fingerprint,
};
use super::hash::Fnv1aHasher;
use super::manifest::{
    ActionsMetadata, BucketsMetadata, CompatibilityMetadata, GameMetadata,
    LayoutMetadata, Manifest, RakeConfig, SeatDescriptor, StrategyMetadata,
    TrainingMetadata,
};
use crate::blueprint_mp::config::{
    BlueprintMpConfig, ForcedBetKind, MpStreetSizes,
};
use crate::blueprint_mp::game_tree::{
    MpGameNode, MpGameTree, TreeAction,
};
use crate::blueprint_mp::storage::MpStorage;

use super::export_common::NS_MP_ARENA as MP_ARENA_NS;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Export error for the MP eager exporter.
#[derive(Debug)]
pub enum ExportError {
    /// Config bucket counts do not match storage bucket counts.
    BucketCountMismatch {
        /// Bucket counts from config.
        config_counts: [u16; 4],
        /// Bucket counts from storage.
        storage_counts: [u16; 4],
    },
    /// A required file is missing from the disk bundle.
    MissingFile {
        /// Human-readable detail.
        detail: String,
    },
    /// The metadata `kind` field does not match `"blueprint_mp"`.
    BadMetadataKind {
        /// The actual kind value found.
        actual: String,
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
            Self::BucketCountMismatch {
                config_counts,
                storage_counts,
            } => write!(
                f,
                "bucket count mismatch: config={config_counts:?}, \
                 storage={storage_counts:?}"
            ),
            Self::MissingFile { detail } => {
                write!(f, "missing file: {detail}")
            }
            Self::BadMetadataKind { actual } => {
                write!(f, "bad metadata kind: expected \"blueprint_mp\", \
                       got \"{actual}\"")
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

/// Training provenance metadata for MP export.
pub struct MpTrainingInfo {
    /// Number of meta-iterations completed.
    pub iterations: u64,
    /// Wall-clock training time in minutes.
    pub elapsed_minutes: f64,
}

/// Result of in-memory export: rows, actions, probabilities, and manifest.
#[derive(Debug)]
pub struct ExportOutput {
    /// Row descriptors in spec sort order.
    pub rows: Vec<super::descriptors::RowDescriptor>,
    /// Action descriptors (referenced by row offsets).
    pub actions: Vec<ActionDescriptor>,
    /// Probability values (referenced by row offsets).
    pub probs: Vec<f32>,
    /// Bundle manifest.
    pub manifest: Manifest,
}

// ---------------------------------------------------------------------------
// Action mapping for MP TreeAction
// ---------------------------------------------------------------------------

/// Map an MP [`TreeAction`] to an [`ActionKind`] and aggressiveness flag.
fn map_mp_action_kind(action: &TreeAction) -> (ActionKind, bool) {
    match action {
        TreeAction::Fold => (ActionKind::Fold, false),
        TreeAction::Check => (ActionKind::Check, false),
        TreeAction::Call => (ActionKind::Call, false),
        TreeAction::Lead(_) => (ActionKind::Bet, true),
        TreeAction::Raise(_) => (ActionKind::Raise, true),
        TreeAction::AllIn => (ActionKind::AllInBetRaise, true),
    }
}

/// Compute `amount_chips` for an MP tree action.
///
/// MP `Lead(v)` and `Raise(v)` store raise-TO amounts in **chips**
/// on all streets (verified: `game_tree.rs` lines 460-543).
/// No BB conversion is needed.
#[allow(clippy::cast_sign_loss)]
fn mp_action_amount_chips(action: &TreeAction) -> u32 {
    match action {
        TreeAction::Lead(v) | TreeAction::Raise(v) => v.round() as u32,
        _ => 0,
    }
}

/// Derive the `size_key` for an MP tree action.
fn mp_action_size_key(action: &TreeAction) -> String {
    match action {
        TreeAction::Lead(v) | TreeAction::Raise(v) => {
            format!("{v}chips")
        }
        TreeAction::AllIn => "allin".to_string(),
        _ => String::new(),
    }
}

/// Derive a human-readable label for an MP tree action.
fn mp_action_label(action: &TreeAction) -> String {
    match action {
        TreeAction::Fold => "Fold".to_string(),
        TreeAction::Check => "Check".to_string(),
        TreeAction::Call => "Call".to_string(),
        TreeAction::Lead(v) => format!("Bet {v}"),
        TreeAction::Raise(v) => format!("Raise {v}"),
        TreeAction::AllIn => "All-In".to_string(),
    }
}

// ---------------------------------------------------------------------------
// MP action schema fingerprint (hardened: typed values, not Debug)
// ---------------------------------------------------------------------------

/// Compute an action schema fingerprint for MP tree actions.
///
/// Uses the same structural approach as
/// `blueprint_v2::storage::action_schema_fingerprint` (kind discriminant and
/// amount bits via [`Fnv1aHasher`]), but adapted for the MP
/// [`TreeAction`] variants.
pub(crate) fn mp_action_schema_fingerprint(actions: &[TreeAction]) -> u64 {
    let mut h = Fnv1aHasher::new();
    h.mix_u64(actions.len() as u64);
    for action in actions {
        match *action {
            TreeAction::Fold => h.mix_u64(0),
            TreeAction::Check => h.mix_u64(1),
            TreeAction::Call => h.mix_u64(2),
            TreeAction::Lead(v) => {
                h.mix_u64(3);
                h.mix_u64(v.to_bits());
            }
            TreeAction::Raise(v) => {
                h.mix_u64(4);
                h.mix_u64(v.to_bits());
            }
            TreeAction::AllIn => h.mix_u64(5),
        }
    }
    h.finish()
}

// ---------------------------------------------------------------------------
// MP action abstraction fingerprint (hardened: typed values, not Debug)
// ---------------------------------------------------------------------------

/// Compute a stable fingerprint over the MP action abstraction config.
///
/// Hashes per-street sizing lists field-by-field rather than using
/// Debug formatting.  Each entry is hashed by its canonical
/// representation.
fn mp_action_abstraction_fingerprint(config: &BlueprintMpConfig) -> u64 {
    let mut h = Fnv1aHasher::new();
    hash_mp_street_sizes(&mut h, &config.action_abstraction.preflop);
    hash_mp_street_sizes(&mut h, &config.action_abstraction.flop);
    hash_mp_street_sizes(&mut h, &config.action_abstraction.turn);
    hash_mp_street_sizes(&mut h, &config.action_abstraction.river);
    if let Some(max_fp) = config.action_abstraction.max_flop_players {
        h.mix_u8(1);
        h.mix_u8(max_fp);
    } else {
        h.mix_u8(0);
    }
    h.finish()
}

/// Hash a single street's sizing config into the hasher.
fn hash_mp_street_sizes(h: &mut Fnv1aHasher, sizes: &MpStreetSizes) {
    // Lead sizes
    h.mix_u64(sizes.lead.len() as u64);
    for entry in &sizes.lead {
        let s = yaml_value_canonical(entry);
        h.mix_bytes(s.as_bytes());
    }
    // Raise depth sizes
    h.mix_u64(sizes.raise.len() as u64);
    for depth in &sizes.raise {
        h.mix_u64(depth.len() as u64);
        for entry in depth {
            let s = yaml_value_canonical(entry);
            h.mix_bytes(s.as_bytes());
        }
    }
}

/// Canonical string representation of a YAML value for fingerprinting.
fn yaml_value_canonical(v: &serde_yaml::Value) -> String {
    match v {
        serde_yaml::Value::String(s) => s.clone(),
        serde_yaml::Value::Number(n) => format!("{n}"),
        _ => format!("{v:?}"),
    }
}

// ---------------------------------------------------------------------------
// Core in-memory export
// ---------------------------------------------------------------------------

/// In-memory export of live [`MpStorage`] strategy into universal dense
/// bundle arrays and manifest.
///
/// # Errors
///
/// Returns [`ExportError::BucketCountMismatch`] if config and storage
/// bucket counts disagree.
pub fn export_mp_strategy_to_universal(
    config: &BlueprintMpConfig,
    tree: &MpGameTree,
    storage: &MpStorage,
    training: &MpTrainingInfo,
) -> Result<ExportOutput, ExportError> {
    let config_counts = config.clustering.bucket_counts();
    validate_bucket_counts(config_counts, storage.bucket_counts)?;

    let mut entries = collect_mp_row_entries(tree, storage, config_counts);
    entries.sort_by_key(RowEntry::sort_key);

    let (rows, actions, probs) = flatten_entries(&entries);
    let manifest = build_mp_manifest(
        config, training, &rows, &actions, &probs, config_counts,
    );

    Ok(ExportOutput { rows, actions, probs, manifest })
}

/// Hard error if config vs storage bucket counts disagree.
fn validate_bucket_counts(
    config_counts: [u16; 4],
    storage_counts: [u16; 4],
) -> Result<(), ExportError> {
    if config_counts != storage_counts {
        return Err(ExportError::BucketCountMismatch {
            config_counts,
            storage_counts,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Row collection (unified)
// ---------------------------------------------------------------------------

/// Collect all `(decision_node, bucket)` rows from the MP tree using a
/// caller-supplied probability function.
///
/// `prob_fn(node_idx, decision_idx, bucket, num_actions)` returns the
/// probability vector for one (node, bucket) pair.  The `node_idx` is
/// the index into `tree.nodes`; `decision_idx` is the sequential count
/// of decision nodes seen so far (used by the projected strategy path).
fn collect_row_entries_with(
    tree: &MpGameTree,
    bucket_counts: [u16; 4],
    mut prob_fn: impl FnMut(u32, usize, u16, usize) -> Vec<f32>,
) -> Vec<RowEntry> {
    let mut entries = Vec::new();
    let mut decision_idx = 0usize;

    for (node_idx, node) in tree.nodes.iter().enumerate() {
        let (seat, street, tree_actions) = match node {
            MpGameNode::Decision {
                seat, street, actions, ..
            } => (*seat, *street, actions),
            _ => continue,
        };

        let street_u8 = street as u8;
        let buckets = bucket_counts[street_u8 as usize];
        let num_actions = tree_actions.len();
        let schema_fp = mp_action_schema_fingerprint(tree_actions);
        let action_descs = build_mp_action_descriptors(tree_actions);

        for bucket in 0..buckets {
            let probs = prob_fn(
                node_idx as u32, decision_idx, bucket, num_actions,
            );
            let fp = row_key_fingerprint(
                MP_ARENA_NS,
                seat.index(),
                street_u8,
                node_idx as u32,
                u32::from(bucket),
            );
            entries.push(RowEntry {
                namespace: MP_ARENA_NS,
                seat: seat.index(),
                street: street_u8,
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

/// Collect rows using live [`MpStorage`] average-strategy sums.
fn collect_mp_row_entries(
    tree: &MpGameTree,
    storage: &MpStorage,
    bucket_counts: [u16; 4],
) -> Vec<RowEntry> {
    let mut avg_buf = vec![0.0_f64; 64];
    collect_row_entries_with(tree, bucket_counts, |node_idx, _, bucket, n| {
        if avg_buf.len() < n {
            avg_buf.resize(n, 0.0);
        }
        storage.average_strategy(node_idx, bucket, n, &mut avg_buf);
        avg_buf[..n].iter().map(|&p| p as f32).collect()
    })
}

/// Build action descriptors for an MP node's actions.
fn build_mp_action_descriptors(
    tree_actions: &[TreeAction],
) -> Vec<ActionDescriptor> {
    tree_actions
        .iter()
        .enumerate()
        .map(|(idx, action)| {
            let (kind, is_aggressive) = map_mp_action_kind(action);
            ActionDescriptor {
                kind,
                amount_chips: mp_action_amount_chips(action),
                size_key: mp_action_size_key(action),
                label: mp_action_label(action),
                is_aggressive,
                source_action_index: idx as u16,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Manifest construction
// ---------------------------------------------------------------------------

/// Compute a stable game fingerprint for the MP config.
fn mp_game_fingerprint(config: &BlueprintMpConfig) -> u64 {
    let mut h = Fnv1aHasher::new();
    h.mix_u64(u64::from(config.game.num_players));
    h.mix_u64(config.game.stack_depth.to_bits());
    for blind in &config.game.blinds {
        h.mix_u8(blind.seat);
        h.mix_u64(blind.amount.to_bits());
    }
    h.finish()
}

/// Build game metadata from MP config.
fn build_mp_game_metadata(config: &BlueprintMpConfig) -> GameMetadata {
    let stack = config.game.stack_depth;
    let seats: Vec<SeatDescriptor> = config
        .game
        .blinds
        .iter()
        .enumerate()
        .map(|(i, blind)| SeatDescriptor {
            seat_id: blind.seat,
            label: seat_label_for_blind(&blind.kind, i),
            blind_ante_role: blind_role_string(&blind.kind),
            starting_stack: stack,
        })
        .collect();

    // Fill in seats without blinds
    let n = config.game.num_players;
    let mut all_seats: Vec<SeatDescriptor> = (0..n)
        .map(|i| {
            seats
                .iter()
                .find(|s| s.seat_id == i)
                .cloned()
                .unwrap_or_else(|| SeatDescriptor {
                    seat_id: i,
                    label: format!("seat_{i}"),
                    blind_ante_role: "none".to_string(),
                    starting_stack: stack,
                })
        })
        .collect();
    all_seats.sort_by_key(|s| s.seat_id);

    let button_seat = find_mp_button(&config.game);

    GameMetadata {
        game_kind: "holdem_no_limit".to_string(),
        num_players: n,
        seats: all_seats,
        button_seat,
        small_blind: sb_amount(&config.game),
        big_blind: bb_amount(&config.game),
        antes: vec![],
        straddles: vec![],
        stack_units: "chips".to_string(),
        rake: RakeConfig {
            rate: config.game.rake_rate,
            cap: config.game.rake_cap,
        },
        max_flop_players: config.action_abstraction.max_flop_players,
    }
}

fn seat_label_for_blind(kind: &ForcedBetKind, _idx: usize) -> String {
    match kind {
        ForcedBetKind::SmallBlind => "SB".to_string(),
        ForcedBetKind::BigBlind => "BB".to_string(),
        ForcedBetKind::Ante => "Ante".to_string(),
        ForcedBetKind::BbAnte => "BBAnte".to_string(),
        ForcedBetKind::Straddle => "Straddle".to_string(),
    }
}

fn blind_role_string(kind: &ForcedBetKind) -> String {
    match kind {
        ForcedBetKind::SmallBlind => "small_blind".to_string(),
        ForcedBetKind::BigBlind => "big_blind".to_string(),
        ForcedBetKind::Ante => "ante".to_string(),
        ForcedBetKind::BbAnte => "bb_ante".to_string(),
        ForcedBetKind::Straddle => "straddle".to_string(),
    }
}

fn sb_amount(game: &crate::blueprint_mp::config::MpGameConfig) -> f64 {
    game.blinds
        .iter()
        .find(|b| b.kind == ForcedBetKind::SmallBlind)
        .map_or(0.0, |b| b.amount)
}

fn bb_amount(game: &crate::blueprint_mp::config::MpGameConfig) -> f64 {
    game.blinds
        .iter()
        .find(|b| b.kind == ForcedBetKind::BigBlind)
        .map_or(0.0, |b| b.amount)
}

/// Find the button seat per MP `game_tree` conventions.
///
/// In 2-player: SB = BTN. In N-player: BTN is the seat before SB.
fn find_mp_button(
    game: &crate::blueprint_mp::config::MpGameConfig,
) -> u8 {
    let n = game.num_players;
    if n == 2 {
        game.blinds
            .iter()
            .find(|b| b.kind == ForcedBetKind::SmallBlind)
            .map_or(0, |b| b.seat)
    } else {
        game.blinds
            .iter()
            .find(|b| b.kind == ForcedBetKind::SmallBlind)
            .map_or(n - 1, |b| (b.seat + n - 1) % n)
    }
}

/// Build training metadata for the MP manifest.
fn build_mp_training_metadata(info: &MpTrainingInfo) -> TrainingMetadata {
    TrainingMetadata {
        source_backend: "mp_eager_dense".to_string(),
        unit_kind: "MetaIteration".to_string(),
        units_completed: info.iterations,
        elapsed_minutes: info.elapsed_minutes,
        strategy_unit: "average_strategy".to_string(),
        command: None,
        config_path: None,
        config_fingerprint: None,
    }
}

/// Build layout metadata from the flattened arrays.
fn build_mp_layout_metadata(
    rows: &[super::descriptors::RowDescriptor],
    actions: &[ActionDescriptor],
    probs: &[f32],
) -> LayoutMetadata {
    LayoutMetadata {
        row_count: rows.len() as u64,
        action_descriptor_count: actions.len() as u64,
        probability_count: probs.len() as u64,
        row_sort_order:
            "namespace_seat_street_node_bucket_fingerprint".to_string(),
        row_namespace: vec!["mp_arena".to_string()],
        missing_row_policy: "reject".to_string(),
    }
}

fn build_mp_manifest(
    config: &BlueprintMpConfig,
    training: &MpTrainingInfo,
    rows: &[super::descriptors::RowDescriptor],
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
        optional_features: vec![
            "row_key_fingerprint_v1".to_string(),
            "mp_action_abstraction_fingerprint_v1".to_string(),
        ],
        game_fingerprint: mp_game_fingerprint(config),
        source_config_fingerprint: None,
        game: build_mp_game_metadata(config),
        training: build_mp_training_metadata(training),
        strategy: StrategyMetadata {
            normalization_tolerance: 1e-4,
        },
        layout: build_mp_layout_metadata(rows, actions, probs),
        actions: ActionsMetadata {
            action_abstraction_fingerprint:
                mp_action_abstraction_fingerprint(config),
        },
        buckets: build_mp_buckets_metadata(config, bucket_counts),
        compatibility: CompatibilityMetadata {
            legacy_fallback: false,
            missing_row_policy: "reject".to_string(),
            resumable: false,
        },
        files: BTreeMap::new(),
    }
}

fn build_mp_buckets_metadata(
    _config: &BlueprintMpConfig,
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
        bucket_mode: "eager_runtime_assigned".to_string(),
        bucket_semantic_fingerprint: bucket_semantic_fingerprint(
            bucket_counts,
        ),
        street_bucket_counts,
        bucket_files: vec![],
        bucket_generator_version: env!("CARGO_PKG_VERSION").to_string(),
        per_flop_bucket_config: None,
    }
}

// ---------------------------------------------------------------------------
// Disk wrapper: export from saved snapshot
// ---------------------------------------------------------------------------

/// Snapshot metadata from an MP `metadata.json`.
#[derive(serde::Deserialize)]
struct MpSnapshotMetadata {
    kind: Option<String>,
    iterations: Option<u64>,
    elapsed_minutes: Option<u64>,
    bucket_counts: Option<[u16; 4]>,
}

/// Require a file to exist, returning `MissingFile` otherwise.
fn require_file(path: &Path, label: &str) -> Result<(), ExportError> {
    if let Some(detail) = super::export_common::check_file_exists(path, label) {
        Err(ExportError::MissingFile { detail })
    } else {
        Ok(())
    }
}

/// Load an MP config from the output directory.
fn load_mp_config(mp_output_dir: &Path) -> Result<BlueprintMpConfig, ExportError> {
    let config_path = mp_output_dir.join("config.yaml");
    require_file(&config_path, "config.yaml")?;
    let config_text =
        std::fs::read_to_string(&config_path).map_err(ExportError::Io)?;
    serde_yaml::from_str(&config_text)
        .map_err(|e| ExportError::Json(e.to_string()))
}

/// Loaded MP snapshot: projected strategy plus parsed metadata.
struct MpSnapshotBundle {
    strategy: crate::blueprint_v2::bundle::BlueprintV2Strategy,
    metadata: MpSnapshotMetadata,
}

/// Load and validate a snapshot's strategy and metadata from disk.
fn load_mp_snapshot(
    snapshot_dir: &Path,
) -> Result<MpSnapshotBundle, ExportError> {
    let strategy_path = snapshot_dir.join("strategy.bin");
    require_file(&strategy_path, "strategy.bin")?;

    let metadata_path = snapshot_dir.join("metadata.json");
    require_file(&metadata_path, "metadata.json")?;
    let meta_text =
        std::fs::read_to_string(&metadata_path).map_err(ExportError::Io)?;
    let metadata: MpSnapshotMetadata = serde_json::from_str(&meta_text)
        .map_err(|e| ExportError::Json(e.to_string()))?;

    if let Some(kind) = &metadata.kind
        && kind != "blueprint_mp"
    {
        return Err(ExportError::BadMetadataKind {
            actual: kind.clone(),
        });
    }

    let strategy = crate::blueprint_v2::bundle::BlueprintV2Strategy::load(
        &strategy_path,
    )
    .map_err(ExportError::Io)?;

    Ok(MpSnapshotBundle { strategy, metadata })
}

/// Validate bucket counts between config and snapshot, returning error
/// on mismatch.
fn validate_snapshot_bucket_counts(
    config_counts: [u16; 4],
    snapshot: &MpSnapshotBundle,
) -> Result<(), ExportError> {
    if snapshot.strategy.bucket_counts != config_counts {
        return Err(ExportError::BucketCountMismatch {
            config_counts,
            storage_counts: snapshot.strategy.bucket_counts,
        });
    }
    if let Some(meta_counts) = snapshot.metadata.bucket_counts
        && meta_counts != config_counts
    {
        return Err(ExportError::BucketCountMismatch {
            config_counts,
            storage_counts: meta_counts,
        });
    }
    Ok(())
}

/// Export a saved MP eager snapshot to universal dense format.
///
/// Loads `config.yaml`, builds the tree, loads `strategy.bin` and
/// `metadata.json`, validates bucket counts, and emits the bundle.
///
/// Write a universal dense bundle from in-memory MP eager training state.
///
/// Called at snapshot time when `format` is `universal` or `both`.
///
/// # Errors
///
/// Returns [`ExportError`] on export or I/O failure.
pub fn write_mp_universal_snapshot(
    config: &BlueprintMpConfig,
    tree: &MpGameTree,
    storage: &MpStorage,
    iterations: u64,
    elapsed_minutes: f64,
    config_yaml_path: &Path,
    out_dir: &Path,
) -> Result<(), ExportError> {
    let training = MpTrainingInfo {
        iterations,
        elapsed_minutes,
    };

    let mut output = export_mp_strategy_to_universal(
        config, tree, storage, &training,
    )?;

    super::export_common::retain_config_yaml(
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

/// # Errors
///
/// Returns [`ExportError`] for missing files, bucket count mismatches,
/// or write errors.
pub fn export_mp_bundle(
    mp_output_dir: &Path,
    snapshot: &str,
    out_dir: &Path,
) -> Result<(), ExportError> {
    let config = load_mp_config(mp_output_dir)?;
    let snap = load_mp_snapshot(&mp_output_dir.join(snapshot))?;
    validate_snapshot_bucket_counts(
        config.clustering.bucket_counts(), &snap,
    )?;

    let tree = MpGameTree::build(&config.game, &config.action_abstraction);
    let training = MpTrainingInfo {
        iterations: snap.metadata.iterations.unwrap_or(0),
        elapsed_minutes: snap.metadata.elapsed_minutes.unwrap_or(0) as f64,
    };

    let mut output = export_mp_strategy_from_projected(
        &config, &tree, &snap.strategy, &training,
    );

    // Retain config.yaml so the bundle is self-contained for the Explorer.
    super::export_common::retain_config_yaml(
        &mp_output_dir.join("config.yaml"),
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

/// Export from a projected [`BlueprintV2Strategy`] (disk path).
///
/// The strategy already contains f32 probabilities projected via
/// `average_strategy`, so we pass them through bitwise.
fn export_mp_strategy_from_projected(
    config: &BlueprintMpConfig,
    tree: &MpGameTree,
    strategy: &crate::blueprint_v2::bundle::BlueprintV2Strategy,
    training: &MpTrainingInfo,
) -> ExportOutput {
    let bucket_counts = config.clustering.bucket_counts();
    let mut entries = collect_projected_row_entries(
        tree, strategy, bucket_counts,
    );
    entries.sort_by_key(RowEntry::sort_key);

    let (rows, actions, probs) = flatten_entries(&entries);
    let manifest = build_mp_manifest(
        config, training, &rows, &actions, &probs, bucket_counts,
    );

    ExportOutput { rows, actions, probs, manifest }
}

/// Collect rows from a projected [`BlueprintV2Strategy`].
fn collect_projected_row_entries(
    tree: &MpGameTree,
    strategy: &crate::blueprint_v2::bundle::BlueprintV2Strategy,
    bucket_counts: [u16; 4],
) -> Vec<RowEntry> {
    collect_row_entries_with(tree, bucket_counts, |_, dec_idx, bucket, n| {
        extract_projected_probs(strategy, dec_idx, bucket, n)
    })
}

/// Extract probabilities for a single bucket from the projected strategy.
fn extract_projected_probs(
    strategy: &crate::blueprint_v2::bundle::BlueprintV2Strategy,
    decision_idx: usize,
    bucket: u16,
    num_actions: usize,
) -> Vec<f32> {
    let legacy_probs =
        strategy.get_action_probs(decision_idx, bucket);
    if legacy_probs.len() == num_actions {
        legacy_probs.to_vec()
    } else {
        let uniform = 1.0f32 / num_actions as f32;
        vec![uniform; num_actions]
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_mp_action_fold() {
        let (kind, agg) = map_mp_action_kind(&TreeAction::Fold);
        assert_eq!(kind, ActionKind::Fold);
        assert!(!agg);
    }

    #[test]
    fn map_mp_action_check() {
        let (kind, agg) = map_mp_action_kind(&TreeAction::Check);
        assert_eq!(kind, ActionKind::Check);
        assert!(!agg);
    }

    #[test]
    fn map_mp_action_call() {
        let (kind, agg) = map_mp_action_kind(&TreeAction::Call);
        assert_eq!(kind, ActionKind::Call);
        assert!(!agg);
    }

    #[test]
    fn map_mp_action_lead() {
        let (kind, agg) = map_mp_action_kind(&TreeAction::Lead(10.0));
        assert_eq!(kind, ActionKind::Bet);
        assert!(agg);
    }

    #[test]
    fn map_mp_action_raise() {
        let (kind, agg) = map_mp_action_kind(&TreeAction::Raise(20.0));
        assert_eq!(kind, ActionKind::Raise);
        assert!(agg);
    }

    #[test]
    fn map_mp_action_allin() {
        let (kind, agg) = map_mp_action_kind(&TreeAction::AllIn);
        assert_eq!(kind, ActionKind::AllInBetRaise);
        assert!(agg);
    }

    #[test]
    fn mp_action_amount_lead_in_chips() {
        // Lead(10.0) = 10 chips, no conversion needed
        assert_eq!(mp_action_amount_chips(&TreeAction::Lead(10.0)), 10);
    }

    #[test]
    fn mp_action_amount_raise_in_chips() {
        // Raise(20.0) = 20 chips
        assert_eq!(mp_action_amount_chips(&TreeAction::Raise(20.0)), 20);
    }

    #[test]
    fn mp_action_amount_fold_is_zero() {
        assert_eq!(mp_action_amount_chips(&TreeAction::Fold), 0);
    }

    #[test]
    fn mp_action_size_key_lead() {
        assert_eq!(mp_action_size_key(&TreeAction::Lead(10.0)), "10chips");
    }

    #[test]
    fn mp_action_size_key_allin() {
        assert_eq!(mp_action_size_key(&TreeAction::AllIn), "allin");
    }

    #[test]
    fn mp_action_size_key_fold() {
        assert_eq!(mp_action_size_key(&TreeAction::Fold), "");
    }

    #[test]
    fn mp_schema_fingerprint_deterministic() {
        let actions = vec![TreeAction::Fold, TreeAction::Call, TreeAction::Lead(10.0)];
        let fp1 = mp_action_schema_fingerprint(&actions);
        let fp2 = mp_action_schema_fingerprint(&actions);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn mp_schema_fingerprint_changes_with_amount() {
        let a1 = vec![TreeAction::Lead(10.0)];
        let a2 = vec![TreeAction::Lead(20.0)];
        assert_ne!(
            mp_action_schema_fingerprint(&a1),
            mp_action_schema_fingerprint(&a2)
        );
    }

    #[test]
    fn mp_schema_fingerprint_changes_with_kind() {
        let a1 = vec![TreeAction::Lead(10.0)];
        let a2 = vec![TreeAction::Raise(10.0)];
        assert_ne!(
            mp_action_schema_fingerprint(&a1),
            mp_action_schema_fingerprint(&a2)
        );
    }

    #[test]
    fn validate_bucket_counts_ok() {
        assert!(validate_bucket_counts([10, 10, 10, 10], [10, 10, 10, 10]).is_ok());
    }

    #[test]
    fn validate_bucket_counts_mismatch() {
        let result = validate_bucket_counts([10, 10, 10, 10], [5, 5, 5, 5]);
        assert!(matches!(result, Err(ExportError::BucketCountMismatch { .. })));
    }
}
