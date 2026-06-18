//! MP lazy sparse-to-universal exporter (Phase 4).
//!
//! Converts [`SparseSnapshotEntry`] rows (realized infosets only) from
//! lazy MP training into the universal dense bundle format, including
//! the format's first semantic-key side table (`strategy.semantic.bin`).
//!
//! ## Semantic identity
//!
//! Namespace: `mp_semantic` (2). Each row carries the full sparse key
//! identity: seat, street (from packed bucket high bits), local bucket
//! (low 14 bits), global bucket (full packed u16 widened to u32), and
//! the action history fields (`history_hi`, `history_lo`, `history_hash`,
//! `history_len`) stored in the side table.
//!
//! `source_node_idx` is `u32::MAX` because lazy rows do not reference
//! arena tree nodes.
//!
//! ## Opaque actions
//!
//! `SparseSnapshotEntry` carries only `num_actions`; action kinds and
//! amounts are NOT recoverable (histories >32 actions exist only as a
//! hash, so replay is impossible in general). Each lazy row emits
//! `num_actions` descriptors with kind `Opaque`, `amount_chips = 0`,
//! empty `size_key`/`label`, `is_aggressive = false`, and
//! `source_action_index = i` (the per-row action index the trainer
//! used). Consumers needing real action semantics must replay public
//! state from the history key (future work).
//!
//! ## Probability conversion
//!
//! Strategy sums are normalized exactly like
//! `SparseMpStorage::average_strategy`: sum > 0 implies each/sum as
//! f64 then `as f32`; sum == 0 implies uniform `1/n`. Present
//! zero-mass rows export as uniform probabilities. Present-zero-mass
//! vs absent is distinguishable only by row presence (no per-row
//! flags yet).
//!
//! ## Non-resumable
//!
//! Lazy MP universal exports are analysis-only. The manifest sets
//! `compatibility.resumable = false` and no `cfr.snapshot.bin` is
//! written.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::BTreeMap;
use std::path::Path;

use super::bundle::{write_bundle, BundleData};
use super::descriptors::{ActionDescriptor, ActionKind};
use super::error::FormatError;
use super::export_common::{
    RowEntry, bucket_semantic_fingerprint, flatten_entries, now_rfc3339_approx,
    NS_MP_SEMANTIC, SEMANTIC_KEY_MP_HISTORY_V1,
};
use super::hash::Fnv1aHasher;
use super::manifest::{
    ActionsMetadata, BucketsMetadata, CompatibilityMetadata, FileEntry,
    GameMetadata, LayoutMetadata, Manifest, RakeConfig, SeatDescriptor,
    StrategyMetadata, TrainingMetadata,
};
use crate::blueprint_mp::config::{BlueprintMpConfig, ForcedBetKind};
use crate::blueprint_mp::sparse_storage::{MpInfosetKey, SparseSnapshotEntry};

// Re-export from descriptors so external consumers using this module's
// path (`mp_lazy_export::SemanticKeyRecord`) still compile.
pub use super::descriptors::{SemanticKeyRecord, SEMANTIC_RECORD_SIZE};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Lightweight config for lazy export (avoids requiring full
/// `BlueprintMpConfig` which needs YAML parsing).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LazyExportConfig {
    pub num_players: u8,
    pub stack_depth: f64,
    pub bucket_counts: [u16; 4],
    /// Small blind amount in chips (0.0 if no SB, e.g. ante-only games).
    #[serde(default)]
    pub small_blind: f64,
    /// Big blind amount in chips (0.0 if no BB, e.g. ante-only games).
    #[serde(default)]
    pub big_blind: f64,
}

/// Training provenance metadata for lazy MP export.
pub struct LazyTrainingInfo {
    /// Number of meta-iterations completed.
    pub iterations: u64,
    /// Wall-clock training time in minutes.
    pub elapsed_minutes: f64,
}

/// Result of in-memory export.
#[derive(Debug)]
pub struct ExportOutput {
    pub rows: Vec<super::descriptors::RowDescriptor>,
    pub actions: Vec<ActionDescriptor>,
    pub probs: Vec<f32>,
    pub semantic_records: Vec<SemanticKeyRecord>,
    pub manifest: Manifest,
}

/// Export error for the MP lazy exporter.
#[derive(Debug)]
pub enum ExportError {
    /// The metadata `kind` field does not match expected value.
    BadMetadataKind {
        expected: String,
        actual: String,
    },
    /// A required file is missing.
    MissingFile { detail: String },
    /// Format error from the universal bundle writer.
    Format(FormatError),
    /// I/O error.
    Io(std::io::Error),
    /// Deserialization error.
    Deserialize(String),
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMetadataKind { expected, actual } => write!(
                f,
                "bad metadata kind: expected \"{expected}\", got \"{actual}\""
            ),
            Self::MissingFile { detail } => {
                write!(f, "missing file: {detail}")
            }
            Self::Format(e) => write!(f, "format error: {e}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Deserialize(e) => write!(f, "deserialization error: {e}"),
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

// ---------------------------------------------------------------------------
// Row key fingerprint (mp_semantic version)
// ---------------------------------------------------------------------------

/// Compute FNV-1a row key fingerprint for a lazy MP semantic key.
///
/// Inputs: `(namespace, seat, packed_bucket, history_hi, history_lo,
/// history_hash, history_len)`. This makes every realized row unique;
/// the reader's duplicate-identity rejection is the backstop.
fn mp_semantic_row_key_fingerprint(key: &MpInfosetKey) -> u64 {
    let mut h = Fnv1aHasher::new();
    h.mix_u16(NS_MP_SEMANTIC);
    h.mix_u8(key.seat);
    h.mix_u16(key.bucket);
    h.mix_u64(key.history_hi);
    h.mix_u64(key.history_lo);
    h.mix_u64(key.history_hash);
    h.mix_u16(key.history_len);
    h.finish()
}

// ---------------------------------------------------------------------------
// Action schema fingerprint (opaque version)
// ---------------------------------------------------------------------------

/// Compute action schema fingerprint for opaque lazy rows.
///
/// Inputs: `(opaque_marker=0xFF, num_actions)`.
fn opaque_action_schema_fingerprint(num_actions: u8) -> u64 {
    let mut h = Fnv1aHasher::new();
    h.mix_u8(0xFF); // opaque marker
    h.mix_u8(num_actions);
    h.finish()
}

// ---------------------------------------------------------------------------
// Probability normalization (matches SparseMpStorage::average_strategy)
// ---------------------------------------------------------------------------

/// Normalize strategy sums to probabilities, matching sparse storage
/// semantics: sum > 0 -> each/sum as f64 then as f32; sum == 0 ->
/// uniform 1/n.
fn normalize_strategy_sums(sums: &[u64]) -> Vec<f32> {
    let n = sums.len();
    if n == 0 {
        return Vec::new();
    }
    let total: f64 = sums.iter().map(|&s| s as f64).sum();
    if total > 0.0 {
        sums.iter().map(|&s| (s as f64 / total) as f32).collect()
    } else {
        let uniform = (1.0_f64 / n as f64) as f32;
        vec![uniform; n]
    }
}

// ---------------------------------------------------------------------------
// Row collection
// ---------------------------------------------------------------------------

/// Build opaque action descriptors for a lazy row.
fn build_opaque_actions(num_actions: u8) -> Vec<ActionDescriptor> {
    (0..num_actions)
        .map(|i| ActionDescriptor {
            kind: ActionKind::Opaque,
            amount_chips: 0,
            size_key: String::new(),
            label: String::new(),
            is_aggressive: false,
            source_action_index: u16::from(i),
        })
        .collect()
}

/// Collect row entries from sparse snapshot entries.
fn collect_lazy_entries(
    entries: &[SparseSnapshotEntry],
) -> (Vec<RowEntry>, Vec<SemanticKeyRecord>) {
    let mut row_entries = Vec::with_capacity(entries.len());
    let mut semantic_records = Vec::with_capacity(entries.len());

    for entry in entries {
        let key = &entry.key;
        let street = (key.bucket >> 14) as u8;
        let local_bucket = key.bucket & 0x3FFF;
        let global_bucket = u32::from(key.bucket);
        let fp = mp_semantic_row_key_fingerprint(key);
        let schema_fp =
            opaque_action_schema_fingerprint(entry.num_actions);
        let probs = normalize_strategy_sums(&entry.strategy_sums);
        let actions = build_opaque_actions(entry.num_actions);

        let sem_idx = semantic_records.len() as u64;
        semantic_records.push(SemanticKeyRecord {
            history_hi: key.history_hi,
            history_lo: key.history_lo,
            history_hash: key.history_hash,
            history_len: key.history_len,
        });

        row_entries.push(RowEntry {
            namespace: NS_MP_SEMANTIC,
            seat: key.seat,
            street,
            local_bucket,
            source_node_idx: u32::MAX,
            global_bucket,
            row_key_fp: fp,
            action_schema_fp: schema_fp,
            actions,
            probs,
            semantic_key_kind: SEMANTIC_KEY_MP_HISTORY_V1,
            semantic_key_index: sem_idx,
        });
    }

    (row_entries, semantic_records)
}

// ---------------------------------------------------------------------------
// In-memory export
// ---------------------------------------------------------------------------

/// Export sparse snapshot entries to universal dense bundle arrays and
/// manifest. No tree construction needed.
pub fn export_lazy_sparse_to_universal(
    config: &LazyExportConfig,
    entries: &[SparseSnapshotEntry],
    training: &LazyTrainingInfo,
) -> ExportOutput {
    let (mut row_entries, semantic_records) = collect_lazy_entries(entries);
    row_entries.sort_by_key(RowEntry::sort_key);

    // After sorting, we need to remap semantic_key_index to reflect
    // the new order. Build a mapping from sorted row order to the
    // original semantic record index.
    // Since semantic records are 1:1 with entries, and entries may be
    // reordered by sort, we need to build semantic records in sorted
    // order.
    let sorted_sem: Vec<SemanticKeyRecord> = row_entries
        .iter()
        .map(|r| semantic_records[r.semantic_key_index as usize])
        .collect();

    // Reassign semantic_key_index to sequential order after sort
    for (i, entry) in row_entries.iter_mut().enumerate() {
        entry.semantic_key_index = i as u64;
    }

    let (rows, actions, probs) = flatten_entries(&row_entries);
    let layout = LayoutMetadata {
        row_count: rows.len() as u64,
        action_descriptor_count: actions.len() as u64,
        probability_count: probs.len() as u64,
        row_sort_order:
            "namespace_seat_street_node_bucket_fingerprint".to_string(),
        row_namespace: vec!["mp_semantic".to_string()],
        missing_row_policy: "uniform_legal".to_string(),
    };
    let manifest = build_lazy_manifest(config, training, layout);

    ExportOutput {
        rows,
        actions,
        probs,
        semantic_records: sorted_sem,
        manifest,
    }
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

fn build_lazy_manifest(
    config: &LazyExportConfig,
    training: &LazyTrainingInfo,
    layout: LayoutMetadata,
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
        required_features: vec!["mp_semantic_rows_v1".to_string()],
        optional_features: vec![
            "mp_semantic_row_key_fingerprint_v1".to_string(),
            "opaque_action_schema_fingerprint_v1".to_string(),
        ],
        game_fingerprint: lazy_game_fingerprint(config),
        source_config_fingerprint: None,
        game: build_lazy_game_metadata(config),
        training: build_lazy_training_metadata(training),
        strategy: StrategyMetadata {
            normalization_tolerance: 1e-4,
        },
        layout,
        actions: ActionsMetadata {
            action_abstraction_fingerprint: 0,
        },
        buckets: build_lazy_buckets_metadata(config.bucket_counts),
        compatibility: CompatibilityMetadata {
            legacy_fallback: false,
            missing_row_policy: "uniform_legal".to_string(),
            resumable: false,
        },
        files: BTreeMap::new(),
    }
}

fn build_lazy_game_metadata(config: &LazyExportConfig) -> GameMetadata {
    let seats = (0..config.num_players)
        .map(|i| SeatDescriptor {
            seat_id: i,
            label: format!("seat_{i}"),
            blind_ante_role: "none".to_string(),
            starting_stack: config.stack_depth,
        })
        .collect();
    GameMetadata {
        game_kind: "holdem_no_limit".to_string(),
        num_players: config.num_players,
        seats,
        button_seat: 0,
        small_blind: config.small_blind,
        big_blind: config.big_blind,
        antes: vec![],
        straddles: vec![],
        stack_units: "chips".to_string(),
        rake: RakeConfig { rate: 0.0, cap: 0.0 },
        max_flop_players: None,
    }
}

fn build_lazy_training_metadata(
    training: &LazyTrainingInfo,
) -> TrainingMetadata {
    TrainingMetadata {
        source_backend: "mp_lazy_sparse_projected".to_string(),
        unit_kind: "MetaIteration".to_string(),
        units_completed: training.iterations,
        elapsed_minutes: training.elapsed_minutes,
        strategy_unit: "average_strategy".to_string(),
        command: None,
        config_path: None,
        config_fingerprint: None,
    }
}

fn build_lazy_buckets_metadata(bc: [u16; 4]) -> BucketsMetadata {
    let mut sbc = BTreeMap::new();
    sbc.insert("preflop".to_string(), u64::from(bc[0]));
    sbc.insert("flop".to_string(), u64::from(bc[1]));
    sbc.insert("turn".to_string(), u64::from(bc[2]));
    sbc.insert("river".to_string(), u64::from(bc[3]));
    BucketsMetadata {
        bucket_mode: "lazy_runtime_assigned".to_string(),
        bucket_semantic_fingerprint: bucket_semantic_fingerprint(bc),
        street_bucket_counts: sbc,
        bucket_files: vec![],
        bucket_generator_version: env!("CARGO_PKG_VERSION").to_string(),
        per_flop_bucket_config: None,
    }
}

fn lazy_game_fingerprint(config: &LazyExportConfig) -> u64 {
    let mut h = Fnv1aHasher::new();
    h.mix_u8(config.num_players);
    h.mix_u64(config.stack_depth.to_bits());
    h.finish()
}

// ---------------------------------------------------------------------------
// Bundle writer (with semantic side table)
// ---------------------------------------------------------------------------

/// Write a lazy bundle to disk, including the semantic side table.
///
/// # Errors
///
/// Returns `FormatError` on I/O failure.
pub fn write_lazy_bundle(
    dir: &Path,
    output: &ExportOutput,
) -> Result<(), FormatError> {
    // First write the three standard payloads + semantic side table
    std::fs::create_dir_all(dir)?;

    let sem_entry = write_semantic_file(dir, &output.semantic_records)?;

    // Write the standard bundle, then patch the manifest to include
    // the semantic file entry.
    let mut manifest = output.manifest.clone();
    manifest.files.insert(
        "strategy.semantic.bin".to_string(),
        sem_entry,
    );

    write_bundle(
        dir,
        &manifest,
        &BundleData {
            rows: &output.rows,
            actions: &output.actions,
            probs: &output.probs,
        },
    )
}

/// Write the semantic side table file, reusing the shared `write_payload`
/// pattern from bundle.rs.
fn write_semantic_file(
    dir: &Path,
    records: &[SemanticKeyRecord],
) -> Result<FileEntry, FormatError> {
    use super::bundle::{write_payload_file, PayloadSpec};
    use super::header::MAGIC_SEMANTIC;

    write_payload_file(
        dir,
        &PayloadSpec {
            name: "strategy.semantic.bin",
            magic: MAGIC_SEMANTIC,
            record_count: records.len(),
            write_fn: Box::new(|w| {
                for record in records {
                    record.write_to_writer(w)?;
                }
                Ok(())
            }),
        },
    )
}

// ---------------------------------------------------------------------------
// Disk wrapper
// ---------------------------------------------------------------------------

/// Snapshot metadata for lazy sparse exports.
#[derive(serde::Deserialize)]
struct LazySnapshotMetadata {
    kind: Option<String>,
    iterations: Option<u64>,
    elapsed_minutes: Option<u64>,
}

/// Export a saved lazy sparse snapshot to universal dense format.
///
/// Reads `config.yaml` from `bundle_dir` (the parent training output
/// directory) and `sparse_entries.bin` + `metadata.json` from
/// `bundle_dir/<snapshot_name>/`. Validates `kind ==
/// "blueprint_mp_lazy_sparse"`.
///
/// This mirrors the eager export's `(bundle_dir, snapshot, out)`
/// pattern: config lives in the bundle root, data in the snapshot
/// subdirectory.
///
/// # Errors
///
/// Returns `ExportError` for missing files, kind mismatches, or write
/// errors.
pub fn export_lazy_bundle_from_disk(
    bundle_dir: &Path,
    snapshot_name: &str,
    out_dir: &Path,
) -> Result<(), ExportError> {
    let snapshot_dir = bundle_dir.join(snapshot_name);
    let config = load_lazy_config(bundle_dir)?;
    let metadata = load_lazy_metadata(&snapshot_dir)?;
    let entries = load_sparse_entries(&snapshot_dir)?;

    let training = LazyTrainingInfo {
        iterations: metadata.iterations.unwrap_or(0),
        elapsed_minutes: metadata.elapsed_minutes.unwrap_or(0) as f64,
    };

    let mut output =
        export_lazy_sparse_to_universal(&config, &entries, &training);

    // Retain config.yaml so the bundle is self-contained for the Explorer.
    retain_lazy_config_yaml(
        &bundle_dir.join("config.yaml"),
        out_dir,
        &mut output.manifest,
    )?;

    write_lazy_bundle(out_dir, &output)?;
    Ok(())
}

/// Copy source config.yaml into the output bundle and set manifest's
/// `config_path` so the Explorer can rebuild the game tree.
fn retain_lazy_config_yaml(
    source: &Path,
    out_dir: &Path,
    manifest: &mut super::manifest::Manifest,
) -> Result<(), ExportError> {
    std::fs::create_dir_all(out_dir)?;
    std::fs::copy(source, out_dir.join("config.yaml"))?;
    manifest.training.config_path = Some("config.yaml".to_string());
    Ok(())
}

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

fn load_lazy_config(
    bundle_dir: &Path,
) -> Result<LazyExportConfig, ExportError> {
    let path = bundle_dir.join("config.yaml");
    require_file(&path, "config.yaml")?;
    let text = std::fs::read_to_string(&path)?;
    let full: BlueprintMpConfig = serde_yaml::from_str(&text)
        .map_err(|e| ExportError::Deserialize(e.to_string()))?;
    let sb = full.game.blinds.iter()
        .find(|b| b.kind == ForcedBetKind::SmallBlind)
        .map_or(0.0, |b| b.amount);
    let bb = full.game.blinds.iter()
        .find(|b| b.kind == ForcedBetKind::BigBlind)
        .map_or(0.0, |b| b.amount);
    Ok(LazyExportConfig {
        num_players: full.game.num_players,
        stack_depth: full.game.stack_depth,
        bucket_counts: full.clustering.bucket_counts(),
        small_blind: sb,
        big_blind: bb,
    })
}

fn load_lazy_metadata(
    dir: &Path,
) -> Result<LazySnapshotMetadata, ExportError> {
    let path = dir.join("metadata.json");
    require_file(&path, "metadata.json")?;
    let text = std::fs::read_to_string(&path)?;
    let metadata: LazySnapshotMetadata = serde_json::from_str(&text)
        .map_err(|e| ExportError::Deserialize(e.to_string()))?;
    let kind = metadata.kind.as_deref().unwrap_or("");
    if kind != "blueprint_mp_lazy_sparse" {
        return Err(ExportError::BadMetadataKind {
            expected: "blueprint_mp_lazy_sparse".to_string(),
            actual: kind.to_string(),
        });
    }
    Ok(metadata)
}

fn load_sparse_entries(
    dir: &Path,
) -> Result<Vec<SparseSnapshotEntry>, ExportError> {
    let path = dir.join("sparse_entries.bin");
    require_file(&path, "sparse_entries.bin")?;
    let file = std::fs::File::open(&path)?;
    let reader = std::io::BufReader::new(file);
    bincode::deserialize_from(reader)
        .map_err(|e| ExportError::Deserialize(e.to_string()))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // SemanticKeyRecord round-trip tests now live in descriptors.rs.

    #[test]
    fn normalize_strategy_sums_positive() {
        let probs = normalize_strategy_sums(&[100, 200, 300]);
        let sum: f64 = probs.iter().map(|&p| f64::from(p)).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!((probs[0] - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_strategy_sums_zero_mass() {
        let probs = normalize_strategy_sums(&[0, 0, 0]);
        for &p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn normalize_strategy_sums_empty() {
        let probs = normalize_strategy_sums(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn mp_semantic_row_key_fingerprint_deterministic() {
        let key = MpInfosetKey {
            seat: 0,
            bucket: 5,
            history_hi: 0x1234,
            history_lo: 0x5678,
            history_hash: 0xAAAA,
            history_len: 4,
        };
        let fp1 = mp_semantic_row_key_fingerprint(&key);
        let fp2 = mp_semantic_row_key_fingerprint(&key);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn mp_semantic_row_key_fingerprint_varies_with_seat() {
        let key1 = MpInfosetKey {
            seat: 0,
            bucket: 5,
            history_hi: 0,
            history_lo: 0,
            history_hash: 0,
            history_len: 1,
        };
        let key2 = MpInfosetKey {
            seat: 1,
            ..key1
        };
        assert_ne!(
            mp_semantic_row_key_fingerprint(&key1),
            mp_semantic_row_key_fingerprint(&key2)
        );
    }

    #[test]
    fn mp_semantic_row_key_fingerprint_varies_with_history() {
        let key1 = MpInfosetKey {
            seat: 0,
            bucket: 5,
            history_hi: 0x1111,
            history_lo: 0x2222,
            history_hash: 0xAAAA,
            history_len: 4,
        };
        let key2 = MpInfosetKey {
            history_hash: 0xBBBB,
            ..key1
        };
        assert_ne!(
            mp_semantic_row_key_fingerprint(&key1),
            mp_semantic_row_key_fingerprint(&key2)
        );
    }

    #[test]
    fn opaque_action_schema_fingerprint_deterministic() {
        let fp1 = opaque_action_schema_fingerprint(3);
        let fp2 = opaque_action_schema_fingerprint(3);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn opaque_action_schema_fingerprint_varies() {
        let fp1 = opaque_action_schema_fingerprint(3);
        let fp2 = opaque_action_schema_fingerprint(4);
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn build_lazy_game_metadata_populates_blinds() {
        let config = LazyExportConfig {
            num_players: 6,
            stack_depth: 100.0,
            bucket_counts: [169, 100, 50, 50],
            small_blind: 0.5,
            big_blind: 1.0,
        };
        let meta = build_lazy_game_metadata(&config);
        assert!(
            (meta.small_blind - 0.5).abs() < f64::EPSILON,
            "small_blind should be 0.5, got {}",
            meta.small_blind
        );
        assert!(
            (meta.big_blind - 1.0).abs() < f64::EPSILON,
            "big_blind should be 1.0, got {}",
            meta.big_blind
        );
    }

    #[test]
    fn build_lazy_game_metadata_zero_blinds_for_ante_only() {
        let config = LazyExportConfig {
            num_players: 6,
            stack_depth: 100.0,
            bucket_counts: [169, 100, 50, 50],
            small_blind: 0.0,
            big_blind: 0.0,
        };
        let meta = build_lazy_game_metadata(&config);
        assert!(
            meta.small_blind.abs() < f64::EPSILON,
            "ante-only game should have small_blind=0"
        );
        assert!(
            meta.big_blind.abs() < f64::EPSILON,
            "ante-only game should have big_blind=0"
        );
    }

    #[test]
    fn build_opaque_actions_correct_count() {
        let actions = build_opaque_actions(3);
        assert_eq!(actions.len(), 3);
        for (i, a) in actions.iter().enumerate() {
            assert_eq!(a.kind, ActionKind::Opaque);
            assert_eq!(a.source_action_index, i as u16);
            assert_eq!(a.amount_chips, 0);
            assert!(a.size_key.is_empty());
            assert!(a.label.is_empty());
            assert!(!a.is_aggressive);
        }
    }
}
