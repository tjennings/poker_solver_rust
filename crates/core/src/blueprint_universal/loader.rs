//! Unified bundle detection and loading for legacy HU and universal bundles.
//!
//! ## Bundle resolution order
//!
//! Both legacy and universal bundles may nest their content inside the bundle
//! root, `final/`, or the latest `snapshot_NNNN/` subdirectory. The resolver
//! checks these locations in order:
//!
//! 1. `dir/final/` — if it contains the sentinel file (`blueprint.json` for
//!    universal, `strategy.bin` for legacy).
//! 2. `dir/` (the root) — if the sentinel is directly present.
//! 3. `dir/snapshot_NNNN/` — the lexicographically last snapshot subdirectory
//!    that contains the sentinel.
//!
//! This matches the Explorer's existing `strategy.bin` discovery order
//! (final -> root -> latest snapshot) and extends it to universal bundles.

#![allow(clippy::cast_possible_truncation)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::bundle::BundleReader;
use super::descriptors::ActionDescriptor;
use super::manifest::Manifest;

use crate::blueprint_v2::bundle::{load_config, BlueprintV2Strategy};
use crate::blueprint_v2::config::BlueprintV2Config;
use crate::blueprint_v2::game_tree::{GameNode, GameTree, TreeAction};

/// The four recognized bundle kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleKind {
    /// Legacy HU bundle (`config.yaml` + `strategy.bin`, no `blueprint.json`).
    LegacyHu,
    /// Universal HU bundle (`blueprint.json` with `hu_arena` namespace).
    UniversalHu,
    /// Universal MP eager bundle (`blueprint.json` with `mp_arena` namespace).
    UniversalMpEager,
    /// Universal MP lazy bundle (`blueprint.json` with `mp_semantic` namespace).
    UniversalMpLazy,
}

impl std::fmt::Display for BundleKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LegacyHu => write!(f, "legacy_hu"),
            Self::UniversalHu => write!(f, "universal_hu"),
            Self::UniversalMpEager => write!(f, "universal_mp_eager"),
            Self::UniversalMpLazy => write!(f, "universal_mp_lazy"),
        }
    }
}

/// Errors from bundle detection and loading.
#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    /// The directory contains neither `blueprint.json` nor `config.yaml`.
    #[error("not a bundle: {path} contains neither blueprint.json nor config.yaml")]
    NotABundle { path: String },

    /// The `blueprint.json` manifest could not be parsed.
    #[error("failed to parse blueprint.json in {path}: {detail}")]
    ManifestParse { path: String, detail: String },

    /// The manifest declares an unrecognized `source_backend`.
    #[error("unknown source_backend \"{backend}\" in {path}")]
    UnknownSourceBackend { backend: String, path: String },

    /// A required payload file declared in the manifest is missing.
    #[error("missing payload file \"{name}\" in {path}")]
    MissingPayloadFile { name: String, path: String },

    /// The manifest's `row_namespace` does not match the `source_backend`.
    #[error("namespace mismatch in {path}: source_backend \"{backend}\" expects namespace {expected}, got {actual:?}")]
    NamespaceMismatch {
        path: String,
        backend: String,
        expected: String,
        actual: Vec<String>,
    },

    /// A required feature is missing from the manifest's `required_features`.
    #[error("missing required feature \"{feature}\" in {path} for source_backend \"{backend}\"")]
    MissingRequiredFeature {
        feature: String,
        backend: String,
        path: String,
    },

    /// The universal format reader returned an error.
    #[error("format error: {0}")]
    Format(#[from] super::error::FormatError),

    /// An I/O error during legacy bundle loading.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Internal data for a loaded legacy HU bundle.
#[derive(Debug)]
pub(crate) struct LegacyHuData {
    config: BlueprintV2Config,
    strategy: BlueprintV2Strategy,
    tree: GameTree,
    decision_index_map: Vec<u32>,
    source_backend: String,
}

/// Internal data for a loaded universal bundle.
#[derive(Debug)]
pub(crate) struct UniversalData {
    manifest: Manifest,
    reader: BundleReader,
}

/// A loaded bundle ready for strategy queries.
///
/// Uses an enum rather than a trait because there are exactly four closed
/// variants and a single consumer holds one loaded bundle. An enum avoids
/// dynamic dispatch ceremony and matches the project's preference for
/// shared functions/enums over speculative traits (Phases 3-4 precedent).
#[derive(Debug)]
#[allow(private_interfaces)]
pub enum LoadedBundle {
    /// Legacy HU bundle loaded from `config.yaml` + `strategy.bin`.
    LegacyHu(Box<LegacyHuData>),
    /// Universal HU bundle with lookup index.
    UniversalHu {
        data: Box<UniversalData>,
        /// Maps `(source_node_idx, global_bucket)` -> row index.
        index: HashMap<(u32, u32), usize>,
    },
    /// Universal MP eager bundle with lookup index.
    UniversalMpEager {
        data: Box<UniversalData>,
        /// Maps `(source_node_idx, seat, global_bucket)` -> row index.
        index: HashMap<(u32, u8, u32), usize>,
    },
    /// Universal MP lazy bundle with lookup index.
    UniversalMpLazy {
        data: Box<UniversalData>,
        /// Maps `(seat, street, local_bucket, history_hash, history_len)`
        /// -> row index.
        index: HashMap<(u8, u8, u16, u64, u16), usize>,
    },
}

/// Semantic identity key for an MP lazy infoset query.
///
/// Groups the five identity fields into a single struct to keep the
/// `query_mp_lazy` parameter count within the 4-parameter guideline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MpLazyKey {
    /// The seat (player position) index.
    pub seat: u8,
    /// Street index (0 = preflop, 1 = flop, 2 = turn, 3 = river).
    pub street: u8,
    /// The local bucket for this street.
    pub local_bucket: u16,
    /// Hash of the action history prefix.
    pub history_hash: u64,
    /// Length of the action history prefix.
    pub history_len: u16,
}

/// A borrowed view of an infoset's action probabilities and descriptors.
#[derive(Debug)]
pub struct InfosetView<'a> {
    /// The action probabilities for this infoset (one per action).
    pub probs: &'a [f32],
    /// The action descriptors for this infoset.
    pub actions: Vec<ActionDescriptor>,
}

// ── LoadedBundle query API ──────────────────────────────────────────

impl LoadedBundle {
    /// The kind of bundle that was loaded.
    #[must_use]
    pub fn kind(&self) -> BundleKind {
        match self {
            Self::LegacyHu(..) => BundleKind::LegacyHu,
            Self::UniversalHu { .. } => BundleKind::UniversalHu,
            Self::UniversalMpEager { .. } => BundleKind::UniversalMpEager,
            Self::UniversalMpLazy { .. } => BundleKind::UniversalMpLazy,
        }
    }

    /// The `source_backend` string (e.g. `"hu_dense"`, `"mp_eager_dense"`).
    #[must_use]
    pub fn source_backend(&self) -> &str {
        match self {
            Self::LegacyHu(d) => &d.source_backend,
            Self::UniversalHu { data, .. }
            | Self::UniversalMpEager { data, .. }
            | Self::UniversalMpLazy { data, .. } => {
                &data.manifest.training.source_backend
            }
        }
    }

    /// Number of players in the game.
    #[must_use]
    pub fn num_players(&self) -> u8 {
        match self {
            Self::LegacyHu(d) => d.config.game.players,
            Self::UniversalHu { data, .. }
            | Self::UniversalMpEager { data, .. }
            | Self::UniversalMpLazy { data, .. } => {
                data.manifest.game.num_players
            }
        }
    }

    /// The manifest for universal bundles, `None` for legacy.
    #[must_use]
    pub fn manifest(&self) -> Option<&Manifest> {
        match self {
            Self::LegacyHu(..) => None,
            Self::UniversalHu { data, .. }
            | Self::UniversalMpEager { data, .. }
            | Self::UniversalMpLazy { data, .. } => Some(&data.manifest),
        }
    }

    /// Training iterations completed.
    #[must_use]
    pub fn iterations(&self) -> u64 {
        match self {
            Self::LegacyHu(d) => d.strategy.iterations,
            Self::UniversalHu { data, .. }
            | Self::UniversalMpEager { data, .. }
            | Self::UniversalMpLazy { data, .. } => {
                data.manifest.training.units_completed
            }
        }
    }

    /// Query a HU infoset by arena node index and bucket.
    ///
    /// Works for both `LegacyHu` and `UniversalHu` bundles.
    /// Returns `None` if the infoset is not found.
    #[must_use]
    pub fn query_hu(
        &self,
        arena_node_idx: u32,
        bucket: u16,
    ) -> Option<InfosetView<'_>> {
        match self {
            Self::LegacyHu(d) => query_legacy_hu(d, arena_node_idx, bucket),
            Self::UniversalHu { data, index } => {
                let row_idx = *index.get(&(
                    arena_node_idx,
                    u32::from(bucket),
                ))?;
                build_view_from_reader(&data.reader, row_idx)
            }
            _ => None,
        }
    }

    /// Query an MP eager infoset by arena node index, seat, and bucket.
    ///
    /// Returns `None` if the infoset is not found.
    #[must_use]
    pub fn query_mp_eager(
        &self,
        arena_node_idx: u32,
        seat: u8,
        bucket: u16,
    ) -> Option<InfosetView<'_>> {
        match self {
            Self::UniversalMpEager { data, index } => {
                let key = (arena_node_idx, seat, u32::from(bucket));
                let row_idx = *index.get(&key)?;
                build_view_from_reader(&data.reader, row_idx)
            }
            _ => None,
        }
    }

    /// Query an MP lazy infoset by semantic identity.
    ///
    /// Returns `None` if the infoset is not found.
    #[must_use]
    pub fn query_mp_lazy(
        &self,
        key: &MpLazyKey,
    ) -> Option<InfosetView<'_>> {
        match self {
            Self::UniversalMpLazy { data, index } => {
                let tuple = (
                    key.seat,
                    key.street,
                    key.local_bucket,
                    key.history_hash,
                    key.history_len,
                );
                let row_idx = *index.get(&tuple)?;
                build_view_from_reader(&data.reader, row_idx)
            }
            _ => None,
        }
    }
}

// ── Query helpers ───────────────────────────────────────────────────

/// Query a legacy HU bundle by arena node index and bucket.
fn query_legacy_hu(
    d: &LegacyHuData,
    arena_node_idx: u32,
    bucket: u16,
) -> Option<InfosetView<'_>> {
    let idx = arena_node_idx as usize;
    if idx >= d.decision_index_map.len() {
        return None;
    }
    let decision_idx = d.decision_index_map[idx];
    if decision_idx == u32::MAX {
        return None;
    }
    let probs = d.strategy.get_action_probs(decision_idx as usize, bucket);
    if probs.is_empty() {
        return None;
    }
    let actions = synthesize_hu_actions(&d.tree, arena_node_idx);
    Some(InfosetView { probs, actions })
}

/// Synthesize action descriptors from the game tree node's actions
/// so legacy HU and universal HU return comparable shapes.
fn synthesize_hu_actions(
    tree: &GameTree,
    arena_node_idx: u32,
) -> Vec<ActionDescriptor> {
    let node = &tree.nodes[arena_node_idx as usize];
    match node {
        GameNode::Decision { actions, .. } => actions
            .iter()
            .enumerate()
            .map(|(i, a)| tree_action_to_descriptor(a, i as u16))
            .collect(),
        _ => Vec::new(),
    }
}

/// Convert a `TreeAction` to an `ActionDescriptor`.
fn tree_action_to_descriptor(
    action: &TreeAction,
    source_idx: u16,
) -> ActionDescriptor {
    use super::descriptors::ActionKind;
    let (kind, amount, aggressive) = match action {
        TreeAction::Fold => (ActionKind::Fold, 0, false),
        TreeAction::Check => (ActionKind::Check, 0, false),
        TreeAction::Call => (ActionKind::Call, 0, false),
        TreeAction::Bet(amt) => {
            #[allow(clippy::cast_sign_loss)]
            let chips = *amt as u32;
            (ActionKind::Bet, chips, true)
        }
        TreeAction::Raise(amt) => {
            #[allow(clippy::cast_sign_loss)]
            let chips = *amt as u32;
            (ActionKind::Raise, chips, true)
        }
        TreeAction::AllIn => (ActionKind::AllInBetRaise, 0, true),
    };
    ActionDescriptor {
        kind,
        amount_chips: amount,
        size_key: String::new(),
        label: String::new(),
        is_aggressive: aggressive,
        source_action_index: source_idx,
    }
}

/// Build an `InfosetView` from a `BundleReader` row index.
fn build_view_from_reader(
    reader: &BundleReader,
    row_idx: usize,
) -> Option<InfosetView<'_>> {
    let row = reader.row(row_idx)?;
    let count = usize::from(row.action_count);
    let prob_start = row.prob_offset as usize;
    let probs = reader.prob_slice(prob_start, count)?;
    let action_start = row.action_offset as usize;
    let mut actions = Vec::with_capacity(count);
    for i in 0..count {
        actions.push(reader.action(action_start + i)?.clone());
    }
    Some(InfosetView { probs, actions })
}

// ── Detection ───────────────────────────────────────────────────────

/// Detect the bundle kind from a directory without loading binary payloads.
///
/// Resolves nesting: checks `final/`, then the root, then the latest
/// `snapshot_NNNN/` subdirectory for the sentinel file.
///
/// # Errors
///
/// Returns a precise [`LoaderError`] variant for each failure mode.
pub fn detect_bundle_kind(dir: &Path) -> Result<BundleKind, LoaderError> {
    let (effective_dir, sentinel) = resolve_bundle_dir(dir)?;
    match sentinel {
        SentinelKind::BlueprintJson => {
            let (kind, _manifest) =
                classify_universal_manifest(&effective_dir)?;
            Ok(kind)
        }
        SentinelKind::ConfigYaml => Ok(BundleKind::LegacyHu),
    }
}

/// Load a bundle from a directory, returning a [`LoadedBundle`] ready
/// for queries.
///
/// # Errors
///
/// Returns a precise [`LoaderError`] variant for each failure mode.
pub fn load_bundle(dir: &Path) -> Result<LoadedBundle, LoaderError> {
    let (effective_dir, sentinel) = resolve_bundle_dir(dir)?;
    match sentinel {
        SentinelKind::ConfigYaml => load_legacy_hu(dir, &effective_dir),
        SentinelKind::BlueprintJson => {
            let (kind, manifest) =
                classify_universal_manifest(&effective_dir)?;
            load_universal(&effective_dir, kind, manifest)
        }
    }
}

// ── Resolution ──────────────────────────────────────────────────────

/// What sentinel file was found during directory resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SentinelKind {
    /// Found `blueprint.json` (universal bundle).
    BlueprintJson,
    /// Found `config.yaml` (legacy HU bundle).
    ConfigYaml,
}

/// Resolve the effective bundle directory, handling nesting inside
/// `final/` or `snapshot_NNNN/`.
///
/// Priority: universal (`blueprint.json`) over legacy (`config.yaml`).
/// Nesting order: `final/` -> root -> latest `snapshot_NNNN/`.
fn resolve_bundle_dir(
    dir: &Path,
) -> Result<(PathBuf, SentinelKind), LoaderError> {
    if let Some(d) = find_sentinel(dir, "blueprint.json") {
        return Ok((d, SentinelKind::BlueprintJson));
    }
    if dir.join("config.yaml").exists() {
        let strat_dir = find_sentinel(dir, "strategy.bin")
            .unwrap_or_else(|| dir.to_path_buf());
        return Ok((strat_dir, SentinelKind::ConfigYaml));
    }
    Err(LoaderError::NotABundle {
        path: dir.display().to_string(),
    })
}

/// Find the directory containing a sentinel file, checking
/// `final/` -> root -> latest `snapshot_NNNN/`.
fn find_sentinel(dir: &Path, sentinel: &str) -> Option<PathBuf> {
    let final_dir = dir.join("final");
    if final_dir.join(sentinel).exists() {
        return Some(final_dir);
    }
    if dir.join(sentinel).exists() {
        return Some(dir.to_path_buf());
    }
    find_latest_snapshot_with(dir, sentinel)
}

/// Find the latest `snapshot_NNNN/` subdirectory containing a file.
fn find_latest_snapshot_with(
    dir: &Path,
    file_name: &str,
) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut snapshots: Vec<_> = entries
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with("snapshot_"))
        })
        .filter(|e| e.path().join(file_name).exists())
        .collect();
    snapshots.sort_by_key(std::fs::DirEntry::file_name);
    snapshots.last().map(std::fs::DirEntry::path)
}

// ── Classification ──────────────────────────────────────────────────

/// Map `source_backend` to its expected `(namespace, BundleKind)`.
///
/// Returns `None` for unrecognized backends. This is the single source
/// of truth for the backend -> namespace/kind mapping, used by
/// `classify_universal_manifest`.
fn backend_metadata(
    backend: &str,
) -> Option<(&'static str, BundleKind)> {
    match backend {
        "hu_dense" | "hu_sparse_projected" => {
            Some(("hu_arena", BundleKind::UniversalHu))
        }
        "mp_eager_dense" => {
            Some(("mp_arena", BundleKind::UniversalMpEager))
        }
        "mp_lazy_sparse_projected" => {
            Some(("mp_semantic", BundleKind::UniversalMpLazy))
        }
        _ => None,
    }
}

/// Map `source_backend` + cross-checks -> `(BundleKind, Manifest)`.
///
/// Returns the parsed manifest alongside the kind so callers can
/// reuse it without re-reading `blueprint.json`.
fn classify_universal_manifest(
    dir: &Path,
) -> Result<(BundleKind, Manifest), LoaderError> {
    let manifest = read_manifest_for_detection(dir)?;
    let backend = &manifest.training.source_backend;
    let path_str = dir.display().to_string();

    let (expected_ns, kind) =
        backend_metadata(backend).ok_or_else(|| {
            LoaderError::UnknownSourceBackend {
                backend: backend.clone(),
                path: path_str.clone(),
            }
        })?;

    if !manifest.layout.row_namespace.contains(&expected_ns.to_string())
    {
        return Err(LoaderError::NamespaceMismatch {
            path: path_str,
            backend: backend.clone(),
            expected: expected_ns.to_string(),
            actual: manifest.layout.row_namespace.clone(),
        });
    }

    if kind == BundleKind::UniversalMpLazy {
        let has_semantic = manifest
            .required_features
            .iter()
            .any(|f| f == "mp_semantic_rows_v1");
        if !has_semantic {
            return Err(LoaderError::MissingRequiredFeature {
                feature: "mp_semantic_rows_v1".to_string(),
                backend: backend.clone(),
                path: path_str,
            });
        }
    }

    check_payload_files_exist(dir, &manifest, &path_str)?;

    Ok((kind, manifest))
}

/// Read `blueprint.json` for detection (not full binary validation).
fn read_manifest_for_detection(
    dir: &Path,
) -> Result<Manifest, LoaderError> {
    let path = dir.join("blueprint.json");
    let text = std::fs::read_to_string(&path).map_err(|e| {
        LoaderError::ManifestParse {
            path: dir.display().to_string(),
            detail: e.to_string(),
        }
    })?;
    serde_json::from_str::<Manifest>(&text).map_err(|e| {
        LoaderError::ManifestParse {
            path: dir.display().to_string(),
            detail: e.to_string(),
        }
    })
}

/// Verify that all required binary payload files exist on disk.
fn check_payload_files_exist(
    dir: &Path,
    manifest: &Manifest,
    path_str: &str,
) -> Result<(), LoaderError> {
    for name in manifest.files.keys() {
        if !dir.join(name).exists() {
            return Err(LoaderError::MissingPayloadFile {
                name: name.clone(),
                path: path_str.to_string(),
            });
        }
    }
    Ok(())
}

// ── Loading ─────────────────────────────────────────────────────────

/// Load a legacy HU bundle.
///
/// `root_dir` has `config.yaml`; `strat_dir` has `strategy.bin`.
fn load_legacy_hu(
    root_dir: &Path,
    strat_dir: &Path,
) -> Result<LoadedBundle, LoaderError> {
    let config = load_config(root_dir)?;
    let strat_path = strat_dir.join("strategy.bin");
    let strategy = BlueprintV2Strategy::load(&strat_path)?;

    let aa = &config.action_abstraction;
    let tree = GameTree::build_with_options(
        config.game.stack_depth,
        config.game.small_blind,
        config.game.big_blind,
        &aa.preflop,
        &aa.flop,
        &aa.turn,
        &aa.river,
        config.game.allow_preflop_limp,
    );
    let decision_index_map = tree.decision_index_map();

    let source_backend = match config.training.storage_backend.as_str() {
        "sparse" | "lazy" => "hu_sparse_projected".to_string(),
        _ => "hu_dense".to_string(),
    };

    Ok(LoadedBundle::LegacyHu(Box::new(LegacyHuData {
        config,
        strategy,
        tree,
        decision_index_map,
        source_backend,
    })))
}

/// Load a universal bundle of any kind.
///
/// Accepts the pre-parsed `manifest` from `classify_universal_manifest`
/// so that `blueprint.json` is not re-read a second time.
fn load_universal(
    dir: &Path,
    kind: BundleKind,
    manifest: Manifest,
) -> Result<LoadedBundle, LoaderError> {
    let reader = BundleReader::open(dir)?;
    let data = Box::new(UniversalData { manifest, reader });
    match kind {
        BundleKind::UniversalHu => {
            let index = build_hu_index(&data.reader);
            Ok(LoadedBundle::UniversalHu { data, index })
        }
        BundleKind::UniversalMpEager => {
            let index = build_mp_eager_index(&data.reader);
            Ok(LoadedBundle::UniversalMpEager { data, index })
        }
        BundleKind::UniversalMpLazy => {
            let index = build_mp_lazy_index(&data.reader);
            Ok(LoadedBundle::UniversalMpLazy { data, index })
        }
        BundleKind::LegacyHu => unreachable!(),
    }
}

// ── Index builders ──────────────────────────────────────────────────

/// Build a lookup index for universal HU bundles.
fn build_hu_index(reader: &BundleReader) -> HashMap<(u32, u32), usize> {
    let mut index = HashMap::with_capacity(reader.row_count());
    for i in 0..reader.row_count() {
        if let Some(row) = reader.row(i) {
            index.insert((row.source_node_idx, row.global_bucket), i);
        }
    }
    index
}

/// Build a lookup index for universal MP eager bundles.
fn build_mp_eager_index(
    reader: &BundleReader,
) -> HashMap<(u32, u8, u32), usize> {
    let mut index = HashMap::with_capacity(reader.row_count());
    for i in 0..reader.row_count() {
        if let Some(row) = reader.row(i) {
            index.insert(
                (row.source_node_idx, row.seat, row.global_bucket),
                i,
            );
        }
    }
    index
}

/// Build a lookup index for universal MP lazy bundles.
fn build_mp_lazy_index(
    reader: &BundleReader,
) -> HashMap<(u8, u8, u16, u64, u16), usize> {
    let mut index = HashMap::with_capacity(reader.row_count());
    for i in 0..reader.row_count() {
        let (Some(row), Some(sem)) =
            (reader.row(i), reader.semantic_record(i))
        else {
            continue;
        };
        index.insert(
            (
                row.seat,
                row.street,
                row.local_bucket,
                sem.history_hash,
                sem.history_len,
            ),
            i,
        );
    }
    index
}
