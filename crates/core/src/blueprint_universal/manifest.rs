//! Manifest types for `blueprint.json`.
//!
//! Unknown JSON fields are silently ignored on deserialization (no
//! `deny_unknown_fields`). Bundle rejection is driven by `required_features`
//! checks, version compatibility, and length/checksum validation.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Top-level manifest for a universal dense blueprint bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub format_name: String,
    pub format_version: u16,
    pub compat_min_reader: u16,
    pub created_at: String,
    pub producer: String,
    pub producer_git: String,
    pub required_features: Vec<String>,
    pub optional_features: Vec<String>,
    /// Stable 64-bit hash identifying the game configuration.
    pub game_fingerprint: u64,
    /// Source config fingerprint, present when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_config_fingerprint: Option<u64>,
    pub game: GameMetadata,
    pub training: TrainingMetadata,
    pub strategy: StrategyMetadata,
    pub layout: LayoutMetadata,
    pub actions: ActionsMetadata,
    pub buckets: BucketsMetadata,
    pub compatibility: CompatibilityMetadata,
    pub files: BTreeMap<String, FileEntry>,
}

/// Game identity metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameMetadata {
    pub game_kind: String,
    pub num_players: u8,
    /// Ordered seat descriptors.
    pub seats: Vec<SeatDescriptor>,
    /// Seat index of the button/dealer.
    pub button_seat: u8,
    pub small_blind: f64,
    pub big_blind: f64,
    /// Per-seat antes (empty if no antes).
    pub antes: Vec<f64>,
    /// Per-seat straddles (empty if no straddles).
    pub straddles: Vec<f64>,
    /// Unit for stack values, e.g. `"chips"` or `"big_blinds"`.
    pub stack_units: String,
    /// Rake configuration, present even when zero.
    pub rake: RakeConfig,
    /// Maximum number of players on the flop when a preflop cap is configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_flop_players: Option<u8>,
}

/// Descriptor for a single seat at the table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SeatDescriptor {
    pub seat_id: u8,
    pub label: String,
    pub blind_ante_role: String,
    pub starting_stack: f64,
}

/// Rake rate and cap configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RakeConfig {
    pub rate: f64,
    pub cap: f64,
}

/// Training provenance metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetadata {
    pub source_backend: String,
    pub unit_kind: String,
    pub units_completed: u64,
    pub elapsed_minutes: f64,
    /// The strategy representation stored in the bundle.
    pub strategy_unit: String,
    /// CLI command used to produce the bundle, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    /// Path to the source config file, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_path: Option<String>,
    /// Fingerprint of the source config, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_fingerprint: Option<u64>,
}

/// Strategy payload metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetadata {
    pub normalization_tolerance: f64,
}

/// Layout metadata for row/action/probability counts and ordering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutMetadata {
    pub row_count: u64,
    pub action_descriptor_count: u64,
    pub probability_count: u64,
    pub row_sort_order: String,
    /// Allowed namespaces in the row file.
    pub row_namespace: Vec<String>,
    pub missing_row_policy: String,
}

/// Action abstraction metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionsMetadata {
    pub action_abstraction_fingerprint: u64,
}

/// Bucket abstraction metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketsMetadata {
    pub bucket_mode: String,
    pub bucket_semantic_fingerprint: u64,
    /// Per-street bucket counts (e.g. "preflop" -> 169).
    pub street_bucket_counts: BTreeMap<String, u64>,
    /// References to bucket files (embedded or external).
    pub bucket_files: Vec<BucketFileRef>,
    /// Version of the bucket generator that produced the buckets.
    pub bucket_generator_version: String,
    /// Per-flop bucket configuration, when used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_flop_bucket_config: Option<PerFlopBucketConfig>,
}

/// Reference to a bucket file with integrity metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BucketFileRef {
    pub name: String,
    pub path: String,
    pub byte_size: u64,
    pub sha256: String,
}

/// Per-flop bucket mode and canonicalization parameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PerFlopBucketConfig {
    pub mode: String,
    pub canonicalization_params: BTreeMap<String, String>,
}

/// Compatibility and legacy behavior metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityMetadata {
    pub legacy_fallback: bool,
    pub missing_row_policy: String,
}

/// Metadata for a single file in the bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub size: u64,
    pub sha256: String,
}
