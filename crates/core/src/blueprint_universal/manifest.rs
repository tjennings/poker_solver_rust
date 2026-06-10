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
    pub small_blind: f64,
    pub big_blind: f64,
}

/// Training provenance metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetadata {
    pub source_backend: String,
    pub unit_kind: String,
    pub units_completed: u64,
}

/// Strategy payload metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetadata {
    pub strategy_unit: String,
    pub normalization_tolerance: f64,
}

/// Layout metadata for row/action/probability counts and ordering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutMetadata {
    pub row_count: u64,
    pub action_descriptor_count: u64,
    pub probability_count: u64,
    pub row_sort_order: String,
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

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for constructing a [`Manifest`] with sensible defaults for testing.
pub struct ManifestBuilder {
    format_name: String,
    format_version: u16,
    compat_min_reader: u16,
    producer: String,
    producer_git: String,
    num_players: u8,
    row_count: u64,
    action_descriptor_count: u64,
    probability_count: u64,
    normalization_tolerance: f64,
    required_features: Vec<String>,
    optional_features: Vec<String>,
}

impl ManifestBuilder {
    #[must_use]
    pub fn format_name(mut self, v: String) -> Self {
        self.format_name = v;
        self
    }
    #[must_use]
    pub fn format_version(mut self, v: u16) -> Self {
        self.format_version = v;
        self
    }
    #[must_use]
    pub fn compat_min_reader(mut self, v: u16) -> Self {
        self.compat_min_reader = v;
        self
    }
    #[must_use]
    pub fn producer(mut self, v: String) -> Self {
        self.producer = v;
        self
    }
    #[must_use]
    pub fn producer_git(mut self, v: String) -> Self {
        self.producer_git = v;
        self
    }
    #[must_use]
    pub fn num_players(mut self, v: u8) -> Self {
        self.num_players = v;
        self
    }
    #[must_use]
    pub fn row_count(mut self, v: u64) -> Self {
        self.row_count = v;
        self
    }
    #[must_use]
    pub fn action_descriptor_count(mut self, v: u64) -> Self {
        self.action_descriptor_count = v;
        self
    }
    #[must_use]
    pub fn probability_count(mut self, v: u64) -> Self {
        self.probability_count = v;
        self
    }
    #[must_use]
    pub fn normalization_tolerance(mut self, v: f64) -> Self {
        self.normalization_tolerance = v;
        self
    }

    /// Build the manifest with file entries to be filled in by the writer.
    #[must_use]
    pub fn build(self) -> Manifest {
        Manifest {
            format_name: self.format_name,
            format_version: self.format_version,
            compat_min_reader: self.compat_min_reader,
            created_at: "2026-01-01T00:00:00Z".to_string(),
            producer: self.producer,
            producer_git: self.producer_git,
            required_features: self.required_features,
            optional_features: self.optional_features,
            game: GameMetadata {
                game_kind: "holdem_no_limit".to_string(),
                num_players: self.num_players,
                small_blind: 1.0,
                big_blind: 2.0,
            },
            training: TrainingMetadata {
                source_backend: "hu_dense".to_string(),
                unit_kind: "Iteration".to_string(),
                units_completed: 0,
            },
            strategy: StrategyMetadata {
                strategy_unit: "average_strategy".to_string(),
                normalization_tolerance: self.normalization_tolerance,
            },
            layout: LayoutMetadata {
                row_count: self.row_count,
                action_descriptor_count: self.action_descriptor_count,
                probability_count: self.probability_count,
                row_sort_order: "namespace_seat_street_node_bucket_fingerprint"
                    .to_string(),
                missing_row_policy: "reject".to_string(),
            },
            actions: ActionsMetadata {
                action_abstraction_fingerprint: 0,
            },
            buckets: BucketsMetadata {
                bucket_mode: "none".to_string(),
                bucket_semantic_fingerprint: 0,
            },
            compatibility: CompatibilityMetadata {
                legacy_fallback: false,
                missing_row_policy: "reject".to_string(),
            },
            files: BTreeMap::new(),
        }
    }
}

impl Manifest {
    /// Start building a manifest.
    #[must_use]
    pub fn builder() -> ManifestBuilder {
        ManifestBuilder {
            format_name: "dense_blueprint".to_string(),
            format_version: 1,
            compat_min_reader: 1,
            producer: String::new(),
            producer_git: "unknown".to_string(),
            num_players: 2,
            row_count: 0,
            action_descriptor_count: 0,
            probability_count: 0,
            normalization_tolerance: 1e-4,
            required_features: Vec::new(),
            optional_features: Vec::new(),
        }
    }
}
