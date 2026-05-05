use crate::datagen::storage::{record_size, NUM_COMBOS};
use crate::model::network::INPUT_SIZE;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::path::Path;

pub const TURN_BOUNDARY_SCHEMA_VERSION: u32 = 1;
pub const TURN_BOUNDARY_BOARD_SIZE: u8 = 4;

/// Dataset-level metadata written next to binary TrainingRecord shards.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub schema_version: u32,
    pub street: DatasetStreet,
    pub record_schema: RecordSchema,
    pub target_source: TargetSource,
    pub source: SourceMetadata,
    pub coverage: CoverageSummary,
    pub validation: ValidationSummary,
    pub shards: Vec<ShardMetadata>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetStreet {
    TurnBoundary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordSchema {
    pub format: String,
    pub board_size: u8,
    pub record_size_bytes: usize,
    pub input_size: usize,
    pub output_size: usize,
    pub normalization: ValueNormalization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueNormalization {
    ChipCfvOverPotPlusStack,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetSource {
    RiverNet,
    ExactRiver,
    Mixed,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SourceMetadata {
    pub generator_commit: Option<String>,
    pub config_hash: Option<String>,
    pub source_model_path: Option<String>,
    pub source_model_checksum: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CoverageSummary {
    #[serde(default)]
    pub total_records: u64,
    #[serde(default)]
    pub by_spr_bucket: BTreeMap<String, u64>,
    #[serde(default)]
    pub by_pot_bucket: BTreeMap<String, u64>,
    #[serde(default)]
    pub by_stack_bucket: BTreeMap<String, u64>,
    #[serde(default)]
    pub by_raise_depth: BTreeMap<String, u64>,
    #[serde(default)]
    pub by_board_texture: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ValidationSummary {
    pub sampled_records: Option<u64>,
    pub avg_exploitability_chips: Option<f64>,
    pub max_exploitability_chips: Option<f64>,
    pub max_target_ratio: Option<f64>,
    pub oracle_mae_chips: Option<f64>,
    pub oracle_weighted_mae_chips: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardMetadata {
    pub path: String,
    pub records: u64,
    pub board_size: u8,
    pub record_size_bytes: usize,
    pub target_source: Option<TargetSource>,
}

impl DatasetManifest {
    pub fn new_turn_boundary(target_source: TargetSource) -> Self {
        Self {
            schema_version: TURN_BOUNDARY_SCHEMA_VERSION,
            street: DatasetStreet::TurnBoundary,
            record_schema: RecordSchema::turn_boundary(),
            target_source,
            source: SourceMetadata::default(),
            coverage: CoverageSummary::default(),
            validation: ValidationSummary::default(),
            shards: Vec::new(),
        }
    }

    pub fn read_yaml(path: impl AsRef<Path>) -> Result<Self, ManifestIoError> {
        let file = File::open(path)?;
        Ok(serde_yaml::from_reader(file)?)
    }

    pub fn write_yaml(&self, path: impl AsRef<Path>) -> Result<(), ManifestIoError> {
        let file = File::create(path)?;
        Ok(serde_yaml::to_writer(file, self)?)
    }

    pub fn validate_turn_boundary(&self) -> Result<(), ManifestValidationError> {
        if self.schema_version != TURN_BOUNDARY_SCHEMA_VERSION {
            return Err(ManifestValidationError::SchemaVersion {
                actual: self.schema_version,
            });
        }
        if self.street != DatasetStreet::TurnBoundary {
            return Err(ManifestValidationError::Street);
        }
        self.record_schema.validate_turn_boundary()?;

        let expected_record_size = record_size(TURN_BOUNDARY_BOARD_SIZE as usize);
        for shard in &self.shards {
            if shard.board_size != TURN_BOUNDARY_BOARD_SIZE {
                return Err(ManifestValidationError::ShardBoardSize {
                    path: shard.path.clone(),
                    actual: shard.board_size,
                });
            }
            if shard.record_size_bytes != expected_record_size {
                return Err(ManifestValidationError::ShardRecordSize {
                    path: shard.path.clone(),
                    actual: shard.record_size_bytes,
                });
            }
        }

        Ok(())
    }
}

impl RecordSchema {
    pub fn turn_boundary() -> Self {
        Self {
            format: "cfvnet_training_record_v1".to_string(),
            board_size: TURN_BOUNDARY_BOARD_SIZE,
            record_size_bytes: record_size(TURN_BOUNDARY_BOARD_SIZE as usize),
            input_size: INPUT_SIZE,
            output_size: NUM_COMBOS,
            normalization: ValueNormalization::ChipCfvOverPotPlusStack,
        }
    }

    fn validate_turn_boundary(&self) -> Result<(), ManifestValidationError> {
        if self.board_size != TURN_BOUNDARY_BOARD_SIZE {
            return Err(ManifestValidationError::BoardSize {
                actual: self.board_size,
            });
        }
        let expected_record_size = record_size(TURN_BOUNDARY_BOARD_SIZE as usize);
        if self.record_size_bytes != expected_record_size {
            return Err(ManifestValidationError::RecordSize {
                actual: self.record_size_bytes,
            });
        }
        if self.input_size != INPUT_SIZE {
            return Err(ManifestValidationError::InputSize {
                actual: self.input_size,
            });
        }
        if self.output_size != NUM_COMBOS {
            return Err(ManifestValidationError::OutputSize {
                actual: self.output_size,
            });
        }
        if self.normalization != ValueNormalization::ChipCfvOverPotPlusStack {
            return Err(ManifestValidationError::Normalization);
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ManifestIoError {
    #[error("manifest I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("manifest YAML failed: {0}")]
    Yaml(#[from] serde_yaml::Error),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ManifestValidationError {
    #[error("expected schema_version=1, got {actual}")]
    SchemaVersion { actual: u32 },
    #[error("expected street=turn_boundary")]
    Street,
    #[error("expected board_size=4, got {actual}")]
    BoardSize { actual: u8 },
    #[error("expected record_size_bytes={}, got {actual}", record_size(TURN_BOUNDARY_BOARD_SIZE as usize))]
    RecordSize { actual: usize },
    #[error("expected input_size={INPUT_SIZE}, got {actual}")]
    InputSize { actual: usize },
    #[error("expected output_size={NUM_COMBOS}, got {actual}")]
    OutputSize { actual: usize },
    #[error("expected normalization=chip_cfv_over_pot_plus_stack")]
    Normalization,
    #[error("shard {path} expected board_size=4, got {actual}")]
    ShardBoardSize { path: String, actual: u8 },
    #[error("shard {path} has record_size_bytes={actual}")]
    ShardRecordSize { path: String, actual: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn turn_boundary_schema_matches_binary_record_contract() {
        let schema = RecordSchema::turn_boundary();
        assert_eq!(schema.board_size, 4);
        assert_eq!(schema.record_size_bytes, record_size(4));
        assert_eq!(schema.input_size, INPUT_SIZE);
        assert_eq!(schema.output_size, NUM_COMBOS);
        assert_eq!(
            schema.normalization,
            ValueNormalization::ChipCfvOverPotPlusStack
        );
    }

    #[test]
    fn manifest_round_trips_as_yaml() {
        let mut manifest = DatasetManifest::new_turn_boundary(TargetSource::RiverNet);
        manifest.source.generator_commit = Some("abc123".to_string());
        manifest.coverage.total_records = 256;
        manifest
            .coverage
            .by_raise_depth
            .insert("4bet_plus".to_string(), 12);
        manifest.shards.push(ShardMetadata {
            path: "turn_000001.bin".to_string(),
            records: 256,
            board_size: TURN_BOUNDARY_BOARD_SIZE,
            record_size_bytes: record_size(4),
            target_source: None,
        });

        let file = NamedTempFile::new().unwrap();
        manifest.write_yaml(file.path()).unwrap();
        let loaded = DatasetManifest::read_yaml(file.path()).unwrap();

        assert_eq!(loaded, manifest);
        loaded.validate_turn_boundary().unwrap();
    }

    #[test]
    fn validation_rejects_river_board_size() {
        let mut manifest = DatasetManifest::new_turn_boundary(TargetSource::RiverNet);
        manifest.record_schema.board_size = 5;

        let err = manifest.validate_turn_boundary().unwrap_err();
        assert_eq!(err, ManifestValidationError::BoardSize { actual: 5 });
    }
}
