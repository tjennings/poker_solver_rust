use crate::datagen::storage::{NUM_COMBOS, TrainingRecord, record_size};
use crate::model::network::INPUT_SIZE;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

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
    pub by_boundary_ordinal: BTreeMap<String, u64>,
    #[serde(default)]
    pub by_allin_proximity: BTreeMap<String, u64>,
    #[serde(default)]
    pub by_board_texture: BTreeMap<String, u64>,
    #[serde(default)]
    pub by_range_entropy: BTreeMap<String, u64>,
    #[serde(default)]
    pub by_range_source: BTreeMap<String, u64>,
    #[serde(default)]
    pub by_target_source: BTreeMap<String, u64>,
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

    pub fn add_turn_boundary_shard(
        &mut self,
        dataset_dir: impl AsRef<Path>,
        shard_path: impl AsRef<Path>,
        records: u64,
        target_source: Option<TargetSource>,
    ) -> Result<(), ManifestValidationError> {
        let path = manifest_shard_path(dataset_dir, shard_path)?;
        self.shards.push(ShardMetadata {
            path,
            records,
            board_size: TURN_BOUNDARY_BOARD_SIZE,
            record_size_bytes: record_size(TURN_BOUNDARY_BOARD_SIZE as usize),
            target_source,
        });
        self.coverage.total_records = self.shards.iter().map(|shard| shard.records).sum();
        Ok(())
    }
}

impl CoverageSummary {
    pub fn record_turn_boundary(
        &mut self,
        rec: &TrainingRecord,
        target_source: TargetSource,
        range_source: &str,
        raise_depth: &str,
        boundary_ordinal: &str,
    ) {
        self.total_records += 1;
        increment(&mut self.by_pot_bucket, pot_bucket(rec.pot));
        increment(&mut self.by_stack_bucket, stack_bucket(rec.effective_stack));
        increment(
            &mut self.by_spr_bucket,
            spr_bucket(rec.effective_stack, rec.pot),
        );
        increment(
            &mut self.by_allin_proximity,
            allin_proximity_bucket(rec.effective_stack, rec.pot),
        );
        increment(&mut self.by_raise_depth, raise_depth.to_string());
        increment(&mut self.by_boundary_ordinal, boundary_ordinal.to_string());
        increment(&mut self.by_board_texture, board_texture_bucket(&rec.board));
        increment(&mut self.by_range_entropy, range_entropy_bucket(rec));
        increment(&mut self.by_range_source, range_source.to_string());
        increment(
            &mut self.by_target_source,
            target_source.as_str().to_string(),
        );
    }
}

impl TargetSource {
    pub fn as_str(self) -> &'static str {
        match self {
            TargetSource::RiverNet => "river_net",
            TargetSource::ExactRiver => "exact_river",
            TargetSource::Mixed => "mixed",
        }
    }
}

fn increment(map: &mut BTreeMap<String, u64>, key: String) {
    *map.entry(key).or_insert(0) += 1;
}

fn pot_bucket(pot: f32) -> String {
    match pot {
        p if p < 10.0 => "pot_lt_10",
        p if p < 25.0 => "pot_10_25",
        p if p < 50.0 => "pot_25_50",
        p if p < 100.0 => "pot_50_100",
        p if p < 200.0 => "pot_100_200",
        _ => "pot_200_plus",
    }
    .to_string()
}

fn stack_bucket(stack: f32) -> String {
    match stack {
        s if s <= 0.0 => "stack_0",
        s if s < 25.0 => "stack_1_25",
        s if s < 50.0 => "stack_25_50",
        s if s < 100.0 => "stack_50_100",
        s if s < 200.0 => "stack_100_200",
        _ => "stack_200_plus",
    }
    .to_string()
}

fn spr_bucket(stack: f32, pot: f32) -> String {
    let spr = if pot > 0.0 {
        stack / pot
    } else {
        f32::INFINITY
    };
    match spr {
        s if s < 0.5 => "spr_lt_0_5",
        s if s < 1.5 => "spr_0_5_1_5",
        s if s < 4.0 => "spr_1_5_4",
        s if s < 8.0 => "spr_4_8",
        s if s < 20.0 => "spr_8_20",
        _ => "spr_20_plus",
    }
    .to_string()
}

fn allin_proximity_bucket(stack: f32, pot: f32) -> String {
    let ratio = if pot > 0.0 {
        stack / pot
    } else {
        f32::INFINITY
    };
    match ratio {
        r if r <= 0.25 => "near_allin_le_0_25p",
        r if r <= 0.5 => "near_allin_0_25_0_5p",
        r if r <= 1.0 => "near_allin_0_5_1p",
        _ => "not_near_allin",
    }
    .to_string()
}

fn board_texture_bucket(board: &[u8]) -> String {
    let mut rank_counts = [0_u8; 13];
    let mut suit_counts = [0_u8; 4];
    for &card in board {
        rank_counts[(card / 4) as usize] += 1;
        suit_counts[(card % 4) as usize] += 1;
    }

    let paired = match rank_counts.iter().copied().max().unwrap_or(0) {
        4 => "quads",
        3 => "trips",
        2 => "paired",
        _ => "unpaired",
    };
    let suit = match suit_counts.iter().copied().max().unwrap_or(0) {
        4 => "monotone",
        3 => "three_flush",
        2 => "two_tone",
        _ => "rainbow",
    };
    let connected = if has_four_card_run(&rank_counts) {
        "connected"
    } else {
        "disconnected"
    };
    format!("{paired}_{suit}_{connected}")
}

fn has_four_card_run(rank_counts: &[u8; 13]) -> bool {
    let mut present = [false; 14];
    for (rank, &count) in rank_counts.iter().enumerate() {
        if count > 0 {
            present[rank] = true;
            if rank == 12 {
                present[13] = true;
            }
        }
    }
    present.windows(4).any(|window| window.iter().all(|&v| v))
}

fn range_entropy_bucket(rec: &TrainingRecord) -> String {
    let range = if rec.player == 0 {
        &rec.oop_range
    } else {
        &rec.ip_range
    };
    let total: f64 = range.iter().map(|&v| f64::from(v.max(0.0))).sum();
    if total <= 0.0 {
        return "entropy_empty".to_string();
    }

    let mut entropy = 0.0_f64;
    let mut support = 0_u32;
    for &value in range {
        let p = f64::from(value.max(0.0)) / total;
        if p > 0.0 {
            entropy -= p * p.ln();
            support += 1;
        }
    }
    if support <= 1 {
        return "entropy_zero".to_string();
    }
    let normalized = entropy / f64::from(support).ln();
    match normalized {
        h if h < 0.35 => "entropy_low",
        h if h < 0.70 => "entropy_medium",
        _ => "entropy_high",
    }
    .to_string()
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
    #[error("shard path {path} is not inside dataset directory {dataset_dir}")]
    ShardOutsideDataset { dataset_dir: String, path: String },
}

/// Convert a shard path to the relative path stored in a manifest.
pub fn manifest_shard_path(
    dataset_dir: impl AsRef<Path>,
    shard_path: impl AsRef<Path>,
) -> Result<String, ManifestValidationError> {
    let dataset_dir = dataset_dir.as_ref();
    let shard_path = shard_path.as_ref();
    let relative: PathBuf = if shard_path.is_absolute() {
        let stripped = shard_path.strip_prefix(dataset_dir).map_err(|_| {
            ManifestValidationError::ShardOutsideDataset {
                dataset_dir: dataset_dir.display().to_string(),
                path: shard_path.display().to_string(),
            }
        })?;
        stripped.to_path_buf()
    } else {
        shard_path.to_path_buf()
    };
    Ok(relative.to_string_lossy().replace('\\', "/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn sample_record() -> TrainingRecord {
        let mut rec = TrainingRecord {
            board: vec![0, 4, 8, 12],
            pot: 20.0,
            effective_stack: 10.0,
            player: 0,
            game_value: 0.0,
            oop_range: [0.0; NUM_COMBOS],
            ip_range: [0.0; NUM_COMBOS],
            cfvs: [0.0; NUM_COMBOS],
            valid_mask: [0; NUM_COMBOS],
        };
        rec.oop_range[0] = 1.0;
        rec
    }

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
    fn coverage_records_turn_boundary_strata() {
        let mut coverage = CoverageSummary::default();
        coverage.record_turn_boundary(
            &sample_record(),
            TargetSource::RiverNet,
            "rsp",
            "4bet_plus",
            "boundary_03",
        );

        assert_eq!(coverage.total_records, 1);
        assert_eq!(coverage.by_pot_bucket.get("pot_10_25"), Some(&1));
        assert_eq!(coverage.by_stack_bucket.get("stack_1_25"), Some(&1));
        assert_eq!(coverage.by_spr_bucket.get("spr_0_5_1_5"), Some(&1));
        assert_eq!(
            coverage.by_allin_proximity.get("near_allin_0_25_0_5p"),
            Some(&1)
        );
        assert_eq!(coverage.by_raise_depth.get("4bet_plus"), Some(&1));
        assert_eq!(coverage.by_boundary_ordinal.get("boundary_03"), Some(&1));
        assert_eq!(coverage.by_range_entropy.get("entropy_zero"), Some(&1));
        assert_eq!(coverage.by_range_source.get("rsp"), Some(&1));
        assert_eq!(coverage.by_target_source.get("river_net"), Some(&1));
        assert_eq!(
            coverage.by_board_texture.get("unpaired_monotone_connected"),
            Some(&1)
        );
    }

    #[test]
    fn validation_rejects_river_board_size() {
        let mut manifest = DatasetManifest::new_turn_boundary(TargetSource::RiverNet);
        manifest.record_schema.board_size = 5;

        let err = manifest.validate_turn_boundary().unwrap_err();
        assert_eq!(err, ManifestValidationError::BoardSize { actual: 5 });
    }

    #[test]
    fn add_shard_records_relative_manifest_path_and_total() {
        let dir = tempfile::tempdir().unwrap();
        let shard0 = dir.path().join("turn.bin");
        let shard1 = dir.path().join("turn_00001.bin");
        let mut manifest = DatasetManifest::new_turn_boundary(TargetSource::RiverNet);

        manifest
            .add_turn_boundary_shard(dir.path(), &shard0, 128, None)
            .unwrap();
        manifest
            .add_turn_boundary_shard(dir.path(), &shard1, 64, Some(TargetSource::ExactRiver))
            .unwrap();

        assert_eq!(manifest.shards[0].path, "turn.bin");
        assert_eq!(manifest.shards[1].path, "turn_00001.bin");
        assert_eq!(
            manifest.shards[1].target_source,
            Some(TargetSource::ExactRiver)
        );
        assert_eq!(manifest.coverage.total_records, 192);
        manifest.validate_turn_boundary().unwrap();
    }

    #[test]
    fn absolute_shard_outside_dataset_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = DatasetManifest::new_turn_boundary(TargetSource::RiverNet);

        let err = manifest
            .add_turn_boundary_shard(dir.path(), "/tmp/outside.bin", 1, None)
            .unwrap_err();

        assert!(matches!(
            err,
            ManifestValidationError::ShardOutsideDataset { .. }
        ));
    }
}
