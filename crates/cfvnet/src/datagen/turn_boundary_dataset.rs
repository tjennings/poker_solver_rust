use std::path::{Path, PathBuf};

use crate::datagen::domain::RecordWriter;
use crate::datagen::manifest::{
    DatasetManifest, FLOP_BOUNDARY_BOARD_SIZE, ManifestIoError, ManifestValidationError,
    SourceMetadata, TURN_BOUNDARY_BOARD_SIZE, TargetSource,
};
use crate::datagen::storage::{TrainingRecord, record_size};

/// Writes turn-boundary oracle records and their manifest as one dataset.
pub struct TurnBoundaryDatasetWriter {
    writer: RecordWriter,
    output_path: PathBuf,
    dataset_dir: PathBuf,
    manifest_path: PathBuf,
    manifest: DatasetManifest,
}

/// Writes flop-boundary oracle records and their manifest as one dataset.
pub struct FlopBoundaryDatasetWriter {
    writer: RecordWriter,
    output_path: PathBuf,
    dataset_dir: PathBuf,
    manifest_path: PathBuf,
    manifest: DatasetManifest,
}

impl TurnBoundaryDatasetWriter {
    pub fn create(
        output_path: impl AsRef<Path>,
        per_file: Option<u64>,
        target_source: TargetSource,
        source: SourceMetadata,
    ) -> Result<Self, String> {
        let output_path = output_path.as_ref().to_path_buf();
        let dataset_dir = output_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        std::fs::create_dir_all(&dataset_dir)
            .map_err(|e| format!("create dataset dir {}: {e}", dataset_dir.display()))?;

        let manifest_path = dataset_dir.join("manifest.yaml");
        let mut manifest = DatasetManifest::new_turn_boundary(target_source);
        manifest.source = source;

        Ok(Self {
            writer: RecordWriter::create(&output_path, per_file)?,
            output_path,
            dataset_dir,
            manifest_path,
            manifest,
        })
    }

    pub fn write(&mut self, records: &[TrainingRecord]) -> Result<(), String> {
        self.write_with_coverage(records, "unknown", "unknown", "unknown")
    }

    pub fn write_with_coverage(
        &mut self,
        records: &[TrainingRecord],
        range_source: &str,
        raise_depth: &str,
        boundary_ordinal: &str,
    ) -> Result<(), String> {
        for rec in records {
            validate_turn_boundary_record(rec)?;
        }
        self.writer.write(records)?;
        for rec in records {
            self.manifest.coverage.record_turn_boundary(
                rec,
                self.manifest.target_source,
                range_source,
                raise_depth,
                boundary_ordinal,
            );
        }
        Ok(())
    }

    pub fn count(&self) -> u64 {
        self.writer.count()
    }

    pub fn finish(mut self) -> Result<DatasetManifest, TurnBoundaryDatasetWriterError> {
        self.writer
            .flush()
            .map_err(TurnBoundaryDatasetWriterError::Write)?;
        self.manifest.shards = self.writer.shard_metadata(
            &self.dataset_dir,
            TURN_BOUNDARY_BOARD_SIZE,
            record_size(TURN_BOUNDARY_BOARD_SIZE as usize),
            Some(self.manifest.target_source),
        )?;
        self.manifest.coverage.total_records =
            self.manifest.shards.iter().map(|shard| shard.records).sum();
        self.manifest.validate_turn_boundary()?;
        self.manifest.write_yaml(&self.manifest_path)?;
        Ok(self.manifest)
    }

    pub fn output_path(&self) -> &Path {
        &self.output_path
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }
}

impl FlopBoundaryDatasetWriter {
    pub fn create(
        output_path: impl AsRef<Path>,
        per_file: Option<u64>,
        target_source: TargetSource,
        source: SourceMetadata,
    ) -> Result<Self, String> {
        let output_path = output_path.as_ref().to_path_buf();
        let dataset_dir = output_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        std::fs::create_dir_all(&dataset_dir)
            .map_err(|e| format!("create dataset dir {}: {e}", dataset_dir.display()))?;

        let manifest_path = dataset_dir.join("manifest.yaml");
        let mut manifest = DatasetManifest::new_flop_boundary(target_source);
        manifest.source = source;

        Ok(Self {
            writer: RecordWriter::create(&output_path, per_file)?,
            output_path,
            dataset_dir,
            manifest_path,
            manifest,
        })
    }

    pub fn write_with_coverage(
        &mut self,
        records: &[TrainingRecord],
        range_source: &str,
        raise_depth: &str,
        boundary_ordinal: &str,
    ) -> Result<(), String> {
        for rec in records {
            validate_flop_boundary_record(rec)?;
        }
        self.writer.write(records)?;
        for rec in records {
            self.manifest.coverage.record_turn_boundary(
                rec,
                self.manifest.target_source,
                range_source,
                raise_depth,
                boundary_ordinal,
            );
        }
        Ok(())
    }

    pub fn count(&self) -> u64 {
        self.writer.count()
    }

    pub fn finish(mut self) -> Result<DatasetManifest, TurnBoundaryDatasetWriterError> {
        self.writer
            .flush()
            .map_err(TurnBoundaryDatasetWriterError::Write)?;
        self.manifest.shards = self.writer.shard_metadata(
            &self.dataset_dir,
            FLOP_BOUNDARY_BOARD_SIZE,
            record_size(FLOP_BOUNDARY_BOARD_SIZE as usize),
            Some(self.manifest.target_source),
        )?;
        self.manifest.coverage.total_records =
            self.manifest.shards.iter().map(|shard| shard.records).sum();
        self.manifest.validate_flop_boundary()?;
        self.manifest.write_yaml(&self.manifest_path)?;
        Ok(self.manifest)
    }

    pub fn output_path(&self) -> &Path {
        &self.output_path
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }
}

fn validate_turn_boundary_record(rec: &TrainingRecord) -> Result<(), String> {
    if rec.board.len() != TURN_BOUNDARY_BOARD_SIZE as usize {
        return Err(format!(
            "turn-boundary records require board_size={}, got {}",
            TURN_BOUNDARY_BOARD_SIZE,
            rec.board.len()
        ));
    }
    if rec.player > 1 {
        return Err(format!("invalid player {}, expected 0 or 1", rec.player));
    }
    if rec.pot <= 0.0 {
        return Err(format!("pot must be positive, got {}", rec.pot));
    }
    if rec.effective_stack < 0.0 {
        return Err(format!(
            "effective_stack must be non-negative, got {}",
            rec.effective_stack
        ));
    }
    Ok(())
}

fn validate_flop_boundary_record(rec: &TrainingRecord) -> Result<(), String> {
    if rec.board.len() != FLOP_BOUNDARY_BOARD_SIZE as usize {
        return Err(format!(
            "flop-boundary records require board_size={}, got {}",
            FLOP_BOUNDARY_BOARD_SIZE,
            rec.board.len()
        ));
    }
    if rec.player > 1 {
        return Err(format!("invalid player {}, expected 0 or 1", rec.player));
    }
    if rec.pot <= 0.0 {
        return Err(format!("pot must be positive, got {}", rec.pot));
    }
    if rec.effective_stack < 0.0 {
        return Err(format!(
            "effective_stack must be non-negative, got {}",
            rec.effective_stack
        ));
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum TurnBoundaryDatasetWriterError {
    #[error("failed to write records: {0}")]
    Write(String),
    #[error("manifest validation failed: {0}")]
    ManifestValidation(#[from] ManifestValidationError),
    #[error("manifest write failed: {0}")]
    ManifestIo(#[from] ManifestIoError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datagen::storage::NUM_COMBOS;
    use tempfile::TempDir;

    fn sample_record(player: u8, pot: f32) -> TrainingRecord {
        TrainingRecord {
            board: vec![0, 4, 8, 12],
            pot,
            effective_stack: 100.0,
            player,
            game_value: 0.05,
            oop_range: [0.0; NUM_COMBOS],
            ip_range: [0.0; NUM_COMBOS],
            cfvs: [0.0; NUM_COMBOS],
            valid_mask: [0; NUM_COMBOS],
        }
    }

    fn sample_flop_record(player: u8, pot: f32) -> TrainingRecord {
        let mut rec = sample_record(player, pot);
        rec.board = vec![0, 4, 8];
        rec
    }

    #[test]
    fn finish_writes_manifest_for_rotated_shards() {
        let dir = TempDir::new().unwrap();
        let output = dir.path().join("turn_boundary.bin");
        let mut writer = TurnBoundaryDatasetWriter::create(
            &output,
            Some(2),
            TargetSource::RiverNet,
            SourceMetadata::default(),
        )
        .unwrap();

        for i in 0..5 {
            writer
                .write(&[sample_record((i % 2) as u8, 10.0 + i as f32)])
                .unwrap();
        }

        let manifest_path = writer.manifest_path().to_path_buf();
        let manifest = writer.finish().unwrap();

        assert!(manifest_path.exists());
        assert_eq!(manifest.coverage.total_records, 5);
        assert_eq!(manifest.shards.len(), 3);
        assert_eq!(manifest.shards[0].path, "turn_boundary.bin");
        assert_eq!(manifest.shards[0].records, 2);
        assert_eq!(manifest.shards[1].path, "turn_boundary_00001.bin");
        assert_eq!(manifest.shards[1].records, 2);
        assert_eq!(manifest.shards[2].path, "turn_boundary_00002.bin");
        assert_eq!(manifest.shards[2].records, 1);

        let loaded = DatasetManifest::read_yaml(&manifest_path).unwrap();
        assert_eq!(loaded, manifest);
        loaded.validate_turn_boundary().unwrap();
        assert_eq!(loaded.coverage.by_target_source.get("river_net"), Some(&5));
        assert_eq!(loaded.coverage.by_raise_depth.get("unknown"), Some(&5));
        assert_eq!(loaded.coverage.by_boundary_ordinal.get("unknown"), Some(&5));
    }

    #[test]
    fn rejects_non_turn_boundary_record() {
        let dir = TempDir::new().unwrap();
        let output = dir.path().join("turn_boundary.bin");
        let mut writer = TurnBoundaryDatasetWriter::create(
            &output,
            None,
            TargetSource::RiverNet,
            SourceMetadata::default(),
        )
        .unwrap();
        let mut rec = sample_record(0, 10.0);
        rec.board.push(16);

        let err = writer.write(&[rec]).unwrap_err();
        assert!(err.contains("board_size=4"));
    }

    #[test]
    fn finish_writes_flop_manifest_for_rotated_shards() {
        let dir = TempDir::new().unwrap();
        let output = dir.path().join("flop_boundary.bin");
        let mut writer = FlopBoundaryDatasetWriter::create(
            &output,
            Some(2),
            TargetSource::TurnNet,
            SourceMetadata::default(),
        )
        .unwrap();

        for i in 0..5 {
            writer
                .write_with_coverage(
                    &[sample_flop_record((i % 2) as u8, 10.0 + i as f32)],
                    "rsp",
                    "flop_subgame",
                    "turn_boundary",
                )
                .unwrap();
        }

        let manifest_path = writer.manifest_path().to_path_buf();
        let manifest = writer.finish().unwrap();

        assert!(manifest_path.exists());
        assert_eq!(manifest.coverage.total_records, 5);
        assert_eq!(manifest.shards.len(), 3);
        assert_eq!(manifest.shards[0].path, "flop_boundary.bin");
        assert_eq!(manifest.shards[0].records, 2);
        assert_eq!(manifest.shards[0].board_size, FLOP_BOUNDARY_BOARD_SIZE);

        let loaded = DatasetManifest::read_yaml(&manifest_path).unwrap();
        assert_eq!(loaded, manifest);
        loaded.validate_flop_boundary().unwrap();
        assert_eq!(loaded.coverage.by_target_source.get("turn_net"), Some(&5));
        assert_eq!(loaded.coverage.by_raise_depth.get("flop_subgame"), Some(&5));
        assert_eq!(
            loaded.coverage.by_boundary_ordinal.get("turn_boundary"),
            Some(&5)
        );
    }

    #[test]
    fn rejects_non_flop_boundary_record() {
        let dir = TempDir::new().unwrap();
        let output = dir.path().join("flop_boundary.bin");
        let mut writer = FlopBoundaryDatasetWriter::create(
            &output,
            None,
            TargetSource::TurnNet,
            SourceMetadata::default(),
        )
        .unwrap();

        let err = writer
            .write_with_coverage(&[sample_record(0, 10.0)], "rsp", "flop", "turn")
            .unwrap_err();
        assert!(err.contains("board_size=3"));
    }
}
