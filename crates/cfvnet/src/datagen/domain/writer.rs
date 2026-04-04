use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use crate::datagen::storage::{write_record, TrainingRecord};

/// Writes training records to binary files with count tracking and optional per-file rotation.
pub struct RecordWriter {
    base_path: PathBuf,
    per_file: Option<u64>,
    writer: BufWriter<File>,
    records_in_file: u64,
    file_index: u32,
    total_count: u64,
}

impl RecordWriter {
    pub fn create(path: &Path, per_file: Option<u64>) -> Result<Self, String> {
        let writer = Self::open_file(path, 0)?;
        Ok(Self {
            base_path: path.to_path_buf(),
            per_file,
            writer,
            records_in_file: 0,
            file_index: 0,
            total_count: 0,
        })
    }

    pub fn write(&mut self, records: &[TrainingRecord]) -> Result<(), String> {
        for rec in records {
            if let Some(limit) = self.per_file {
                if self.records_in_file >= limit {
                    self.rotate()?;
                }
            }
            write_record(&mut self.writer, rec)
                .map_err(|e| format!("write record: {e}"))?;
            self.records_in_file += 1;
            self.total_count += 1;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), String> {
        use std::io::Write;
        self.writer.flush().map_err(|e| format!("flush: {e}"))
    }

    pub fn count(&self) -> u64 {
        self.total_count
    }

    fn rotate(&mut self) -> Result<(), String> {
        self.flush()?;
        self.file_index += 1;
        self.writer = Self::open_file(&self.base_path, self.file_index)?;
        self.records_in_file = 0;
        let path = Self::file_path(&self.base_path, self.file_index);
        eprintln!("[writer] rotated to {}", path.display());
        Ok(())
    }

    fn open_file(base: &Path, index: u32) -> Result<BufWriter<File>, String> {
        let path = Self::file_path(base, index);
        let file = File::create(&path)
            .map_err(|e| format!("create {}: {e}", path.display()))?;
        Ok(BufWriter::with_capacity(1 << 20, file))
    }

    fn file_path(base: &Path, index: u32) -> PathBuf {
        if index == 0 {
            base.to_path_buf()
        } else {
            let stem = base.file_stem().unwrap_or_default().to_string_lossy();
            let parent = base.parent().unwrap_or(Path::new("."));
            parent.join(format!("{stem}_{index:05}.bin"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datagen::storage::read_record;
    use std::io::BufReader;
    use tempfile::{NamedTempFile, TempDir};

    fn sample_record(player: u8, pot: f32) -> TrainingRecord {
        TrainingRecord {
            board: vec![0, 4, 8, 12],
            pot,
            effective_stack: 100.0,
            player,
            game_value: 0.05,
            oop_range: [0.0; 1326],
            ip_range: [0.0; 1326],
            cfvs: [0.0; 1326],
            valid_mask: [0; 1326],
        }
    }

    fn count_records_in_file(path: &Path) -> u64 {
        let mut reader = BufReader::new(std::fs::File::open(path).unwrap());
        let mut count = 0;
        while read_record(&mut reader).is_ok() {
            count += 1;
        }
        count
    }

    #[test]
    fn count_starts_at_zero() {
        let tmp = NamedTempFile::new().unwrap();
        let writer = RecordWriter::create(tmp.path(), None).unwrap();
        assert_eq!(writer.count(), 0);
    }

    #[test]
    fn write_increments_count() {
        let tmp = NamedTempFile::new().unwrap();
        let mut writer = RecordWriter::create(tmp.path(), None).unwrap();
        let records = vec![sample_record(0, 100.0), sample_record(1, 100.0)];
        writer.write(&records).unwrap();
        assert_eq!(writer.count(), 2);
    }

    #[test]
    fn written_records_roundtrip_correctly() {
        let tmp = NamedTempFile::new().unwrap();
        let mut writer = RecordWriter::create(tmp.path(), None).unwrap();
        let records = vec![sample_record(0, 150.0), sample_record(1, 200.0)];
        writer.write(&records).unwrap();
        writer.flush().unwrap();

        let mut reader = BufReader::new(std::fs::File::open(tmp.path()).unwrap());
        let r0 = read_record(&mut reader).unwrap();
        let r1 = read_record(&mut reader).unwrap();
        assert_eq!(r0.player, 0);
        assert_eq!(r0.pot, 150.0);
        assert_eq!(r1.player, 1);
        assert_eq!(r1.pot, 200.0);
    }

    #[test]
    fn multiple_write_calls_accumulate_count() {
        let tmp = NamedTempFile::new().unwrap();
        let mut writer = RecordWriter::create(tmp.path(), None).unwrap();
        writer.write(&[sample_record(0, 100.0)]).unwrap();
        writer.write(&[sample_record(1, 100.0)]).unwrap();
        writer.write(&[sample_record(0, 200.0)]).unwrap();
        assert_eq!(writer.count(), 3);
    }

    #[test]
    fn write_empty_slice_does_not_change_count() {
        let tmp = NamedTempFile::new().unwrap();
        let mut writer = RecordWriter::create(tmp.path(), None).unwrap();
        writer.write(&[]).unwrap();
        assert_eq!(writer.count(), 0);
    }

    #[test]
    fn rotation_creates_correct_number_of_files() {
        let dir = TempDir::new().unwrap();
        let base = dir.path().join("data.bin");
        let mut writer = RecordWriter::create(&base, Some(3)).unwrap();
        for i in 0..10 {
            writer.write(&[sample_record(0, i as f32)]).unwrap();
        }
        writer.flush().unwrap();

        // 10 records at 3 per file = 4 files (3+3+3+1).
        assert!(dir.path().join("data.bin").exists(), "first file should use original name");
        assert!(dir.path().join("data_00001.bin").exists(), "second file");
        assert!(dir.path().join("data_00002.bin").exists(), "third file");
        assert!(dir.path().join("data_00003.bin").exists(), "fourth file");
        assert!(!dir.path().join("data_00004.bin").exists(), "no fifth file");
    }

    #[test]
    fn rotation_distributes_records_correctly() {
        let dir = TempDir::new().unwrap();
        let base = dir.path().join("data.bin");
        let mut writer = RecordWriter::create(&base, Some(3)).unwrap();
        for i in 0..10 {
            writer.write(&[sample_record(0, i as f32)]).unwrap();
        }
        writer.flush().unwrap();

        assert_eq!(count_records_in_file(&dir.path().join("data.bin")), 3);
        assert_eq!(count_records_in_file(&dir.path().join("data_00001.bin")), 3);
        assert_eq!(count_records_in_file(&dir.path().join("data_00002.bin")), 3);
        assert_eq!(count_records_in_file(&dir.path().join("data_00003.bin")), 1);
    }

    #[test]
    fn rotation_total_count_tracks_all_records() {
        let dir = TempDir::new().unwrap();
        let base = dir.path().join("data.bin");
        let mut writer = RecordWriter::create(&base, Some(3)).unwrap();
        for i in 0..10 {
            writer.write(&[sample_record(0, i as f32)]).unwrap();
        }
        assert_eq!(writer.count(), 10);
    }

    #[test]
    fn no_rotation_when_per_file_is_none() {
        let dir = TempDir::new().unwrap();
        let base = dir.path().join("data.bin");
        let mut writer = RecordWriter::create(&base, None).unwrap();
        for i in 0..10 {
            writer.write(&[sample_record(0, i as f32)]).unwrap();
        }
        writer.flush().unwrap();

        assert_eq!(count_records_in_file(&dir.path().join("data.bin")), 10);
        assert!(!dir.path().join("data_00001.bin").exists(), "no rotation files");
    }

    #[test]
    fn rotation_with_exact_boundary() {
        // 6 records with per_file=3 = exactly 2 files, no leftover.
        let dir = TempDir::new().unwrap();
        let base = dir.path().join("data.bin");
        let mut writer = RecordWriter::create(&base, Some(3)).unwrap();
        for i in 0..6 {
            writer.write(&[sample_record(0, i as f32)]).unwrap();
        }
        writer.flush().unwrap();

        assert_eq!(count_records_in_file(&dir.path().join("data.bin")), 3);
        assert_eq!(count_records_in_file(&dir.path().join("data_00001.bin")), 3);
        assert!(!dir.path().join("data_00002.bin").exists());
        assert_eq!(writer.count(), 6);
    }

    #[test]
    fn rotation_with_per_file_one() {
        let dir = TempDir::new().unwrap();
        let base = dir.path().join("data.bin");
        let mut writer = RecordWriter::create(&base, Some(1)).unwrap();
        for i in 0..3 {
            writer.write(&[sample_record(0, i as f32)]).unwrap();
        }
        writer.flush().unwrap();

        assert_eq!(count_records_in_file(&dir.path().join("data.bin")), 1);
        assert_eq!(count_records_in_file(&dir.path().join("data_00001.bin")), 1);
        assert_eq!(count_records_in_file(&dir.path().join("data_00002.bin")), 1);
        assert!(!dir.path().join("data_00003.bin").exists());
    }

    #[test]
    fn rotation_records_are_readable_in_order() {
        let dir = TempDir::new().unwrap();
        let base = dir.path().join("data.bin");
        let mut writer = RecordWriter::create(&base, Some(2)).unwrap();
        for i in 0..5 {
            writer.write(&[sample_record(0, (i * 10) as f32)]).unwrap();
        }
        writer.flush().unwrap();

        // File 0: pots 0, 10
        let mut reader = BufReader::new(std::fs::File::open(dir.path().join("data.bin")).unwrap());
        assert_eq!(read_record(&mut reader).unwrap().pot, 0.0);
        assert_eq!(read_record(&mut reader).unwrap().pot, 10.0);

        // File 1: pots 20, 30
        let mut reader = BufReader::new(std::fs::File::open(dir.path().join("data_00001.bin")).unwrap());
        assert_eq!(read_record(&mut reader).unwrap().pot, 20.0);
        assert_eq!(read_record(&mut reader).unwrap().pot, 30.0);

        // File 2: pot 40
        let mut reader = BufReader::new(std::fs::File::open(dir.path().join("data_00002.bin")).unwrap());
        assert_eq!(read_record(&mut reader).unwrap().pot, 40.0);
    }
}
