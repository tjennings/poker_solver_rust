use std::io::BufWriter;
use std::path::Path;

use crate::datagen::storage::{write_record, TrainingRecord};

/// Writes training records to a binary file with count tracking.
pub struct RecordWriter {
    writer: BufWriter<std::fs::File>,
    count: u64,
}

impl RecordWriter {
    pub fn create(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::create(path)
            .map_err(|e| format!("create {}: {e}", path.display()))?;
        Ok(Self {
            writer: BufWriter::with_capacity(1 << 20, file),
            count: 0,
        })
    }

    pub fn write(&mut self, records: &[TrainingRecord]) -> Result<(), String> {
        for rec in records {
            write_record(&mut self.writer, rec)
                .map_err(|e| format!("write record: {e}"))?;
            self.count += 1;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), String> {
        use std::io::Write;
        self.writer.flush().map_err(|e| format!("flush: {e}"))
    }

    pub fn count(&self) -> u64 {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datagen::storage::read_record;
    use std::io::BufReader;
    use tempfile::NamedTempFile;

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

    #[test]
    fn count_starts_at_zero() {
        let tmp = NamedTempFile::new().unwrap();
        let writer = RecordWriter::create(tmp.path()).unwrap();
        assert_eq!(writer.count(), 0);
    }

    #[test]
    fn write_increments_count() {
        let tmp = NamedTempFile::new().unwrap();
        let mut writer = RecordWriter::create(tmp.path()).unwrap();
        let records = vec![sample_record(0, 100.0), sample_record(1, 100.0)];
        writer.write(&records).unwrap();
        assert_eq!(writer.count(), 2);
    }

    #[test]
    fn written_records_roundtrip_correctly() {
        let tmp = NamedTempFile::new().unwrap();
        let mut writer = RecordWriter::create(tmp.path()).unwrap();
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
        let mut writer = RecordWriter::create(tmp.path()).unwrap();
        writer.write(&[sample_record(0, 100.0)]).unwrap();
        writer.write(&[sample_record(1, 100.0)]).unwrap();
        writer.write(&[sample_record(0, 200.0)]).unwrap();
        assert_eq!(writer.count(), 3);
    }

    #[test]
    fn write_empty_slice_does_not_change_count() {
        let tmp = NamedTempFile::new().unwrap();
        let mut writer = RecordWriter::create(tmp.path()).unwrap();
        writer.write(&[]).unwrap();
        assert_eq!(writer.count(), 0);
    }
}
