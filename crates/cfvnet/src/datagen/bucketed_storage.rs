use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

/// A single bucketed training record.
///
/// - `input`: `[oop_reach(num_buckets) | ip_reach(num_buckets) | pot/initial_stack]`
///   -- length `2 * num_buckets + 1`
/// - `target`: `[oop_cfvs(num_buckets) | ip_cfvs(num_buckets)]`
///   -- length `2 * num_buckets`
#[derive(Debug, Clone)]
pub struct BucketedRecord {
    pub input: Vec<f32>,  // length 2*num_buckets + 1
    pub target: Vec<f32>, // length 2*num_buckets
}

/// Read the file header: a single little-endian `u32` containing `num_buckets`.
pub fn read_bucketed_header<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Read a single bucketed record given `num_buckets`.
///
/// Returns `UnexpectedEof` when the stream has no more data.
pub fn read_bucketed_record<R: Read>(
    reader: &mut R,
    num_buckets: usize,
) -> io::Result<BucketedRecord> {
    let input_len = 2 * num_buckets + 1;
    let target_len = 2 * num_buckets;

    let mut input = vec![0.0f32; input_len];
    let mut target = vec![0.0f32; target_len];

    // Read input floats via bytemuck zero-copy cast
    let input_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut input);
    reader.read_exact(input_bytes)?;

    // Read target floats
    let target_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut target);
    reader.read_exact(target_bytes)?;

    Ok(BucketedRecord { input, target })
}

/// Write the file header: a single little-endian `u32` containing `num_buckets`.
pub fn write_bucketed_header<W: Write>(writer: &mut W, num_buckets: u32) -> io::Result<()> {
    writer.write_all(&num_buckets.to_le_bytes())
}

/// Write a single bucketed record (input then target, as raw LE f32 bytes).
pub fn write_bucketed_record<W: Write>(
    writer: &mut W,
    record: &BucketedRecord,
) -> io::Result<()> {
    let input_bytes: &[u8] = bytemuck::cast_slice(&record.input);
    writer.write_all(input_bytes)?;
    let target_bytes: &[u8] = bytemuck::cast_slice(&record.target);
    writer.write_all(target_bytes)
}

/// Byte size of a single bucketed record (no header).
///
/// Each record stores `(2*num_buckets+1)` input floats and `(2*num_buckets)`
/// target floats, i.e. `(4*num_buckets + 1) * 4` bytes.
pub const fn bucketed_record_size(num_buckets: usize) -> usize {
    (2 * num_buckets + 1 + 2 * num_buckets) * std::mem::size_of::<f32>()
}

/// Header size in bytes (single `u32`).
const HEADER_SIZE: u64 = 4;

/// Count records in a bucketed file by file-size arithmetic.
///
/// The stream position is restored after the call.
pub fn count_bucketed_records<S: Seek>(stream: &mut S, num_buckets: usize) -> io::Result<u64> {
    let current = stream.stream_position()?;
    let end = stream.seek(SeekFrom::End(0))?;
    stream.seek(SeekFrom::Start(current))?;

    let data_bytes = end.saturating_sub(HEADER_SIZE);
    let rec_size = bucketed_record_size(num_buckets) as u64;
    if rec_size == 0 {
        return Ok(0);
    }
    Ok(data_bytes / rec_size)
}

/// Count records across multiple bucketed files.
///
/// Opens each file, reads the header to obtain `num_buckets`, then uses the
/// file size to compute the record count. All files must share the same
/// `num_buckets`; a mismatch returns an `InvalidData` error.
///
/// Returns `(num_buckets, total_records)`. If `files` is empty, returns `(0, 0)`.
pub fn count_bucketed_records_in_files(files: &[PathBuf]) -> io::Result<(u32, u64)> {
    let mut total = 0u64;
    let mut expected_buckets: Option<u32> = None;

    for path in files {
        let mut file = std::fs::File::open(path)?;
        let file_size = file.metadata()?.len();
        if file_size < HEADER_SIZE {
            eprintln!("Warning: skipping empty file {}", path.display());
            continue;
        }
        let nb = read_bucketed_header(&mut file)?;

        if let Some(expected) = expected_buckets {
            if nb != expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "bucket count mismatch: {} has {nb} but expected {expected}",
                        path.display()
                    ),
                ));
            }
        } else {
            expected_buckets = Some(nb);
        }

        let rec_size = bucketed_record_size(nb as usize) as u64;
        let data_bytes = file_size.saturating_sub(HEADER_SIZE);
        total += data_bytes / rec_size;
    }

    Ok((expected_buckets.unwrap_or(0), total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufWriter, Seek, SeekFrom};
    use tempfile::NamedTempFile;

    /// Build a deterministic record for a given `num_buckets`.
    fn make_record(num_buckets: usize, seed: f32) -> BucketedRecord {
        let input_len = 2 * num_buckets + 1;
        let target_len = 2 * num_buckets;
        let input: Vec<f32> = (0..input_len).map(|i| seed + i as f32 * 0.01).collect();
        let target: Vec<f32> = (0..target_len).map(|i| seed - i as f32 * 0.02).collect();
        BucketedRecord { input, target }
    }

    #[test]
    fn roundtrip_record() {
        let num_buckets: u32 = 10;
        let records = vec![make_record(10, 1.0), make_record(10, 2.5)];

        let mut file = NamedTempFile::new().unwrap();

        // Write header + records
        {
            let mut w = BufWriter::new(&mut file);
            write_bucketed_header(&mut w, num_buckets).unwrap();
            for rec in &records {
                write_bucketed_record(&mut w, rec).unwrap();
            }
            // flush via drop
        }

        // Read back
        file.seek(SeekFrom::Start(0)).unwrap();
        let nb = read_bucketed_header(&mut file).unwrap();
        assert_eq!(nb, num_buckets);

        for original in &records {
            let loaded = read_bucketed_record(&mut file, nb as usize).unwrap();
            assert_eq!(original.input.len(), loaded.input.len());
            assert_eq!(original.target.len(), loaded.target.len());
            for (a, b) in original.input.iter().zip(&loaded.input) {
                assert!((a - b).abs() < 1e-7, "input mismatch: {a} vs {b}");
            }
            for (a, b) in original.target.iter().zip(&loaded.target) {
                assert!((a - b).abs() < 1e-7, "target mismatch: {a} vs {b}");
            }
        }

        // Should get UnexpectedEof when trying to read past the end
        let err = read_bucketed_record(&mut file, nb as usize).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn record_size_calculation() {
        let num_buckets = 10usize;
        let rec = make_record(num_buckets, 0.0);

        let mut buf = Vec::new();
        write_bucketed_record(&mut buf, &rec).unwrap();

        assert_eq!(
            buf.len(),
            bucketed_record_size(num_buckets),
            "written bytes should match bucketed_record_size"
        );

        // Sanity: (4*10 + 1) * 4 = 164
        assert_eq!(bucketed_record_size(num_buckets), (4 * 10 + 1) * 4);
    }

    #[test]
    fn count_records_correct() {
        let num_buckets: u32 = 8;
        let n = 7;

        let mut file = NamedTempFile::new().unwrap();
        {
            let mut w = BufWriter::new(&mut file);
            write_bucketed_header(&mut w, num_buckets).unwrap();
            for i in 0..n {
                let rec = make_record(num_buckets as usize, i as f32);
                write_bucketed_record(&mut w, &rec).unwrap();
            }
        }

        file.seek(SeekFrom::Start(0)).unwrap();
        let count = count_bucketed_records(&mut file, num_buckets as usize).unwrap();
        assert_eq!(count, n);
    }

    #[test]
    fn count_records_in_files() {
        let num_buckets: u32 = 5;

        // Write 3 records to file A
        let mut file_a = NamedTempFile::new().unwrap();
        {
            let mut w = BufWriter::new(&mut file_a);
            write_bucketed_header(&mut w, num_buckets).unwrap();
            for i in 0..3 {
                let rec = make_record(num_buckets as usize, i as f32);
                write_bucketed_record(&mut w, &rec).unwrap();
            }
        }

        // Write 5 records to file B
        let mut file_b = NamedTempFile::new().unwrap();
        {
            let mut w = BufWriter::new(&mut file_b);
            write_bucketed_header(&mut w, num_buckets).unwrap();
            for i in 0..5 {
                let rec = make_record(num_buckets as usize, i as f32 + 100.0);
                write_bucketed_record(&mut w, &rec).unwrap();
            }
        }

        let paths = vec![
            file_a.path().to_path_buf(),
            file_b.path().to_path_buf(),
        ];
        let (nb, total) = count_bucketed_records_in_files(&paths).unwrap();
        assert_eq!(nb, num_buckets);
        assert_eq!(total, 8);
    }

    #[test]
    fn mismatched_buckets_errors() {
        // File A: 5 buckets
        let mut file_a = NamedTempFile::new().unwrap();
        {
            let mut w = BufWriter::new(&mut file_a);
            write_bucketed_header(&mut w, 5).unwrap();
            write_bucketed_record(&mut w, &make_record(5, 1.0)).unwrap();
        }

        // File B: 10 buckets
        let mut file_b = NamedTempFile::new().unwrap();
        {
            let mut w = BufWriter::new(&mut file_b);
            write_bucketed_header(&mut w, 10).unwrap();
            write_bucketed_record(&mut w, &make_record(10, 1.0)).unwrap();
        }

        let paths = vec![
            file_a.path().to_path_buf(),
            file_b.path().to_path_buf(),
        ];
        let err = count_bucketed_records_in_files(&paths).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(
            msg.contains("mismatch"),
            "error should mention mismatch: {msg}"
        );
    }
}
