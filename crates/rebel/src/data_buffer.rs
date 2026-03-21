// Disk-backed reservoir buffer

use memmap2::Mmap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use rand::Rng;

/// A fixed-size record for mmap-friendly random access.
#[derive(Clone)]
pub struct BufferRecord {
    pub board: [u8; 5],
    pub board_card_count: u8,
    pub pot: f32,
    pub effective_stack: f32,
    pub player: u8,
    pub game_value: f32,
    pub oop_reach: [f32; 1326],
    pub ip_reach: [f32; 1326],
    pub cfvs: [f32; 1326],
    pub valid_mask: [u8; 1326],
}

impl BufferRecord {
    /// Byte size of one record on disk.
    // board(5) + board_card_count(1) + pot(4) + effective_stack(4) + player(1) + game_value(4)
    // + oop_reach(1326*4) + ip_reach(1326*4) + cfvs(1326*4) + valid_mask(1326*1)
    // = 19 + 15912 + 1326 = 17257
    pub const SIZE: usize = 5 + 1 + 4 + 4 + 1 + 4 + 1326 * 4 + 1326 * 4 + 1326 * 4 + 1326;

    /// Serialize this record as little-endian bytes into `buf`.
    pub fn serialize(&self, buf: &mut Vec<u8>) {
        buf.reserve(Self::SIZE);
        buf.extend_from_slice(&self.board);
        buf.push(self.board_card_count);
        buf.extend_from_slice(&self.pot.to_le_bytes());
        buf.extend_from_slice(&self.effective_stack.to_le_bytes());
        buf.push(self.player);
        buf.extend_from_slice(&self.game_value.to_le_bytes());
        for &v in &self.oop_reach {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.ip_reach {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.cfvs {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&self.valid_mask);
    }

    /// Deserialize a record from a byte slice (must be at least `SIZE` bytes).
    pub fn deserialize(data: &[u8]) -> Self {
        assert!(
            data.len() >= Self::SIZE,
            "data too short: {} < {}",
            data.len(),
            Self::SIZE
        );

        let mut offset = 0;

        let mut board = [0u8; 5];
        board.copy_from_slice(&data[offset..offset + 5]);
        offset += 5;

        let board_card_count = data[offset];
        offset += 1;

        let pot = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        offset += 4;

        let effective_stack = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        offset += 4;

        let player = data[offset];
        offset += 1;

        let game_value = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        offset += 4;

        let mut oop_reach = [0.0f32; 1326];
        for v in &mut oop_reach {
            *v = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            offset += 4;
        }

        let mut ip_reach = [0.0f32; 1326];
        for v in &mut ip_reach {
            *v = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            offset += 4;
        }

        let mut cfvs = [0.0f32; 1326];
        for v in &mut cfvs {
            *v = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            offset += 4;
        }

        let mut valid_mask = [0u8; 1326];
        valid_mask.copy_from_slice(&data[offset..offset + 1326]);

        Self {
            board,
            board_card_count,
            pot,
            effective_stack,
            player,
            game_value,
            oop_reach,
            ip_reach,
            cfvs,
            valid_mask,
        }
    }
}

/// Disk-backed reservoir buffer with mmap random sampling.
pub struct DiskBuffer {
    #[allow(dead_code)]
    path: PathBuf,
    file: File,
    count: usize,
    max_records: usize,
    total_appended: usize,
}

impl DiskBuffer {
    /// Create a new buffer file at `path` with the given capacity.
    pub fn create(path: impl AsRef<Path>, max_records: usize) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        Ok(Self {
            path,
            file,
            count: 0,
            max_records,
            total_appended: 0,
        })
    }

    /// Open an existing buffer file, inferring count from file size.
    pub fn open(path: impl AsRef<Path>, max_records: usize) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().read(true).write(true).open(&path)?;
        let file_size = file.metadata()?.len() as usize;
        let count = file_size / BufferRecord::SIZE;
        Ok(Self {
            path,
            file,
            count,
            max_records,
            // When reopening, we don't know the true total_appended;
            // set it to count so future reservoir sampling works correctly.
            total_appended: count,
        })
    }

    /// Current number of records stored.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Append a record. If full, use reservoir sampling to decide whether to replace.
    pub fn append(&mut self, record: &BufferRecord) -> io::Result<()> {
        self.total_appended += 1;

        if self.count < self.max_records {
            // Buffer not yet full: append at end
            self.write_record(self.count, record)?;
            self.count += 1;
        } else {
            // Reservoir sampling: replace index j with probability max_records / total_appended
            let j = rand::rng().random_range(0..self.total_appended);
            if j < self.max_records {
                self.write_record(j, record)?;
            }
        }
        Ok(())
    }

    /// Read a single record at the given index.
    pub fn read_record(&self, index: usize) -> io::Result<BufferRecord> {
        if index >= self.count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("index {} out of bounds (count {})", index, self.count),
            ));
        }
        let offset = (index * BufferRecord::SIZE) as u64;
        let mut buf = vec![0u8; BufferRecord::SIZE];
        // Use a clone of the file handle so we can read from an immutable reference.
        // File::try_clone shares the underlying OS handle but gives an independent seek position.
        let mut reader = self.file.try_clone()?;
        reader.seek(SeekFrom::Start(offset))?;
        reader.read_exact(&mut buf)?;
        Ok(BufferRecord::deserialize(&buf))
    }

    /// Write a single record at the given index.
    pub fn write_record(&mut self, index: usize, record: &BufferRecord) -> io::Result<()> {
        let offset = (index * BufferRecord::SIZE) as u64;
        let mut buf = Vec::with_capacity(BufferRecord::SIZE);
        record.serialize(&mut buf);
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(&buf)?;
        self.file.flush()?;
        Ok(())
    }

    /// Sample `n` random records using mmap.
    pub fn sample<R: Rng>(&self, rng: &mut R, n: usize) -> io::Result<Vec<BufferRecord>> {
        if self.count == 0 {
            return Ok(Vec::new());
        }
        // SAFETY:
        // 1. The file has nonzero length (checked by the `self.count == 0` guard above).
        // 2. The mapped region must not be concurrently modified while the `Mmap`
        //    is in scope.  `DiskBuffer` enforces this by taking `&self` (shared
        //    reference), which prevents any `&mut self` writer from running
        //    concurrently in safe Rust.  External processes must not write to
        //    the file while this mmap is live.
        let mmap = unsafe { Mmap::map(&self.file)? };
        let mut records = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = rng.random_range(0..self.count);
            let start = idx * BufferRecord::SIZE;
            let end = start + BufferRecord::SIZE;
            records.push(BufferRecord::deserialize(&mmap[start..end]));
        }
        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use tempfile::TempDir;

    fn make_record(pot: f32) -> BufferRecord {
        let mut rec = BufferRecord {
            board: [0xFF; 5],
            board_card_count: 3,
            pot,
            effective_stack: 100.0,
            player: 0,
            game_value: 42.0,
            oop_reach: [0.0; 1326],
            ip_reach: [0.0; 1326],
            cfvs: [0.0; 1326],
            valid_mask: [0; 1326],
        };
        // Set first three board cards to something recognizable
        rec.board[0] = 10;
        rec.board[1] = 20;
        rec.board[2] = 30;
        // Set a few reach/cfv values to non-zero for roundtrip checking
        rec.oop_reach[0] = 1.0;
        rec.ip_reach[1] = 0.5;
        rec.cfvs[2] = -3.14;
        rec.valid_mask[0] = 1;
        rec.valid_mask[1] = 1;
        rec.valid_mask[2] = 1;
        rec
    }

    #[test]
    fn test_buffer_record_size() {
        // board(5) + board_card_count(1) + pot(4) + effective_stack(4) + player(1) + game_value(4)
        // + oop_reach(5304) + ip_reach(5304) + cfvs(5304) + valid_mask(1326)
        let expected = 5 + 1 + 4 + 4 + 1 + 4 + 1326 * 4 + 1326 * 4 + 1326 * 4 + 1326;
        assert_eq!(BufferRecord::SIZE, expected);
        assert_eq!(BufferRecord::SIZE, 17257);
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let rec = make_record(200.0);
        let mut buf = Vec::new();
        rec.serialize(&mut buf);
        assert_eq!(buf.len(), BufferRecord::SIZE);

        let rec2 = BufferRecord::deserialize(&buf);
        assert_eq!(rec2.board, rec.board);
        assert_eq!(rec2.board_card_count, rec.board_card_count);
        assert_eq!(rec2.pot, rec.pot);
        assert_eq!(rec2.effective_stack, rec.effective_stack);
        assert_eq!(rec2.player, rec.player);
        assert_eq!(rec2.game_value, rec.game_value);
        assert_eq!(rec2.oop_reach[0], 1.0);
        assert_eq!(rec2.ip_reach[1], 0.5);
        assert_eq!(rec2.cfvs[2], -3.14);
        assert_eq!(rec2.valid_mask[0], 1);
        assert_eq!(rec2.valid_mask[1], 1);
        assert_eq!(rec2.valid_mask[2], 1);
        // Check a zero element
        assert_eq!(rec2.oop_reach[100], 0.0);
        assert_eq!(rec2.valid_mask[100], 0);
    }

    #[test]
    fn test_buffer_append_and_count() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("buffer.bin");
        let mut buf = DiskBuffer::create(&path, 100).unwrap();
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());

        buf.append(&make_record(1.0)).unwrap();
        buf.append(&make_record(2.0)).unwrap();
        buf.append(&make_record(3.0)).unwrap();
        assert_eq!(buf.len(), 3);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_buffer_read_write_record() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("buffer.bin");
        let mut buf = DiskBuffer::create(&path, 100).unwrap();

        for i in 0..5 {
            buf.append(&make_record(i as f32 * 10.0)).unwrap();
        }
        assert_eq!(buf.len(), 5);

        for i in 0..5 {
            let rec = buf.read_record(i).unwrap();
            assert_eq!(rec.pot, i as f32 * 10.0, "pot mismatch at index {}", i);
        }
    }

    #[test]
    fn test_buffer_random_sample() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("buffer.bin");
        let mut buf = DiskBuffer::create(&path, 200).unwrap();

        for i in 0..100 {
            buf.append(&make_record(i as f32)).unwrap();
        }
        assert_eq!(buf.len(), 100);

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let samples = buf.sample(&mut rng, 10).unwrap();
        assert_eq!(samples.len(), 10);

        for s in &samples {
            assert!(
                s.pot >= 0.0 && s.pot < 100.0,
                "sampled pot {} out of range [0, 100)",
                s.pot
            );
        }
    }

    #[test]
    fn test_buffer_reservoir_replacement() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("buffer.bin");
        let mut buf = DiskBuffer::create(&path, 10).unwrap();

        for i in 0..20 {
            buf.append(&make_record(i as f32)).unwrap();
        }
        // Buffer capped at max_records
        assert_eq!(buf.len(), 10);

        // All stored pots should be valid values from 0..20
        for i in 0..10 {
            let rec = buf.read_record(i).unwrap();
            assert!(
                rec.pot >= 0.0 && rec.pot < 20.0,
                "record {} has invalid pot {}",
                i,
                rec.pot
            );
        }
    }

    #[test]
    fn test_buffer_open_existing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("buffer.bin");

        {
            let mut buf = DiskBuffer::create(&path, 100).unwrap();
            buf.append(&make_record(11.0)).unwrap();
            buf.append(&make_record(22.0)).unwrap();
            buf.append(&make_record(33.0)).unwrap();
            assert_eq!(buf.len(), 3);
        }

        // Re-open
        let buf2 = DiskBuffer::open(&path, 100).unwrap();
        assert_eq!(buf2.len(), 3);
        let rec0 = buf2.read_record(0).unwrap();
        assert_eq!(rec0.pot, 11.0);
        let rec2 = buf2.read_record(2).unwrap();
        assert_eq!(rec2.pot, 33.0);
    }
}
