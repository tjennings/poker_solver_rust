use std::io::{self, Read, Seek, SeekFrom, Write};

pub const NUM_COMBOS: usize = 1326;

/// Compute the byte size of one training record for a given board size.
///
/// Layout: board_size(1) + board(board_size) + pot(4) + stack(4)
///       + oop_range(1326*4) + ip_range(1326*4) + oop_cfvs(1326*4)
///       + ip_cfvs(1326*4) + valid_mask(1326)
pub const fn record_size(board_size: usize) -> usize {
    1 // board_size prefix byte
    + board_size
    + 4  // pot: f32
    + 4  // effective_stack: f32
    + NUM_COMBOS * 4  // oop_range
    + NUM_COMBOS * 4  // ip_range
    + NUM_COMBOS * 4  // oop_cfvs
    + NUM_COMBOS * 4  // ip_cfvs
    + NUM_COMBOS      // valid_mask
}

/// One training sample: a solved subgame with CFVs for both players.
#[derive(Debug, Clone)]
pub struct TrainingRecord {
    pub board: Vec<u8>,
    pub pot: f32,
    pub effective_stack: f32,
    pub oop_range: [f32; NUM_COMBOS],
    pub ip_range: [f32; NUM_COMBOS],
    pub oop_cfvs: [f32; NUM_COMBOS],
    pub ip_cfvs: [f32; NUM_COMBOS],
    pub valid_mask: [u8; NUM_COMBOS],
}

/// Write a single training record in little-endian (native) format.
///
/// Format: `[board_size: u8] [board: board_size × u8] [pot: f32] ...`
///
/// The f32 arrays are written as a single bulk `write_all` call via `bytemuck`,
/// which is correct on little-endian platforms (all current targets).
pub fn write_record<W: Write>(w: &mut W, rec: &TrainingRecord) -> io::Result<()> {
    let board_size = rec.board.len() as u8;
    w.write_all(&[board_size])?;
    w.write_all(&rec.board)?;
    w.write_all(&rec.pot.to_le_bytes())?;
    w.write_all(&rec.effective_stack.to_le_bytes())?;
    w.write_all(bytemuck::cast_slice(&rec.oop_range))?;
    w.write_all(bytemuck::cast_slice(&rec.ip_range))?;
    w.write_all(bytemuck::cast_slice(&rec.oop_cfvs))?;
    w.write_all(bytemuck::cast_slice(&rec.ip_cfvs))?;
    w.write_all(&rec.valid_mask)?;
    Ok(())
}

/// Read a single training record from little-endian format.
///
/// The first byte is the board size, followed by that many board card bytes.
pub fn read_record<R: Read>(r: &mut R) -> io::Result<TrainingRecord> {
    let mut buf1 = [0u8; 1];
    r.read_exact(&mut buf1)?;
    let board_size = buf1[0] as usize;

    let mut board = vec![0u8; board_size];
    r.read_exact(&mut board)?;

    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4)?;
    let pot = f32::from_le_bytes(buf4);

    r.read_exact(&mut buf4)?;
    let effective_stack = f32::from_le_bytes(buf4);

    let mut oop_range = [0.0f32; NUM_COMBOS];
    r.read_exact(bytemuck::cast_slice_mut(&mut oop_range))?;

    let mut ip_range = [0.0f32; NUM_COMBOS];
    r.read_exact(bytemuck::cast_slice_mut(&mut ip_range))?;

    let mut oop_cfvs = [0.0f32; NUM_COMBOS];
    r.read_exact(bytemuck::cast_slice_mut(&mut oop_cfvs))?;

    let mut ip_cfvs = [0.0f32; NUM_COMBOS];
    r.read_exact(bytemuck::cast_slice_mut(&mut ip_cfvs))?;

    let mut valid_mask = [0u8; NUM_COMBOS];
    r.read_exact(&mut valid_mask)?;

    Ok(TrainingRecord {
        board,
        pot,
        effective_stack,
        oop_range,
        ip_range,
        oop_cfvs,
        ip_cfvs,
        valid_mask,
    })
}

/// Count the number of complete records for a uniform board size.
///
/// All records in the file must have the same `board_size`.
pub fn count_records<S: Seek>(s: &mut S, board_size: usize) -> io::Result<u64> {
    let end = s.seek(SeekFrom::End(0))?;
    s.seek(SeekFrom::Start(0))?;
    Ok(end / record_size(board_size) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom};
    use tempfile::NamedTempFile;

    fn sample_record() -> TrainingRecord {
        let mut oop_range = [0.0f32; 1326];
        let mut ip_range = [0.0f32; 1326];
        let mut oop_cfvs = [0.0f32; 1326];
        let mut ip_cfvs = [0.0f32; 1326];
        let mut valid_mask = [0u8; 1326];

        // Set a few values to be non-trivial
        oop_range[0] = 0.5;
        oop_range[1] = 0.5;
        ip_range[100] = 1.0;
        oop_cfvs[0] = 0.123;
        oop_cfvs[100] = -0.456;
        ip_cfvs[0] = -0.123;
        ip_cfvs[100] = 0.456;
        valid_mask[0] = 1;
        valid_mask[1] = 1;
        valid_mask[100] = 1;

        TrainingRecord {
            board: vec![0, 4, 8, 12, 16],
            pot: 100.0,
            effective_stack: 50.0,
            oop_range,
            ip_range,
            oop_cfvs,
            ip_cfvs,
            valid_mask,
        }
    }

    #[test]
    fn round_trip_single_record() {
        let record = sample_record();
        let mut file = NamedTempFile::new().unwrap();

        write_record(&mut file, &record).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let loaded = read_record(&mut file).unwrap();
        assert_eq!(record.board, loaded.board);
        assert_eq!(record.pot, loaded.pot);
        assert_eq!(record.effective_stack, loaded.effective_stack);
        assert_eq!(record.oop_range, loaded.oop_range);
        assert_eq!(record.ip_range, loaded.ip_range);
        assert_eq!(record.oop_cfvs, loaded.oop_cfvs);
        assert_eq!(record.ip_cfvs, loaded.ip_cfvs);
        assert_eq!(record.valid_mask, loaded.valid_mask);
    }

    #[test]
    fn round_trip_multiple_records() {
        let r1 = sample_record();
        let mut r2 = sample_record();
        r2.pot = 200.0;
        r2.oop_cfvs[0] = 0.999;
        r2.ip_cfvs[0] = -0.999;

        let mut file = NamedTempFile::new().unwrap();
        write_record(&mut file, &r1).unwrap();
        write_record(&mut file, &r2).unwrap();

        file.seek(SeekFrom::Start(0)).unwrap();

        let loaded1 = read_record(&mut file).unwrap();
        let loaded2 = read_record(&mut file).unwrap();

        assert_eq!(r1.board, loaded1.board);
        assert_eq!(r2.pot, loaded2.pot);
        assert_eq!(r2.oop_cfvs[0], loaded2.oop_cfvs[0]);
        assert_eq!(r2.ip_cfvs[0], loaded2.ip_cfvs[0]);
    }

    #[test]
    fn record_count_is_correct() {
        let record = sample_record();
        let mut file = NamedTempFile::new().unwrap();

        for _ in 0..5 {
            write_record(&mut file, &record).unwrap();
        }

        file.seek(SeekFrom::Start(0)).unwrap();
        let count = count_records(&mut file, 5).unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn record_size_is_consistent() {
        let expected = 1 + 5 + 4 + 4 // board_size, board, pot, stack
            + 1326 * 4   // oop_range
            + 1326 * 4   // ip_range
            + 1326 * 4   // oop_cfvs
            + 1326 * 4   // ip_cfvs
            + 1326; // valid_mask
        assert_eq!(record_size(5), expected);
    }

    #[test]
    fn record_roundtrip_4_card_board() {
        let mut rec = sample_record();
        rec.board = vec![0, 4, 8, 12];

        let mut file = NamedTempFile::new().unwrap();
        write_record(&mut file, &rec).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let loaded = read_record(&mut file).unwrap();
        assert_eq!(rec.board, loaded.board);
        assert_eq!(loaded.board.len(), 4);
        assert_eq!(rec.pot, loaded.pot);
        assert_eq!(rec.effective_stack, loaded.effective_stack);
        assert_eq!(rec.oop_range, loaded.oop_range);
        assert_eq!(rec.ip_range, loaded.ip_range);
        assert_eq!(rec.oop_cfvs, loaded.oop_cfvs);
        assert_eq!(rec.ip_cfvs, loaded.ip_cfvs);
        assert_eq!(rec.valid_mask, loaded.valid_mask);
    }

    #[test]
    fn record_roundtrip_5_card_board() {
        let rec = sample_record();
        let mut file = NamedTempFile::new().unwrap();
        write_record(&mut file, &rec).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let loaded = read_record(&mut file).unwrap();
        assert_eq!(rec.board, loaded.board);
        assert_eq!(loaded.board.len(), 5);
    }

    #[test]
    fn count_records_variable_board() {
        // 5-card records
        let rec5 = sample_record();
        let mut file5 = NamedTempFile::new().unwrap();
        for _ in 0..3 {
            write_record(&mut file5, &rec5).unwrap();
        }
        file5.seek(SeekFrom::Start(0)).unwrap();
        assert_eq!(count_records(&mut file5, 5).unwrap(), 3);

        // 4-card records
        let mut rec4 = sample_record();
        rec4.board = vec![0, 4, 8, 12];
        let mut file4 = NamedTempFile::new().unwrap();
        for _ in 0..4 {
            write_record(&mut file4, &rec4).unwrap();
        }
        file4.seek(SeekFrom::Start(0)).unwrap();
        assert_eq!(count_records(&mut file4, 4).unwrap(), 4);
    }

    #[test]
    fn record_v2_roundtrip() {
        // Verify both oop_cfvs and ip_cfvs survive roundtrip with distinct values
        let mut rec = sample_record();

        // Set distinct patterns in oop_cfvs and ip_cfvs
        for i in 0..NUM_COMBOS {
            rec.oop_cfvs[i] = (i as f32) * 0.001;
            rec.ip_cfvs[i] = -(i as f32) * 0.002;
        }

        let mut file = NamedTempFile::new().unwrap();
        write_record(&mut file, &rec).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let loaded = read_record(&mut file).unwrap();

        // Verify every element roundtrips exactly
        assert_eq!(rec.oop_cfvs, loaded.oop_cfvs);
        assert_eq!(rec.ip_cfvs, loaded.ip_cfvs);

        // Verify they are truly distinct arrays
        assert_ne!(loaded.oop_cfvs, loaded.ip_cfvs);

        // Spot-check a few values
        assert_eq!(loaded.oop_cfvs[500], 500.0 * 0.001);
        assert_eq!(loaded.ip_cfvs[500], -500.0 * 0.002);
    }
}
