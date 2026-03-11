use std::io::{self, Read, Seek, SeekFrom, Write};

pub const NUM_COMBOS: usize = 1326;

/// Fixed size of one training record in bytes.
pub const RECORD_SIZE: usize = 5 // board: 5 × u8
    + 4                          // pot: f32
    + 4                          // effective_stack: f32
    + 1                          // player: u8
    + 4                          // game_value: f32
    + NUM_COMBOS * 4             // oop_range: f32
    + NUM_COMBOS * 4             // ip_range: f32
    + NUM_COMBOS * 4             // cfvs: f32
    + NUM_COMBOS;                // valid_mask: u8

/// One training sample: a single player's perspective of a solved river subgame.
#[derive(Debug, Clone)]
pub struct TrainingRecord {
    pub board: [u8; 5],
    pub pot: f32,
    pub effective_stack: f32,
    pub player: u8, // 0=OOP, 1=IP
    pub game_value: f32,
    pub oop_range: [f32; NUM_COMBOS],
    pub ip_range: [f32; NUM_COMBOS],
    pub cfvs: [f32; NUM_COMBOS],
    pub valid_mask: [u8; NUM_COMBOS],
}

/// Write a single training record in little-endian format.
pub fn write_record<W: Write>(w: &mut W, rec: &TrainingRecord) -> io::Result<()> {
    w.write_all(&rec.board)?;
    w.write_all(&rec.pot.to_le_bytes())?;
    w.write_all(&rec.effective_stack.to_le_bytes())?;
    w.write_all(&[rec.player])?;
    w.write_all(&rec.game_value.to_le_bytes())?;
    for &v in &rec.oop_range {
        w.write_all(&v.to_le_bytes())?;
    }
    for &v in &rec.ip_range {
        w.write_all(&v.to_le_bytes())?;
    }
    for &v in &rec.cfvs {
        w.write_all(&v.to_le_bytes())?;
    }
    w.write_all(&rec.valid_mask)?;
    Ok(())
}

/// Read a single training record from little-endian format.
pub fn read_record<R: Read>(r: &mut R) -> io::Result<TrainingRecord> {
    let mut board = [0u8; 5];
    r.read_exact(&mut board)?;

    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4)?;
    let pot = f32::from_le_bytes(buf4);

    r.read_exact(&mut buf4)?;
    let effective_stack = f32::from_le_bytes(buf4);

    let mut buf1 = [0u8; 1];
    r.read_exact(&mut buf1)?;
    let player = buf1[0];

    r.read_exact(&mut buf4)?;
    let game_value = f32::from_le_bytes(buf4);

    let mut oop_range = [0.0f32; NUM_COMBOS];
    for v in &mut oop_range {
        r.read_exact(&mut buf4)?;
        *v = f32::from_le_bytes(buf4);
    }

    let mut ip_range = [0.0f32; NUM_COMBOS];
    for v in &mut ip_range {
        r.read_exact(&mut buf4)?;
        *v = f32::from_le_bytes(buf4);
    }

    let mut cfvs = [0.0f32; NUM_COMBOS];
    for v in &mut cfvs {
        r.read_exact(&mut buf4)?;
        *v = f32::from_le_bytes(buf4);
    }

    let mut valid_mask = [0u8; NUM_COMBOS];
    r.read_exact(&mut valid_mask)?;

    Ok(TrainingRecord {
        board,
        pot,
        effective_stack,
        player,
        game_value,
        oop_range,
        ip_range,
        cfvs,
        valid_mask,
    })
}

/// Count the number of complete records by dividing file size by record size.
pub fn count_records<S: Seek>(s: &mut S) -> io::Result<u64> {
    let end = s.seek(SeekFrom::End(0))?;
    s.seek(SeekFrom::Start(0))?;
    Ok(end / RECORD_SIZE as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom};
    use tempfile::NamedTempFile;

    fn sample_record() -> TrainingRecord {
        let mut oop_range = [0.0f32; 1326];
        let mut ip_range = [0.0f32; 1326];
        let mut cfvs = [0.0f32; 1326];
        let mut valid_mask = [0u8; 1326];

        // Set a few values to be non-trivial
        oop_range[0] = 0.5;
        oop_range[1] = 0.5;
        ip_range[100] = 1.0;
        cfvs[0] = 0.123;
        cfvs[100] = -0.456;
        valid_mask[0] = 1;
        valid_mask[1] = 1;
        valid_mask[100] = 1;

        TrainingRecord {
            board: [0, 4, 8, 12, 16],
            pot: 100.0,
            effective_stack: 50.0,
            player: 0,
            oop_range,
            ip_range,
            cfvs,
            valid_mask,
            game_value: 0.05,
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
        assert_eq!(record.player, loaded.player);
        assert_eq!(record.oop_range, loaded.oop_range);
        assert_eq!(record.ip_range, loaded.ip_range);
        assert_eq!(record.cfvs, loaded.cfvs);
        assert_eq!(record.valid_mask, loaded.valid_mask);
        assert!((record.game_value - loaded.game_value).abs() < 1e-7);
    }

    #[test]
    fn round_trip_multiple_records() {
        let r1 = sample_record();
        let mut r2 = sample_record();
        r2.player = 1;
        r2.pot = 200.0;
        r2.game_value = -0.1;

        let mut file = NamedTempFile::new().unwrap();
        write_record(&mut file, &r1).unwrap();
        write_record(&mut file, &r2).unwrap();

        file.seek(SeekFrom::Start(0)).unwrap();

        let loaded1 = read_record(&mut file).unwrap();
        let loaded2 = read_record(&mut file).unwrap();

        assert_eq!(r1.board, loaded1.board);
        assert_eq!(r2.pot, loaded2.pot);
        assert_eq!(r2.player, loaded2.player);
    }

    #[test]
    fn record_count_is_correct() {
        let record = sample_record();
        let mut file = NamedTempFile::new().unwrap();

        for _ in 0..5 {
            write_record(&mut file, &record).unwrap();
        }

        file.seek(SeekFrom::Start(0)).unwrap();
        let count = count_records(&mut file).unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn record_size_is_consistent() {
        let expected = 5 + 4 + 4 + 1 + 4 // board, pot, stack, player, game_value
            + 1326 * 4   // oop_range
            + 1326 * 4   // ip_range
            + 1326 * 4   // cfvs
            + 1326; // valid_mask
        assert_eq!(RECORD_SIZE, expected);
    }
}
