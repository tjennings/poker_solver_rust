//! Per-flop bucket file: turn + river bucket assignments for one canonical flop.
//!
//! ## Wire format (little-endian)
//!
//! | Field | Size |
//! |-|-|
//! | Magic `PFB1` | 4 |
//! | flop_cards (3 x encoded card) | 3 |
//! | turn_bucket_count | 2 |
//! | river_bucket_count | 2 |
//! | turn_count (u8) | 1 |
//! | turn_cards (turn_count x encoded card) | turn_count |
//! | turn_buckets (turn_count x 1326 x u16) | turn_count x 1326 x 2 |
//! | For each turn card: | |
//! |   river_count (u8) | 1 |
//! |   river_cards (river_count x encoded card) | river_count |
//! |   river_buckets (river_count x 1326 x u16) | river_count x 1326 x 2 |

use std::io::{self, Read, Write};
use std::path::Path;

use rs_poker::core::Card;

use super::bucket_file::{decode_card, encode_card};

const MAGIC: [u8; 4] = *b"PFB1";
const COMBOS: usize = 1326;

/// Per-flop bucket file holding turn + river bucket assignments for one
/// canonical flop.
#[derive(Debug)]
pub struct PerFlopBucketFile {
    /// The 3 canonical flop cards.
    pub flop_cards: [Card; 3],
    /// Number of turn buckets used.
    pub turn_bucket_count: u16,
    /// Number of river buckets used.
    pub river_bucket_count: u16,
    /// Canonical turn cards for this flop.
    pub turn_cards: Vec<Card>,
    /// Flat: `turn_buckets[turn_idx * COMBOS + combo_idx]`
    pub turn_buckets: Vec<u16>,
    /// Per-turn river cards: `river_cards_per_turn[turn_idx]` = vec of river cards.
    pub river_cards_per_turn: Vec<Vec<Card>>,
    /// Per-turn river buckets: `river_buckets_per_turn[turn_idx][river_idx * COMBOS + combo_idx]`
    pub river_buckets_per_turn: Vec<Vec<u16>>,
}

impl PerFlopBucketFile {
    /// Look up the turn bucket for a given turn index and combo index.
    #[must_use]
    pub fn get_turn_bucket(&self, turn_idx: usize, combo_idx: usize) -> u16 {
        self.turn_buckets[turn_idx * COMBOS + combo_idx]
    }

    /// Look up the river bucket for a given turn index, river index, and combo index.
    #[must_use]
    pub fn get_river_bucket(&self, turn_idx: usize, river_idx: usize, combo_idx: usize) -> u16 {
        self.river_buckets_per_turn[turn_idx][river_idx * COMBOS + combo_idx]
    }

    /// Write the per-flop bucket file to an arbitrary writer.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing fails.
    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(&MAGIC)?;
        for &c in &self.flop_cards {
            w.write_all(&[encode_card(c)])?;
        }
        w.write_all(&self.turn_bucket_count.to_le_bytes())?;
        w.write_all(&self.river_bucket_count.to_le_bytes())?;
        #[allow(clippy::cast_possible_truncation)]
        w.write_all(&[self.turn_cards.len() as u8])?;
        for &c in &self.turn_cards {
            w.write_all(&[encode_card(c)])?;
        }
        for &b in &self.turn_buckets {
            w.write_all(&b.to_le_bytes())?;
        }
        for (turn_idx, river_cards) in self.river_cards_per_turn.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            w.write_all(&[river_cards.len() as u8])?;
            for &c in river_cards {
                w.write_all(&[encode_card(c)])?;
            }
            for &b in &self.river_buckets_per_turn[turn_idx] {
                w.write_all(&b.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Read a per-flop bucket file from an arbitrary reader.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the magic bytes don't match or reading fails.
    pub fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }

        let mut card_buf = [0u8; 1];
        let mut flop_raw = [0u8; 3];
        r.read_exact(&mut flop_raw)?;
        let flop_cards = [
            decode_card(flop_raw[0]),
            decode_card(flop_raw[1]),
            decode_card(flop_raw[2]),
        ];

        let mut u16_buf = [0u8; 2];
        r.read_exact(&mut u16_buf)?;
        let turn_bucket_count = u16::from_le_bytes(u16_buf);
        r.read_exact(&mut u16_buf)?;
        let river_bucket_count = u16::from_le_bytes(u16_buf);

        let mut u8_buf = [0u8; 1];
        r.read_exact(&mut u8_buf)?;
        let turn_count = u8_buf[0] as usize;

        let mut turn_cards = Vec::with_capacity(turn_count);
        for _ in 0..turn_count {
            r.read_exact(&mut card_buf)?;
            turn_cards.push(decode_card(card_buf[0]));
        }

        let total_turn_buckets = turn_count * COMBOS;
        let mut turn_data = vec![0u8; total_turn_buckets * 2];
        r.read_exact(&mut turn_data)?;
        let mut turn_buckets = Vec::with_capacity(total_turn_buckets);
        for chunk in turn_data.chunks_exact(2) {
            turn_buckets.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        }

        let mut river_cards_per_turn = Vec::with_capacity(turn_count);
        let mut river_buckets_per_turn = Vec::with_capacity(turn_count);
        for _ in 0..turn_count {
            r.read_exact(&mut u8_buf)?;
            let river_count = u8_buf[0] as usize;
            let mut river_cards = Vec::with_capacity(river_count);
            for _ in 0..river_count {
                r.read_exact(&mut card_buf)?;
                river_cards.push(decode_card(card_buf[0]));
            }
            let total_river_buckets = river_count * COMBOS;
            let mut river_data = vec![0u8; total_river_buckets * 2];
            r.read_exact(&mut river_data)?;
            let mut river_buckets = Vec::with_capacity(total_river_buckets);
            for chunk in river_data.chunks_exact(2) {
                river_buckets.push(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
            river_cards_per_turn.push(river_cards);
            river_buckets_per_turn.push(river_buckets);
        }

        Ok(Self {
            flop_cards,
            turn_bucket_count,
            river_bucket_count,
            turn_cards,
            turn_buckets,
            river_cards_per_turn,
            river_buckets_per_turn,
        })
    }

    /// Save the per-flop bucket file to disk.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if file creation or writing fails.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut w = io::BufWriter::new(file);
        self.write_to(&mut w)
    }

    /// Load a per-flop bucket file from disk.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be opened or contains invalid data.
    pub fn load(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mut r = io::BufReader::new(file);
        Self::read_from(&mut r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rs_poker::core::{Card, Suit, Value};

    fn make_test_flop() -> [Card; 3] {
        [
            Card::new(Value::Queen, Suit::Spade),
            Card::new(Value::Jack, Suit::Heart),
            Card::new(Value::Two, Suit::Diamond),
        ]
    }

    #[test]
    fn round_trip_write_read() {
        let flop = make_test_flop();
        let turn_cards = vec![
            Card::new(Value::Ace, Suit::Club),
            Card::new(Value::King, Suit::Club),
        ];
        let river_cards_per_turn = vec![
            vec![
                Card::new(Value::Ten, Suit::Club),
                Card::new(Value::Nine, Suit::Club),
                Card::new(Value::Eight, Suit::Club),
            ],
            vec![
                Card::new(Value::Seven, Suit::Club),
                Card::new(Value::Six, Suit::Club),
                Card::new(Value::Five, Suit::Club),
            ],
        ];

        let pf = PerFlopBucketFile {
            flop_cards: flop,
            turn_bucket_count: 10,
            river_bucket_count: 10,
            turn_cards: turn_cards.clone(),
            turn_buckets: vec![0u16; 2 * COMBOS],
            river_cards_per_turn: river_cards_per_turn.clone(),
            river_buckets_per_turn: vec![
                vec![0u16; 3 * COMBOS],
                vec![0u16; 3 * COMBOS],
            ],
        };

        let mut buf = Vec::new();
        pf.write_to(&mut buf).unwrap();

        let mut cursor = std::io::Cursor::new(&buf);
        let loaded = PerFlopBucketFile::read_from(&mut cursor).unwrap();

        assert_eq!(loaded.flop_cards, flop);
        assert_eq!(loaded.turn_bucket_count, 10);
        assert_eq!(loaded.river_bucket_count, 10);
        assert_eq!(loaded.turn_cards, turn_cards);
        assert_eq!(loaded.turn_buckets.len(), 2 * COMBOS);
        assert_eq!(loaded.river_cards_per_turn.len(), 2);
        assert_eq!(loaded.river_buckets_per_turn[0].len(), 3 * COMBOS);
    }

    #[test]
    fn get_turn_bucket_lookup() {
        let mut pf = PerFlopBucketFile {
            flop_cards: make_test_flop(),
            turn_bucket_count: 10,
            river_bucket_count: 10,
            turn_cards: vec![Card::new(Value::Ace, Suit::Club)],
            turn_buckets: vec![0u16; COMBOS],
            river_cards_per_turn: vec![vec![]],
            river_buckets_per_turn: vec![vec![]],
        };
        pf.turn_buckets[42] = 7;
        assert_eq!(pf.get_turn_bucket(0, 42), 7);
    }

    #[test]
    fn get_turn_bucket_multi_turn() {
        let mut pf = PerFlopBucketFile {
            flop_cards: make_test_flop(),
            turn_bucket_count: 10,
            river_bucket_count: 10,
            turn_cards: vec![
                Card::new(Value::Ace, Suit::Club),
                Card::new(Value::King, Suit::Club),
            ],
            turn_buckets: vec![0u16; 2 * COMBOS],
            river_cards_per_turn: vec![vec![], vec![]],
            river_buckets_per_turn: vec![vec![], vec![]],
        };
        // Set bucket for turn_idx=1, combo_idx=100
        pf.turn_buckets[1 * COMBOS + 100] = 9;
        assert_eq!(pf.get_turn_bucket(1, 100), 9);
        // Confirm turn_idx=0 is still 0
        assert_eq!(pf.get_turn_bucket(0, 100), 0);
    }

    #[test]
    fn get_river_bucket_lookup() {
        let river_card = Card::new(Value::Ten, Suit::Club);
        let mut pf = PerFlopBucketFile {
            flop_cards: make_test_flop(),
            turn_bucket_count: 10,
            river_bucket_count: 10,
            turn_cards: vec![Card::new(Value::Ace, Suit::Club)],
            turn_buckets: vec![0u16; COMBOS],
            river_cards_per_turn: vec![vec![river_card]],
            river_buckets_per_turn: vec![vec![0u16; COMBOS]],
        };
        pf.river_buckets_per_turn[0][99] = 5;
        assert_eq!(pf.get_river_bucket(0, 0, 99), 5);
    }

    #[test]
    fn get_river_bucket_multi_river() {
        let mut pf = PerFlopBucketFile {
            flop_cards: make_test_flop(),
            turn_bucket_count: 10,
            river_bucket_count: 10,
            turn_cards: vec![Card::new(Value::Ace, Suit::Club)],
            turn_buckets: vec![0u16; COMBOS],
            river_cards_per_turn: vec![vec![
                Card::new(Value::Ten, Suit::Club),
                Card::new(Value::Nine, Suit::Club),
            ]],
            river_buckets_per_turn: vec![vec![0u16; 2 * COMBOS]],
        };
        // Set bucket for river_idx=1, combo_idx=50
        pf.river_buckets_per_turn[0][1 * COMBOS + 50] = 3;
        assert_eq!(pf.get_river_bucket(0, 1, 50), 3);
        assert_eq!(pf.get_river_bucket(0, 0, 50), 0);
    }

    #[test]
    fn round_trip_with_nonzero_buckets() {
        let flop = make_test_flop();
        let turn_cards = vec![Card::new(Value::Ace, Suit::Club)];
        let river_cards_per_turn = vec![vec![Card::new(Value::Ten, Suit::Club)]];

        let mut turn_buckets = vec![0u16; COMBOS];
        for i in 0..COMBOS {
            #[allow(clippy::cast_possible_truncation)]
            { turn_buckets[i] = (i % 10) as u16; }
        }

        let mut river_buckets = vec![0u16; COMBOS];
        for i in 0..COMBOS {
            #[allow(clippy::cast_possible_truncation)]
            { river_buckets[i] = (i % 8) as u16; }
        }

        let pf = PerFlopBucketFile {
            flop_cards: flop,
            turn_bucket_count: 10,
            river_bucket_count: 8,
            turn_cards: turn_cards.clone(),
            turn_buckets: turn_buckets.clone(),
            river_cards_per_turn: river_cards_per_turn.clone(),
            river_buckets_per_turn: vec![river_buckets.clone()],
        };

        let mut buf = Vec::new();
        pf.write_to(&mut buf).unwrap();
        let loaded = PerFlopBucketFile::read_from(&mut std::io::Cursor::new(&buf)).unwrap();

        assert_eq!(loaded.turn_buckets, turn_buckets);
        assert_eq!(loaded.river_buckets_per_turn[0], river_buckets);
    }

    #[test]
    fn read_rejects_bad_magic() {
        let mut buf = vec![0u8; 20];
        buf[0..4].copy_from_slice(b"JUNK");
        let err = PerFlopBucketFile::read_from(&mut std::io::Cursor::new(&buf)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn save_and_load_file() {
        let flop = make_test_flop();
        let pf = PerFlopBucketFile {
            flop_cards: flop,
            turn_bucket_count: 5,
            river_bucket_count: 5,
            turn_cards: vec![Card::new(Value::Ace, Suit::Club)],
            turn_buckets: vec![1u16; COMBOS],
            river_cards_per_turn: vec![vec![Card::new(Value::Ten, Suit::Club)]],
            river_buckets_per_turn: vec![vec![2u16; COMBOS]],
        };

        let dir = tempfile::tempdir().expect("tempdir failed");
        let path = dir.path().join("test.pfb");
        pf.save(&path).unwrap();
        let loaded = PerFlopBucketFile::load(&path).unwrap();

        assert_eq!(loaded.flop_cards, flop);
        assert_eq!(loaded.turn_buckets, vec![1u16; COMBOS]);
        assert_eq!(loaded.river_buckets_per_turn[0], vec![2u16; COMBOS]);
    }
}
