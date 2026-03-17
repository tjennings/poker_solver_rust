//! Binary format for storing bucket assignments per street.
//!
//! Used by the clustering pipeline to persist results and by the training
//! engine to load bucket lookups at runtime.
//!
//! ## Wire format (little-endian)
//!
//! | Offset | Size | Field |
//! |-|-|-|
//! | 0 | 4 | Magic `BKT2` |
//! | 4 | 1 | Version (1 or 2) |
//! | 5 | 1 | Street (0-3) |
//! | 6 | 2 | `bucket_count` |
//! | 8 | 4 | `board_count` |
//! | 12 | 2 | `combos_per_board` |
//! | 14 | B*8 | Board table (`PackedBoard` u64s, v2+ only, B = `board_count`) |
//! | 14+B*8 | N*2 | Flat `u16` bucket assignments |

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use rs_poker::core::{Card, Suit, Value};

use super::Street;

// ---------------------------------------------------------------------------
// PackedBoard — compact hashable board key
// ---------------------------------------------------------------------------

/// Compact canonical board representation for hashing and serialization.
///
/// Each card occupies 8 bits: `(value_rank << 2) | (suit as u8)`, packed
/// left-to-right from the most-significant byte. Up to 5 cards fit in 40 of
/// the 64 available bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PackedBoard(pub u64);

impl PackedBoard {
    /// Pack a slice of canonical cards into a `u64`.
    ///
    /// # Panics
    ///
    /// Panics if `cards.len() > 5`.
    #[must_use]
    pub fn from_cards(cards: &[Card]) -> Self {
        assert!(cards.len() <= 5, "PackedBoard supports at most 5 cards");
        let mut packed: u64 = 0;
        for (i, card) in cards.iter().enumerate() {
            let byte = encode_card(*card);
            packed |= u64::from(byte) << (56 - i * 8);
        }
        Self(packed)
    }

    /// Unpack back to a vector of cards.
    #[must_use]
    pub fn to_cards(self, num_cards: usize) -> Vec<Card> {
        (0..num_cards)
            .map(|i| {
                #[allow(clippy::cast_possible_truncation)]
                let byte = (self.0 >> (56 - i * 8)) as u8;
                decode_card(byte)
            })
            .collect()
    }
}

/// Encode a single card as an 8-bit value: `(value_rank << 2) | suit_index`.
pub(crate) fn encode_card(card: Card) -> u8 {
    let value_rank = crate::card_utils::value_rank(card.value);
    let suit_idx = card.suit as u8;
    (value_rank << 2) | suit_idx
}

/// Decode an 8-bit packed card back into `Card`.
pub(crate) fn decode_card(byte: u8) -> Card {
    let rank = byte >> 2;
    let suit_idx = byte & 0x03;
    // value_rank maps Two→2 … Ace→14, so discriminant = rank - 2.
    let value = Value::from(rank - 2);
    let suit = Suit::from(suit_idx);
    Card::new(value, suit)
}

// ---------------------------------------------------------------------------
// BucketFile — binary bucket storage
// ---------------------------------------------------------------------------

const MAGIC: [u8; 4] = *b"BKT2";
pub(crate) const VERSION: u8 = 2;

/// Header size in bytes: 4 (magic) + 1 (version) + 1 (street) + 2 (`bucket_count`)
///                       + 4 (`board_count`) + 2 (`combos_per_board`) = 14
const HEADER_SIZE: usize = 14;

/// Metadata describing a bucket file's dimensions.
#[derive(Debug, Clone)]
pub struct BucketFileHeader {
    pub street: Street,
    pub bucket_count: u16,
    pub board_count: u32,
    pub combos_per_board: u16,
    pub version: u8,
}

/// A bucket file holding per-combo bucket assignments for every board.
///
/// Buckets are stored flat: `buckets[board_idx * combos_per_board + combo_idx]`.
#[derive(Debug)]
pub struct BucketFile {
    pub header: BucketFileHeader,
    /// Canonical boards in order, one per `board_idx`. Empty for v1 files.
    pub boards: Vec<PackedBoard>,
    /// Flat array: `buckets[board_idx * combos_per_board + combo_idx]`
    pub buckets: Vec<u16>,
}

impl BucketFile {
    /// Write the bucket file to an arbitrary writer.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing to the underlying writer fails.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&MAGIC)?;
        writer.write_all(&[self.header.version])?;
        writer.write_all(&[self.header.street as u8])?;
        writer.write_all(&self.header.bucket_count.to_le_bytes())?;
        writer.write_all(&self.header.board_count.to_le_bytes())?;
        writer.write_all(&self.header.combos_per_board.to_le_bytes())?;

        // Write board table (v2+).
        for &board in &self.boards {
            writer.write_all(&board.0.to_le_bytes())?;
        }

        // Write bucket data as raw little-endian u16 values.
        for &bucket in &self.buckets {
            writer.write_all(&bucket.to_le_bytes())?;
        }

        Ok(())
    }

    /// Save the bucket file to disk.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if file creation or writing fails.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.write_to(&mut writer)
    }

    /// Read a bucket file from an arbitrary reader, validating header fields.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if:
    /// - The magic bytes do not match `BKT2`.
    /// - The version is unsupported.
    /// - The street value is out of range.
    /// - Reading from the underlying reader fails.
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut header_buf = [0u8; HEADER_SIZE];
        reader.read_exact(&mut header_buf)?;

        // Validate magic bytes.
        if header_buf[0..4] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "bad magic: expected {:?}, got {:?}",
                    MAGIC,
                    &header_buf[0..4]
                ),
            ));
        }

        // Validate version — accept v1 and v2, reject future versions.
        let version = header_buf[4];
        if version > VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported version: {version} (max supported: {VERSION})"),
            ));
        }

        let street = Street::from_u8(header_buf[5]).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid street value: {}", header_buf[5]),
            )
        })?;
        let bucket_count = u16::from_le_bytes([header_buf[6], header_buf[7]]);
        let board_count = u32::from_le_bytes([header_buf[8], header_buf[9], header_buf[10], header_buf[11]]);
        let combos_per_board = u16::from_le_bytes([header_buf[12], header_buf[13]]);

        // Read board table for v2+.
        let boards = if version >= 2 {
            let mut boards = Vec::with_capacity(board_count as usize);
            for _ in 0..board_count {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                boards.push(PackedBoard(u64::from_le_bytes(buf)));
            }
            boards
        } else {
            Vec::new()
        };

        let total = board_count as usize * combos_per_board as usize;
        let mut data_buf = vec![0u8; total * 2];
        reader.read_exact(&mut data_buf)?;

        let mut buckets = Vec::with_capacity(total);
        for chunk in data_buf.chunks_exact(2) {
            buckets.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        }

        let header = BucketFileHeader {
            street,
            bucket_count,
            board_count,
            combos_per_board,
            version,
        };

        Ok(Self { header, boards, buckets })
    }

    /// Load a bucket file from disk.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be opened or contains invalid data.
    pub fn load(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::read_from(&mut reader)
    }

    /// Build a lookup map from `PackedBoard` to board index.
    ///
    /// Used by the clustering pipeline to find a board's index for bucket
    /// lookups when building potential-aware histograms.
    #[must_use]
    pub fn board_index_map(&self) -> rustc_hash::FxHashMap<PackedBoard, u32> {
        self.boards
            .iter()
            .enumerate()
            .map(|(i, &board)| (board, i as u32))
            .collect()
    }

    /// Look up the bucket assignment for a specific board and combo index.
    #[must_use]
    pub fn get_bucket(&self, board_idx: u32, combo_idx: u16) -> u16 {
        let idx = board_idx as usize * self.header.combos_per_board as usize + combo_idx as usize;
        self.buckets[idx]
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use rs_poker::core::{Card, Suit, Value};

    /// Helper: create a bucket file with deterministic data.
    fn make_test_file(boards: u32, combos: u16, buckets_k: u16) -> BucketFile {
        let total = boards as usize * combos as usize;
        let mut buckets = Vec::with_capacity(total);
        for i in 0..total {
            #[allow(clippy::cast_possible_truncation)]
            buckets.push((i % buckets_k as usize) as u16);
        }
        // Generate dummy packed boards for each board index.
        let board_entries: Vec<PackedBoard> = (0..boards)
            .map(|i| PackedBoard(u64::from(i)))
            .collect();
        BucketFile {
            header: BucketFileHeader {
                street: Street::Flop,
                bucket_count: buckets_k,
                board_count: boards,
                combos_per_board: combos,
                version: VERSION,
            },
            boards: board_entries,
            buckets,
        }
    }

    #[test]
    fn test_bucket_file_round_trip() {
        let original = make_test_file(3, 1326, 200);

        let mut buf = Vec::new();
        original.write_to(&mut buf).expect("write failed");

        // Header (14) + board table (3 * 8) + bucket data (3 * 1326 * 2).
        assert_eq!(buf.len(), HEADER_SIZE + 3 * 8 + 3 * 1326 * 2);

        let mut cursor = Cursor::new(&buf);
        let loaded = BucketFile::read_from(&mut cursor).expect("read failed");

        assert_eq!(loaded.header.street, Street::Flop);
        assert_eq!(loaded.header.bucket_count, 200);
        assert_eq!(loaded.header.board_count, 3);
        assert_eq!(loaded.header.combos_per_board, 1326);
        assert_eq!(loaded.buckets.len(), original.buckets.len());
        assert_eq!(loaded.buckets, original.buckets);

        // Spot-check get_bucket.
        for board in 0..3_u32 {
            for combo in [0_u16, 100, 500, 1325] {
                assert_eq!(
                    loaded.get_bucket(board, combo),
                    original.get_bucket(board, combo),
                );
            }
        }
    }

    #[test]
    fn test_bucket_file_save_load() {
        let dir = tempfile::tempdir().expect("tempdir failed");
        let path = dir.path().join("test.bkt");

        let original = make_test_file(5, 1326, 500);
        original.save(&path).expect("save failed");

        let loaded = BucketFile::load(&path).expect("load failed");

        assert_eq!(loaded.header.street, original.header.street);
        assert_eq!(loaded.header.bucket_count, original.header.bucket_count);
        assert_eq!(loaded.header.board_count, original.header.board_count);
        assert_eq!(loaded.header.combos_per_board, original.header.combos_per_board);
        assert_eq!(loaded.buckets, original.buckets);
    }

    #[test]
    fn test_bucket_file_bad_magic() {
        let mut buf = vec![0u8; HEADER_SIZE + 4]; // minimal data
        buf[0..4].copy_from_slice(b"JUNK");

        let mut cursor = Cursor::new(&buf);
        let err = BucketFile::read_from(&mut cursor).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("bad magic"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn bucket_file_v2_round_trip_with_boards() {
        let boards = vec![
            PackedBoard::from_cards(&[
                Card::new(Value::Ace, Suit::Spade),
                Card::new(Value::King, Suit::Spade),
                Card::new(Value::Queen, Suit::Heart),
            ]),
            PackedBoard::from_cards(&[
                Card::new(Value::Two, Suit::Spade),
                Card::new(Value::Three, Suit::Heart),
                Card::new(Value::Four, Suit::Diamond),
            ]),
        ];
        let bf = BucketFile {
            header: BucketFileHeader {
                street: Street::Flop,
                bucket_count: 10,
                board_count: 2,
                combos_per_board: 1326,
                version: 2,
            },
            boards: boards.clone(),
            buckets: vec![0_u16; 2 * 1326],
        };

        let mut buf = Vec::new();
        bf.write_to(&mut buf).unwrap();

        let loaded = BucketFile::read_from(&mut Cursor::new(buf)).unwrap();
        assert_eq!(loaded.boards.len(), 2);
        assert_eq!(loaded.boards[0], boards[0]);
        assert_eq!(loaded.boards[1], boards[1]);
        assert_eq!(loaded.header.version, 2);
    }

    #[test]
    fn bucket_file_v1_read_no_boards() {
        // Build a v1-format buffer manually: header + bucket data, no board table.
        let board_count: u32 = 1;
        let combos_per_board: u16 = 4;
        let bucket_count: u16 = 2;
        let buckets: Vec<u16> = vec![0, 1, 0, 1];

        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.push(1); // version 1
        buf.push(0); // street = Preflop
        buf.extend_from_slice(&bucket_count.to_le_bytes());
        buf.extend_from_slice(&board_count.to_le_bytes());
        buf.extend_from_slice(&combos_per_board.to_le_bytes());
        for &b in &buckets {
            buf.extend_from_slice(&b.to_le_bytes());
        }

        let loaded = BucketFile::read_from(&mut Cursor::new(buf)).unwrap();
        assert_eq!(loaded.header.version, 1);
        assert!(loaded.boards.is_empty());
        assert_eq!(loaded.buckets, buckets);
    }

    // -- PackedBoard tests --------------------------------------------------

    #[test]
    fn packed_board_round_trip_flop() {
        let cards = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Heart),
        ];
        let packed = PackedBoard::from_cards(&cards);
        let recovered = packed.to_cards(3);
        assert_eq!(recovered, cards.to_vec());
    }

    #[test]
    fn packed_board_round_trip_river() {
        let cards = [
            Card::new(Value::Two, Suit::Club),
            Card::new(Value::Five, Suit::Diamond),
            Card::new(Value::Ten, Suit::Heart),
            Card::new(Value::Jack, Suit::Spade),
            Card::new(Value::Ace, Suit::Spade),
        ];
        let packed = PackedBoard::from_cards(&cards);
        let recovered = packed.to_cards(5);
        assert_eq!(recovered, cards.to_vec());
    }

    #[test]
    fn packed_board_hash_eq() {
        use std::collections::HashMap;
        let cards = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Queen, Suit::Diamond),
        ];
        let p1 = PackedBoard::from_cards(&cards);
        let p2 = PackedBoard::from_cards(&cards);
        assert_eq!(p1, p2);
        let mut map = HashMap::new();
        map.insert(p1, 42_u32);
        assert_eq!(map[&p2], 42);
    }

    #[test]
    fn board_index_map_round_trip() {
        let boards = vec![
            PackedBoard::from_cards(&[
                Card::new(Value::Ace, Suit::Spade),
                Card::new(Value::King, Suit::Spade),
                Card::new(Value::Queen, Suit::Heart),
            ]),
            PackedBoard::from_cards(&[
                Card::new(Value::Two, Suit::Spade),
                Card::new(Value::Three, Suit::Heart),
                Card::new(Value::Four, Suit::Diamond),
            ]),
        ];
        let bf = BucketFile {
            header: BucketFileHeader {
                street: Street::Flop,
                bucket_count: 10,
                board_count: 2,
                combos_per_board: 1326,
                version: VERSION,
            },
            boards: boards.clone(),
            buckets: vec![0; 2 * 1326],
        };
        let map = bf.board_index_map();
        assert_eq!(map.len(), 2);
        assert_eq!(map[&boards[0]], 0);
        assert_eq!(map[&boards[1]], 1);
    }

    #[test]
    fn board_index_map_empty_boards() {
        let bf = BucketFile {
            header: BucketFileHeader {
                street: Street::Flop,
                bucket_count: 10,
                board_count: 0,
                combos_per_board: 1326,
                version: VERSION,
            },
            boards: vec![],
            buckets: vec![],
        };
        let map = bf.board_index_map();
        assert!(map.is_empty());
    }

    #[test]
    fn packed_board_different_boards_different_keys() {
        let a = PackedBoard::from_cards(&[
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Spade),
        ]);
        let b = PackedBoard::from_cards(&[
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Jack, Suit::Spade),
        ]);
        assert_ne!(a, b);
    }
}
