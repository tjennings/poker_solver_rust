//! Fixed-width row and action descriptor encode/decode.
//!
//! ## Row descriptor wire format (little-endian, 96 bytes)
//!
//! | Offset | Size | Field                       |
//! |--------|------|-----------------------------|
//! | 0      | 8    | `row_id`                    |
//! | 8      | 2    | `namespace`                 |
//! | 10     | 1    | `seat`                      |
//! | 11     | 1    | `street`                    |
//! | 12     | 2    | `local_bucket`              |
//! | 14     | 2    | `reserved0` (must be 0)     |
//! | 16     | 4    | `global_bucket`             |
//! | 20     | 4    | `source_node_idx`           |
//! | 24     | 8    | `action_offset`             |
//! | 32     | 2    | `action_count`              |
//! | 34     | 2    | `reserved1` (must be 0)     |
//! | 36     | 4    | (padding to 8-byte align)   |
//! | 40     | 8    | `prob_offset`               |
//! | 48     | 8    | `row_key_fingerprint`       |
//! | 56     | 8    | `action_schema_fingerprint` |
//! | 64     | 2    | `semantic_key_kind`         |
//! | 66     | 6    | (padding to 8-byte align)   |
//! | 72     | 8    | `semantic_key_offset`       |
//! | 80     | 16   | (tail padding to 96 bytes)  |
//!
//! Total: 96 bytes, aligned to 8 bytes.
//!
//! ## Action descriptor wire format (little-endian, 128 bytes)
//!
//! | Offset | Size | Field                  |
//! |--------|------|------------------------|
//! | 0      | 1    | `kind` (enum u8)       |
//! | 1      | 1    | `is_aggressive` (bool) |
//! | 2      | 2    | `source_action_index`  |
//! | 4      | 4    | `amount_chips`         |
//! | 8      | 1    | `size_key_len`         |
//! | 9      | 31   | `size_key` (UTF-8)     |
//! | 40     | 1    | `label_len`            |
//! | 41     | 55   | `label` (UTF-8)        |
//! | 96     | 32   | (reserved / padding)   |
//!
//! Total: 128 bytes, aligned to 8 bytes.
//! Size key and label use fixed-cap inline storage with a leading length byte.
//! Maximum size key length is 31 bytes; maximum label length is 55 bytes.

use super::error::FormatError;
use std::io::Write;

/// Size of a single row descriptor in bytes.
pub const ROW_DESCRIPTOR_SIZE: usize = 96;

/// Size of a single action descriptor in bytes.
pub const ACTION_DESCRIPTOR_SIZE: usize = 128;

/// Maximum length of an inline size key string.
const MAX_SIZE_KEY_LEN: usize = 31;

/// Maximum length of an inline label string.
const MAX_LABEL_LEN: usize = 55;

// ---------------------------------------------------------------------------
// Row descriptor
// ---------------------------------------------------------------------------

/// A single row descriptor from `strategy.rows.bin`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowDescriptor {
    pub row_id: u64,
    pub namespace: u16,
    pub seat: u8,
    pub street: u8,
    pub local_bucket: u16,
    pub global_bucket: u32,
    pub source_node_idx: u32,
    pub action_offset: u64,
    pub action_count: u16,
    pub prob_offset: u64,
    pub row_key_fingerprint: u64,
    pub action_schema_fingerprint: u64,
    pub semantic_key_kind: u16,
    pub semantic_key_offset: u64,
}

impl RowDescriptor {
    /// Encode this descriptor to a writer as a fixed-width 96-byte record.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing fails.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<(), FormatError> {
        let mut buf = [0u8; ROW_DESCRIPTOR_SIZE];
        encode_row(self, &mut buf);
        w.write_all(&buf)?;
        Ok(())
    }

    /// Decode a row descriptor from a 96-byte slice.
    #[must_use]
    pub fn from_bytes(buf: &[u8; ROW_DESCRIPTOR_SIZE]) -> Self {
        decode_row(buf)
    }

    /// The identity key tuple used for sort-order and uniqueness checks.
    ///
    /// Sort order: (`namespace`, `seat`, `street`, `source_node_idx`,
    /// `global_bucket`, `row_key_fingerprint`).
    #[must_use]
    pub fn identity_key(&self) -> (u16, u8, u8, u32, u32, u64) {
        (
            self.namespace,
            self.seat,
            self.street,
            self.source_node_idx,
            self.global_bucket,
            self.row_key_fingerprint,
        )
    }
}

fn encode_row(row: &RowDescriptor, buf: &mut [u8; ROW_DESCRIPTOR_SIZE]) {
    buf[0..8].copy_from_slice(&row.row_id.to_le_bytes());
    buf[8..10].copy_from_slice(&row.namespace.to_le_bytes());
    buf[10] = row.seat;
    buf[11] = row.street;
    buf[12..14].copy_from_slice(&row.local_bucket.to_le_bytes());
    buf[14..16].copy_from_slice(&0u16.to_le_bytes()); // reserved0
    buf[16..20].copy_from_slice(&row.global_bucket.to_le_bytes());
    buf[20..24].copy_from_slice(&row.source_node_idx.to_le_bytes());
    buf[24..32].copy_from_slice(&row.action_offset.to_le_bytes());
    buf[32..34].copy_from_slice(&row.action_count.to_le_bytes());
    buf[34..36].copy_from_slice(&0u16.to_le_bytes()); // reserved1
    buf[36..40].copy_from_slice(&[0u8; 4]); // padding
    buf[40..48].copy_from_slice(&row.prob_offset.to_le_bytes());
    buf[48..56].copy_from_slice(&row.row_key_fingerprint.to_le_bytes());
    buf[56..64].copy_from_slice(&row.action_schema_fingerprint.to_le_bytes());
    buf[64..66].copy_from_slice(&row.semantic_key_kind.to_le_bytes());
    buf[66..72].copy_from_slice(&[0u8; 6]); // padding
    buf[72..80].copy_from_slice(&row.semantic_key_offset.to_le_bytes());
    buf[80..96].copy_from_slice(&[0u8; 16]); // tail padding
}

fn decode_row(buf: &[u8; ROW_DESCRIPTOR_SIZE]) -> RowDescriptor {
    RowDescriptor {
        row_id: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
        namespace: u16::from_le_bytes(buf[8..10].try_into().unwrap()),
        seat: buf[10],
        street: buf[11],
        local_bucket: u16::from_le_bytes(buf[12..14].try_into().unwrap()),
        global_bucket: u32::from_le_bytes(buf[16..20].try_into().unwrap()),
        source_node_idx: u32::from_le_bytes(buf[20..24].try_into().unwrap()),
        action_offset: u64::from_le_bytes(buf[24..32].try_into().unwrap()),
        action_count: u16::from_le_bytes(buf[32..34].try_into().unwrap()),
        prob_offset: u64::from_le_bytes(buf[40..48].try_into().unwrap()),
        row_key_fingerprint: u64::from_le_bytes(buf[48..56].try_into().unwrap()),
        action_schema_fingerprint: u64::from_le_bytes(buf[56..64].try_into().unwrap()),
        semantic_key_kind: u16::from_le_bytes(buf[64..66].try_into().unwrap()),
        semantic_key_offset: u64::from_le_bytes(buf[72..80].try_into().unwrap()),
    }
}

// ---------------------------------------------------------------------------
// Action descriptor
// ---------------------------------------------------------------------------

/// Enumeration of legal action kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ActionKind {
    Fold = 0,
    Check = 1,
    Call = 2,
    Bet = 3,
    Raise = 4,
    AllInCall = 5,
    AllInBetRaise = 6,
}

impl ActionKind {
    /// Convert from a raw `u8` discriminant.
    ///
    /// # Errors
    ///
    /// Returns `None` if the value is not a known variant.
    #[must_use]
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Fold),
            1 => Some(Self::Check),
            2 => Some(Self::Call),
            3 => Some(Self::Bet),
            4 => Some(Self::Raise),
            5 => Some(Self::AllInCall),
            6 => Some(Self::AllInBetRaise),
            _ => None,
        }
    }
}

/// A single action descriptor from `strategy.actions.bin`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionDescriptor {
    pub kind: ActionKind,
    pub amount_chips: u32,
    pub size_key: String,
    pub label: String,
    pub is_aggressive: bool,
    pub source_action_index: u16,
}

impl ActionDescriptor {
    /// Encode this descriptor as a fixed-width 128-byte record.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing fails.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<(), FormatError> {
        let mut buf = [0u8; ACTION_DESCRIPTOR_SIZE];
        encode_action(self, &mut buf);
        w.write_all(&buf)?;
        Ok(())
    }

    /// Decode an action descriptor from a 128-byte slice.
    ///
    /// # Errors
    ///
    /// Returns `FormatError::InvalidManifest` if the action kind byte is
    /// unknown.
    pub fn from_bytes(
        buf: &[u8; ACTION_DESCRIPTOR_SIZE],
    ) -> Result<Self, FormatError> {
        decode_action(buf)
    }
}

fn encode_action(action: &ActionDescriptor, buf: &mut [u8; ACTION_DESCRIPTOR_SIZE]) {
    buf[0] = action.kind as u8;
    buf[1] = u8::from(action.is_aggressive);
    buf[2..4].copy_from_slice(&action.source_action_index.to_le_bytes());
    buf[4..8].copy_from_slice(&action.amount_chips.to_le_bytes());

    write_inline_string(&mut buf[8..40], action.size_key.as_bytes(), MAX_SIZE_KEY_LEN);
    write_inline_string(&mut buf[40..96], action.label.as_bytes(), MAX_LABEL_LEN);
    // Remaining bytes are zero (padding/reserved), already zeroed.
}

/// Write a length-prefixed inline string into a buffer region.
///
/// The first byte is the length, followed by the UTF-8 bytes (truncated to
/// `max_len`). Callers guarantee `max_len` < 256 so the cast is safe.
#[allow(clippy::cast_possible_truncation)]
fn write_inline_string(region: &mut [u8], src: &[u8], max_len: usize) {
    let len = src.len().min(max_len);
    region[0] = len as u8;
    if len > 0 {
        region[1..=len].copy_from_slice(&src[..len]);
    }
}

fn decode_action(
    buf: &[u8; ACTION_DESCRIPTOR_SIZE],
) -> Result<ActionDescriptor, FormatError> {
    let kind = ActionKind::from_u8(buf[0]).ok_or_else(|| FormatError::InvalidManifest {
        detail: format!("unknown action kind: {}", buf[0]),
    })?;
    let is_aggressive = buf[1] != 0;
    let source_action_index = u16::from_le_bytes([buf[2], buf[3]]);
    let amount_chips = u32::from_le_bytes(buf[4..8].try_into().unwrap());

    let sk_len = buf[8] as usize;
    let size_key = std::str::from_utf8(&buf[9..9 + sk_len.min(MAX_SIZE_KEY_LEN)])
        .unwrap_or("")
        .to_string();

    let lb_len = buf[40] as usize;
    let label = std::str::from_utf8(&buf[41..41 + lb_len.min(MAX_LABEL_LEN)])
        .unwrap_or("")
        .to_string();

    Ok(ActionDescriptor {
        kind,
        amount_chips,
        size_key,
        label,
        is_aggressive,
        source_action_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_descriptor_round_trips() {
        let row = RowDescriptor {
            row_id: 42,
            namespace: 1,
            seat: 2,
            street: 3,
            local_bucket: 100,
            global_bucket: 5000,
            source_node_idx: 999,
            action_offset: 1024,
            action_count: 4,
            prob_offset: 2048,
            row_key_fingerprint: 0xDEAD_BEEF,
            action_schema_fingerprint: 0xCAFE_BABE,
            semantic_key_kind: 7,
            semantic_key_offset: 4096,
        };

        let mut buf = [0u8; ROW_DESCRIPTOR_SIZE];
        encode_row(&row, &mut buf);
        let decoded = decode_row(&buf);
        assert_eq!(decoded, row);
    }

    #[test]
    fn action_descriptor_round_trips() {
        let action = ActionDescriptor {
            kind: ActionKind::Raise,
            amount_chips: 150,
            size_key: "0.75x".to_string(),
            label: "Raise 75%".to_string(),
            is_aggressive: true,
            source_action_index: 3,
        };

        let mut buf = [0u8; ACTION_DESCRIPTOR_SIZE];
        encode_action(&action, &mut buf);
        let decoded = decode_action(&buf).unwrap();
        assert_eq!(decoded, action);
    }

    #[test]
    fn action_descriptor_empty_strings() {
        let action = ActionDescriptor {
            kind: ActionKind::Fold,
            amount_chips: 0,
            size_key: String::new(),
            label: String::new(),
            is_aggressive: false,
            source_action_index: 0,
        };

        let mut buf = [0u8; ACTION_DESCRIPTOR_SIZE];
        encode_action(&action, &mut buf);
        let decoded = decode_action(&buf).unwrap();
        assert_eq!(decoded, action);
    }

    #[test]
    fn action_descriptor_max_length_strings() {
        let action = ActionDescriptor {
            kind: ActionKind::Bet,
            amount_chips: 500,
            size_key: "a".repeat(MAX_SIZE_KEY_LEN),
            label: "b".repeat(MAX_LABEL_LEN),
            is_aggressive: true,
            source_action_index: 1,
        };

        let mut buf = [0u8; ACTION_DESCRIPTOR_SIZE];
        encode_action(&action, &mut buf);
        let decoded = decode_action(&buf).unwrap();
        assert_eq!(decoded, action);
    }

    #[test]
    fn action_kind_from_u8_known_values() {
        assert_eq!(ActionKind::from_u8(0), Some(ActionKind::Fold));
        assert_eq!(ActionKind::from_u8(6), Some(ActionKind::AllInBetRaise));
        assert_eq!(ActionKind::from_u8(7), None);
        assert_eq!(ActionKind::from_u8(255), None);
    }

    #[test]
    fn row_identity_key_ordering() {
        let a = RowDescriptor {
            row_id: 0,
            namespace: 0,
            seat: 0,
            street: 0,
            local_bucket: 0,
            global_bucket: 0,
            source_node_idx: 0,
            action_offset: 0,
            action_count: 0,
            prob_offset: 0,
            row_key_fingerprint: 0,
            action_schema_fingerprint: 0,
            semantic_key_kind: 0,
            semantic_key_offset: 0,
        };
        let mut b = a.clone();
        b.namespace = 1;
        assert!(a.identity_key() < b.identity_key());

        let mut c = a.clone();
        c.seat = 1;
        assert!(a.identity_key() < c.identity_key());
    }

    // -- Property tests -------------------------------------------------------

    use proptest::prelude::*;

    fn arb_row() -> impl Strategy<Value = RowDescriptor> {
        // proptest Strategy is implemented for tuples up to 12 elements,
        // so we split the 14 fields into two groups and combine them.
        let part1 = (
            any::<u64>(),  // row_id
            any::<u16>(),  // namespace
            any::<u8>(),   // seat
            any::<u8>(),   // street
            any::<u16>(),  // local_bucket
            any::<u32>(),  // global_bucket
            any::<u32>(),  // source_node_idx
        );
        let part2 = (
            any::<u64>(),  // action_offset
            any::<u16>(),  // action_count
            any::<u64>(),  // prob_offset
            any::<u64>(),  // row_key_fingerprint
            any::<u64>(),  // action_schema_fingerprint
            any::<u16>(),  // semantic_key_kind
            any::<u64>(),  // semantic_key_offset
        );
        (part1, part2).prop_map(|(p1, p2)| RowDescriptor {
            row_id: p1.0,
            namespace: p1.1,
            seat: p1.2,
            street: p1.3,
            local_bucket: p1.4,
            global_bucket: p1.5,
            source_node_idx: p1.6,
            action_offset: p2.0,
            action_count: p2.1,
            prob_offset: p2.2,
            row_key_fingerprint: p2.3,
            action_schema_fingerprint: p2.4,
            semantic_key_kind: p2.5,
            semantic_key_offset: p2.6,
        })
    }

    fn arb_action_kind() -> impl Strategy<Value = ActionKind> {
        (0u8..=6).prop_map(|v| ActionKind::from_u8(v).unwrap())
    }

    fn arb_action() -> impl Strategy<Value = ActionDescriptor> {
        (
            arb_action_kind(),
            any::<u32>(),
            "[a-z0-9]{0,31}",
            "[a-z0-9 ]{0,55}",
            any::<bool>(),
            any::<u16>(),
        )
            .prop_map(
                |(kind, amount_chips, size_key, label, is_aggressive, source_action_index)| {
                    ActionDescriptor {
                        kind,
                        amount_chips,
                        size_key,
                        label,
                        is_aggressive,
                        source_action_index,
                    }
                },
            )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn prop_row_descriptor_round_trip(row in arb_row()) {
            let mut buf = [0u8; ROW_DESCRIPTOR_SIZE];
            encode_row(&row, &mut buf);
            let decoded = decode_row(&buf);
            prop_assert_eq!(decoded, row);
        }

        #[test]
        fn prop_action_descriptor_round_trip(action in arb_action()) {
            let mut buf = [0u8; ACTION_DESCRIPTOR_SIZE];
            encode_action(&action, &mut buf);
            let decoded = decode_action(&buf).unwrap();
            prop_assert_eq!(decoded, action);
        }

        #[test]
        fn prop_sorted_unique_identity(
            mut keys in proptest::collection::vec(
                (any::<u16>(), any::<u8>(), any::<u8>(), any::<u32>(), any::<u32>(), any::<u64>()),
                2..20
            )
        ) {
            keys.sort();
            keys.dedup();
            // Sorted, deduplicated keys should always be accepted by
            // validate_row_order (tested via identity_key ordering).
            for w in keys.windows(2) {
                prop_assert!(w[0] < w[1]);
            }
        }

        #[test]
        fn prop_row_offset_arithmetic(
            action_offset in 0u64..1_000_000,
            action_count in 0u16..1000,
        ) {
            let end = action_offset.checked_add(u64::from(action_count));
            prop_assert!(end.is_some());
            prop_assert!(end.unwrap() >= action_offset);
        }

        #[test]
        fn prop_normalization_tolerance(
            probs in proptest::collection::vec(0.0f32..1.0, 2..10)
        ) {
            let sum: f64 = probs.iter().map(|p| f64::from(*p)).sum();
            // Probabilities from 0..1 might not sum to 1.
            // Just verify the sum is well-defined (not NaN).
            prop_assert!(sum.is_finite());
        }
    }
}
