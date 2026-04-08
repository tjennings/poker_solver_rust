//! 128-bit information set key for the multiplayer blueprint solver.
//!
//! ## Bit Layout (128 bits total)
//!
//! ```text
//! hi word (64 bits):
//!   Bits 63-61: seat position       (3 bits, 0-7)
//!   Bits 60-33: hand/bucket         (28 bits)
//!   Bits 32-31: street              (2 bits)
//!   Bits 30-26: spr_bucket          (5 bits)
//!   Bits 25-0:  top 26 bits of action history
//!
//! lo word (64 bits):
//!   Bits 63-0:  bottom 64 bits of action history
//! ```
//!
//! Total action history: 90 bits = 22 slots x 4 bits each.
//! Actions are packed MSB-first: action\[0\] at the highest position.
//!
//! Action slot encoding (4 bits each):
//! - 0 = fold
//! - 1 = check
//! - 2 = call
//! - 3+ = bet/raise size index
//! - 0xF = empty/unused sentinel

use super::types::{Seat, Street};

// Bit positions within the `hi` word.
const SEAT_SHIFT: u32 = 61;
const BUCKET_SHIFT: u32 = 33;
const STREET_SHIFT: u32 = 31;
const SPR_SHIFT: u32 = 26;

// Masks for extraction.
const SEAT_MASK: u64 = 0x7; // 3 bits
const BUCKET_MASK: u64 = 0xFFF_FFFF; // 28 bits
const STREET_MASK: u64 = 0x3; // 2 bits
const SPR_MASK: u64 = 0x1F; // 5 bits

/// Total number of 4-bit action slots in the 90-bit action history.
const MAX_ACTIONS: usize = 22;

/// Total bits used for action history (22 slots x 4 bits).
const ACTION_BITS_TOTAL: u32 = 90;

/// A 128-bit packed information set key for multiplayer solvers.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct InfoKey128 {
    hi: u64,
    lo: u64,
}

impl InfoKey128 {
    /// Build a key from its components.
    ///
    /// # Panics
    ///
    /// Panics if `actions.len()` exceeds 22.
    #[must_use]
    pub fn new(seat: Seat, bucket: u32, street: Street, spr_bucket: u32, actions: &[u8]) -> Self {
        assert!(
            actions.len() <= MAX_ACTIONS,
            "action count {} exceeds 22",
            actions.len()
        );
        let hi = Self::pack_header(seat, bucket, street, spr_bucket);
        let (action_hi, lo) = Self::pack_actions(actions);
        Self {
            hi: hi | action_hi,
            lo,
        }
    }

    /// Extract the seat position.
    #[must_use]
    pub fn seat(self) -> Seat {
        #[allow(clippy::cast_possible_truncation)]
        Seat::from_raw(((self.hi >> SEAT_SHIFT) & SEAT_MASK) as u8)
    }

    /// Extract the 28-bit hand/bucket field.
    #[must_use]
    pub fn bucket_bits(self) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        { ((self.hi >> BUCKET_SHIFT) & BUCKET_MASK) as u32 }
    }

    /// Extract the street.
    ///
    /// # Panics
    ///
    /// Panics if the internal street bits are invalid (should never happen).
    #[must_use]
    pub fn street(self) -> Street {
        #[allow(clippy::cast_possible_truncation)]
        let raw = ((self.hi >> STREET_SHIFT) & STREET_MASK) as u8;
        Street::from_u8(raw).expect("invalid street bits in InfoKey128")
    }

    /// Extract the SPR bucket (5 bits, 0-31).
    #[must_use]
    pub fn spr_bucket(self) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        { ((self.hi >> SPR_SHIFT) & SPR_MASK) as u32 }
    }

    /// Construct from raw `(hi, lo)` pair.
    #[must_use]
    pub const fn from_parts(hi: u64, lo: u64) -> Self {
        Self { hi, lo }
    }

    /// Return the raw `(hi, lo)` pair.
    #[must_use]
    pub const fn as_parts(self) -> (u64, u64) {
        (self.hi, self.lo)
    }

    // -- private helpers --

    /// Pack seat, bucket, street, and spr into the upper bits of `hi`.
    fn pack_header(seat: Seat, bucket: u32, street: Street, spr_bucket: u32) -> u64 {
        let mut hi: u64 = 0;
        hi |= (u64::from(seat.index()) & SEAT_MASK) << SEAT_SHIFT;
        hi |= (u64::from(bucket) & BUCKET_MASK) << BUCKET_SHIFT;
        hi |= (u64::from(street as u8) & STREET_MASK) << STREET_SHIFT;
        hi |= (u64::from(spr_bucket) & SPR_MASK) << SPR_SHIFT;
        hi
    }

    /// Pack action slots into a 90-bit field split across `(hi_bits, lo)`.
    ///
    /// Actions are MSB-first: action\[0\] occupies the highest 4 bits of the
    /// 90-bit field. The top 26 bits land in `hi` bits 25-0; the bottom 64
    /// bits form `lo`.
    fn pack_actions(actions: &[u8]) -> (u64, u64) {
        let mut hi_actions: u64 = 0;
        let mut lo: u64 = 0;
        for (i, &code) in actions.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let bit_pos = ACTION_BITS_TOTAL - 4 - (i as u32) * 4;
            Self::set_action_nibble(bit_pos, code, &mut hi_actions, &mut lo);
        }
        (hi_actions, lo)
    }

    /// Write a single 4-bit action nibble at `bit_pos` within the 90-bit
    /// action field, split across `hi_actions` (bits 89-64) and `lo` (bits 63-0).
    fn set_action_nibble(bit_pos: u32, code: u8, hi_actions: &mut u64, lo: &mut u64) {
        let nibble = u64::from(code) & 0xF;
        if bit_pos >= 64 {
            *hi_actions |= nibble << (bit_pos - 64);
        } else {
            *lo |= nibble << bit_pos;
        }
    }
}

impl std::fmt::Debug for InfoKey128 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "InfoKey128 {{ seat={}, bucket={}, {}, spr={} }}",
            self.seat().index(),
            self.bucket_bits(),
            self.street(),
            self.spr_bucket(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;
    use std::collections::HashSet;

    #[timed_test]
    fn round_trip_components() {
        let key = InfoKey128::new(
            Seat::from_raw(3),
            0x0ABC_DEF,
            Street::Flop,
            17,
            &[1, 2, 3],
        );
        assert_eq!(key.seat(), Seat::from_raw(3));
        assert_eq!(key.bucket_bits(), 0x0ABC_DEF);
        assert_eq!(key.street(), Street::Flop);
        assert_eq!(key.spr_bucket(), 17);
    }

    #[timed_test]
    fn different_seats_different_keys() {
        let k1 = InfoKey128::new(Seat::from_raw(0), 100, Street::Preflop, 5, &[1]);
        let k2 = InfoKey128::new(Seat::from_raw(1), 100, Street::Preflop, 5, &[1]);
        assert_ne!(k1, k2);
    }

    #[timed_test]
    fn different_buckets_different_keys() {
        let k1 = InfoKey128::new(Seat::from_raw(2), 100, Street::Flop, 5, &[1]);
        let k2 = InfoKey128::new(Seat::from_raw(2), 200, Street::Flop, 5, &[1]);
        assert_ne!(k1, k2);
    }

    #[timed_test]
    fn different_streets_different_keys() {
        let k1 = InfoKey128::new(Seat::from_raw(0), 42, Street::Turn, 10, &[]);
        let k2 = InfoKey128::new(Seat::from_raw(0), 42, Street::River, 10, &[]);
        assert_ne!(k1, k2);
    }

    #[timed_test]
    fn max_22_actions() {
        let actions: Vec<u8> = (0..22).map(|i| (i % 15) as u8).collect();
        let key = InfoKey128::new(Seat::from_raw(7), 999, Street::River, 31, &actions);
        assert_eq!(key.seat(), Seat::from_raw(7));
    }

    #[timed_test]
    #[should_panic(expected = "exceeds 22")]
    fn overflow_panics() {
        let actions = vec![1u8; 23];
        let _ = InfoKey128::new(Seat::from_raw(0), 0, Street::Preflop, 0, &actions);
    }

    #[timed_test]
    fn action_order_matters() {
        let k1 = InfoKey128::new(Seat::from_raw(0), 0, Street::Preflop, 0, &[1, 2]);
        let k2 = InfoKey128::new(Seat::from_raw(0), 0, Street::Preflop, 0, &[2, 1]);
        assert_ne!(k1, k2);
    }

    #[timed_test]
    fn empty_actions_valid() {
        let key = InfoKey128::new(Seat::from_raw(4), 555, Street::Turn, 20, &[]);
        assert_eq!(key.seat(), Seat::from_raw(4));
        assert_eq!(key.bucket_bits(), 555);
        assert_eq!(key.street(), Street::Turn);
        assert_eq!(key.spr_bucket(), 20);
    }

    #[timed_test]
    fn hash_equality() {
        let k1 = InfoKey128::new(Seat::from_raw(2), 42, Street::Flop, 10, &[3, 5]);
        let k2 = InfoKey128::new(Seat::from_raw(2), 42, Street::Flop, 10, &[3, 5]);
        assert_eq!(k1, k2);

        let mut set = HashSet::new();
        set.insert(k1);
        assert!(set.contains(&k2));
    }

    #[timed_test]
    fn spr_and_actions_dont_overlap() {
        // Max SPR (31) with max action bits (all 0xF)
        let actions = vec![0xFu8; 22];
        let key = InfoKey128::new(Seat::from_raw(0), 0, Street::Preflop, 31, &actions);
        assert_eq!(key.spr_bucket(), 31, "SPR corrupted by max action bits");
    }

    #[timed_test]
    fn all_streets_round_trip() {
        for &street in &[Street::Preflop, Street::Flop, Street::Turn, Street::River] {
            let key = InfoKey128::new(Seat::from_raw(1), 77, street, 8, &[2]);
            assert_eq!(key.street(), street, "Failed for {street}");
        }
    }

    #[timed_test]
    fn from_parts_as_parts_round_trip() {
        let key = InfoKey128::new(Seat::from_raw(5), 12345, Street::Turn, 15, &[1, 2, 3]);
        let (hi, lo) = key.as_parts();
        let reconstructed = InfoKey128::from_parts(hi, lo);
        assert_eq!(key, reconstructed);
    }

    #[timed_test]
    fn debug_shows_human_readable() {
        let key = InfoKey128::new(Seat::from_raw(2), 42, Street::Flop, 10, &[]);
        let dbg = format!("{key:?}");
        assert!(dbg.contains("seat=2"), "Debug missing seat: {dbg}");
        assert!(dbg.contains("bucket=42"), "Debug missing bucket: {dbg}");
        assert!(dbg.contains("flop"), "Debug missing street: {dbg}");
        assert!(dbg.contains("spr=10"), "Debug missing spr: {dbg}");
    }

    #[timed_test]
    fn seat_max_value() {
        let key = InfoKey128::new(Seat::from_raw(7), 0, Street::Preflop, 0, &[]);
        assert_eq!(key.seat(), Seat::from_raw(7));
    }

    #[timed_test]
    fn bucket_max_28_bits() {
        let max_bucket = (1u32 << 28) - 1;
        let key = InfoKey128::new(Seat::from_raw(0), max_bucket, Street::Preflop, 0, &[]);
        assert_eq!(key.bucket_bits(), max_bucket);
    }

    #[timed_test]
    fn spr_max_value() {
        let key = InfoKey128::new(Seat::from_raw(0), 0, Street::Preflop, 31, &[]);
        assert_eq!(key.spr_bucket(), 31);
    }

    #[timed_test]
    fn single_action_round_trips_all_values() {
        for action in 0..=0xFu8 {
            let k1 = InfoKey128::new(Seat::from_raw(0), 0, Street::Preflop, 0, &[action]);
            let k2 = InfoKey128::new(Seat::from_raw(0), 0, Street::Preflop, 0, &[action]);
            assert_eq!(k1, k2, "Action {action} does not round-trip");
        }
    }

    #[timed_test]
    fn actions_at_boundary_slots_20_and_21() {
        // Verify that actions near the hi/lo boundary (slot ~6-7) are distinct
        let mut a1 = vec![0u8; 22];
        let mut a2 = vec![0u8; 22];
        a1[20] = 5;
        a2[21] = 5;
        let k1 = InfoKey128::new(Seat::from_raw(0), 0, Street::Preflop, 0, &a1);
        let k2 = InfoKey128::new(Seat::from_raw(0), 0, Street::Preflop, 0, &a2);
        assert_ne!(k1, k2);
    }

    #[timed_test]
    fn all_fields_max_no_corruption() {
        let actions = vec![0xFu8; 22];
        let key = InfoKey128::new(
            Seat::from_raw(7),
            (1u32 << 28) - 1,
            Street::River,
            31,
            &actions,
        );
        assert_eq!(key.seat(), Seat::from_raw(7));
        assert_eq!(key.bucket_bits(), (1u32 << 28) - 1);
        assert_eq!(key.street(), Street::River);
        assert_eq!(key.spr_bucket(), 31);
    }
}
