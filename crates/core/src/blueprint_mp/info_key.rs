//! 128-bit information set key for the multiplayer blueprint solver.
//!
//! ## Bit Layout (128 bits total)
//!
//! ```text
//! hi word (64 bits):
//!   Bits 63-61: seat position       (3 bits, 0-7)
//!   Bits 60-33: hand/bucket         (28 bits)
//!   Bits 32-26: reserved            (7 bits)
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

use super::types::Seat;

// Bit positions within the `hi` word.
const SEAT_SHIFT: u32 = 61;
const BUCKET_SHIFT: u32 = 33;

// Masks for extraction.
const SEAT_MASK: u64 = 0x7; // 3 bits
const BUCKET_MASK: u64 = 0xFFF_FFFF; // 28 bits

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
    pub fn new(seat: Seat, bucket: u32, actions: &[u8]) -> Self {
        assert!(
            actions.len() <= MAX_ACTIONS,
            "action count {} exceeds 22",
            actions.len()
        );
        let hi = Self::pack_header(seat, bucket);
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
        {
            ((self.hi >> BUCKET_SHIFT) & BUCKET_MASK) as u32
        }
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

    /// Pack seat and bucket into the upper bits of `hi`.
    fn pack_header(seat: Seat, bucket: u32) -> u64 {
        let mut hi: u64 = 0;
        hi |= (u64::from(seat.index()) & SEAT_MASK) << SEAT_SHIFT;
        hi |= (u64::from(bucket) & BUCKET_MASK) << BUCKET_SHIFT;
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
            "InfoKey128 {{ seat={}, bucket={} }}",
            self.seat().index(),
            self.bucket_bits(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use test_macros::timed_test;

    #[timed_test]
    fn round_trip_components() {
        let key = InfoKey128::new(Seat::from_raw(3), 0x0ABC_DEF, &[1, 2, 3]);
        assert_eq!(key.seat(), Seat::from_raw(3));
        assert_eq!(key.bucket_bits(), 0x0ABC_DEF);
    }

    #[timed_test]
    fn different_seats_different_keys() {
        let k1 = InfoKey128::new(Seat::from_raw(0), 100, &[1]);
        let k2 = InfoKey128::new(Seat::from_raw(1), 100, &[1]);
        assert_ne!(k1, k2);
    }

    #[timed_test]
    fn different_buckets_different_keys() {
        let k1 = InfoKey128::new(Seat::from_raw(2), 100, &[1]);
        let k2 = InfoKey128::new(Seat::from_raw(2), 200, &[1]);
        assert_ne!(k1, k2);
    }

    #[timed_test]
    fn bucket_namespace_makes_streets_different() {
        let flop_bucket = (1 << 14) | 42;
        let turn_bucket = (2 << 14) | 42;
        let k1 = InfoKey128::new(Seat::from_raw(0), flop_bucket, &[]);
        let k2 = InfoKey128::new(Seat::from_raw(0), turn_bucket, &[]);
        assert_ne!(k1, k2);
    }

    #[timed_test]
    fn max_22_actions() {
        let actions: Vec<u8> = (0..22).map(|i| (i % 15) as u8).collect();
        let key = InfoKey128::new(Seat::from_raw(7), 999, &actions);
        assert_eq!(key.seat(), Seat::from_raw(7));
    }

    #[timed_test]
    #[should_panic(expected = "exceeds 22")]
    fn overflow_panics() {
        let actions = vec![1u8; 23];
        let _ = InfoKey128::new(Seat::from_raw(0), 0, &actions);
    }

    #[timed_test]
    fn action_order_matters() {
        let k1 = InfoKey128::new(Seat::from_raw(0), 0, &[1, 2]);
        let k2 = InfoKey128::new(Seat::from_raw(0), 0, &[2, 1]);
        assert_ne!(k1, k2);
    }

    #[timed_test]
    fn empty_actions_valid() {
        let key = InfoKey128::new(Seat::from_raw(4), 555, &[]);
        assert_eq!(key.seat(), Seat::from_raw(4));
        assert_eq!(key.bucket_bits(), 555);
    }

    #[timed_test]
    fn hash_equality() {
        let k1 = InfoKey128::new(Seat::from_raw(2), 42, &[3, 5]);
        let k2 = InfoKey128::new(Seat::from_raw(2), 42, &[3, 5]);
        assert_eq!(k1, k2);

        let mut set = HashSet::new();
        set.insert(k1);
        assert!(set.contains(&k2));
    }

    #[timed_test]
    fn from_parts_as_parts_round_trip() {
        let key = InfoKey128::new(Seat::from_raw(5), 12345, &[1, 2, 3]);
        let (hi, lo) = key.as_parts();
        let reconstructed = InfoKey128::from_parts(hi, lo);
        assert_eq!(key, reconstructed);
    }

    #[timed_test]
    fn debug_shows_human_readable() {
        let key = InfoKey128::new(Seat::from_raw(2), 42, &[]);
        let dbg = format!("{key:?}");
        assert!(dbg.contains("seat=2"), "Debug missing seat: {dbg}");
        assert!(dbg.contains("bucket=42"), "Debug missing bucket: {dbg}");
    }

    #[timed_test]
    fn seat_max_value() {
        let key = InfoKey128::new(Seat::from_raw(7), 0, &[]);
        assert_eq!(key.seat(), Seat::from_raw(7));
    }

    #[timed_test]
    fn bucket_max_28_bits() {
        let max_bucket = (1u32 << 28) - 1;
        let key = InfoKey128::new(Seat::from_raw(0), max_bucket, &[]);
        assert_eq!(key.bucket_bits(), max_bucket);
    }

    #[timed_test]
    fn single_action_round_trips_all_values() {
        for action in 0..=0xFu8 {
            let k1 = InfoKey128::new(Seat::from_raw(0), 0, &[action]);
            let k2 = InfoKey128::new(Seat::from_raw(0), 0, &[action]);
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
        let k1 = InfoKey128::new(Seat::from_raw(0), 0, &a1);
        let k2 = InfoKey128::new(Seat::from_raw(0), 0, &a2);
        assert_ne!(k1, k2);
    }

    #[timed_test]
    fn all_fields_max_no_corruption() {
        let actions = vec![0xFu8; 22];
        let key = InfoKey128::new(Seat::from_raw(7), (1u32 << 28) - 1, &actions);
        assert_eq!(key.seat(), Seat::from_raw(7));
        assert_eq!(key.bucket_bits(), (1u32 << 28) - 1);
    }
}
