use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

use super::MAX_PLAYERS;

// === Seat ===

/// Player seat index (0-based). Validated against table size on construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Seat(u8);

impl Seat {
    /// Create a new seat, panicking if `index >= num_players` or exceeds `MAX_PLAYERS`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= num_players` or `num_players > MAX_PLAYERS`.
    #[must_use]
    pub fn new(index: u8, num_players: u8) -> Self {
        assert!(
            (index as usize) < num_players as usize && (num_players as usize) <= MAX_PLAYERS,
            "Seat {index} out of range for {num_players} players (max {MAX_PLAYERS})"
        );
        Self(index)
    }

    /// Create a seat without validation. Use in hot paths where the index
    /// is already known to be valid.
    #[must_use]
    pub const fn from_raw(index: u8) -> Self {
        Self(index)
    }

    /// Return the underlying seat index.
    #[must_use]
    pub const fn index(self) -> u8 {
        self.0
    }
}

// === PlayerSet ===

/// Bitfield representing a set of players (up to 8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlayerSet(u8);

impl PlayerSet {
    /// An empty set with no players.
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// A set containing seats `0..num_players`.
    #[must_use]
    pub fn all(num_players: u8) -> Self {
        debug_assert!(num_players as usize <= MAX_PLAYERS);
        if num_players >= 8 { Self(0xFF) } else { Self((1u8 << num_players) - 1) }
    }

    /// Whether the set contains the given seat.
    #[must_use]
    pub const fn contains(self, seat: Seat) -> bool {
        self.0 & (1 << seat.0) != 0
    }

    /// Add a seat to the set.
    pub fn insert(&mut self, seat: Seat) {
        self.0 |= 1 << seat.0;
    }

    /// Remove a seat from the set.
    pub fn remove(&mut self, seat: Seat) {
        self.0 &= !(1 << seat.0);
    }

    /// Number of players in the set.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn count(self) -> u8 {
        // count_ones() on a u8 returns at most 8, so truncation is impossible.
        self.0.count_ones() as u8
    }

    /// Iterate over seats in ascending order.
    pub fn iter(self) -> impl Iterator<Item = Seat> {
        (0u8..8).filter(move |&i| self.0 & (1 << i) != 0).map(Seat)
    }

    /// Find the next seat after `seat` (clockwise wrap). Returns `None` if
    /// fewer than 2 players remain in the set.
    #[must_use]
    pub fn next_after(self, seat: Seat, num_players: u8) -> Option<Seat> {
        if self.count() < 2 {
            return None;
        }
        for offset in 1..num_players {
            let candidate = (seat.0 + offset) % num_players;
            if self.contains(Seat(candidate)) {
                return Some(Seat(candidate));
            }
        }
        None
    }

    /// Raw bit representation.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Construct from raw bits.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }
}

// === Chips ===

/// Chip amount newtype wrapping `f64`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Chips(pub f64);

impl Chips {
    /// Zero chips.
    pub const ZERO: Self = Self(0.0);

    /// Whether the chip amount is exactly zero.
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }
}

impl Add for Chips {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Chips {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl AddAssign for Chips {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for Chips {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Mul<f64> for Chips {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self(self.0 * rhs)
    }
}

impl Mul<u8> for Chips {
    type Output = Self;
    fn mul(self, rhs: u8) -> Self {
        Self(self.0 * f64::from(rhs))
    }
}

// === Bucket ===

/// Bucket index for hand abstraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Bucket(pub u16);

// === Street ===

/// Poker street.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Street {
    Preflop = 0,
    Flop = 1,
    Turn = 2,
    River = 3,
}

impl Street {
    /// Parse from a raw byte.
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Preflop),
            1 => Some(Self::Flop),
            2 => Some(Self::Turn),
            3 => Some(Self::River),
            _ => None,
        }
    }

    /// Returns the next street, or `None` if already on the river.
    #[must_use]
    pub const fn next(self) -> Option<Self> {
        match self {
            Self::Preflop => Some(Self::Flop),
            Self::Flop => Some(Self::Turn),
            Self::Turn => Some(Self::River),
            Self::River => None,
        }
    }

    /// Street as a `usize` index (0-3).
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    // ── Seat tests ──

    #[timed_test]
    fn seat_new_valid_indices() {
        let s = Seat::new(0, 6);
        assert_eq!(s.index(), 0);
        let s = Seat::new(5, 6);
        assert_eq!(s.index(), 5);
    }

    #[timed_test]
    #[should_panic(expected = "out of range")]
    fn seat_new_panics_when_index_equals_num_players() {
        let _ = Seat::new(6, 6);
    }

    #[timed_test]
    #[should_panic(expected = "out of range")]
    fn seat_new_panics_when_index_exceeds_max() {
        let _ = Seat::new(8, 9);
    }

    #[timed_test]
    fn seat_from_raw_no_validation() {
        let s = Seat::from_raw(255);
        assert_eq!(s.index(), 255);
    }

    // ── PlayerSet tests ──

    #[timed_test]
    fn player_set_empty_has_zero_count() {
        let ps = PlayerSet::empty();
        assert_eq!(ps.count(), 0);
        assert_eq!(ps.bits(), 0);
    }

    #[timed_test]
    fn player_set_all_sets_n_bits() {
        let ps = PlayerSet::all(6);
        assert_eq!(ps.count(), 6);
        assert_eq!(ps.bits(), 0b0011_1111);
    }

    #[timed_test]
    fn player_set_insert_and_contains() {
        let mut ps = PlayerSet::empty();
        let s2 = Seat::from_raw(2);
        assert!(!ps.contains(s2));
        ps.insert(s2);
        assert!(ps.contains(s2));
        assert_eq!(ps.count(), 1);
    }

    #[timed_test]
    fn player_set_remove() {
        let mut ps = PlayerSet::all(4);
        let s1 = Seat::from_raw(1);
        assert!(ps.contains(s1));
        ps.remove(s1);
        assert!(!ps.contains(s1));
        assert_eq!(ps.count(), 3);
    }

    #[timed_test]
    fn player_set_insert_idempotent() {
        let mut ps = PlayerSet::empty();
        let s0 = Seat::from_raw(0);
        ps.insert(s0);
        ps.insert(s0);
        assert_eq!(ps.count(), 1);
    }

    #[timed_test]
    fn player_set_remove_absent_is_noop() {
        let mut ps = PlayerSet::empty();
        let s3 = Seat::from_raw(3);
        ps.remove(s3);
        assert_eq!(ps.count(), 0);
    }

    #[timed_test]
    fn player_set_iter_returns_seats_in_order() {
        let mut ps = PlayerSet::empty();
        ps.insert(Seat::from_raw(1));
        ps.insert(Seat::from_raw(3));
        ps.insert(Seat::from_raw(5));
        let seats: Vec<u8> = ps.iter().map(|s| s.index()).collect();
        assert_eq!(seats, vec![1, 3, 5]);
    }

    #[timed_test]
    fn player_set_iter_empty_yields_nothing() {
        let ps = PlayerSet::empty();
        assert_eq!(ps.iter().count(), 0);
    }

    #[timed_test]
    fn player_set_next_after_wraps_clockwise() {
        // Players at seats 0, 2, 4 in a 6-player game
        let mut ps = PlayerSet::empty();
        ps.insert(Seat::from_raw(0));
        ps.insert(Seat::from_raw(2));
        ps.insert(Seat::from_raw(4));

        // Next after seat 0 should be seat 2
        assert_eq!(
            ps.next_after(Seat::from_raw(0), 6),
            Some(Seat::from_raw(2))
        );
        // Next after seat 2 should be seat 4
        assert_eq!(
            ps.next_after(Seat::from_raw(2), 6),
            Some(Seat::from_raw(4))
        );
        // Next after seat 4 should wrap to seat 0
        assert_eq!(
            ps.next_after(Seat::from_raw(4), 6),
            Some(Seat::from_raw(0))
        );
    }

    #[timed_test]
    fn player_set_next_after_single_player_returns_none() {
        let mut ps = PlayerSet::empty();
        ps.insert(Seat::from_raw(3));
        assert_eq!(ps.next_after(Seat::from_raw(3), 6), None);
    }

    #[timed_test]
    fn player_set_next_after_empty_returns_none() {
        let ps = PlayerSet::empty();
        assert_eq!(ps.next_after(Seat::from_raw(0), 6), None);
    }

    #[timed_test]
    fn player_set_from_bits_round_trips() {
        let ps = PlayerSet::from_bits(0b1010_0101);
        assert_eq!(ps.bits(), 0b1010_0101);
    }

    #[timed_test]
    fn player_set_all_max_players() {
        let ps = PlayerSet::all(8);
        assert_eq!(ps.count(), 8);
        assert_eq!(ps.bits(), 0xFF);
    }

    // ── Chips tests ──

    #[timed_test]
    fn chips_zero_constant() {
        assert_eq!(Chips::ZERO, Chips(0.0));
        assert!(Chips::ZERO.is_zero());
    }

    #[timed_test]
    fn chips_is_zero_false_for_nonzero() {
        assert!(!Chips(1.5).is_zero());
    }

    #[timed_test]
    fn chips_add() {
        assert_eq!(Chips(10.0) + Chips(5.0), Chips(15.0));
    }

    #[timed_test]
    fn chips_sub() {
        assert_eq!(Chips(10.0) - Chips(3.0), Chips(7.0));
    }

    #[timed_test]
    fn chips_add_assign() {
        let mut c = Chips(10.0);
        c += Chips(5.0);
        assert_eq!(c, Chips(15.0));
    }

    #[timed_test]
    fn chips_sub_assign() {
        let mut c = Chips(10.0);
        c -= Chips(3.0);
        assert_eq!(c, Chips(7.0));
    }

    #[timed_test]
    fn chips_mul_f64() {
        assert_eq!(Chips(10.0) * 2.5, Chips(25.0));
    }

    #[timed_test]
    fn chips_mul_u8() {
        assert_eq!(Chips(10.0) * 3u8, Chips(30.0));
    }

    // ── Street tests ──

    #[timed_test]
    fn street_from_u8_round_trips() {
        for v in 0..=3u8 {
            let street = Street::from_u8(v).unwrap();
            assert_eq!(street.index(), v as usize);
        }
    }

    #[timed_test]
    fn street_from_u8_invalid_returns_none() {
        assert!(Street::from_u8(4).is_none());
        assert!(Street::from_u8(255).is_none());
    }

    #[timed_test]
    fn street_next_progression() {
        assert_eq!(Street::Preflop.next(), Some(Street::Flop));
        assert_eq!(Street::Flop.next(), Some(Street::Turn));
        assert_eq!(Street::Turn.next(), Some(Street::River));
        assert_eq!(Street::River.next(), None);
    }

    #[timed_test]
    fn street_index_matches_discriminant() {
        assert_eq!(Street::Preflop.index(), 0);
        assert_eq!(Street::Flop.index(), 1);
        assert_eq!(Street::Turn.index(), 2);
        assert_eq!(Street::River.index(), 3);
    }

    // ── Bucket tests ──

    #[timed_test]
    fn bucket_default_is_zero() {
        assert_eq!(Bucket::default(), Bucket(0));
    }
}
