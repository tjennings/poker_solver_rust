//! Pre-computed equity + expected-next-equity cache for MCCFR bucketing.
//!
//! Eliminates the ~30× overhead of computing expected delta on the fly by
//! pre-computing `(equity, expected_next_equity)` for every canonical
//! `(hand, board)` combination at the flop and turn streets.
//!
//! **Bottom-up generation:**
//! 1. Turn table first: for each canonical turn board × valid hand, compute
//!    `equity` on the 4-card board and `expected_next_equity` by averaging
//!    over all ~46 possible river cards.
//! 2. Flop table second: for each canonical flop board × valid hand, compute
//!    `equity` on the 3-card board and `expected_next_equity` by averaging
//!    the turn-table equity entries over all ~47 possible turn cards. This
//!    gives two-street lookahead (the flop expected-next already accounts
//!    for the river).

// Practical game trees have < 2^32 entries; card indices fit in u8.
#![allow(clippy::cast_possible_truncation)]

use std::io;
use std::path::Path;

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::poker::{Card, Suit, Value};
use crate::showdown_equity::compute_equity;
use super::cluster_pipeline::{build_deck, enumerate_canonical_flops, enumerate_canonical_turns};

/// Cached equity and expected-next-equity for a single (hand, board) pair.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CacheEntry {
    pub equity: f32,
    pub expected_next_equity: f32,
}

/// Pre-computed equity+delta lookup tables for flop and turn streets.
#[derive(Debug, Serialize, Deserialize)]
pub struct EquityDeltaCache {
    /// Turn entries: canonical (hand, board) → (equity, expected_river_equity).
    turn: FxHashMap<u64, CacheEntry>,
    /// Flop entries: canonical (hand, board) → (equity, expected_turn_equity).
    flop: FxHashMap<u64, CacheEntry>,
}

// ---------------------------------------------------------------------------
// Card → 0..51 index packing
// ---------------------------------------------------------------------------

/// Map a card to a unique index in 0..51.
///
/// Layout: value_index * 4 + suit_index, where values go Two=0..Ace=12
/// and suits go Spade=0, Heart=1, Diamond=2, Club=3.
fn card_index(card: Card) -> u8 {
    let v = u8::from(card.value); // Two=0 .. Ace=12
    let s = match card.suit {
        Suit::Spade => 0u8,
        Suit::Heart => 1,
        Suit::Diamond => 2,
        Suit::Club => 3,
    };
    v * 4 + s
}

/// Pack a hand (2 cards) + board (3 or 4 cards) into a u64 key.
///
/// Cards are sorted by index before packing to ensure canonical ordering.
/// Each card occupies 6 bits (max value 51 < 64). Up to 6 cards = 36 bits.
fn pack_key(hand: [Card; 2], board: &[Card]) -> u64 {
    let mut indices: arrayvec::ArrayVec<u8, 6> = arrayvec::ArrayVec::new();
    indices.push(card_index(hand[0]));
    indices.push(card_index(hand[1]));
    for &c in board {
        indices.push(card_index(c));
    }
    indices.sort_unstable();

    let mut key = 0u64;
    for (i, &idx) in indices.iter().enumerate() {
        key |= u64::from(idx) << (i * 6);
    }
    key
}

/// Check if a hand conflicts with (shares cards with) the board.
#[cfg(test)]
fn conflicts(hand: [Card; 2], board: &[Card]) -> bool {
    for &bc in board {
        if hand[0].value == bc.value && hand[0].suit == bc.suit {
            return true;
        }
        if hand[1].value == bc.value && hand[1].suit == bc.suit {
            return true;
        }
    }
    false
}

/// Enumerate all C(N, 2) two-card hands from remaining cards not on the board.
fn valid_hands(board: &[Card]) -> Vec<[Card; 2]> {
    let deck = build_deck();
    let mut hands = Vec::new();
    for i in 0..deck.len() {
        if board.iter().any(|b| b.value == deck[i].value && b.suit == deck[i].suit) {
            continue;
        }
        for j in (i + 1)..deck.len() {
            if board.iter().any(|b| b.value == deck[j].value && b.suit == deck[j].suit) {
                continue;
            }
            hands.push([deck[i], deck[j]]);
        }
    }
    hands
}

/// Cards remaining in the deck after removing hand and board.
fn remaining_cards(hand: [Card; 2], board: &[Card]) -> Vec<Card> {
    let deck = build_deck();
    deck.into_iter()
        .filter(|c| {
            !(c.value == hand[0].value && c.suit == hand[0].suit)
                && !(c.value == hand[1].value && c.suit == hand[1].suit)
                && !board.iter().any(|b| b.value == c.value && b.suit == c.suit)
        })
        .collect()
}

impl EquityDeltaCache {
    /// Look up cached values for a turn board (4 cards) + hand.
    #[must_use]
    pub fn turn_lookup(&self, hand: [Card; 2], board: &[Card]) -> Option<&CacheEntry> {
        debug_assert_eq!(board.len(), 4);
        self.turn.get(&pack_key(hand, board))
    }

    /// Look up cached values for a flop board (3 cards) + hand.
    #[must_use]
    pub fn flop_lookup(&self, hand: [Card; 2], board: &[Card]) -> Option<&CacheEntry> {
        debug_assert_eq!(board.len(), 3);
        self.flop.get(&pack_key(hand, board))
    }

    /// Number of entries in the turn table.
    #[must_use]
    pub fn turn_entries(&self) -> usize {
        self.turn.len()
    }

    /// Number of entries in the flop table.
    #[must_use]
    pub fn flop_entries(&self) -> usize {
        self.flop.len()
    }

    /// Save cache to a binary file.
    ///
    /// # Errors
    /// Returns `Err` if the file cannot be created or written.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, self)
            .map_err(|e| io::Error::other(e.to_string()))
    }

    /// Load cache from a binary file.
    ///
    /// # Errors
    /// Returns `Err` if the file cannot be read or is invalid.
    pub fn load(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        bincode::deserialize_from(reader)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Generate the full cache bottom-up: turn table first, then flop table.
    ///
    /// `progress` is called with `(street_name, fraction_complete)` where
    /// fraction is in [0.0, 1.0].
    pub fn generate(progress: impl Fn(&str, f64) + Sync) -> Self {
        let turn_table = Self::generate_turn_table(|f| progress("turn", f));
        let flop_table = Self::generate_flop_table(&turn_table, |f| progress("flop", f));

        Self {
            turn: turn_table,
            flop: flop_table,
        }
    }

    /// Build the turn table: for each canonical turn board, compute equity
    /// and expected river equity for every valid hand.
    fn generate_turn_table(progress: impl Fn(f64) + Sync) -> FxHashMap<u64, CacheEntry> {
        let canonical_turns = enumerate_canonical_turns();
        let total = canonical_turns.len();

        let completed = std::sync::atomic::AtomicUsize::new(0);

        // Process each board in parallel, collecting per-board maps.
        let per_board: Vec<Vec<(u64, CacheEntry)>> = canonical_turns
            .par_iter()
            .map(|wb| {
                let board = &wb.cards;
                let hands = valid_hands(board);
                let mut entries = Vec::with_capacity(hands.len());

                for hand in &hands {
                    let equity = compute_equity(*hand, board) as f32;

                    // Expected river equity: average over remaining cards.
                    let rem = remaining_cards(*hand, board);
                    let mut total_eq = 0.0f64;
                    let mut board5 = [board[0], board[1], board[2], board[3],
                                      Card::new(Value::Two, Suit::Spade)];
                    for &river_card in &rem {
                        board5[4] = river_card;
                        total_eq += compute_equity(*hand, &board5);
                    }
                    let expected = if rem.is_empty() {
                        0.5
                    } else {
                        total_eq / rem.len() as f64
                    } as f32;

                    let key = pack_key(*hand, board);
                    entries.push((key, CacheEntry { equity, expected_next_equity: expected }));
                }

                let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                progress(done as f64 / total as f64);

                entries
            })
            .collect();

        // Merge into a single map.
        let total_entries: usize = per_board.iter().map(|v| v.len()).sum();
        let mut map = FxHashMap::with_capacity_and_hasher(total_entries, Default::default());
        for entries in per_board {
            for (key, entry) in entries {
                map.insert(key, entry);
            }
        }
        map
    }

    /// Build the flop table using the turn table for two-street lookahead.
    ///
    /// For each canonical flop board × valid hand:
    /// - `equity` = showdown equity on the 3-card board.
    /// - `expected_next_equity` = average of turn-table equity values over
    ///   all possible turn cards (not the raw turn equity, but the turn-table
    ///   equity which already averages over the river).
    fn generate_flop_table(
        turn_table: &FxHashMap<u64, CacheEntry>,
        progress: impl Fn(f64) + Sync,
    ) -> FxHashMap<u64, CacheEntry> {
        let canonical_flops = enumerate_canonical_flops();
        let total = canonical_flops.len();

        let completed = std::sync::atomic::AtomicUsize::new(0);

        let per_board: Vec<Vec<(u64, CacheEntry)>> = canonical_flops
            .par_iter()
            .map(|wb| {
                let board = &wb.cards;
                let hands = valid_hands(board);
                let mut entries = Vec::with_capacity(hands.len());

                for hand in &hands {
                    let equity = compute_equity(*hand, board) as f32;

                    // Expected turn equity via table lookup.
                    let rem = remaining_cards(*hand, board);
                    let mut total_eq = 0.0f64;
                    let mut count = 0u32;
                    let mut board4 = [board[0], board[1], board[2],
                                      Card::new(Value::Two, Suit::Spade)];

                    for &turn_card in &rem {
                        board4[3] = turn_card;
                        let key = pack_key(*hand, &board4);
                        if let Some(entry) = turn_table.get(&key) {
                            // Use the turn table's equity (which already
                            // accounts for expected river equity).
                            total_eq += f64::from(entry.equity);
                            count += 1;
                        } else {
                            // Fallback: compute directly (shouldn't happen
                            // with canonical boards but handles edge cases).
                            total_eq += compute_equity(*hand, &board4);
                            count += 1;
                        }
                    }

                    let expected = if count == 0 {
                        0.5
                    } else {
                        total_eq / f64::from(count)
                    } as f32;

                    let key = pack_key(*hand, board);
                    entries.push((key, CacheEntry { equity, expected_next_equity: expected }));
                }

                let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                progress(done as f64 / total as f64);

                entries
            })
            .collect();

        let total_entries: usize = per_board.iter().map(|v| v.len()).sum();
        let mut map = FxHashMap::with_capacity_and_hasher(total_entries, Default::default());
        for entries in per_board {
            for (key, entry) in entries {
                map.insert(key, entry);
            }
        }
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(name: &str) -> Card {
        let bytes = name.as_bytes();
        let value = match bytes[0] {
            b'A' => Value::Ace,
            b'K' => Value::King,
            b'Q' => Value::Queen,
            b'J' => Value::Jack,
            b'T' => Value::Ten,
            b'9' => Value::Nine,
            b'8' => Value::Eight,
            b'7' => Value::Seven,
            b'6' => Value::Six,
            b'5' => Value::Five,
            b'4' => Value::Four,
            b'3' => Value::Three,
            b'2' => Value::Two,
            _ => panic!("bad card value: {name}"),
        };
        let suit = match bytes[1] {
            b'h' => Suit::Heart,
            b'd' => Suit::Diamond,
            b'c' => Suit::Club,
            b's' => Suit::Spade,
            _ => panic!("bad card suit: {name}"),
        };
        Card::new(value, suit)
    }

    #[test]
    fn pack_key_deterministic() {
        let hand = [c("Ah"), c("Kd")];
        let board = [c("7s"), c("2d"), c("5c")];
        let k1 = pack_key(hand, &board);
        let k2 = pack_key(hand, &board);
        assert_eq!(k1, k2);

        // Same cards in different order should produce the same key.
        let hand_rev = [c("Kd"), c("Ah")];
        let k3 = pack_key(hand_rev, &board);
        assert_eq!(k1, k3);
    }

    #[test]
    fn card_index_unique() {
        let deck = build_deck();
        let mut indices: Vec<u8> = deck.iter().map(|c| card_index(*c)).collect();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(indices.len(), 52);
        assert_eq!(*indices.last().unwrap(), 51);
    }

    #[test]
    fn conflicts_detects_shared_cards() {
        let hand = [c("Ah"), c("Kd")];
        let board = [c("Ah"), c("7s"), c("2d")];
        assert!(conflicts(hand, &board));

        let board2 = [c("Qh"), c("7s"), c("2d")];
        assert!(!conflicts(hand, &board2));
    }

    #[test]
    fn turn_lookup_basic() {
        // Generate a tiny cache with just one canonical turn board
        // to verify basic functionality.
        let hand = [c("Ah"), c("Kd")];
        let board = [c("7s"), c("2d"), c("5c"), c("Jh")];
        let key = pack_key(hand, &board);

        let mut turn = FxHashMap::default();
        let equity = compute_equity(hand, &board) as f32;
        turn.insert(key, CacheEntry {
            equity,
            expected_next_equity: 0.5,
        });

        let cache = EquityDeltaCache {
            turn,
            flop: FxHashMap::default(),
        };

        let entry = cache.turn_lookup(hand, &board).expect("should find entry");
        assert!((entry.equity - equity).abs() < 1e-6);
    }
}
