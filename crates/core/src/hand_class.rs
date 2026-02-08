//! Hand classification utility for poker hands.
//!
//! Classifies a poker hand's made-hand strength and draw potential given
//! hole cards and a board (flop/turn/river). Only the strongest made-hand
//! class is kept; draw classes are independent and stack on top.

use std::fmt;
use std::str::FromStr;

use crate::poker::{Card, Hand, Rank, Rankable, Suit, Value};

/// A single hand classification category.
///
/// Made hands are ordered strongest-first. Draw categories are independent
/// and evaluated separately from made hands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum HandClass {
    // Made hands (strongest first)
    StraightFlush = 0,
    FourOfAKind = 1,
    FullHouse = 2,
    Flush = 3,
    Straight = 4,
    Set = 5,
    Trips = 6,
    TwoPair = 7,
    TopPair = 8,
    SecondPair = 9,
    ThirdPair = 10,
    LowPair = 11,
    Underpair = 12,
    AceHigh = 13,
    KingHigh = 14,
    // Draws
    FlushDrawNuts = 15,
    FlushDraw = 16,
    BackdoorFlushDraw = 17,
    Oesd = 18,
    Gutshot = 19,
}

impl HandClass {
    const ALL: [Self; 20] = [
        Self::StraightFlush,
        Self::FourOfAKind,
        Self::FullHouse,
        Self::Flush,
        Self::Straight,
        Self::Set,
        Self::Trips,
        Self::TwoPair,
        Self::TopPair,
        Self::SecondPair,
        Self::ThirdPair,
        Self::LowPair,
        Self::Underpair,
        Self::AceHigh,
        Self::KingHigh,
        Self::FlushDrawNuts,
        Self::FlushDraw,
        Self::BackdoorFlushDraw,
        Self::Oesd,
        Self::Gutshot,
    ];

    fn from_discriminant(d: u8) -> Option<Self> {
        Self::ALL.get(d as usize).copied()
    }
}

impl fmt::Display for HandClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::StraightFlush => "StraightFlush",
            Self::FourOfAKind => "FourOfAKind",
            Self::FullHouse => "FullHouse",
            Self::Flush => "Flush",
            Self::Straight => "Straight",
            Self::Set => "Set",
            Self::Trips => "Trips",
            Self::TwoPair => "TwoPair",
            Self::TopPair => "TopPair",
            Self::SecondPair => "SecondPair",
            Self::ThirdPair => "ThirdPair",
            Self::LowPair => "LowPair",
            Self::Underpair => "Underpair",
            Self::AceHigh => "AceHigh",
            Self::KingHigh => "KingHigh",
            Self::FlushDrawNuts => "FlushDrawNuts",
            Self::FlushDraw => "FlushDraw",
            Self::BackdoorFlushDraw => "BackdoorFlushDraw",
            Self::Oesd => "Oesd",
            Self::Gutshot => "Gutshot",
        };
        write!(f, "{s}")
    }
}

/// Error returned when parsing an invalid `HandClass` string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseHandClassError(String);

impl fmt::Display for ParseHandClassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown hand class: '{}'", self.0)
    }
}

impl std::error::Error for ParseHandClassError {}

impl FromStr for HandClass {
    type Err = ParseHandClassError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "StraightFlush" => Ok(Self::StraightFlush),
            "FourOfAKind" => Ok(Self::FourOfAKind),
            "FullHouse" => Ok(Self::FullHouse),
            "Flush" => Ok(Self::Flush),
            "Straight" => Ok(Self::Straight),
            "Set" => Ok(Self::Set),
            "Trips" => Ok(Self::Trips),
            "TwoPair" => Ok(Self::TwoPair),
            "TopPair" => Ok(Self::TopPair),
            "SecondPair" => Ok(Self::SecondPair),
            "ThirdPair" => Ok(Self::ThirdPair),
            "LowPair" => Ok(Self::LowPair),
            "Underpair" => Ok(Self::Underpair),
            "AceHigh" => Ok(Self::AceHigh),
            "KingHigh" => Ok(Self::KingHigh),
            "FlushDrawNuts" => Ok(Self::FlushDrawNuts),
            "FlushDraw" => Ok(Self::FlushDraw),
            "BackdoorFlushDraw" => Ok(Self::BackdoorFlushDraw),
            "Oesd" => Ok(Self::Oesd),
            "Gutshot" => Ok(Self::Gutshot),
            _ => Err(ParseHandClassError(s.to_string())),
        }
    }
}

/// A set of active hand classifications, stored as a `u32` bitset.
///
/// Each bit corresponds to a `HandClass` discriminant. This provides
/// O(1) insert, contains, and iteration, and fits in a single register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HandClassification {
    bits: u32,
}

impl HandClassification {
    /// Create an empty classification.
    #[must_use]
    pub fn new() -> Self {
        Self { bits: 0 }
    }

    /// Check whether a class is present.
    #[must_use]
    pub fn has(self, class: HandClass) -> bool {
        self.bits & (1 << class as u8) != 0
    }

    /// Add a class to the set.
    pub fn add(&mut self, class: HandClass) {
        self.bits |= 1 << class as u8;
    }

    /// Check whether the set is empty.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.bits == 0
    }

    /// Iterate over all active classes in discriminant order.
    pub fn iter(self) -> impl Iterator<Item = HandClass> {
        let bits = self.bits;
        (0u8..20).filter_map(move |i| {
            if bits & (1 << i) != 0 {
                HandClass::from_discriminant(i)
            } else {
                None
            }
        })
    }

    /// Convert to a vector of class name strings.
    #[must_use]
    pub fn to_strings(self) -> Vec<String> {
        self.iter().map(|c| c.to_string()).collect()
    }

    /// Parse from a slice of class name strings.
    ///
    /// # Errors
    ///
    /// Returns an error if any string is not a valid `HandClass` name.
    pub fn from_strings(strings: &[&str]) -> Result<Self, ParseHandClassError> {
        let mut result = Self::new();
        for s in strings {
            let class: HandClass = s.parse()?;
            result.add(class);
        }
        Ok(result)
    }
}

/// Error type for hand classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassifyError {
    /// Board must have 3, 4, or 5 cards.
    InvalidBoardSize(usize),
}

impl fmt::Display for ClassifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBoardSize(n) => {
                write!(f, "invalid board size: {n} (expected 3, 4, or 5)")
            }
        }
    }
}

impl std::error::Error for ClassifyError {}

/// Classify a poker hand given hole cards and a board.
///
/// Returns the strongest made-hand class plus any applicable draw classes.
/// Board must be 3-5 cards (flop, turn, or river).
///
/// # Errors
///
/// Returns `ClassifyError::InvalidBoardSize` if `board.len()` is not 3, 4, or 5.
pub fn classify(hole: [Card; 2], board: &[Card]) -> Result<HandClassification, ClassifyError> {
    if !(3..=5).contains(&board.len()) {
        return Err(ClassifyError::InvalidBoardSize(board.len()));
    }

    let mut result = HandClassification::new();

    classify_made_hand(hole, board, &mut result);

    if board.len() < 5 {
        classify_draws(hole, board, &mut result);
    }

    Ok(result)
}

/// Convert a `Value` to a numeric rank (Two=2, ..., Ace=14).
fn value_rank(v: Value) -> u8 {
    u8::from(v) + 2
}

/// Classify the strongest made hand and add it to the result.
fn classify_made_hand(hole: [Card; 2], board: &[Card], result: &mut HandClassification) {
    let mut cards: Vec<Card> = Vec::with_capacity(7);
    cards.extend_from_slice(&hole);
    cards.extend_from_slice(board);

    let hand = Hand::new_with_cards(cards);
    let rank = hand.rank();

    // Check high-level categories first (strongest to weakest)
    match rank {
        Rank::StraightFlush(_) => {
            result.add(HandClass::StraightFlush);
            return;
        }
        Rank::FourOfAKind(_) => {
            result.add(HandClass::FourOfAKind);
            return;
        }
        Rank::FullHouse(_) => {
            result.add(HandClass::FullHouse);
            return;
        }
        Rank::Flush(_) => {
            result.add(HandClass::Flush);
            return;
        }
        Rank::Straight(_) => {
            result.add(HandClass::Straight);
            return;
        }
        _ => {}
    }

    // For ThreeOfAKind, TwoPair, OnePair, HighCard — we need hole-card analysis
    classify_pair_type(hole, board, rank, result);
}

/// Classify pair-type hands (set, trips, two pair, top/second/third/low pair, underpair)
/// and no-pair hands (ace high, king high).
fn classify_pair_type(
    hole: [Card; 2],
    board: &[Card],
    rank: Rank,
    result: &mut HandClassification,
) {
    let hole_values = [hole[0].value, hole[1].value];
    let mut board_ranks: Vec<u8> = board.iter().map(|c| value_rank(c.value)).collect();
    board_ranks.sort_unstable();
    board_ranks.dedup();
    board_ranks.reverse(); // highest first

    let is_pocket_pair = hole_values[0] == hole_values[1];

    match rank {
        Rank::ThreeOfAKind(_) => classify_three_of_a_kind(hole_values, is_pocket_pair, result),
        Rank::TwoPair(_) => classify_two_pair(hole_values, board, result),
        Rank::OnePair(_) => classify_one_pair(hole_values, &board_ranks, is_pocket_pair, result),
        Rank::HighCard(_) => classify_high_card(hole_values, result),
        _ => {} // Already handled above
    }
}

/// Classify three-of-a-kind as either Set (pocket pair + board card) or Trips (one hole + board pair).
fn classify_three_of_a_kind(
    _hole_values: [Value; 2],
    is_pocket_pair: bool,
    result: &mut HandClassification,
) {
    if is_pocket_pair {
        result.add(HandClass::Set);
    } else {
        result.add(HandClass::Trips);
    }
    // If neither hole card matches each other, it's trips from hole + board pair
    // (handled by the else branch since !is_pocket_pair means one hole card matches a board pair)
}

/// Classify two-pair hands. Must use at least one hole card.
fn classify_two_pair(hole_values: [Value; 2], board: &[Card], result: &mut HandClassification) {
    let board_values: Vec<Value> = board.iter().map(|c| c.value).collect();

    // Check if at least one hole card pairs something
    let h0_pairs_board = board_values.contains(&hole_values[0]);
    let h1_pairs_board = board_values.contains(&hole_values[1]);
    let is_pocket_pair = hole_values[0] == hole_values[1];

    if h0_pairs_board || h1_pairs_board || is_pocket_pair {
        result.add(HandClass::TwoPair);
    }
}

/// Classify one-pair hands relative to board ranks.
fn classify_one_pair(
    hole_values: [Value; 2],
    board_ranks: &[u8],
    is_pocket_pair: bool,
    result: &mut HandClassification,
) {
    if is_pocket_pair {
        let pair_rank = value_rank(hole_values[0]);
        classify_pocket_pair_position(pair_rank, board_ranks, result);
        return;
    }

    // One hole card pairs a board card — find which one
    let h0_rank = value_rank(hole_values[0]);
    let h1_rank = value_rank(hole_values[1]);

    // Which hole card rank matches a board rank?
    let pair_rank = if board_ranks.contains(&h0_rank) {
        h0_rank
    } else if board_ranks.contains(&h1_rank) {
        h1_rank
    } else {
        // Board has a pair but neither hole card makes it — no classification
        return;
    };

    classify_pair_position(pair_rank, board_ranks, result);
}

/// Classify a pocket pair relative to board ranks.
///
/// Position is determined by how many distinct board ranks are above the pair:
/// 0 above → `TopPair` (overpair), 1 above → `SecondPair`, etc.
fn classify_pocket_pair_position(
    pair_rank: u8,
    board_ranks: &[u8],
    result: &mut HandClassification,
) {
    let ranks_above = board_ranks.iter().filter(|&&r| r > pair_rank).count();
    match ranks_above {
        0 => result.add(HandClass::TopPair),
        1 => result.add(HandClass::SecondPair),
        2 => result.add(HandClass::ThirdPair),
        _ => {
            if board_ranks.iter().all(|&r| r > pair_rank) {
                result.add(HandClass::Underpair);
            } else {
                result.add(HandClass::LowPair);
            }
        }
    }
}

/// Given a pair rank and the sorted (descending) distinct board ranks,
/// classify as top/second/third/low pair.
fn classify_pair_position(pair_rank: u8, board_ranks: &[u8], result: &mut HandClassification) {
    // board_ranks is sorted highest-first, deduplicated
    if let Some(pos) = board_ranks.iter().position(|&r| r == pair_rank) {
        match pos {
            0 => result.add(HandClass::TopPair),
            1 => result.add(HandClass::SecondPair),
            2 => result.add(HandClass::ThirdPair),
            _ => result.add(HandClass::LowPair),
        }
    } else {
        // Pocket pair above all board cards: treat as overpair = top pair
        // (pocket pair that's higher than all board ranks)
        if pair_rank > board_ranks[0] {
            result.add(HandClass::TopPair);
        } else {
            result.add(HandClass::Underpair);
        }
    }
}

/// Classify no-pair hands as ace-high or king-high.
fn classify_high_card(hole_values: [Value; 2], result: &mut HandClassification) {
    let max_hole = std::cmp::max(value_rank(hole_values[0]), value_rank(hole_values[1]));
    if max_hole == value_rank(Value::Ace) {
        result.add(HandClass::AceHigh);
    } else if max_hole == value_rank(Value::King) {
        result.add(HandClass::KingHigh);
    }
    // Other high cards get no classification
}

/// Classify draw potential (flush draws, straight draws).
///
/// Draws are suppressed when a completed hand of that type is already made:
/// - No flush draws when a flush or straight flush is already made
/// - No straight draws when a straight, straight flush, or better is made
fn classify_draws(hole: [Card; 2], board: &[Card], result: &mut HandClassification) {
    let sf = result.has(HandClass::StraightFlush);
    let has_flush = result.has(HandClass::Flush) || sf;
    let has_straight = result.has(HandClass::Straight) || sf;

    if !has_flush {
        classify_flush_draws(hole, board, result);
    }
    if !has_straight {
        classify_straight_draws(hole, board, result);
    }
}

/// Detect flush draws, nut flush draws, and backdoor flush draws.
fn classify_flush_draws(hole: [Card; 2], board: &[Card], result: &mut HandClassification) {
    let all_cards: Vec<Card> = std::iter::once(hole[0])
        .chain(std::iter::once(hole[1]))
        .chain(board.iter().copied())
        .collect();

    let mut best_flush_draw: Option<HandClass> = None;

    for &suit in &[Suit::Spade, Suit::Heart, Suit::Diamond, Suit::Club] {
        let suited_cards: Vec<&Card> = all_cards.iter().filter(|c| c.suit == suit).collect();
        let hole_in_suit: Vec<&Card> = hole.iter().filter(|c| c.suit == suit).collect();

        // Must use at least one hole card
        if hole_in_suit.is_empty() {
            continue;
        }

        let count = suited_cards.len();

        if count >= 4 {
            // 4-to-a-flush: check if nut
            let is_nut = is_nut_flush_draw(hole, board, suit);
            let class = if is_nut {
                HandClass::FlushDrawNuts
            } else {
                HandClass::FlushDraw
            };

            best_flush_draw = Some(match best_flush_draw {
                Some(existing) => stronger_flush_draw(existing, class),
                None => class,
            });
        } else if count == 3 && board.len() == 3 {
            // 3-to-a-flush on flop: backdoor flush draw
            let class = HandClass::BackdoorFlushDraw;
            best_flush_draw = Some(match best_flush_draw {
                Some(existing) => stronger_flush_draw(existing, class),
                None => class,
            });
        }
    }

    if let Some(class) = best_flush_draw {
        result.add(class);
    }
}

/// Determine which of two flush draw classes is stronger.
fn stronger_flush_draw(a: HandClass, b: HandClass) -> HandClass {
    // FlushDrawNuts (15) < FlushDraw (16) < BackdoorFlushDraw (17) by discriminant,
    // but FlushDrawNuts is strongest, so lower discriminant = stronger
    if (a as u8) <= (b as u8) { a } else { b }
}

/// Check if the hole cards give a nut flush draw in the given suit.
///
/// The nut flush draw means the hole card in the draw suit is the highest
/// missing card of that suit (i.e., no higher card of that suit exists
/// outside the visible cards).
fn is_nut_flush_draw(hole: [Card; 2], board: &[Card], suit: Suit) -> bool {
    // Find the highest hole card in this suit
    let max_hole_in_suit = hole
        .iter()
        .filter(|c| c.suit == suit)
        .map(|c| value_rank(c.value))
        .max();

    let Some(max_hole_rank) = max_hole_in_suit else {
        return false;
    };

    // Collect all visible cards in this suit
    let visible_in_suit: Vec<u8> = hole
        .iter()
        .chain(board.iter())
        .filter(|c| c.suit == suit)
        .map(|c| value_rank(c.value))
        .collect();

    // Check if any card of this suit higher than our hole card is NOT visible
    // (i.e., could be in the deck). If no such card exists, we have the nut draw.
    for rank in (max_hole_rank + 1)..=value_rank(Value::Ace) {
        if !visible_in_suit.contains(&rank) {
            return false; // A higher card of this suit is out there
        }
    }

    true
}

/// Detect straight draws (OESD and gutshot).
fn classify_straight_draws(hole: [Card; 2], board: &[Card], result: &mut HandClassification) {
    let all_values: Vec<u8> = hole
        .iter()
        .chain(board.iter())
        .map(|c| value_rank(c.value))
        .collect();

    // Build a bitset of present values (bits 2..=14 for Two through Ace)
    let mut present: u16 = 0;
    for &v in &all_values {
        present |= 1 << v;
    }

    // For wheel straights, Ace also counts as 1
    let ace_bit = 1u16 << value_rank(Value::Ace);
    if present & ace_bit != 0 {
        present |= 1 << 1; // Ace as low
    }

    // Count how many missing single cards would complete a 5-card straight
    let mut completions = 0u32;
    let hole_ranks: [u8; 2] = [value_rank(hole[0].value), value_rank(hole[1].value)];

    // Check each possible 5-card straight window
    for low in 1..=10u8 {
        let window: u16 = (0..5).fold(0u16, |acc, i| acc | (1 << (low + i)));

        let missing = window & !present;
        let missing_count = missing.count_ones();

        if missing_count == 1 {
            // Exactly one card completes this straight
            // But we must use at least one hole card in the straight
            let hole_in_window = hole_ranks.iter().any(|&r| {
                let r_bit = 1u16 << r;
                window & r_bit != 0
            }) || (hole_ranks.contains(&value_rank(Value::Ace))
                && window & (1 << 1) != 0);

            if hole_in_window {
                completions += 1;
            }
        }
    }

    // 8+ completions = OESD (typically 2 windows × 4 suits = 8 outs)
    // 4+ completions = Gutshot (typically 1 window × 4 suits = 4 outs)
    // Note: completions count distinct straight windows, not out cards
    if completions >= 2 {
        result.add(HandClass::Oesd);
    } else if completions >= 1 {
        result.add(HandClass::Gutshot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    fn card(value: Value, suit: Suit) -> Card {
        Card::new(value, suit)
    }

    // Convenience aliases
    use Suit::{Club, Diamond, Heart, Spade};
    use Value::{Ace, Eight, Five, Jack, King, Nine, Queen, Seven, Six, Ten, Three, Two};

    // === Type tests ===

    #[timed_test]
    fn hand_class_display_roundtrip() {
        for class in HandClass::ALL {
            let s = class.to_string();
            let parsed: HandClass = s.parse().unwrap();
            assert_eq!(parsed, class, "roundtrip failed for {class:?}");
        }
    }

    #[timed_test]
    fn hand_class_parse_error() {
        let result: Result<HandClass, _> = "NotAClass".parse();
        assert!(result.is_err());
    }

    #[timed_test]
    fn classification_bitset_add_has() {
        let mut c = HandClassification::new();
        assert!(c.is_empty());

        c.add(HandClass::TopPair);
        assert!(c.has(HandClass::TopPair));
        assert!(!c.has(HandClass::FlushDraw));
        assert!(!c.is_empty());
    }

    #[timed_test]
    fn classification_iter() {
        let mut c = HandClassification::new();
        c.add(HandClass::TopPair);
        c.add(HandClass::FlushDrawNuts);

        let classes: Vec<HandClass> = c.iter().collect();
        assert_eq!(classes, vec![HandClass::TopPair, HandClass::FlushDrawNuts]);
    }

    #[timed_test]
    fn classification_to_from_strings() {
        let mut c = HandClassification::new();
        c.add(HandClass::TwoPair);
        c.add(HandClass::Oesd);

        let strings = c.to_strings();
        assert_eq!(strings, vec!["TwoPair", "Oesd"]);

        let strs: Vec<&str> = strings.iter().map(String::as_str).collect();
        let parsed = HandClassification::from_strings(&strs).unwrap();
        assert_eq!(parsed, c);
    }

    #[timed_test]
    fn classification_from_strings_error() {
        let result = HandClassification::from_strings(&["TopPair", "BadClass"]);
        assert!(result.is_err());
    }

    // === Board validation ===

    #[timed_test]
    fn classify_rejects_invalid_board_size() {
        let hole = [card(Ace, Spade), card(King, Spade)];
        assert!(classify(hole, &[card(Two, Heart)]).is_err());
        assert!(classify(hole, &[]).is_err());
    }

    // === Made hand tests ===

    fn assert_classes(hole: [Card; 2], board: &[Card], expected: &[HandClass]) {
        let result = classify(hole, board).unwrap();
        let actual: Vec<HandClass> = result.iter().collect();
        assert_eq!(
            actual, expected,
            "\nhole: {hole:?}\nboard: {board:?}\nexpected: {expected:?}\n  actual: {actual:?}"
        );
    }

    #[timed_test]
    fn top_pair() {
        assert_classes(
            [card(Ace, Diamond), card(Two, Club)],
            &[card(Ace, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[HandClass::TopPair],
        );
    }

    #[timed_test]
    fn two_pair_not_also_top_pair() {
        assert_classes(
            [card(Ace, Diamond), card(Two, Club)],
            &[card(Ace, Heart), card(Two, Spade), card(Ten, Diamond)],
            &[HandClass::TwoPair],
        );
    }

    #[timed_test]
    fn set_from_pocket_pair() {
        assert_classes(
            [card(Seven, Diamond), card(Seven, Club)],
            &[card(Seven, Heart), card(King, Spade), card(Three, Diamond)],
            &[HandClass::Set],
        );
    }

    #[timed_test]
    fn trips_from_board_pair() {
        assert_classes(
            [card(King, Diamond), card(Five, Club)],
            &[card(Five, Heart), card(Five, Spade), card(Three, Diamond)],
            &[HandClass::Trips],
        );
    }

    #[timed_test]
    fn second_pair() {
        // Kd Jc on Ah Ks Td — second pair + gutshot (Q makes AKQJT)
        assert_classes(
            [card(King, Diamond), card(Jack, Club)],
            &[card(Ace, Heart), card(King, Spade), card(Ten, Diamond)],
            &[HandClass::SecondPair, HandClass::Gutshot],
        );
    }

    #[timed_test]
    fn third_pair() {
        assert_classes(
            [card(Jack, Diamond), card(Two, Club)],
            &[card(Ace, Heart), card(King, Spade), card(Jack, Heart)],
            &[HandClass::ThirdPair],
        );
    }

    #[timed_test]
    fn underpair() {
        assert_classes(
            [card(Five, Diamond), card(Five, Club)],
            &[card(Ace, Heart), card(King, Spade), card(Ten, Diamond)],
            &[HandClass::Underpair],
        );
    }

    #[timed_test]
    fn full_house_not_also_set() {
        assert_classes(
            [card(Ace, Diamond), card(Ace, Heart)],
            &[card(Ace, Spade), card(King, Heart), card(King, Diamond)],
            &[HandClass::FullHouse],
        );
    }

    #[timed_test]
    fn four_of_a_kind() {
        assert_classes(
            [card(Seven, Diamond), card(Seven, Club)],
            &[card(Seven, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[HandClass::FourOfAKind],
        );
    }

    #[timed_test]
    fn straight_flush() {
        assert_classes(
            [card(Queen, Club), card(Jack, Club)],
            &[card(Ten, Club), card(Nine, Club), card(Eight, Club)],
            &[HandClass::StraightFlush],
        );
    }

    #[timed_test]
    fn ace_high() {
        assert_classes(
            [card(Ace, Club), card(Eight, Heart)],
            &[card(King, Spade), card(Seven, Diamond), card(Three, Club)],
            &[HandClass::AceHigh],
        );
    }

    #[timed_test]
    fn king_high() {
        assert_classes(
            [card(King, Club), card(Eight, Heart)],
            &[card(Queen, Spade), card(Seven, Diamond), card(Three, Club)],
            &[HandClass::KingHigh],
        );
    }

    // === Draw tests ===

    #[timed_test]
    fn nut_flush_draw() {
        // Ad 5d on Kd 7d 3s — ace high + nut flush draw
        assert_classes(
            [card(Ace, Diamond), card(Five, Diamond)],
            &[
                card(King, Diamond),
                card(Seven, Diamond),
                card(Three, Spade),
            ],
            &[HandClass::AceHigh, HandClass::FlushDrawNuts],
        );
    }

    #[timed_test]
    fn non_nut_flush_draw() {
        assert_classes(
            [card(Five, Diamond), card(Three, Diamond)],
            &[card(King, Diamond), card(Seven, Diamond), card(Two, Spade)],
            &[HandClass::FlushDraw],
        );
    }

    #[timed_test]
    fn backdoor_flush_draw_on_flop() {
        assert_classes(
            [card(Ace, Diamond), card(Five, Diamond)],
            &[card(King, Diamond), card(Seven, Spade), card(Three, Spade)],
            &[HandClass::AceHigh, HandClass::BackdoorFlushDraw],
        );
    }

    #[timed_test]
    fn flush_draw_supersedes_backdoor() {
        // 4-to-flush in diamonds, 3-to-flush in spades — only FlushDrawNuts (+ AceHigh)
        assert_classes(
            [card(Ace, Diamond), card(Five, Diamond)],
            &[
                card(King, Diamond),
                card(Seven, Diamond),
                card(Three, Spade),
            ],
            &[HandClass::AceHigh, HandClass::FlushDrawNuts],
        );
    }

    #[timed_test]
    fn oesd() {
        assert_classes(
            [card(Nine, Heart), card(Eight, Club)],
            &[card(Seven, Spade), card(Six, Diamond), card(Two, Heart)],
            &[HandClass::Oesd],
        );
    }

    #[timed_test]
    fn gutshot() {
        assert_classes(
            [card(Jack, Heart), card(Nine, Club)],
            &[card(Ten, Spade), card(Seven, Diamond), card(Two, Heart)],
            &[HandClass::Gutshot],
        );
    }

    // === Combined made + draw ===

    #[timed_test]
    fn top_pair_plus_nut_flush_draw() {
        assert_classes(
            [card(Ace, Diamond), card(Five, Diamond)],
            &[card(Ace, Heart), card(Seven, Diamond), card(Three, Diamond)],
            &[HandClass::TopPair, HandClass::FlushDrawNuts],
        );
    }

    #[timed_test]
    fn ace_high_plus_nut_flush_draw() {
        assert_classes(
            [card(Ace, Diamond), card(King, Diamond)],
            &[
                card(Queen, Diamond),
                card(Seven, Diamond),
                card(Three, Spade),
            ],
            &[HandClass::AceHigh, HandClass::FlushDrawNuts],
        );
    }

    #[timed_test]
    fn straight_plus_flush_draw() {
        assert_classes(
            [card(Ace, Heart), card(King, Heart)],
            &[card(Queen, Heart), card(Jack, Heart), card(Ten, Spade)],
            &[HandClass::Straight, HandClass::FlushDrawNuts],
        );
    }

    // === River: no draws ===

    #[timed_test]
    fn river_no_draws() {
        assert_classes(
            [card(Five, Diamond), card(Five, Club)],
            &[
                card(Ace, Heart),
                card(King, Spade),
                card(Ten, Diamond),
                card(Five, Heart),
                card(Two, Club),
            ],
            &[HandClass::Set],
        );
    }

    #[timed_test]
    fn river_full_house() {
        assert_classes(
            [card(Ace, Diamond), card(Two, Club)],
            &[
                card(Ace, Heart),
                card(Two, Spade),
                card(Ten, Diamond),
                card(Two, Heart),
                card(King, Club),
            ],
            &[HandClass::FullHouse],
        );
    }

    // === Turn tests ===

    #[timed_test]
    fn turn_flush_draw_no_backdoor() {
        // Ad 5d on Kd 7s 3s 2d — ace high + nut flush draw + gutshot (4 makes A5432 wheel)
        assert_classes(
            [card(Ace, Diamond), card(Five, Diamond)],
            &[
                card(King, Diamond),
                card(Seven, Spade),
                card(Three, Spade),
                card(Two, Diamond),
            ],
            &[
                HandClass::AceHigh,
                HandClass::FlushDrawNuts,
                HandClass::Gutshot,
            ],
        );
    }

    #[timed_test]
    fn turn_straight_made() {
        assert_classes(
            [card(Nine, Heart), card(Eight, Heart)],
            &[
                card(Seven, Heart),
                card(Six, Heart),
                card(Two, Spade),
                card(Ten, Spade),
            ],
            &[HandClass::Straight, HandClass::FlushDraw],
        );
    }

    #[timed_test]
    fn straight_river() {
        assert_classes(
            [card(Ace, Diamond), card(King, Club)],
            &[
                card(Queen, Heart),
                card(Jack, Spade),
                card(Ten, Diamond),
                card(Three, Club),
                card(Two, Spade),
            ],
            &[HandClass::Straight],
        );
    }

    // === Nut flush draw edge case ===

    #[timed_test]
    fn ace_of_suit_is_nut_flush_draw() {
        // Ac 8c on Qc 7s 3h — Ac is the highest club, so nut flush draw
        assert_classes(
            [card(Ace, Club), card(Eight, Club)],
            &[card(Queen, Club), card(Seven, Spade), card(Three, Heart)],
            &[HandClass::AceHigh, HandClass::BackdoorFlushDraw],
        );
    }

    #[timed_test]
    fn low_pair_on_board() {
        // 2c pairs the board Two which is below the third-highest board rank
        assert_classes(
            [card(Two, Club), card(Three, Heart)],
            &[
                card(Ace, Spade),
                card(King, Diamond),
                card(Ten, Heart),
                card(Two, Diamond),
            ],
            &[HandClass::LowPair],
        );
    }

    // === Additional plan test cases ===

    #[timed_test]
    fn flush_on_board() {
        // 5 suited cards = flush, no draws emitted
        assert_classes(
            [card(Ace, Heart), card(Three, Heart)],
            &[
                card(King, Heart),
                card(Nine, Heart),
                card(Two, Heart),
                card(Six, Spade),
                card(Jack, Diamond),
            ],
            &[HandClass::Flush],
        );
    }

    #[timed_test]
    fn overpair_is_top_pair() {
        // AA on K73 — pocket pair above all board cards
        assert_classes(
            [card(Ace, Diamond), card(Ace, Club)],
            &[card(King, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[HandClass::TopPair],
        );
    }

    #[timed_test]
    fn pocket_pair_between_board_ranks() {
        // 88 on AK3 — 2 board ranks above (A, K) → ThirdPair
        assert_classes(
            [card(Eight, Diamond), card(Eight, Club)],
            &[card(Ace, Heart), card(King, Spade), card(Three, Diamond)],
            &[HandClass::ThirdPair],
        );
    }

    #[timed_test]
    fn second_pair_pocket_pair() {
        // QQ on K73 — below K, above 7 and 3 → second pair
        assert_classes(
            [card(Queen, Diamond), card(Queen, Club)],
            &[card(King, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[HandClass::SecondPair],
        );
    }

    #[timed_test]
    fn no_class_for_low_high_card() {
        // No ace or king high, no pair → empty (no classification)
        assert_classes(
            [card(Jack, Diamond), card(Nine, Club)],
            &[card(Queen, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[],
        );
    }

    #[timed_test]
    fn board_pair_no_hole_match() {
        // Board has a pair (77) but hole cards don't match → no pair for us
        // rs_poker says OnePair but neither hole card makes it
        assert_classes(
            [card(Ace, Diamond), card(King, Club)],
            &[card(Seven, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[],
        );
    }

    #[timed_test]
    fn straight_draw_wheel() {
        // A3 on 259 — gutshot to wheel (need 4 for 5-4-3-2-A) + ace high
        assert_classes(
            [card(Ace, Diamond), card(Three, Club)],
            &[card(Two, Heart), card(Five, Spade), card(Nine, Diamond)],
            &[HandClass::AceHigh, HandClass::Gutshot],
        );
    }
}
