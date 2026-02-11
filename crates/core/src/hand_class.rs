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
    // Made hands (strongest first, 0-12)
    StraightFlush = 0,
    FourOfAKind = 1,
    FullHouse = 2,
    Flush = 3,
    Straight = 4,
    Set = 5,
    Trips = 6,
    TwoPair = 7,
    Overpair = 8,
    Pair = 9,
    Underpair = 10,
    Overcards = 11,
    HighCard = 12,
    // Draws (13-18)
    ComboDraw = 13,
    FlushDraw = 14,
    BackdoorFlushDraw = 15,
    Oesd = 16,
    Gutshot = 17,
    BackdoorStraightDraw = 18,
}

impl HandClass {
    /// Total number of hand class variants.
    pub const COUNT: usize = 19;

    /// Number of made-hand classes (discriminants `0..NUM_MADE`).
    pub const NUM_MADE: usize = Self::ComboDraw as usize;

    /// Number of draw classes (discriminants `NUM_MADE..COUNT`).
    pub const NUM_DRAWS: usize = Self::COUNT - Self::NUM_MADE;

    /// Sentinel returned by [`HandClassification::strongest_made_id`] when
    /// no made hand is present. Equal to `NUM_MADE`.
    #[allow(clippy::cast_possible_truncation)]
    pub const DRAW_ONLY_ID: u8 = Self::NUM_MADE as u8;

    /// Returns `true` if `class_id` (from `strongest_made_id`) represents a
    /// made hand rather than the draw-only sentinel.
    #[must_use]
    pub const fn is_made_hand_id(class_id: u8) -> bool {
        (class_id as usize) < Self::NUM_MADE
    }

    /// All variants in discriminant order.
    pub const ALL: [Self; Self::COUNT] = [
        Self::StraightFlush,
        Self::FourOfAKind,
        Self::FullHouse,
        Self::Flush,
        Self::Straight,
        Self::Set,
        Self::Trips,
        Self::TwoPair,
        Self::Overpair,
        Self::Pair,
        Self::Underpair,
        Self::Overcards,
        Self::HighCard,
        Self::ComboDraw,
        Self::FlushDraw,
        Self::BackdoorFlushDraw,
        Self::Oesd,
        Self::Gutshot,
        Self::BackdoorStraightDraw,
    ];

    /// Convert a discriminant (0-18) back to a `HandClass`.
    #[must_use]
    pub fn from_discriminant(d: u8) -> Option<Self> {
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
            Self::Overpair => "Overpair",
            Self::Pair => "Pair",
            Self::Underpair => "Underpair",
            Self::Overcards => "Overcards",
            Self::HighCard => "HighCard",
            Self::ComboDraw => "ComboDraw",
            Self::FlushDraw => "FlushDraw",
            Self::BackdoorFlushDraw => "BackdoorFlushDraw",
            Self::Oesd => "Oesd",
            Self::Gutshot => "Gutshot",
            Self::BackdoorStraightDraw => "BackdoorStraightDraw",
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
        // Try exact match first, then case-insensitive
        Self::from_name(s).ok_or_else(|| ParseHandClassError(s.to_string()))
    }
}

impl HandClass {
    /// Parse a hand class name, case-insensitive.
    ///
    /// Accepts both `"Pair"` and `"pair"` formats. Also accepts legacy names
    /// from the 28-variant enum (e.g. `"TopPair"` → `Pair`, `"NutFlush"` → `Flush`).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        let lower = name.to_ascii_lowercase();
        match lower.as_str() {
            "straightflush" | "straight_flush" => Some(Self::StraightFlush),
            "fourofakind" | "four_of_a_kind" | "quads" => Some(Self::FourOfAKind),
            "fullhouse" | "full_house" => Some(Self::FullHouse),
            "flush" | "nutflush" | "nut_flush" => Some(Self::Flush),
            "straight" => Some(Self::Straight),
            "set" | "topset" | "top_set" | "bottomset" | "bottom_set" => Some(Self::Set),
            "trips" => Some(Self::Trips),
            "twopair" | "two_pair" => Some(Self::TwoPair),
            "overpair" | "over_pair" => Some(Self::Overpair),
            "pair" | "toppairtopkicker" | "top_pair_top_kicker" | "tptk" | "toppair"
            | "top_pair" | "secondpair" | "second_pair" | "thirdpair" | "third_pair"
            | "lowpair" | "low_pair" => Some(Self::Pair),
            "underpair" | "under_pair" => Some(Self::Underpair),
            "overcards" | "over_cards" => Some(Self::Overcards),
            "highcard" | "high_card" | "acehigh" | "ace_high" | "kinghigh" | "king_high" => {
                Some(Self::HighCard)
            }
            "combodraw" | "combo_draw" => Some(Self::ComboDraw),
            "flushdraw" | "flush_draw" | "flushdraewnuts" | "flushdraw_nuts"
            | "flush_draw_nuts" | "flushdrawnuts" => Some(Self::FlushDraw),
            "backdoorflushdraw" | "backdoor_flush_draw" => Some(Self::BackdoorFlushDraw),
            "oesd" | "open_ended" => Some(Self::Oesd),
            "gutshot" | "gut_shot" => Some(Self::Gutshot),
            "backdoorstraightdraw" | "backdoor_straight_draw" => {
                Some(Self::BackdoorStraightDraw)
            }
            _ => None,
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

    /// Create a classification from raw bits.
    #[must_use]
    pub fn from_bits(bits: u32) -> Self {
        Self { bits }
    }

    /// Raw bits for use as a compact bucket key.
    ///
    /// Each bit corresponds to a `HandClass` discriminant, so the value
    /// uniquely identifies the combination of made-hand and draw classes.
    #[must_use]
    pub fn bits(self) -> u32 {
        self.bits
    }

    /// Return the discriminant (0..`NUM_MADE`-1) of the strongest made-hand
    /// class present, or [`HandClass::DRAW_ONLY_ID`] if no made hand is set.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn strongest_made_id(self) -> u8 {
        let made_mask = self.bits & ((1 << HandClass::NUM_MADE) - 1);
        if made_mask == 0 {
            HandClass::DRAW_ONLY_ID
        } else {
            made_mask.trailing_zeros() as u8
        }
    }

    /// Return draw flags as a [`HandClass::NUM_DRAWS`]-bit value.
    ///
    /// Bit 0 = `ComboDraw`, bit 1 = `FlushDraw`, bit 2 = `BackdoorFlushDraw`,
    /// bit 3 = `Oesd`, bit 4 = `Gutshot`, bit 5 = `BackdoorStraightDraw`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn draw_flags(self) -> u8 {
        let mask = (1u32 << HandClass::NUM_DRAWS) - 1;
        ((self.bits >> HandClass::NUM_MADE) & mask) as u8
    }

    /// Check whether the set is empty.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.bits == 0
    }

    /// Iterate over all active classes in discriminant order.
    pub fn iter(self) -> impl Iterator<Item = HandClass> {
        let bits = self.bits;
        #[allow(clippy::cast_possible_truncation)]
        (0u8..(HandClass::COUNT as u8)).filter_map(move |i| {
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
    let board_has_pair = board_ranks.windows(2).any(|w| w[0] == w[1]);
    board_ranks.dedup();
    board_ranks.reverse(); // highest first

    let is_pocket_pair = hole_values[0] == hole_values[1];

    match rank {
        Rank::ThreeOfAKind(_) => {
            classify_three_of_a_kind(hole_values, is_pocket_pair, &board_ranks, result);
        }
        Rank::TwoPair(_) => {
            classify_two_pair(hole_values, &board_ranks, is_pocket_pair, board_has_pair, result);
        }
        Rank::OnePair(_) => classify_one_pair(hole_values, &board_ranks, is_pocket_pair, result),
        Rank::HighCard(_) => classify_high_card(hole_values, board, result),
        _ => {} // Already handled above
    }

    // Board has pair/trips but hole cards don't contribute — fall through to high card
    if result.is_empty() {
        classify_high_card(hole_values, board, result);
    }
}

/// Classify three-of-a-kind as Set (pocket pair hits board) or Trips (board pair + hole card).
fn classify_three_of_a_kind(
    _hole_values: [Value; 2],
    is_pocket_pair: bool,
    _board_ranks: &[u8],
    result: &mut HandClassification,
) {
    if is_pocket_pair {
        result.add(HandClass::Set);
    } else {
        result.add(HandClass::Trips);
    }
}

/// Classify two-pair hands.
///
/// - Board has a pair → classify the hole-card pair as top/second/third/low pair.
/// - No board pair → both hole cards pair distinct board cards → `TwoPair`.
fn classify_two_pair(
    hole_values: [Value; 2],
    board_ranks: &[u8],
    is_pocket_pair: bool,
    board_has_pair: bool,
    result: &mut HandClassification,
) {
    if board_has_pair {
        // One pair comes from the board; classify the hole-card pair position
        if is_pocket_pair {
            classify_pocket_pair_position(value_rank(hole_values[0]), board_ranks, result);
        } else {
            let h0_rank = value_rank(hole_values[0]);
            let h1_rank = value_rank(hole_values[1]);
            if board_ranks.contains(&h0_rank) {
                classify_pair_position(h0_rank, hole_values, board_ranks, result);
            } else if board_ranks.contains(&h1_rank) {
                classify_pair_position(h1_rank, hole_values, board_ranks, result);
            }
        }
    } else {
        // No board pair — both hole cards independently pair board cards
        let h0_rank = value_rank(hole_values[0]);
        let h1_rank = value_rank(hole_values[1]);
        if board_ranks.contains(&h0_rank) && board_ranks.contains(&h1_rank) && !is_pocket_pair {
            result.add(HandClass::TwoPair);
        }
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

    classify_pair_position(pair_rank, hole_values, board_ranks, result);
}

/// Classify a pocket pair relative to board ranks.
///
/// 0 above → `Overpair`, all above → `Underpair`, otherwise → `Pair`.
fn classify_pocket_pair_position(
    pair_rank: u8,
    board_ranks: &[u8],
    result: &mut HandClassification,
) {
    let ranks_above = board_ranks.iter().filter(|&&r| r > pair_rank).count();
    if ranks_above == 0 {
        result.add(HandClass::Overpair);
    } else if board_ranks.iter().all(|&r| r > pair_rank) {
        result.add(HandClass::Underpair);
    } else {
        result.add(HandClass::Pair);
    }
}

/// Given a pair rank, hole values, and the sorted (descending) distinct board ranks,
/// classify as `Pair` (one hole card pairs a board card), `Overpair`, or `Underpair`.
fn classify_pair_position(
    pair_rank: u8,
    _hole_values: [Value; 2],
    board_ranks: &[u8],
    result: &mut HandClassification,
) {
    if board_ranks.contains(&pair_rank) {
        result.add(HandClass::Pair);
    } else if pair_rank > board_ranks[0] {
        result.add(HandClass::Overpair);
    } else {
        result.add(HandClass::Underpair);
    }
}

/// Classify no-pair hands: overcards, or high card (ace/king high).
fn classify_high_card(hole_values: [Value; 2], board: &[Card], result: &mut HandClassification) {
    let h0 = value_rank(hole_values[0]);
    let h1 = value_rank(hole_values[1]);
    let max_board = board.iter().map(|c| value_rank(c.value)).max().unwrap_or(0);

    if h0 > max_board && h1 > max_board {
        result.add(HandClass::Overcards);
        return;
    }

    let max_hole = std::cmp::max(h0, h1);
    if max_hole >= value_rank(Value::King) {
        result.add(HandClass::HighCard);
    }
}

/// Classify draw potential (flush draws, straight draws, combo draws).
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

    // Combo draw: both a flush draw and a straight draw
    let has_flush_draw = result.has(HandClass::FlushDraw);
    let has_straight_draw = result.has(HandClass::Oesd) || result.has(HandClass::Gutshot);
    if has_flush_draw && has_straight_draw {
        result.add(HandClass::ComboDraw);
    }
}

/// Detect flush draws and backdoor flush draws.
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
            let class = HandClass::FlushDraw;
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
    // FlushDraw (14) < BackdoorFlushDraw (15) by discriminant;
    // lower discriminant = stronger
    if (a as u8) <= (b as u8) { a } else { b }
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
    } else if board.len() == 3 {
        // Backdoor straight draw: flop only, 3-to-a-straight needing 2 runners
        if has_backdoor_straight(hole, board, present) {
            result.add(HandClass::BackdoorStraightDraw);
        }
    }
}

/// Detect a backdoor straight draw on the flop.
///
/// Looks for any 5-card straight window where exactly 2 cards are missing
/// and at least one hole card is in the window.
fn has_backdoor_straight(hole: [Card; 2], _board: &[Card], present: u16) -> bool {
    let hole_ranks: [u8; 2] = [value_rank(hole[0].value), value_rank(hole[1].value)];

    for low in 1..=10u8 {
        let window: u16 = (0..5).fold(0u16, |acc, i| acc | (1 << (low + i)));
        let missing = window & !present;
        let missing_count = missing.count_ones();

        if missing_count == 2 {
            let hole_in_window = hole_ranks.iter().any(|&r| {
                let r_bit = 1u16 << r;
                window & r_bit != 0
            }) || (hole_ranks.contains(&value_rank(Value::Ace)) && window & (1 << 1) != 0);

            if hole_in_window {
                return true;
            }
        }
    }
    false
}

/// Compute intra-class strength for a made hand.
///
/// Returns a value 1-14 (higher = stronger within the class). The "key rank"
/// depends on the class: for sets it's the set rank, for pairs the kicker, etc.
/// If the class has no meaningful differentiation, returns 1.
///
/// Only meaningful for made-hand classes (discriminants 0-20).
#[must_use]
pub fn intra_class_strength(hole: [Card; 2], board: &[Card], class: HandClass) -> u8 {
    let raw = raw_intra_strength(hole, board, class);
    raw.clamp(1, 14)
}

/// Core strength computation — may return 0 or >14 for edge cases; caller clamps.
fn raw_intra_strength(hole: [Card; 2], board: &[Card], class: HandClass) -> u8 {
    let h0 = value_rank(hole[0].value);
    let h1 = value_rank(hole[1].value);
    let board_ranks: Vec<u8> = board.iter().map(|c| value_rank(c.value)).collect();

    match class {
        HandClass::StraightFlush | HandClass::Straight => {
            let top = straight_top_rank(hole, board);
            15u8.saturating_sub(top)
        }
        HandClass::FourOfAKind => {
            let quad = find_quad_rank(hole, board);
            15u8.saturating_sub(quad)
        }
        HandClass::FullHouse => {
            let trips = find_trips_rank(hole, board);
            15u8.saturating_sub(trips)
        }
        HandClass::Flush => {
            let max_hole_flush = max_hole_flush_rank(hole, board);
            15u8.saturating_sub(max_hole_flush)
        }
        HandClass::Set => {
            // Pocket pair matches board → set rank = pocket rank
            15u8.saturating_sub(h0) // h0 == h1 for pocket pair
        }
        HandClass::Trips => {
            // Board pair + one hole card matches → kicker is the other hole card
            let kicker = h0.max(h1);
            15u8.saturating_sub(kicker)
        }
        HandClass::TwoPair => {
            // Higher of the two pairing hole cards
            let higher = h0.max(h1);
            15u8.saturating_sub(higher)
        }
        HandClass::Overpair | HandClass::Underpair => 15u8.saturating_sub(h0), // pocket pair rank
        HandClass::Pair => {
            // Pair rank of the hole card that pairs the board
            let pair_rank = if board_ranks.contains(&h0) {
                h0
            } else {
                h1
            };
            15u8.saturating_sub(pair_rank)
        }
        HandClass::Overcards => 15u8.saturating_sub(h0.max(h1)),
        HandClass::HighCard => {
            // Highest hole card rank
            15u8.saturating_sub(h0.max(h1))
        }
        // Draw classes — no intra-class strength
        _ => 1,
    }
}

/// Find the top card rank of the made straight (or straight flush).
///
/// Checks each 5-card window from high to low; returns the top rank of
/// the first window where all 5 ranks are present (using at least one hole card).
fn straight_top_rank(hole: [Card; 2], board: &[Card]) -> u8 {
    let mut present: u16 = 0;
    for &c in hole.iter().chain(board.iter()) {
        present |= 1 << value_rank(c.value);
    }
    // Ace-low: duplicate ace as rank 1
    if present & (1 << 14) != 0 {
        present |= 1 << 1;
    }
    for top in (5..=14u8).rev() {
        let low = top - 4;
        let window: u16 = (low..=top).fold(0u16, |acc, r| acc | (1 << r));
        if present & window == window {
            return top;
        }
    }
    5 // fallback (wheel)
}

/// Find the rank of the four-of-a-kind.
fn find_quad_rank(hole: [Card; 2], board: &[Card]) -> u8 {
    let mut counts = [0u8; 15];
    for &c in hole.iter().chain(board.iter()) {
        counts[value_rank(c.value) as usize] += 1;
    }
    for r in (2..=14u8).rev() {
        if counts[r as usize] == 4 {
            return r;
        }
    }
    0
}

/// Find the trips rank in a full house.
fn find_trips_rank(hole: [Card; 2], board: &[Card]) -> u8 {
    let mut counts = [0u8; 15];
    for &c in hole.iter().chain(board.iter()) {
        counts[value_rank(c.value) as usize] += 1;
    }
    for r in (2..=14u8).rev() {
        if counts[r as usize] >= 3 {
            return r;
        }
    }
    0
}

/// Find the highest hole card rank in the flush suit.
fn max_hole_flush_rank(hole: [Card; 2], board: &[Card]) -> u8 {
    let all: Vec<Card> = hole.iter().copied().chain(board.iter().copied()).collect();
    let flush_suit = [Suit::Spade, Suit::Heart, Suit::Diamond, Suit::Club]
        .into_iter()
        .find(|&s| all.iter().filter(|c| c.suit == s).count() >= 5);

    let Some(suit) = flush_suit else { return 0 };
    hole.iter()
        .filter(|c| c.suit == suit)
        .map(|c| value_rank(c.value))
        .max()
        .unwrap_or(0)
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
    fn hand_class_legacy_aliases() {
        // Old 28-variant names should parse to the merged 19-variant enum
        assert_eq!(HandClass::from_name("NutFlush"), Some(HandClass::Flush));
        assert_eq!(HandClass::from_name("TopSet"), Some(HandClass::Set));
        assert_eq!(HandClass::from_name("BottomSet"), Some(HandClass::Set));
        assert_eq!(HandClass::from_name("TopPair"), Some(HandClass::Pair));
        assert_eq!(HandClass::from_name("TopPairTopKicker"), Some(HandClass::Pair));
        assert_eq!(HandClass::from_name("TPTK"), Some(HandClass::Pair));
        assert_eq!(HandClass::from_name("SecondPair"), Some(HandClass::Pair));
        assert_eq!(HandClass::from_name("ThirdPair"), Some(HandClass::Pair));
        assert_eq!(HandClass::from_name("LowPair"), Some(HandClass::Pair));
        assert_eq!(HandClass::from_name("AceHigh"), Some(HandClass::HighCard));
        assert_eq!(HandClass::from_name("KingHigh"), Some(HandClass::HighCard));
        assert_eq!(HandClass::from_name("FlushDrawNuts"), Some(HandClass::FlushDraw));
    }

    #[timed_test]
    fn classification_bitset_add_has() {
        let mut c = HandClassification::new();
        assert!(c.is_empty());

        c.add(HandClass::Pair);
        assert!(c.has(HandClass::Pair));
        assert!(!c.has(HandClass::FlushDraw));
        assert!(!c.is_empty());
    }

    #[timed_test]
    fn classification_iter() {
        let mut c = HandClassification::new();
        c.add(HandClass::Pair);
        c.add(HandClass::FlushDraw);

        let classes: Vec<HandClass> = c.iter().collect();
        assert_eq!(classes, vec![HandClass::Pair, HandClass::FlushDraw]);
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
        let result = HandClassification::from_strings(&["Pair", "BadClass"]);
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
    fn pair_on_board() {
        // A2 on A-7-3: Ace pairs top board rank → Pair
        assert_classes(
            [card(Ace, Diamond), card(Two, Club)],
            &[card(Ace, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[HandClass::Pair, HandClass::BackdoorStraightDraw],
        );
    }

    #[timed_test]
    fn two_pair_not_also_pair() {
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
    fn pair_second_position() {
        // Kd Jc on Ah Ks Td — pairs K (second-highest board rank) → Pair + gutshot
        assert_classes(
            [card(King, Diamond), card(Jack, Club)],
            &[card(Ace, Heart), card(King, Spade), card(Ten, Diamond)],
            &[HandClass::Pair, HandClass::Gutshot],
        );
    }

    #[timed_test]
    fn pair_third_position() {
        // Jd 2c on Ah Ks Jh — pairs J (third-highest) → Pair
        assert_classes(
            [card(Jack, Diamond), card(Two, Club)],
            &[card(Ace, Heart), card(King, Spade), card(Jack, Heart)],
            &[HandClass::Pair, HandClass::BackdoorStraightDraw],
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
    fn high_card_ace() {
        // Ac 8h on Ks 7d 3c → HighCard (ace high)
        assert_classes(
            [card(Ace, Club), card(Eight, Heart)],
            &[card(King, Spade), card(Seven, Diamond), card(Three, Club)],
            &[HandClass::HighCard],
        );
    }

    #[timed_test]
    fn high_card_king() {
        // Kc 8h on Qs 7d 3c → HighCard (king high)
        assert_classes(
            [card(King, Club), card(Eight, Heart)],
            &[card(Queen, Spade), card(Seven, Diamond), card(Three, Club)],
            &[HandClass::HighCard],
        );
    }

    // === Draw tests ===

    #[timed_test]
    fn flush_draw_on_flop() {
        // Ad 5d on Kd 7d 3s — high card + flush draw + backdoor straight
        assert_classes(
            [card(Ace, Diamond), card(Five, Diamond)],
            &[
                card(King, Diamond),
                card(Seven, Diamond),
                card(Three, Spade),
            ],
            &[HandClass::HighCard, HandClass::FlushDraw, HandClass::BackdoorStraightDraw],
        );
    }

    #[timed_test]
    fn flush_draw_low_cards() {
        assert_classes(
            [card(Five, Diamond), card(Three, Diamond)],
            &[card(King, Diamond), card(Seven, Diamond), card(Two, Spade)],
            &[HandClass::FlushDraw, HandClass::BackdoorStraightDraw],
        );
    }

    #[timed_test]
    fn backdoor_flush_draw_on_flop() {
        assert_classes(
            [card(Ace, Diamond), card(Five, Diamond)],
            &[card(King, Diamond), card(Seven, Spade), card(Three, Spade)],
            &[HandClass::HighCard, HandClass::BackdoorFlushDraw, HandClass::BackdoorStraightDraw],
        );
    }

    #[timed_test]
    fn flush_draw_supersedes_backdoor() {
        // 4-to-flush in diamonds → FlushDraw (+ HighCard)
        assert_classes(
            [card(Ace, Diamond), card(Five, Diamond)],
            &[
                card(King, Diamond),
                card(Seven, Diamond),
                card(Three, Spade),
            ],
            &[HandClass::HighCard, HandClass::FlushDraw, HandClass::BackdoorStraightDraw],
        );
    }

    #[timed_test]
    fn oesd() {
        // 9h 8c on 7s 6d 2h: 9,8 > all board (7,6,2) → Overcards + OESD
        assert_classes(
            [card(Nine, Heart), card(Eight, Club)],
            &[card(Seven, Spade), card(Six, Diamond), card(Two, Heart)],
            &[HandClass::Overcards, HandClass::Oesd],
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
    fn pair_plus_flush_draw() {
        // A5d on Ah 7d 3d → Pair + FlushDraw + backdoor straight
        assert_classes(
            [card(Ace, Diamond), card(Five, Diamond)],
            &[card(Ace, Heart), card(Seven, Diamond), card(Three, Diamond)],
            &[HandClass::Pair, HandClass::FlushDraw, HandClass::BackdoorStraightDraw],
        );
    }

    #[timed_test]
    fn overcards_plus_flush_draw() {
        // AKd on Qd 7d 3s — both hole cards above all board ranks + backdoor straight
        assert_classes(
            [card(Ace, Diamond), card(King, Diamond)],
            &[
                card(Queen, Diamond),
                card(Seven, Diamond),
                card(Three, Spade),
            ],
            &[HandClass::Overcards, HandClass::FlushDraw, HandClass::BackdoorStraightDraw],
        );
    }

    #[timed_test]
    fn straight_plus_flush_draw() {
        assert_classes(
            [card(Ace, Heart), card(King, Heart)],
            &[card(Queen, Heart), card(Jack, Heart), card(Ten, Spade)],
            &[HandClass::Straight, HandClass::FlushDraw],
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
        // Ad 5d on Kd 7s 3s 2d — high card + combo draw (flush + gutshot)
        assert_classes(
            [card(Ace, Diamond), card(Five, Diamond)],
            &[
                card(King, Diamond),
                card(Seven, Spade),
                card(Three, Spade),
                card(Two, Diamond),
            ],
            &[
                HandClass::HighCard,
                HandClass::ComboDraw,
                HandClass::FlushDraw,
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

    // === Flush draw edge case ===

    #[timed_test]
    fn ace_of_suit_backdoor_only() {
        // Ac 8c on Qc 7s 3h — only 3-to-a-flush → BackdoorFlushDraw
        assert_classes(
            [card(Ace, Club), card(Eight, Club)],
            &[card(Queen, Club), card(Seven, Spade), card(Three, Heart)],
            &[HandClass::HighCard, HandClass::BackdoorFlushDraw],
        );
    }

    #[timed_test]
    fn pair_low_position_on_board() {
        // 2c pairs the board Two → Pair
        assert_classes(
            [card(Two, Club), card(Three, Heart)],
            &[
                card(Ace, Spade),
                card(King, Diamond),
                card(Ten, Heart),
                card(Two, Diamond),
            ],
            &[HandClass::Pair],
        );
    }

    // === Additional test cases ===

    #[timed_test]
    fn flush_on_board() {
        // 5 suited cards with Ah → Flush (no nut distinction), no draws on river
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
    fn overpair() {
        // AA on K73 — pocket pair above all board cards
        assert_classes(
            [card(Ace, Diamond), card(Ace, Club)],
            &[card(King, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[HandClass::Overpair],
        );
    }

    #[timed_test]
    fn pocket_pair_between_board_ranks() {
        // 88 on AK3 — not overpair, not underpair → Pair
        assert_classes(
            [card(Eight, Diamond), card(Eight, Club)],
            &[card(Ace, Heart), card(King, Spade), card(Three, Diamond)],
            &[HandClass::Pair],
        );
    }

    #[timed_test]
    fn pocket_pair_one_above() {
        // QQ on K73 — 1 rank above → Pair (not Overpair, not Underpair)
        assert_classes(
            [card(Queen, Diamond), card(Queen, Club)],
            &[card(King, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[HandClass::Pair],
        );
    }

    #[timed_test]
    fn no_class_for_low_high_card() {
        // No ace or king high, no pair → only backdoor straight draw
        assert_classes(
            [card(Jack, Diamond), card(Nine, Club)],
            &[card(Queen, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[HandClass::BackdoorStraightDraw],
        );
    }

    #[timed_test]
    fn board_pair_no_hole_match() {
        // Board has a pair (77) but hole cards don't match — AK is overcards
        assert_classes(
            [card(Ace, Diamond), card(King, Club)],
            &[card(Seven, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[HandClass::Overcards],
        );
    }

    #[timed_test]
    fn straight_draw_wheel() {
        // A3 on 259 — gutshot to wheel (need 4 for 5-4-3-2-A) + high card
        assert_classes(
            [card(Ace, Diamond), card(Three, Club)],
            &[card(Two, Heart), card(Five, Spade), card(Nine, Diamond)],
            &[HandClass::HighCard, HandClass::Gutshot],
        );
    }

    // === TwoPair vs board-paired two pair ===

    #[timed_test]
    fn two_pair_both_hole_cards_pair_board() {
        // A7 on A-7-3: both hole cards pair distinct board cards → TwoPair
        assert_classes(
            [card(Ace, Diamond), card(Seven, Club)],
            &[card(Ace, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[HandClass::TwoPair],
        );
    }

    #[timed_test]
    fn two_pair_board_paired_classifies_as_pair() {
        // Kd 5c on Kh 7s 7d 3h: board pair 7-7, K pairs board rank → Pair
        assert_classes(
            [card(King, Diamond), card(Five, Club)],
            &[
                card(King, Heart),
                card(Seven, Spade),
                card(Seven, Diamond),
                card(Three, Heart),
            ],
            &[HandClass::Pair],
        );
    }

    #[timed_test]
    fn two_pair_board_paired_pocket_pair() {
        // KK on A-7-7-3: pocket pair below A, not below all → Pair
        assert_classes(
            [card(King, Diamond), card(King, Club)],
            &[
                card(Ace, Heart),
                card(Seven, Spade),
                card(Seven, Diamond),
                card(Three, Heart),
            ],
            &[HandClass::Pair],
        );
    }

    #[timed_test]
    fn two_pair_board_paired_hole_makes_pair() {
        // A3 on K-7-7-3: 3 pairs board 3 → Pair
        assert_classes(
            [card(Ace, Diamond), card(Three, Club)],
            &[
                card(King, Heart),
                card(Seven, Spade),
                card(Seven, Diamond),
                card(Three, Heart),
            ],
            &[HandClass::Pair],
        );
    }

    #[timed_test]
    fn two_pair_both_hole_cards_on_turn() {
        // J9 on J-9-K-2: both hole cards pair, no board pair → TwoPair
        assert_classes(
            [card(Jack, Diamond), card(Nine, Club)],
            &[
                card(Jack, Heart),
                card(Nine, Spade),
                card(King, Diamond),
                card(Two, Heart),
            ],
            &[HandClass::TwoPair],
        );
    }

    #[timed_test]
    fn two_pair_board_paired_no_hole_contribution() {
        // Board has pair 7-7, neither hole card pairs anything — A2 is high card (river, no draws)
        assert_classes(
            [card(Ace, Diamond), card(Two, Club)],
            &[
                card(King, Heart),
                card(Seven, Spade),
                card(Seven, Diamond),
                card(Three, Heart),
                card(Six, Club),
            ],
            &[HandClass::HighCard],
        );
    }

    // === Merged class tests ===

    #[timed_test]
    fn ace_pairs_top_board_rank() {
        // AK on A-7-3: Ace pairs top board rank → Pair
        assert_classes(
            [card(Ace, Diamond), card(King, Club)],
            &[card(Ace, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[HandClass::Pair],
        );
    }

    #[timed_test]
    fn king_pairs_top_board_rank() {
        // AK on K-7-3: K pairs top board rank → Pair
        assert_classes(
            [card(Ace, Diamond), card(King, Club)],
            &[card(King, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[HandClass::Pair],
        );
    }

    #[timed_test]
    fn queen_pairs_top_board_rank() {
        // KQ on K-7-3: K pairs top rank → Pair
        assert_classes(
            [card(King, Diamond), card(Queen, Club)],
            &[card(King, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[HandClass::Pair],
        );
    }

    #[timed_test]
    fn overcards() {
        // AK on 7-5-3: both hole cards above all board ranks + backdoor straight
        assert_classes(
            [card(Ace, Diamond), card(King, Club)],
            &[card(Seven, Heart), card(Five, Spade), card(Three, Diamond)],
            &[HandClass::Overcards, HandClass::BackdoorStraightDraw],
        );
    }

    #[timed_test]
    fn overcards_vs_high_card() {
        // AQ on K-7-3: Queen(12) < King(13), not overcards → HighCard + backdoor straight
        assert_classes(
            [card(Ace, Diamond), card(Queen, Club)],
            &[card(King, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[HandClass::HighCard, HandClass::BackdoorStraightDraw],
        );
    }

    #[timed_test]
    fn flush_made_nut() {
        // Ah Kd on Qh Jh 5h 3h 2c → Flush (was NutFlush)
        assert_classes(
            [card(Ace, Heart), card(King, Diamond)],
            &[
                card(Queen, Heart),
                card(Jack, Heart),
                card(Five, Heart),
                card(Three, Heart),
                card(Two, Club),
            ],
            &[HandClass::Flush],
        );
    }

    #[timed_test]
    fn flush_made_non_nut() {
        // Kh Kd on Qh Jh 5h 3h 2c → Flush (same class as nut)
        assert_classes(
            [card(King, Heart), card(King, Diamond)],
            &[
                card(Queen, Heart),
                card(Jack, Heart),
                card(Five, Heart),
                card(Three, Heart),
                card(Two, Club),
            ],
            &[HandClass::Flush],
        );
    }

    #[timed_test]
    fn set_top_position() {
        // KK on K-7-3: set matching highest board rank → Set
        assert_classes(
            [card(King, Diamond), card(King, Club)],
            &[card(King, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[HandClass::Set],
        );
    }

    #[timed_test]
    fn set_middle_position() {
        // 77 on K-7-3: set matching middle board rank → Set
        assert_classes(
            [card(Seven, Diamond), card(Seven, Club)],
            &[card(King, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[HandClass::Set],
        );
    }

    #[timed_test]
    fn set_bottom_position() {
        // 33 on K-7-3: set matching lowest board rank → Set
        assert_classes(
            [card(Three, Heart), card(Three, Club)],
            &[card(King, Heart), card(Seven, Spade), card(Three, Diamond)],
            &[HandClass::Set],
        );
    }

    #[timed_test]
    fn combo_draw() {
        // 8h7h on 9h6h2c: flush draw + OESD = combo draw
        assert_classes(
            [card(Eight, Heart), card(Seven, Heart)],
            &[card(Nine, Heart), card(Six, Heart), card(Two, Club)],
            &[HandClass::ComboDraw, HandClass::FlushDraw, HandClass::Oesd],
        );
    }

    #[timed_test]
    fn backdoor_straight_draw() {
        assert_classes(
            [card(Jack, Diamond), card(Ten, Club)],
            &[card(Ace, Heart), card(Five, Spade), card(Two, Diamond)],
            &[HandClass::BackdoorStraightDraw],
        );
    }

    #[timed_test]
    fn no_backdoor_straight_draw_on_turn() {
        assert_classes(
            [card(Jack, Diamond), card(Ten, Club)],
            &[
                card(Ace, Heart),
                card(Five, Spade),
                card(Two, Diamond),
                card(Queen, Club),
            ],
            &[HandClass::Gutshot],
        );
    }

    #[timed_test]
    fn overpair_in_two_pair_context() {
        // AA on A-7-7-3 → actually FullHouse
        assert_classes(
            [card(Ace, Diamond), card(Ace, Club)],
            &[
                card(Ace, Heart),
                card(Seven, Spade),
                card(Seven, Diamond),
                card(Three, Heart),
            ],
            &[HandClass::FullHouse],
        );
    }

    #[timed_test]
    fn combo_draw_with_flush() {
        // Ah 7h on 8h 6h 5c: flush draw + OESD = combo draw
        assert_classes(
            [card(Ace, Heart), card(Seven, Heart)],
            &[card(Eight, Heart), card(Six, Heart), card(Five, Club)],
            &[
                HandClass::HighCard,
                HandClass::ComboDraw,
                HandClass::FlushDraw,
                HandClass::Oesd,
            ],
        );
    }

    #[timed_test]
    fn set_on_turn() {
        // KK on K-7-3-2 → Set
        assert_classes(
            [card(King, Diamond), card(King, Club)],
            &[
                card(King, Heart),
                card(Seven, Spade),
                card(Three, Diamond),
                card(Two, Club),
            ],
            &[HandClass::Set],
        );
    }

    #[timed_test]
    fn set_on_turn_bottom() {
        // 22 on K-7-3-2 → Set
        assert_classes(
            [card(Two, Diamond), card(Two, Club)],
            &[
                card(King, Heart),
                card(Seven, Spade),
                card(Three, Diamond),
                card(Two, Heart),
            ],
            &[HandClass::Set],
        );
    }

    // === strongest_made_id tests ===

    #[timed_test]
    fn strongest_made_id_straight_flush() {
        let mut c = HandClassification::new();
        c.add(HandClass::StraightFlush);
        c.add(HandClass::FlushDraw); // draw should be ignored
        assert_eq!(c.strongest_made_id(), 0);
    }

    #[timed_test]
    fn strongest_made_id_pair_with_draw() {
        let mut c = HandClassification::new();
        c.add(HandClass::Pair);
        c.add(HandClass::Oesd);
        assert_eq!(c.strongest_made_id(), HandClass::Pair as u8);
    }

    #[timed_test]
    fn strongest_made_id_no_made_hand() {
        let mut c = HandClassification::new();
        c.add(HandClass::FlushDraw);
        c.add(HandClass::Gutshot);
        assert_eq!(c.strongest_made_id(), 13);
    }

    #[timed_test]
    fn strongest_made_id_empty() {
        let c = HandClassification::new();
        assert_eq!(c.strongest_made_id(), 13);
    }

    // === draw_flags tests ===

    #[timed_test]
    fn draw_flags_combo() {
        let mut c = HandClassification::new();
        c.add(HandClass::ComboDraw);
        c.add(HandClass::FlushDraw);
        c.add(HandClass::Oesd);
        let flags = c.draw_flags();
        assert_eq!(flags & 1, 1, "ComboDraw bit");      // bit 0
        assert_eq!((flags >> 1) & 1, 1, "FlushDraw bit"); // bit 1
        assert_eq!((flags >> 3) & 1, 1, "Oesd bit");      // bit 3
    }

    #[timed_test]
    fn draw_flags_none() {
        let mut c = HandClassification::new();
        c.add(HandClass::Pair);
        assert_eq!(c.draw_flags(), 0);
    }

    // === intra_class_strength tests ===

    #[timed_test]
    fn strength_overpair_aces_vs_kings() {
        let board = &[card(Queen, Heart), card(Seven, Spade), card(Three, Diamond)];
        let aa = [card(Ace, Diamond), card(Ace, Club)];
        let kk = [card(King, Diamond), card(King, Club)];
        let s_aa = intra_class_strength(aa, board, HandClass::Overpair);
        let s_kk = intra_class_strength(kk, board, HandClass::Overpair);
        assert!(s_aa < s_kk, "AA ({s_aa}) should be stronger (lower) than KK ({s_kk})");
    }

    #[timed_test]
    fn strength_pair_rank() {
        let board = &[card(King, Heart), card(Seven, Spade), card(Three, Diamond)];
        // AK on K73 — pair rank is K(13), 15-13=2
        let ak = [card(Ace, Diamond), card(King, Club)];
        let s_ak = intra_class_strength(ak, board, HandClass::Pair);
        assert_eq!(s_ak, 2);
        // Q7 on K73 — pair rank is 7, 15-7=8
        let q7 = [card(Queen, Diamond), card(Seven, Club)];
        let s_q7 = intra_class_strength(q7, board, HandClass::Pair);
        assert_eq!(s_q7, 8);
        assert!(s_ak < s_q7, "K-pair ({s_ak}) stronger than 7-pair ({s_q7})");
    }

    #[timed_test]
    fn strength_straight_ace_high_vs_nine_high() {
        let board_a = &[
            card(Queen, Heart),
            card(Jack, Spade),
            card(Ten, Diamond),
            card(Two, Club),
            card(Three, Heart),
        ];
        let ak = [card(Ace, Diamond), card(King, Club)];
        let s_a = intra_class_strength(ak, board_a, HandClass::Straight);

        let board_9 = &[
            card(Seven, Heart),
            card(Six, Spade),
            card(Five, Diamond),
            card(Two, Club),
            card(Three, Heart),
        ];
        let n89 = [card(Nine, Diamond), card(Eight, Club)];
        let s_9 = intra_class_strength(n89, board_9, HandClass::Straight);

        assert!(s_a < s_9, "A-high ({s_a}) stronger than 9-high ({s_9})");
    }

    #[timed_test]
    fn strength_set_rank() {
        let board = &[card(King, Heart), card(Seven, Spade), card(Three, Diamond)];
        let kk = [card(King, Diamond), card(King, Club)];
        let s = intra_class_strength(kk, board, HandClass::Set);
        // King = rank 13, so 15 - 13 = 2
        assert_eq!(s, 2);
    }

    #[timed_test]
    fn strength_quads() {
        let board = &[
            card(Seven, Heart),
            card(Seven, Spade),
            card(Three, Diamond),
            card(King, Club),
            card(Two, Heart),
        ];
        let s77 = [card(Seven, Diamond), card(Seven, Club)];
        let s = intra_class_strength(s77, board, HandClass::FourOfAKind);
        assert_eq!(s, 8);
    }

    #[timed_test]
    fn strength_full_house() {
        let board = &[
            card(Ace, Heart),
            card(Ace, Spade),
            card(King, Diamond),
            card(King, Club),
            card(Two, Heart),
        ];
        let aa = [card(Ace, Diamond), card(Three, Club)];
        let s = intra_class_strength(aa, board, HandClass::FullHouse);
        assert_eq!(s, 1);
    }

    #[timed_test]
    fn strength_flush_ace_high() {
        let board = &[
            card(Queen, Heart),
            card(Jack, Heart),
            card(Five, Heart),
            card(Three, Heart),
            card(Two, Club),
        ];
        let ah = [card(Ace, Heart), card(King, Diamond)];
        let s = intra_class_strength(ah, board, HandClass::Flush);
        // Ace = rank 14, so 15 - 14 = 1
        assert_eq!(s, 1);
    }

    #[timed_test]
    fn strength_high_card() {
        let board = &[card(Queen, Heart), card(Seven, Spade), card(Three, Diamond)];
        let ak = [card(Ace, Diamond), card(King, Club)];
        let s = intra_class_strength(ak, board, HandClass::HighCard);
        // max(14, 13) = 14, 15 - 14 = 1
        assert_eq!(s, 1);
    }

    #[timed_test]
    fn strength_draw_class_returns_1() {
        let board = &[card(Nine, Heart), card(Six, Heart), card(Two, Club)];
        let h87 = [card(Eight, Heart), card(Seven, Heart)];
        let s = intra_class_strength(h87, board, HandClass::Oesd);
        assert_eq!(s, 1);
    }
}
