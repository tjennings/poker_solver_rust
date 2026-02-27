//! 1,755 canonical (suit-isomorphic) flop representations.
//!
//! In Hold'em, there are C(52,3) = 22,100 possible 3-card flops, but only
//! 1,755 strategically distinct flop types once suit relabeling is considered.
//! Two flops are equivalent if one can be obtained from the other by
//! permuting suits.
//!
//! Each canonical flop carries rich metadata: suit texture, rank texture,
//! high card class, connectedness, and a weight (number of raw flops it
//! represents).

use std::collections::HashMap;
use std::fmt;

use crate::abstraction::CanonicalBoard;
use crate::poker::{Card, Suit, Value};

/// Suit texture of a flop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SuitTexture {
    /// All three cards have different suits.
    Rainbow,
    /// Exactly two cards share a suit.
    TwoTone,
    /// All three cards share the same suit.
    Monotone,
}

/// Rank texture of a flop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RankTexture {
    /// All three ranks are distinct.
    Unpaired,
    /// Exactly one pair (two cards share a rank).
    Paired,
    /// All three cards share the same rank (trips).
    Trips,
}

/// Classification of the highest card on the flop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HighCardClass {
    /// Highest card is Ten or above (T, J, Q, K, A).
    Broadway,
    /// Highest card is Six through Nine.
    Middle,
    /// Highest card is Five or below (2, 3, 4, 5).
    Low,
}

/// Connectedness metrics for a flop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Connectedness {
    /// Gap between highest and middle distinct ranks (0 = connected).
    pub gap_high_mid: u8,
    /// Gap between middle and lowest distinct ranks (0 = connected).
    pub gap_mid_low: u8,
    /// Whether any 5-card straight can use all 3 ranks (including wheel).
    pub has_straight_potential: bool,
}

/// A canonical (suit-isomorphic) flop with metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalFlop {
    cards: [Card; 3],
    suit_texture: SuitTexture,
    rank_texture: RankTexture,
    high_card_class: HighCardClass,
    connectedness: Connectedness,
    weight: u16,
}

impl CanonicalFlop {
    /// The three canonical cards of this flop.
    #[must_use]
    pub fn cards(&self) -> &[Card; 3] {
        &self.cards
    }

    /// Suit texture (rainbow, two-tone, or monotone).
    #[must_use]
    pub fn suit_texture(&self) -> SuitTexture {
        self.suit_texture
    }

    /// Rank texture (unpaired, paired, or trips).
    #[must_use]
    pub fn rank_texture(&self) -> RankTexture {
        self.rank_texture
    }

    /// High card classification (broadway, middle, or low).
    #[must_use]
    pub fn high_card_class(&self) -> HighCardClass {
        self.high_card_class
    }

    /// Connectedness metrics.
    #[must_use]
    pub fn connectedness(&self) -> &Connectedness {
        &self.connectedness
    }

    /// Number of raw (non-canonical) flops this represents.
    #[must_use]
    pub fn weight(&self) -> u16 {
        self.weight
    }
}

/// Generate all 52 cards of a standard deck.
fn full_deck() -> Vec<Card> {
    let mut deck = Vec::with_capacity(52);
    for value in Value::values() {
        for suit in Suit::suits() {
            deck.push(Card::new(value, suit));
        }
    }
    deck
}

use crate::card_utils::value_rank;

/// Classify the suit texture of three cards.
fn classify_suit_texture(cards: [Card; 3]) -> SuitTexture {
    let s0 = cards[0].suit;
    let s1 = cards[1].suit;
    let s2 = cards[2].suit;

    if s0 == s1 && s1 == s2 {
        SuitTexture::Monotone
    } else if s0 == s1 || s0 == s2 || s1 == s2 {
        SuitTexture::TwoTone
    } else {
        SuitTexture::Rainbow
    }
}

/// Classify the rank texture of three cards.
fn classify_rank_texture(cards: [Card; 3]) -> RankTexture {
    let v0 = cards[0].value;
    let v1 = cards[1].value;
    let v2 = cards[2].value;

    if v0 == v1 && v1 == v2 {
        RankTexture::Trips
    } else if v0 == v1 || v0 == v2 || v1 == v2 {
        RankTexture::Paired
    } else {
        RankTexture::Unpaired
    }
}

/// Classify the highest card on the flop.
fn classify_high_card(cards: [Card; 3]) -> HighCardClass {
    let max_rank = cards
        .iter()
        .map(|c| value_rank(c.value))
        .max()
        // Safety: cards always has 3 elements
        .unwrap_or(0);

    if max_rank >= value_rank(Value::Ten) {
        HighCardClass::Broadway
    } else if max_rank >= value_rank(Value::Six) {
        HighCardClass::Middle
    } else {
        HighCardClass::Low
    }
}

/// Compute connectedness metrics for three cards.
///
/// For paired/trips flops, gaps are computed using the distinct ranks present.
/// If fewer than 3 distinct ranks exist, missing gaps default to 0.
fn compute_connectedness(cards: [Card; 3]) -> Connectedness {
    let mut ranks: Vec<u8> = cards.iter().map(|c| value_rank(c.value)).collect();
    ranks.sort_unstable();
    ranks.dedup();

    let (gap_high_mid, gap_mid_low) = match ranks.len() {
        3 => (ranks[2] - ranks[1] - 1, ranks[1] - ranks[0] - 1),
        2 => (ranks[1] - ranks[0] - 1, 0),
        _ => (0, 0),
    };

    let has_straight_potential = check_straight_potential(&ranks);

    Connectedness {
        gap_high_mid,
        gap_mid_low,
        has_straight_potential,
    }
}

/// Check if all distinct ranks can fit within a 5-rank window.
///
/// Standard windows: [r, r+4] for r in 1..=10 (where Ace=14).
/// Wheel window: {14, 2, 3, 4, 5} (Ace plays low).
fn check_straight_potential(distinct_ranks: &[u8]) -> bool {
    // Standard windows: any 5 consecutive ranks [low, low+4]
    // where low ranges from 2 to 10 (so high ranges from 6 to 14)
    for low in 2..=10u8 {
        let high = low + 4;
        if distinct_ranks.iter().all(|&r| r >= low && r <= high) {
            return true;
        }
    }

    // Wheel window: A-2-3-4-5 (ranks: 14, 2, 3, 4, 5)
    let wheel = [2u8, 3, 4, 5, 14];
    if distinct_ranks.iter().all(|&r| wheel.contains(&r)) {
        return true;
    }

    false
}

/// Sort canonical cards for consistent `HashMap` key.
///
/// Cards are sorted by value descending, then suit ascending.
fn sort_cards_for_key(cards: &mut [Card; 3]) {
    cards.sort_by(|a, b| b.value.cmp(&a.value).then(a.suit.cmp(&b.suit)));
}

/// Generate all 1,755 canonical (suit-isomorphic) flops.
///
/// Enumerates all C(52,3) = 22,100 raw flops, canonicalizes each using
/// `CanonicalBoard::from_cards()`, and deduplicates. Each canonical flop
/// includes metadata and a weight (occurrence count).
///
/// The result is sorted deterministically by cards (value descending, suit
/// ascending).
///
/// # Panics
///
/// Cannot panic. The internal call to `CanonicalBoard::from_cards` always
/// succeeds for a 3-card input.
#[must_use]
pub fn all_flops() -> Vec<CanonicalFlop> {
    let deck = full_deck();
    let n = deck.len();

    // Canonicalize all C(52,3) flops and count occurrences
    let mut counts: HashMap<[Card; 3], u16> = HashMap::new();

    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let raw = [deck[i], deck[j], deck[k]];
                // CanonicalBoard::from_cards never fails for 3 cards
                let canonical =
                    CanonicalBoard::from_cards(&raw).expect("3-card board is always valid");

                let mut key: [Card; 3] =
                    [canonical.cards[0], canonical.cards[1], canonical.cards[2]];
                sort_cards_for_key(&mut key);

                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }

    // Build CanonicalFlop for each unique canonical form
    let mut flops: Vec<CanonicalFlop> = counts
        .into_iter()
        .map(|(cards, weight)| CanonicalFlop {
            suit_texture: classify_suit_texture(cards),
            rank_texture: classify_rank_texture(cards),
            high_card_class: classify_high_card(cards),
            connectedness: compute_connectedness(cards),
            cards,
            weight,
        })
        .collect();

    // Sort deterministically: by cards (value desc, suit asc already encoded in Card's Ord)
    flops.sort_by_key(|f| f.cards);

    flops
}

/// Build a lookup map from canonical flop cards to their combinatorial weight.
///
/// The keys are the same `[Card; 3]` arrays returned by `all_flops()` and
/// `canonical_flops()` in `postflop_hands`.
#[must_use]
pub fn flop_weight_map() -> HashMap<[Card; 3], u16> {
    all_flops()
        .into_iter()
        .map(|f| (*f.cards(), f.weight()))
        .collect()
}

/// Look up the combinatorial weight for a list of canonical flop boards.
///
/// Returns a weight for each flop in the input order. Flops not found in the
/// canonical set get weight 1 (conservative fallback).
#[must_use]
pub fn lookup_flop_weights(flops: &[[Card; 3]]) -> Vec<u16> {
    let map = flop_weight_map();
    flops
        .iter()
        .map(|cards| {
            let mut key = *cards;
            sort_cards_for_key(&mut key);
            map.get(&key).copied().unwrap_or(1)
        })
        .collect()
}

impl fmt::Display for CanonicalFlop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let suit_label = match self.suit_texture {
            SuitTexture::Rainbow => "rainbow",
            SuitTexture::TwoTone => "two-tone",
            SuitTexture::Monotone => "monotone",
        };
        let rank_label = match self.rank_texture {
            RankTexture::Unpaired => "unpaired",
            RankTexture::Paired => "paired",
            RankTexture::Trips => "trips",
        };
        let high_label = match self.high_card_class {
            HighCardClass::Broadway => "Broadway",
            HighCardClass::Middle => "Middle",
            HighCardClass::Low => "Low",
        };

        write!(
            f,
            "{} {} {} [{suit_label}, {rank_label}, {high_label}, w={}]",
            self.cards[0], self.cards[1], self.cards[2], self.weight
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    // === Core invariants ===

    #[timed_test]
    fn total_canonical_flops_is_1755() {
        let flops = all_flops();
        assert_eq!(flops.len(), 1755);
    }

    #[timed_test]
    fn total_weight_sum_is_22100() {
        let flops = all_flops();
        let total: u32 = flops.iter().map(|f| u32::from(f.weight())).sum();
        assert_eq!(total, 22_100);
    }

    #[timed_test]
    fn all_canonical_forms_are_unique() {
        let flops = all_flops();
        let mut seen = std::collections::HashSet::new();
        for flop in &flops {
            assert!(seen.insert(flop.cards), "duplicate cards: {:?}", flop.cards);
        }
    }

    // === Rank texture counts ===

    #[timed_test]
    fn unpaired_flop_count() {
        let flops = all_flops();
        let count = flops
            .iter()
            .filter(|f| f.rank_texture() == RankTexture::Unpaired)
            .count();
        assert_eq!(count, 1430);
    }

    #[timed_test]
    fn paired_flop_count() {
        let flops = all_flops();
        let count = flops
            .iter()
            .filter(|f| f.rank_texture() == RankTexture::Paired)
            .count();
        assert_eq!(count, 312);
    }

    #[timed_test]
    fn trips_flop_count() {
        let flops = all_flops();
        let count = flops
            .iter()
            .filter(|f| f.rank_texture() == RankTexture::Trips)
            .count();
        assert_eq!(count, 13);
    }

    // === Weight by texture ===

    #[timed_test]
    fn rainbow_unpaired_weight_is_24() {
        let flops = all_flops();
        for flop in &flops {
            if flop.rank_texture() == RankTexture::Unpaired
                && flop.suit_texture() == SuitTexture::Rainbow
            {
                assert_eq!(
                    flop.weight(),
                    24,
                    "rainbow unpaired flop {} has wrong weight",
                    flop
                );
            }
        }
    }

    #[timed_test]
    fn two_tone_unpaired_weight_is_12() {
        let flops = all_flops();
        for flop in &flops {
            if flop.rank_texture() == RankTexture::Unpaired
                && flop.suit_texture() == SuitTexture::TwoTone
            {
                assert_eq!(
                    flop.weight(),
                    12,
                    "two-tone unpaired flop {} has wrong weight",
                    flop
                );
            }
        }
    }

    #[timed_test]
    fn monotone_unpaired_weight_is_4() {
        let flops = all_flops();
        for flop in &flops {
            if flop.rank_texture() == RankTexture::Unpaired
                && flop.suit_texture() == SuitTexture::Monotone
            {
                assert_eq!(
                    flop.weight(),
                    4,
                    "monotone unpaired flop {} has wrong weight",
                    flop
                );
            }
        }
    }

    #[timed_test]
    fn paired_weight_is_12() {
        let flops = all_flops();
        for flop in &flops {
            if flop.rank_texture() == RankTexture::Paired {
                assert_eq!(flop.weight(), 12, "paired flop {} has wrong weight", flop);
            }
        }
    }

    #[timed_test]
    fn trips_weight_is_4() {
        let flops = all_flops();
        for flop in &flops {
            if flop.rank_texture() == RankTexture::Trips {
                assert_eq!(flop.weight(), 4, "trips flop {} has wrong weight", flop);
            }
        }
    }

    // === Known flop checks ===

    fn find_flop(flops: &[CanonicalFlop], c0: Card, c1: Card, c2: Card) -> Option<&CanonicalFlop> {
        let canonical =
            CanonicalBoard::from_cards(&[c0, c1, c2]).expect("3-card board is always valid");
        let mut key: [Card; 3] = [canonical.cards[0], canonical.cards[1], canonical.cards[2]];
        sort_cards_for_key(&mut key);
        flops.iter().find(|f| f.cards == key)
    }

    #[timed_test]
    fn akq_rainbow_is_broadway_connected_with_straight_potential() {
        let flops = all_flops();
        let flop = find_flop(
            &flops,
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Queen, Suit::Diamond),
        )
        .expect("AKQ rainbow should exist");

        assert_eq!(flop.suit_texture(), SuitTexture::Rainbow);
        assert_eq!(flop.rank_texture(), RankTexture::Unpaired);
        assert_eq!(flop.high_card_class(), HighCardClass::Broadway);
        assert_eq!(flop.connectedness().gap_high_mid, 0);
        assert_eq!(flop.connectedness().gap_mid_low, 0);
        assert!(flop.connectedness().has_straight_potential);
    }

    #[timed_test]
    fn two_three_four_monotone_is_low_connected_with_straight_potential() {
        let flops = all_flops();
        let flop = find_flop(
            &flops,
            Card::new(Value::Two, Suit::Spade),
            Card::new(Value::Three, Suit::Spade),
            Card::new(Value::Four, Suit::Spade),
        )
        .expect("234 monotone should exist");

        assert_eq!(flop.suit_texture(), SuitTexture::Monotone);
        assert_eq!(flop.rank_texture(), RankTexture::Unpaired);
        assert_eq!(flop.high_card_class(), HighCardClass::Low);
        assert_eq!(flop.connectedness().gap_high_mid, 0);
        assert_eq!(flop.connectedness().gap_mid_low, 0);
        assert!(flop.connectedness().has_straight_potential);
    }

    #[timed_test]
    fn aaa_trips_has_correct_metadata() {
        let flops = all_flops();
        // Trips are canonicalized - find one with Ace value
        let flop = flops
            .iter()
            .find(|f| f.rank_texture() == RankTexture::Trips && f.cards[0].value == Value::Ace)
            .expect("AAA trips should exist");

        assert_eq!(flop.rank_texture(), RankTexture::Trips);
        assert_eq!(flop.high_card_class(), HighCardClass::Broadway);
        assert_eq!(flop.weight(), 4);
        assert_eq!(flop.connectedness().gap_high_mid, 0);
        assert_eq!(flop.connectedness().gap_mid_low, 0);
    }

    #[timed_test]
    fn ace_seven_two_rainbow_is_broadway_no_straight_potential() {
        let flops = all_flops();
        let flop = find_flop(
            &flops,
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Seven, Suit::Heart),
            Card::new(Value::Two, Suit::Diamond),
        )
        .expect("A72 rainbow should exist");

        assert_eq!(flop.high_card_class(), HighCardClass::Broadway);
        assert!(!flop.connectedness().has_straight_potential);
    }

    #[timed_test]
    fn wheel_flop_a23_has_straight_potential() {
        let flops = all_flops();
        let flop = find_flop(
            &flops,
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Two, Suit::Heart),
            Card::new(Value::Three, Suit::Diamond),
        )
        .expect("A23 rainbow should exist");

        assert!(
            flop.connectedness().has_straight_potential,
            "A-2-3 should have straight potential via wheel"
        );
    }

    #[timed_test]
    fn nine_eight_seven_is_middle_connected() {
        let flops = all_flops();
        let flop = find_flop(
            &flops,
            Card::new(Value::Nine, Suit::Spade),
            Card::new(Value::Eight, Suit::Heart),
            Card::new(Value::Seven, Suit::Diamond),
        )
        .expect("987 rainbow should exist");

        assert_eq!(flop.high_card_class(), HighCardClass::Middle);
        assert_eq!(flop.connectedness().gap_high_mid, 0);
        assert_eq!(flop.connectedness().gap_mid_low, 0);
        assert!(flop.connectedness().has_straight_potential);
    }

    // === Display ===

    #[timed_test]
    fn display_format_includes_metadata() {
        let flops = all_flops();
        let flop = find_flop(
            &flops,
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Diamond),
        )
        .expect("AK7 rainbow should exist");

        let display = flop.to_string();
        assert!(
            display.contains("rainbow"),
            "display should contain texture: {display}"
        );
        assert!(
            display.contains("unpaired"),
            "display should contain rank: {display}"
        );
        assert!(
            display.contains("Broadway"),
            "display should contain class: {display}"
        );
        assert!(
            display.contains("w=24"),
            "display should contain weight: {display}"
        );
    }

    // === Suit texture counts ===

    #[timed_test]
    fn suit_texture_breakdown_counts() {
        let flops = all_flops();
        let rainbow = flops
            .iter()
            .filter(|f| f.suit_texture() == SuitTexture::Rainbow)
            .count();
        let two_tone = flops
            .iter()
            .filter(|f| f.suit_texture() == SuitTexture::TwoTone)
            .count();
        let monotone = flops
            .iter()
            .filter(|f| f.suit_texture() == SuitTexture::Monotone)
            .count();

        // Each unpaired rank combo (C(13,3)=286) has: 1 rainbow, 3 two-tone, 1 monotone = 5 suit patterns
        // → 286 rainbow unpaired + 858 two-tone unpaired + 286 monotone unpaired = 1430
        // Each paired combo (C(13,1)*C(12,1)=156) has: 1 rainbow-ish, 1 two-tone-ish = 2 suit patterns
        // → 156 + 156 = 312 paired
        // Trips: 13 (only one suit pattern each)
        assert_eq!(rainbow + two_tone + monotone, 1755);

        // Unpaired rainbow: C(13,3) = 286
        let rainbow_unpaired = flops
            .iter()
            .filter(|f| {
                f.suit_texture() == SuitTexture::Rainbow
                    && f.rank_texture() == RankTexture::Unpaired
            })
            .count();
        assert_eq!(rainbow_unpaired, 286);

        // Unpaired monotone: C(13,3) = 286
        let monotone_unpaired = flops
            .iter()
            .filter(|f| {
                f.suit_texture() == SuitTexture::Monotone
                    && f.rank_texture() == RankTexture::Unpaired
            })
            .count();
        assert_eq!(monotone_unpaired, 286);
    }

    // === Weight lookup ===

    #[timed_test]
    fn lookup_flop_weights_returns_correct_values() {
        // AKQ rainbow → weight 24, AKQ monotone → weight 4
        let rainbow = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Queen, Suit::Diamond),
        ];
        let monotone = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Spade),
        ];
        let weights = lookup_flop_weights(&[rainbow, monotone]);
        assert_eq!(weights[0], 24, "rainbow AKQ should have weight 24");
        assert_eq!(weights[1], 4, "monotone AKQ should have weight 4");
    }

    #[timed_test]
    fn lookup_flop_weights_all_flops_sum_to_22100() {
        let all = all_flops();
        let cards: Vec<[Card; 3]> = all.iter().map(|f| *f.cards()).collect();
        let weights = lookup_flop_weights(&cards);
        let total: u32 = weights.iter().map(|&w| u32::from(w)).sum();
        assert_eq!(total, 22_100);
    }

    // === Performance ===

    #[timed_test]
    fn all_flops_completes_quickly() {
        let start = std::time::Instant::now();
        let _flops = all_flops();
        let elapsed = start.elapsed();
        // Debug mode is ~5-10x slower than release; 500ms is generous for debug
        assert!(
            elapsed.as_millis() < 500,
            "all_flops() took {}ms, expected < 500ms",
            elapsed.as_millis()
        );
    }
}
