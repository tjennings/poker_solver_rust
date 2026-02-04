//! Preflop equity calculations for canonical hand matchups.
//!
//! Provides equity (win probability) between two canonical preflop hands.
//! Equity is calculated via Monte Carlo simulation for accuracy.

use crate::hands::{all_hands, CanonicalHand};
use crate::poker::{Card, Hand, Rank, Rankable, Suit, Value};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Default number of Monte Carlo samples for equity calculations.
/// Lower values are faster but less accurate.
pub const DEFAULT_EQUITY_SAMPLES: u32 = 1_000;

/// Cache for equity calculations to avoid recomputation.
static EQUITY_CACHE: LazyLock<Mutex<HashMap<(CanonicalHand, CanonicalHand), f64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Calculate equity of `hand1` vs `hand2` (probability hand1 wins).
///
/// Uses Monte Carlo simulation with a default sample size.
/// Results are cached for repeated lookups.
///
/// # Returns
///
/// A value in `[0.0, 1.0]` representing hand1's equity.
/// - 1.0 means hand1 always wins
/// - 0.5 means a coin flip
/// - 0.0 means hand1 always loses
#[must_use]
pub fn equity(hand1: CanonicalHand, hand2: CanonicalHand) -> f64 {
    // Check cache first
    if let Ok(cache) = EQUITY_CACHE.lock()
        && let Some(&eq) = cache.get(&(hand1, hand2))
    {
        return eq;
    }

    // Calculate equity
    let eq = calculate_equity(hand1, hand2, DEFAULT_EQUITY_SAMPLES);

    // Store in cache
    if let Ok(mut cache) = EQUITY_CACHE.lock() {
        cache.insert((hand1, hand2), eq);
        // Only store inverse if hands are different (avoid overwriting same key)
        if hand1 != hand2 {
            cache.insert((hand2, hand1), 1.0 - eq);
        }
    }

    eq
}

/// Calculate equity with a specific number of Monte Carlo samples.
///
/// # Arguments
///
/// * `hand1` - First canonical hand
/// * `hand2` - Second canonical hand
/// * `samples` - Number of Monte Carlo samples
///
/// # Returns
///
/// Equity of hand1 vs hand2 (probability hand1 wins).
#[must_use]
pub fn calculate_equity(hand1: CanonicalHand, hand2: CanonicalHand, samples: u32) -> f64 {
    let mut wins1 = 0u32;
    let mut wins2 = 0u32;
    let mut ties = 0u32;

    // We need a deterministic RNG for reproducibility
    let mut rng_state: u64 = 0x1234_5678_9ABC_DEF0;

    for i in 0..samples {
        // Mix in the iteration counter for variety
        let seed = rng_state.wrapping_add(u64::from(i).wrapping_mul(0x9E37_79B9_7F4A_7C15));

        // Simple xorshift RNG
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;

        // Deal specific cards for each canonical hand
        let (cards1, cards2) = deal_hands(hand1, hand2, seed);

        // Deal community cards (5 cards)
        let board = deal_board(cards1, cards2, seed.wrapping_add(0xDEAD_BEEF));

        // Evaluate hands
        let rank1 = evaluate_hand(cards1, board);
        let rank2 = evaluate_hand(cards2, board);

        match rank1.cmp(&rank2) {
            std::cmp::Ordering::Greater => wins1 += 1,
            std::cmp::Ordering::Less => wins2 += 1,
            std::cmp::Ordering::Equal => ties += 1,
        }
    }

    // Equity = wins + (ties / 2)
    let total = f64::from(wins1 + wins2 + ties);
    (f64::from(wins1) + f64::from(ties) / 2.0) / total
}

const SUITS: [Suit; 4] = [Suit::Spade, Suit::Heart, Suit::Diamond, Suit::Club];

/// Deal specific cards for two canonical hands.
fn deal_hands(hand1: CanonicalHand, hand2: CanonicalHand, seed: u64) -> ([Card; 2], [Card; 2]) {
    // Select suits for hand1 based on seed
    #[allow(clippy::cast_possible_truncation)]
    let suit_idx1 = (seed % 4) as usize;
    let suit_idx2 = if hand1.is_suited() {
        suit_idx1 // Same suit for suited hands
    } else {
        // For pairs or offsuit - ensure different suit
        #[allow(clippy::cast_possible_truncation)]
        let offset = ((seed >> 8) % 3) as usize;
        (suit_idx1 + 1 + offset) % 4
    };

    let card1_1 = Card::new(hand1.high_value(), SUITS[suit_idx1]);
    let card1_2 = Card::new(hand1.low_value(), SUITS[suit_idx2]);

    // Select suits for hand2, avoiding conflicts with hand1
    let mut used_cards = vec![card1_1, card1_2];

    let card2_1 = find_available_card(hand2.high_value(), &used_cards, seed >> 16);
    used_cards.push(card2_1);

    let card2_2 = if hand2.is_suited() {
        // Same suit as card2_1
        let target_suit = card2_1.suit;
        let card = Card::new(hand2.low_value(), target_suit);
        if used_cards.contains(&card) {
            // If blocked, try other suits (shouldn't happen for valid canonical hands)
            find_available_card(hand2.low_value(), &used_cards, seed >> 24)
        } else {
            card
        }
    } else {
        // For pairs or offsuit - different suit from card2_1
        find_available_card_offsuit(hand2.low_value(), card2_1.suit, &used_cards, seed >> 24)
    };

    ([card1_1, card1_2], [card2_1, card2_2])
}

/// Find an available card with the given value.
fn find_available_card(value: Value, used: &[Card], seed: u64) -> Card {
    #[allow(clippy::cast_possible_truncation)]
    let start = (seed % 4) as usize;
    for i in 0..4 {
        let suit = SUITS[(start + i) % 4];
        let card = Card::new(value, suit);
        if !used.contains(&card) {
            return card;
        }
    }
    // Fallback (shouldn't happen in valid scenarios)
    Card::new(value, SUITS[0])
}

/// Find an available card with a different suit.
fn find_available_card_offsuit(value: Value, avoid_suit: Suit, used: &[Card], seed: u64) -> Card {
    #[allow(clippy::cast_possible_truncation)]
    let start = (seed % 4) as usize;
    for i in 0..4 {
        let suit = SUITS[(start + i) % 4];
        if suit == avoid_suit {
            continue;
        }
        let card = Card::new(value, suit);
        if !used.contains(&card) {
            return card;
        }
    }
    // Fallback to any available
    find_available_card(value, used, seed)
}

/// Deal 5 community cards avoiding the hole cards.
fn deal_board(hand1: [Card; 2], hand2: [Card; 2], seed: u64) -> [Card; 5] {
    let mut used: Vec<Card> = vec![hand1[0], hand1[1], hand2[0], hand2[1]];
    let mut board = [Card::new(Value::Two, Suit::Spade); 5];
    let mut rng = seed;

    for card in &mut board {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;

        // Pick a card from remaining deck
        let remaining = 52 - used.len();
        #[allow(clippy::cast_possible_truncation)]
        let idx = (rng % remaining as u64) as usize;

        *card = nth_available_card(idx, &used);
        used.push(*card);
    }

    board
}

/// Get the nth available card (not in used list).
fn nth_available_card(n: usize, used: &[Card]) -> Card {
    let values = [
        Value::Two,
        Value::Three,
        Value::Four,
        Value::Five,
        Value::Six,
        Value::Seven,
        Value::Eight,
        Value::Nine,
        Value::Ten,
        Value::Jack,
        Value::Queen,
        Value::King,
        Value::Ace,
    ];

    let mut count = 0;
    for &value in &values {
        for &suit in &SUITS {
            let card = Card::new(value, suit);
            if !used.contains(&card) {
                if count == n {
                    return card;
                }
                count += 1;
            }
        }
    }

    // Fallback
    Card::new(Value::Two, Suit::Spade)
}

/// Evaluate a 7-card hand (2 hole + 5 board) and return its rank.
fn evaluate_hand(hole: [Card; 2], board: [Card; 5]) -> Rank {
    // Build all 7 cards
    let cards = vec![
        hole[0], hole[1], board[0], board[1], board[2], board[3], board[4],
    ];
    let hand = Hand::new_with_cards(cards);

    // rs_poker evaluates the best 5-card hand from 7 cards
    hand.rank()
}

/// Clear the equity cache.
pub fn clear_cache() {
    if let Ok(mut cache) = EQUITY_CACHE.lock() {
        cache.clear();
    }
}

/// Get the number of cached equity values.
#[must_use]
pub fn cache_size() -> usize {
    EQUITY_CACHE.lock().map_or(0, |c| c.len())
}

/// Pre-compute all equity values between canonical hands.
///
/// This can be called once before training to avoid equity calculation
/// overhead during CFR traversal. Computes all 169×169 = 28,561 equity pairs.
///
/// Returns the number of equity pairs computed.
pub fn prewarm_cache() -> usize {
    let hands: Vec<_> = all_hands().collect();
    let mut computed = 0;

    for &h1 in &hands {
        for &h2 in &hands {
            // Equity function caches both directions, so skip if already cached
            let already_cached = EQUITY_CACHE
                .lock()
                .is_ok_and(|c| c.contains_key(&(h1, h2)));

            if !already_cached {
                let _ = equity(h1, h2);
                computed += 1;
            }
        }
    }

    computed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hands::CanonicalHand;

    #[test]
    fn aa_beats_kk_most_of_time() {
        let aa = CanonicalHand::parse("AA").unwrap();
        let kk = CanonicalHand::parse("KK").unwrap();

        let eq = calculate_equity(aa, kk, 5000);

        // AA should beat KK about 80% of the time
        assert!(eq > 0.75, "AA equity vs KK: {eq}");
        assert!(eq < 0.90, "AA equity vs KK: {eq}");
    }

    #[test]
    fn aks_vs_qjo_is_reasonable() {
        let aks = CanonicalHand::parse("AKs").unwrap();
        let qjo = CanonicalHand::parse("QJo").unwrap();

        let eq = calculate_equity(aks, qjo, 5000);

        // AKs should be a decent favorite over QJo
        assert!(eq > 0.55, "AKs equity vs QJo: {eq}");
        assert!(eq < 0.75, "AKs equity vs QJo: {eq}");
    }

    #[test]
    fn equity_is_symmetric() {
        let ak = CanonicalHand::parse("AKs").unwrap();
        let qj = CanonicalHand::parse("QJs").unwrap();

        let eq_ak = calculate_equity(ak, qj, 5000);
        let eq_qj = calculate_equity(qj, ak, 5000);

        // Should be roughly inverse
        let sum = eq_ak + eq_qj;
        assert!(
            (sum - 1.0).abs() < 0.05,
            "Equities should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn coinflip_hands_are_close() {
        // AKo vs 22 is approximately a coinflip
        let ako = CanonicalHand::parse("AKo").unwrap();
        let twos = CanonicalHand::parse("22").unwrap();

        let eq = calculate_equity(ako, twos, 5000);

        // Should be roughly 50/50
        assert!(eq > 0.40, "AKo equity vs 22: {eq}");
        assert!(eq < 0.60, "AKo equity vs 22: {eq}");
    }

    #[test]
    fn cache_stores_values() {
        clear_cache();

        let aa = CanonicalHand::parse("AA").unwrap();
        let kk = CanonicalHand::parse("KK").unwrap();

        // First call calculates
        let _ = equity(aa, kk);

        // Should have cached both directions
        assert!(cache_size() >= 2);
    }

    #[test]
    fn dominated_hand_loses() {
        // A2o vs AKs - A2 is dominated
        let a2o = CanonicalHand::parse("A2o").unwrap();
        let aks = CanonicalHand::parse("AKs").unwrap();

        let eq = calculate_equity(a2o, aks, 5000);

        // A2o should lose most of the time
        assert!(eq < 0.35, "A2o equity vs AKs: {eq}");
    }
}
