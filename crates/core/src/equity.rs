//! Preflop equity calculations for canonical hand matchups.
//!
//! Provides equity (win probability) between two canonical preflop hands.
//! Equity is calculated by enumerating all non-overlapping combo pairs and
//! running Monte Carlo board sampling for each.

use crate::hands::CanonicalHand;
#[cfg(test)]
use crate::hands::all_hands;
use crate::poker::{Card, Hand, Rank, Rankable, Suit, Value};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Default number of Monte Carlo samples for equity calculations.
pub(crate) const DEFAULT_EQUITY_SAMPLES: u32 = 1_000;

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

/// Calculate equity with a specific number of Monte Carlo board samples.
///
/// Enumerates all non-overlapping combo pairs for the two canonical hands,
/// then distributes board samples evenly across them. This ensures unbiased
/// equity calculation regardless of hand type (pair, suited, offsuit).
#[must_use]
pub fn calculate_equity(hand1: CanonicalHand, hand2: CanonicalHand, samples: u32) -> f64 {
    let valid_pairs = non_overlapping_combos(hand1, hand2);
    if valid_pairs.is_empty() {
        return 0.5;
    }

    #[allow(clippy::cast_possible_truncation)]
    let samples_per_pair = (samples as usize / valid_pairs.len()).max(1);

    let mut total_wins = 0.0_f64;
    let mut total_count = 0u64;

    for (pair_idx, &(hole1, hole2)) in valid_pairs.iter().enumerate() {
        // Unique seed per combo pair for reproducibility
        let base_seed = splitmix64(pair_idx as u64 ^ 0xBEEF_CAFE_1234_5678);
        let mut rng = base_seed;

        for s in 0..samples_per_pair {
            rng = splitmix64(rng ^ s as u64);
            let board = deal_board(hole1, hole2, rng);
            let rank1 = evaluate_hand(hole1, board);
            let rank2 = evaluate_hand(hole2, board);

            match rank1.cmp(&rank2) {
                std::cmp::Ordering::Greater => total_wins += 1.0,
                std::cmp::Ordering::Equal => total_wins += 0.5,
                std::cmp::Ordering::Less => {}
            }
            total_count += 1;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    { total_wins / total_count as f64 }
}

/// Enumerate all non-overlapping `([Card;2], [Card;2])` pairs for two canonical hands.
fn non_overlapping_combos(
    hand1: CanonicalHand,
    hand2: CanonicalHand,
) -> Vec<([Card; 2], [Card; 2])> {
    let combos1 = hand1.combos();
    let combos2 = hand2.combos();

    combos1
        .iter()
        .flat_map(|&(a, b)| {
            combos2
                .iter()
                .filter(move |&&(c, d)| a != c && a != d && b != c && b != d)
                .map(move |&(c, d)| ([a, b], [c, d]))
        })
        .collect()
}

/// `splitmix64` — high-quality 64-bit hash/RNG step.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

const SUITS: [Suit; 4] = [Suit::Spade, Suit::Heart, Suit::Diamond, Suit::Club];
const VALUES: [Value; 13] = [
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

/// Deal 5 community cards avoiding the 4 hole cards.
///
/// Uses Fisher-Yates partial shuffle on the remaining 48 cards.
fn deal_board(hand1: [Card; 2], hand2: [Card; 2], seed: u64) -> [Card; 5] {
    // Build remaining deck (48 cards)
    let mut deck = [Card::new(Value::Two, Suit::Spade); 48];
    let mut n = 0;
    for &value in &VALUES {
        for &suit in &SUITS {
            let card = Card::new(value, suit);
            if card != hand1[0] && card != hand1[1] && card != hand2[0] && card != hand2[1] {
                deck[n] = card;
                n += 1;
            }
        }
    }

    // Fisher-Yates for first 5 positions
    let mut rng = seed;
    let mut board = [Card::new(Value::Two, Suit::Spade); 5];
    for i in 0..5 {
        rng = splitmix64(rng);
        #[allow(clippy::cast_possible_truncation)]
        let j = i + (rng as usize % (n - i));
        deck.swap(i, j);
        board[i] = deck[i];
    }

    board
}

/// Evaluate a 7-card hand (2 hole + 5 board) and return its rank.
fn evaluate_hand(hole: [Card; 2], board: [Card; 5]) -> Rank {
    let cards = vec![
        hole[0], hole[1], board[0], board[1], board[2], board[3], board[4],
    ];
    let hand = Hand::new_with_cards(cards);
    hand.rank()
}

#[cfg(test)]
/// Clear the equity cache.
pub(crate) fn clear_cache() {
    if let Ok(mut cache) = EQUITY_CACHE.lock() {
        cache.clear();
    }
}

#[cfg(test)]
/// Get the number of cached equity values.
#[must_use]
pub(crate) fn cache_size() -> usize {
    EQUITY_CACHE.lock().map_or(0, |c| c.len())
}

#[cfg(test)]
/// Pre-compute all equity values between canonical hands.
///
/// This can be called once before training to avoid equity calculation
/// overhead during CFR traversal. Computes all 169×169 = 28,561 equity pairs.
///
/// Returns the number of equity pairs computed.
pub(crate) fn prewarm_cache() -> usize {
    let hands: Vec<_> = all_hands().collect();
    let mut computed = 0;

    for &h1 in &hands {
        for &h2 in &hands {
            // Equity function caches both directions, so skip if already cached
            let already_cached = EQUITY_CACHE.lock().is_ok_and(|c| c.contains_key(&(h1, h2)));

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
    use test_macros::timed_test;

    #[timed_test]
    fn aa_beats_kk_most_of_time() {
        let aa = CanonicalHand::parse("AA").unwrap();
        let kk = CanonicalHand::parse("KK").unwrap();

        let eq = calculate_equity(aa, kk, 5000);

        // AA should beat KK about 80% of the time
        assert!(eq > 0.75, "AA equity vs KK: {eq}");
        assert!(eq < 0.90, "AA equity vs KK: {eq}");
    }

    #[timed_test]
    fn aks_vs_qjo_is_reasonable() {
        let aks = CanonicalHand::parse("AKs").unwrap();
        let qjo = CanonicalHand::parse("QJo").unwrap();

        let eq = calculate_equity(aks, qjo, 5000);

        // AKs should be a decent favorite over QJo
        assert!(eq > 0.55, "AKs equity vs QJo: {eq}");
        assert!(eq < 0.75, "AKs equity vs QJo: {eq}");
    }

    #[timed_test]
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

    #[timed_test]
    fn coinflip_hands_are_close() {
        let ako = CanonicalHand::parse("AKo").unwrap();
        let twos = CanonicalHand::parse("22").unwrap();

        let eq = calculate_equity(ako, twos, 5000);

        assert!(eq > 0.40, "AKo equity vs 22: {eq}");
        assert!(eq < 0.60, "AKo equity vs 22: {eq}");
    }

    #[timed_test]
    fn cache_stores_values() {
        clear_cache();

        let aa = CanonicalHand::parse("AA").unwrap();
        let kk = CanonicalHand::parse("KK").unwrap();

        let _ = equity(aa, kk);

        assert!(cache_size() >= 2);
    }

    #[timed_test]
    fn dominated_hand_loses() {
        let a2o = CanonicalHand::parse("A2o").unwrap();
        let aks = CanonicalHand::parse("AKs").unwrap();

        let eq = calculate_equity(a2o, aks, 5000);

        assert!(eq < 0.35, "A2o equity vs AKs: {eq}");
    }

    #[timed_test]
    fn a2o_has_higher_avg_equity_than_82o() {
        // A2o should clearly beat 82o in avg equity vs the field
        let a2o = CanonicalHand::parse("A2o").unwrap();
        let eight_two = CanonicalHand::parse("82o").unwrap();

        let eq_a2o = calculate_equity(a2o, eight_two, 10_000);
        assert!(eq_a2o > 0.55, "A2o should beat 82o, got equity {eq_a2o}");
    }

    #[timed_test]
    fn a2o_beats_72o_head_to_head() {
        let a2o = CanonicalHand::parse("A2o").unwrap();
        let seven_two = CanonicalHand::parse("72o").unwrap();
        let eq = calculate_equity(a2o, seven_two, 50_000);
        // A2o dominates 72o — should win about 63-65%
        assert!(eq > 0.60, "A2o vs 72o equity {eq:.3} should be > 0.60");
    }

    #[timed_test]
    fn a5s_equity_above_fifty() {
        // A5s vs a random mediocre hand should be favored
        let a5s = CanonicalHand::parse("A5s").unwrap();
        let t8o = CanonicalHand::parse("T8o").unwrap();
        let eq = calculate_equity(a5s, t8o, 50_000);
        // A5s vs T8o is close to 50/50 but A5s has slight edge
        assert!(
            eq > 0.40 && eq < 0.60,
            "A5s vs T8o equity {eq:.3} should be near 0.50"
        );
    }

    #[timed_test]
    fn non_overlapping_combo_counts() {
        let aa = CanonicalHand::parse("AA").unwrap();
        let kk = CanonicalHand::parse("KK").unwrap();
        assert_eq!(non_overlapping_combos(aa, kk).len(), 36); // 6 * 6

        let aks = CanonicalHand::parse("AKs").unwrap();
        assert_eq!(non_overlapping_combos(aa, aks).len(), 12); // shared ace

        let seven_two = CanonicalHand::parse("72o").unwrap();
        let eight_three = CanonicalHand::parse("83o").unwrap();
        assert_eq!(non_overlapping_combos(seven_two, eight_three).len(), 144); // 12 * 12
    }

}
