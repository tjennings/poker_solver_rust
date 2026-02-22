//! Expected Hand Strength (EHS) computation for preflop hand abstraction.
//!
//! Wraps the existing `HandStrengthCalculator` to provide the EHS feature vectors
//! used in k-means clustering of hands into buckets.

use crate::abstraction::HandStrengthCalculator;
use crate::poker::{Card, Suit, Value};

/// Feature vector for k-means hand clustering: `[EHS, positive_potential, negative_potential]`.
pub type EhsFeatures = [f64; 3];

/// Compute EHS feature vector for a hand on a given board.
///
/// Delegates to `HandStrengthCalculator` which performs exhaustive opponent enumeration.
/// The `_samples` parameter is retained for API compatibility but not used (enumeration is exact).
#[must_use]
pub fn ehs_features(hole: [Card; 2], board: &[Card]) -> EhsFeatures {
    let calc = HandStrengthCalculator::new();
    let hs = match board.len() {
        3 => calc.calculate_flop(board, (hole[0], hole[1])),
        4 => calc.calculate_turn(board, (hole[0], hole[1])),
        5 => calc.calculate_river(board, (hole[0], hole[1])),
        _ => return [0.5, 0.0, 0.0],
    };
    [f64::from(hs.ehs), f64::from(hs.ppot), f64::from(hs.npot)]
}

/// Compute average EHS feature vector over sampled boards, given a set of blocked cards.
///
/// Samples `num_samples` runouts from the remaining deck, computes EHS features for each,
/// and returns the average. Used for clustering (`canonical_hand`, `flop_texture`) pairs.
#[must_use]
pub fn avg_ehs_features(
    hole: [Card; 2],
    known_board: &[Card],
    num_samples: u32,
    seed: u64,
) -> EhsFeatures {
    let boards = sample_runouts(hole, known_board, num_samples, seed);
    average_features(
        &boards
            .iter()
            .map(|b| ehs_features(hole, b))
            .collect::<Vec<_>>(),
    )
}

/// Sample `n` random runouts completing `known_board` to the next street.
///
/// Returns boards of length `known_board.len() + 1` (adds one card per sample).
#[allow(clippy::cast_possible_truncation)]
fn sample_runouts(hole: [Card; 2], known_board: &[Card], n: u32, seed: u64) -> Vec<Vec<Card>> {
    let live = live_deck(hole, known_board);
    // live.len() ≤ 52, safe to truncate to u32
    let n = n.min(live.len() as u32);
    (0..n)
        .map(|i| {
            // splitmix64 output used as index; usize truncation is harmless for deck indexing
            #[allow(clippy::cast_possible_truncation)]
            let idx = splitmix64(seed ^ u64::from(i)) as usize % live.len();
            let mut board = known_board.to_vec();
            board.push(live[idx]);
            board
        })
        .collect()
}

/// Build the live deck: all 52 cards minus hole cards and known board.
fn live_deck(hole: [Card; 2], board: &[Card]) -> Vec<Card> {
    all_cards()
        .filter(|c| *c != hole[0] && *c != hole[1] && !board.contains(c))
        .collect()
}

/// Average a slice of `EhsFeatures` vectors.
#[allow(clippy::cast_precision_loss)]
fn average_features(features: &[EhsFeatures]) -> EhsFeatures {
    if features.is_empty() {
        return [f64::NAN, f64::NAN, f64::NAN]; // sentinel: hand blocked on this board
    }
    // features.len() ≤ millions in typical use; precision loss is acceptable
    let n = features.len() as f64;
    let sum = features.iter().fold([0.0f64; 3], |acc, f| {
        [acc[0] + f[0], acc[1] + f[1], acc[2] + f[2]]
    });
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

/// Iterator over all 52 cards.
fn all_cards() -> impl Iterator<Item = Card> {
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
    const SUITS: [Suit; 4] = [Suit::Spade, Suit::Heart, Suit::Diamond, Suit::Club];
    VALUES
        .into_iter()
        .flat_map(|v| SUITS.into_iter().map(move |s| Card::new(v, s)))
}

/// `splitmix64` — high-quality 64-bit hash/RNG step (same as used in `equity.rs`).
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::{Card, Suit, Value};
    use test_macros::timed_test;

    fn make_card(v: Value, s: Suit) -> Card {
        Card::new(v, s)
    }

    #[timed_test]
    fn ehs_features_river_returns_three_components() {
        let hole = [
            make_card(Value::Ace, Suit::Spade),
            make_card(Value::King, Suit::Spade),
        ];
        let board = vec![
            make_card(Value::Two, Suit::Spade),
            make_card(Value::Five, Suit::Heart),
            make_card(Value::Eight, Suit::Diamond),
            make_card(Value::Ten, Suit::Club),
            make_card(Value::Queen, Suit::Heart),
        ];
        let features = ehs_features(hole, &board);
        assert_eq!(features.len(), 3);
        assert!(
            features[0] >= 0.0 && features[0] <= 1.0,
            "EHS out of range: {}",
            features[0]
        );
        // River: no potential
        assert_eq!(features[1], 0.0, "river ppot should be 0");
        assert_eq!(features[2], 0.0, "river npot should be 0");
    }

    #[timed_test]
    fn ehs_features_turn_draws_have_positive_potential() {
        // Flush draw on turn: As 5s on 2s 8s Tc Qd
        let hole = [
            make_card(Value::Ace, Suit::Spade),
            make_card(Value::Five, Suit::Spade),
        ];
        let board = vec![
            make_card(Value::Two, Suit::Spade),
            make_card(Value::Eight, Suit::Spade),
            make_card(Value::Ten, Suit::Club),
            make_card(Value::Queen, Suit::Diamond),
        ];
        let features = ehs_features(hole, &board);
        assert!(
            features[1] > 0.05,
            "flush draw ppot should be > 0.05, got {}",
            features[1]
        );
    }

    #[timed_test(60)]
    #[ignore = "slow"]
    fn ehs_features_flop_draws_have_positive_potential() {
        // Flush draw on flop: As 5s on 2s 8s Tc
        let hole = [
            make_card(Value::Ace, Suit::Spade),
            make_card(Value::Five, Suit::Spade),
        ];
        let board = vec![
            make_card(Value::Two, Suit::Spade),
            make_card(Value::Eight, Suit::Spade),
            make_card(Value::Ten, Suit::Club),
        ];
        let features = ehs_features(hole, &board);
        assert!(
            features[1] > 0.05,
            "flush draw ppot should be > 0.05, got {}",
            features[1]
        );
    }

    #[timed_test]
    fn ehs_features_strong_hand_high_ehs() {
        // AA on 2-7 rainbow board: very strong
        let hole = [
            make_card(Value::Ace, Suit::Spade),
            make_card(Value::Ace, Suit::Heart),
        ];
        let board = vec![
            make_card(Value::Two, Suit::Diamond),
            make_card(Value::Seven, Suit::Club),
            make_card(Value::Three, Suit::Heart),
            make_card(Value::Four, Suit::Spade),
            make_card(Value::Eight, Suit::Diamond),
        ];
        let features = ehs_features(hole, &board);
        assert!(
            features[0] > 0.7,
            "AA on low board EHS should be > 0.7, got {}",
            features[0]
        );
    }

    #[timed_test]
    fn ehs_features_unknown_board_len_returns_default() {
        let hole = [
            make_card(Value::Ace, Suit::Spade),
            make_card(Value::King, Suit::Heart),
        ];
        let features = ehs_features(hole, &[]);
        assert_eq!(features, [0.5, 0.0, 0.0]);
    }

    #[timed_test]
    fn average_features_empty_returns_nan() {
        let result = average_features(&[]);
        assert!(result[0].is_nan(), "empty features should return NaN sentinel");
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());
    }

    #[timed_test]
    fn average_features_single_passthrough() {
        let feat = [0.7, 0.1, 0.2];
        let result = average_features(&[feat]);
        for i in 0..3 {
            assert!((result[i] - feat[i]).abs() < 1e-10, "mismatch at index {i}");
        }
    }

    #[timed_test]
    fn average_features_computes_mean() {
        let feats = [[0.2, 0.0, 0.0], [0.8, 0.0, 0.0]];
        let result = average_features(&feats);
        assert!((result[0] - 0.5).abs() < 1e-10);
    }

    #[timed_test]
    fn live_deck_excludes_hole_and_board() {
        let hole = [
            make_card(Value::Ace, Suit::Spade),
            make_card(Value::King, Suit::Heart),
        ];
        let board = vec![make_card(Value::Two, Suit::Diamond)];
        let deck = live_deck(hole, &board);
        assert_eq!(deck.len(), 49);
        assert!(!deck.contains(&hole[0]));
        assert!(!deck.contains(&hole[1]));
        assert!(!deck.contains(&board[0]));
    }

    #[timed_test]
    fn sample_runouts_returns_correct_board_length() {
        let hole = [
            make_card(Value::Ace, Suit::Spade),
            make_card(Value::King, Suit::Heart),
        ];
        let board = vec![
            make_card(Value::Two, Suit::Diamond),
            make_card(Value::Three, Suit::Club),
            make_card(Value::Four, Suit::Spade),
        ];
        let boards = sample_runouts(hole, &board, 10, 42);
        assert_eq!(boards.len(), 10);
        for b in &boards {
            assert_eq!(b.len(), 4, "each runout should add one card");
        }
    }

    #[timed_test]
    fn avg_ehs_features_in_range() {
        let hole = [
            make_card(Value::Ace, Suit::Spade),
            make_card(Value::King, Suit::Spade),
        ];
        let board = vec![
            make_card(Value::Two, Suit::Spade),
            make_card(Value::Five, Suit::Heart),
            make_card(Value::Eight, Suit::Club),
        ];
        let feats = avg_ehs_features(hole, &board, 5, 42);
        assert!(
            feats[0] >= 0.0 && feats[0] <= 1.0,
            "avg EHS out of range: {}",
            feats[0]
        );
    }
}
