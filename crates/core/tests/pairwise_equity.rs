//! Integration tests for pairwise bucket equity computation.
//!
//! Verifies that `compute_pairwise_bucket_equity` produces accurate equity
//! values that reflect actual hand-vs-hand strength, unlike the centroid-ratio
//! approximation.

use poker_solver_core::hands::CanonicalHand;
use poker_solver_core::poker::{Card, Suit, Value};
use poker_solver_core::preflop::hand_buckets::{
    compute_bucket_pair_equity, compute_pairwise_bucket_equity,
};

fn card(v: Value, s: Suit) -> Card {
    Card::new(v, s)
}

/// Helper: find hand index for a given canonical hand name.
fn hand_index(hands: &[CanonicalHand], name: &str) -> usize {
    let target = CanonicalHand::parse(name).unwrap();
    hands.iter().position(|h| *h == target).unwrap()
}

/// Use a small set of hands for fast tests.
fn small_hand_set() -> Vec<CanonicalHand> {
    ["AA", "KK", "AKs", "AKo", "65s", "72o", "KJo", "QTs", "33", "T9o"]
        .iter()
        .map(|s| CanonicalHand::parse(s).unwrap())
        .collect()
}

/// River board: AA vs 72o on a dry board. AA should have >0.85 equity.
#[test]
fn pairwise_river_aa_dominates_72o() {
    let hands = small_hand_set();
    let river_board: &[Card] = &[
        card(Value::King, Suit::Diamond),
        card(Value::Nine, Suit::Club),
        card(Value::Five, Suit::Spade),
        card(Value::Three, Suit::Heart),
        card(Value::Eight, Suit::Diamond),
    ];

    let aa_idx = hand_index(&hands, "AA");
    let trash_idx = hand_index(&hands, "72o");

    let num_boards = 1;
    let mut assignments = vec![2u16; hands.len() * num_boards];
    assignments[aa_idx] = 0;
    assignments[trash_idx] = 1;

    let boards: Vec<&[Card]> = vec![river_board];
    let eq = compute_pairwise_bucket_equity(&hands, &boards, &assignments, 3, num_boards, 1.0);

    let aa_vs_72o = eq.get(0, 1);
    assert!(
        aa_vs_72o > 0.85,
        "AA vs 72o on river should be >0.85, got {aa_vs_72o}"
    );
}

/// Self-equity should be approximately 0.5.
#[test]
fn pairwise_self_equity_approximately_half() {
    let hands = small_hand_set();
    let river_board: &[Card] = &[
        card(Value::King, Suit::Diamond),
        card(Value::Nine, Suit::Club),
        card(Value::Five, Suit::Spade),
        card(Value::Three, Suit::Heart),
        card(Value::Eight, Suit::Diamond),
    ];

    // All hands in bucket 0.
    let num_boards = 1;
    let assignments = vec![0u16; hands.len() * num_boards];
    let boards: Vec<&[Card]> = vec![river_board];
    let eq = compute_pairwise_bucket_equity(&hands, &boards, &assignments, 1, num_boards, 1.0);

    let self_eq = eq.get(0, 0);
    assert!(
        (self_eq - 0.5).abs() < 0.02,
        "self-equity should be ~0.5, got {self_eq}"
    );
}

/// Nut straight on flop should have high equity vs weak hands (exhaustive).
#[test]
fn pairwise_flop_nut_straight_dominates() {
    let hands = small_hand_set();
    // 234 rainbow flop — 65s makes the nut straight.
    let flop: &[Card] = &[
        card(Value::Two, Suit::Diamond),
        card(Value::Three, Suit::Club),
        card(Value::Four, Suit::Heart),
    ];

    let straight_idx = hand_index(&hands, "65s");
    let weak_idx = hand_index(&hands, "KJo");

    let num_boards = 1;
    let mut assignments = vec![2u16; hands.len() * num_boards];
    assignments[straight_idx] = 0;
    assignments[weak_idx] = 1;

    let boards: Vec<&[Card]> = vec![flop];
    let eq = compute_pairwise_bucket_equity(&hands, &boards, &assignments, 3, num_boards, 1.0);

    let straight_vs_weak = eq.get(0, 1);
    assert!(
        straight_vs_weak > 0.75,
        "nut straight vs KJo on 234 flop should be >0.75, got {straight_vs_weak}"
    );
}

/// Sampled equity should be close to exhaustive equity (within tolerance).
#[test]
fn pairwise_sampled_close_to_exhaustive() {
    let hands = small_hand_set();
    let flop: &[Card] = &[
        card(Value::Two, Suit::Diamond),
        card(Value::Three, Suit::Club),
        card(Value::Four, Suit::Heart),
    ];

    let strong_idx = hand_index(&hands, "65s");
    let weak_idx = hand_index(&hands, "KJo");

    let num_boards = 1;
    let mut assignments = vec![2u16; hands.len() * num_boards];
    assignments[strong_idx] = 0;
    assignments[weak_idx] = 1;

    let boards: Vec<&[Card]> = vec![flop];
    let exact = compute_pairwise_bucket_equity(&hands, &boards, &assignments, 3, num_boards, 1.0);
    let sampled =
        compute_pairwise_bucket_equity(&hands, &boards, &assignments, 3, num_boards, 0.2);

    let diff = (exact.get(0, 1) - sampled.get(0, 1)).abs();
    assert!(
        diff < 0.06,
        "sampled vs exhaustive should be within 0.06, diff = {diff} (exact={}, sampled={})",
        exact.get(0, 1),
        sampled.get(0, 1)
    );
}

/// Pairwise equity should be more extreme than centroid-ratio for dominant hands.
#[test]
fn pairwise_more_extreme_than_centroid_ratio() {
    let hands = small_hand_set();
    let river_board: &[Card] = &[
        card(Value::King, Suit::Diamond),
        card(Value::Nine, Suit::Club),
        card(Value::Five, Suit::Spade),
        card(Value::Three, Suit::Heart),
        card(Value::Eight, Suit::Diamond),
    ];

    let aa_idx = hand_index(&hands, "AA");
    let trash_idx = hand_index(&hands, "72o");

    let num_boards = 1;
    let mut assignments = vec![2u16; hands.len() * num_boards];
    assignments[aa_idx] = 0;
    assignments[trash_idx] = 1;

    // Pairwise equity.
    let boards: Vec<&[Card]> = vec![river_board];
    let pairwise = compute_pairwise_bucket_equity(&hands, &boards, &assignments, 3, num_boards, 1.0);

    // Old centroid-ratio equity (for comparison).
    // Fake avg equities: AA ~0.85, 72o ~0.10, others 0.5.
    let mut avg_equities = vec![0.5f64; hands.len()];
    avg_equities[aa_idx] = 0.85;
    avg_equities[trash_idx] = 0.10;
    let centroid = compute_bucket_pair_equity(&assignments, 3, &avg_equities);

    let pw_eq = pairwise.get(0, 1);
    let cr_eq = centroid.get(0, 1);

    // Pairwise should be higher (AA truly dominates 72o).
    assert!(
        pw_eq > cr_eq,
        "pairwise ({pw_eq}) should be > centroid-ratio ({cr_eq}) for AA vs 72o"
    );
}
