//! Integration tests for the card abstraction pipeline.
//!
//! These tests verify the full pipeline works end-to-end: creating abstractions
//! from boundaries, saving/loading, and bucket lookups.

use poker_solver_core::abstraction::{BucketBoundaries, CardAbstraction, Street};
use poker_solver_core::poker::{Card, Suit, Value};
use tempfile::tempdir;
use test_macros::timed_test;

/// Helper: build synthetic boundaries with uniform distribution
fn synthetic_boundaries(flop_buckets: usize, turn_buckets: usize, river_buckets: usize) -> BucketBoundaries {
    BucketBoundaries {
        flop: (1..flop_buckets).map(|i| i as f32 / flop_buckets as f32).collect(),
        turn: (1..turn_buckets).map(|i| i as f32 / turn_buckets as f32).collect(),
        river: (1..river_buckets).map(|i| i as f32 / river_buckets as f32).collect(),
    }
}

#[timed_test]
fn full_pipeline_generate_save_load() {
    let boundaries = synthetic_boundaries(10, 10, 20);
    let abstraction = CardAbstraction::from_boundaries(boundaries);

    let dir = tempdir().expect("Failed to create temp directory");
    let path = dir.path().join("test_boundaries.bin");
    abstraction.save(&path).expect("Failed to save boundaries");

    let loaded = CardAbstraction::load(&path).expect("Failed to load boundaries");
    assert_eq!(loaded.num_buckets(Street::Flop), 10);
    assert_eq!(loaded.num_buckets(Street::Turn), 10);
    assert_eq!(loaded.num_buckets(Street::River), 20);
}

#[timed_test]
fn bucket_lookup_returns_valid_index() {
    let boundaries = synthetic_boundaries(100, 100, 200);
    let abstraction = CardAbstraction::from_boundaries(boundaries);

    // Test river lookup (cheap — no flop EHS2)
    let board = vec![
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Spade),
        Card::new(Value::Queen, Suit::Heart),
        Card::new(Value::Jack, Suit::Diamond),
        Card::new(Value::Two, Suit::Club),
    ];
    let holding = (
        Card::new(Value::Ten, Suit::Spade),
        Card::new(Value::Nine, Suit::Spade),
    );

    let bucket = abstraction
        .get_bucket(&board, holding)
        .expect("Failed to get bucket");
    assert!(bucket < 200, "Bucket {bucket} should be < 200");
}

#[timed_test]
fn duplicate_card_rejected() {
    let boundaries = synthetic_boundaries(10, 10, 10);
    let abstraction = CardAbstraction::from_boundaries(boundaries);

    let board = vec![
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Spade),
        Card::new(Value::Queen, Suit::Heart),
        Card::new(Value::Jack, Suit::Diamond),
        Card::new(Value::Two, Suit::Club),
    ];
    // Holding has Ace of Spades which is on board
    let holding = (
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::Nine, Suit::Heart),
    );

    let result = abstraction.get_bucket(&board, holding);
    assert!(result.is_err(), "Should reject duplicate card");
}

#[timed_test]
fn different_hands_get_different_buckets_on_river() {
    // With enough buckets, strong and weak hands should be in different buckets
    let boundaries = BucketBoundaries {
        flop: (1..100).map(|i| i as f32 / 100.0).collect(),
        turn: (1..100).map(|i| i as f32 / 100.0).collect(),
        river: (1..100).map(|i| i as f32 / 100.0).collect(),
    };
    let abstraction = CardAbstraction::from_boundaries(boundaries);

    // Strong hand: Broadway straight
    let board1 = vec![
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Heart),
        Card::new(Value::Queen, Suit::Diamond),
        Card::new(Value::Jack, Suit::Club),
        Card::new(Value::Two, Suit::Spade),
    ];
    let strong_holding = (
        Card::new(Value::Ten, Suit::Heart),
        Card::new(Value::Nine, Suit::Heart),
    );

    // Weak hand: No pair, low cards
    let board2 = vec![
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Spade),
        Card::new(Value::Queen, Suit::Spade),
        Card::new(Value::Jack, Suit::Spade),
        Card::new(Value::Nine, Suit::Diamond),
    ];
    let weak_holding = (
        Card::new(Value::Two, Suit::Heart),
        Card::new(Value::Three, Suit::Club),
    );

    let strong_bucket = abstraction
        .get_bucket(&board1, strong_holding)
        .expect("Failed to get strong hand bucket");
    let weak_bucket = abstraction
        .get_bucket(&board2, weak_holding)
        .expect("Failed to get weak hand bucket");

    // Strong hand should be in higher bucket than weak hand
    assert!(
        strong_bucket > weak_bucket,
        "Strong hand bucket {strong_bucket} should be > weak hand bucket {weak_bucket}"
    );
}

#[timed_test(10)]
fn turn_bucket_lookup_works() {
    // Turn EHS2 is cheap (~990 combos), use synthetic boundaries
    let boundaries = synthetic_boundaries(50, 50, 100);
    let abstraction = CardAbstraction::from_boundaries(boundaries);

    // Test turn lookup (4 cards)
    let board = vec![
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Spade),
        Card::new(Value::Queen, Suit::Heart),
        Card::new(Value::Seven, Suit::Diamond),
    ];
    let holding = (
        Card::new(Value::Jack, Suit::Spade),
        Card::new(Value::Ten, Suit::Spade),
    );

    let bucket = abstraction
        .get_bucket(&board, holding)
        .expect("Failed to get turn bucket");
    assert!(bucket < 50, "Turn bucket {bucket} should be < 50");
}

#[timed_test]
fn invalid_board_size_rejected() {
    let boundaries = BucketBoundaries {
        flop: vec![0.5],
        turn: vec![0.5],
        river: vec![0.5],
    };
    let abstraction = CardAbstraction::from_boundaries(boundaries);

    // 2 cards is invalid (not flop, turn, or river)
    let board = vec![
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Spade),
    ];
    let holding = (
        Card::new(Value::Queen, Suit::Heart),
        Card::new(Value::Jack, Suit::Diamond),
    );

    let result = abstraction.get_bucket(&board, holding);
    assert!(result.is_err(), "Should reject invalid board size");
}

#[timed_test]
fn same_hand_same_bucket_deterministic() {
    let boundaries = BucketBoundaries {
        flop: (1..50).map(|i| i as f32 / 50.0).collect(),
        turn: (1..50).map(|i| i as f32 / 50.0).collect(),
        river: (1..100).map(|i| i as f32 / 100.0).collect(),
    };
    let abstraction = CardAbstraction::from_boundaries(boundaries);

    let board = vec![
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Heart),
        Card::new(Value::Queen, Suit::Diamond),
        Card::new(Value::Jack, Suit::Club),
        Card::new(Value::Two, Suit::Spade),
    ];
    let holding = (
        Card::new(Value::Ten, Suit::Heart),
        Card::new(Value::Nine, Suit::Heart),
    );

    // Same hand should always return same bucket
    let bucket1 = abstraction
        .get_bucket(&board, holding)
        .expect("Failed to get bucket");
    let bucket2 = abstraction
        .get_bucket(&board, holding)
        .expect("Failed to get bucket");
    let bucket3 = abstraction
        .get_bucket(&board, holding)
        .expect("Failed to get bucket");

    assert_eq!(bucket1, bucket2, "Bucket lookup should be deterministic");
    assert_eq!(bucket2, bucket3, "Bucket lookup should be deterministic");
}
