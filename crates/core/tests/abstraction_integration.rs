//! Integration tests for the card abstraction pipeline.
//!
//! These tests verify the full pipeline works end-to-end: generating boundaries,
//! creating abstractions, saving/loading, and bucket lookups.

use poker_solver_core::abstraction::{
    AbstractionConfig, BoundaryGenerator, BucketBoundaries, CardAbstraction, Street,
};
use poker_solver_core::poker::{Card, Suit, Value};
use tempfile::tempdir;

#[test]
fn full_pipeline_generate_save_load() {
    // Generate small boundaries
    let config = AbstractionConfig {
        flop_buckets: 10,
        turn_buckets: 10,
        river_buckets: 20,
        samples_per_street: 50,
    };
    let generator = BoundaryGenerator::new(config);
    let boundaries = generator.generate(12345);

    // Create abstraction and save
    let abstraction = CardAbstraction::from_boundaries(boundaries);

    let dir = tempdir().expect("Failed to create temp directory");
    let path = dir.path().join("test_boundaries.bin");
    abstraction.save(&path).expect("Failed to save boundaries");

    // Load and verify
    let loaded = CardAbstraction::load(&path).expect("Failed to load boundaries");
    assert_eq!(loaded.num_buckets(Street::Flop), 10);
    assert_eq!(loaded.num_buckets(Street::Turn), 10);
    assert_eq!(loaded.num_buckets(Street::River), 20);
}

#[test]
fn bucket_lookup_returns_valid_index() {
    let config = AbstractionConfig {
        flop_buckets: 100,
        turn_buckets: 100,
        river_buckets: 200,
        samples_per_street: 100,
    };
    let generator = BoundaryGenerator::new(config);
    let boundaries = generator.generate(42);
    let abstraction = CardAbstraction::from_boundaries(boundaries);

    // Test river lookup
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
    assert!(bucket < 200, "Bucket {} should be < 200", bucket);
}

#[test]
fn duplicate_card_rejected() {
    let config = AbstractionConfig {
        flop_buckets: 10,
        turn_buckets: 10,
        river_buckets: 10,
        samples_per_street: 10,
    };
    let generator = BoundaryGenerator::new(config);
    let boundaries = generator.generate(1);
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

#[test]
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
        "Strong hand bucket {} should be > weak hand bucket {}",
        strong_bucket,
        weak_bucket
    );
}

#[test]
fn flop_bucket_lookup_works() {
    let config = AbstractionConfig {
        flop_buckets: 50,
        turn_buckets: 50,
        river_buckets: 100,
        samples_per_street: 100,
    };
    let generator = BoundaryGenerator::new(config);
    let boundaries = generator.generate(99);
    let abstraction = CardAbstraction::from_boundaries(boundaries);

    // Test flop lookup (3 cards)
    let board = vec![
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Spade),
        Card::new(Value::Queen, Suit::Heart),
    ];
    let holding = (
        Card::new(Value::Jack, Suit::Spade),
        Card::new(Value::Ten, Suit::Spade),
    );

    let bucket = abstraction
        .get_bucket(&board, holding)
        .expect("Failed to get flop bucket");
    assert!(bucket < 50, "Flop bucket {} should be < 50", bucket);
}

#[test]
fn turn_bucket_lookup_works() {
    let config = AbstractionConfig {
        flop_buckets: 50,
        turn_buckets: 50,
        river_buckets: 100,
        samples_per_street: 100,
    };
    let generator = BoundaryGenerator::new(config);
    let boundaries = generator.generate(77);
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
    assert!(bucket < 50, "Turn bucket {} should be < 50", bucket);
}

#[test]
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

#[test]
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
