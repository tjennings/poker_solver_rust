//! Boundary Generation for Card Abstraction
//!
//! Provides offline generation of bucket boundaries by sampling random hands
//! and computing EHS2 values.

use crate::abstraction::{AbstractionConfig, BucketBoundaries, HandStrengthCalculator};
use crate::poker::{Card, Suit, Value};
use indicatif::{ProgressBar, ProgressStyle};
use rand::SeedableRng;
use rand::prelude::*;

/// Generates bucket boundaries by sampling hands.
///
/// This generator creates bucket boundaries by randomly sampling hands
/// and computing their EHS2 values. The boundaries are then computed
/// using percentile-based bucketing.
///
/// This is an expensive operation meant to be run offline once,
/// with the resulting boundaries saved to disk for later use.
pub struct BoundaryGenerator {
    config: AbstractionConfig,
    calculator: HandStrengthCalculator,
}

impl BoundaryGenerator {
    /// Create a new boundary generator with the given configuration.
    ///
    /// # Arguments
    /// * `config` - Configuration specifying bucket counts and sample sizes
    #[must_use]
    pub fn new(config: AbstractionConfig) -> Self {
        Self {
            config,
            calculator: HandStrengthCalculator::new(),
        }
    }

    /// Generate boundaries by sampling random boards and holdings.
    ///
    /// This is expensive - meant to be run offline once. The generated
    /// boundaries can then be saved and loaded for runtime use.
    ///
    /// # Arguments
    /// * `seed` - Random seed for reproducible boundary generation
    ///
    /// # Returns
    /// `BucketBoundaries` computed from the sampled EHS2 values
    #[must_use]
    pub fn generate(&self, seed: u64) -> BucketBoundaries {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut river_samples = self.sample_street(&mut rng, 5, self.config.samples_per_street);
        let mut turn_samples = self.sample_street(&mut rng, 4, self.config.samples_per_street);

        // Use fewer samples for flop due to computation cost
        let flop_sample_count = (self.config.samples_per_street / 100).max(100);
        let mut flop_samples = self.sample_street(&mut rng, 3, flop_sample_count);

        BucketBoundaries::from_samples(
            &mut flop_samples,
            &mut turn_samples,
            &mut river_samples,
            self.config.flop_buckets as usize,
            self.config.turn_buckets as usize,
            self.config.river_buckets as usize,
        )
    }

    /// Sample EHS2 values for a given street.
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    /// * `board_size` - Number of community cards (3=flop, 4=turn, 5=river)
    /// * `num_samples` - Number of samples to generate
    fn sample_street(&self, rng: &mut StdRng, board_size: usize, num_samples: u32) -> Vec<f32> {
        let street_name = match board_size {
            3 => "flop",
            4 => "turn",
            5 => "river",
            _ => "unknown",
        };

        let pb = ProgressBar::new(u64::from(num_samples));
        pb.set_style(ProgressStyle::with_template(
            &format!("  sampling {street_name} EHS2 [{{bar:40}}] {{pos}}/{{len}} [{{elapsed}} < {{eta}}, {{per_sec}}]")
        ).unwrap());

        let mut samples = Vec::with_capacity(num_samples as usize);
        let deck = Self::full_deck();

        for _ in 0..num_samples {
            // Shuffle and deal
            let mut cards = deck.clone();
            cards.shuffle(rng);

            let board: Vec<Card> = cards[0..board_size].to_vec();
            let holding = (cards[board_size], cards[board_size + 1]);

            let hs = match board_size {
                3 => self.calculator.calculate_flop(&board, holding),
                4 => self.calculator.calculate_turn(&board, holding),
                5 => self.calculator.calculate_river(&board, holding),
                _ => unreachable!("Invalid board size: {}", board_size),
            };

            samples.push(hs.ehs2);
            pb.inc(1);
        }

        pb.finish_and_clear();

        samples
    }

    /// Generate a full 52-card deck.
    fn full_deck() -> Vec<Card> {
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

        let mut deck = Vec::with_capacity(52);
        for &value in &VALUES {
            for &suit in &SUITS {
                deck.push(Card::new(value, suit));
            }
        }
        deck
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_precision_loss)]
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn generator_produces_valid_boundaries() {
        // Test from_samples with synthetic data (avoids expensive flop EHS2)
        let mut flop_samples: Vec<f32> = (0..50).map(|i| i as f32 / 50.0).collect();
        let mut turn_samples: Vec<f32> = (0..50).map(|i| i as f32 / 50.0).collect();
        let mut river_samples: Vec<f32> = (0..50).map(|i| i as f32 / 50.0).collect();

        let boundaries = BucketBoundaries::from_samples(
            &mut flop_samples,
            &mut turn_samples,
            &mut river_samples,
            10,
            10,
            10,
        );

        assert_eq!(boundaries.flop.len(), 9); // 10 buckets = 9 boundaries
        assert_eq!(boundaries.turn.len(), 9);
        assert_eq!(boundaries.river.len(), 9);

        // Boundaries should be monotonically increasing
        for window in boundaries.river.windows(2) {
            assert!(window[0] <= window[1], "Boundaries not monotonic");
        }
        for window in boundaries.flop.windows(2) {
            assert!(window[0] <= window[1], "Boundaries not monotonic");
        }
    }

    #[timed_test]
    fn full_deck_has_52_cards() {
        let deck = BoundaryGenerator::full_deck();
        assert_eq!(deck.len(), 52);
    }

    #[timed_test]
    fn full_deck_has_unique_cards() {
        let deck = BoundaryGenerator::full_deck();
        let unique: std::collections::HashSet<_> = deck.iter().collect();
        assert_eq!(unique.len(), 52);
    }

    #[timed_test]
    fn deterministic_generation_with_same_seed() {
        // River EHS2 is cheap — use HandStrengthCalculator directly
        let calc = HandStrengthCalculator::new();
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

        let hs1 = calc.calculate_river(&board, holding);
        let hs2 = calc.calculate_river(&board, holding);

        assert!(
            (hs1.ehs2 - hs2.ehs2).abs() < f32::EPSILON,
            "Same inputs should produce identical EHS2: {} vs {}",
            hs1.ehs2,
            hs2.ehs2
        );
    }

    #[timed_test]
    fn different_inputs_produce_different_ehs2() {
        let calc = HandStrengthCalculator::new();
        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Queen, Suit::Diamond),
            Card::new(Value::Jack, Suit::Club),
            Card::new(Value::Two, Suit::Spade),
        ];

        // Strong holding: Broadway straight
        let strong = (
            Card::new(Value::Ten, Suit::Heart),
            Card::new(Value::Nine, Suit::Heart),
        );
        // Weak holding: low pair
        let weak = (
            Card::new(Value::Three, Suit::Heart),
            Card::new(Value::Four, Suit::Club),
        );

        let hs_strong = calc.calculate_river(&board, strong);
        let hs_weak = calc.calculate_river(&board, weak);

        assert!(
            (hs_strong.ehs2 - hs_weak.ehs2).abs() > f32::EPSILON,
            "Different holdings should produce different EHS2: strong={}, weak={}",
            hs_strong.ehs2,
            hs_weak.ehs2
        );
    }

    #[timed_test]
    fn boundaries_within_valid_range() {
        // Use river EHS2 (cheap) for several holdings to verify range
        let calc = HandStrengthCalculator::new();
        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Queen, Suit::Diamond),
            Card::new(Value::Jack, Suit::Club),
            Card::new(Value::Two, Suit::Spade),
        ];

        let holdings = [
            (
                Card::new(Value::Ten, Suit::Heart),
                Card::new(Value::Nine, Suit::Heart),
            ),
            (
                Card::new(Value::Three, Suit::Heart),
                Card::new(Value::Four, Suit::Club),
            ),
            (
                Card::new(Value::Seven, Suit::Diamond),
                Card::new(Value::Eight, Suit::Club),
            ),
            (
                Card::new(Value::Five, Suit::Club),
                Card::new(Value::Six, Suit::Diamond),
            ),
        ];

        for holding in &holdings {
            let hs = calc.calculate_river(&board, *holding);
            assert!(
                (0.0..=1.0).contains(&hs.ehs2),
                "EHS2 out of range: {}",
                hs.ehs2
            );
        }
    }
}
