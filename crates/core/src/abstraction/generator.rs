//! Boundary Generation for Card Abstraction
//!
//! Provides offline generation of bucket boundaries by sampling random hands
//! and computing EHS2 values.

use crate::abstraction::{AbstractionConfig, BucketBoundaries, HandStrengthCalculator};
use crate::poker::{Card, Suit, Value};
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

        eprintln!("Generating river samples...");
        let mut river_samples = self.sample_street(&mut rng, 5, self.config.samples_per_street);

        eprintln!("Generating turn samples...");
        let mut turn_samples = self.sample_street(&mut rng, 4, self.config.samples_per_street);

        eprintln!("Generating flop samples (slow)...");
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
        let mut samples = Vec::with_capacity(num_samples as usize);
        let deck = Self::full_deck();

        for i in 0..num_samples {
            if i % 1000 == 0 && i > 0 {
                eprintln!("  Sampled {i}/{num_samples}");
            }

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
        }

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
    use super::*;

    #[test]
    fn generator_produces_valid_boundaries() {
        let config = AbstractionConfig {
            flop_buckets: 10,
            turn_buckets: 10,
            river_buckets: 10,
            samples_per_street: 100, // Small for testing
        };
        let generator = BoundaryGenerator::new(config);
        let boundaries = generator.generate(42);

        assert_eq!(boundaries.flop.len(), 9); // 10 buckets = 9 boundaries
        assert_eq!(boundaries.turn.len(), 9);
        assert_eq!(boundaries.river.len(), 9);

        // Boundaries should be monotonically increasing
        for window in boundaries.river.windows(2) {
            assert!(window[0] <= window[1], "Boundaries not monotonic");
        }
    }

    #[test]
    fn full_deck_has_52_cards() {
        let deck = BoundaryGenerator::full_deck();
        assert_eq!(deck.len(), 52);
    }

    #[test]
    fn full_deck_has_unique_cards() {
        let deck = BoundaryGenerator::full_deck();
        let unique: std::collections::HashSet<_> = deck.iter().collect();
        assert_eq!(unique.len(), 52);
    }

    #[test]
    fn deterministic_generation_with_same_seed() {
        let config = AbstractionConfig {
            flop_buckets: 5,
            turn_buckets: 5,
            river_buckets: 5,
            samples_per_street: 50,
        };
        let generator = BoundaryGenerator::new(config);

        let boundaries1 = generator.generate(12345);
        let boundaries2 = generator.generate(12345);

        assert_eq!(boundaries1.flop, boundaries2.flop);
        assert_eq!(boundaries1.turn, boundaries2.turn);
        assert_eq!(boundaries1.river, boundaries2.river);
    }

    #[test]
    fn different_seeds_produce_different_boundaries() {
        let config = AbstractionConfig {
            flop_buckets: 5,
            turn_buckets: 5,
            river_buckets: 5,
            samples_per_street: 50,
        };
        let generator = BoundaryGenerator::new(config);

        let boundaries1 = generator.generate(11111);
        let boundaries2 = generator.generate(22222);

        // At least one street's boundaries should differ
        let all_same = boundaries1.flop == boundaries2.flop
            && boundaries1.turn == boundaries2.turn
            && boundaries1.river == boundaries2.river;
        assert!(
            !all_same,
            "Different seeds should produce different boundaries"
        );
    }

    #[test]
    fn boundaries_within_valid_range() {
        let config = AbstractionConfig {
            flop_buckets: 5,
            turn_buckets: 5,
            river_buckets: 5,
            samples_per_street: 100,
        };
        let generator = BoundaryGenerator::new(config);
        let boundaries = generator.generate(99);

        // All boundaries should be between 0 and 1 (EHS2 range)
        for &b in &boundaries.flop {
            assert!(b >= 0.0 && b <= 1.0, "Flop boundary out of range: {}", b);
        }
        for &b in &boundaries.turn {
            assert!(b >= 0.0 && b <= 1.0, "Turn boundary out of range: {}", b);
        }
        for &b in &boundaries.river {
            assert!(b >= 0.0 && b <= 1.0, "River boundary out of range: {}", b);
        }
    }
}
