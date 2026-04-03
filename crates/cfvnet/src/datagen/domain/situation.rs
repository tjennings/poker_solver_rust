use crate::config::DatagenConfig;
use crate::datagen::sampler::{sample_situation, Situation};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Produces random Situations from config. Implements Iterator.
/// Skips degenerate situations (effective_stack <= 0) internally.
pub struct SituationGenerator {
    config: DatagenConfig,
    initial_stack: i32,
    board_size: usize,
    rng: ChaCha8Rng,
    remaining: u64,
}

impl SituationGenerator {
    pub fn new(
        config: &DatagenConfig,
        initial_stack: i32,
        board_size: usize,
        seed: u64,
        count: u64,
    ) -> Self {
        Self {
            config: config.clone(),
            initial_stack,
            board_size,
            rng: ChaCha8Rng::seed_from_u64(seed),
            remaining: count,
        }
    }
}

impl Iterator for SituationGenerator {
    type Item = Situation;

    fn next(&mut self) -> Option<Situation> {
        while self.remaining > 0 {
            self.remaining -= 1;
            let sit =
                sample_situation(&self.config, self.initial_stack, self.board_size, &mut self.rng);
            if sit.effective_stack > 0 {
                return Some(sit);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DatagenConfig;

    #[test]
    fn generator_produces_valid_situations() {
        let config = DatagenConfig::default();
        let sit_gen = SituationGenerator::new(&config, 200, 4, 42, 10);
        let situations: Vec<_> = sit_gen.collect();
        assert!(!situations.is_empty());
        for sit in &situations {
            assert!(sit.effective_stack > 0);
            assert_eq!(sit.board_size, 4);
        }
    }

    #[test]
    fn generator_respects_count() {
        let config = DatagenConfig::default();
        let sit_gen = SituationGenerator::new(&config, 200, 4, 42, 3);
        let count = sit_gen.count();
        assert!(count <= 3);
    }

    #[test]
    fn generator_skips_degenerate_situations() {
        // With a very high pot interval, effective_stack may go to 0 or negative.
        // The generator should skip those.
        let config = DatagenConfig {
            pot_intervals: vec![[380, 400]],
            ..DatagenConfig::default()
        };
        let sit_gen = SituationGenerator::new(&config, 200, 4, 42, 20);
        for sit in sit_gen {
            assert!(sit.effective_stack > 0);
        }
    }

    #[test]
    fn generator_deterministic_with_same_seed() {
        let config = DatagenConfig::default();
        let sit_gen1 = SituationGenerator::new(&config, 200, 4, 99, 5);
        let sit_gen2 = SituationGenerator::new(&config, 200, 4, 99, 5);
        let sits1: Vec<_> = sit_gen1.collect();
        let sits2: Vec<_> = sit_gen2.collect();
        assert_eq!(sits1.len(), sits2.len());
        for (a, b) in sits1.iter().zip(sits2.iter()) {
            assert_eq!(a.pot, b.pot);
            assert_eq!(a.effective_stack, b.effective_stack);
            assert_eq!(a.board, b.board);
        }
    }

    #[test]
    fn generator_returns_none_when_exhausted() {
        let config = DatagenConfig::default();
        let mut sit_gen = SituationGenerator::new(&config, 200, 4, 42, 1);
        let _first = sit_gen.next(); // consume the one item (or skip degenerate)
        // After count is exhausted, should always return None
        // (may return None on first call too if degenerate)
        for _ in 0..5 {
            assert!(sit_gen.next().is_none());
        }
    }
}
