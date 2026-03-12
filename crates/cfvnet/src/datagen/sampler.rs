use rand::Rng;

use crate::config::DatagenConfig;
use super::range_gen::{generate_rsp_range, NUM_COMBOS};

/// A single training situation before solving.
#[derive(Debug, Clone)]
pub struct Situation {
    pub board: Vec<u8>,
    pub pot: i32,
    pub effective_stack: i32,
    pub ranges: [[f32; NUM_COMBOS]; 2], // [OOP, IP]
}

/// Sample a random situation: board, pot, effective stack, and two R(S,p) ranges.
///
/// `board_size` must be 4 (turn) or 5 (river).
pub fn sample_situation<R: Rng>(
    config: &DatagenConfig,
    initial_stack: i32,
    board_size: usize,
    rng: &mut R,
) -> Situation {
    let board = sample_board(board_size, rng);
    let pot = sample_pot(&config.pot_intervals, rng);
    let max_stack = initial_stack - pot / 2;
    let effective_stack = if max_stack < 5 {
        5
    } else {
        rng.gen_range(5..=max_stack)
    };
    let oop_range = generate_rsp_range(&board, rng);
    let ip_range = generate_rsp_range(&board, rng);
    Situation {
        board,
        pot,
        effective_stack,
        ranges: [oop_range, ip_range],
    }
}

/// Sample `num_cards` unique cards from 0..52 for the board.
///
/// # Panics
///
/// Panics if `num_cards` is greater than 52.
pub fn sample_board<R: Rng>(num_cards: usize, rng: &mut R) -> Vec<u8> {
    assert!(num_cards <= 52, "cannot sample {num_cards} cards from a 52-card deck");
    let mut board = Vec::with_capacity(num_cards);
    let mut used = [false; 52];
    for _ in 0..num_cards {
        loop {
            let c: u8 = rng.gen_range(0..52);
            if !used[c as usize] {
                used[c as usize] = true;
                board.push(c);
                break;
            }
        }
    }
    board
}

/// Sample a pot from stratified intervals: pick a uniform random interval, then
/// a uniform random value within that interval.
fn sample_pot<R: Rng>(intervals: &[[i32; 2]], rng: &mut R) -> i32 {
    let idx = rng.gen_range(0..intervals.len());
    let [lo, hi] = intervals[idx];
    rng.gen_range(lo..hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn test_config() -> DatagenConfig {
        DatagenConfig {
            num_samples: 100,
            street: "river".into(),
            pot_intervals: vec![[4, 20], [20, 80], [80, 200], [200, 400]],
            solver_iterations: 100,
            target_exploitability: 0.01,
            threads: 1,
            seed: 42,
        }
    }

    const INITIAL_STACK: i32 = 200;

    #[test]
    fn board_has_five_unique_cards() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = test_config();
        for _ in 0..100 {
            let sit = sample_situation(&config, INITIAL_STACK, 5, &mut rng);
            let board = &sit.board;
            assert_eq!(board.len(), 5);
            for i in 0..5 {
                for j in (i + 1)..5 {
                    assert_ne!(
                        board[i], board[j],
                        "duplicate card: {} at positions {i} and {j}",
                        board[i]
                    );
                }
                assert!(board[i] < 52, "card {} out of range", board[i]);
            }
        }
    }

    #[test]
    fn pot_within_configured_intervals() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = test_config();
        for _ in 0..200 {
            let sit = sample_situation(&config, INITIAL_STACK, 5, &mut rng);
            let in_some_interval = config
                .pot_intervals
                .iter()
                .any(|[lo, hi]| sit.pot >= *lo && sit.pot < *hi);
            assert!(
                in_some_interval,
                "pot {} not in any interval {:?}",
                sit.pot, config.pot_intervals
            );
        }
    }

    #[test]
    fn stack_within_valid_range() {
        let config = test_config();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..200 {
            let sit = sample_situation(&config, INITIAL_STACK, 5, &mut rng);
            let max_stack = INITIAL_STACK - sit.pot / 2;
            assert!(
                sit.effective_stack >= 5,
                "stack must be >= 5, got: {}",
                sit.effective_stack
            );
            if max_stack >= 5 {
                assert!(
                    sit.effective_stack <= max_stack,
                    "stack {} exceeds max {} for pot {}",
                    sit.effective_stack,
                    max_stack,
                    sit.pot
                );
            }
        }
    }

    #[test]
    fn ranges_are_valid() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = test_config();
        for _ in 0..50 {
            let sit = sample_situation(&config, INITIAL_STACK, 5, &mut rng);
            for player in 0..2 {
                let range = &sit.ranges[player];
                let sum: f32 = range.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-4,
                    "player {player} range sums to {sum}"
                );
                for i in 0..1326 {
                    let (c1, c2) = range_solver::card::index_to_card_pair(i);
                    if sit.board.contains(&c1) || sit.board.contains(&c2) {
                        assert_eq!(
                            range[i], 0.0,
                            "player {player} combo {i} conflicts with board"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let config = test_config();
        let mut rng1 = ChaCha8Rng::seed_from_u64(99);
        let mut rng2 = ChaCha8Rng::seed_from_u64(99);
        let s1 = sample_situation(&config, INITIAL_STACK, 5, &mut rng1);
        let s2 = sample_situation(&config, INITIAL_STACK, 5, &mut rng2);
        assert_eq!(s1.board, s2.board);
        assert_eq!(s1.pot, s2.pot);
        assert_eq!(s1.effective_stack, s2.effective_stack);
        assert_eq!(s1.ranges[0], s2.ranges[0]);
        assert_eq!(s1.ranges[1], s2.ranges[1]);
    }

    #[test]
    fn sample_board_4_cards() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..100 {
            let board = sample_board(4, &mut rng);
            assert_eq!(board.len(), 4);
            for i in 0..4 {
                assert!(board[i] < 52);
                for j in (i + 1)..4 {
                    assert_ne!(board[i], board[j], "duplicate cards in 4-card board");
                }
            }
        }
    }

    #[test]
    fn sample_board_5_cards() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..100 {
            let board = sample_board(5, &mut rng);
            assert_eq!(board.len(), 5);
            for i in 0..5 {
                assert!(board[i] < 52);
                for j in (i + 1)..5 {
                    assert_ne!(board[i], board[j], "duplicate cards in 5-card board");
                }
            }
        }
    }
}
