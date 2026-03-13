use rand::Rng;

use crate::config::DatagenConfig;
use super::range_gen::{compute_hand_strengths, generate_rsp_range_with_strengths, NUM_COMBOS};

/// A single training situation before solving.
#[derive(Debug, Clone)]
pub struct Situation {
    /// Board cards as fixed-size array. Only the first `board_size` elements are valid.
    pub board: [u8; 5],
    /// Number of valid board cards (4 for turn, 5 for river).
    pub board_size: usize,
    pub pot: i32,
    pub effective_stack: i32,
    pub ranges: [[f32; NUM_COMBOS]; 2], // [OOP, IP]
}

impl Situation {
    /// Return a slice of the valid board cards.
    #[inline]
    pub fn board_cards(&self) -> &[u8] {
        &self.board[..self.board_size]
    }
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
    let pot = sample_pot(config.pot_range, rng);
    let max_stack = initial_stack - pot / 2;
    let effective_stack = if max_stack < 5 {
        5
    } else {
        rng.gen_range(5..=max_stack)
    };
    let strengths = compute_hand_strengths(&board[..board_size]);
    let oop_range = generate_rsp_range_with_strengths(&strengths, rng);
    let ip_range = generate_rsp_range_with_strengths(&strengths, rng);
    Situation {
        board,
        board_size,
        pot,
        effective_stack,
        ranges: [oop_range, ip_range],
    }
}

/// Sample `num_cards` unique cards from 0..52 for the board.
///
/// Returns a `[u8; 5]` with the first `num_cards` slots filled and the rest zeroed.
///
/// # Panics
///
/// Panics if `num_cards` is greater than 5 or 52.
pub fn sample_board<R: Rng>(num_cards: usize, rng: &mut R) -> [u8; 5] {
    assert!(num_cards <= 5, "board cannot have more than 5 cards, got {num_cards}");
    let mut board = [0u8; 5];
    let mut used = [false; 52];
    for slot in board.iter_mut().take(num_cards) {
        loop {
            let c: u8 = rng.gen_range(0..52);
            if !used[c as usize] {
                used[c as usize] = true;
                *slot = c;
                break;
            }
        }
    }
    board
}

/// Sample a pot using log-uniform distribution over the given range.
///
/// This gives equal probability density per multiplicative factor, so small
/// pots (e.g. 4-20) get similar representation as large pots (e.g. 100-400).
fn sample_pot<R: Rng>(pot_range: [i32; 2], rng: &mut R) -> i32 {
    let [lo, hi] = pot_range;
    let log_lo = (lo as f64).ln();
    let log_hi = (hi as f64).ln();
    let log_pot = rng.gen_range(log_lo..log_hi);
    log_pot.exp().round() as i32
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
            pot_range: [4, 400],
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
            assert_eq!(sit.board_size, 5);
            let board = sit.board_cards();
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
    fn pot_within_configured_range() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = test_config();
        let [lo, hi] = config.pot_range;
        for _ in 0..200 {
            let sit = sample_situation(&config, INITIAL_STACK, 5, &mut rng);
            assert!(
                sit.pot >= lo && sit.pot <= hi,
                "pot {} not in range [{}, {}]",
                sit.pot, lo, hi
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
                    if sit.board_cards().contains(&c1) || sit.board_cards().contains(&c2) {
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
            // First 4 slots filled, 5th is padding (zero).
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
            for i in 0..5 {
                assert!(board[i] < 52);
                for j in (i + 1)..5 {
                    assert_ne!(board[i], board[j], "duplicate cards in 5-card board");
                }
            }
        }
    }
}
