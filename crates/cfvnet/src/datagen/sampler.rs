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
    let (pot, effective_stack) = if let Some(ref spr_intervals) = config.spr_intervals {
        sample_pot_stack_by_spr(&config.pot_intervals, spr_intervals, initial_stack, rng)
    } else {
        let pot = sample_pot(&config.pot_intervals, rng);
        let max_stack = initial_stack - pot / 2;
        let effective_stack = if max_stack < 5 {
            5
        } else {
            rng.gen_range(5..=max_stack)
        };
        (pot, effective_stack)
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

/// Sample a pot from stratified intervals: pick a uniform random interval, then
/// a uniform random value within that interval.
fn sample_pot<R: Rng>(intervals: &[[i32; 2]], rng: &mut R) -> i32 {
    let idx = rng.gen_range(0..intervals.len());
    let [lo, hi] = intervals[idx];
    rng.gen_range(lo..hi)
}

/// Sample a (pot, stack) pair via hybrid stratified rejection sampling.
///
/// Picks an SPR bucket uniformly (guaranteeing uniform SPR marginals), then
/// picks a pot bucket uniformly from those feasible for the target SPR.
/// This produces distributions that are uniform on SPR and approximately
/// uniform on pot size.
fn sample_pot_stack_by_spr<R: Rng>(
    pot_intervals: &[[i32; 2]],
    spr_intervals: &[[f64; 2]],
    initial_stack: i32,
    rng: &mut R,
) -> (i32, i32) {
    let stack_f = initial_stack as f64;
    for _ in 0..10_000 {
        // 1. Pick SPR bucket uniformly
        let spr_idx = rng.gen_range(0..spr_intervals.len());
        let [spr_lo, spr_hi] = spr_intervals[spr_idx];
        let target_spr = rng.gen_range(spr_lo..spr_hi);

        // 2. Compute feasible pot range for this SPR
        let pot_min_f = if target_spr > 0.0 {
            (5.0 / target_spr).ceil()
        } else {
            (2.0 * (stack_f - 5.0)).ceil()
        };
        let pot_max_f = stack_f / (target_spr + 0.5);

        let feasible_lo = pot_min_f.max(1.0) as i32;
        let feasible_hi = pot_max_f as i32;
        if feasible_lo > feasible_hi {
            continue;
        }

        // 3. Collect feasible pot buckets (those with non-empty intersection)
        let mut feasible_buckets: Vec<[i32; 2]> = Vec::new();
        for &[ilo, ihi] in pot_intervals {
            let lo = ilo.max(feasible_lo);
            let hi = ihi.min(feasible_hi + 1);
            if lo < hi {
                feasible_buckets.push([lo, hi]);
            }
        }
        if feasible_buckets.is_empty() {
            continue;
        }

        // 4. Pick a feasible pot bucket uniformly, then sample within it
        let bucket = feasible_buckets[rng.gen_range(0..feasible_buckets.len())];
        let pot = rng.gen_range(bucket[0]..bucket[1]);

        // 5. Derive stack from SPR
        let max_stack = initial_stack - pot / 2;
        if max_stack < 5 {
            continue;
        }
        let target_stack = (target_spr * pot as f64).round() as i32;
        let stack = target_stack.clamp(5, max_stack);

        // 6. Verify actual SPR lands in chosen bucket
        let actual_spr = stack as f64 / pot as f64;
        if actual_spr >= spr_lo && actual_spr < spr_hi {
            return (pot, stack);
        }
    }
    panic!(
        "sample_pot_stack_by_spr: exceeded 10000 attempts; \
         check that pot_intervals and spr_intervals are feasible for initial_stack={initial_stack}"
    );
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
            mode: "model".into(),
            pot_intervals: vec![[4, 20], [20, 80], [80, 200], [200, 400]],
            spr_intervals: None,
            solver_iterations: 100,
            target_exploitability: 0.01,
            threads: 1,
            seed: Some(42),
            leaf_eval_interval: 0,
            bet_size_fuzz: 0.0,
            river_output: None,
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

    fn test_spr_intervals() -> Vec<[f64; 2]> {
        vec![[0.0, 0.5], [0.5, 1.5], [1.5, 4.0], [4.0, 8.0], [8.0, 50.0]]
    }

    fn test_pot_intervals() -> Vec<[i32; 2]> {
        vec![[4, 20], [20, 80], [80, 200]]
    }

    #[test]
    fn spr_stratified_covers_all_buckets() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let pot_intervals = test_pot_intervals();
        let spr_intervals = test_spr_intervals();
        let mut bucket_hits = vec![0u32; spr_intervals.len()];

        for _ in 0..1000 {
            let (pot, stack) = sample_pot_stack_by_spr(
                &pot_intervals, &spr_intervals, INITIAL_STACK, &mut rng,
            );
            let actual_spr = stack as f64 / pot as f64;
            for (i, [lo, hi]) in spr_intervals.iter().enumerate() {
                if actual_spr >= *lo && actual_spr < *hi {
                    bucket_hits[i] += 1;
                    break;
                }
            }
        }

        for (i, hits) in bucket_hits.iter().enumerate() {
            assert!(
                *hits > 0,
                "SPR bucket {} ({:?}) got zero hits",
                i, spr_intervals[i]
            );
        }
    }

    #[test]
    fn spr_stratified_respects_bounds() {
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let pot_intervals = test_pot_intervals();
        let spr_intervals = test_spr_intervals();

        for _ in 0..500 {
            let (pot, stack) = sample_pot_stack_by_spr(
                &pot_intervals, &spr_intervals, INITIAL_STACK, &mut rng,
            );
            let max_stack = INITIAL_STACK - pot / 2;

            assert!(stack >= 5, "stack {} < 5", stack);
            assert!(
                stack <= max_stack,
                "stack {} > max_stack {} for pot {}",
                stack, max_stack, pot
            );

            let in_some_pot_interval = pot_intervals
                .iter()
                .any(|[lo, hi]| pot >= *lo && pot < *hi);
            assert!(
                in_some_pot_interval,
                "pot {} not in any interval {:?}",
                pot, pot_intervals
            );
        }
    }

    #[test]
    fn spr_stratified_actual_spr_in_bucket() {
        let mut rng = ChaCha8Rng::seed_from_u64(77);
        let pot_intervals = test_pot_intervals();
        let spr_intervals = test_spr_intervals();

        for _ in 0..500 {
            let (pot, stack) = sample_pot_stack_by_spr(
                &pot_intervals, &spr_intervals, INITIAL_STACK, &mut rng,
            );
            let actual_spr = stack as f64 / pot as f64;
            let in_some_spr_bucket = spr_intervals
                .iter()
                .any(|[lo, hi]| actual_spr >= *lo && actual_spr < *hi);
            assert!(
                in_some_spr_bucket,
                "actual SPR {} not in any bucket {:?} (pot={}, stack={})",
                actual_spr, spr_intervals, pot, stack
            );
        }
    }

    #[test]
    fn backward_compatible_without_spr() {
        let config = test_config();
        assert!(config.spr_intervals.is_none());
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        for _ in 0..100 {
            let sit = sample_situation(&config, INITIAL_STACK, 5, &mut rng);
            let in_some_interval = config
                .pot_intervals
                .iter()
                .any(|[lo, hi]| sit.pot >= *lo && sit.pot < *hi);
            assert!(in_some_interval, "pot {} not in any interval", sit.pot);
            assert!(sit.effective_stack >= 5);
        }
    }

    #[test]
    fn sample_situation_uses_spr_when_configured() {
        let spr_intervals = vec![[0.0, 0.5], [0.5, 1.5], [1.5, 4.0]];
        let config = DatagenConfig {
            spr_intervals: Some(spr_intervals.clone()),
            ..test_config()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        for _ in 0..200 {
            let sit = sample_situation(&config, INITIAL_STACK, 5, &mut rng);
            let actual_spr = sit.effective_stack as f64 / sit.pot as f64;
            let in_some_spr_bucket = spr_intervals
                .iter()
                .any(|[lo, hi]| actual_spr >= *lo && actual_spr < *hi);
            assert!(
                in_some_spr_bucket,
                "actual SPR {} not in any bucket when spr_intervals configured",
                actual_spr
            );
        }
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

    #[test]
    fn spr_and_pot_buckets_both_covered() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let pot_intervals = test_pot_intervals();
        let spr_intervals = test_spr_intervals();
        let num_spr = spr_intervals.len();
        let num_pot = pot_intervals.len();
        // 2D grid: cell[spr_idx][pot_idx]
        let mut cell_hits = vec![vec![0u32; num_pot]; num_spr];

        for _ in 0..5000 {
            let (pot, stack) = sample_pot_stack_by_spr(
                &pot_intervals, &spr_intervals, INITIAL_STACK, &mut rng,
            );
            let actual_spr = stack as f64 / pot as f64;
            let spr_idx = spr_intervals.iter().position(|[lo, hi]| actual_spr >= *lo && actual_spr < *hi);
            let pot_idx = pot_intervals.iter().position(|[lo, hi]| pot >= *lo && pot < *hi);
            if let (Some(si), Some(pi)) = (spr_idx, pot_idx) {
                cell_hits[si][pi] += 1;
            }
        }

        // Check pot bucket marginals are roughly uniform (no bucket < 25% of max)
        let mut pot_marginals = vec![0u32; num_pot];
        for si in 0..num_spr {
            for pi in 0..num_pot {
                pot_marginals[pi] += cell_hits[si][pi];
            }
        }
        let max_pot = *pot_marginals.iter().max().unwrap();
        for (pi, &count) in pot_marginals.iter().enumerate() {
            assert!(
                count * 4 >= max_pot,
                "pot bucket {} ({:?}) underrepresented: {} vs max {}",
                pi, pot_intervals[pi], count, max_pot
            );
        }

        // Check SPR bucket marginals are roughly uniform
        let mut spr_marginals = vec![0u32; num_spr];
        for si in 0..num_spr {
            for pi in 0..num_pot {
                spr_marginals[si] += cell_hits[si][pi];
            }
        }
        let max_spr = *spr_marginals.iter().max().unwrap();
        for (si, &count) in spr_marginals.iter().enumerate() {
            assert!(
                count * 4 >= max_spr,
                "SPR bucket {} ({:?}) underrepresented: {} vs max {}",
                si, spr_intervals[si], count, max_spr
            );
        }
    }
}
