use rand::Rng;
use range_solver::card::index_to_card_pair;

/// Number of possible hole card combos in HUNL.
pub const NUM_COMBOS: usize = 1326;

/// Compute hand strength for each of the 1326 combos on a board of 4 or 5 cards.
///
/// For a 5-card board, returns ordinal ranks directly from the 7-card evaluator.
/// For a 4-card board, averages raw strength over all 48 possible river cards,
/// then converts to ordinal ranks.
///
/// Returns an array where `strengths[i]` is the hand rank (higher = stronger)
/// for the combo at index `i`. Board-conflicting combos get strength 0.
///
/// # Panics
///
/// Panics if `board.len()` is not 4 or 5.
pub fn compute_hand_strengths(board: &[u8]) -> [u16; NUM_COMBOS] {
    assert!(
        board.len() == 4 || board.len() == 5,
        "board must have 4 or 5 cards, got {}",
        board.len()
    );

    let board_mask = slice_to_mask(board);

    if board.len() == 5 {
        compute_strengths_5(board, board_mask)
    } else {
        compute_strengths_4(board, board_mask)
    }
}

/// Fast path for 5-card boards: one evaluation per combo.
fn compute_strengths_5(board: &[u8], board_mask: u64) -> [u16; NUM_COMBOS] {
    let mut raw = [0i32; NUM_COMBOS];
    for (i, strength) in raw.iter_mut().enumerate() {
        let (c1, c2) = index_to_card_pair(i);
        let hand_mask: u64 = (1 << c1) | (1 << c2);
        if hand_mask & board_mask != 0 {
            continue;
        }
        *strength = evaluate_7_slice(c1, c2, board);
    }
    raw_to_ordinal(&raw)
}

/// 4-card board: average raw strength across all 48 possible river cards.
fn compute_strengths_4(board: &[u8], board_mask: u64) -> [u16; NUM_COMBOS] {
    let mut avg = [0.0f64; NUM_COMBOS];
    for (i, avg_val) in avg.iter_mut().enumerate() {
        let (c1, c2) = index_to_card_pair(i);
        let hand_mask: u64 = (1 << c1) | (1 << c2);
        if hand_mask & board_mask != 0 {
            continue;
        }
        let used = board_mask | hand_mask;
        let mut sum = 0i64;
        let mut count = 0u32;
        for river in 0u8..52 {
            if used & (1 << river) != 0 {
                continue;
            }
            let full_board = [board[0], board[1], board[2], board[3], river];
            sum += i64::from(evaluate_7_slice(c1, c2, &full_board));
            count += 1;
        }
        if count > 0 {
            *avg_val = sum as f64 / f64::from(count);
        }
    }

    // Convert averaged strengths to ordinal ranks via sorting.
    let mut pairs: Vec<(u64, usize)> = (0..NUM_COMBOS)
        .filter(|&i| avg[i] > 0.0)
        .map(|i| (avg[i].to_bits(), i))
        .collect();
    pairs.sort_unstable();

    let mut strengths = [0u16; NUM_COMBOS];
    let mut rank = 1u16;
    for window_idx in 0..pairs.len() {
        if window_idx > 0 && pairs[window_idx].0 != pairs[window_idx - 1].0 {
            rank = window_idx as u16 + 1;
        }
        strengths[pairs[window_idx].1] = rank;
    }
    strengths
}

/// Convert raw i32 strengths to ordinal u16 ranks.
fn raw_to_ordinal(raw: &[i32; NUM_COMBOS]) -> [u16; NUM_COMBOS] {
    let mut pairs: Vec<(i32, usize)> = (0..NUM_COMBOS)
        .filter(|&i| raw[i] != 0)
        .map(|i| (raw[i], i))
        .collect();
    pairs.sort_unstable();

    let mut strengths = [0u16; NUM_COMBOS];
    let mut rank = 1u16;
    for window_idx in 0..pairs.len() {
        if window_idx > 0 && pairs[window_idx].0 != pairs[window_idx - 1].0 {
            rank = window_idx as u16 + 1;
        }
        strengths[pairs[window_idx].1] = rank;
    }
    strengths
}

/// Generate a random range using the DeepStack R(S,p) procedure.
///
/// Returns a 1326-element array of reach probabilities summing to 1.0.
/// Board-conflicting combos have zero reach.
pub fn generate_rsp_range<R: Rng>(board: &[u8], rng: &mut R) -> [f32; NUM_COMBOS] {
    let strengths = compute_hand_strengths(board);
    generate_rsp_range_with_strengths(&strengths, rng)
}

/// Like [`generate_rsp_range`], but accepts pre-computed hand strengths.
///
/// Use this when generating multiple ranges for the same board to avoid
/// recomputing `compute_hand_strengths` for each call.
pub fn generate_rsp_range_with_strengths<R: Rng>(
    strengths: &[u16; NUM_COMBOS],
    rng: &mut R,
) -> [f32; NUM_COMBOS] {
    // Collect valid (non-blocked) combo indices, sorted by ascending strength.
    let mut valid: Vec<usize> = (0..NUM_COMBOS)
        .filter(|&i| strengths[i] > 0)
        .collect();
    valid.sort_by_key(|&i| strengths[i]);

    let mut range = [0.0f32; NUM_COMBOS];
    rsp_recursive(&valid, 1.0, &mut range, rng);
    range
}

/// Recursive R(S, p) implementation.
///
/// Splits probability mass `p` randomly between the stronger and weaker halves
/// of `hands` (which must be sorted by ascending strength). Stronger hands
/// receive the larger portion on average, creating ranges correlated with
/// hand strength.
fn rsp_recursive<R: Rng>(hands: &[usize], p: f64, range: &mut [f32; NUM_COMBOS], rng: &mut R) {
    if hands.is_empty() || p <= 0.0 {
        return;
    }
    if hands.len() == 1 {
        range[hands[0]] += p as f32;
        return;
    }

    let draw: f64 = rng.r#gen::<f64>() * p;
    let remainder = p - draw;

    // Give the larger portion to stronger hands, smaller to weaker.
    // This creates a positive correlation between hand strength and reach.
    let p_strong = draw.max(remainder);
    let p_weak = draw.min(remainder);

    let mid = hands.len() / 2;
    let (weaker, stronger) = hands.split_at(mid);

    rsp_recursive(stronger, p_strong, range, rng);
    rsp_recursive(weaker, p_weak, range, rng);
}

// ---------------------------------------------------------------------------
// Inline 7-card hand evaluator
//
// This is a self-contained evaluator that produces an i32 encoding where
// higher values correspond to stronger hands. The encoding places the hand
// category in the high bits and kicker information in the low bits, so
// standard integer comparison gives correct hand ordering.
//
// Card encoding: card_id = 4 * rank + suit
//   rank: 2→0, 3→1, ..., A→12
//   suit: club→0, diamond→1, heart→2, spade→3
// ---------------------------------------------------------------------------

/// Evaluate a 7-card hand (2 hole cards + 5 board cards).
/// Returns an i32 where higher = stronger. Zero is never returned for valid input.
fn evaluate_7_slice(c1: u8, c2: u8, board: &[u8]) -> i32 {
    let cards = [
        c1 as usize,
        c2 as usize,
        board[0] as usize,
        board[1] as usize,
        board[2] as usize,
        board[3] as usize,
        board[4] as usize,
    ];

    let mut rankset = 0i32;
    let mut rankset_suit = [0i32; 4];
    let mut rank_count = [0i32; 13];

    for &card in &cards {
        let rank = card / 4;
        let suit = card % 4;
        rankset |= 1 << rank;
        rankset_suit[suit] |= 1 << rank;
        rank_count[rank] += 1;
    }

    let mut rankset_of_count = [0i32; 5];
    for rank in 0..13 {
        rankset_of_count[rank_count[rank] as usize] |= 1 << rank;
    }

    let mut flush_suit: i32 = -1;
    for suit in 0..4 {
        if rankset_suit[suit as usize].count_ones() >= 5 {
            flush_suit = suit;
        }
    }

    let is_straight = find_straight(rankset);

    if flush_suit >= 0 {
        let is_straight_flush = find_straight(rankset_suit[flush_suit as usize]);
        if is_straight_flush != 0 {
            (8 << 26) | is_straight_flush
        } else {
            (5 << 26) | keep_n_msb(rankset_suit[flush_suit as usize], 5)
        }
    } else if rankset_of_count[4] != 0 {
        let remaining = keep_n_msb(rankset ^ rankset_of_count[4], 1);
        (7 << 26) | (rankset_of_count[4] << 13) | remaining
    } else if rankset_of_count[3].count_ones() == 2 {
        let trips = keep_n_msb(rankset_of_count[3], 1);
        let pair = rankset_of_count[3] ^ trips;
        (6 << 26) | (trips << 13) | pair
    } else if rankset_of_count[3] != 0 && rankset_of_count[2] != 0 {
        let pair = keep_n_msb(rankset_of_count[2], 1);
        (6 << 26) | (rankset_of_count[3] << 13) | pair
    } else if is_straight != 0 {
        (4 << 26) | is_straight
    } else if rankset_of_count[3] != 0 {
        let remaining = keep_n_msb(rankset_of_count[1], 2);
        (3 << 26) | (rankset_of_count[3] << 13) | remaining
    } else if rankset_of_count[2].count_ones() >= 2 {
        let pairs = keep_n_msb(rankset_of_count[2], 2);
        let remaining = keep_n_msb(rankset ^ pairs, 1);
        (2 << 26) | (pairs << 13) | remaining
    } else if rankset_of_count[2] != 0 {
        let remaining = keep_n_msb(rankset_of_count[1], 3);
        (1 << 26) | (rankset_of_count[2] << 13) | remaining
    } else {
        keep_n_msb(rankset, 5)
    }
}

/// Keep the `n` most-significant set bits of `x`, clearing all others.
#[inline]
fn keep_n_msb(mut x: i32, n: i32) -> i32 {
    let mut ret = 0;
    for _ in 0..n {
        let bit = 1 << (x.leading_zeros() ^ 31);
        x ^= bit;
        ret |= bit;
    }
    ret
}

/// Detect a 5-card straight in a rank bitset. Returns the top-rank bit of the
/// straight, or 0 if none exists. Handles the wheel (A-2-3-4-5).
#[inline]
fn find_straight(rankset: i32) -> i32 {
    const WHEEL: i32 = 0b1_0000_0000_1111;
    let is_straight =
        rankset & (rankset << 1) & (rankset << 2) & (rankset << 3) & (rankset << 4);
    if is_straight != 0 {
        keep_n_msb(is_straight, 1)
    } else if (rankset & WHEEL) == WHEEL {
        1 << 3
    } else {
        0
    }
}

/// Convert a board slice into a 64-bit mask for fast conflict checks.
#[inline]
fn slice_to_mask(board: &[u8]) -> u64 {
    board.iter().fold(0u64, |mask, &card| mask | (1 << card))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn test_board() -> [u8; 5] {
        // Qs Jh 2c 8d 3s = card IDs
        [
            4 * 10 + 3, // Qs
            4 * 9 + 2,  // Jh
            4 * 0 + 0,  // 2c
            4 * 6 + 1,  // 8d
            4 * 1 + 3,  // 3s
        ]
    }

    #[test]
    fn range_sums_to_one() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let board = test_board();
        let range = generate_rsp_range(&board, &mut rng);
        let sum: f32 = range.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "range sum = {sum}");
    }

    #[test]
    fn range_all_non_negative() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let board = test_board();
        let range = generate_rsp_range(&board, &mut rng);
        for (i, &v) in range.iter().enumerate() {
            assert!(v >= 0.0, "range[{i}] = {v} is negative");
        }
    }

    #[test]
    fn board_blocked_combos_are_zero() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let board = test_board();
        let range = generate_rsp_range(&board, &mut rng);

        for card in &board {
            for i in 0..1326 {
                let (c1, c2) = range_solver::card::index_to_card_pair(i);
                if c1 == *card || c2 == *card {
                    assert_eq!(
                        range[i], 0.0,
                        "combo {i} (cards {c1},{c2}) conflicts with board card {card}"
                    );
                }
            }
        }
    }

    #[test]
    fn strong_hands_have_higher_mean_reach() {
        let board = test_board();
        let mut top_sum = 0.0f64;
        let mut bottom_sum = 0.0f64;
        let n = 1000;

        let strengths = compute_hand_strengths(&board);
        let valid: Vec<usize> = (0..1326)
            .filter(|&i| {
                let (c1, c2) = range_solver::card::index_to_card_pair(i);
                !board.contains(&c1) && !board.contains(&c2)
            })
            .collect();

        let mut sorted = valid.clone();
        sorted.sort_by(|&a, &b| strengths[a].cmp(&strengths[b]));
        let top_quarter = &sorted[sorted.len() * 3 / 4..];
        let bottom_quarter = &sorted[..sorted.len() / 4];

        for seed in 0..n {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let range = generate_rsp_range(&board, &mut rng);
            top_sum += top_quarter.iter().map(|&i| range[i] as f64).sum::<f64>();
            bottom_sum += bottom_quarter.iter().map(|&i| range[i] as f64).sum::<f64>();
        }

        assert!(
            top_sum > bottom_sum,
            "top quarter mean {:.6} should exceed bottom quarter mean {:.6}",
            top_sum / (n as f64 * top_quarter.len() as f64),
            bottom_sum / (n as f64 * bottom_quarter.len() as f64)
        );
    }

    #[test]
    fn deterministic_with_same_seed() {
        let board = test_board();
        let mut rng1 = ChaCha8Rng::seed_from_u64(123);
        let mut rng2 = ChaCha8Rng::seed_from_u64(123);
        let range1 = generate_rsp_range(&board, &mut rng1);
        let range2 = generate_rsp_range(&board, &mut rng2);
        assert_eq!(range1, range2);
    }

    #[test]
    fn hand_strengths_4_card_board() {
        // 4-card board: Qs Jh 2c 8d
        let board = [
            4 * 10 + 3, // Qs
            4 * 9 + 2,  // Jh
            4 * 0 + 0,  // 2c
            4 * 6 + 1,  // 8d
        ];
        let strengths = compute_hand_strengths(&board);

        // Board-blocked combos should have strength 0.
        for i in 0..NUM_COMBOS {
            let (c1, c2) = range_solver::card::index_to_card_pair(i);
            if board.contains(&c1) || board.contains(&c2) {
                assert_eq!(strengths[i], 0, "blocked combo {i} should have strength 0");
            }
        }

        // Non-blocked combos should have positive strength.
        let non_blocked: Vec<usize> = (0..NUM_COMBOS)
            .filter(|&i| {
                let (c1, c2) = range_solver::card::index_to_card_pair(i);
                !board.contains(&c1) && !board.contains(&c2)
            })
            .collect();
        assert!(!non_blocked.is_empty());
        for &i in &non_blocked {
            assert!(strengths[i] > 0, "valid combo {i} should have positive strength");
        }
    }

    #[test]
    fn range_4_card_board_sums_to_one() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let board: [u8; 4] = [
            4 * 10 + 3, // Qs
            4 * 9 + 2,  // Jh
            4 * 0 + 0,  // 2c
            4 * 6 + 1,  // 8d
        ];
        let range = generate_rsp_range(&board, &mut rng);
        let sum: f32 = range.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "4-card range sum = {sum}");
    }
}
