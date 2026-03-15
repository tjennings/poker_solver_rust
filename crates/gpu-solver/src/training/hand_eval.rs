//! Hand strength evaluation for sampled river boards.
//!
//! For each sampled board (5 cards) and each of the 1326 possible hole-card
//! combos, evaluate the 7-card hand strength using the range-solver's CPU
//! evaluator. This is a pragmatic approach: boards are downloaded from the
//! GPU (tiny: 5 * N u32s), evaluated on CPU, and the results uploaded back.
//! The transfer is small (~5 MB for 1000 spots * 1326 hands * 4 bytes) and
//! happens once per batch, not per solver iteration.

use range_solver::card::{evaluate_hand_strength, index_to_card_pair, Card};

/// Build the canonical 1326 combo card pairs in index order.
///
/// Combo index `i` maps to `(c1, c2)` via `index_to_card_pair(i)`, where
/// `c1 < c2` and each card is in `0..52`.
fn build_combo_cards() -> Vec<(Card, Card)> {
    (0..1326).map(index_to_card_pair).collect()
}

/// Evaluate hand strengths for all 1326 combos on each board.
///
/// `boards` contains `num_situations * 5` card indices (u32, each in 0..52).
/// Returns `num_situations * 1326` u32 strength values. Higher values
/// indicate stronger hands. Combos that conflict with the board (share a
/// card) receive strength 0.
///
/// The evaluation uses `rayon` to parallelise across situations.
pub fn evaluate_all_hand_strengths(boards: &[u32], num_situations: usize) -> Vec<u32> {
    assert_eq!(
        boards.len(),
        num_situations * 5,
        "boards must contain exactly num_situations * 5 card indices"
    );

    let combo_cards = build_combo_cards();

    // Use rayon to parallelise across situations for large batches.
    use rayon::prelude::*;

    let chunks: Vec<Vec<u32>> = (0..num_situations)
        .into_par_iter()
        .map(|sit| {
            let board: [Card; 5] = [
                boards[sit * 5] as u8,
                boards[sit * 5 + 1] as u8,
                boards[sit * 5 + 2] as u8,
                boards[sit * 5 + 3] as u8,
                boards[sit * 5 + 4] as u8,
            ];

            let mut sit_strengths = vec![0u32; 1326];
            for (combo_idx, &(c1, c2)) in combo_cards.iter().enumerate() {
                // Skip if combo conflicts with board
                if board.contains(&c1) || board.contains(&c2) {
                    continue;
                }
                sit_strengths[combo_idx] =
                    u32::from(evaluate_hand_strength(&board, (c1, c2)));
            }
            sit_strengths
        })
        .collect();

    // Flatten into a single vec
    let mut strengths = Vec::with_capacity(num_situations * 1326);
    for chunk in chunks {
        strengths.extend_from_slice(&chunk);
    }
    strengths
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_solver::card::card_from_str;

    /// Helper: convert card strings to u32 board array.
    fn board_from_strs(cards: &[&str]) -> Vec<u32> {
        cards
            .iter()
            .map(|s| u32::from(card_from_str(s).unwrap()))
            .collect()
    }

    #[test]
    fn test_evaluate_single_board() {
        // Board: As Kh Qd Jc Ts (broadway board)
        let board = board_from_strs(&["As", "Kh", "Qd", "Jc", "Ts"]);
        let strengths = evaluate_all_hand_strengths(&board, 1);

        assert_eq!(strengths.len(), 1326);

        // Combos that use board cards should have strength 0
        let as_card = card_from_str("As").unwrap();
        let kh_card = card_from_str("Kh").unwrap();
        // index_to_card_pair for two board cards should give a blocked combo
        let blocked_idx =
            range_solver::card::card_pair_to_index(as_card, kh_card);
        assert_eq!(
            strengths[blocked_idx], 0,
            "Combo using board cards should have 0 strength"
        );

        // Non-blocked combos should have non-zero strength
        let non_blocked_count = strengths.iter().filter(|&&s| s > 0).count();
        assert!(
            non_blocked_count > 0,
            "At least some combos should have non-zero strength"
        );

        // On a 5-card board, combos using 0 board cards = C(47,2) = 1081
        // Combos using exactly 1 board card = 5 * 47 = 235
        // Combos using exactly 2 board cards = C(5,2) = 10
        // Total blocked = 235 + 10 = 245
        // Total non-blocked = 1326 - 245 = 1081
        assert_eq!(non_blocked_count, 1081);
    }

    #[test]
    fn test_hand_ordering_aa_beats_kk_beats_qq() {
        // Board: 2c 3d 7h 9s Tc — no straight/flush possible for AA/KK/QQ
        let board = board_from_strs(&["2c", "3d", "7h", "9s", "Tc"]);
        let strengths = evaluate_all_hand_strengths(&board, 1);

        // AA: e.g. Ah As
        let ah = card_from_str("Ah").unwrap();
        let as_ = card_from_str("As").unwrap();
        let aa_idx = range_solver::card::card_pair_to_index(ah, as_);

        // KK: e.g. Kh Ks
        let kh = card_from_str("Kh").unwrap();
        let ks = card_from_str("Ks").unwrap();
        let kk_idx = range_solver::card::card_pair_to_index(kh, ks);

        // QQ: e.g. Qh Qs
        let qh = card_from_str("Qh").unwrap();
        let qs = card_from_str("Qs").unwrap();
        let qq_idx = range_solver::card::card_pair_to_index(qh, qs);

        assert!(
            strengths[aa_idx] > strengths[kk_idx],
            "AA ({}) should beat KK ({})",
            strengths[aa_idx],
            strengths[kk_idx]
        );
        assert!(
            strengths[kk_idx] > strengths[qq_idx],
            "KK ({}) should beat QQ ({})",
            strengths[kk_idx],
            strengths[qq_idx]
        );
    }

    #[test]
    fn test_multiple_boards() {
        // Two boards: one with low cards, one with high cards
        let mut boards = Vec::new();
        boards.extend(board_from_strs(&["2c", "3d", "4h", "5s", "7c"]));
        boards.extend(board_from_strs(&["Tc", "Jd", "Qh", "Ks", "9c"]));

        let strengths = evaluate_all_hand_strengths(&boards, 2);
        assert_eq!(strengths.len(), 2 * 1326);

        // Both boards should have some non-zero strengths
        let board1_nonzero = strengths[..1326].iter().filter(|&&s| s > 0).count();
        let board2_nonzero = strengths[1326..].iter().filter(|&&s| s > 0).count();
        assert!(board1_nonzero > 0);
        assert!(board2_nonzero > 0);
    }

    #[test]
    fn test_straight_beats_pair() {
        // Board: 5c 6d 7h Ks 2c
        let board = board_from_strs(&["5c", "6d", "7h", "Ks", "2c"]);
        let strengths = evaluate_all_hand_strengths(&board, 1);

        // 8h 9h makes a straight (5-6-7-8-9)
        let c8h = card_from_str("8h").unwrap();
        let c9h = card_from_str("9h").unwrap();
        let straight_idx = range_solver::card::card_pair_to_index(c8h, c9h);

        // Ah Ad makes a pair of aces (second pair after kings)
        let ah = card_from_str("Ah").unwrap();
        let ad = card_from_str("Ad").unwrap();
        let pair_idx = range_solver::card::card_pair_to_index(ah, ad);

        assert!(
            strengths[straight_idx] > strengths[pair_idx],
            "Straight ({}) should beat pair of aces ({})",
            strengths[straight_idx],
            strengths[pair_idx]
        );
    }

    #[test]
    fn test_empty_boards() {
        let strengths = evaluate_all_hand_strengths(&[], 0);
        assert!(strengths.is_empty());
    }

    #[test]
    fn test_combo_cards_ordering() {
        let combos = build_combo_cards();
        assert_eq!(combos.len(), 1326);

        // Verify they match index_to_card_pair
        for (i, &(c1, c2)) in combos.iter().enumerate() {
            let expected = index_to_card_pair(i);
            assert_eq!((c1, c2), expected, "Mismatch at combo index {i}");
        }
    }
}
