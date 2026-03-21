// PBS — Public Belief State

use range_solver::card::card_pair_to_index;

pub const NUM_COMBOS: usize = 1326;

/// Maps two card IDs (0-51, encoded as 4*rank + suit) to a combo index (0-1325).
/// Wraps range-solver's `card_pair_to_index`; order of arguments does not matter.
#[inline]
pub fn combo_index(c1: u8, c2: u8) -> usize {
    card_pair_to_index(c1, c2)
}

/// Public Belief State — the minimal game state visible to both players
/// in a ReBeL subgame.
#[derive(Clone)]
///
/// Contains the public board cards, pot/stack geometry, and per-player
/// reach probabilities over all 1326 hole-card combos.
pub struct Pbs {
    /// Board cards (0-51 encoding: 4*rank + suit), length 0..=5.
    pub board: Vec<u8>,
    /// Total pot in chips.
    pub pot: i32,
    /// Effective stack in chips.
    pub effective_stack: i32,
    /// Reach probabilities per player: `reach_probs[player][combo_index]`.
    /// Player 0 = OOP, player 1 = IP.
    pub reach_probs: Box<[[f32; NUM_COMBOS]; 2]>,
}

impl Pbs {
    /// Creates a new PBS with uniform reach probabilities (1.0 for every combo),
    /// then zeros out combos that are blocked by board cards.
    pub fn new_uniform(board: Vec<u8>, pot: i32, effective_stack: i32) -> Self {
        let mut pbs = Pbs {
            board,
            pot,
            effective_stack,
            reach_probs: Box::new([[1.0f32; NUM_COMBOS]; 2]),
        };
        pbs.zero_blocked_combos();
        pbs
    }

    /// Zeros the reach probability of every combo that contains a board card,
    /// for both players.
    pub fn zero_blocked_combos(&mut self) {
        for &board_card in &self.board {
            for other in 0..52u8 {
                if other == board_card {
                    continue;
                }
                let idx = combo_index(board_card, other);
                self.reach_probs[0][idx] = 0.0;
                self.reach_probs[1][idx] = 0.0;
            }
        }
    }

    /// Returns normalized beliefs for the given player (0 = OOP, 1 = IP).
    ///
    /// Each belief is the player's reach probability for that combo divided
    /// by the sum of all reach probabilities. Blocked combos get belief 0.0.
    pub fn beliefs(&self, player: usize) -> Vec<f32> {
        let reaches = &self.reach_probs[player];
        let sum: f32 = reaches.iter().sum();
        if sum == 0.0 {
            return vec![0.0; NUM_COMBOS];
        }
        let inv_sum = 1.0 / sum;
        reaches.iter().map(|&r| r * inv_sum).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combo_index_symmetric() {
        // combo_index(a, b) == combo_index(b, a) for several pairs
        assert_eq!(combo_index(0, 1), combo_index(1, 0));
        assert_eq!(combo_index(5, 20), combo_index(20, 5));
        assert_eq!(combo_index(51, 0), combo_index(0, 51));
        assert_eq!(combo_index(10, 30), combo_index(30, 10));
        assert_eq!(combo_index(48, 49), combo_index(49, 48));
        // Should be in range [0, 1325]
        assert!(combo_index(0, 1) < NUM_COMBOS);
        assert!(combo_index(50, 51) < NUM_COMBOS);
    }

    #[test]
    fn test_pbs_new_uniform() {
        // 5-card board: As Kh Qd Jc Ts = cards 51, 46, 40, 35, 32
        let board = vec![51, 46, 40, 35, 32];
        let pbs = Pbs::new_uniform(board.clone(), 100, 200);

        assert_eq!(pbs.board, board);
        assert_eq!(pbs.pot, 100);
        assert_eq!(pbs.effective_stack, 200);

        // Check that non-blocked combos have reach 1.0
        // Pick a combo that doesn't contain any board card: 2d2c = cards 0,1
        let idx = combo_index(0, 1);
        assert_eq!(pbs.reach_probs[0][idx], 1.0);
        assert_eq!(pbs.reach_probs[1][idx], 1.0);

        // Another non-blocked combo: 3d3c = cards 4,5
        let idx2 = combo_index(4, 5);
        assert_eq!(pbs.reach_probs[0][idx2], 1.0);
        assert_eq!(pbs.reach_probs[1][idx2], 1.0);
    }

    #[test]
    fn test_pbs_blocks_board_combos() {
        // Board: As(51) Kh(46)
        let board = vec![51, 46];
        let pbs = Pbs::new_uniform(board, 100, 200);

        // Any combo containing card 51 (As) should be blocked
        // As + any other card (except board cards) should be 0.0
        for other in 0..51u8 {
            if other == 46 {
                continue; // skip the other board card
            }
            let idx = combo_index(51, other);
            assert_eq!(
                pbs.reach_probs[0][idx], 0.0,
                "combo with As (51) and card {} should be blocked for player 0",
                other
            );
            assert_eq!(
                pbs.reach_probs[1][idx], 0.0,
                "combo with As (51) and card {} should be blocked for player 1",
                other
            );
        }

        // Any combo containing card 46 (Kh) should be blocked
        for other in 0..52u8 {
            if other == 46 || other == 51 {
                continue;
            }
            let idx = combo_index(46, other);
            assert_eq!(
                pbs.reach_probs[0][idx], 0.0,
                "combo with Kh (46) and card {} should be blocked for player 0",
                other
            );
        }

        // Non-blocked combo should still be 1.0
        let idx = combo_index(0, 1); // 2d2c — not on board
        assert_eq!(pbs.reach_probs[0][idx], 1.0);
        assert_eq!(pbs.reach_probs[1][idx], 1.0);
    }

    #[test]
    fn test_pbs_beliefs_normalize() {
        // Board with 3 cards
        let board = vec![51, 46, 40]; // As Kh Qd
        let pbs = Pbs::new_uniform(board, 200, 500);

        let beliefs_oop = pbs.beliefs(0);
        let beliefs_ip = pbs.beliefs(1);

        assert_eq!(beliefs_oop.len(), NUM_COMBOS);
        assert_eq!(beliefs_ip.len(), NUM_COMBOS);

        // Beliefs should sum to 1.0 (within floating point tolerance)
        let sum_oop: f32 = beliefs_oop.iter().sum();
        let sum_ip: f32 = beliefs_ip.iter().sum();
        assert!(
            (sum_oop - 1.0).abs() < 1e-5,
            "OOP beliefs sum = {}, expected ~1.0",
            sum_oop
        );
        assert!(
            (sum_ip - 1.0).abs() < 1e-5,
            "IP beliefs sum = {}, expected ~1.0",
            sum_ip
        );

        // Blocked combos should have belief 0.0
        let blocked_idx = combo_index(51, 0); // As + 2d — blocked
        assert_eq!(beliefs_oop[blocked_idx], 0.0);
        assert_eq!(beliefs_ip[blocked_idx], 0.0);

        // Non-blocked combos should have positive belief
        let unblocked_idx = combo_index(0, 1); // 2d2c — not blocked
        assert!(beliefs_oop[unblocked_idx] > 0.0);
        assert!(beliefs_ip[unblocked_idx] > 0.0);
    }
}
