// Self-play training loop — Algorithm 1 from the ReBeL paper.
//
// Generates training data by playing hands against itself with subgame
// solving at each decision point. At each PBS:
//   1. Solve depth-limited subgame (value net at leaves)
//   2. Record (PBS, root CFVs) as a training example
//   3. Sample an action from the average strategy
//   4. Update beliefs via Bayes rule
//   5. Advance to the next PBS
//
// This is a simplified version that solves at street boundaries rather
// than at every intra-street decision node. A full implementation would
// traverse the betting tree within each street.

use rand::Rng;
use rand::SeedableRng;

use crate::blueprint_sampler::deal_hand;
use crate::data_buffer::DiskBuffer;
use crate::generate::pbs_to_buffer_record;
use crate::pbs::{combo_index, Pbs, NUM_COMBOS};
use crate::solver::{solve_depth_limited_pbs, SolveConfig, SolveResult};
use poker_solver_core::blueprint_v2::cfv_subgame_solver::LeafEvaluator;

/// A training example from self-play: a PBS and its computed CFVs.
pub struct TrainingExample {
    /// The public belief state at the time of solving.
    pub pbs: Pbs,
    /// Counterfactual values for each player: `[OOP, IP]`, indexed by combo.
    pub cfvs: Box<[[f32; NUM_COMBOS]; 2]>,
}

/// Configuration for the self-play loop.
pub struct SelfPlayConfig {
    /// Total number of hands to play.
    pub num_hands: usize,
    /// Number of CFR iterations per subgame solve.
    pub cfr_iterations: u32,
    /// Exploration probability: with this probability, one player takes
    /// a uniform random action instead of the solved strategy.
    pub exploration_epsilon: f32,
    /// Starting stack size in chips.
    pub initial_stack: i32,
    /// Small blind in chips.
    pub small_blind: i32,
    /// Big blind in chips.
    pub big_blind: i32,
    /// Number of hands between training batches.
    pub hands_per_training_batch: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

/// Street identifiers for the self-play state machine.
///
/// We use our own simple enum here rather than importing the core Street
/// enum, to keep the self-play logic self-contained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Street {
    Preflop,
    Flop,
    Turn,
    River,
}

/// Number of board cards visible at each street.
fn board_card_count(street: Street) -> usize {
    match street {
        Street::Preflop => 0,
        Street::Flop => 3,
        Street::Turn => 4,
        Street::River => 5,
    }
}

/// The sequence of streets in a hand.
const STREET_ORDER: [Street; 4] = [
    Street::Preflop,
    Street::Flop,
    Street::Turn,
    Street::River,
];

/// Play one hand via self-play with subgame solving at each street boundary.
///
/// At each street boundary:
/// 1. Build the PBS for the current street (board cards, pot, stack, reach probs)
/// 2. Solve the depth-limited subgame at this PBS
/// 3. Record (PBS, root CFVs) as a training example
/// 4. If not the last street, determine whether the hand continues:
///    - Sample a "meta-action" (continue or fold) from the solved CFVs
///    - With probability `exploration_epsilon`, one player takes a random action
/// 5. Update beliefs based on the action taken
/// 6. Advance to the next street
///
/// This is a simplified version that treats each street boundary as a single
/// decision point. A full implementation would traverse the intra-street
/// betting tree, sampling actions at each decision node.
///
/// Returns collected training examples for this hand.
pub fn play_self_play_hand<R: Rng>(
    evaluator: &dyn LeafEvaluator,
    solve_config: &SolveConfig,
    sp_config: &SelfPlayConfig,
    rng: &mut R,
) -> Vec<TrainingExample> {
    let mut examples = Vec::new();
    let deal = deal_hand(rng);

    // Initial pot = small blind + big blind
    let initial_pot = sp_config.small_blind + sp_config.big_blind;

    // Effective stack = starting stack minus the larger blind
    let initial_eff_stack =
        sp_config.initial_stack - sp_config.small_blind.max(sp_config.big_blind);

    // Pot and effective stack for the simplified version.
    // In a full implementation these would be updated by betting actions;
    // here they remain at their initial values since we only solve at
    // street boundaries and don't model intra-street bet sizing.
    let pot = initial_pot;
    let effective_stack = initial_eff_stack;

    // Initialize reach probabilities: uniform for both players.
    let mut reach_probs = Box::new([[1.0f32; NUM_COMBOS]; 2]);

    // Decide exploration for this hand.
    let exploring = rng.random::<f32>() < sp_config.exploration_epsilon;
    let explore_player: usize = if exploring {
        rng.random_range(0..2usize)
    } else {
        usize::MAX // sentinel: no exploration
    };

    for &street in &STREET_ORDER {
        let n_board = board_card_count(street);

        // Build the PBS for this street.
        let board = deal.board[..n_board].to_vec();
        let mut pbs = Pbs {
            board,
            pot,
            effective_stack,
            reach_probs: reach_probs.clone(),
        };
        pbs.zero_blocked_combos();

        // Solve the subgame at this PBS.
        let solve_result = match solve_depth_limited_pbs(&pbs, solve_config, evaluator) {
            Ok(r) => r,
            Err(_) => {
                // Solving failed (e.g., pot or stack constraints).
                // End the hand here — no more examples.
                break;
            }
        };

        // Record the training example.
        examples.push(TrainingExample {
            pbs: pbs.clone(),
            cfvs: Box::new([solve_result.oop_cfvs, solve_result.ip_cfvs]),
        });

        // On the river, no more streets to advance to.
        if street == Street::River {
            break;
        }

        // Determine whether the hand continues to the next street.
        //
        // In a full implementation, we would traverse the solved betting tree
        // and sample actions at each decision node. Here, we use a simplified
        // approach: we check whether the hand should fold based on the
        // CFV structure, and update beliefs with a "continue" action.
        //
        // For the simplified version, we always continue to the next street
        // (no fold detection), but update reach probabilities using the
        // solve result as a soft belief update.
        //
        // The reach update is based on the proportion of CFV that is
        // positive (indicating the combo wants to continue). This is a
        // heuristic approximation of the full intra-street belief update.
        update_beliefs_simplified(
            &mut reach_probs,
            &solve_result,
            &deal.board[..n_board],
            exploring && explore_player < 2,
            explore_player,
            rng,
        );

        // Zero out combos blocked by the new street's board cards.
        let next_n_board = board_card_count(STREET_ORDER[street_index(street) + 1]);
        for &board_card in &deal.board[n_board..next_n_board] {
            for other in 0..52u8 {
                if other == board_card {
                    continue;
                }
                let idx = combo_index(board_card, other);
                reach_probs[0][idx] = 0.0;
                reach_probs[1][idx] = 0.0;
            }
        }
    }

    examples
}

/// Returns the index of a street in `STREET_ORDER`.
fn street_index(street: Street) -> usize {
    match street {
        Street::Preflop => 0,
        Street::Flop => 1,
        Street::Turn => 2,
        Street::River => 3,
    }
}

/// Simplified belief update between streets.
///
/// In the full ReBeL algorithm, beliefs are updated action-by-action within
/// the solved betting tree. Here, we approximate this by scaling reach
/// probabilities based on the solve result.
///
/// For each combo, we compute a "continuation probability" derived from the
/// normalized CFV magnitude. Combos with strongly negative CFVs (wanting to
/// fold) get reduced reach. This is a coarse approximation, but captures the
/// essential idea of Bayesian belief updating.
///
/// When `exploring` is true and `explore_player` identifies a valid player,
/// that player's reach probabilities are left unchanged (uniform exploration).
fn update_beliefs_simplified<R: Rng>(
    reach_probs: &mut [[f32; NUM_COMBOS]; 2],
    solve_result: &SolveResult,
    _board: &[u8],
    exploring: bool,
    explore_player: usize,
    _rng: &mut R,
) {
    let cfvs_per_player = [&solve_result.oop_cfvs, &solve_result.ip_cfvs];

    for player in 0..2usize {
        // If this player is exploring, keep their reach unchanged.
        if exploring && player == explore_player {
            continue;
        }

        let cfvs = cfvs_per_player[player];

        // Find the range of CFVs to normalize.
        let mut min_cfv = f32::MAX;
        let mut max_cfv = f32::MIN;
        for (i, &cfv) in cfvs.iter().enumerate() {
            if reach_probs[player][i] > 0.0 {
                min_cfv = min_cfv.min(cfv);
                max_cfv = max_cfv.max(cfv);
            }
        }

        // If all CFVs are the same, no belief update needed.
        let range = max_cfv - min_cfv;
        if range < 1e-8 {
            continue;
        }

        // Scale reach by a soft continuation probability based on CFV.
        // Map CFVs to [0.1, 1.0] range: higher CFV = more likely to continue.
        let inv_range = 1.0 / range;
        for i in 0..NUM_COMBOS {
            if reach_probs[player][i] > 0.0 {
                let normalized = (cfvs[i] - min_cfv) * inv_range; // [0, 1]
                let continuation_prob = 0.1 + 0.9 * normalized; // [0.1, 1.0]
                reach_probs[player][i] *= continuation_prob;
            }
        }
    }
}

/// Run the self-play training loop.
///
/// Plays `sp_config.num_hands` hands via self-play, collecting training
/// examples at each street boundary. All examples are converted to
/// `BufferRecord`s and appended to the disk buffer.
///
/// Returns the total number of training examples generated.
pub fn self_play_training_loop(
    evaluator: &dyn LeafEvaluator,
    solve_config: &SolveConfig,
    sp_config: &SelfPlayConfig,
    buffer: &mut DiskBuffer,
) -> usize {
    let mut total_examples = 0usize;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(sp_config.seed);

    for hand_idx in 0..sp_config.num_hands {
        let examples = play_self_play_hand(evaluator, solve_config, sp_config, &mut rng);

        // Convert each training example to buffer records (one per player).
        for example in &examples {
            for player in 0..2u8 {
                let mut rec = pbs_to_buffer_record(&example.pbs, player);
                // Fill in the CFVs and game value from the solve result.
                rec.cfvs = example.cfvs[player as usize];
                rec.game_value = crate::solver::weighted_game_value(
                    &example.cfvs[player as usize],
                    &example.pbs.reach_probs[player as usize],
                );
                if let Err(e) = buffer.append(&rec) {
                    eprintln!(
                        "Warning: failed to append record for hand {hand_idx}, player {player}: {e}"
                    );
                }
            }
            total_examples += 1;
        }

        // Progress logging every 100 hands.
        if (hand_idx + 1) % 100 == 0 {
            eprintln!(
                "Self-play progress: {}/{} hands, {} examples so far",
                hand_idx + 1,
                sp_config.num_hands,
                total_examples
            );
        }
    }

    total_examples
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_example_construction() {
        // Verify TrainingExample fields can be constructed and accessed.
        let board = vec![0, 4, 8, 12, 16]; // 2c, 3c, 4c, 5c, 6c
        let pbs = Pbs::new_uniform(board.clone(), 100, 200);

        let mut cfvs = Box::new([[0.0f32; NUM_COMBOS]; 2]);
        cfvs[0][0] = 1.5;
        cfvs[0][100] = -0.3;
        cfvs[1][0] = -1.5;
        cfvs[1][100] = 0.3;

        let example = TrainingExample { pbs, cfvs };

        assert_eq!(example.pbs.board, board);
        assert_eq!(example.pbs.pot, 100);
        assert_eq!(example.pbs.effective_stack, 200);
        assert_eq!(example.cfvs[0][0], 1.5);
        assert_eq!(example.cfvs[0][100], -0.3);
        assert_eq!(example.cfvs[1][0], -1.5);
        assert_eq!(example.cfvs[1][100], 0.3);

        // CFV arrays are the right size
        assert_eq!(example.cfvs[0].len(), NUM_COMBOS);
        assert_eq!(example.cfvs[1].len(), NUM_COMBOS);
    }

    #[test]
    fn test_self_play_config_defaults() {
        let config = SelfPlayConfig {
            num_hands: 1000,
            cfr_iterations: 200,
            exploration_epsilon: 0.25,
            initial_stack: 400,
            small_blind: 1,
            big_blind: 2,
            hands_per_training_batch: 100,
            seed: 42,
        };

        assert_eq!(config.num_hands, 1000);
        assert_eq!(config.cfr_iterations, 200);
        assert!((config.exploration_epsilon - 0.25).abs() < 1e-6);
        assert_eq!(config.initial_stack, 400);
        assert_eq!(config.small_blind, 1);
        assert_eq!(config.big_blind, 2);
        assert_eq!(config.hands_per_training_batch, 100);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_street_order() {
        assert_eq!(STREET_ORDER.len(), 4);
        assert_eq!(STREET_ORDER[0], Street::Preflop);
        assert_eq!(STREET_ORDER[1], Street::Flop);
        assert_eq!(STREET_ORDER[2], Street::Turn);
        assert_eq!(STREET_ORDER[3], Street::River);
    }

    #[test]
    fn test_board_card_count() {
        assert_eq!(board_card_count(Street::Preflop), 0);
        assert_eq!(board_card_count(Street::Flop), 3);
        assert_eq!(board_card_count(Street::Turn), 4);
        assert_eq!(board_card_count(Street::River), 5);
    }

    #[test]
    fn test_street_index() {
        assert_eq!(street_index(Street::Preflop), 0);
        assert_eq!(street_index(Street::Flop), 1);
        assert_eq!(street_index(Street::Turn), 2);
        assert_eq!(street_index(Street::River), 3);
    }

    #[test]
    fn test_update_beliefs_simplified_no_crash() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut reach = [[1.0f32; NUM_COMBOS]; 2];
        let result = SolveResult {
            oop_cfvs: [0.0f32; NUM_COMBOS],
            ip_cfvs: [0.0f32; NUM_COMBOS],
            oop_game_value: 0.0,
            ip_game_value: 0.0,
            exploitability: 0.0,
        };
        let board = [0u8, 4, 8];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Should not panic with zero CFVs (all equal -> no update).
        update_beliefs_simplified(&mut reach, &result, &board, false, usize::MAX, &mut rng);

        // Reach should be unchanged when all CFVs are equal.
        for i in 0..NUM_COMBOS {
            assert_eq!(reach[0][i], 1.0, "OOP reach should be unchanged at combo {i}");
            assert_eq!(reach[1][i], 1.0, "IP reach should be unchanged at combo {i}");
        }
    }

    #[test]
    fn test_update_beliefs_simplified_varies_reach() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut reach = [[1.0f32; NUM_COMBOS]; 2];
        let mut oop_cfvs = [0.0f32; NUM_COMBOS];
        let ip_cfvs = [0.0f32; NUM_COMBOS];

        // Set varying CFVs for OOP: combo 0 has high value, combo 1 has low value.
        oop_cfvs[0] = 1.0;
        oop_cfvs[1] = -1.0;
        // Leave the rest at 0.0

        let result = SolveResult {
            oop_cfvs,
            ip_cfvs,
            oop_game_value: 0.0,
            ip_game_value: 0.0,
            exploitability: 0.0,
        };
        let board: [u8; 0] = []; // empty board (preflop)
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        update_beliefs_simplified(&mut reach, &result, &board, false, usize::MAX, &mut rng);

        // Combo 0 (high CFV) should have higher reach than combo 1 (low CFV).
        assert!(
            reach[0][0] > reach[0][1],
            "High-CFV combo should keep higher reach: {} vs {}",
            reach[0][0],
            reach[0][1]
        );

        // Both should still be positive (min continuation prob is 0.1).
        assert!(reach[0][0] > 0.0, "reach should be positive");
        assert!(reach[0][1] > 0.0, "reach should be positive");

        // IP reach should be unchanged since IP CFVs are all 0.0.
        for i in 0..NUM_COMBOS {
            assert_eq!(reach[1][i], 1.0, "IP reach should be unchanged");
        }
    }

    #[test]
    fn test_update_beliefs_exploring_player_unchanged() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut reach = [[1.0f32; NUM_COMBOS]; 2];
        let mut oop_cfvs = [0.5f32; NUM_COMBOS];
        let ip_cfvs = [0.5f32; NUM_COMBOS];
        oop_cfvs[0] = 1.0;
        oop_cfvs[1] = -1.0;

        let result = SolveResult {
            oop_cfvs,
            ip_cfvs,
            oop_game_value: 0.0,
            ip_game_value: 0.0,
            exploitability: 0.0,
        };
        let board: [u8; 0] = [];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Player 0 (OOP) is exploring — their reach should not change.
        update_beliefs_simplified(&mut reach, &result, &board, true, 0, &mut rng);

        // OOP reach should be unchanged.
        for i in 0..NUM_COMBOS {
            assert_eq!(
                reach[0][i], 1.0,
                "Exploring player's reach should be unchanged at combo {i}"
            );
        }
    }

    #[test]
    fn test_training_example_to_buffer_record() {
        // Verify that training examples can be converted to buffer records.
        let board = vec![0, 4, 8, 12, 16];
        let pbs = Pbs::new_uniform(board.clone(), 100, 200);

        let mut cfvs = Box::new([[0.0f32; NUM_COMBOS]; 2]);
        cfvs[0][0] = 0.5;
        cfvs[1][0] = -0.5;

        let example = TrainingExample {
            pbs: pbs.clone(),
            cfvs,
        };

        // Convert to buffer record for OOP.
        let mut rec = pbs_to_buffer_record(&example.pbs, 0);
        rec.cfvs = example.cfvs[0];
        rec.game_value = crate::solver::weighted_game_value(
            &example.cfvs[0],
            &example.pbs.reach_probs[0],
        );

        assert_eq!(rec.player, 0);
        assert_eq!(rec.cfvs[0], 0.5);
        assert_eq!(rec.board_card_count, 5);
        assert_eq!(rec.pot, 100.0);
        assert!(rec.game_value.is_finite());
    }

    #[test]
    #[ignore] // Requires a LeafEvaluator (e.g., trained CfvNet)
    fn test_play_self_play_hand_integration() {
        // This test would:
        // 1. Create or load a LeafEvaluator
        // 2. Configure SolveConfig and SelfPlayConfig
        // 3. Call play_self_play_hand
        // 4. Verify that training examples are produced
        // 5. Verify that CFVs are non-zero for non-blocked combos
        //
        // To run: cargo test -p rebel self_play::tests::test_play_self_play_hand_integration -- --ignored
        // Requires: a trained value network model
    }

    #[test]
    #[ignore] // Requires a LeafEvaluator (e.g., trained CfvNet)
    fn test_self_play_training_loop_integration() {
        // This test would:
        // 1. Create or load a LeafEvaluator
        // 2. Configure SolveConfig and SelfPlayConfig
        // 3. Create a DiskBuffer
        // 4. Call self_play_training_loop with a small num_hands
        // 5. Verify records were appended to the buffer
        // 6. Verify buffer records have non-zero CFVs
        //
        // To run: cargo test -p rebel self_play::tests::test_self_play_training_loop_integration -- --ignored
        // Requires: a trained value network model
    }
}
