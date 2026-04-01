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

use std::sync::Arc;

use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::blueprint_sampler::{deal_hand, play_preflop_under_blueprint};
use crate::inference_server::InferenceHandle;
use crate::pbs::{combo_index, Pbs, NUM_COMBOS};
use crate::replay_buffer::{ReplayBuffer, ReplayEntry};
use crate::solver::{solve_depth_limited_pbs, SolveConfig, SolveResult};
use cfvnet::model::network::INPUT_SIZE;
use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::game_tree::GameTree;
use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
use poker_solver_core::blueprint_v2::LeafEvaluator;
use poker_solver_core::poker::Card as RsPokerCard;

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

// ---------------------------------------------------------------------------
// InferenceHandle → LeafEvaluator adapter
// ---------------------------------------------------------------------------

/// Adapter that wraps an [`InferenceHandle`] to implement the
/// [`LeafEvaluator`] trait. This bridges the self-play loop to the
/// inference server for boundary evaluation during subgame solving.
///
/// When Task 3 (`solve_subgame_iterative`) is complete, this adapter will
/// be replaced by a direct call to that function. For now it allows the
/// self-play loop to use InferenceHandle with the existing solver API.
struct InferenceLeafEvaluator<'a> {
    handle: &'a InferenceHandle,
}

impl LeafEvaluator for InferenceLeafEvaluator<'_> {
    fn evaluate(
        &self,
        combos: &[[RsPokerCard; 2]],
        board: &[RsPokerCard],
        pot: f64,
        effective_stack: f64,
        oop_range: &[f64],
        ip_range: &[f64],
        traverser: u8,
    ) -> Vec<f64> {
        // Build 2720-element input vector for the inference server.
        let mut input = vec![0.0f32; INPUT_SIZE];

        // Convert solver-ordered ranges back to canonical 1326 ordering.
        // oop_range and ip_range are in solver combo ordering (same length as combos).
        // We need canonical 1326 ordering for the net.
        let mut canonical_oop = [0.0f32; NUM_COMBOS];
        let mut canonical_ip = [0.0f32; NUM_COMBOS];
        for (i, combo) in combos.iter().enumerate() {
            let idx = rs_poker_pair_to_canonical(combo);
            canonical_oop[idx] = oop_range[i] as f32;
            canonical_ip[idx] = ip_range[i] as f32;
        }

        // Normalize ranges to sum to 1.0.
        let oop_sum: f32 = canonical_oop.iter().sum();
        let ip_sum: f32 = canonical_ip.iter().sum();
        if oop_sum > 0.0 {
            let inv = 1.0 / oop_sum;
            for v in &mut canonical_oop {
                *v *= inv;
            }
        }
        if ip_sum > 0.0 {
            let inv = 1.0 / ip_sum;
            for v in &mut canonical_ip {
                *v *= inv;
            }
        }

        // Fill input: OOP range (0..1326), IP range (1326..2652).
        input[..NUM_COMBOS].copy_from_slice(&canonical_oop);
        input[NUM_COMBOS..2 * NUM_COMBOS].copy_from_slice(&canonical_ip);

        // Board one-hot (2652..2704).
        for card in board {
            let idx = rs_poker_card_to_u8(card) as usize;
            input[2 * NUM_COMBOS + idx] = 1.0;
        }

        // Rank presence (2704..2717).
        for card in board {
            let rank = rs_poker_card_to_u8(card) / 4;
            input[2 * NUM_COMBOS + 52 + rank as usize] = 1.0;
        }

        // Pot / 400.0 (position 2717).
        input[2 * NUM_COMBOS + 52 + 13] = pot as f32 / 400.0;
        // Effective stack / 400.0 (position 2718).
        input[2 * NUM_COMBOS + 52 + 13 + 1] = effective_stack as f32 / 400.0;
        // Player indicator (position 2719).
        input[2 * NUM_COMBOS + 52 + 13 + 2] = traverser as f32;

        // Submit to inference server and get 1326 canonical CFVs.
        let cfvs_canonical = self.handle.evaluate(input);

        // Map canonical 1326 CFVs back to solver combo ordering.
        combos
            .iter()
            .map(|combo| {
                let idx = rs_poker_pair_to_canonical(combo);
                cfvs_canonical[idx] as f64
            })
            .collect()
    }
}

/// Convert an rs_poker Card pair to canonical combo index (0..1325).
fn rs_poker_pair_to_canonical(pair: &[RsPokerCard; 2]) -> usize {
    let c1 = rs_poker_card_to_u8(&pair[0]);
    let c2 = rs_poker_card_to_u8(&pair[1]);
    combo_index(c1, c2)
}

/// Convert an rs_poker Card to the u8 encoding (4*rank + suit).
fn rs_poker_card_to_u8(card: &RsPokerCard) -> u8 {
    use poker_solver_core::poker::{Suit, Value};

    let rank = match card.value {
        Value::Two => 0u8,
        Value::Three => 1,
        Value::Four => 2,
        Value::Five => 3,
        Value::Six => 4,
        Value::Seven => 5,
        Value::Eight => 6,
        Value::Nine => 7,
        Value::Ten => 8,
        Value::Jack => 9,
        Value::Queen => 10,
        Value::King => 11,
        Value::Ace => 12,
    };
    let suit = match card.suit {
        Suit::Club => 0u8,
        Suit::Diamond => 1,
        Suit::Heart => 2,
        Suit::Spade => 3,
    };
    4 * rank + suit
}

// ---------------------------------------------------------------------------
// build_training_input — encode PBS → 2720-float net input
// ---------------------------------------------------------------------------

/// Encode a PBS and player indicator into a 2720-element input vector
/// suitable for the CfvNet inference server.
///
/// Layout (matches cfvnet):
/// - `[0..1326]`: OOP reach probabilities (normalized to sum to 1.0)
/// - `[1326..2652]`: IP reach probabilities (normalized)
/// - `[2652..2704]`: board card one-hot (52 positions)
/// - `[2704..2717]`: rank presence (13 positions)
/// - `[2717]`: pot / 400.0
/// - `[2718]`: effective_stack / 400.0
/// - `[2719]`: player indicator (0.0 or 1.0)
pub fn build_training_input(pbs: &Pbs, player: u8) -> Vec<f32> {
    let mut input = vec![0.0f32; INPUT_SIZE];

    // OOP range: normalize to sum to 1.0.
    let oop_sum: f32 = pbs.reach_probs[0].iter().sum();
    if oop_sum > 0.0 {
        let inv = 1.0 / oop_sum;
        for i in 0..NUM_COMBOS {
            input[i] = pbs.reach_probs[0][i] * inv;
        }
    }

    // IP range: normalize to sum to 1.0.
    let ip_sum: f32 = pbs.reach_probs[1].iter().sum();
    if ip_sum > 0.0 {
        let inv = 1.0 / ip_sum;
        for i in 0..NUM_COMBOS {
            input[NUM_COMBOS + i] = pbs.reach_probs[1][i] * inv;
        }
    }

    // Board one-hot: positions 2652..2704.
    for &card in &pbs.board {
        input[2 * NUM_COMBOS + card as usize] = 1.0;
    }

    // Rank presence: positions 2704..2717.
    for &card in &pbs.board {
        let rank = card / 4;
        input[2 * NUM_COMBOS + 52 + rank as usize] = 1.0;
    }

    // Pot: position 2717.
    input[2 * NUM_COMBOS + 52 + 13] = pbs.pot as f32 / 400.0;

    // Effective stack: position 2718.
    input[2 * NUM_COMBOS + 52 + 13 + 1] = pbs.effective_stack as f32 / 400.0;

    // Player indicator: position 2719.
    input[2 * NUM_COMBOS + 52 + 13 + 2] = player as f32;

    input
}

/// Street identifiers for the self-play state machine.
///
/// We use our own simple enum here rather than importing the core Street
/// enum, to keep the self-play logic self-contained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Street {
    #[allow(dead_code)]
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

/// The sequence of postflop streets (preflop is handled by blueprint).
const STREET_ORDER: [Street; 3] = [
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
    handle: &InferenceHandle,
    solve_config: &SolveConfig,
    sp_config: &SelfPlayConfig,
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    buckets: &AllBuckets,
    rng: &mut R,
) -> Vec<TrainingExample> {
    let mut examples = Vec::new();
    let deal = deal_hand(rng);

    // --- Play preflop under blueprint policy ---
    let preflop_result = match play_preflop_under_blueprint(
        strategy, tree, buckets, &deal,
        sp_config.initial_stack, sp_config.small_blind, sp_config.big_blind,
        rng,
    ) {
        Some(r) => r,
        None => return examples, // hand ended preflop (fold)
    };

    // Use preflop-updated state for postflop solving
    let pot = preflop_result.pot;
    let effective_stack = preflop_result.effective_stack;
    let mut reach_probs = preflop_result.reach_probs;

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

        // Solve the subgame at this PBS via the inference server.
        let evaluator = InferenceLeafEvaluator { handle };
        let solve_result = match solve_depth_limited_pbs(&pbs, solve_config, &evaluator) {
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
        Street::Preflop => panic!("preflop not in STREET_ORDER"),
        Street::Flop => 0,
        Street::Turn => 1,
        Street::River => 2,
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
/// examples at each street boundary. Training examples are pushed to the
/// replay buffer as `ReplayEntry` records (one per player per PBS).
///
/// Returns the total number of training examples generated.
pub fn self_play_training_loop(
    handle: &InferenceHandle,
    solve_config: &SolveConfig,
    sp_config: &SelfPlayConfig,
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    buckets: &AllBuckets,
    replay_buffer: &Arc<ReplayBuffer>,
) -> usize {
    let total_examples = std::sync::atomic::AtomicUsize::new(0);
    let hands_done = std::sync::atomic::AtomicUsize::new(0);
    let start = std::time::Instant::now();

    eprintln!("Starting self-play: {} hands, {} CFR iters/solve, parallel workers",
        sp_config.num_hands, sp_config.cfr_iterations);

    // Parallel self-play: each thread gets its own RNG seeded deterministically.
    // All threads share the inference handle (which batches GPU calls).
    (0..sp_config.num_hands).into_par_iter().for_each(|hand_idx| {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(
            sp_config.seed.wrapping_add(hand_idx as u64),
        );

        let examples = play_self_play_hand(
            handle, solve_config, sp_config,
            strategy, tree, buckets,
            &mut rng,
        );

        // Push each training example to the replay buffer (one entry per player).
        for example in &examples {
            for player in 0..2u8 {
                let input = build_training_input(&example.pbs, player);
                let target = example.cfvs[player as usize].to_vec();
                replay_buffer.push(ReplayEntry { input, target });
            }
        }
        handle.notify_solve_complete();
        total_examples.fetch_add(examples.len(), std::sync::atomic::Ordering::Relaxed);
        let done = hands_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

        // Progress logging every 100 hands.
        if done % 100 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = done as f64 / elapsed;
            let remaining = (sp_config.num_hands - done) as f64 / rate;
            let examples_so_far = total_examples.load(std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "Self-play: {}/{} hands, {} examples, {:.1} hands/s, buffer={}, eta {:.0}s",
                done,
                sp_config.num_hands,
                examples_so_far,
                rate,
                replay_buffer.len(),
                remaining,
            );
        }
    });

    total_examples.load(std::sync::atomic::Ordering::Relaxed)
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
    fn test_street_order_starts_at_flop() {
        assert_eq!(STREET_ORDER.len(), 3);
        assert_eq!(STREET_ORDER[0], Street::Flop);
        assert_eq!(STREET_ORDER[1], Street::Turn);
        assert_eq!(STREET_ORDER[2], Street::River);
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
        assert_eq!(street_index(Street::Flop), 0);
        assert_eq!(street_index(Street::Turn), 1);
        assert_eq!(street_index(Street::River), 2);
    }

    #[test]
    #[should_panic(expected = "preflop not in STREET_ORDER")]
    fn test_street_index_preflop_panics() {
        street_index(Street::Preflop);
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

    // -----------------------------------------------------------------------
    // build_training_input tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_training_input_length() {
        let board = vec![0, 4, 8, 12, 16]; // 2c, 3c, 4c, 5c, 6c
        let pbs = Pbs::new_uniform(board, 100, 200);
        let input = build_training_input(&pbs, 0);
        assert_eq!(input.len(), INPUT_SIZE, "input should be {INPUT_SIZE} floats");
    }

    #[test]
    fn test_build_training_input_ranges_normalized() {
        let board = vec![0, 4, 8, 12, 16];
        let pbs = Pbs::new_uniform(board, 200, 400);

        let input = build_training_input(&pbs, 0);

        // OOP range (0..1326) should sum to ~1.0.
        let oop_sum: f32 = input[..NUM_COMBOS].iter().sum();
        assert!(
            (oop_sum - 1.0).abs() < 1e-4,
            "OOP range should sum to ~1.0, got {oop_sum}"
        );

        // IP range (1326..2652) should sum to ~1.0.
        let ip_sum: f32 = input[NUM_COMBOS..2 * NUM_COMBOS].iter().sum();
        assert!(
            (ip_sum - 1.0).abs() < 1e-4,
            "IP range should sum to ~1.0, got {ip_sum}"
        );
    }

    #[test]
    fn test_build_training_input_board_one_hot() {
        let board = vec![0, 4, 8]; // 2c, 3c, 4c (flop)
        let pbs = Pbs::new_uniform(board.clone(), 100, 200);
        let input = build_training_input(&pbs, 0);

        // Board one-hot region: 2652..2704.
        let board_start = 2 * NUM_COMBOS; // 2652
        for card in 0..52usize {
            let expected = if board.contains(&(card as u8)) {
                1.0
            } else {
                0.0
            };
            assert_eq!(
                input[board_start + card], expected,
                "board one-hot mismatch at card {card}"
            );
        }
    }

    #[test]
    fn test_build_training_input_rank_presence() {
        let board = vec![0, 4, 8]; // 2c(rank 0), 3c(rank 1), 4c(rank 2)
        let pbs = Pbs::new_uniform(board, 100, 200);
        let input = build_training_input(&pbs, 0);

        // Rank presence region: 2704..2717.
        let rank_start = 2 * NUM_COMBOS + 52; // 2704
        for rank in 0..13usize {
            let expected = if rank <= 2 { 1.0 } else { 0.0 };
            assert_eq!(
                input[rank_start + rank], expected,
                "rank presence mismatch at rank {rank}"
            );
        }
    }

    #[test]
    fn test_build_training_input_pot_stack_player() {
        let board = vec![0, 4, 8, 12, 16];
        let pbs = Pbs::new_uniform(board, 200, 400);

        let input_oop = build_training_input(&pbs, 0);
        let input_ip = build_training_input(&pbs, 1);

        // Pot: position 2717 = pot / 400.0 = 200 / 400 = 0.5.
        assert!(
            (input_oop[2717] - 0.5).abs() < 1e-6,
            "pot should be 0.5, got {}",
            input_oop[2717]
        );

        // Stack: position 2718 = stack / 400.0 = 400 / 400 = 1.0.
        assert!(
            (input_oop[2718] - 1.0).abs() < 1e-6,
            "stack should be 1.0, got {}",
            input_oop[2718]
        );

        // Player indicator: position 2719.
        assert_eq!(input_oop[2719], 0.0, "OOP player indicator should be 0.0");
        assert_eq!(input_ip[2719], 1.0, "IP player indicator should be 1.0");
    }

    #[test]
    fn test_build_training_input_blocked_combos_zero() {
        let board = vec![0, 4, 8, 12, 16]; // 2c, 3c, 4c, 5c, 6c
        let pbs = Pbs::new_uniform(board, 100, 200);
        let input = build_training_input(&pbs, 0);

        // Combo containing card 0 (2c, which is on the board) should be 0.
        let blocked_idx = combo_index(0, 1);
        assert_eq!(
            input[blocked_idx], 0.0,
            "blocked combo should have 0 reach"
        );
    }

    // -----------------------------------------------------------------------
    // rs_poker_card_to_u8 test
    // -----------------------------------------------------------------------

    #[test]
    fn test_rs_poker_card_to_u8_roundtrip() {
        use poker_solver_core::poker::{Suit, Value};

        // 2c = 0
        let card = RsPokerCard::new(Value::Two, Suit::Club);
        assert_eq!(rs_poker_card_to_u8(&card), 0);

        // As = 51
        let card = RsPokerCard::new(Value::Ace, Suit::Spade);
        assert_eq!(rs_poker_card_to_u8(&card), 51);

        // Kh = 4*11 + 2 = 46
        let card = RsPokerCard::new(Value::King, Suit::Heart);
        assert_eq!(rs_poker_card_to_u8(&card), 46);

        // 8d = 4*6 + 1 = 25
        let card = RsPokerCard::new(Value::Eight, Suit::Diamond);
        assert_eq!(rs_poker_card_to_u8(&card), 25);
    }

    // -----------------------------------------------------------------------
    // Integration tests (require running inference server)
    // -----------------------------------------------------------------------

    #[test]
    #[ignore] // Requires a running InferenceHandle (e.g., GPU inference server)
    fn test_play_self_play_hand_integration() {
        // This test would:
        // 1. Create an InferenceHandle connected to an inference server
        // 2. Configure SolveConfig and SelfPlayConfig
        // 3. Call play_self_play_hand
        // 4. Verify that training examples are produced
        // 5. Verify that CFVs are non-zero for non-blocked combos
        //
        // To run: cargo test -p rebel self_play::tests::test_play_self_play_hand_integration -- --ignored
        // Requires: a running inference server with a trained value network
    }

    #[test]
    #[ignore] // Requires a running InferenceHandle (e.g., GPU inference server)
    fn test_self_play_training_loop_integration() {
        // This test would:
        // 1. Create an InferenceHandle connected to an inference server
        // 2. Configure SolveConfig and SelfPlayConfig
        // 3. Create a ReplayBuffer
        // 4. Call self_play_training_loop with a small num_hands
        // 5. Verify entries were pushed to the replay buffer
        // 6. Verify replay entries have correct input/target dimensions
        //
        // To run: cargo test -p rebel self_play::tests::test_self_play_training_loop_integration -- --ignored
        // Requires: a running inference server with a trained value network
    }
}
