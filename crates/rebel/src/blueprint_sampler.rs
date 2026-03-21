// Blueprint sampler — play hands under blueprint policy, snapshot PBSs at street boundaries.
//
// Generates training data for ReBeL by:
// 1. Dealing random hands
// 2. Traversing the game tree under blueprint strategy
// 3. Snapshotting Public Belief States at each street boundary (Chance node)

use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::game_tree::{GameNode, GameTree, TreeAction};
use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
use poker_solver_core::blueprint_v2::Street;
use poker_solver_core::poker::{Card, Suit, Value};
use rand::Rng;
use range_solver::card::index_to_card_pair;

use crate::belief_update::{sample_action, update_reach};
use crate::pbs::{Pbs, NUM_COMBOS};

/// A dealt hand: hole cards for two players plus the full 5-card board.
///
/// Card encoding: range-solver format (0-51: `4 * rank + suit`,
/// where rank 0=2..12=A, suit 0=club, 1=diamond, 2=heart, 3=spade).
pub struct Deal {
    /// Hole cards for each player: `[OOP, IP]`.
    pub hole_cards: [[u8; 2]; 2],
    /// Full 5-card board: flop(3) + turn(1) + river(1).
    pub board: [u8; 5],
}

/// Deal a random hand using Fisher-Yates shuffle on the first 9 cards.
///
/// Shuffles 9 cards from a 52-card deck:
/// - Cards 0-1: OOP hole cards
/// - Cards 2-3: IP hole cards
/// - Cards 4-8: Board (flop + turn + river)
pub fn deal_hand<R: Rng>(rng: &mut R) -> Deal {
    let mut deck: Vec<u8> = (0..52).collect();

    // Fisher-Yates: shuffle first 9 positions
    for i in 0..9 {
        let j = rng.random_range(i..52);
        deck.swap(i, j);
    }

    Deal {
        hole_cards: [[deck[0], deck[1]], [deck[2], deck[3]]],
        board: [deck[4], deck[5], deck[6], deck[7], deck[8]],
    }
}

/// Convert a range-solver card ID (0-51: `4*rank + suit`) to an `rs_poker::core::Card`.
///
/// Range-solver suit encoding: club=0, diamond=1, heart=2, spade=3.
/// rs_poker suit encoding: Spade=0, Club=1, Heart=2, Diamond=3.
fn rs_id_to_card(id: u8) -> Card {
    let rank = id / 4;
    let suit_id = id % 4;
    let value = Value::from(rank);
    let suit = match suit_id {
        0 => Suit::Club,
        1 => Suit::Diamond,
        2 => Suit::Heart,
        3 => Suit::Spade,
        _ => unreachable!(),
    };
    Card::new(value, suit)
}

/// Precompute the bucket for each of 1326 combos given a board and street.
///
/// Board-blocked combos get bucket 0 (irrelevant since their reach is 0).
pub fn compute_combo_buckets(
    buckets: &AllBuckets,
    street: Street,
    board: &[u8],
) -> [u16; 1326] {
    let mut result = [0u16; 1326];

    // Build a set of board cards for fast blocking check
    let mut board_mask: u64 = 0;
    for &card in board {
        board_mask |= 1u64 << card;
    }

    // Convert board to rs_poker Cards for AllBuckets API
    let rs_board: Vec<Card> = board.iter().map(|&c| rs_id_to_card(c)).collect();

    for (combo_idx, bucket_slot) in result.iter_mut().enumerate() {
        let (c1, c2) = index_to_card_pair(combo_idx);

        // Check if either card is blocked by the board
        let hand_mask: u64 = (1u64 << c1) | (1u64 << c2);
        if hand_mask & board_mask != 0 {
            *bucket_slot = 0;
            continue;
        }

        let hole = [rs_id_to_card(c1), rs_id_to_card(c2)];
        *bucket_slot = buckets.get_bucket(street, hole, &rs_board);
    }

    result
}

/// Compute how invested amounts change after taking an action.
///
/// Returns the new invested amounts for both players.
/// `actor` is the index (0 or 1) of the player taking the action.
/// `starting_stack` is as a f64 (matching the tree's convention).
fn apply_action(
    action: &TreeAction,
    invested: [f64; 2],
    actor: usize,
    starting_stack: f64,
) -> [f64; 2] {
    let mut new_invested = invested;
    match action {
        TreeAction::Fold | TreeAction::Check => {
            // No change to invested amounts
        }
        TreeAction::Call => {
            // Match the opponent's investment
            new_invested[actor] = invested[1 - actor];
        }
        TreeAction::Bet(amount) | TreeAction::Raise(amount) => {
            // Amount is the total invested by actor (not incremental)
            new_invested[actor] = *amount;
        }
        TreeAction::AllIn => {
            new_invested[actor] = starting_stack;
        }
    }
    new_invested
}

/// Board slice visible at a given street.
fn board_for_street(board: &[u8; 5], street: Street) -> &[u8] {
    match street {
        Street::Preflop => &[],
        Street::Flop => &board[..3],
        Street::Turn => &board[..4],
        Street::River => &board[..5],
    }
}

/// Play one hand under blueprint policy, returning PBS snapshots at street boundaries.
///
/// For each decision node:
/// 1. Look up the bucket for the dealt hand's actual cards
/// 2. Get action probabilities from blueprint strategy
/// 3. Sample an action
/// 4. Update ALL combos' reach probabilities for the acting player
///
/// At each Chance node (street boundary), snapshot the PBS.
#[allow(clippy::too_many_arguments)]
pub fn play_hand<R: Rng>(
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    buckets: &AllBuckets,
    deal: &Deal,
    initial_stack: i32,
    small_blind: i32,
    big_blind: i32,
    rng: &mut R,
) -> Vec<Pbs> {
    let starting_stack = f64::from(initial_stack);

    // Pre-build the decision_idx mapping from arena node index
    let decision_idx_map = tree.decision_index_map();

    // Initialize reach probabilities (uniform, then zero board-blocked for preflop)
    let mut reach = [[1.0f32; NUM_COMBOS]; 2];

    // Zero out combos blocked by each player's dealt cards
    // (NOT needed — reach probs represent the range over all possible hands,
    // not conditioned on the actual deal. Board blocking happens at snapshot time.)

    let mut snapshots = Vec::new();

    // Start traversal
    traverse(
        strategy,
        tree,
        buckets,
        deal,
        &decision_idx_map,
        &mut reach,
        &mut snapshots,
        tree.root,
        [f64::from(small_blind), f64::from(big_blind)],
        starting_stack,
        initial_stack,
        rng,
    );

    snapshots
}

/// Recursive tree traversal under blueprint policy.
#[allow(clippy::too_many_arguments)]
fn traverse<R: Rng>(
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    buckets: &AllBuckets,
    deal: &Deal,
    decision_idx_map: &[u32],
    reach: &mut [[f32; NUM_COMBOS]; 2],
    snapshots: &mut Vec<Pbs>,
    node_idx: u32,
    invested: [f64; 2],
    starting_stack: f64,
    initial_stack: i32,
    rng: &mut R,
) {
    match &tree.nodes[node_idx as usize] {
        GameNode::Terminal { .. } => {
            // Hand is over, nothing to snapshot
        }

        GameNode::Chance { next_street, child } => {
            // Street boundary — snapshot the PBS for the NEXT street's board
            let next_board = board_for_street(&deal.board, *next_street);
            let pot = (invested[0] + invested[1]).round() as i32;
            let max_invested = invested[0].max(invested[1]);
            let effective_stack = initial_stack - max_invested.round() as i32;

            // Build the PBS snapshot
            let mut pbs = Pbs {
                board: next_board.to_vec(),
                pot,
                effective_stack,
                reach_probs: Box::new(*reach),
            };
            // Zero out board-blocked combos
            pbs.zero_blocked_combos();

            snapshots.push(pbs);

            // Continue traversal into the next street
            traverse(
                strategy,
                tree,
                buckets,
                deal,
                decision_idx_map,
                reach,
                snapshots,
                *child,
                invested,
                starting_stack,
                initial_stack,
                rng,
            );
        }

        GameNode::Decision {
            player,
            street,
            actions,
            children,
            ..
        } => {
            let player = *player;
            let street = *street;
            let decision_idx = decision_idx_map[node_idx as usize];
            debug_assert_ne!(decision_idx, u32::MAX, "Decision node must have valid decision_idx");

            // Get the actual hole cards bucket for the acting player
            let hole = deal.hole_cards[player as usize];
            let board_slice = board_for_street(&deal.board, street);
            let rs_hole = [rs_id_to_card(hole[0]), rs_id_to_card(hole[1])];
            let rs_board: Vec<Card> = board_slice.iter().map(|&c| rs_id_to_card(c)).collect();
            let actual_bucket = buckets.get_bucket(street, rs_hole, &rs_board);

            // Get action probabilities for the actual hand's bucket
            let action_probs = strategy.get_action_probs(decision_idx as usize, actual_bucket);

            if action_probs.is_empty() || actions.is_empty() {
                return; // Malformed node, shouldn't happen
            }

            // Sample an action based on blueprint probabilities
            let chosen_action_idx = sample_action(action_probs, rng);

            // Build action_probs_per_bucket for reach update.
            // For the acting player, update reach for ALL combos based on their bucket's action probs.
            let combo_buckets = compute_combo_buckets(buckets, street, board_slice);
            let num_buckets = buckets.bucket_counts[street as usize] as usize;
            let num_actions = actions.len();

            let mut action_probs_per_bucket: Vec<Vec<f32>> = Vec::with_capacity(num_buckets);
            for b in 0..num_buckets {
                let probs = strategy.get_action_probs(decision_idx as usize, b as u16);
                if probs.len() == num_actions {
                    action_probs_per_bucket.push(probs.to_vec());
                } else {
                    // Fallback: uniform
                    let uniform = 1.0 / num_actions as f32;
                    action_probs_per_bucket.push(vec![uniform; num_actions]);
                }
            }

            // Update reach probabilities for the acting player
            update_reach(
                &mut reach[player as usize],
                &combo_buckets,
                &action_probs_per_bucket,
                chosen_action_idx,
            );

            // Update invested and recurse
            let new_invested = apply_action(
                &actions[chosen_action_idx],
                invested,
                player as usize,
                starting_stack,
            );

            traverse(
                strategy,
                tree,
                buckets,
                deal,
                decision_idx_map,
                reach,
                snapshots,
                children[chosen_action_idx],
                new_invested,
                starting_stack,
                initial_stack,
                rng,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use range_solver::card::card_pair_to_index;

    #[test]
    fn test_deal_hand_unique_cards() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let deal = deal_hand(&mut rng);

        // Collect all 9 cards
        let mut all_cards: Vec<u8> = Vec::with_capacity(9);
        for player_hole in &deal.hole_cards {
            all_cards.extend_from_slice(player_hole);
        }
        all_cards.extend_from_slice(&deal.board);

        // All 9 cards should be unique
        assert_eq!(all_cards.len(), 9);
        let mut sorted = all_cards.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            9,
            "Expected 9 unique cards, got {} (cards: {:?})",
            sorted.len(),
            all_cards
        );

        // All cards should be in [0, 52)
        for &card in &all_cards {
            assert!(card < 52, "Card {} out of range [0, 52)", card);
        }
    }

    #[test]
    fn test_deal_hand_deterministic() {
        let mut rng1 = ChaCha8Rng::seed_from_u64(12345);
        let mut rng2 = ChaCha8Rng::seed_from_u64(12345);

        let deal1 = deal_hand(&mut rng1);
        let deal2 = deal_hand(&mut rng2);

        assert_eq!(
            deal1.hole_cards, deal2.hole_cards,
            "Same seed must produce same hole cards"
        );
        assert_eq!(
            deal1.board, deal2.board,
            "Same seed must produce same board"
        );
    }

    #[test]
    fn test_deal_hand_different_seeds_differ() {
        let mut rng1 = ChaCha8Rng::seed_from_u64(1);
        let mut rng2 = ChaCha8Rng::seed_from_u64(2);

        let deal1 = deal_hand(&mut rng1);
        let deal2 = deal_hand(&mut rng2);

        // It's statistically near-impossible for two different seeds to produce the same deal
        let same = deal1.hole_cards == deal2.hole_cards && deal1.board == deal2.board;
        assert!(
            !same,
            "Different seeds should produce different deals (astronomically unlikely otherwise)"
        );
    }

    #[test]
    fn test_rs_id_to_card_roundtrip() {
        // Verify the range-solver -> rs_poker conversion is correct
        // by checking known values.

        // 2c = card_id 0: rank 0 (Two), suit 0 (club)
        let c = rs_id_to_card(0);
        assert_eq!(c.value, Value::Two);
        assert_eq!(c.suit, Suit::Club);

        // As = card_id 51: rank 12 (Ace), suit 3 (spade)
        let c = rs_id_to_card(51);
        assert_eq!(c.value, Value::Ace);
        assert_eq!(c.suit, Suit::Spade);

        // Kh = card_id 46: rank 11 (King), suit 2 (heart)
        let c = rs_id_to_card(46);
        assert_eq!(c.value, Value::King);
        assert_eq!(c.suit, Suit::Heart);

        // 3d = card_id 5: rank 1 (Three), suit 1 (diamond)
        let c = rs_id_to_card(5);
        assert_eq!(c.value, Value::Three);
        assert_eq!(c.suit, Suit::Diamond);

        // Verify card_to_rs_id inverse: convert rs_poker Card back to u8
        for id in 0..52u8 {
            let card = rs_id_to_card(id);
            let rank = card.value as u8;
            let suit = match card.suit {
                Suit::Club => 0u8,
                Suit::Diamond => 1,
                Suit::Heart => 2,
                Suit::Spade => 3,
            };
            let reconstructed = 4 * rank + suit;
            assert_eq!(
                reconstructed, id,
                "Card ID {} roundtrip failed: got {}",
                id, reconstructed
            );
        }
    }

    #[test]
    fn test_apply_action_fold() {
        let invested = [5.0, 10.0];
        let result = apply_action(&TreeAction::Fold, invested, 0, 100.0);
        assert_eq!(result, [5.0, 10.0], "Fold should not change invested");
    }

    #[test]
    fn test_apply_action_check() {
        let invested = [10.0, 10.0];
        let result = apply_action(&TreeAction::Check, invested, 1, 100.0);
        assert_eq!(result, [10.0, 10.0], "Check should not change invested");
    }

    #[test]
    fn test_apply_action_call() {
        // Player 0 calls, matching player 1's investment
        let invested = [5.0, 10.0];
        let result = apply_action(&TreeAction::Call, invested, 0, 100.0);
        assert_eq!(result, [10.0, 10.0], "Call should match opponent's investment");

        // Player 1 calls, matching player 0's investment
        let invested = [20.0, 10.0];
        let result = apply_action(&TreeAction::Call, invested, 1, 100.0);
        assert_eq!(result, [20.0, 20.0], "Call should match opponent's investment");
    }

    #[test]
    fn test_apply_action_bet() {
        // Player 0 bets 20 total
        let invested = [10.0, 10.0];
        let result = apply_action(&TreeAction::Bet(20.0), invested, 0, 100.0);
        assert_eq!(
            result,
            [20.0, 10.0],
            "Bet sets actor's invested to the bet amount"
        );
    }

    #[test]
    fn test_apply_action_raise() {
        // Player 1 raises to 30 total
        let invested = [5.0, 10.0];
        let result = apply_action(&TreeAction::Raise(30.0), invested, 1, 100.0);
        assert_eq!(
            result,
            [5.0, 30.0],
            "Raise sets actor's invested to the raise amount"
        );
    }

    #[test]
    fn test_apply_action_allin() {
        let invested = [5.0, 10.0];
        let result = apply_action(&TreeAction::AllIn, invested, 0, 100.0);
        assert_eq!(
            result,
            [100.0, 10.0],
            "AllIn sets actor's invested to starting stack"
        );

        let result2 = apply_action(&TreeAction::AllIn, invested, 1, 100.0);
        assert_eq!(
            result2,
            [5.0, 100.0],
            "AllIn sets actor's invested to starting stack"
        );
    }

    #[test]
    fn test_invested_tracking_sequence() {
        // Simulate a realistic preflop betting sequence:
        // SB posts 0.5, BB posts 1.0
        // SB raises to 2.5 (Raise(2.5))
        // BB calls
        let stack = 100.0;
        let mut invested = [0.5, 1.0];

        // SB raises to 2.5
        invested = apply_action(&TreeAction::Raise(2.5), invested, 0, stack);
        assert!(
            (invested[0] - 2.5).abs() < 1e-9,
            "After SB raise: invested[0]={}, expected 2.5",
            invested[0]
        );
        assert!(
            (invested[1] - 1.0).abs() < 1e-9,
            "After SB raise: invested[1]={}, expected 1.0",
            invested[1]
        );

        // BB calls
        invested = apply_action(&TreeAction::Call, invested, 1, stack);
        assert!(
            (invested[0] - 2.5).abs() < 1e-9,
            "After BB call: invested[0]={}, expected 2.5",
            invested[0]
        );
        assert!(
            (invested[1] - 2.5).abs() < 1e-9,
            "After BB call: invested[1]={}, expected 2.5",
            invested[1]
        );

        let pot = invested[0] + invested[1];
        assert!(
            (pot - 5.0).abs() < 1e-9,
            "Pot should be 5.0, got {}",
            pot
        );
    }

    #[test]
    fn test_invested_tracking_postflop() {
        // Postflop: both players have 5.0 invested from preflop
        // Player 0 bets 10.0 (total invested 10.0 means bet of 5.0 into pot of 10.0)
        // Player 1 raises to 30.0
        // Player 0 calls
        let stack = 100.0;
        let mut invested = [5.0, 5.0];

        // Player 0 bets to 10.0 total
        invested = apply_action(&TreeAction::Bet(10.0), invested, 0, stack);
        assert!(
            (invested[0] - 10.0).abs() < 1e-9 && (invested[1] - 5.0).abs() < 1e-9,
            "After bet: {:?}",
            invested
        );

        // Player 1 raises to 30.0 total
        invested = apply_action(&TreeAction::Raise(30.0), invested, 1, stack);
        assert!(
            (invested[0] - 10.0).abs() < 1e-9 && (invested[1] - 30.0).abs() < 1e-9,
            "After raise: {:?}",
            invested
        );

        // Player 0 calls
        invested = apply_action(&TreeAction::Call, invested, 0, stack);
        assert!(
            (invested[0] - 30.0).abs() < 1e-9 && (invested[1] - 30.0).abs() < 1e-9,
            "After call: {:?}",
            invested
        );

        let pot = invested[0] + invested[1];
        assert!(
            (pot - 60.0).abs() < 1e-9,
            "Pot should be 60.0, got {}",
            pot
        );

        let effective_stack = stack - invested[0].max(invested[1]);
        assert!(
            (effective_stack - 70.0).abs() < 1e-9,
            "Effective stack should be 70.0, got {}",
            effective_stack
        );
    }

    #[test]
    fn test_board_for_street() {
        let board: [u8; 5] = [10, 20, 30, 40, 50];

        assert_eq!(board_for_street(&board, Street::Preflop), &[] as &[u8]);
        assert_eq!(board_for_street(&board, Street::Flop), &[10, 20, 30]);
        assert_eq!(board_for_street(&board, Street::Turn), &[10, 20, 30, 40]);
        assert_eq!(board_for_street(&board, Street::River), &[10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_deal_hand_many_deals_all_valid() {
        // Deal 1000 hands and verify each is valid
        let mut rng = ChaCha8Rng::seed_from_u64(999);
        for i in 0..1000 {
            let deal = deal_hand(&mut rng);

            let mut all_cards: Vec<u8> = Vec::with_capacity(9);
            all_cards.extend_from_slice(&deal.hole_cards[0]);
            all_cards.extend_from_slice(&deal.hole_cards[1]);
            all_cards.extend_from_slice(&deal.board);

            // All in range
            for &c in &all_cards {
                assert!(c < 52, "Deal {}: card {} out of range", i, c);
            }

            // All unique
            let mut sorted = all_cards.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                9,
                "Deal {}: expected 9 unique cards, got {}",
                i,
                sorted.len()
            );
        }
    }

    #[test]
    fn test_compute_combo_buckets_blocks_board_cards_preflop() {
        // For preflop, AllBuckets uses canonical hand index which doesn't
        // need bucket files. We can test the blocking logic at preflop.
        let bucket_counts = [169, 10, 10, 10];
        let buckets = AllBuckets::new(bucket_counts, [None, None, None, None]);

        // Preflop with empty board
        let board: [u8; 0] = [];
        let combo_buckets = compute_combo_buckets(&buckets, Street::Preflop, &board);

        // With no board, nothing should be blocked.
        // All combos should have a valid bucket in [0, 169)
        for combo_idx in 0..NUM_COMBOS {
            assert!(
                combo_buckets[combo_idx] < 169,
                "Combo {} should have valid preflop bucket, got {}",
                combo_idx,
                combo_buckets[combo_idx]
            );
        }

        // Verify specific known hands get correct buckets:
        // AcAs (cards 48, 51) should map to AA bucket
        // AhAs (cards 50, 51) should map to the same AA bucket
        let aa_1 = card_pair_to_index(48, 51); // AcAs
        let aa_2 = card_pair_to_index(50, 51); // AhAs
        assert_eq!(
            combo_buckets[aa_1], combo_buckets[aa_2],
            "All AA combos should map to same canonical bucket"
        );
    }

    #[test]
    fn test_compute_combo_buckets_board_blocking() {
        // Test board-blocking logic directly without needing postflop buckets.
        // Use preflop street but with a fake "board" to verify blocking.
        // Since preflop ignores the board for bucket lookup, we manually test
        // the blocking logic.
        let bucket_counts = [169, 10, 10, 10];
        let buckets = AllBuckets::new(bucket_counts, [None, None, None, None]);

        // With board cards [0, 5, 10], combos containing those should be blocked
        let board = [0u8, 5, 10];

        // Call compute_combo_buckets for preflop — the bucket lookup won't use board,
        // but the blocking logic should still zero out combos with board cards.
        let combo_buckets = compute_combo_buckets(&buckets, Street::Preflop, &board);

        // Combo (0, 1) contains card 0 (board card) -> bucket should be 0
        let blocked_idx = card_pair_to_index(0, 1);
        assert_eq!(
            combo_buckets[blocked_idx], 0,
            "Combo containing board card 0 should have bucket 0"
        );

        // Combo (5, 2) contains card 5 (board card) -> bucket should be 0
        let blocked_idx2 = card_pair_to_index(5, 2);
        assert_eq!(
            combo_buckets[blocked_idx2], 0,
            "Combo containing board card 5 should have bucket 0"
        );

        // Combo (1, 2) does NOT contain any board card -> should have valid bucket
        let unblocked_idx = card_pair_to_index(1, 2);
        assert!(
            combo_buckets[unblocked_idx] < 169,
            "Unblocked combo should have valid bucket, got {}",
            combo_buckets[unblocked_idx]
        );
    }

    #[test]
    #[ignore] // Requires a trained blueprint, game tree, and bucket files to run
    fn test_play_hand_integration() {
        // This test would:
        // 1. Load a trained BlueprintV2Strategy from disk
        // 2. Build or load a GameTree
        // 3. Load AllBuckets with bucket files
        // 4. Deal a hand and play it under blueprint policy
        // 5. Verify that PBS snapshots are generated at street boundaries
        // 6. Verify reach probabilities are updated correctly
        //
        // To run: cargo test -p rebel blueprint_sampler::tests::test_play_hand_integration -- --ignored
        // Requires: a trained blueprint model in the expected path
    }
}
