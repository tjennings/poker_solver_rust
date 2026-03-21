// PBS generation pipeline — parallel blueprint sampling to disk buffer.
//
// Plays hands under blueprint policy in parallel, snapshots PBSs at
// street boundaries, converts to BufferRecords, and appends to the
// disk buffer for later CFV solving.

use std::sync::Mutex;

use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::game_tree::GameTree;
use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use range_solver::card::index_to_card_pair;
use rayon::prelude::*;

use crate::blueprint_sampler::{deal_hand, play_hand};
use crate::config::RebelConfig;
use crate::data_buffer::{BufferRecord, DiskBuffer};
use crate::pbs::{Pbs, NUM_COMBOS};

/// Convert a PBS to a `BufferRecord` for one player's perspective.
///
/// CFVs are zeroed (filled later by the subgame solver).
///
/// - `board` is padded to 5 with `0xFF` for unused slots.
/// - `valid_mask` is 1 for non-blocked combos, 0 for board-blocked.
/// - `oop_reach` / `ip_reach` are copied from `pbs.reach_probs[0]` / `[1]`.
pub fn pbs_to_buffer_record(pbs: &Pbs, player: u8) -> BufferRecord {
    // Pad board to 5 cards with 0xFF for unused slots
    let mut board = [0xFF_u8; 5];
    let card_count = pbs.board.len().min(5);
    board[..card_count].copy_from_slice(&pbs.board[..card_count]);

    // Build the valid_mask: 1 for non-blocked combos, 0 for board-blocked.
    // A combo is board-blocked if either of its two cards appears on the board.
    let mut board_mask: u64 = 0;
    for &card in &pbs.board {
        board_mask |= 1u64 << card;
    }

    let mut valid_mask = [0u8; NUM_COMBOS];
    for (combo_idx, mask_slot) in valid_mask.iter_mut().enumerate() {
        let (c1, c2) = index_to_card_pair(combo_idx);
        let hand_mask: u64 = (1u64 << c1) | (1u64 << c2);
        if hand_mask & board_mask == 0 {
            *mask_slot = 1;
        }
    }

    BufferRecord {
        board,
        board_card_count: card_count as u8,
        pot: pbs.pot as f32,
        effective_stack: pbs.effective_stack as f32,
        player,
        game_value: 0.0,
        oop_reach: pbs.reach_probs[0],
        ip_reach: pbs.reach_probs[1],
        cfvs: [0.0; NUM_COMBOS],
        valid_mask,
    }
}

/// Generate PBS snapshots from blueprint play and write to buffer.
///
/// Plays `num_hands` hands under blueprint policy in parallel,
/// snapshots PBSs at street boundaries, converts to `BufferRecord`s,
/// and appends to the disk buffer.
///
/// Returns total number of PBSs generated.
pub fn generate_pbs(
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    buckets: &AllBuckets,
    config: &RebelConfig,
    buffer: &Mutex<DiskBuffer>,
) -> usize {
    let num_hands = config.seed.num_hands;
    let base_seed = config.seed.seed;
    let initial_stack = config.game.initial_stack;
    let small_blind = config.game.small_blind;
    let big_blind = config.game.big_blind;

    // Parallel iteration: each hand gets a deterministic RNG seeded
    // from (base_seed + hand_idx).  We collect per-hand PBS counts
    // and sum them for the return value.
    let total_pbs: usize = (0..num_hands)
        .into_par_iter()
        .map(|hand_idx| {
            let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(hand_idx as u64));

            // Deal and play one hand under blueprint policy
            let deal = deal_hand(&mut rng);
            let snapshots = play_hand(
                strategy,
                tree,
                buckets,
                &deal,
                initial_stack,
                small_blind,
                big_blind,
                &mut rng,
            );

            // Convert each PBS snapshot to two BufferRecords (one per player)
            // and append to the shared buffer.
            let pbs_count = snapshots.len();
            if pbs_count > 0 {
                let mut buf = buffer.lock().expect("buffer mutex poisoned");
                for pbs in &snapshots {
                    let rec_oop = pbs_to_buffer_record(pbs, 0);
                    let rec_ip = pbs_to_buffer_record(pbs, 1);
                    buf.append(&rec_oop).expect("failed to append OOP record");
                    buf.append(&rec_ip).expect("failed to append IP record");
                }
            }

            pbs_count
        })
        .sum();

    total_pbs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbs::combo_index;

    #[test]
    fn test_pbs_to_buffer_record_basic() {
        // Create a PBS with known values on a 5-card board
        let board = vec![0, 4, 8, 12, 16]; // 2c, 3c, 4c, 5c, 6c
        let pbs = Pbs::new_uniform(board.clone(), 200, 350);

        let rec = pbs_to_buffer_record(&pbs, 0);

        // Board should be copied exactly (already 5 cards, no padding needed)
        assert_eq!(rec.board, [0, 4, 8, 12, 16]);
        assert_eq!(rec.board_card_count, 5);

        // Pot and effective stack
        assert_eq!(rec.pot, 200.0);
        assert_eq!(rec.effective_stack, 350.0);

        // Player
        assert_eq!(rec.player, 0);

        // Game value and CFVs should be zero
        assert_eq!(rec.game_value, 0.0);
        for &cfv in &rec.cfvs {
            assert_eq!(cfv, 0.0);
        }

        // Reach probs should match
        assert_eq!(rec.oop_reach, pbs.reach_probs[0]);
        assert_eq!(rec.ip_reach, pbs.reach_probs[1]);

        // A non-blocked combo should have valid_mask=1 and reach=1.0
        // Cards 1 and 2 (2d, 2h) are not on the board
        let idx = combo_index(1, 2);
        assert_eq!(rec.valid_mask[idx], 1);
        assert_eq!(rec.oop_reach[idx], 1.0);
        assert_eq!(rec.ip_reach[idx], 1.0);

        // A blocked combo should have valid_mask=0 and reach=0.0
        // Card 0 (2c) is on the board
        let blocked_idx = combo_index(0, 1);
        assert_eq!(rec.valid_mask[blocked_idx], 0);
        assert_eq!(rec.oop_reach[blocked_idx], 0.0);
        assert_eq!(rec.ip_reach[blocked_idx], 0.0);
    }

    #[test]
    fn test_pbs_to_buffer_record_player_field() {
        let board = vec![0, 4, 8, 12, 16];
        let pbs = Pbs::new_uniform(board, 100, 200);

        let rec_oop = pbs_to_buffer_record(&pbs, 0);
        let rec_ip = pbs_to_buffer_record(&pbs, 1);

        assert_eq!(rec_oop.player, 0);
        assert_eq!(rec_ip.player, 1);

        // Both should share the same reach probs and valid mask
        assert_eq!(rec_oop.oop_reach, rec_ip.oop_reach);
        assert_eq!(rec_oop.ip_reach, rec_ip.ip_reach);
        assert_eq!(rec_oop.valid_mask, rec_ip.valid_mask);
    }

    #[test]
    fn test_pbs_to_buffer_record_board_padding() {
        // PBS with a 3-card board (flop) — should pad to 5 with 0xFF
        let board = vec![10, 20, 30]; // 3 cards
        let pbs = Pbs::new_uniform(board, 50, 400);

        let rec = pbs_to_buffer_record(&pbs, 0);

        assert_eq!(rec.board[0], 10);
        assert_eq!(rec.board[1], 20);
        assert_eq!(rec.board[2], 30);
        assert_eq!(rec.board[3], 0xFF); // padded
        assert_eq!(rec.board[4], 0xFF); // padded
        assert_eq!(rec.board_card_count, 3);
    }

    #[test]
    fn test_pbs_to_buffer_record_turn_padding() {
        // PBS with a 4-card board (turn) — pad last slot with 0xFF
        let board = vec![10, 20, 30, 40];
        let pbs = Pbs::new_uniform(board, 80, 300);

        let rec = pbs_to_buffer_record(&pbs, 1);

        assert_eq!(rec.board[0], 10);
        assert_eq!(rec.board[1], 20);
        assert_eq!(rec.board[2], 30);
        assert_eq!(rec.board[3], 40);
        assert_eq!(rec.board[4], 0xFF); // padded
        assert_eq!(rec.board_card_count, 4);
    }

    #[test]
    fn test_pbs_to_buffer_record_valid_mask() {
        // Board: cards 0, 1, 2 (2c, 2d, 2h)
        // Any combo containing card 0, 1, or 2 should have mask=0
        // All other combos should have mask=1
        let board = vec![0, 1, 2];
        let pbs = Pbs::new_uniform(board.clone(), 100, 200);

        let rec = pbs_to_buffer_record(&pbs, 0);

        // Build the set of blocked combos independently
        let mut board_mask: u64 = 0;
        for &card in &board {
            board_mask |= 1u64 << card;
        }

        let mut expected_valid = 0usize;
        let mut expected_blocked = 0usize;

        for combo_idx in 0..NUM_COMBOS {
            let (c1, c2) = index_to_card_pair(combo_idx);
            let hand_mask: u64 = (1u64 << c1) | (1u64 << c2);
            if hand_mask & board_mask != 0 {
                assert_eq!(
                    rec.valid_mask[combo_idx], 0,
                    "combo {} (cards {},{}) should be blocked by board {:?}",
                    combo_idx, c1, c2, board
                );
                expected_blocked += 1;
            } else {
                assert_eq!(
                    rec.valid_mask[combo_idx], 1,
                    "combo {} (cards {},{}) should be valid (not blocked by board {:?})",
                    combo_idx, c1, c2, board
                );
                expected_valid += 1;
            }
        }

        // With 3 board cards, each blocks 51 combos but some overlap.
        // 3 cards block combos containing any of them.
        // Number of blocked combos = C(3,2) + 3*49 = 3 + 147 = 150
        // (3 combos among the board cards + each of 3 board cards paired with 49 non-board cards)
        assert_eq!(expected_blocked, 150);
        assert_eq!(expected_valid, NUM_COMBOS - 150);
    }

    #[test]
    fn test_pbs_to_buffer_record_roundtrip_through_serialize() {
        // Verify that the record produced by pbs_to_buffer_record survives
        // serialize/deserialize roundtrip.
        let board = vec![51, 46, 40]; // As, Kh, Qd
        let pbs = Pbs::new_uniform(board, 300, 150);
        let rec = pbs_to_buffer_record(&pbs, 1);

        let mut buf = Vec::new();
        rec.serialize(&mut buf);
        let rec2 = BufferRecord::deserialize(&buf);

        assert_eq!(rec2.board, rec.board);
        assert_eq!(rec2.board_card_count, rec.board_card_count);
        assert_eq!(rec2.pot, rec.pot);
        assert_eq!(rec2.effective_stack, rec.effective_stack);
        assert_eq!(rec2.player, rec.player);
        assert_eq!(rec2.game_value, rec.game_value);
        assert_eq!(rec2.oop_reach, rec.oop_reach);
        assert_eq!(rec2.ip_reach, rec.ip_reach);
        assert_eq!(rec2.cfvs, rec.cfvs);
        assert_eq!(rec2.valid_mask, rec.valid_mask);
    }

    #[test]
    #[ignore] // Requires BlueprintV2Strategy, GameTree, and AllBuckets from disk
    fn test_generate_pbs_integration() {
        // This test would:
        // 1. Load a trained BlueprintV2Strategy
        // 2. Build or load a GameTree
        // 3. Load AllBuckets with cluster files
        // 4. Create a DiskBuffer
        // 5. Call generate_pbs with a small num_hands
        // 6. Verify records were appended to the buffer
        // 7. Read records back and verify fields are populated correctly
        //
        // To run: cargo test -p rebel generate::tests::test_generate_pbs_integration -- --ignored
        // Requires: trained blueprint model and cluster files
    }
}
