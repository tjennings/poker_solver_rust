// PBS generation pipeline — parallel blueprint sampling to disk buffer.
//
// Plays hands under blueprint policy in parallel, snapshots PBSs at
// street boundaries, converts to BufferRecords, and appends to the
// disk buffer for later CFV solving.

use std::sync::atomic::{AtomicUsize, Ordering};
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
use crate::solver::{solve_depth_limited_pbs, solve_river_pbs, SolveConfig};
use poker_solver_core::blueprint_v2::LeafEvaluator;

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

    let pb = indicatif::ProgressBar::new(num_hands as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{msg} [{bar:40}] {pos}/{len} ({per_sec}, eta {eta})"
        )
        .unwrap()
        .progress_chars("=> "),
    );
    pb.set_message("Generating PBSs");

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

            pb.inc(1);
            pbs_count
        })
        .sum();

    pb.finish_with_message("PBS generation complete");
    total_pbs
}

/// Reconstruct a `Pbs` from a `BufferRecord`.
///
/// Maps BufferRecord fields back to Pbs:
/// - `board` is sliced to `board[..board_card_count]`
/// - `pot` and `effective_stack` are cast from f32 to i32
/// - `reach_probs` are taken from `oop_reach` / `ip_reach`
pub fn buffer_record_to_pbs(rec: &BufferRecord) -> Pbs {
    let board = rec.board[..rec.board_card_count as usize].to_vec();
    let mut reach_probs = Box::new([[0.0f32; NUM_COMBOS]; 2]);
    reach_probs[0] = rec.oop_reach;
    reach_probs[1] = rec.ip_reach;
    Pbs {
        board,
        pot: rec.pot as i32,
        effective_stack: rec.effective_stack as i32,
        reach_probs,
    }
}

/// Solve PBS records in the buffer and fill in their CFVs.
///
/// For each record:
/// 1. Read the `BufferRecord`
/// 2. Convert to a `Pbs` via `buffer_record_to_pbs`
/// 3. Solve:
///    - If `evaluator` is `Some`, uses `solve_depth_limited_pbs` (supports all streets).
///    - If `evaluator` is `None`, uses `solve_river_pbs` (river PBSs only).
/// 4. Write the solved CFVs and game_value back
///
/// Uses rayon for parallel solving across threads.
/// Returns the number of successfully solved records.
pub fn solve_buffer_records(
    buffer: &mut DiskBuffer,
    solve_config: &SolveConfig,
    evaluator: Option<&(dyn LeafEvaluator + Sync)>,
    threads: usize,
) -> usize {
    let total = buffer.len();
    if total == 0 {
        return 0;
    }

    let chunk_size = 1000;
    let solved_count = AtomicUsize::new(0);

    let pb = indicatif::ProgressBar::new(total as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{msg} [{bar:40}] {pos}/{len} ({per_sec}, eta {eta})"
        )
        .unwrap()
        .progress_chars("=> "),
    );
    pb.set_message("Solving PBSs");

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to create rayon thread pool for solving");

    for chunk_start in (0..total).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(total);

        // Read all records in this chunk
        let records: Vec<(usize, BufferRecord)> = (chunk_start..chunk_end)
            .map(|idx| {
                let rec = buffer
                    .read_record(idx)
                    .unwrap_or_else(|e| panic!("failed to read record {idx}: {e}"));
                (idx, rec)
            })
            .collect();

        // Solve in parallel
        let results: Vec<(usize, BufferRecord)> = pool.install(|| {
            records
                .into_par_iter()
                .filter_map(|(idx, mut rec)| {
                    // Skip already-solved records (non-zero game_value)
                    if rec.game_value != 0.0 {
                        pb.inc(1);
                        return None;
                    }

                    let pbs = buffer_record_to_pbs(&rec);

                    let solve_result = if let Some(eval) = evaluator {
                        // Depth-limited solving: supports all streets (3-5 cards).
                        solve_depth_limited_pbs(&pbs, solve_config, eval)
                    } else {
                        // River-only solving: requires exactly 5 board cards.
                        if pbs.board.len() != 5 {
                            pb.inc(1);
                            return None;
                        }
                        solve_river_pbs(&pbs, solve_config)
                    };

                    match solve_result {
                        Ok(result) => {
                            // Fill CFVs based on player perspective
                            if rec.player == 0 {
                                rec.cfvs = result.oop_cfvs;
                                rec.game_value = result.oop_game_value;
                            } else {
                                rec.cfvs = result.ip_cfvs;
                                rec.game_value = result.ip_game_value;
                            }
                            solved_count.fetch_add(1, Ordering::Relaxed);
                            Some((idx, rec))
                        }
                        Err(e) => {
                            eprintln!("  Warning: failed to solve record {idx}: {e}");
                            pb.inc(1);
                            None
                        }
                    }
                })
                .collect()
        });

        // Write solved records back to buffer
        for (idx, rec) in &results {
            buffer
                .write_record(*idx, rec)
                .unwrap_or_else(|e| panic!("failed to write record {idx}: {e}"));
            pb.inc(1);
        }
    }

    let solved = solved_count.load(Ordering::Relaxed);
    pb.finish_with_message(format!("Solved {solved}/{total} records"));
    solved
}

/// Convert a pair of bet size fraction arrays `[bets, raises]` to per-player
/// `BetSizeOptions` with `PotRelative` entries and an all-in option.
fn fractions_to_bet_size_options(
    fractions: &[Vec<f64>; 2],
) -> [range_solver::bet_size::BetSizeOptions; 2] {
    use range_solver::bet_size::{BetSize, BetSizeOptions};

    // OOP: fractions[0] = bets, fractions[1] = raises
    let build_one = |bet_fracs: &[f64], raise_fracs: &[f64]| {
        let mut bets: Vec<BetSize> = bet_fracs
            .iter()
            .map(|&f| BetSize::PotRelative(f))
            .collect();
        bets.push(BetSize::AllIn);
        bets.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let mut raises: Vec<BetSize> = raise_fracs
            .iter()
            .map(|&f| BetSize::PotRelative(f))
            .collect();
        raises.push(BetSize::AllIn);
        raises.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        BetSizeOptions {
            bet: bets,
            raise: raises,
        }
    };

    // Both players use the same bet/raise structure for simplicity.
    // fractions[0] = bet sizes, fractions[1] = raise sizes.
    let opts = build_one(&fractions[0], &fractions[1]);
    [opts.clone(), opts]
}

/// Build a `SolveConfig` from the rebel `SeedConfig`.
///
/// Converts bet size fractions for each street to `BetSizeOptions` with
/// `PotRelative` entries, adding an all-in option.
pub fn build_solve_config(seed: &crate::config::SeedConfig) -> SolveConfig {
    let river_sizes = fractions_to_bet_size_options(&seed.bet_sizes.river);
    let turn_sizes = fractions_to_bet_size_options(&seed.bet_sizes.turn);
    let flop_sizes = fractions_to_bet_size_options(&seed.bet_sizes.flop);

    SolveConfig {
        bet_sizes: river_sizes[0].clone(),
        turn_bet_sizes: Some(turn_sizes),
        flop_bet_sizes: Some(flop_sizes),
        solver_iterations: seed.solver_iterations,
        target_exploitability: seed.target_exploitability,
        add_allin_threshold: seed.add_allin_threshold,
        force_allin_threshold: seed.force_allin_threshold,
    }
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

    #[test]
    fn test_buffer_record_to_pbs_roundtrip() {
        // Create a PBS, convert to BufferRecord, convert back, verify fields match.
        let board = vec![0, 4, 8, 12, 16]; // 2c, 3c, 4c, 5c, 6c (5-card river board)
        let pbs = Pbs::new_uniform(board.clone(), 200, 350);

        let rec = pbs_to_buffer_record(&pbs, 0);
        let pbs2 = buffer_record_to_pbs(&rec);

        // Board should match
        assert_eq!(pbs2.board, board);

        // Pot and effective stack should match (after f32 roundtrip)
        assert_eq!(pbs2.pot, 200);
        assert_eq!(pbs2.effective_stack, 350);

        // Reach probs should match exactly (both go through [f32; 1326])
        assert_eq!(pbs2.reach_probs[0], pbs.reach_probs[0]);
        assert_eq!(pbs2.reach_probs[1], pbs.reach_probs[1]);

        // Verify blocked combos are still zero
        let blocked_idx = combo_index(0, 1); // card 0 is on board
        assert_eq!(pbs2.reach_probs[0][blocked_idx], 0.0);
        assert_eq!(pbs2.reach_probs[1][blocked_idx], 0.0);

        // Verify non-blocked combos are still 1.0
        let valid_idx = combo_index(1, 2); // cards 1,2 not on board
        assert_eq!(pbs2.reach_probs[0][valid_idx], 1.0);
        assert_eq!(pbs2.reach_probs[1][valid_idx], 1.0);
    }

    #[test]
    fn test_buffer_record_to_pbs_short_board() {
        // 3-card board — padded to 5 in record, should round-trip back to 3
        let board = vec![10, 20, 30];
        let pbs = Pbs::new_uniform(board.clone(), 50, 400);

        let rec = pbs_to_buffer_record(&pbs, 1);
        let pbs2 = buffer_record_to_pbs(&rec);

        assert_eq!(pbs2.board, board);
        assert_eq!(pbs2.pot, 50);
        assert_eq!(pbs2.effective_stack, 400);
    }

    #[test]
    fn test_build_solve_config() {
        use crate::config::{BetSizeConfig, SeedConfig};
        use range_solver::bet_size::BetSize;

        let seed = SeedConfig {
            num_hands: 1000,
            seed: 42,
            threads: 4,
            solver_iterations: 512,
            target_exploitability: 0.01,
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            bet_sizes: BetSizeConfig {
                flop: [vec![0.33, 0.67], vec![0.33, 0.67]],
                turn: [vec![0.5, 0.75], vec![0.5, 0.75]],
                river: [vec![0.5, 1.0], vec![0.75]],
            },
        };

        let sc = build_solve_config(&seed);

        assert_eq!(sc.solver_iterations, 512);
        assert!((sc.target_exploitability - 0.01).abs() < 1e-6);
        assert!((sc.add_allin_threshold - 1.5).abs() < 1e-9);
        assert!((sc.force_allin_threshold - 0.15).abs() < 1e-9);

        // Bets should contain PotRelative(0.5), PotRelative(1.0), AllIn — sorted
        assert!(sc.bet_sizes.bet.contains(&BetSize::PotRelative(0.5)));
        assert!(sc.bet_sizes.bet.contains(&BetSize::PotRelative(1.0)));
        assert!(sc.bet_sizes.bet.contains(&BetSize::AllIn));

        // Raises should contain PotRelative(0.75), AllIn — sorted
        assert!(sc.bet_sizes.raise.contains(&BetSize::PotRelative(0.75)));
        assert!(sc.bet_sizes.raise.contains(&BetSize::AllIn));
    }

    #[test]
    #[ignore] // Requires solving (takes ~1s per PBS)
    fn test_solve_buffer_records_integration() {
        use crate::solver::SolveConfig;
        use range_solver::bet_size::BetSizeOptions;

        let dir = tempfile::tempdir().unwrap();
        let buffer_path = dir.path().join("test_solve.bin");

        let mut buffer = DiskBuffer::create(&buffer_path, 100).unwrap();

        // Create a few river PBSs with uniform reach and write to buffer
        let boards = vec![
            vec![0, 4, 8, 12, 16],  // 2c,3c,4c,5c,6c
            vec![51, 46, 40, 25, 7], // As,Kh,Qd,8d,3s
        ];

        for board in &boards {
            let pbs = Pbs::new_uniform(board.clone(), 100, 100);
            let rec_oop = pbs_to_buffer_record(&pbs, 0);
            let rec_ip = pbs_to_buffer_record(&pbs, 1);
            buffer.append(&rec_oop).unwrap();
            buffer.append(&rec_ip).unwrap();
        }
        assert_eq!(buffer.len(), 4);

        let solve_config = SolveConfig {
            bet_sizes: BetSizeOptions::try_from(("50%,a", "")).expect("valid bet sizes"),
            turn_bet_sizes: None,
            flop_bet_sizes: None,
            solver_iterations: 100,
            target_exploitability: 0.05,
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
        };

        let solved = solve_buffer_records(&mut buffer, &solve_config, None, 2);
        assert_eq!(solved, 4, "expected all 4 records to be solved");

        // Verify CFVs are non-zero for at least some combos
        for i in 0..4 {
            let rec = buffer.read_record(i).unwrap();
            let has_nonzero = rec.cfvs.iter().any(|&v| v != 0.0);
            assert!(
                has_nonzero,
                "record {} should have at least some non-zero CFVs after solving",
                i
            );
            assert!(
                rec.game_value.is_finite(),
                "record {} game_value should be finite, got {}",
                i, rec.game_value
            );
        }
    }
}
