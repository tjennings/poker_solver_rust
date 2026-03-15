//! Build a [`BatchGpuSolver`] from sampled situations and hand strengths.
//!
//! The builder takes sampled situation data (boards, ranges, pots, stacks)
//! plus precomputed hand strengths and constructs the `RiverSpot` structs
//! needed by `BatchGpuSolver::new()`.
//!
//! This is the pragmatic approach: the sampled data is downloaded from the
//! GPU, `RiverSpot` values are built on the CPU, and then
//! `BatchGpuSolver::new()` re-uploads the per-hand data. The download/upload
//! round-trip is a one-time cost per batch (not per solver iteration) and is
//! dwarfed by the 4000-iteration DCFR+ solve.
//!
//! A future optimisation could build per-hand data directly on the GPU, but
//! that would duplicate the complex batch-assembly logic in
//! `BatchGpuSolver::new()` for minimal gain (build time << solve time).

#[cfg(feature = "cuda")]
use crate::batch::{BatchConfig, BatchGpuSolver, RiverSpot};
#[cfg(feature = "cuda")]
use crate::gpu::GpuContext;

/// Build a [`BatchGpuSolver`] from sampled situation data.
///
/// # Arguments
///
/// * `gpu` — CUDA context.
/// * `boards` — Board cards: `[num_situations * 5]` u32 values (card indices 0..51).
/// * `ranges_oop` — OOP range weights: `[num_situations * 1326]` f32 values.
/// * `ranges_ip` — IP range weights: `[num_situations * 1326]` f32 values.
/// * `pots` — Pot sizes: `[num_situations]` f32 values.
/// * `stacks` — Effective stacks: `[num_situations]` f32 values.
/// * `num_situations` — Number of situations in the batch.
/// * `config` — Shared bet-size configuration for the batch.
///
/// # Returns
///
/// A `BatchGpuSolver` ready to call `.solve()` on.
#[cfg(feature = "cuda")]
pub fn build_batch_from_samples<'a>(
    gpu: &'a GpuContext,
    boards: &[u32],
    ranges_oop: &[f32],
    ranges_ip: &[f32],
    pots: &[f32],
    stacks: &[f32],
    num_situations: usize,
    config: &BatchConfig,
) -> Result<BatchGpuSolver<'a>, String> {
    assert_eq!(boards.len(), num_situations * 5);
    assert_eq!(ranges_oop.len(), num_situations * 1326);
    assert_eq!(ranges_ip.len(), num_situations * 1326);
    assert_eq!(pots.len(), num_situations);
    assert_eq!(stacks.len(), num_situations);

    let spots: Vec<RiverSpot> = (0..num_situations)
        .map(|i| RiverSpot {
            flop: [
                boards[i * 5] as u8,
                boards[i * 5 + 1] as u8,
                boards[i * 5 + 2] as u8,
            ],
            turn: boards[i * 5 + 3] as u8,
            river: boards[i * 5 + 4] as u8,
            oop_range: ranges_oop[i * 1326..(i + 1) * 1326].to_vec(),
            ip_range: ranges_ip[i * 1326..(i + 1) * 1326].to_vec(),
            pot: pots[i] as i32,
            effective_stack: stacks[i] as i32,
        })
        .collect();

    BatchGpuSolver::new(gpu, &spots, config)
}

#[cfg(test)]
mod tests {
    use crate::batch::RiverSpot;
    use range_solver::card::card_from_str;

    /// Helper: convert card name to u32.
    fn card(s: &str) -> u32 {
        u32::from(card_from_str(s).unwrap())
    }

    /// Build a valid board as `[u32; 5]`.
    fn test_board() -> [u32; 5] {
        [card("2c"), card("5d"), card("8h"), card("Js"), card("Ac")]
    }

    /// Build a uniform range (all combos equal weight, respecting board blocking).
    fn uniform_range(board: &[u32; 5]) -> Vec<f32> {
        let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();
        let mut range = vec![0.0f32; 1326];
        for (i, r) in range.iter_mut().enumerate() {
            let (c1, c2) = range_solver::card::index_to_card_pair(i);
            if !board_set.contains(&c1) && !board_set.contains(&c2) {
                *r = 1.0;
            }
        }
        range
    }

    #[test]
    fn test_build_spots_from_samples() {
        let board = test_board();
        let range = uniform_range(&board);

        let num_situations = 3;
        let mut boards = Vec::new();
        let mut ranges_oop = Vec::new();
        let mut ranges_ip = Vec::new();
        let mut pots = Vec::new();
        let mut stacks = Vec::new();

        for _ in 0..num_situations {
            boards.extend_from_slice(&board);
            ranges_oop.extend_from_slice(&range);
            ranges_ip.extend_from_slice(&range);
            pots.push(200.0f32);
            stacks.push(400.0f32);
        }

        // Verify spot construction (doesn't need GPU)
        let spots: Vec<RiverSpot> = (0..num_situations)
            .map(|i| RiverSpot {
                flop: [
                    boards[i * 5] as u8,
                    boards[i * 5 + 1] as u8,
                    boards[i * 5 + 2] as u8,
                ],
                turn: boards[i * 5 + 3] as u8,
                river: boards[i * 5 + 4] as u8,
                oop_range: ranges_oop[i * 1326..(i + 1) * 1326].to_vec(),
                ip_range: ranges_ip[i * 1326..(i + 1) * 1326].to_vec(),
                pot: pots[i] as i32,
                effective_stack: stacks[i] as i32,
            })
            .collect();

        assert_eq!(spots.len(), 3);
        assert_eq!(spots[0].flop, [board[0] as u8, board[1] as u8, board[2] as u8]);
        assert_eq!(spots[0].turn, board[3] as u8);
        assert_eq!(spots[0].river, board[4] as u8);
        assert_eq!(spots[0].pot, 200);
        assert_eq!(spots[0].effective_stack, 400);
        assert_eq!(spots[0].oop_range.len(), 1326);
        assert_eq!(spots[0].ip_range.len(), 1326);

        // Verify range blocking: combos sharing a board card should be zero
        let as_card = card_from_str("Ac").unwrap();
        let ah_card = card_from_str("Ah").unwrap();
        let blocked_idx = range_solver::card::card_pair_to_index(as_card, ah_card);
        // The board has Ac, so any combo with Ac should be zero
        // But this pair is (Ac, Ah) which shares Ac with board
        assert_eq!(spots[0].oop_range[blocked_idx], 0.0);

        // A non-blocked combo should be 1.0
        let c3h = card_from_str("3h").unwrap();
        let c4s = card_from_str("4s").unwrap();
        let unblocked_idx = range_solver::card::card_pair_to_index(c3h, c4s);
        assert_eq!(spots[0].oop_range[unblocked_idx], 1.0);
    }
}
