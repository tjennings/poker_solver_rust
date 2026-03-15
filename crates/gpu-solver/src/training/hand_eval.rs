//! GPU-native hand strength evaluation for sampled river boards.
//!
//! For each sampled board (5 cards) and each of the 1326 possible hole-card
//! combos, evaluate the 7-card hand strength entirely on the GPU using a
//! CUDA kernel that mirrors the CPU evaluator in `range-solver/src/hand.rs`.
//!
//! The kernel computes the same internal i32 hand value and performs the
//! same binary search against HAND_TABLE, producing identical u16 strength
//! rankings as the CPU version.
//!
//! No data is downloaded to the CPU during evaluation. The combo_cards
//! lookup and HAND_TABLE are uploaded once and cached.

#[cfg(feature = "cuda")]
use crate::gpu::{GpuContext, GpuError};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use range_solver::card::{evaluate_hand_strength, index_to_card_pair, Card};

/// Build the canonical 1326 combo card pairs in index order.
///
/// Combo index `i` maps to `(c1, c2)` via `index_to_card_pair(i)`, where
/// `c1 < c2` and each card is in `0..52`.
fn build_combo_cards() -> Vec<(Card, Card)> {
    (0..1326).map(index_to_card_pair).collect()
}

/// Build the flat `[1326 * 2]` combo_cards array for GPU upload.
///
/// Element `[i * 2]` = card1, `[i * 2 + 1]` = card2 for combo index i.
pub fn build_combo_cards_flat() -> Vec<u32> {
    let combos = build_combo_cards();
    let mut flat = Vec::with_capacity(1326 * 2);
    for (c1, c2) in combos {
        flat.push(c1 as u32);
        flat.push(c2 as u32);
    }
    flat
}

/// Get the HAND_TABLE from range-solver for GPU upload.
///
/// Returns the sorted array of 4824 i32 hand values used for binary search
/// to convert internal hand representations to u16 strength indices.
fn get_hand_table() -> Vec<i32> {
    // Access the hand table from range-solver.
    // We need to use the same table that the CPU evaluator uses.
    range_solver::hand_table_data().to_vec()
}

/// Evaluate hand strengths for all 1326 combos on each board using the CPU.
///
/// This is the original CPU implementation, kept for validation and as a
/// fallback. `boards` contains `num_situations * 5` card indices (u32).
/// Returns `num_situations * 1326` u32 strength values.
pub fn evaluate_all_hand_strengths(boards: &[u32], num_situations: usize) -> Vec<u32> {
    assert_eq!(
        boards.len(),
        num_situations * 5,
        "boards must contain exactly num_situations * 5 card indices"
    );

    let combo_cards = build_combo_cards();

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
                if board.contains(&c1) || board.contains(&c2) {
                    continue;
                }
                sit_strengths[combo_idx] =
                    u32::from(evaluate_hand_strength(&board, (c1, c2)));
            }
            sit_strengths
        })
        .collect();

    let mut strengths = Vec::with_capacity(num_situations * 1326);
    for chunk in chunks {
        strengths.extend_from_slice(&chunk);
    }
    strengths
}

/// GPU-native hand strength evaluator.
///
/// Uploads the combo_cards lookup table and HAND_TABLE once, then evaluates
/// hand strengths for any number of boards entirely on the GPU.
#[cfg(feature = "cuda")]
pub struct GpuHandEvaluator {
    /// Flat combo cards `[1326 * 2]` on GPU.
    gpu_combo_cards: CudaSlice<u32>,
    /// HAND_TABLE `[4824]` on GPU.
    gpu_hand_table: CudaSlice<i32>,
    /// Length of the hand table (4824).
    hand_table_len: u32,
}

#[cfg(feature = "cuda")]
impl GpuHandEvaluator {
    /// Create a new GPU hand evaluator, uploading lookup tables to the GPU.
    pub fn new(gpu: &GpuContext) -> Result<Self, GpuError> {
        let combo_cards_flat = build_combo_cards_flat();
        let hand_table = get_hand_table();
        let hand_table_len = hand_table.len() as u32;

        let gpu_combo_cards = gpu.upload(&combo_cards_flat)?;
        let gpu_hand_table = gpu.upload(&hand_table)?;

        Ok(Self {
            gpu_combo_cards,
            gpu_hand_table,
            hand_table_len,
        })
    }

    /// Reference to the GPU combo_cards buffer `[1326 * 2]`.
    pub fn combo_cards(&self) -> &CudaSlice<u32> {
        &self.gpu_combo_cards
    }

    /// Evaluate hand strengths for boards already on GPU.
    ///
    /// # Arguments
    /// * `gpu` - CUDA context for kernel launch.
    /// * `boards` - GPU buffer of `[num_situations * 5]` card indices.
    /// * `num_situations` - Number of boards.
    ///
    /// # Returns
    /// GPU buffer of `[num_situations * 1326]` u32 strength values.
    pub fn evaluate_on_gpu(
        &self,
        gpu: &GpuContext,
        boards: &CudaSlice<u32>,
        num_situations: usize,
    ) -> Result<CudaSlice<u32>, GpuError> {
        let total = num_situations * 1326;
        let mut strengths = gpu.alloc_zeros::<u32>(total)?;

        if num_situations == 0 {
            return Ok(strengths);
        }

        let kernel = gpu.compile_and_load(
            include_str!("../../kernels/evaluate_hand_strengths.cu"),
            "evaluate_hand_strengths",
        )?;

        let cfg = LaunchConfig::for_num_elems(total as u32);
        unsafe {
            gpu.stream
                .launch_builder(&kernel)
                .arg(&mut strengths)
                .arg(boards)
                .arg(&self.gpu_combo_cards)
                .arg(&self.gpu_hand_table)
                .arg(&self.hand_table_len)
                .arg(&(num_situations as u32))
                .launch(cfg)?;
        }

        Ok(strengths)
    }
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
        assert_eq!(non_blocked_count, 1081);
    }

    #[test]
    fn test_hand_ordering_aa_beats_kk_beats_qq() {
        // Board: 2c 3d 7h 9s Tc — no straight/flush possible for AA/KK/QQ
        let board = board_from_strs(&["2c", "3d", "7h", "9s", "Tc"]);
        let strengths = evaluate_all_hand_strengths(&board, 1);

        let ah = card_from_str("Ah").unwrap();
        let as_ = card_from_str("As").unwrap();
        let aa_idx = range_solver::card::card_pair_to_index(ah, as_);

        let kh = card_from_str("Kh").unwrap();
        let ks = card_from_str("Ks").unwrap();
        let kk_idx = range_solver::card::card_pair_to_index(kh, ks);

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
        let mut boards = Vec::new();
        boards.extend(board_from_strs(&["2c", "3d", "4h", "5s", "7c"]));
        boards.extend(board_from_strs(&["Tc", "Jd", "Qh", "Ks", "9c"]));

        let strengths = evaluate_all_hand_strengths(&boards, 2);
        assert_eq!(strengths.len(), 2 * 1326);

        let board1_nonzero = strengths[..1326].iter().filter(|&&s| s > 0).count();
        let board2_nonzero = strengths[1326..].iter().filter(|&&s| s > 0).count();
        assert!(board1_nonzero > 0);
        assert!(board2_nonzero > 0);
    }

    #[test]
    fn test_straight_beats_pair() {
        let board = board_from_strs(&["5c", "6d", "7h", "Ks", "2c"]);
        let strengths = evaluate_all_hand_strengths(&board, 1);

        let c8h = card_from_str("8h").unwrap();
        let c9h = card_from_str("9h").unwrap();
        let straight_idx = range_solver::card::card_pair_to_index(c8h, c9h);

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

        for (i, &(c1, c2)) in combos.iter().enumerate() {
            let expected = index_to_card_pair(i);
            assert_eq!((c1, c2), expected, "Mismatch at combo index {i}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_gpu_hand_eval_matches_cpu() {
        use crate::gpu::GpuContext;

        let gpu = GpuContext::new(0).expect("CUDA device required");
        let evaluator = GpuHandEvaluator::new(&gpu).unwrap();

        // Test with multiple boards
        let mut boards = Vec::new();
        boards.extend(board_from_strs(&["2c", "3d", "7h", "9s", "Tc"]));
        boards.extend(board_from_strs(&["As", "Kh", "Qd", "Jc", "Ts"]));
        boards.extend(board_from_strs(&["5c", "6d", "7h", "Ks", "2d"]));
        let num_situations = 3;

        // CPU evaluation
        let cpu_strengths = evaluate_all_hand_strengths(&boards, num_situations);

        // GPU evaluation
        let gpu_boards = gpu.upload(&boards).unwrap();
        let gpu_strengths = evaluator.evaluate_on_gpu(&gpu, &gpu_boards, num_situations).unwrap();
        let gpu_result: Vec<u32> = gpu.download(&gpu_strengths).unwrap();

        assert_eq!(cpu_strengths.len(), gpu_result.len());

        // GPU and CPU must produce identical results
        for (i, (cpu, gpu_val)) in cpu_strengths.iter().zip(gpu_result.iter()).enumerate() {
            assert_eq!(
                *cpu, *gpu_val,
                "Mismatch at index {i} (sit={}, combo={}): CPU={cpu}, GPU={gpu_val}",
                i / 1326,
                i % 1326
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_gpu_hand_eval_empty() {
        use crate::gpu::GpuContext;

        let gpu = GpuContext::new(0).expect("CUDA device required");
        let evaluator = GpuHandEvaluator::new(&gpu).unwrap();

        let boards: Vec<u32> = vec![];
        let gpu_boards = gpu.upload(&boards).unwrap();
        let gpu_strengths = evaluator.evaluate_on_gpu(&gpu, &gpu_boards, 0).unwrap();
        let result: Vec<u32> = gpu.download(&gpu_strengths).unwrap();
        assert!(result.is_empty());
    }
}
