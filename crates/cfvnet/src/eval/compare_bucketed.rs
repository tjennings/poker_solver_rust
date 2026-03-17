use std::collections::HashMap;

use poker_solver_core::blueprint_v2::bucket_file::{BucketFile, PackedBoard};

use crate::config::{DatagenConfig, GameConfig};
use crate::datagen::bucket_mapping::encode_bucketed_record;
use crate::datagen::sampler::Situation;
use crate::datagen::solver::solve_situation;
use crate::eval::compare::{default_solve_config, generate_comparison_spot, ComparisonSummary, SpotResult};
use crate::eval::metrics::{mean_absolute_error, max_absolute_error, pot_relative_to_mbb};

/// Convert a cfvnet card index (0..52, `4*rank + suit` where rank: 2=0..A=12,
/// suit: C=0 D=1 H=2 S=3) to an `rs_poker::core::Card`.
fn cfvnet_card_to_rs_poker(card_id: u8) -> rs_poker::core::Card {
    let rank = card_id / 4;
    let suit = card_id % 4;
    rs_poker::core::Card::new(
        rs_poker::core::Value::from(rank),
        match suit {
            0 => rs_poker::core::Suit::Club,
            1 => rs_poker::core::Suit::Diamond,
            2 => rs_poker::core::Suit::Heart,
            _ => rs_poker::core::Suit::Spade,
        },
    )
}

/// Convert a `Situation`'s board cards to a `PackedBoard` using the same
/// canonical ordering as the clustering pipeline (sorted by value_rank desc,
/// suit asc).
fn board_to_packed(spot: &Situation) -> PackedBoard {
    let cards: Vec<rs_poker::core::Card> = spot
        .board_cards()
        .iter()
        .map(|&c| cfvnet_card_to_rs_poker(c))
        .collect();
    poker_solver_core::blueprint_v2::cluster_pipeline::canonical_key(&cards)
}

/// Run comparison of bucketed model predictions against exact solver.
///
/// For each spot:
/// 1. Sample a random river situation
/// 2. Solve exactly with range-solver (1326-dim)
/// 3. Map to bucket space using `encode_bucketed_record` (ground truth)
/// 4. Call `predict_fn` with the bucketed input to get model predictions
/// 5. Compare predicted vs actual bucket CFVs
pub fn run_bucketed_comparison<F>(
    game_config: &GameConfig,
    datagen: &DatagenConfig,
    num_spots: usize,
    base_seed: u64,
    bucket_file: &BucketFile,
    num_buckets: usize,
    predict_fn: F,
) -> Result<ComparisonSummary, String>
where
    F: Fn(&[f32]) -> Vec<f32>, // Takes bucketed input [2*nb+1], returns predicted [2*nb]
{
    let solve_config = default_solve_config(game_config, datagen)?;
    let initial_stack = game_config.initial_stack;

    // Build board lookup cache once.
    let board_cache: HashMap<PackedBoard, u32> = bucket_file.board_index_map();

    let mut maes = Vec::with_capacity(num_spots);
    let mut max_errors = Vec::with_capacity(num_spots);
    let mut mbbs = Vec::with_capacity(num_spots);
    let mut spots = Vec::with_capacity(num_spots);
    let mut skipped = 0usize;

    let mut i = 0u64;
    while spots.len() < num_spots {
        let spot = generate_comparison_spot(base_seed + i, initial_stack, datagen);
        i += 1;

        // Skip spots with zero effective stack.
        if spot.effective_stack <= 0 {
            skipped += 1;
            continue;
        }

        // Look up this board in the bucket file.
        let packed = board_to_packed(&spot);
        let board_idx = match board_cache.get(&packed) {
            Some(&idx) => idx,
            None => {
                // Board not in bucket file -- skip.
                skipped += 1;
                continue;
            }
        };

        let result = solve_situation(&spot, &solve_config)?;

        // Map to bucket space: both players' CFVs + ranges.
        let (input, target) = encode_bucketed_record(
            &spot.ranges[0],
            &spot.ranges[1],
            &result.oop_evs,
            &result.ip_evs,
            bucket_file,
            board_idx,
            num_buckets,
            spot.pot as f32,
            initial_stack as f32,
        );

        // Get model prediction.
        let predicted = predict_fn(&input);

        // Build mask: bucket has nonzero reach from the respective player.
        // input[0..num_buckets] = OOP reach, input[num_buckets..2*num_buckets] = IP reach.
        let mask: Vec<bool> = (0..2 * num_buckets)
            .map(|j| input[j] > 0.0)
            .collect();

        let mae = mean_absolute_error(&predicted, &target, &mask);
        let max_err = max_absolute_error(&predicted, &target, &mask);
        let mbb = pot_relative_to_mbb(mae, f64::from(spot.pot), 2.0);

        maes.push(mae);
        max_errors.push(max_err);
        mbbs.push(mbb);
        spots.push(SpotResult {
            board: spot.board,
            board_size: spot.board_size,
            pot: spot.pot,
            effective_stack: spot.effective_stack,
            mae,
            mbb,
        });
    }

    if skipped > 0 {
        eprintln!("  (skipped {skipped} spots: board not in bucket file or zero stack)");
    }

    let n = num_spots as f64;
    Ok(ComparisonSummary {
        num_spots,
        mean_mae: maes.iter().sum::<f64>() / n,
        mean_max_error: max_errors.iter().sum::<f64>() / n,
        mean_mbb: mbbs.iter().sum::<f64>() / n,
        worst_mae: maes.iter().copied().fold(0.0_f64, f64::max),
        worst_mbb: mbbs.iter().copied().fold(0.0_f64, f64::max),
        spots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::metrics::mean_absolute_error;

    #[test]
    fn perfect_prediction_gives_zero_error() {
        // Test the metrics functions directly with identical prediction and target.
        let pred = vec![0.1f32, 0.2, -0.1, 0.3, 0.0, 0.15];
        let actual = pred.clone();
        let mask = vec![true; 6];

        let mae = mean_absolute_error(&pred, &actual, &mask);
        assert!(mae < 1e-6, "perfect prediction should have zero MAE, got {mae}");
    }

    #[test]
    fn masked_entries_ignored() {
        let pred = vec![0.0f32, 0.0, 100.0];
        let actual = vec![0.0f32, 0.0, 0.0];
        let mask = vec![true, true, false]; // third entry masked out

        let mae = mean_absolute_error(&pred, &actual, &mask);
        assert!(mae < 1e-6, "masked entries should be ignored, got MAE={mae}");
    }

    #[test]
    fn cfvnet_card_conversion_round_trip() {
        // card 0 = 2c, card 51 = As
        let c0 = cfvnet_card_to_rs_poker(0);
        assert_eq!(c0.value, rs_poker::core::Value::Two);
        assert_eq!(c0.suit, rs_poker::core::Suit::Club);

        let c51 = cfvnet_card_to_rs_poker(51);
        assert_eq!(c51.value, rs_poker::core::Value::Ace);
        assert_eq!(c51.suit, rs_poker::core::Suit::Spade);

        let c48 = cfvnet_card_to_rs_poker(48);
        assert_eq!(c48.value, rs_poker::core::Value::Ace);
        assert_eq!(c48.suit, rs_poker::core::Suit::Club);
    }
}
