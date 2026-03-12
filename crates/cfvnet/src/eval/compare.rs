use range_solver::bet_size::BetSizeOptions;

use crate::config::{DatagenConfig, GameConfig};
use crate::datagen::sampler::{sample_situation, Situation};
use crate::datagen::solver::{solve_situation, SolveConfig, SolveResult};
use crate::eval::metrics::{compute_prediction_metrics, PredictionMetrics};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Generate a random comparison spot from a seed.
pub fn generate_comparison_spot(seed: u64, initial_stack: i32, datagen: &DatagenConfig) -> Situation {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    sample_situation(datagen, initial_stack, 5, &mut rng)
}

/// Default solve config for comparison spots.
pub fn default_solve_config(game: &GameConfig) -> Result<SolveConfig, String> {
    let bet_str = game.bet_sizes.join(",");
    let bet_sizes = BetSizeOptions::try_from((bet_str.as_str(), ""))
        .map_err(|e| format!("invalid bet sizes: {e}"))?;
    Ok(SolveConfig {
        bet_sizes,
        solver_iterations: 500,
        target_exploitability: 0.005,
        add_allin_threshold: game.add_allin_threshold,
        force_allin_threshold: game.force_allin_threshold,
    })
}

/// Compare a single predicted CFV vector against ground-truth solve.
pub fn compare_single_spot(
    predicted: &[f32],
    actual: &[f32],
    valid_mask: &[bool],
    pot: f32,
) -> PredictionMetrics {
    compute_prediction_metrics(predicted, actual, valid_mask, pot)
}

/// Per-spot comparison result with board context for reporting.
pub struct SpotResult {
    pub board: [u8; 5],
    pub board_size: usize,
    pub pot: i32,
    pub effective_stack: i32,
    pub mae: f64,
    pub mbb: f64,
}

/// Summary of comparison across multiple spots.
pub struct ComparisonSummary {
    pub num_spots: usize,
    pub mean_mae: f64,
    pub mean_max_error: f64,
    pub mean_mbb: f64,
    pub worst_mae: f64,
    pub worst_mbb: f64,
    pub spots: Vec<SpotResult>,
}

/// Run comparison on N random spots.
///
/// `predict_fn` takes a `Situation` and `SolveResult` and returns predicted
/// `[f32; 1326]` CFVs for OOP. For now this is used with the exact solver
/// as a self-test.
pub fn run_comparison<F>(
    game_config: &GameConfig,
    datagen: &DatagenConfig,
    num_spots: usize,
    base_seed: u64,
    predict_fn: F,
) -> Result<ComparisonSummary, String>
where
    F: Fn(&Situation, &SolveResult) -> Vec<f32>,
{
    let solve_config = default_solve_config(game_config)?;
    let mut maes = Vec::with_capacity(num_spots);
    let mut max_errors = Vec::with_capacity(num_spots);
    let mut mbbs = Vec::with_capacity(num_spots);
    let mut spots = Vec::with_capacity(num_spots);

    for i in 0..num_spots {
        let spot = generate_comparison_spot(base_seed + i as u64, game_config.initial_stack, datagen);
        let result = solve_situation(&spot, &solve_config)?;
        let predicted = predict_fn(&spot, &result);
        let mask: Vec<bool> = result.valid_mask.to_vec();
        let metrics = compare_single_spot(&predicted, &result.oop_evs, &mask, spot.pot as f32);
        maes.push(metrics.mae);
        max_errors.push(metrics.max_error);
        mbbs.push(metrics.mbb_error);
        spots.push(SpotResult {
            board: spot.board,
            board_size: spot.board_size,
            pot: spot.pot,
            effective_stack: spot.effective_stack,
            mae: metrics.mae,
            mbb: metrics.mbb_error,
        });
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
    use crate::config::GameConfig;

    #[test]
    fn comparison_with_perfect_oracle_shows_zero_error() {
        let game = GameConfig {
            initial_stack: 200,
            bet_sizes: vec!["50%".into(), "a".into()],
            ..Default::default()
        };
        let datagen = DatagenConfig::default();

        // Use exact OOP EVs as "predictions" — error should be zero
        let summary = run_comparison(&game, &datagen, 2, 42, |_sit, result| result.oop_evs.to_vec())
            .unwrap();

        assert!(
            summary.mean_mae < 1e-6,
            "perfect oracle should have zero MAE, got {}",
            summary.mean_mae
        );
        assert!(
            summary.worst_mae < 1e-6,
            "perfect oracle worst MAE should be zero, got {}",
            summary.worst_mae
        );
        assert_eq!(summary.num_spots, 2);
    }

    #[test]
    fn summary_contains_per_spot_results() {
        let game = GameConfig {
            initial_stack: 200,
            bet_sizes: vec!["50%".into(), "a".into()],
            ..Default::default()
        };
        let datagen = DatagenConfig::default();

        let summary = run_comparison(&game, &datagen, 3, 42, |_sit, result| result.oop_evs.to_vec())
            .unwrap();

        assert_eq!(summary.spots.len(), 3);
        for spot in &summary.spots {
            assert!(spot.board_size == 4 || spot.board_size == 5);
            assert!(spot.pot > 0);
            assert!(spot.effective_stack > 0);
            assert!(spot.mae.is_finite());
            assert!(spot.mbb.is_finite());
        }
    }
}
