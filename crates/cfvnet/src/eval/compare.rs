use crate::config::{DatagenConfig, GameConfig};
use crate::datagen::sampler::{sample_situation, Situation};
use crate::datagen::solver::{solve_situation, SolveConfig, SolveResult};
use crate::eval::metrics::{compute_prediction_metrics, PredictionMetrics};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Generate a random comparison spot from a seed.
pub fn generate_comparison_spot(seed: u64, initial_stack: i32) -> Situation {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let config = DatagenConfig::default();
    sample_situation(&config, initial_stack, 5, &mut rng)
}

/// Default solve config for comparison spots.
pub fn default_solve_config(game: &GameConfig) -> SolveConfig {
    SolveConfig {
        bet_sizes: game.bet_sizes.clone(),
        solver_iterations: 500,
        target_exploitability: 0.005,
        add_allin_threshold: game.add_allin_threshold,
        force_allin_threshold: game.force_allin_threshold,
    }
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

/// Summary of comparison across multiple spots.
pub struct ComparisonSummary {
    pub num_spots: usize,
    pub mean_mae: f64,
    pub mean_max_error: f64,
    pub mean_mbb: f64,
    pub worst_mae: f64,
    pub worst_mbb: f64,
}

/// Run comparison on N random spots.
///
/// `predict_fn` takes a `Situation` and `SolveResult` and returns predicted
/// `[f32; 1326]` CFVs for OOP. For now this is used with the exact solver
/// as a self-test.
pub fn run_comparison<F>(
    game_config: &GameConfig,
    num_spots: usize,
    base_seed: u64,
    predict_fn: F,
) -> Result<ComparisonSummary, String>
where
    F: Fn(&Situation, &SolveResult) -> Vec<f32>,
{
    let solve_config = default_solve_config(game_config);
    let mut maes = Vec::with_capacity(num_spots);
    let mut max_errors = Vec::with_capacity(num_spots);
    let mut mbbs = Vec::with_capacity(num_spots);

    for i in 0..num_spots {
        let spot = generate_comparison_spot(base_seed + i as u64, game_config.initial_stack);
        let result = solve_situation(&spot, &solve_config)?;
        let predicted = predict_fn(&spot, &result);
        let mask: Vec<bool> = result.valid_mask.to_vec();
        let metrics = compare_single_spot(&predicted, &result.oop_evs, &mask, spot.pot as f32);
        maes.push(metrics.mae);
        max_errors.push(metrics.max_error);
        mbbs.push(metrics.mbb_error);
    }

    let n = num_spots as f64;
    Ok(ComparisonSummary {
        num_spots,
        mean_mae: maes.iter().sum::<f64>() / n,
        mean_max_error: max_errors.iter().sum::<f64>() / n,
        mean_mbb: mbbs.iter().sum::<f64>() / n,
        worst_mae: maes.iter().copied().fold(0.0_f64, f64::max),
        worst_mbb: mbbs.iter().copied().fold(0.0_f64, f64::max),
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

        // Use exact OOP EVs as "predictions" — error should be zero
        let summary = run_comparison(&game, 2, 42, |_sit, result| result.oop_evs.to_vec())
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
}
