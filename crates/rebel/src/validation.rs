//! Validation metrics for ReBeL CFV networks.
//!
//! Provides MSE computation, per-street breakdown, and held-out validation
//! set generation by solving random subgames exactly.

use poker_solver_core::blueprint_v2::Street;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::blueprint_sampler::deal_hand;
use crate::data_buffer::BufferRecord;
use crate::generate::pbs_to_buffer_record;
use crate::pbs::{Pbs, NUM_COMBOS};
use crate::solver::{solve_river_pbs, SolveConfig};

/// Compute masked mean squared error between predicted and actual CFVs.
///
/// Only counts combos where `mask[i] > 0` (non-board-blocked).
/// Returns 0.0 if no valid combos exist.
pub fn compute_mse(predicted: &[f32], actual: &[f32], mask: &[f32]) -> f32 {
    let mut sum_sq = 0.0_f64;
    let mut count = 0u64;

    for i in 0..predicted.len().min(actual.len()).min(mask.len()) {
        if mask[i] > 0.0 {
            let diff = (predicted[i] - actual[i]) as f64;
            sum_sq += diff * diff;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    (sum_sq / count as f64) as f32
}

/// Determine street from board card count.
///
/// - 5 cards -> River
/// - 4 cards -> Turn
/// - 3 cards -> Flop
/// - 0 cards -> Preflop
///
/// Panics on invalid counts (1, 2, or >5).
pub fn street_from_board_count(count: usize) -> Street {
    match count {
        0 => Street::Preflop,
        3 => Street::Flop,
        4 => Street::Turn,
        5 => Street::River,
        _ => panic!("invalid board card count: {count}"),
    }
}

/// Per-street validation results.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Overall MSE across all records.
    pub overall_mse: f32,
    /// Per-street breakdown: (street, mse, count).
    pub per_street: Vec<(Street, f32, usize)>,
    /// Total number of records validated.
    pub total_records: usize,
}

/// Validate model predictions against solved reference data.
///
/// Takes tuples of `(predicted_cfvs, actual_cfvs, valid_mask, board_card_count)`
/// and computes MSE overall and per-street.
pub fn validate_predictions(
    predictions: &[([f32; 1326], [f32; 1326], [u8; 1326], u8)],
) -> ValidationReport {
    if predictions.is_empty() {
        return ValidationReport {
            overall_mse: 0.0,
            per_street: Vec::new(),
            total_records: 0,
        };
    }

    // Accumulators for overall MSE
    let mut overall_sum_sq = 0.0_f64;
    let mut overall_count = 0u64;

    // Per-street accumulators: keyed by Street discriminant
    // (Preflop=0, Flop=1, Turn=2, River=3)
    let mut street_sum_sq = [0.0_f64; 4];
    let mut street_combo_count = [0u64; 4];
    let mut street_record_count = [0usize; 4];

    for (predicted, actual, mask, board_card_count) in predictions {
        let street = street_from_board_count(*board_card_count as usize);
        let street_idx = street as usize;

        street_record_count[street_idx] += 1;

        for i in 0..NUM_COMBOS {
            if mask[i] > 0 {
                let diff = (predicted[i] - actual[i]) as f64;
                let sq = diff * diff;
                overall_sum_sq += sq;
                overall_count += 1;
                street_sum_sq[street_idx] += sq;
                street_combo_count[street_idx] += 1;
            }
        }
    }

    let overall_mse = if overall_count > 0 {
        (overall_sum_sq / overall_count as f64) as f32
    } else {
        0.0
    };

    // Build per-street results (only include streets that have data)
    let streets = [Street::Preflop, Street::Flop, Street::Turn, Street::River];
    let mut per_street = Vec::new();
    for &street in &streets {
        let idx = street as usize;
        if street_record_count[idx] > 0 {
            let mse = if street_combo_count[idx] > 0 {
                (street_sum_sq[idx] / street_combo_count[idx] as f64) as f32
            } else {
                0.0
            };
            per_street.push((street, mse, street_record_count[idx]));
        }
    }

    ValidationReport {
        overall_mse,
        per_street,
        total_records: predictions.len(),
    }
}

/// Generate a held-out validation set by solving random river subgames exactly.
///
/// For each sample:
/// 1. Deal a random board (5 cards for river)
/// 2. Create a uniform-reach PBS
/// 3. Solve exactly with range-solver
/// 4. Convert to BufferRecords with solved CFVs
///
/// Only generates river validation examples (exact solving without evaluator).
/// Turn/flop would require a LeafEvaluator.
///
/// `num_per_street`: how many validation examples per street (currently only river)
pub fn generate_validation_set(
    num_per_street: usize,
    solve_config: &SolveConfig,
    seed: u64,
) -> Vec<BufferRecord> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut records = Vec::with_capacity(num_per_street * 2); // 2 players per PBS

    for _ in 0..num_per_street {
        let deal = deal_hand(&mut rng);
        let board = deal.board.to_vec();

        // Create uniform-reach PBS for river
        let pbs = Pbs::new_uniform(board, 100, 200);

        match solve_river_pbs(&pbs, solve_config) {
            Ok(result) => {
                // OOP record
                let mut rec_oop = pbs_to_buffer_record(&pbs, 0);
                rec_oop.cfvs = result.oop_cfvs;
                rec_oop.game_value = result.oop_game_value;
                records.push(rec_oop);

                // IP record
                let mut rec_ip = pbs_to_buffer_record(&pbs, 1);
                rec_ip.cfvs = result.ip_cfvs;
                rec_ip.game_value = result.ip_game_value;
                records.push(rec_ip);
            }
            Err(e) => {
                eprintln!("Warning: failed to solve validation PBS: {e}");
            }
        }
    }

    records
}

/// Format a `ValidationReport` as a human-readable string.
pub fn format_report(report: &ValidationReport) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "Validation Report ({} records)\n",
        report.total_records
    ));
    out.push_str(&format!(
        "  Overall MSE: {:.6}\n",
        report.overall_mse
    ));

    if !report.per_street.is_empty() {
        out.push_str("  Per-street:\n");
        for &(street, mse, count) in &report.per_street {
            out.push_str(&format!(
                "    {:?}: MSE={:.6}, records={}\n",
                street, mse, count
            ));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_mse_exact() {
        let predicted = [1.0_f32, 2.0, 3.0];
        let actual = [1.1_f32, 2.2, 2.8];
        let mask = [1.0_f32, 1.0, 1.0];

        let mse = compute_mse(&predicted, &actual, &mask);
        // (0.1^2 + 0.2^2 + 0.2^2) / 3 = (0.01 + 0.04 + 0.04) / 3 = 0.03
        assert!(
            (mse - 0.03).abs() < 1e-5,
            "expected ~0.03, got {mse}"
        );
    }

    #[test]
    fn test_compute_mse_with_mask() {
        let predicted = [1.0_f32, 2.0, 3.0, 4.0];
        let actual = [1.1_f32, 2.2, 2.8, 100.0]; // last value is far off
        let mask = [1.0_f32, 1.0, 1.0, 0.0]; // last is masked out

        let mse = compute_mse(&predicted, &actual, &mask);
        // Only first 3 count: (0.01 + 0.04 + 0.04) / 3 = 0.03
        assert!(
            (mse - 0.03).abs() < 1e-5,
            "expected ~0.03, got {mse}"
        );
    }

    #[test]
    fn test_compute_mse_empty() {
        let predicted = [1.0_f32, 2.0, 3.0];
        let actual = [1.1_f32, 2.2, 2.8];
        let mask = [0.0_f32, 0.0, 0.0]; // all masked

        let mse = compute_mse(&predicted, &actual, &mask);
        assert_eq!(mse, 0.0, "all-masked should return 0.0");
    }

    #[test]
    fn test_street_from_board_count() {
        assert_eq!(street_from_board_count(5), Street::River);
        assert_eq!(street_from_board_count(4), Street::Turn);
        assert_eq!(street_from_board_count(3), Street::Flop);
        assert_eq!(street_from_board_count(0), Street::Preflop);
    }

    #[test]
    #[should_panic(expected = "invalid board card count")]
    fn test_street_from_board_count_invalid() {
        street_from_board_count(2);
    }

    #[test]
    fn test_validate_predictions_basic() {
        // Two river predictions (board_card_count = 5)
        let mut pred1 = [0.0_f32; 1326];
        let mut actual1 = [0.0_f32; 1326];
        let mut mask1 = [0_u8; 1326];

        // Set up a few valid combos
        pred1[0] = 1.0;
        actual1[0] = 1.1;
        mask1[0] = 1;

        pred1[1] = 2.0;
        actual1[1] = 2.0; // exact match
        mask1[1] = 1;

        let mut pred2 = [0.0_f32; 1326];
        let mut actual2 = [0.0_f32; 1326];
        let mut mask2 = [0_u8; 1326];

        pred2[0] = 3.0;
        actual2[0] = 3.5;
        mask2[0] = 1;

        let predictions = vec![
            (pred1, actual1, mask1, 5_u8),
            (pred2, actual2, mask2, 5_u8),
        ];

        let report = validate_predictions(&predictions);

        assert_eq!(report.total_records, 2);
        assert_eq!(report.per_street.len(), 1); // only river
        assert_eq!(report.per_street[0].0, Street::River);
        assert_eq!(report.per_street[0].2, 2); // 2 records

        // Overall MSE: (0.1^2 + 0.0^2 + 0.5^2) / 3 = (0.01 + 0.0 + 0.25) / 3 = 0.08666...
        assert!(
            (report.overall_mse - 0.0866667).abs() < 1e-4,
            "expected ~0.0867, got {}",
            report.overall_mse
        );
        assert!(report.overall_mse > 0.0);
    }

    #[test]
    fn test_validate_predictions_empty() {
        let report = validate_predictions(&[]);
        assert_eq!(report.total_records, 0);
        assert_eq!(report.overall_mse, 0.0);
        assert!(report.per_street.is_empty());
    }

    #[test]
    fn test_validate_predictions_multi_street() {
        let mut pred_r = [0.0_f32; 1326];
        let mut actual_r = [0.0_f32; 1326];
        let mut mask_r = [0_u8; 1326];
        pred_r[0] = 1.0;
        actual_r[0] = 1.0;
        mask_r[0] = 1;

        let mut pred_t = [0.0_f32; 1326];
        let mut actual_t = [0.0_f32; 1326];
        let mut mask_t = [0_u8; 1326];
        pred_t[0] = 2.0;
        actual_t[0] = 3.0;
        mask_t[0] = 1;

        let predictions = vec![
            (pred_r, actual_r, mask_r, 5_u8), // river
            (pred_t, actual_t, mask_t, 4_u8), // turn
        ];

        let report = validate_predictions(&predictions);

        assert_eq!(report.total_records, 2);
        assert_eq!(report.per_street.len(), 2);

        // Find river and turn entries
        let river_entry = report.per_street.iter().find(|e| e.0 == Street::River).unwrap();
        let turn_entry = report.per_street.iter().find(|e| e.0 == Street::Turn).unwrap();

        // River: MSE = 0.0 (exact match)
        assert_eq!(river_entry.1, 0.0);
        assert_eq!(river_entry.2, 1);

        // Turn: MSE = 1.0 (diff = 1.0, sq = 1.0, count = 1)
        assert!((turn_entry.1 - 1.0).abs() < 1e-5);
        assert_eq!(turn_entry.2, 1);
    }

    #[test]
    fn test_format_report() {
        let report = ValidationReport {
            overall_mse: 0.0042,
            per_street: vec![
                (Street::River, 0.003, 100),
                (Street::Turn, 0.005, 50),
            ],
            total_records: 150,
        };

        let formatted = format_report(&report);

        assert!(
            formatted.contains("150 records"),
            "should contain record count"
        );
        assert!(
            formatted.contains("0.004200"),
            "should contain overall MSE: got:\n{formatted}"
        );
        assert!(
            formatted.contains("River"),
            "should contain River street"
        );
        assert!(
            formatted.contains("Turn"),
            "should contain Turn street"
        );
        assert!(
            formatted.contains("Per-street"),
            "should contain per-street header"
        );
    }

    #[test]
    fn test_generate_validation_set() {
        use range_solver::bet_size::BetSizeOptions;

        let solve_config = SolveConfig {
            bet_sizes: BetSizeOptions::try_from(("50%,a", "")).expect("valid bet sizes"),
            turn_bet_sizes: None,
            flop_bet_sizes: None,
            solver_iterations: 100,
            target_exploitability: 0.05,
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
        };

        let records = generate_validation_set(2, &solve_config, 42);

        // 2 PBSs * 2 players = 4 records
        assert_eq!(records.len(), 4, "expected 4 records (2 per PBS)");

        for rec in &records {
            // All should be river (5 board cards)
            assert_eq!(
                rec.board_card_count, 5,
                "expected 5 board cards, got {}",
                rec.board_card_count
            );

            // Should have some non-zero CFVs
            let has_nonzero = rec.cfvs.iter().any(|&v| v != 0.0);
            assert!(
                has_nonzero,
                "validation record should have non-zero CFVs"
            );

            // Game value should be finite
            assert!(
                rec.game_value.is_finite(),
                "game_value should be finite, got {}",
                rec.game_value
            );

            // Valid mask should have some valid combos
            let valid_count: usize = rec.valid_mask.iter().map(|&m| m as usize).sum();
            assert!(
                valid_count > 0,
                "should have some valid combos in mask"
            );
        }
    }
}
