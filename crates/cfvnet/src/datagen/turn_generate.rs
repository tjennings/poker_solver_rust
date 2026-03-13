//! Turn training data generation pipeline.
//!
//! Samples random turn situations, solves them using [`CfvSubgameSolver`] with
//! a [`RiverNetEvaluator`] as the leaf evaluator, extracts root CFVs, and
//! writes [`TrainingRecord`]s with 4-card boards.

use std::io::BufWriter;
use std::path::Path;

use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
use indicatif::{ProgressBar, ProgressStyle};
use poker_solver_core::blueprint_v2::cfv_subgame_solver::{CfvSubgameSolver, LeafEvaluator};
use poker_solver_core::blueprint_v2::game_tree::GameTree;
use poker_solver_core::blueprint_v2::subgame_cfr::SubgameHands;
use poker_solver_core::blueprint_v2::Street;
use poker_solver_core::poker::{Card, Suit, Value};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use range_solver::card::card_pair_to_index;

use super::range_gen::NUM_COMBOS;
use super::sampler::sample_situation;
use super::storage::{write_record, TrainingRecord};
use crate::config::CfvnetConfig;
use crate::eval::river_net_evaluator::RiverNetEvaluator;
use crate::model::network::{CfvNet, input_size};

type B = NdArray;

/// Convert a range-solver `u8` card to an `rs_poker::core::Card`.
fn u8_to_rs_card(id: u8) -> Card {
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

/// Convert an `rs_poker::core::Card` to a range-solver `u8` card.
fn rs_card_to_u8(card: Card) -> u8 {
    let rank = card.value as u8;
    let suit = match card.suit {
        Suit::Club => 0,
        Suit::Diamond => 1,
        Suit::Heart => 2,
        Suit::Spade => 3,
    };
    4 * rank + suit
}

/// Parse config bet size strings (e.g. `["50%", "100%", "a"]`) into pot fractions.
///
/// Entries like "a" (all-in) are skipped — the game tree builder adds all-in
/// automatically.
fn parse_bet_sizes(sizes: &[String]) -> Vec<f64> {
    sizes
        .iter()
        .filter_map(|s| {
            let trimmed = s.trim();
            if trimmed.eq_ignore_ascii_case("a") {
                return None;
            }
            let num_str = trimmed.trim_end_matches('%');
            num_str.parse::<f64>().ok().map(|v| v / 100.0)
        })
        .collect()
}

/// Solve a single turn situation and return per-player root CFVs mapped to 1326 indices.
///
/// Returns `(oop_cfvs_1326, ip_cfvs_1326, valid_mask, game_value_oop, game_value_ip)`.
#[allow(clippy::type_complexity)]
fn solve_turn_situation(
    board_u8: &[u8],
    pot: f64,
    effective_stack: f64,
    ranges: &[[f32; NUM_COMBOS]; 2],
    bet_sizes: &[Vec<f64>],
    solver_iterations: u32,
    evaluator: Box<dyn LeafEvaluator>,
) -> (
    [f32; NUM_COMBOS],
    [f32; NUM_COMBOS],
    [u8; NUM_COMBOS],
    f32,
    f32,
) {
    // Convert board from u8 to Card.
    let board_cards: Vec<Card> = board_u8.iter().map(|&c| u8_to_rs_card(c)).collect();

    // Build tree and enumerate combos.
    let invested = [pot / 2.0; 2];
    let starting_stack = effective_stack + pot / 2.0;
    let tree = GameTree::build_subgame(
        Street::Turn,
        pot,
        invested,
        starting_stack,
        bet_sizes,
        Some(1), // depth_limit=1: river boundaries become DepthBoundary
    );

    let hands = SubgameHands::enumerate(&board_cards);
    let mut solver = CfvSubgameSolver::new(
        tree,
        hands.clone(),
        &board_cards,
        evaluator,
        starting_stack,
    );

    solver.train(solver_iterations);

    // Extract root CFVs for both players.
    let oop_cfvs_combo = solver.root_cfvs(0);
    let ip_cfvs_combo = solver.root_cfvs(1);

    // Map combo-indexed CFVs to 1326-indexed arrays.
    let mut oop_cfvs = [0.0_f32; NUM_COMBOS];
    let mut ip_cfvs = [0.0_f32; NUM_COMBOS];
    let mut valid_mask = [0_u8; NUM_COMBOS];

    for (combo_idx, combo) in hands.combos.iter().enumerate() {
        let c0 = rs_card_to_u8(combo[0]);
        let c1 = rs_card_to_u8(combo[1]);
        let idx_1326 = card_pair_to_index(c0, c1);

        // CFVs from the solver are in chip units; normalize to pot-relative.
        let half_pot = pot / 2.0;
        let norm = if half_pot > 0.0 { half_pot } else { 1.0 };
        oop_cfvs[idx_1326] = (oop_cfvs_combo[combo_idx] / norm) as f32;
        ip_cfvs[idx_1326] = (ip_cfvs_combo[combo_idx] / norm) as f32;
        valid_mask[idx_1326] = 1;
    }

    // Compute weighted game values.
    let oop_gv = weighted_sum(&ranges[0], &oop_cfvs);
    let ip_gv = weighted_sum(&ranges[1], &ip_cfvs);

    (oop_cfvs, ip_cfvs, valid_mask, oop_gv, ip_gv)
}

/// Compute `sum(range[i] * cfvs[i])` for all combos.
fn weighted_sum(range: &[f32; NUM_COMBOS], cfvs: &[f32; NUM_COMBOS]) -> f32 {
    range.iter().zip(cfvs.iter()).map(|(&r, &c)| r * c).sum()
}

/// Generate turn training data by sampling situations, solving with
/// `CfvSubgameSolver` + `RiverNetEvaluator`, and writing paired records.
///
/// The river model is loaded once and cloned for each situation's evaluator.
///
/// # Errors
///
/// Returns an error if the config is invalid, the river model cannot be
/// loaded, or file IO fails.
pub fn generate_turn_training_data(
    config: &CfvnetConfig,
    output_path: &Path,
) -> Result<(), String> {
    let river_model_path = config
        .game
        .river_model_path
        .as_deref()
        .ok_or("river_model_path is required for turn datagen")?;

    let num_samples = config.datagen.num_samples;
    let seed = config.datagen.seed;
    let solver_iterations = config.datagen.solver_iterations;
    let bet_sizes_f64 = parse_bet_sizes(&config.game.bet_sizes);
    if bet_sizes_f64.is_empty() {
        return Err("no valid percentage bet sizes found in config".into());
    }
    let bet_sizes_vec = vec![bet_sizes_f64];

    // Load river model.
    let device = <B as burn::tensor::backend::Backend>::Device::default();
    let in_size = input_size(5); // River model takes 5-card boards.
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    let model = CfvNet::<B>::new(
        &device,
        config.training.hidden_layers,
        config.training.hidden_size,
        in_size,
    )
    .load_file(river_model_path, &recorder, &device)
    .map_err(|e| format!("failed to load river model: {e}"))?;

    // Sample all situations deterministically.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let situations: Vec<_> = (0..num_samples)
        .map(|_| sample_situation(&config.datagen, config.game.initial_stack, 4, &mut rng))
        .collect();

    let pb = ProgressBar::new(num_samples);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len} [{elapsed_precise}] {msg}")
            .expect("valid progress bar template"),
    );

    // Open output file.
    let file =
        std::fs::File::create(output_path).map_err(|e| format!("create output: {e}"))?;
    let mut writer = BufWriter::new(file);

    // Process situations sequentially (each solve is already parallel internally).
    for sit in &situations {
        if sit.effective_stack <= 0 {
            pb.inc(1);
            continue;
        }

        // Clone model for a fresh evaluator each solve.
        let eval_model = model.clone();
        let evaluator: Box<dyn LeafEvaluator> =
            Box::new(RiverNetEvaluator::new(eval_model, device));

        let (oop_cfvs, ip_cfvs, valid_mask, _oop_gv, _ip_gv) = solve_turn_situation(
            sit.board_cards(),
            f64::from(sit.pot),
            f64::from(sit.effective_stack),
            &sit.ranges,
            &bet_sizes_vec,
            solver_iterations,
            evaluator,
        );

        let board_vec = sit.board_cards().to_vec();

        let rec = TrainingRecord {
            board: board_vec,
            pot: sit.pot as f32,
            effective_stack: sit.effective_stack as f32,
            oop_range: sit.ranges[0],
            ip_range: sit.ranges[1],
            oop_cfvs,
            ip_cfvs,
            valid_mask,
        };
        write_record(&mut writer, &rec).map_err(|e| format!("write: {e}"))?;

        pb.inc(1);
    }

    pb.finish_with_message("done");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        CfvnetConfig, DatagenConfig, EvaluationConfig, GameConfig, TrainingConfig,
    };
    use crate::datagen::storage;
    use tempfile::NamedTempFile;

    fn turn_test_config(num_samples: u64) -> CfvnetConfig {
        CfvnetConfig {
            game: GameConfig {
                initial_stack: 200,
                bet_sizes: vec!["50%".into(), "a".into()],
                board_size: 4,
                river_model_path: None, // tests don't load a real model
                ..Default::default()
            },
            datagen: DatagenConfig {
                num_samples,
                street: "turn".into(),
                solver_iterations: 50,
                target_exploitability: 0.05,
                threads: 1,
                seed: 42,
                ..Default::default()
            },
            training: TrainingConfig {
                hidden_layers: 1,
                hidden_size: 8,
                ..Default::default()
            },
            evaluation: EvaluationConfig::default(),
        }
    }

    #[test]
    fn parse_bet_sizes_basic() {
        let sizes = vec!["50%".into(), "100%".into(), "a".into()];
        let parsed = parse_bet_sizes(&sizes);
        assert_eq!(parsed.len(), 2);
        assert!((parsed[0] - 0.5).abs() < 1e-10);
        assert!((parsed[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn parse_bet_sizes_only_allin() {
        let sizes = vec!["a".into()];
        let parsed = parse_bet_sizes(&sizes);
        assert!(parsed.is_empty());
    }

    #[test]
    fn u8_card_roundtrip() {
        for id in 0u8..52 {
            let card = u8_to_rs_card(id);
            let back = rs_card_to_u8(card);
            assert_eq!(id, back, "roundtrip failed for card {id}");
        }
    }

    #[test]
    fn solve_single_turn_situation() {
        // Use a tiny untrained model as the river evaluator.
        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let in_size = input_size(5);
        let model = CfvNet::<B>::new(&device, 1, 8, in_size);
        let evaluator: Box<dyn LeafEvaluator> =
            Box::new(RiverNetEvaluator::new(model, device));

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let datagen_config = DatagenConfig {
            num_samples: 1,
            street: "turn".into(),
            solver_iterations: 20,
            target_exploitability: 0.05,
            threads: 1,
            seed: 42,
            ..Default::default()
        };
        let sit = sample_situation(&datagen_config, 200, 4, &mut rng);
        assert_eq!(sit.board_size, 4);

        if sit.effective_stack <= 0 {
            return; // Skip degenerate situation.
        }

        let bet_sizes_f64 = parse_bet_sizes(&["50%".into(), "a".into()]);
        let (oop_cfvs, ip_cfvs, valid_mask, _oop_gv, _ip_gv) = solve_turn_situation(
            sit.board_cards(),
            f64::from(sit.pot),
            f64::from(sit.effective_stack),
            &sit.ranges,
            &[bet_sizes_f64],
            20,
            evaluator,
        );

        // Verify shapes.
        assert_eq!(oop_cfvs.len(), NUM_COMBOS);
        assert_eq!(ip_cfvs.len(), NUM_COMBOS);
        assert_eq!(valid_mask.len(), NUM_COMBOS);

        // Some combos must be valid.
        let num_valid: usize = valid_mask.iter().map(|&v| v as usize).sum();
        assert!(num_valid > 0, "expected some valid combos");

        // All values should be finite.
        for (i, &cfv) in oop_cfvs.iter().enumerate() {
            assert!(cfv.is_finite(), "OOP combo {i}: non-finite CFV {cfv}");
        }
        for (i, &cfv) in ip_cfvs.iter().enumerate() {
            assert!(cfv.is_finite(), "IP combo {i}: non-finite CFV {cfv}");
        }
    }

    #[test]
    fn generate_turn_requires_river_model_path() {
        let config = turn_test_config(1);
        let output = NamedTempFile::new().unwrap();
        let result = generate_turn_training_data(&config, output.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("river_model_path"));
    }

    #[test]
    fn solve_writes_4_card_board_records() {
        // Directly test the record writing path without needing a model file.
        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let in_size = input_size(5);
        let model = CfvNet::<B>::new(&device, 1, 8, in_size);
        let evaluator: Box<dyn LeafEvaluator> =
            Box::new(RiverNetEvaluator::new(model, device));

        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let datagen_config = DatagenConfig {
            num_samples: 1,
            street: "turn".into(),
            solver_iterations: 10,
            target_exploitability: 0.05,
            threads: 1,
            seed: 123,
            ..Default::default()
        };
        let sit = sample_situation(&datagen_config, 200, 4, &mut rng);
        if sit.effective_stack <= 0 {
            return;
        }

        let bet_sizes_f64 = parse_bet_sizes(&["50%".into()]);
        let (oop_cfvs, ip_cfvs, valid_mask, _oop_gv, _ip_gv) = solve_turn_situation(
            sit.board_cards(),
            f64::from(sit.pot),
            f64::from(sit.effective_stack),
            &sit.ranges,
            &[bet_sizes_f64],
            10,
            evaluator,
        );

        // Write a single record with both players' CFVs and verify round-trip.
        let output = NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(output.path()).unwrap();
            let mut writer = BufWriter::new(file);

            let board_vec = sit.board_cards().to_vec();

            let rec = TrainingRecord {
                board: board_vec,
                pot: sit.pot as f32,
                effective_stack: sit.effective_stack as f32,
                oop_range: sit.ranges[0],
                ip_range: sit.ranges[1],
                oop_cfvs,
                ip_cfvs,
                valid_mask,
            };
            write_record(&mut writer, &rec).unwrap();
        }

        // Read back and verify.
        let mut file = std::fs::File::open(output.path()).unwrap();
        let rec0 = storage::read_record(&mut file).unwrap();

        assert_eq!(rec0.board.len(), 4, "record should have 4-card board");
        assert!(rec0.pot > 0.0);
        assert!(rec0.effective_stack > 0.0);
    }
}
