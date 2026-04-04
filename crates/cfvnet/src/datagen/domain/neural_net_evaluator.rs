use std::sync::Mutex;

use burn::backend::wgpu::Wgpu;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
use poker_solver_core::blueprint_v2::LeafEvaluator;
use poker_solver_core::poker::Card;
use range_solver::card::card_pair_to_index;

use crate::config::CfvnetConfig;
use crate::datagen::turn_generate::u8_to_rs_card;
use crate::eval::river_net_evaluator::RiverNetEvaluator;
use crate::model::network::{CfvNet, INPUT_SIZE};

use super::evaluator::{BoundaryCfvs, BoundaryEvaluator};
use super::game::Game;

type B = Wgpu;

/// Implements `BoundaryEvaluator` by wrapping a GPU-backed `RiverNetEvaluator`.
///
/// Bridges the domain trait to the existing `LeafEvaluator::evaluate_boundaries`
/// method, converting Game boundary info into the format expected by the neural net.
/// Uses a Mutex internally because `RiverNetEvaluator<Wgpu>` is not Sync.
pub struct NeuralNetEvaluator {
    evaluator: Mutex<RiverNetEvaluator<B>>,
}

impl NeuralNetEvaluator {
    /// Load a river model from disk and wrap it as a `NeuralNetEvaluator`.
    ///
    /// Reads the model config from `config.yaml` next to the model file,
    /// falling back to the main config architecture if not found.
    pub fn load(model_path: &str, config: &CfvnetConfig) -> Result<Self, String> {
        let model_dir = std::path::Path::new(model_path)
            .parent()
            .ok_or("river_model_path has no parent directory")?;
        let river_config_path = model_dir.join("config.yaml");
        let (hidden_layers, hidden_size) = if river_config_path.exists() {
            let yaml = std::fs::read_to_string(&river_config_path)
                .map_err(|e| format!("read river config: {e}"))?;
            let river_cfg: CfvnetConfig = serde_yaml::from_str(&yaml)
                .map_err(|e| format!("parse river config: {e}"))?;
            eprintln!(
                "[domain] river model architecture: {}x{} (from {})",
                river_cfg.training.hidden_layers,
                river_cfg.training.hidden_size,
                river_config_path.display()
            );
            (
                river_cfg.training.hidden_layers,
                river_cfg.training.hidden_size,
            )
        } else {
            eprintln!("[domain] warning: no river config.yaml found, using config architecture");
            (config.training.hidden_layers, config.training.hidden_size)
        };

        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
        let model = CfvNet::<B>::new(&device, hidden_layers, hidden_size, INPUT_SIZE)
            .load_file(model_path, &recorder, &device)
            .map_err(|e| format!("failed to load river model: {e}"))?;
        eprintln!("[domain] river model loaded on wgpu");

        Ok(Self {
            evaluator: Mutex::new(RiverNetEvaluator::new(model, device)),
        })
    }
}

impl BoundaryEvaluator for NeuralNetEvaluator {
    fn evaluate(&self, game: &Game) -> Vec<BoundaryCfvs> {
        let num_boundaries = game.num_boundaries();
        if num_boundaries == 0 {
            return Vec::new();
        }

        let sit = game.situation();
        let board_u8 = sit.board_cards();
        let pot = f64::from(sit.pot);
        let effective_stack = f64::from(sit.effective_stack);

        // Convert board from u8 to Card.
        let board_cards: Vec<Card> = board_u8.iter().map(|&c| u8_to_rs_card(c)).collect();

        // Both players share the same hands on a 4-card board.
        let hands = game.private_cards(0);

        // Convert to [[Card; 2]] for the evaluator.
        let combos: Vec<[Card; 2]> = hands
            .iter()
            .map(|&(c0, c1)| [u8_to_rs_card(c0), u8_to_rs_card(c1)])
            .collect();

        // Build per-combo range arrays from the 1326-indexed input ranges.
        let oop_reach: Vec<f64> = hands
            .iter()
            .map(|&(c0, c1)| f64::from(sit.ranges[0][card_pair_to_index(c0, c1)]))
            .collect();
        let ip_reach: Vec<f64> = hands
            .iter()
            .map(|&(c0, c1)| f64::from(sit.ranges[1][card_pair_to_index(c0, c1)]))
            .collect();

        // Collect all (pot, eff_stack, player) requests.
        let mut requests: Vec<(f64, f64, u8)> = Vec::with_capacity(num_boundaries * 2);
        for ordinal in 0..num_boundaries {
            let bpot = game.boundary_pot(ordinal) as f64;
            let eff_stack_at_boundary = effective_stack - (bpot - pot) / 2.0;
            for player in 0..2u8 {
                requests.push((bpot, eff_stack_at_boundary, player));
            }
        }

        // One batched call -- GPU does one forward pass per river card.
        let evaluator = self.evaluator.lock().expect("evaluator lock poisoned");
        let all_cfvs =
            evaluator.evaluate_boundaries(&combos, &board_cards, &oop_reach, &ip_reach, &requests);

        // Convert Vec<Vec<f64>> results to Vec<BoundaryCfvs>.
        let mut result = Vec::with_capacity(num_boundaries * 2);

        for ordinal in 0..num_boundaries {
            for player in 0..2usize {
                let req_idx = ordinal * 2 + player;
                let cfvs_f32: Vec<f32> = all_cfvs[req_idx].iter().map(|&v| v as f32).collect();

                result.push(BoundaryCfvs {
                    ordinal,
                    player,
                    cfvs: cfvs_f32,
                });
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DatagenConfig;
    use crate::datagen::domain::game::GameBuilder;
    use crate::datagen::sampler::sample_situation;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn build_test_game() -> Game {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = DatagenConfig::default();
        loop {
            let sit = sample_situation(&config, 200, 4, &mut rng);
            if sit.effective_stack <= 0 {
                continue;
            }
            let builder = GameBuilder::new(vec![vec![0.5, 1.0]]);
            if let Some(game) = builder.build(&sit) {
                return game;
            }
        }
    }

    #[test]
    fn neural_net_evaluator_is_object_safe() {
        // Verify that NeuralNetEvaluator can be used as Box<dyn BoundaryEvaluator>.
        // This test only needs to compile — it validates the trait impl exists.
        fn _assert_object_safe(_: Box<dyn BoundaryEvaluator>) {}
    }

    #[test]
    fn bridge_builds_correct_boundary_requests() {
        // Verify the boundary request construction logic.
        let game = build_test_game();
        let num_boundaries = game.num_boundaries();
        assert!(num_boundaries > 0, "turn game should have boundary nodes");

        // Verify boundary pot is positive for all ordinals.
        for ord in 0..num_boundaries {
            let pot = game.boundary_pot(ord);
            assert!(pot > 0, "boundary pot should be positive, got {pot}");
        }
    }

    #[test]
    fn bridge_converts_board_to_cards() {
        let game = build_test_game();
        let board_u8 = game.situation().board_cards();
        assert_eq!(board_u8.len(), 4);

        let board_cards: Vec<Card> = board_u8.iter().map(|&c| u8_to_rs_card(c)).collect();
        assert_eq!(board_cards.len(), 4);
    }

    #[test]
    fn bridge_converts_combos_to_card_pairs() {
        let game = build_test_game();
        let hands = game.private_cards(0);
        assert!(!hands.is_empty());

        let combos: Vec<[Card; 2]> = hands
            .iter()
            .map(|&(c0, c1)| [u8_to_rs_card(c0), u8_to_rs_card(c1)])
            .collect();
        assert_eq!(combos.len(), hands.len());
    }

    #[test]
    fn bridge_builds_range_arrays_from_situation() {
        let game = build_test_game();
        let hands = game.private_cards(0);
        let ranges = &game.situation().ranges;

        let oop_reach: Vec<f64> = hands
            .iter()
            .map(|&(c0, c1)| f64::from(ranges[0][card_pair_to_index(c0, c1)]))
            .collect();
        assert_eq!(oop_reach.len(), hands.len());

        // At least some combos should have non-zero reach.
        assert!(oop_reach.iter().any(|&r| r > 0.0));
    }

    #[test]
    fn bridge_computes_eff_stack_at_boundary() {
        let game = build_test_game();
        let root_pot = f64::from(game.situation().pot);
        let root_eff_stack = f64::from(game.situation().effective_stack);

        for ord in 0..game.num_boundaries() {
            let bpot = game.boundary_pot(ord) as f64;
            let eff_stack_at_boundary = root_eff_stack - (bpot - root_pot) / 2.0;
            // Effective stack at boundary should be less than or equal to root.
            assert!(
                eff_stack_at_boundary <= root_eff_stack + 0.01,
                "eff_stack_at_boundary={eff_stack_at_boundary} > root={root_eff_stack}"
            );
        }
    }
}
