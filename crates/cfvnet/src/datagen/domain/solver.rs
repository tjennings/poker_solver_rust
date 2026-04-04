use std::sync::Arc;

use range_solver::card::card_pair_to_index;

use super::evaluator::BoundaryEvaluator;
use super::game::Game;
use crate::datagen::range_gen::NUM_COMBOS;
use crate::datagen::storage::TrainingRecord;

/// Configuration for the DCFR solver.
#[derive(Clone)]
pub struct SolverConfig {
    pub max_iterations: u32,
    pub target_exploitability: Option<f32>,
    pub leaf_eval_interval: u32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            target_exploitability: None,
            leaf_eval_interval: 0, // 0 = evaluate once at start
        }
    }
}

/// A solved game with computed exploitability.
pub struct SolvedGame {
    pub game: Game,
    pub exploitability: f32,
}

impl SolvedGame {
    /// Extract training records (OOP + IP) from the solved game.
    pub fn extract_records(&self) -> Vec<TrainingRecord> {
        let sit = self.game.situation();
        let pot = f64::from(sit.pot);
        let half_pot = pot / 2.0;
        let norm = if half_pot > 0.0 { half_pot } else { 1.0 };

        let raw_oop = self.game.expected_values(0);
        let raw_ip = self.game.expected_values(1);
        let oop_hands = self.game.private_cards(0);
        let ip_hands = self.game.private_cards(1);

        let mut oop_cfvs = [0.0_f32; NUM_COMBOS];
        let mut ip_cfvs = [0.0_f32; NUM_COMBOS];
        let mut valid_mask = [0_u8; NUM_COMBOS];

        for (i, &(c0, c1)) in oop_hands.iter().enumerate() {
            let idx = card_pair_to_index(c0, c1);
            oop_cfvs[idx] = ((f64::from(raw_oop[i]) - half_pot) / norm) as f32;
            valid_mask[idx] = 1;
        }
        for (i, &(c0, c1)) in ip_hands.iter().enumerate() {
            let idx = card_pair_to_index(c0, c1);
            ip_cfvs[idx] = ((f64::from(raw_ip[i]) - half_pot) / norm) as f32;
            valid_mask[idx] = 1;
        }

        let oop_gv: f32 = sit.ranges[0]
            .iter()
            .zip(oop_cfvs.iter())
            .map(|(&r, &c)| r * c)
            .sum();
        let ip_gv: f32 = sit.ranges[1]
            .iter()
            .zip(ip_cfvs.iter())
            .map(|(&r, &c)| r * c)
            .sum();

        let board = sit.board_cards().to_vec();
        vec![
            TrainingRecord {
                board: board.clone(),
                pot: sit.pot as f32,
                effective_stack: sit.effective_stack as f32,
                player: 0,
                game_value: oop_gv,
                oop_range: sit.ranges[0],
                ip_range: sit.ranges[1],
                cfvs: oop_cfvs,
                valid_mask,
            },
            TrainingRecord {
                board,
                pot: sit.pot as f32,
                effective_stack: sit.effective_stack as f32,
                player: 1,
                game_value: ip_gv,
                oop_range: sit.ranges[0],
                ip_range: sit.ranges[1],
                cfvs: ip_cfvs,
                valid_mask,
            },
        ]
    }
}

/// Step-based DCFR solver wrapping range_solver primitives.
pub struct Solver {
    game: Option<Game>,
    config: SolverConfig,
    evaluator: Arc<dyn BoundaryEvaluator>,
    iteration: u32,
    boundaries_set: bool,
}

impl Solver {
    pub fn new(game: Game, config: &SolverConfig, evaluator: Arc<dyn BoundaryEvaluator>) -> Self {
        Self {
            game: Some(game),
            config: config.clone(),
            evaluator,
            iteration: 0,
            boundaries_set: false,
        }
    }

    /// Run one DCFR iteration. Returns `Some(SolvedGame)` when done.
    pub fn step(&mut self) -> Option<SolvedGame> {
        let game = self.game.as_ref().expect("step called after finish");

        // Evaluate boundaries if needed.
        let needs_eval = !self.boundaries_set
            || (self.config.leaf_eval_interval > 0
                && self.iteration > 0
                && self.iteration % self.config.leaf_eval_interval == 0);

        if needs_eval {
            let cfvs = self.evaluator.evaluate(game);
            for bc in cfvs {
                game.set_boundary_cfvs(bc.ordinal, bc.player, bc.cfvs);
            }
            self.boundaries_set = true;
        }

        // Run one DCFR iteration.
        game.solve_step(self.iteration);
        self.iteration += 1;

        // Check if done by max iterations.
        if self.iteration >= self.config.max_iterations {
            return Some(self.finish());
        }

        // Check exploitability target (every 10 iterations to avoid overhead).
        if let Some(target) = self.config.target_exploitability {
            if self.iteration % 10 == 0 {
                let game = self.game.as_ref().expect("game present");
                let cfvs = self.evaluator.evaluate(game);
                for bc in &cfvs {
                    game.set_boundary_cfvs(bc.ordinal, bc.player, bc.cfvs.clone());
                }
                let exploit = game.compute_exploitability();
                let abs_target = target * game.situation().pot as f32;
                if exploit <= abs_target {
                    return Some(self.finish());
                }
            }
        }

        None
    }

    fn finish(&mut self) -> SolvedGame {
        let mut game = self.game.take().expect("finish called twice");

        // Ensure boundaries are set for final exploitability computation.
        let cfvs = self.evaluator.evaluate(&game);
        for bc in cfvs {
            game.set_boundary_cfvs(bc.ordinal, bc.player, bc.cfvs);
        }

        game.finalize();
        game.back_to_root();
        game.cache_normalized_weights();

        let exploit = game.compute_exploitability();

        SolvedGame {
            game,
            exploitability: exploit,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DatagenConfig;
    use crate::datagen::domain::evaluator::{BoundaryCfvs, BoundaryEvaluator};
    use crate::datagen::domain::game::{Game, GameBuilder};
    use crate::datagen::sampler::sample_situation;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::sync::Arc;

    struct ZeroEvaluator;
    impl BoundaryEvaluator for ZeroEvaluator {
        fn evaluate(&self, game: &Game) -> Vec<BoundaryCfvs> {
            (0..game.num_boundaries())
                .flat_map(|ord| {
                    (0..2).map(move |player| BoundaryCfvs {
                        ordinal: ord,
                        player,
                        cfvs: vec![0.0; game.num_private_hands(player)],
                    })
                })
                .collect()
        }
    }

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
    fn solver_config_default_values() {
        let config = SolverConfig::default();
        assert_eq!(config.max_iterations, 1000);
        assert!(config.target_exploitability.is_none());
        assert_eq!(config.leaf_eval_interval, 0);
    }

    #[test]
    fn solver_produces_solved_game_with_finite_records() {
        range_solver::set_force_sequential(true);
        let game = build_test_game();
        let eval: Arc<dyn BoundaryEvaluator> = Arc::new(ZeroEvaluator);
        let solver_config = SolverConfig {
            max_iterations: 50,
            ..Default::default()
        };
        let mut solver = Solver::new(game, &solver_config, eval);

        let solved = loop {
            match solver.step() {
                None => continue,
                Some(sg) => break sg,
            }
        };

        assert!(solved.exploitability.is_finite());
        let records = solved.extract_records();
        assert_eq!(records.len(), 2); // OOP + IP
        for rec in &records {
            assert_eq!(rec.board.len(), 4);
            assert!(rec.game_value.is_finite());
            for &cfv in &rec.cfvs {
                assert!(cfv.is_finite());
            }
        }
    }

    #[test]
    fn solver_step_returns_none_before_max_iterations() {
        range_solver::set_force_sequential(true);
        let game = build_test_game();
        let eval: Arc<dyn BoundaryEvaluator> = Arc::new(ZeroEvaluator);
        let solver_config = SolverConfig {
            max_iterations: 10,
            ..Default::default()
        };
        let mut solver = Solver::new(game, &solver_config, eval);

        // First 9 steps should return None
        for _ in 0..9 {
            assert!(solver.step().is_none());
        }
        // 10th step should return SolvedGame
        assert!(solver.step().is_some());
    }

    #[test]
    fn solved_game_records_have_correct_player_ids() {
        range_solver::set_force_sequential(true);
        let game = build_test_game();
        let eval: Arc<dyn BoundaryEvaluator> = Arc::new(ZeroEvaluator);
        let solver_config = SolverConfig {
            max_iterations: 20,
            ..Default::default()
        };
        let mut solver = Solver::new(game, &solver_config, eval);

        let solved = loop {
            match solver.step() {
                None => continue,
                Some(sg) => break sg,
            }
        };

        let records = solved.extract_records();
        assert_eq!(records[0].player, 0);
        assert_eq!(records[1].player, 1);
    }

    #[test]
    fn solved_game_records_have_matching_pot_and_stack() {
        range_solver::set_force_sequential(true);
        let game = build_test_game();
        let expected_pot = game.situation().pot as f32;
        let expected_stack = game.situation().effective_stack as f32;
        let eval: Arc<dyn BoundaryEvaluator> = Arc::new(ZeroEvaluator);
        let solver_config = SolverConfig {
            max_iterations: 20,
            ..Default::default()
        };
        let mut solver = Solver::new(game, &solver_config, eval);

        let solved = loop {
            match solver.step() {
                None => continue,
                Some(sg) => break sg,
            }
        };

        let records = solved.extract_records();
        for rec in &records {
            assert_eq!(rec.pot, expected_pot);
            assert_eq!(rec.effective_stack, expected_stack);
        }
    }
}
