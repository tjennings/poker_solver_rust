pub mod evaluator;
pub mod game;
pub mod neural_net_evaluator;
pub mod pipeline;
pub mod situation;
pub mod solver;
pub mod writer;

pub use evaluator::{BoundaryCfvs, BoundaryEvaluator, SolveStrategy};
pub use game::{Game, GameBuilder};
pub use situation::{RangeSource, SituationGenerator};
pub use solver::{SolvedGame, Solver, SolverConfig};
pub use writer::RecordWriter;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DatagenConfig;
    use crate::datagen::domain::evaluator::SolveStrategy;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::sync::Arc;

    /// Zero-valued boundary evaluator for testing pipeline mechanics.
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

    fn depth_limited_strategy() -> SolveStrategy {
        SolveStrategy::DepthLimited {
            evaluator: Arc::new(ZeroEvaluator),
        }
    }

    #[test]
    fn full_domain_pipeline_produces_training_records() {
        range_solver::set_force_sequential(true);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let datagen_config = DatagenConfig::default();
        let sit_gen = SituationGenerator::new(&datagen_config, 200, 4, 42, 5);
        let strategy = depth_limited_strategy();
        let builder = GameBuilder::new(vec![vec![0.5, 1.0]], &strategy);

        let mut total_records = 0;
        for sit in sit_gen {
            let game = match builder.build(&sit, &mut rng) {
                Some(g) => g,
                None => continue,
            };
            let solver_config = SolverConfig {
                max_iterations: 20,
                ..Default::default()
            };
            let mut solver = Solver::new(
                game,
                &solver_config,
                strategy.clone(),
            );
            let solved = loop {
                match solver.step() {
                    None => continue,
                    Some(sg) => break sg,
                }
            };
            assert!(solved.exploitability.is_finite());
            let records = solved.extract_records();
            assert_eq!(records.len(), 2);
            for rec in &records {
                assert_eq!(rec.board.len(), 4);
                assert!(rec.pot > 0.0);
                assert!(rec.effective_stack > 0.0);
                assert!(rec.game_value.is_finite());
                for &cfv in &rec.cfvs {
                    assert!(cfv.is_finite());
                }
            }
            total_records += records.len();
        }
        assert!(total_records > 0, "should produce at least some records");
    }

    #[test]
    fn pipeline_handles_multiple_situations() {
        range_solver::set_force_sequential(true);

        let mut rng = ChaCha8Rng::seed_from_u64(77);
        let datagen_config = DatagenConfig::default();
        let sit_gen = SituationGenerator::new(&datagen_config, 200, 4, 77, 3);
        let strategy = depth_limited_strategy();
        let builder = GameBuilder::new(vec![vec![0.5, 1.0]], &strategy);

        let mut solved_count = 0;
        for sit in sit_gen {
            let game = match builder.build(&sit, &mut rng) {
                Some(g) => g,
                None => continue,
            };
            let solver_config = SolverConfig {
                max_iterations: 10,
                ..Default::default()
            };
            let mut solver = Solver::new(
                game,
                &solver_config,
                strategy.clone(),
            );
            let solved = loop {
                match solver.step() {
                    None => continue,
                    Some(sg) => break sg,
                }
            };
            let records = solved.extract_records();
            assert_eq!(records.len(), 2);
            assert_eq!(records[0].player, 0);
            assert_eq!(records[1].player, 1);
            solved_count += 1;
        }
        assert!(solved_count > 0);
    }
}
