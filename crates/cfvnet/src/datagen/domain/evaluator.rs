use std::sync::Arc;

use super::game::Game;

/// Counterfactual values for a single boundary node.
pub struct BoundaryCfvs {
    pub ordinal: usize,
    pub player: usize,
    pub cfvs: Vec<f32>,
}

/// Evaluates boundary CFVs for a game's boundary nodes.
/// Implementations: neural net (GPU), exact (solve to showdown), zero (testing).
pub trait BoundaryEvaluator: Send + Sync {
    /// Evaluate all boundary nodes and return CFVs for each (ordinal, player).
    fn evaluate(&self, game: &Game) -> Vec<BoundaryCfvs>;
}

/// Strategy for solving a game tree.
#[derive(Clone)]
pub enum SolveStrategy {
    /// Solve to showdown -- no boundaries, no evaluator.
    Exact,
    /// Depth-limited with boundary evaluation.
    DepthLimited {
        evaluator: Arc<dyn BoundaryEvaluator>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DatagenConfig;
    use crate::datagen::domain::game::GameBuilder;
    use crate::datagen::sampler::sample_situation;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::sync::Arc;

    struct MockEvaluator;
    impl BoundaryEvaluator for MockEvaluator {
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

    #[test]
    fn mock_evaluator_produces_cfvs_for_all_boundaries() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = DatagenConfig::default();
        let sit = sample_situation(&config, 200, 4, &mut rng);
        if sit.effective_stack <= 0 {
            return;
        }
        let strategy = SolveStrategy::DepthLimited {
            evaluator: Arc::new(MockEvaluator),
        };
        let builder = GameBuilder::new(vec![vec![0.5, 1.0]], &strategy);
        let game = builder.build(&sit, &mut rng).expect("build");
        let eval = MockEvaluator;
        let results = eval.evaluate(&game);
        assert_eq!(results.len(), game.num_boundaries() * 2);
    }

    #[test]
    fn boundary_cfvs_has_correct_size() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = DatagenConfig::default();
        let sit = sample_situation(&config, 200, 4, &mut rng);
        if sit.effective_stack <= 0 {
            return;
        }
        let strategy = SolveStrategy::DepthLimited {
            evaluator: Arc::new(MockEvaluator),
        };
        let builder = GameBuilder::new(vec![vec![0.5, 1.0]], &strategy);
        let game = builder.build(&sit, &mut rng).expect("build");
        let eval = MockEvaluator;
        let results = eval.evaluate(&game);
        for bc in &results {
            let expected_size = game.num_private_hands(bc.player);
            assert_eq!(bc.cfvs.len(), expected_size);
        }
    }

    #[test]
    fn boundary_evaluator_is_object_safe() {
        // Verify the trait can be used as a trait object (dyn BoundaryEvaluator).
        let eval: Box<dyn BoundaryEvaluator> = Box::new(MockEvaluator);
        let _ = eval; // just need to compile
    }

    #[test]
    fn solve_strategy_exact_variant_exists() {
        let _strategy = SolveStrategy::Exact;
    }

    #[test]
    fn solve_strategy_depth_limited_variant_exists() {
        let eval: Arc<dyn BoundaryEvaluator> = Arc::new(MockEvaluator);
        let _strategy = SolveStrategy::DepthLimited { evaluator: eval };
    }

    #[test]
    fn solve_strategy_is_clone() {
        let eval: Arc<dyn BoundaryEvaluator> = Arc::new(MockEvaluator);
        let strategy = SolveStrategy::DepthLimited {
            evaluator: eval,
        };
        let cloned = strategy.clone();
        // Both should exist without issue.
        let _a = strategy;
        let _b = cloned;
    }
}
