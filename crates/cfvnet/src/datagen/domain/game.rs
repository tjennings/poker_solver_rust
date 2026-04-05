use rand::Rng;

use crate::datagen::sampler::Situation;
use range_solver::game::PostFlopGame;
use range_solver::interface::Game as RsGame;

use super::evaluator::SolveStrategy;

/// A turn game tree built from a situation.
/// Thin wrapper over PostFlopGame with domain-level boundary access.
pub struct Game {
    situation: Situation,
    tree: PostFlopGame,
}

impl Game {
    pub fn new(situation: Situation, tree: PostFlopGame) -> Self {
        Self { situation, tree }
    }

    // Accessors

    pub fn situation(&self) -> &Situation {
        &self.situation
    }

    /// Access the underlying PostFlopGame for GPU topology extraction.
    pub fn inner(&self) -> &PostFlopGame {
        &self.tree
    }

    pub fn num_private_hands(&self, player: usize) -> usize {
        self.tree.num_private_hands(player)
    }

    pub fn private_cards(&self, player: usize) -> &[(u8, u8)] {
        self.tree.private_cards(player)
    }

    // Boundary methods

    pub fn num_boundaries(&self) -> usize {
        self.tree.num_boundary_nodes()
    }

    pub fn boundary_pot(&self, ordinal: usize) -> i32 {
        self.tree.boundary_pot(ordinal)
    }

    pub fn boundary_cfvs_empty(&self, ordinal: usize, player: usize) -> bool {
        self.tree.boundary_cfvs_empty(ordinal, player)
    }

    pub fn set_boundary_cfvs(&self, ordinal: usize, player: usize, cfvs: Vec<f32>) {
        self.tree.set_boundary_cfvs(ordinal, player, cfvs);
    }

    pub fn boundary_reach(&self, ordinal: usize, player: usize) -> Vec<f32> {
        self.tree.boundary_reach(ordinal, player)
    }

    // Solver operations

    pub fn solve_step(&self, iteration: u32) {
        range_solver::solve_step(&self.tree, iteration);
    }

    pub fn finalize(&mut self) {
        range_solver::finalize(&mut self.tree);
    }

    pub fn compute_exploitability(&self) -> f32 {
        range_solver::compute_exploitability(&self.tree)
    }

    pub fn expected_values(&self, player: usize) -> Vec<f32> {
        self.tree.expected_values(player)
    }

    pub fn back_to_root(&mut self) {
        self.tree.back_to_root();
    }

    pub fn cache_normalized_weights(&mut self) {
        self.tree.cache_normalized_weights();
    }
}

/// Builds a Game from a Situation by delegating to the existing
/// `build_turn_game` / `build_turn_game_exact` functions.
pub struct GameBuilder {
    bet_sizes: Vec<Vec<f64>>,
    pub(crate) exact: bool,
    pub(crate) fuzz: f64,
}

impl GameBuilder {
    pub fn new(bet_sizes: Vec<Vec<f64>>, strategy: &SolveStrategy) -> Self {
        Self {
            bet_sizes,
            exact: matches!(strategy, SolveStrategy::Exact),
            fuzz: 0.0,
        }
    }

    pub fn with_fuzz(mut self, fuzz: f64) -> Self {
        self.fuzz = fuzz;
        self
    }

    pub fn build(&self, sit: &Situation, rng: &mut impl Rng) -> Option<Game> {
        let sizes = if self.fuzz > 0.0 {
            super::game_tree::fuzz_bet_sizes(&self.bet_sizes, self.fuzz, rng)
        } else {
            self.bet_sizes.clone()
        };
        let tree = if self.exact {
            super::game_tree::build_turn_game_exact(
                sit.board_cards(),
                f64::from(sit.pot),
                f64::from(sit.effective_stack),
                &sit.ranges,
                &sizes,
            )?
        } else {
            super::game_tree::build_turn_game(
                sit.board_cards(),
                f64::from(sit.pot),
                f64::from(sit.effective_stack),
                &sit.ranges,
                &sizes,
            )?
        };
        Some(Game::new(sit.clone(), tree))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DatagenConfig;
    use crate::datagen::domain::evaluator::{BoundaryEvaluator, BoundaryCfvs, SolveStrategy};
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

    fn depth_limited_strategy() -> SolveStrategy {
        SolveStrategy::DepthLimited {
            evaluator: Arc::new(MockEvaluator),
        }
    }

    #[test]
    fn game_builder_produces_game_with_boundaries() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = DatagenConfig::default();
        let sit = sample_situation(&config, 200, 4, &mut rng);
        let strategy = depth_limited_strategy();
        let builder = GameBuilder::new(vec![vec![0.5, 1.0]], &strategy);
        if sit.effective_stack <= 0 {
            return;
        }
        let game = builder.build(&sit, &mut rng).expect("should build");
        assert!(game.num_boundaries() > 0, "turn game should have boundaries");
    }

    #[test]
    fn game_boundary_methods_delegate_correctly() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = DatagenConfig::default();
        let sit = sample_situation(&config, 200, 4, &mut rng);
        let strategy = depth_limited_strategy();
        let builder = GameBuilder::new(vec![vec![0.5, 1.0]], &strategy);
        if sit.effective_stack <= 0 {
            return;
        }
        let game = builder.build(&sit, &mut rng).expect("should build");
        let pot = game.boundary_pot(0);
        assert!(pot > 0, "boundary pot should be positive");
        assert!(
            game.boundary_cfvs_empty(0, 0),
            "cfvs should start empty"
        );
    }

    #[test]
    fn game_exposes_situation() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = DatagenConfig::default();
        let sit = sample_situation(&config, 200, 4, &mut rng);
        if sit.effective_stack <= 0 {
            return;
        }
        let strategy = depth_limited_strategy();
        let builder = GameBuilder::new(vec![vec![0.5, 1.0]], &strategy);
        let game = builder.build(&sit, &mut rng).expect("should build");
        assert_eq!(game.situation().pot, sit.pot);
        assert_eq!(game.situation().effective_stack, sit.effective_stack);
    }

    #[test]
    fn game_builder_stores_exact_flag() {
        let exact_builder = GameBuilder::new(vec![vec![0.5, 1.0]], &SolveStrategy::Exact);
        assert!(exact_builder.exact);
        let dl_builder = GameBuilder::new(vec![vec![0.5, 1.0]], &depth_limited_strategy());
        assert!(!dl_builder.exact);
    }

    #[test]
    fn game_builder_with_fuzz_sets_fuzz() {
        let strategy = depth_limited_strategy();
        let builder = GameBuilder::new(vec![vec![0.5, 1.0]], &strategy).with_fuzz(0.1);
        assert!((builder.fuzz - 0.1).abs() < 1e-12);
    }

    #[test]
    fn game_builder_default_fuzz_is_zero() {
        let strategy = depth_limited_strategy();
        let builder = GameBuilder::new(vec![vec![0.5, 1.0]], &strategy);
        assert!((builder.fuzz - 0.0).abs() < 1e-12);
    }

    #[test]
    fn game_builder_with_fuzz_builds_game() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = DatagenConfig::default();
        let sit = sample_situation(&config, 200, 4, &mut rng);
        if sit.effective_stack <= 0 {
            return;
        }
        let strategy = depth_limited_strategy();
        let builder = GameBuilder::new(vec![vec![0.5, 1.0]], &strategy).with_fuzz(0.1);
        let game = builder.build(&sit, &mut rng);
        assert!(game.is_some(), "fuzzed build should produce a game");
    }

    #[test]
    fn game_builder_exact_mode_builds_game_for_river() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = DatagenConfig::default();
        // River board (5 cards) with exact mode.
        let sit = sample_situation(&config, 200, 5, &mut rng);
        if sit.effective_stack <= 0 {
            return;
        }
        let builder = GameBuilder::new(vec![vec![0.5, 1.0]], &SolveStrategy::Exact);
        let game = builder.build(&sit, &mut rng);
        assert!(game.is_some(), "exact mode should build a river game");
        let game = game.unwrap();
        assert_eq!(
            game.num_boundaries(),
            0,
            "exact mode river game should have no boundaries"
        );
    }
}
