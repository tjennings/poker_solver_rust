use crate::datagen::sampler::Situation;
use range_solver::game::PostFlopGame;

use crate::datagen::range_gen::NUM_COMBOS;

/// A turn game tree built from a situation.
/// Thin wrapper over PostFlopGame with domain-level boundary access.
pub struct Game {
    pub situation: Situation,
    pub tree: PostFlopGame,
}

impl Game {
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
}

/// Builds a Game from a Situation by delegating to the existing
/// `build_turn_game` function.
pub struct GameBuilder {
    pub bet_sizes: Vec<Vec<f64>>,
    pub depth_limit: Option<usize>,
}

impl GameBuilder {
    pub fn new(bet_sizes: Vec<Vec<f64>>) -> Self {
        Self {
            bet_sizes,
            depth_limit: Some(0),
        }
    }

    pub fn build(&self, sit: &Situation) -> Option<Game> {
        let tree = crate::datagen::turn_generate::build_turn_game(
            sit.board_cards(),
            f64::from(sit.pot),
            f64::from(sit.effective_stack),
            &sit.ranges,
            &self.bet_sizes,
        )?;
        Some(Game {
            situation: sit.clone(),
            tree,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DatagenConfig;
    use crate::datagen::sampler::sample_situation;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn game_builder_produces_game_with_boundaries() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = DatagenConfig::default();
        let sit = sample_situation(&config, 200, 4, &mut rng);
        let builder = GameBuilder::new(vec![vec![0.5, 1.0]]);
        if sit.effective_stack <= 0 {
            return;
        }
        let game = builder.build(&sit).expect("should build");
        assert!(game.num_boundaries() > 0, "turn game should have boundaries");
    }

    #[test]
    fn game_boundary_methods_delegate_correctly() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let config = DatagenConfig::default();
        let sit = sample_situation(&config, 200, 4, &mut rng);
        let builder = GameBuilder::new(vec![vec![0.5, 1.0]]);
        if sit.effective_stack <= 0 {
            return;
        }
        let game = builder.build(&sit).expect("should build");
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
        let builder = GameBuilder::new(vec![vec![0.5, 1.0]]);
        let game = builder.build(&sit).expect("should build");
        assert_eq!(game.situation.pot, sit.pot);
        assert_eq!(game.situation.effective_stack, sit.effective_stack);
    }

    #[test]
    fn game_builder_with_default_depth_limit() {
        let builder = GameBuilder::new(vec![vec![0.5, 1.0]]);
        assert_eq!(builder.depth_limit, Some(0));
    }
}
