//! Subgame Solver for real-time strategy computation
//!
//! Provides real-time strategy lookup with caching. For now it performs
//! cache + blueprint lookup. Full depth-limited CFR solving is future work.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::abstraction::{CardAbstraction, Street};
use crate::game::Action;
use crate::info_key::{InfoKey, canonical_hand_index, encode_action};
use crate::poker::Card;

use super::{BlueprintError, BlueprintStrategy, CacheConfig, SubgameCache, SubgameKey};

/// Configuration for subgame solving
#[derive(Debug, Clone)]
pub struct SubgameConfig {
    /// Maximum depth to search in the game tree
    pub depth_limit: usize,
    /// Time budget for solving in milliseconds
    pub time_budget_ms: u64,
    /// Maximum number of CFR iterations
    pub max_iterations: u32,
}

impl Default for SubgameConfig {
    fn default() -> Self {
        Self {
            depth_limit: 4,
            time_budget_ms: 200,
            max_iterations: 1000,
        }
    }
}

/// Subgame solver for real-time strategy lookup
///
/// Combines blueprint strategy with caching for efficient real-time play.
/// Currently performs cache + blueprint lookup; full depth-limited CFR
/// solving will be added in future work.
pub struct SubgameSolver {
    /// The precomputed blueprint strategy
    blueprint: Arc<BlueprintStrategy>,
    /// Optional card abstraction for bucket computation
    abstraction: Option<Arc<CardAbstraction>>,
    /// Cache for solved subgames
    cache: SubgameCache,
    /// Configuration for solving
    config: SubgameConfig,
}

impl SubgameSolver {
    /// Create a new subgame solver
    ///
    /// # Arguments
    /// * `blueprint` - The precomputed blueprint strategy
    /// * `abstraction` - Optional card abstraction for bucket computation
    /// * `config` - Configuration for solving
    /// * `cache_config` - Configuration for the subgame cache
    ///
    /// # Errors
    /// Returns `BlueprintError::CacheError` if cache creation fails
    pub fn new(
        blueprint: Arc<BlueprintStrategy>,
        abstraction: Option<Arc<CardAbstraction>>,
        config: SubgameConfig,
        cache_config: CacheConfig,
    ) -> Result<Self, BlueprintError> {
        let cache = SubgameCache::new(cache_config)?;
        Ok(Self {
            blueprint,
            abstraction,
            cache,
            config,
        })
    }

    /// Returns the solver configuration
    #[must_use]
    pub fn config(&self) -> &SubgameConfig {
        &self.config
    }

    /// Solve for action probabilities at a given game state
    ///
    /// Checks the cache first, then falls back to blueprint lookup.
    ///
    /// # Arguments
    /// * `board` - Community cards (0-5 cards)
    /// * `holding` - Player's hole cards
    /// * `history` - Action history with street context
    ///
    /// # Errors
    /// Returns `BlueprintError::InfoSetNotFound` if no strategy exists for this state
    pub fn solve(
        &self,
        board: &[Card],
        holding: [Card; 2],
        history: &[(Street, Action)],
    ) -> Result<Vec<f32>, BlueprintError> {
        // Create cache key
        let cache_key = self.make_cache_key(board, holding, history)?;

        // Check cache first
        if let Some(probs) = self.cache.get(&cache_key) {
            return Ok(probs);
        }

        // Fall back to blueprint lookup
        let info_set_key = self.make_info_set_key(board, holding, history)?;
        if let Some(probs) = self.blueprint.lookup(info_set_key) {
            // Cache the result for future lookups
            let probs_vec = probs.to_vec();
            // Ignore cache insert errors - strategy lookup succeeded
            let _ = self.cache.insert(cache_key, probs_vec.clone());
            return Ok(probs_vec);
        }

        Err(BlueprintError::InfoSetNotFound(info_set_key))
    }

    /// Create a cache key from game state
    ///
    /// # Arguments
    /// * `board` - Community cards
    /// * `holding` - Player's hole cards
    /// * `history` - Action history
    ///
    /// # Errors
    /// Returns error if street cannot be determined from board length
    pub fn make_cache_key(
        &self,
        board: &[Card],
        holding: [Card; 2],
        history: &[(Street, Action)],
    ) -> Result<SubgameKey, BlueprintError> {
        // Hash board
        let board_hash = {
            let mut hasher = DefaultHasher::new();
            board.hash(&mut hasher);
            hasher.finish()
        };

        // Get bucket from abstraction (0 if preflop or no abstraction)
        let holding_bucket = if board.is_empty() {
            // Preflop: no bucket needed
            0
        } else if let Some(ref abstraction) = self.abstraction {
            abstraction
                .get_bucket(board, (holding[0], holding[1]))
                .unwrap_or(0)
        } else {
            0
        };

        // Hash history
        let history_hash = {
            let mut hasher = DefaultHasher::new();
            for (street, action) in history {
                street.hash(&mut hasher);
                std::mem::discriminant(action).hash(&mut hasher);
                // Hash bet/raise amounts
                match action {
                    Action::Bet(amount) | Action::Raise(amount) => amount.hash(&mut hasher),
                    _ => {}
                }
            }
            hasher.finish()
        };

        Ok(SubgameKey::new(board_hash, holding_bucket, history_hash))
    }

    /// Create a u64 information set key from game state.
    ///
    /// Uses the same `InfoKey` encoding as `HunlPostflop::info_set_key()`.
    ///
    /// # Errors
    /// Returns error if street cannot be determined from board length
    pub fn make_info_set_key(
        &self,
        board: &[Card],
        holding: [Card; 2],
        history: &[(Street, Action)],
    ) -> Result<u64, BlueprintError> {
        let street = Street::from_board_len(board.len())
            .map_err(|e| BlueprintError::InvalidStrategy(format!("invalid board length: {e}")))?;

        let hand_bits: u32 = if board.is_empty() {
            u32::from(canonical_hand_index(holding))
        } else if let Some(ref abstraction) = self.abstraction {
            abstraction
                .get_bucket(board, (holding[0], holding[1]))
                .map_or(0, u32::from)
        } else {
            u32::from(canonical_hand_index(holding))
        };

        let street_num: u8 = match street {
            Street::Preflop => 0,
            Street::Flop => 1,
            Street::Turn => 2,
            Street::River => 3,
        };

        let action_codes: Vec<u8> = history
            .iter()
            .map(|(_, action)| encode_action(*action))
            .collect();

        // pot/stack buckets are 0 here — SubgameSolver doesn't track pot state
        Ok(InfoKey::new(hand_bits, street_num, 0, &action_codes).as_u64())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::items_after_statements)]
    use super::*;
    use test_macros::timed_test;

    fn create_test_blueprint() -> Arc<BlueprintStrategy> {
        use crate::poker::{Suit, Value};
        let mut strategy = BlueprintStrategy::new();
        // AK preflop, no history
        let ak_holding = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
        ];
        let ak_key = InfoKey::new(u32::from(canonical_hand_index(ak_holding)), 0, 0, &[]).as_u64();
        strategy.insert(ak_key, vec![0.3, 0.5, 0.2]);
        // 42 (Four-Two offsuit) on flop with check-bet history
        let ft_holding = [
            Card::new(Value::Four, Suit::Spade),
            Card::new(Value::Two, Suit::Heart),
        ];
        let ft_key = InfoKey::new(
            u32::from(canonical_hand_index(ft_holding)),
            1,
            0,
            &[2, 4], // check, bet(0)
        )
        .as_u64();
        strategy.insert(ft_key, vec![0.4, 0.6]);
        Arc::new(strategy)
    }

    #[timed_test]
    fn subgame_config_default() {
        let config = SubgameConfig::default();

        assert_eq!(config.depth_limit, 4, "default depth_limit should be 4");
        assert_eq!(
            config.time_budget_ms, 200,
            "default time_budget_ms should be 200"
        );
        assert_eq!(
            config.max_iterations, 1000,
            "default max_iterations should be 1000"
        );
    }

    #[timed_test]
    fn subgame_solver_creates() {
        let blueprint = create_test_blueprint();
        let config = SubgameConfig::default();
        let cache_config = CacheConfig::default();

        let result = SubgameSolver::new(blueprint, None, config, cache_config);

        assert!(result.is_ok(), "SubgameSolver creation should succeed");
        let solver = result.expect("should create solver");
        assert_eq!(
            solver.config().depth_limit,
            4,
            "config should be accessible"
        );
    }

    #[timed_test]
    fn subgame_solver_returns_cached() {
        let blueprint = create_test_blueprint();
        let config = SubgameConfig::default();
        let cache_config = CacheConfig::default();
        let solver = SubgameSolver::new(blueprint, None, config, cache_config)
            .expect("should create solver");

        // Create test cards using rs_poker
        use crate::poker::{Suit, Value};
        let holding = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
        ];
        let board: &[Card] = &[];
        let history: &[(Street, Action)] = &[];

        // Create cache key and insert manually
        let cache_key = solver
            .make_cache_key(board, holding, history)
            .expect("should create cache key");
        let expected_probs = vec![0.1, 0.2, 0.7];
        solver
            .cache
            .insert(cache_key, expected_probs.clone())
            .expect("should insert into cache");

        // Solve should return cached value
        let result = solver.solve(board, holding, history);
        assert!(result.is_ok(), "solve should succeed with cached value");
        assert_eq!(
            result.expect("should have probs"),
            expected_probs,
            "should return cached probabilities"
        );
    }

    #[timed_test]
    fn subgame_solver_falls_back_to_blueprint() {
        let blueprint = create_test_blueprint();
        let config = SubgameConfig::default();
        let cache_config = CacheConfig::default();
        let solver = SubgameSolver::new(blueprint, None, config, cache_config)
            .expect("should create solver");

        use crate::poker::{Suit, Value};
        let holding = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
        ];
        let board: &[Card] = &[];
        let history: &[(Street, Action)] = &[];

        // Solve should find from blueprint (key "AK|P|")
        let result = solver.solve(board, holding, history);
        assert!(result.is_ok(), "solve should find strategy from blueprint");
        assert_eq!(
            result.expect("should have probs"),
            vec![0.3, 0.5, 0.2],
            "should return blueprint probabilities"
        );
    }

    #[timed_test]
    fn subgame_solver_returns_info_set_not_found() {
        let blueprint = create_test_blueprint();
        let config = SubgameConfig::default();
        let cache_config = CacheConfig::default();
        let solver = SubgameSolver::new(blueprint, None, config, cache_config)
            .expect("should create solver");

        use crate::poker::{Suit, Value};
        // Use cards that don't have a blueprint entry
        let holding = [
            Card::new(Value::Two, Suit::Spade),
            Card::new(Value::Three, Suit::Heart),
        ];
        let board: &[Card] = &[];
        let history: &[(Street, Action)] = &[];

        let result = solver.solve(board, holding, history);
        assert!(result.is_err(), "solve should fail for unknown info set");
        match result {
            Err(BlueprintError::InfoSetNotFound(_key)) => {}
            other => panic!("expected InfoSetNotFound error, got {other:?}"),
        }
    }

    #[timed_test]
    fn make_info_set_key_preflop_format() {
        let blueprint = create_test_blueprint();
        let config = SubgameConfig::default();
        let cache_config = CacheConfig::default();
        let solver = SubgameSolver::new(blueprint, None, config, cache_config)
            .expect("should create solver");

        use crate::poker::{Suit, Value};
        let holding = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
        ];
        let board: &[Card] = &[];
        let history: &[(Street, Action)] = &[];

        let key = solver
            .make_info_set_key(board, holding, history)
            .expect("should create key");
        let expected = InfoKey::new(u32::from(canonical_hand_index(holding)), 0, 0, &[]).as_u64();
        assert_eq!(key, expected, "preflop key should match InfoKey encoding");
    }

    #[timed_test]
    fn make_info_set_key_with_history() {
        let blueprint = create_test_blueprint();
        let config = SubgameConfig::default();
        let cache_config = CacheConfig::default();
        let solver = SubgameSolver::new(blueprint, None, config, cache_config)
            .expect("should create solver");

        use crate::poker::{Suit, Value};
        let holding = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
        ];
        let board: &[Card] = &[];
        let history = &[
            (Street::Preflop, Action::Raise(1)),
            (Street::Preflop, Action::Call),
        ];

        let key = solver
            .make_info_set_key(board, holding, history)
            .expect("should create key");
        let expected = InfoKey::new(
            u32::from(canonical_hand_index(holding)),
            0,
            0,
            &[10, 3], // raise(1)=10, call=3
        )
        .as_u64();
        assert_eq!(key, expected, "key with history should encode actions");
    }

    #[timed_test]
    fn make_cache_key_is_deterministic() {
        let blueprint = create_test_blueprint();
        let config = SubgameConfig::default();
        let cache_config = CacheConfig::default();
        let solver = SubgameSolver::new(blueprint, None, config, cache_config)
            .expect("should create solver");

        use crate::poker::{Suit, Value};
        let holding = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
        ];
        let board: &[Card] = &[];
        let history: &[(Street, Action)] = &[];

        let key1 = solver
            .make_cache_key(board, holding, history)
            .expect("should create key");
        let key2 = solver
            .make_cache_key(board, holding, history)
            .expect("should create key");

        assert_eq!(
            key1.to_bytes(),
            key2.to_bytes(),
            "same inputs should produce same cache key"
        );
    }
}
