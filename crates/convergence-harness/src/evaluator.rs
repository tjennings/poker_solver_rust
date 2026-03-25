use crate::solver_trait::{ComboEvMap, StrategyMap};
use range_solver::{compute_exploitability, PostFlopGame};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Compute a stable node ID from an action history.
fn node_id(history: &[usize]) -> u64 {
    let mut hasher = DefaultHasher::new();
    history.hash(&mut hasher);
    hasher.finish()
}

/// Navigate back to parent position after visiting a child.
/// Uses back_to_root + apply_history since PostFlopGame has no pop/undo.
pub(crate) fn navigate_back(game: &mut PostFlopGame, history: &[usize]) {
    game.back_to_root();
    if !history.is_empty() {
        game.apply_history(history);
    }
}

/// Generic DFS walk of the game tree. Calls `visit` at each decision node
/// with the current node ID and the game positioned at that node.
fn walk_tree(
    game: &mut PostFlopGame,
    history: &mut Vec<usize>,
    visit: &mut impl FnMut(&mut PostFlopGame, u64),
) {
    if game.is_terminal_node() {
        return;
    }

    if game.is_chance_node() {
        let possible = game.possible_cards();
        for card in 0..52u8 {
            if possible & (1u64 << card) != 0 {
                history.push(card as usize);
                game.play(card as usize);
                walk_tree(game, history, visit);
                history.pop();
                navigate_back(game, history);
            }
        }
        return;
    }

    // Decision node -- invoke visitor
    let nid = node_id(history);
    visit(game, nid);

    // Recurse into children
    let num_actions = game.available_actions().len();
    for action_idx in 0..num_actions {
        history.push(action_idx);
        game.play(action_idx);
        walk_tree(game, history, visit);
        history.pop();
        navigate_back(game, history);
    }
}

/// Walk the game tree and extract strategy at every decision node.
pub fn extract_strategy(game: &mut PostFlopGame) -> StrategyMap {
    let mut result = StrategyMap::new();
    game.back_to_root();
    let mut history = Vec::new();
    walk_tree(game, &mut history, &mut |g, nid| {
        result.insert(nid, g.strategy());
    });
    game.back_to_root();
    result
}

/// Extract combo EVs at every decision node. Game must be finalized.
pub fn extract_combo_evs(game: &mut PostFlopGame) -> ComboEvMap {
    let mut result = ComboEvMap::new();
    game.back_to_root();
    let mut history = Vec::new();
    walk_tree(game, &mut history, &mut |g, nid| {
        g.cache_normalized_weights();
        let oop_ev = g.expected_values(0);
        let ip_ev = g.expected_values(1);
        result.insert(nid, [oop_ev, ip_ev]);
    });
    game.back_to_root();
    result
}

/// Compute exploitability of the current strategy.
pub fn exploitability(game: &PostFlopGame) -> f64 {
    compute_exploitability(game) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver_trait::ConvergenceSolver;
    use crate::solvers::exhaustive::ExhaustiveSolver;
    use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::{card_from_str, flop_from_str};
    use range_solver::CardConfig;

    /// Build a minimal river game for testing: narrow ranges, single river card,
    /// simple bet sizes. Fast to construct and solve.
    fn build_test_river_game() -> PostFlopGame {
        let oop_range = "AA,KK,QQ,AKs,AKo";
        let ip_range = "JJ,TT,99,AQs,AQo";

        let card_config = CardConfig {
            range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
            flop: flop_from_str("Td9d6h").unwrap(),
            turn: card_from_str("2c").unwrap(),
            river: card_from_str("3s").unwrap(),
        };

        let bet_sizes = BetSizeOptions::try_from(("50%,a", "2.5x")).unwrap();

        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 200,
            rake_rate: 0.0,
            rake_cap: 0.0,
            flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
            turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
            river_bet_sizes: [bet_sizes.clone(), bet_sizes],
            turn_donk_sizes: None,
            river_donk_sizes: None,
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.1,
            depth_limit: None,
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        PostFlopGame::with_config(card_config, action_tree).unwrap()
    }

    // -- node_id tests --

    #[test]
    fn test_node_id_empty_history() {
        let id = node_id(&[]);
        assert_eq!(id, node_id(&[]));
    }

    #[test]
    fn test_node_id_different_histories_produce_different_ids() {
        let id1 = node_id(&[0]);
        let id2 = node_id(&[1]);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_node_id_same_history_produces_same_id() {
        let id1 = node_id(&[0, 1, 2]);
        let id2 = node_id(&[0, 1, 2]);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_node_id_order_matters() {
        let id1 = node_id(&[0, 1]);
        let id2 = node_id(&[1, 0]);
        assert_ne!(id1, id2);
    }

    // -- extract_strategy tests --

    #[test]
    fn test_extract_strategy_returns_nonempty_map() {
        let game = build_test_river_game();
        let mut solver = ExhaustiveSolver::new(game);
        for _ in 0..20 {
            solver.solve_step();
        }
        let mut game = solver.into_game();
        let strategy = extract_strategy(&mut game);
        assert!(!strategy.is_empty(), "Strategy map should not be empty");
    }

    #[test]
    fn test_extract_strategy_root_node_present() {
        let game = build_test_river_game();
        let mut solver = ExhaustiveSolver::new(game);
        for _ in 0..20 {
            solver.solve_step();
        }
        let mut game = solver.into_game();
        let strategy = extract_strategy(&mut game);
        let root_id = node_id(&[]);
        assert!(
            strategy.contains_key(&root_id),
            "Root node should be present in strategy"
        );
    }

    #[test]
    fn test_extract_strategy_probabilities_are_valid() {
        let game = build_test_river_game();
        let mut solver = ExhaustiveSolver::new(game);
        for _ in 0..20 {
            solver.solve_step();
        }
        let mut game = solver.into_game();
        let num_hands_oop = game.private_cards(0).len();
        let num_hands_ip = game.private_cards(1).len();
        let strategy = extract_strategy(&mut game);

        for (_nid, strat) in &strategy {
            assert!(!strat.is_empty(), "Strategy vector should not be empty");
            // All values should be non-negative (probabilities)
            for &val in strat.iter() {
                assert!(
                    val >= -1e-6,
                    "Strategy probability should be >= 0, got {}",
                    val
                );
            }
            // Check if it divides evenly by either player's hand count
            let num_hands = if strat.len() % num_hands_oop == 0 {
                num_hands_oop
            } else {
                num_hands_ip
            };
            let num_actions = strat.len() / num_hands;
            if num_actions > 0 && num_hands > 0 {
                for h in 0..num_hands {
                    let sum: f32 = (0..num_actions).map(|a| strat[a * num_hands + h]).sum();
                    assert!(
                        (sum - 1.0).abs() < 0.01,
                        "Strategy probabilities should sum to ~1.0 for each hand, got {}",
                        sum
                    );
                }
            }
        }
    }

    #[test]
    fn test_extract_strategy_has_multiple_nodes() {
        let game = build_test_river_game();
        let mut solver = ExhaustiveSolver::new(game);
        for _ in 0..20 {
            solver.solve_step();
        }
        let mut game = solver.into_game();
        let strategy = extract_strategy(&mut game);
        assert!(
            strategy.len() >= 2,
            "Should have at least 2 decision nodes, got {}",
            strategy.len()
        );
    }

    // -- extract_combo_evs tests --

    #[test]
    fn test_extract_combo_evs_returns_nonempty_after_finalize() {
        let game = build_test_river_game();
        let mut solver = ExhaustiveSolver::new(game);
        for _ in 0..20 {
            solver.solve_step();
        }
        solver.finalize();
        let mut game = solver.into_game();
        let evs = extract_combo_evs(&mut game);
        assert!(!evs.is_empty(), "Combo EV map should not be empty");
    }

    #[test]
    fn test_extract_combo_evs_root_node_present() {
        let game = build_test_river_game();
        let mut solver = ExhaustiveSolver::new(game);
        for _ in 0..20 {
            solver.solve_step();
        }
        solver.finalize();
        let mut game = solver.into_game();
        let evs = extract_combo_evs(&mut game);
        let root_id = node_id(&[]);
        assert!(
            evs.contains_key(&root_id),
            "Root node should be present in combo EVs"
        );
    }

    #[test]
    fn test_extract_combo_evs_has_both_player_vectors() {
        let game = build_test_river_game();
        let mut solver = ExhaustiveSolver::new(game);
        for _ in 0..20 {
            solver.solve_step();
        }
        solver.finalize();
        let mut game = solver.into_game();
        let evs = extract_combo_evs(&mut game);
        let root_id = node_id(&[]);
        let [oop_ev, ip_ev] = evs.get(&root_id).expect("Root node should exist");
        assert!(!oop_ev.is_empty(), "OOP EV vector should not be empty");
        assert!(!ip_ev.is_empty(), "IP EV vector should not be empty");
    }

    // -- exploitability tests --

    #[test]
    fn test_exploitability_returns_positive_before_solving() {
        let game = build_test_river_game();
        let solver = ExhaustiveSolver::new(game);
        let expl = exploitability(solver.game());
        assert!(expl > 0.0, "Initial exploitability should be positive");
    }

    #[test]
    fn test_exploitability_decreases_after_solving() {
        let game = build_test_river_game();
        let mut solver = ExhaustiveSolver::new(game);
        let expl_before = exploitability(solver.game());
        for _ in 0..20 {
            solver.solve_step();
        }
        let expl_after = exploitability(solver.game());
        assert!(
            expl_after < expl_before,
            "Exploitability should decrease after solving: before={}, after={}",
            expl_before,
            expl_after
        );
    }
}
