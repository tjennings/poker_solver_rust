use crate::solver_trait::{ComboEvMap, StrategyMap};
use range_solver::{compute_exploitability, PostFlopGame};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Compute a stable node ID from an action history.
pub(crate) fn node_id(history: &[usize]) -> u64 {
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
}
