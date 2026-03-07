//! Scenario resolution: walk a game tree following an action path and
//! extract a 13x13 strategy grid for the target decision node.

use poker_solver_core::blueprint_v2::game_tree::{GameNode, GameTree, TreeAction};
use poker_solver_core::blueprint_v2::storage::BlueprintStorage;
use poker_solver_core::poker::Card;

use crate::blueprint_tui_widgets::CellStrategy;

/// Walk `tree` following `actions` from the root. Returns the arena
/// index of the target node, or `None` if any action string is invalid
/// or the path runs into a terminal.
///
/// Chance nodes are skipped automatically — the walk passes through
/// them to reach the next decision node.
pub fn resolve_action_path(tree: &GameTree, actions: &[String]) -> Option<u32> {
    let mut node_idx = tree.root;

    for action_str in actions {
        // Skip through chance nodes transparently.
        node_idx = skip_chance(tree, node_idx);

        let GameNode::Decision {
            actions: ref node_actions,
            ref children,
            ..
        } = tree.nodes[node_idx as usize]
        else {
            return None;
        };

        let matched = match_action(action_str, node_actions)?;
        node_idx = children[matched];
    }

    // Skip trailing chance node so caller always lands on a decision or terminal.
    node_idx = skip_chance(tree, node_idx);
    Some(node_idx)
}

/// Format a `TreeAction` for display.
pub fn format_tree_action(action: &TreeAction) -> String {
    match action {
        TreeAction::Fold => "fold".to_string(),
        TreeAction::Check => "check".to_string(),
        TreeAction::Call => "call".to_string(),
        TreeAction::AllIn => "all-in".to_string(),
        TreeAction::Bet(v) => format!("bet {v:.1}"),
        TreeAction::Raise(v) => format!("raise {v:.1}"),
    }
}

/// Build the 13x13 strategy grid for the decision node at `node_idx`.
///
/// For preflop, bucket = canonical hand index mod `num_buckets`.
/// For postflop streets, equity-based bucketing is used to match the
/// solver's `AllBuckets::get_bucket()` logic.
#[allow(clippy::cast_possible_truncation)]
pub fn extract_strategy_grid(
    tree: &GameTree,
    storage: &BlueprintStorage,
    node_idx: u32,
    board: &[Card],
) -> [[CellStrategy; 13]; 13] {
    let mut grid: [[CellStrategy; 13]; 13] =
        std::array::from_fn(|_| std::array::from_fn(|_| CellStrategy::default()));

    let GameNode::Decision {
        ref actions,
        street,
        ..
    } = tree.nodes[node_idx as usize]
    else {
        return grid;
    };

    let street_idx = street as u8;
    let num_buckets = storage.bucket_counts[street_idx as usize];
    if num_buckets == 0 {
        return grid;
    }

    let action_labels: Vec<String> = actions.iter().map(format_tree_action).collect();

    for row in 0..13u16 {
        for col in 0..13u16 {
            let bucket = if street_idx == 0 {
                // Preflop: canonical hand index mod bucket count.
                let hand = poker_solver_core::hands::CanonicalHand::from_matrix_position(
                    row as usize, col as usize,
                ).expect("valid 13x13 matrix position");
                (hand.index() as u16) % num_buckets
            } else {
                // Postflop: equity-based bucketing matching the solver.
                let hand = poker_solver_core::hands::CanonicalHand::from_matrix_position(
                    row as usize, col as usize,
                ).expect("valid 13x13 matrix position");
                let combo = hand.combos().into_iter().find(|(c1, c2)| {
                    !board.contains(c1) && !board.contains(c2)
                });
                if let Some((c1, c2)) = combo {
                    let equity = poker_solver_core::showdown_equity::compute_equity(
                        [c1, c2], board,
                    );
                    ((equity * f64::from(num_buckets)) as u16).min(num_buckets - 1)
                } else {
                    0 // hand fully blocked by board
                }
            };

            let probs = storage.average_strategy(node_idx, bucket);
            let cell_actions: Vec<(String, f32)> = action_labels
                .iter()
                .zip(probs.iter())
                .map(|(name, &p)| (name.clone(), p as f32))
                .collect();

            grid[row as usize][col as usize] = CellStrategy {
                actions: cell_actions,
            };
        }
    }

    grid
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Follow chain of `Chance` nodes to reach a non-chance node.
fn skip_chance(tree: &GameTree, mut idx: u32) -> u32 {
    while let GameNode::Chance { child, .. } = tree.nodes[idx as usize] {
        idx = child;
    }
    idx
}

/// Find the index of the action in `node_actions` that matches
/// `action_str`. Returns `None` if no match.
fn match_action(action_str: &str, node_actions: &[TreeAction]) -> Option<usize> {
    let lower = action_str.to_ascii_lowercase();

    if lower == "fold" {
        return node_actions
            .iter()
            .position(|a| matches!(a, TreeAction::Fold));
    }
    if lower == "check" {
        return node_actions
            .iter()
            .position(|a| matches!(a, TreeAction::Check));
    }
    if lower == "call" {
        return node_actions
            .iter()
            .position(|a| matches!(a, TreeAction::Call));
    }
    if lower == "allin" || lower == "all-in" {
        return node_actions
            .iter()
            .position(|a| matches!(a, TreeAction::AllIn));
    }

    // "bet-N" or "raise-N": the N-th occurrence of Bet/Raise in this node.
    if let Some(n_str) = lower.strip_prefix("bet-") {
        let n: usize = n_str.parse().ok()?;
        return node_actions
            .iter()
            .enumerate()
            .filter(|(_, a)| matches!(a, TreeAction::Bet(_)))
            .nth(n)
            .map(|(i, _)| i);
    }
    if let Some(n_str) = lower.strip_prefix("raise-") {
        let n: usize = n_str.parse().ok()?;
        return node_actions
            .iter()
            .enumerate()
            .filter(|(_, a)| matches!(a, TreeAction::Raise(_)))
            .nth(n)
            .map(|(i, _)| i);
    }

    None
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    fn toy_tree() -> GameTree {
        GameTree::build(
            10.0,
            0.5,
            1.0,
            &[vec!["2.5bb".into()]],
            &[vec![1.0]],
            &[vec![1.0]],
            &[vec![1.0]],
            1,
        )
    }

    #[timed_test(10)]
    fn resolve_root_node() {
        let tree = toy_tree();
        let result = resolve_action_path(&tree, &[]);
        assert_eq!(result, Some(tree.root));
    }

    #[timed_test(10)]
    fn resolve_raise_call() {
        let tree = toy_tree();
        let path: Vec<String> = vec!["raise-0".into(), "call".into()];
        let result = resolve_action_path(&tree, &path);
        assert!(result.is_some(), "raise-0 then call should resolve");
    }

    #[timed_test(10)]
    fn resolve_invalid_returns_none() {
        let tree = toy_tree();
        let path: Vec<String> = vec!["invalid".into()];
        let result = resolve_action_path(&tree, &path);
        assert_eq!(result, None);
    }

    #[timed_test(10)]
    fn extract_grid_returns_169_cells() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let grid = extract_strategy_grid(&tree, &storage, tree.root, &[]);
        for row in &grid {
            for cell in row {
                assert!(
                    !cell.actions.is_empty(),
                    "every cell should have at least one action"
                );
            }
        }
    }
}
