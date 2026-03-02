//! Game tree construction and EV computation.
//!
//! Builds a decision tree from a concrete deal using blueprint strategy lookups,
//! then computes path-weighted expected value at terminal nodes.

use crate::blueprint::BlueprintStrategy;
use crate::game::{Game, HunlPostflop, Player, PostflopState, TerminalType};
use crate::info_key::format_action_label;

/// Configuration for tree building.
#[derive(Debug, Clone)]
pub struct TreeConfig {
    /// Maximum depth to traverse.
    pub max_depth: usize,
    /// Minimum action probability to include in the tree.
    pub min_prob: f32,
    /// Bet sizes from the game config (for formatting action labels).
    pub bet_sizes: Vec<f32>,
}

/// Kind of terminal node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalKind {
    Fold,
    Showdown,
}

/// Terminal node outcome.
#[derive(Debug, Clone)]
pub struct TerminalInfo {
    /// Type of terminal.
    pub kind: TerminalKind,
    /// EV for Player 1 (SB) at this terminal, in BB.
    pub ev_bb: f64,
    /// Human-readable description.
    pub description: String,
}

/// A node in the game decision tree.
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Action that led to this node ("Root" for the root).
    pub action_label: String,
    /// Probability of taking this action (1.0 for root).
    pub probability: f32,
    /// Which player acts here: 0=P1(SB), 1=P2(BB). None if terminal.
    pub player: Option<u8>,
    /// Packed info set key at this node.
    pub info_key: u64,
    /// Street index (0-3).
    pub street: u8,
    /// Current pot in BB.
    pub pot_bb: f64,
    /// Remaining stacks in BB [P1, P2].
    pub stacks_bb: [f64; 2],
    /// Human-readable hand description.
    pub hand_desc: String,
    /// Whether the blueprint contained a strategy for this key.
    pub has_strategy: bool,
    /// Terminal info if this is a leaf.
    pub terminal: Option<TerminalInfo>,
    /// Product of all ancestor probabilities.
    pub reach_prob: f64,
    /// Child nodes.
    pub children: Vec<TreeNode>,
}

/// Summary of EV across all terminal paths in the tree.
#[derive(Debug, Clone)]
pub struct EvSummary {
    /// Each terminal: (path description, reach probability, EV in BB).
    pub terminals: Vec<(String, f64, f64)>,
    /// Weighted total EV: sum of `reach_prob` * `ev_bb`.
    pub total_ev_bb: f64,
    /// Sum of all terminal reach probabilities.
    pub total_reach: f64,
}

/// Build a game tree from a concrete deal with blueprint strategy lookups.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn build_tree(
    game: &HunlPostflop,
    state: &PostflopState,
    blueprint: &BlueprintStrategy,
    config: &TreeConfig,
    action_label: String,
    probability: f32,
    reach_prob: f64,
    depth: usize,
) -> TreeNode {
    let info_key = game.info_set_key(state);
    let pot_bb = f64::from(state.pot) / 2.0;
    let stacks_bb = [
        f64::from(state.stacks[0]) / 2.0,
        f64::from(state.stacks[1]) / 2.0,
    ];
    let street = state.street as u8;
    let hand_desc = hand_description(state);

    // Terminal node
    if let Some(terminal) = state.terminal {
        let terminal_info = build_terminal_info(game, state, terminal);
        return TreeNode {
            action_label,
            probability,
            player: None,
            info_key,
            street,
            pot_bb,
            stacks_bb,
            hand_desc,
            has_strategy: false,
            terminal: Some(terminal_info),
            reach_prob,
            children: Vec::new(),
        };
    }

    // Depth limit
    if depth >= config.max_depth {
        return TreeNode {
            action_label,
            probability,
            player: player_index(state),
            info_key,
            street,
            pot_bb,
            stacks_bb,
            hand_desc,
            has_strategy: false,
            terminal: None,
            reach_prob,
            children: Vec::new(),
        };
    }

    let actions = game.actions(state);
    let strategy = blueprint.lookup(info_key);
    let has_strategy = strategy.is_some();

    // Build children
    let probs = strategy_probs(strategy, actions.len());
    let mut children = Vec::new();

    for (i, &action) in actions.iter().enumerate() {
        let prob = probs[i];
        if prob < config.min_prob {
            continue;
        }
        let child_state = game.next_state(state, action);
        let child_label = format_action_label(action, &config.bet_sizes);
        let child_reach = reach_prob * f64::from(prob);

        let child = build_tree(
            game,
            &child_state,
            blueprint,
            config,
            child_label,
            prob,
            child_reach,
            depth + 1,
        );
        children.push(child);
    }

    // Sort by probability descending
    children.sort_by(|a, b| {
        b.probability
            .partial_cmp(&a.probability)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    TreeNode {
        action_label,
        probability,
        player: player_index(state),
        info_key,
        street,
        pot_bb,
        stacks_bb,
        hand_desc,
        has_strategy,
        terminal: None,
        reach_prob,
        children,
    }
}

/// Compute path-weighted EV summary from a built tree.
#[must_use]
pub fn compute_deal_ev(root: &TreeNode) -> EvSummary {
    let mut terminals = Vec::new();
    collect_terminals(root, &[], &mut terminals);

    let total_ev_bb = terminals.iter().map(|(_, reach, ev)| reach * ev).sum();
    let total_reach = terminals.iter().map(|(_, reach, _)| reach).sum();

    // Sort by reach probability descending
    terminals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    EvSummary {
        terminals,
        total_ev_bb,
        total_reach,
    }
}

fn collect_terminals(node: &TreeNode, path: &[String], out: &mut Vec<(String, f64, f64)>) {
    if let Some(ref terminal) = node.terminal {
        let path_str = if path.is_empty() {
            node.action_label.clone()
        } else {
            let mut full = path.to_vec();
            full.push(node.action_label.clone());
            full[1..].join(" \u{2192} ") // skip "Root" prefix
        };
        out.push((path_str, node.reach_prob, terminal.ev_bb));
        return;
    }

    if node.children.is_empty() {
        return; // Depth-limited leaf, not a real terminal
    }

    let mut current_path = path.to_vec();
    current_path.push(node.action_label.clone());
    for child in &node.children {
        collect_terminals(child, &current_path, out);
    }
}

fn build_terminal_info(
    game: &HunlPostflop,
    state: &PostflopState,
    terminal: TerminalType,
) -> TerminalInfo {
    let ev_bb = game.utility(state, Player::Player1);

    match terminal {
        TerminalType::Fold(folder) => {
            let winner = if folder == Player::Player1 {
                "P2"
            } else {
                "P1"
            };
            let folder_name = if folder == Player::Player1 {
                "P1"
            } else {
                "P2"
            };
            TerminalInfo {
                kind: TerminalKind::Fold,
                ev_bb,
                description: format!(
                    "{folder_name} folds \u{2192} {winner} wins {:.1} BB",
                    f64::from(state.pot) / 2.0
                ),
            }
        }
        TerminalType::Showdown => {
            let result = if ev_bb > 0.0 {
                format!("P1 wins {ev_bb:+.1} BB")
            } else if ev_bb < 0.0 {
                format!("P2 wins {:+.1} BB", -ev_bb)
            } else {
                "Split pot".to_string()
            };
            TerminalInfo {
                kind: TerminalKind::Showdown,
                ev_bb,
                description: format!("Showdown: {result}"),
            }
        }
    }
}

fn player_index(state: &PostflopState) -> Option<u8> {
    state.to_act.map(|p| match p {
        Player::Player1 => 0,
        Player::Player2 => 1,
    })
}

fn hand_description(state: &PostflopState) -> String {
    let holding = state.current_holding();
    format!("{}{}", holding[0], holding[1])
}

/// Get strategy probabilities, falling back to uniform if not in blueprint.
fn strategy_probs(strategy: Option<&[f32]>, num_actions: usize) -> Vec<f32> {
    match strategy {
        Some(probs) if probs.len() == num_actions => probs.to_vec(),
        _ => {
            #[allow(clippy::cast_precision_loss)]
            let uniform = 1.0 / num_actions as f32;
            vec![uniform; num_actions]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::PostflopConfig;
    use test_macros::timed_test;

    fn create_test_game() -> HunlPostflop {
        let config = PostflopConfig {
            stack_depth: 25,
            bet_sizes: vec![0.33, 0.67, 1.0],
            ..PostflopConfig::default()
        };
        HunlPostflop::new(config, None, 10)
    }

    #[timed_test]
    fn build_tree_depth_zero_returns_leaf() {
        let game = create_test_game();
        let states = game.initial_states();
        let state = &states[0];
        let bp = BlueprintStrategy::new();
        let config = TreeConfig {
            max_depth: 0,
            min_prob: 0.0,
            bet_sizes: vec![0.33, 0.67, 1.0],
        };

        let root = build_tree(&game, state, &bp, &config, "Root".into(), 1.0, 1.0, 0);
        assert!(root.children.is_empty());
        assert!(root.terminal.is_none());
    }

    #[timed_test]
    fn build_tree_depth_one_has_children() {
        let game = create_test_game();
        let states = game.initial_states();
        let state = &states[0];
        let bp = BlueprintStrategy::new();
        let config = TreeConfig {
            max_depth: 1,
            min_prob: 0.0,
            bet_sizes: vec![0.33, 0.67, 1.0],
        };

        let root = build_tree(&game, state, &bp, &config, "Root".into(), 1.0, 1.0, 0);
        assert!(!root.children.is_empty(), "Should have children at depth 1");
        assert!(root.player.is_some());
    }

    #[timed_test]
    fn build_tree_min_prob_prunes() {
        let game = create_test_game();
        let states = game.initial_states();
        let state = &states[0];
        let bp = BlueprintStrategy::new();

        let config_all = TreeConfig {
            max_depth: 1,
            min_prob: 0.0,
            bet_sizes: vec![0.33, 0.67, 1.0],
        };
        let root_all = build_tree(&game, state, &bp, &config_all, "Root".into(), 1.0, 1.0, 0);

        let config_high = TreeConfig {
            max_depth: 1,
            min_prob: 0.5,
            bet_sizes: vec![0.33, 0.67, 1.0],
        };
        let root_high = build_tree(&game, state, &bp, &config_high, "Root".into(), 1.0, 1.0, 0);

        assert!(
            root_high.children.len() <= root_all.children.len(),
            "Higher min_prob should prune more: {} vs {}",
            root_high.children.len(),
            root_all.children.len()
        );
    }

    #[timed_test]
    fn build_tree_fold_creates_terminal() {
        let game = create_test_game();
        let states = game.initial_states();
        let state = &states[0];
        let bp = BlueprintStrategy::new();
        let config = TreeConfig {
            max_depth: 2,
            min_prob: 0.0,
            bet_sizes: vec![0.33, 0.67, 1.0],
        };

        let root = build_tree(&game, state, &bp, &config, "Root".into(), 1.0, 1.0, 0);

        // Find the fold child
        let fold_child = root.children.iter().find(|c| c.action_label == "Fold");
        assert!(fold_child.is_some(), "Should have a Fold child");

        let fold = fold_child.unwrap();
        assert!(fold.terminal.is_some(), "Fold should be terminal");
        assert_eq!(fold.terminal.as_ref().unwrap().kind, TerminalKind::Fold);
    }

    #[timed_test]
    fn compute_ev_from_simple_tree() {
        // Build a tree with known terminal values
        let terminal_win = TreeNode {
            action_label: "Call".into(),
            probability: 0.6,
            player: None,
            info_key: 0,
            street: 0,
            pot_bb: 4.0,
            stacks_bb: [23.0, 23.0],
            hand_desc: String::new(),
            has_strategy: false,
            terminal: Some(TerminalInfo {
                kind: TerminalKind::Showdown,
                ev_bb: 2.0,
                description: "P1 wins".into(),
            }),
            reach_prob: 0.6,
            children: Vec::new(),
        };

        let terminal_fold = TreeNode {
            action_label: "Fold".into(),
            probability: 0.4,
            player: None,
            info_key: 0,
            street: 0,
            pot_bb: 2.0,
            stacks_bb: [24.0, 24.0],
            hand_desc: String::new(),
            has_strategy: false,
            terminal: Some(TerminalInfo {
                kind: TerminalKind::Fold,
                ev_bb: 1.0,
                description: "P2 folds".into(),
            }),
            reach_prob: 0.4,
            children: Vec::new(),
        };

        let root = TreeNode {
            action_label: "Root".into(),
            probability: 1.0,
            player: Some(0),
            info_key: 0,
            street: 0,
            pot_bb: 2.0,
            stacks_bb: [24.0, 24.0],
            hand_desc: String::new(),
            has_strategy: true,
            terminal: None,
            reach_prob: 1.0,
            children: vec![terminal_win, terminal_fold],
        };

        let summary = compute_deal_ev(&root);
        assert_eq!(summary.terminals.len(), 2);
        // EV = 0.6 * 2.0 + 0.4 * 1.0 = 1.6
        assert!(
            (summary.total_ev_bb - 1.6).abs() < 0.001,
            "EV: {}",
            summary.total_ev_bb
        );
        assert!((summary.total_reach - 1.0).abs() < 0.001);
    }

    #[timed_test]
    fn strategy_probs_uniform_fallback() {
        let probs = strategy_probs(None, 4);
        assert_eq!(probs.len(), 4);
        assert!((probs[0] - 0.25).abs() < 0.001);
    }

    #[timed_test]
    fn strategy_probs_uses_blueprint() {
        let bp_probs = vec![0.1, 0.2, 0.3, 0.4];
        let probs = strategy_probs(Some(&bp_probs), 4);
        assert_eq!(probs, bp_probs);
    }
}
