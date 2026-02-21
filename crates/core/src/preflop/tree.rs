/// Preflop game tree built from a `PreflopConfig`.
///
/// Arena-allocated: all nodes live in a flat `Vec`, with children referencing
/// indices into that vec. Built once via `PreflopTree::build` and then
/// traversed by the CFR solver.
use super::config::PreflopConfig;

/// An action available at a preflop decision node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreflopAction {
    Fold,
    Call,
    Raise(f64),
    AllIn,
}

/// How a terminal node was reached.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TerminalType {
    Fold { folder: u8 },
    Showdown,
}

/// A node in the preflop game tree.
#[derive(Debug, Clone)]
pub enum PreflopNode {
    Decision {
        position: u8,
        children: Vec<u32>,
        action_labels: Vec<PreflopAction>,
    },
    Terminal {
        terminal_type: TerminalType,
        pot: u32,
    },
}

/// A complete preflop game tree.
#[derive(Debug, Clone)]
pub struct PreflopTree {
    pub nodes: Vec<PreflopNode>,
}

/// Internal state tracked during recursive tree construction.
#[derive(Debug, Clone)]
struct BuildState {
    /// Amount each player has invested so far.
    invested: Vec<u32>,
    /// Remaining stack per player after investments.
    stacks: Vec<u32>,
    /// Index of the next player to act.
    to_act: u8,
    /// Number of raises so far in this round.
    raise_count: u8,
    /// Number of players who have had a chance to act since the last raise.
    /// When this equals `num_active`, the round closes.
    actions_since_last_raise: u8,
    /// The current bet amount everyone must match.
    current_bet: u32,
    /// Number of players still in the hand (not folded).
    num_active: u8,
    /// Whether each player has folded.
    folded: Vec<bool>,
    /// Number of players total.
    num_players: u8,
}

impl PreflopTree {
    /// Builds the preflop game tree from the given configuration.
    #[must_use]
    pub fn build(config: &PreflopConfig) -> Self {
        let num_players = config.num_players();
        let mut invested = vec![0u32; num_players as usize];
        let mut stacks = config.stacks.clone();

        // Post blinds
        for &(pos, amount) in &config.blinds {
            let actual = amount.min(stacks[pos]);
            invested[pos] = actual;
            stacks[pos] -= actual;
        }

        // Post antes
        for &(pos, amount) in &config.antes {
            let actual = amount.min(stacks[pos]);
            invested[pos] += actual;
            stacks[pos] -= actual;
        }

        let current_bet = invested.iter().copied().max().unwrap_or(0);

        // First to act: position 0 for HU (SB), or first non-blind for multi-way
        let to_act = 0;

        let state = BuildState {
            invested,
            stacks,
            to_act,
            raise_count: 0,
            actions_since_last_raise: 0,
            current_bet,
            num_active: num_players,
            folded: vec![false; num_players as usize],
            num_players,
        };

        let mut nodes = Vec::new();
        build_recursive(config, &state, &mut nodes);

        Self { nodes }
    }
}

/// Recursively builds the tree, returning the index of the node created.
#[allow(clippy::cast_possible_truncation)]
fn build_recursive(
    config: &PreflopConfig,
    state: &BuildState,
    nodes: &mut Vec<PreflopNode>,
) -> u32 {
    // Check if only one player remains (everyone else folded)
    if state.num_active == 1 {
        let pot: u32 = state.invested.iter().sum();
        let folder = find_last_folder(state);
        let idx = nodes.len() as u32;
        nodes.push(PreflopNode::Terminal {
            terminal_type: TerminalType::Fold { folder },
            pot,
        });
        return idx;
    }

    // Check if round is complete: all active players have acted since last raise
    // and all investments are equal among active players
    if is_round_complete(state) {
        let pot: u32 = state.invested.iter().sum();
        let idx = nodes.len() as u32;
        nodes.push(PreflopNode::Terminal {
            terminal_type: TerminalType::Showdown,
            pot,
        });
        return idx;
    }

    // Skip folded players
    let position = find_next_active(state);
    let to_call = state
        .current_bet
        .saturating_sub(state.invested[position as usize]);

    let mut actions = Vec::new();
    let mut child_indices = Vec::new();

    // Reserve a slot for this decision node
    let node_idx = nodes.len() as u32;
    nodes.push(PreflopNode::Terminal {
        terminal_type: TerminalType::Showdown,
        pot: 0,
    }); // placeholder

    // Fold (always available if there's something to call, or if we want to fold anyway)
    if to_call > 0 {
        let child_state = apply_fold(state, position);
        let child_idx = build_recursive(config, &child_state, nodes);
        actions.push(PreflopAction::Fold);
        child_indices.push(child_idx);
    }

    // Call/Check
    {
        let child_state = apply_call(state, position, to_call);
        let child_idx = build_recursive(config, &child_state, nodes);
        actions.push(PreflopAction::Call);
        child_indices.push(child_idx);
    }

    // Raises (if under raise cap)
    if state.raise_count < config.raise_cap {
        let player_stack = state.stacks[position as usize];
        let last_raise_depth = state.raise_count + 1 == config.raise_cap;

        // At the last raise depth, only offer all-in (no sized raises)
        if !last_raise_depth {
            let depth =
                (state.raise_count as usize).min(config.raise_sizes.len().saturating_sub(1));
            for &size in &config.raise_sizes[depth] {
                let raise_amount = compute_raise_amount(state, size);
                if raise_amount < player_stack {
                    let child_state = apply_raise(state, position, raise_amount);
                    let child_idx = build_recursive(config, &child_state, nodes);
                    actions.push(PreflopAction::Raise(size));
                    child_indices.push(child_idx);
                }
            }
        }

        // All-in (available if we have more chips than needed to call)
        if player_stack > to_call {
            let child_state = apply_all_in(state, position);
            let child_idx = build_recursive(config, &child_state, nodes);
            actions.push(PreflopAction::AllIn);
            child_indices.push(child_idx);
        }
    }

    // Replace placeholder with the real decision node
    nodes[node_idx as usize] = PreflopNode::Decision {
        position,
        children: child_indices,
        action_labels: actions,
    };

    node_idx
}

/// Find the most recent folder (for terminal type labeling).
#[allow(clippy::cast_possible_truncation)]
fn find_last_folder(state: &BuildState) -> u8 {
    for (i, &folded) in state.folded.iter().enumerate() {
        if folded {
            return i as u8;
        }
    }
    0
}

/// Find the next active (non-folded) player from `to_act`.
fn find_next_active(state: &BuildState) -> u8 {
    let mut pos = state.to_act;
    for _ in 0..state.num_players {
        if !state.folded[pos as usize] {
            return pos;
        }
        pos = (pos + 1) % state.num_players;
    }
    state.to_act
}

/// Check whether the betting round is complete.
fn is_round_complete(state: &BuildState) -> bool {
    // Not complete until everyone has had at least one action
    if state.actions_since_last_raise < state.num_active {
        return false;
    }

    // All active players must have equal investment (or be all-in)
    let active_investments: Vec<u32> = state
        .invested
        .iter()
        .enumerate()
        .filter(|(i, _)| !state.folded[*i] && state.stacks[*i] > 0)
        .map(|(_, &inv)| inv)
        .collect();

    if active_investments.is_empty() {
        return true;
    }

    let target = active_investments[0];
    active_investments.iter().all(|&inv| inv == target)
}

/// Compute the raise amount (additional chips beyond calling) based on a multiplier.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn compute_raise_amount(state: &BuildState, multiplier: f64) -> u32 {
    // new_bet = current_bet * multiplier (always positive, fits u32)
    let new_bet = (f64::from(state.current_bet) * multiplier) as u32;
    new_bet.max(state.current_bet + 1) // at least a min-raise
}

/// Advance to the next player after the given position.
fn next_player(position: u8, num_players: u8) -> u8 {
    (position + 1) % num_players
}

fn apply_fold(state: &BuildState, position: u8) -> BuildState {
    let mut new_state = state.clone();
    new_state.folded[position as usize] = true;
    new_state.num_active -= 1;
    new_state.actions_since_last_raise += 1;
    new_state.to_act = next_player(position, state.num_players);
    new_state
}

fn apply_call(state: &BuildState, position: u8, to_call: u32) -> BuildState {
    let mut new_state = state.clone();
    let actual_call = to_call.min(state.stacks[position as usize]);
    new_state.invested[position as usize] += actual_call;
    new_state.stacks[position as usize] -= actual_call;
    new_state.actions_since_last_raise += 1;
    new_state.to_act = next_player(position, state.num_players);
    new_state
}

fn apply_raise(state: &BuildState, position: u8, raise_total: u32) -> BuildState {
    let mut new_state = state.clone();
    let to_call = state
        .current_bet
        .saturating_sub(state.invested[position as usize]);
    // Total new investment = call + raise above current bet
    let additional = raise_total;
    let actual = additional.min(state.stacks[position as usize]);
    new_state.invested[position as usize] += actual;
    new_state.stacks[position as usize] -= actual;
    new_state.current_bet = new_state.invested[position as usize];
    new_state.raise_count += 1;
    new_state.actions_since_last_raise = 1; // reset: raiser has acted
    new_state.to_act = next_player(position, state.num_players);
    let _ = to_call;
    new_state
}

fn apply_all_in(state: &BuildState, position: u8) -> BuildState {
    let mut new_state = state.clone();
    let all_chips = state.stacks[position as usize];
    new_state.invested[position as usize] += all_chips;
    new_state.stacks[position as usize] = 0;
    let new_bet = new_state.invested[position as usize];
    if new_bet > state.current_bet {
        new_state.current_bet = new_bet;
        new_state.raise_count += 1;
        new_state.actions_since_last_raise = 1;
    } else {
        new_state.actions_since_last_raise += 1;
    }
    new_state.to_act = next_player(position, state.num_players);
    new_state
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn hu_tree_has_root_decision() {
        let config = PreflopConfig::heads_up(100);
        let tree = PreflopTree::build(&config);
        assert!(!tree.nodes.is_empty());
        assert!(matches!(tree.nodes[0], PreflopNode::Decision { .. }));
    }

    #[timed_test]
    fn hu_tree_root_is_sb_to_act() {
        let config = PreflopConfig::heads_up(100);
        let tree = PreflopTree::build(&config);
        match &tree.nodes[0] {
            PreflopNode::Decision { position, .. } => assert_eq!(*position, 0),
            _ => panic!("root should be a decision node"),
        }
    }

    #[timed_test]
    fn hu_tree_fold_produces_terminal() {
        let config = PreflopConfig::heads_up(100);
        let tree = PreflopTree::build(&config);
        match &tree.nodes[0] {
            PreflopNode::Decision {
                action_labels,
                children,
                ..
            } => {
                // First action should be Fold
                let fold_idx = action_labels
                    .iter()
                    .position(|a| *a == PreflopAction::Fold)
                    .expect("Fold should be available");
                let child = children[fold_idx] as usize;
                assert!(
                    matches!(
                        tree.nodes[child],
                        PreflopNode::Terminal {
                            terminal_type: TerminalType::Fold { folder: 0 },
                            ..
                        }
                    ),
                    "Fold child should be a Fold terminal, got: {:?}",
                    tree.nodes[child]
                );
            }
            _ => panic!("root should be a decision node"),
        }
    }

    #[timed_test]
    fn hu_tree_call_then_check_is_showdown() {
        // SB limps (calls BB) → BB checks → Showdown
        let config = PreflopConfig::heads_up(100);
        let tree = PreflopTree::build(&config);

        // Root: SB's decision. Find "Call" action.
        let (bb_node_idx, _) = find_action_child(&tree, 0, PreflopAction::Call);

        // BB's decision after SB limps. Find "Call" (check, since to_call=0).
        match &tree.nodes[bb_node_idx] {
            PreflopNode::Decision {
                position,
                action_labels,
                children,
                ..
            } => {
                assert_eq!(*position, 1, "BB should act next");
                // BB has Call (check) as an option
                let call_idx = action_labels
                    .iter()
                    .position(|a| *a == PreflopAction::Call)
                    .expect("BB should be able to check");
                let terminal_idx = children[call_idx] as usize;
                assert!(
                    matches!(
                        tree.nodes[terminal_idx],
                        PreflopNode::Terminal {
                            terminal_type: TerminalType::Showdown,
                            ..
                        }
                    ),
                    "After SB limp + BB check, should be showdown, got: {:?}",
                    tree.nodes[terminal_idx]
                );
            }
            other => panic!("Expected BB decision node, got: {:?}", other),
        }
    }

    #[timed_test]
    fn hu_tree_respects_raise_cap() {
        let config = PreflopConfig::heads_up(100);
        let tree = PreflopTree::build(&config);

        // Walk down the tree following only raise actions and count them
        let max_raises = count_max_raises(&tree, 0, 0);
        assert!(
            max_raises <= u32::from(config.raise_cap),
            "Max raises {} exceeds cap {}",
            max_raises,
            config.raise_cap
        );
    }

    #[timed_test]
    fn hu_tree_showdown_pot_after_limp() {
        // SB limps (1 → 2) + BB checks → pot should be 4 (2+2)
        let config = PreflopConfig::heads_up(100);
        let tree = PreflopTree::build(&config);

        let (bb_node, _) = find_action_child(&tree, 0, PreflopAction::Call);
        let (showdown_node, _) = find_action_child(&tree, bb_node, PreflopAction::Call);

        match &tree.nodes[showdown_node] {
            PreflopNode::Terminal { pot, .. } => {
                assert_eq!(*pot, 4, "Pot after limp + check should be 4 (SB=2, BB=2)");
            }
            other => panic!("Expected terminal, got: {:?}", other),
        }
    }

    #[timed_test]
    fn hu_tree_fold_pot_after_sb_fold() {
        // SB folds → BB wins, pot = 3 (SB posted 1, BB posted 2)
        let config = PreflopConfig::heads_up(100);
        let tree = PreflopTree::build(&config);

        let (fold_node, _) = find_action_child(&tree, 0, PreflopAction::Fold);

        match &tree.nodes[fold_node] {
            PreflopNode::Terminal { pot, .. } => {
                assert_eq!(*pot, 3, "Pot after SB fold should be 3 (SB=1, BB=2)");
            }
            other => panic!("Expected terminal, got: {:?}", other),
        }
    }

    /// Helper: find the child index for a given action at a decision node.
    fn find_action_child(
        tree: &PreflopTree,
        node_idx: usize,
        action: PreflopAction,
    ) -> (usize, usize) {
        match &tree.nodes[node_idx] {
            PreflopNode::Decision {
                action_labels,
                children,
                ..
            } => {
                let pos = action_labels
                    .iter()
                    .position(|a| *a == action)
                    .unwrap_or_else(|| {
                        panic!(
                            "Action {:?} not found at node {}. Available: {:?}",
                            action, node_idx, action_labels
                        )
                    });
                (children[pos] as usize, pos)
            }
            other => panic!("Expected decision node at {}, got: {:?}", node_idx, other),
        }
    }

    #[timed_test]
    fn last_raise_depth_only_offers_all_in() {
        // With raise_cap=2, the second raise depth should only have all-in (no sized raises)
        let mut config = PreflopConfig::heads_up(25);
        config.raise_sizes = vec![vec![3.0]];
        config.raise_cap = 2;
        let tree = PreflopTree::build(&config);

        // SB opens with Raise(3.0) → BB's decision
        let (bb_node, _) = find_action_child(&tree, 0, PreflopAction::Raise(3.0));

        // BB at depth 1 (last depth with cap=2): should have Fold, Call, AllIn but NO Raise
        match &tree.nodes[bb_node] {
            PreflopNode::Decision { action_labels, .. } => {
                let has_raise = action_labels
                    .iter()
                    .any(|a| matches!(a, PreflopAction::Raise(_)));
                assert!(
                    !has_raise,
                    "Last raise depth should not have sized raises, got: {:?}",
                    action_labels
                );
                let has_all_in = action_labels.iter().any(|a| *a == PreflopAction::AllIn);
                assert!(
                    has_all_in,
                    "Last raise depth should offer all-in, got: {:?}",
                    action_labels
                );
            }
            other => panic!("Expected decision node, got: {:?}", other),
        }
    }

    /// Recursively count the maximum number of raises along any path.
    fn count_max_raises(tree: &PreflopTree, node_idx: usize, raises_so_far: u32) -> u32 {
        match &tree.nodes[node_idx] {
            PreflopNode::Terminal { .. } => raises_so_far,
            PreflopNode::Decision {
                action_labels,
                children,
                ..
            } => {
                let mut max = raises_so_far;
                for (action, &child) in action_labels.iter().zip(children.iter()) {
                    let extra = match action {
                        PreflopAction::Raise(_) | PreflopAction::AllIn => 1,
                        _ => 0,
                    };
                    let child_max = count_max_raises(tree, child as usize, raises_so_far + extra);
                    if child_max > max {
                        max = child_max;
                    }
                }
                max
            }
        }
    }
}
