//! Concrete subgame tree for real-time subgame solving.
//!
//! Unlike the abstract game tree used in blueprint training, this tree
//! operates on a specific board with no hand abstraction. Each node
//! tracks concrete pot/stack values and the tree can be depth-limited
//! to solve only the current street.

use crate::game::{Action, ALL_IN};
use crate::poker::Card;

/// A node in a concrete subgame tree (specific board, no hand abstraction).
#[derive(Debug, Clone)]
pub enum SubgameNode {
    /// A decision point where a player must act.
    Decision {
        position: u8,
        actions: Vec<Action>,
        children: Vec<u32>,
        pot: u32,
        stacks: Vec<u32>,
    },
    /// Hand is over (fold or showdown).
    Terminal {
        is_fold: bool,
        fold_player: u8,
        pot: u32,
        stacks: Vec<u32>,
    },
    /// Depth boundary -- stop solving here, use blueprint continuation values.
    DepthBoundary {
        pot: u32,
        stacks: Vec<u32>,
    },
}

/// A concrete subgame tree for a specific board.
#[derive(Debug, Clone)]
pub struct SubgameTree {
    pub nodes: Vec<SubgameNode>,
    pub board: Vec<Card>,
    pub bet_sizes: Vec<f32>,
    pub num_players: u8,
}

/// All valid hole card combos for a specific board.
#[derive(Debug, Clone)]
pub struct SubgameHands {
    pub combos: Vec<[Card; 2]>,
}

impl SubgameHands {
    /// Enumerate all valid 2-card combos from the 52-card deck excluding board cards.
    #[must_use]
    pub fn enumerate(board: &[Card]) -> Self {
        let deck = remaining_deck(board);
        let mut combos = Vec::with_capacity(deck.len() * (deck.len() - 1) / 2);
        for i in 0..deck.len() {
            for j in (i + 1)..deck.len() {
                combos.push([deck[i], deck[j]]);
            }
        }
        Self { combos }
    }
}

fn remaining_deck(board: &[Card]) -> Vec<Card> {
    crate::poker::full_deck()
        .into_iter()
        .filter(|c| !board.contains(c))
        .collect()
}

/// Builder for constructing a [`SubgameTree`].
pub struct SubgameTreeBuilder {
    board: Vec<Card>,
    bet_sizes: Vec<f32>,
    pot: u32,
    stacks: Vec<u32>,
    depth_limit: Option<usize>,
}

impl SubgameTreeBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            board: Vec::new(),
            bet_sizes: vec![1.0],
            pot: 0,
            stacks: Vec::new(),
            depth_limit: None,
        }
    }

    #[must_use]
    pub fn board(mut self, board: &[Card]) -> Self {
        self.board = board.to_vec();
        self
    }

    #[must_use]
    pub fn bet_sizes(mut self, sizes: &[f32]) -> Self {
        self.bet_sizes = sizes.to_vec();
        self
    }

    #[must_use]
    pub fn pot(mut self, pot: u32) -> Self {
        self.pot = pot;
        self
    }

    #[must_use]
    pub fn stacks(mut self, stacks: &[u32]) -> Self {
        self.stacks = stacks.to_vec();
        self
    }

    #[must_use]
    pub fn depth_limit(mut self, streets: usize) -> Self {
        self.depth_limit = Some(streets);
        self
    }

    /// Build the subgame tree via recursive DFS.
    #[must_use]
    pub fn build(self) -> SubgameTree {
        let starting_street = board_street(self.board.len());
        let ctx = BuildContext {
            bet_sizes: &self.bet_sizes,
            depth_limit: self.depth_limit,
            starting_street,
        };
        let root = NodeState {
            position: 0,
            pot: self.pot,
            stacks: self.stacks.clone(),
            to_call: 0,
            street: starting_street,
            street_bets: 0,
            street_actions: 0,
        };
        let mut nodes = Vec::new();
        build_recursive(&ctx, &root, &mut nodes);
        SubgameTree {
            nodes,
            board: self.board,
            bet_sizes: self.bet_sizes,
            num_players: 2,
        }
    }
}

impl Default for SubgameTreeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal tree-building logic
// ---------------------------------------------------------------------------

/// Immutable context shared during tree construction.
struct BuildContext<'a> {
    bet_sizes: &'a [f32],
    depth_limit: Option<usize>,
    starting_street: u8,
}

/// Mutable per-node state tracked during DFS.
#[derive(Clone)]
struct NodeState {
    position: u8,
    pot: u32,
    stacks: Vec<u32>,
    to_call: u32,
    street: u8,
    street_bets: u32,
    /// Number of actions taken on the current street (for detecting street end).
    street_actions: u32,
}

/// Map board card count to a street number (0=preflop, 1=flop, 2=turn, 3=river).
fn board_street(board_len: usize) -> u8 {
    match board_len {
        0 => 0,
        4 => 2,
        5 => 3,
        // 3-card board and any other count default to flop
        _ => 1,
    }
}

fn build_recursive(ctx: &BuildContext, state: &NodeState, nodes: &mut Vec<SubgameNode>) -> u32 {
    #[allow(clippy::cast_possible_truncation)]
    let node_idx = nodes.len() as u32;

    let actions = generate_actions(ctx, state);

    // Push placeholder decision node; fill children after recursion.
    nodes.push(SubgameNode::Decision {
        position: state.position,
        actions: actions.clone(),
        children: Vec::new(),
        pot: state.pot,
        stacks: state.stacks.clone(),
    });

    let mut children = Vec::with_capacity(actions.len());
    for &action in &actions {
        let child_idx = apply_action(ctx, state, action, nodes);
        children.push(child_idx);
    }

    if let SubgameNode::Decision { children: ref mut c, .. } = nodes[node_idx as usize] {
        *c = children;
    }

    node_idx
}

/// Generate the legal actions for the current node state.
fn generate_actions(ctx: &BuildContext, state: &NodeState) -> Vec<Action> {
    let mut actions = Vec::new();
    let stack = state.stacks[state.position as usize];

    if state.to_call > 0 {
        actions.push(Action::Fold);
    }

    if state.to_call == 0 {
        actions.push(Action::Check);
    } else if stack >= state.to_call {
        actions.push(Action::Call);
    }

    // Sized bets/raises
    #[allow(clippy::cast_possible_truncation)]
    for (i, _) in ctx.bet_sizes.iter().enumerate() {
        let idx = i as u32;
        if state.to_call == 0 {
            actions.push(Action::Bet(idx));
        } else {
            actions.push(Action::Raise(idx));
        }
    }

    // All-in
    if state.to_call == 0 {
        actions.push(Action::Bet(ALL_IN));
    } else {
        actions.push(Action::Raise(ALL_IN));
    }

    actions
}

/// Apply an action and recurse (or produce a terminal / boundary).
///
/// Returns the node index of the child.
fn apply_action(
    ctx: &BuildContext,
    state: &NodeState,
    action: Action,
    nodes: &mut Vec<SubgameNode>,
) -> u32 {
    match action {
        Action::Fold => push_terminal(state, true, nodes),
        Action::Check => apply_check(ctx, state, nodes),
        Action::Call => apply_call(ctx, state, nodes),
        Action::Bet(idx) | Action::Raise(idx) => apply_bet(ctx, state, idx, nodes),
    }
}

fn push_terminal(state: &NodeState, is_fold: bool, nodes: &mut Vec<SubgameNode>) -> u32 {
    #[allow(clippy::cast_possible_truncation)]
    let idx = nodes.len() as u32;
    nodes.push(SubgameNode::Terminal {
        is_fold,
        fold_player: state.position,
        pot: state.pot,
        stacks: state.stacks.clone(),
    });
    idx
}

fn push_boundary(pot: u32, stacks: &[u32], nodes: &mut Vec<SubgameNode>) -> u32 {
    #[allow(clippy::cast_possible_truncation)]
    let idx = nodes.len() as u32;
    nodes.push(SubgameNode::DepthBoundary {
        pot,
        stacks: stacks.to_vec(),
    });
    idx
}

/// Check action: if both players have checked, advance the street.
fn apply_check(ctx: &BuildContext, state: &NodeState, nodes: &mut Vec<SubgameNode>) -> u32 {
    let new_actions = state.street_actions + 1;
    // Street ends when both players have acted and neither has bet.
    // In HU postflop: position 0 checks, then position 1 checks => advance.
    if new_actions >= 2 {
        return advance_street_or_boundary(ctx, state.pot, &state.stacks, state.street, nodes);
    }
    // Otherwise, pass action to opponent.
    let child = NodeState {
        position: 1 - state.position,
        pot: state.pot,
        stacks: state.stacks.clone(),
        to_call: 0,
        street: state.street,
        street_bets: state.street_bets,
        street_actions: new_actions,
    };
    build_recursive(ctx, &child, nodes)
}

/// Call action: match the bet, then advance the street.
fn apply_call(ctx: &BuildContext, state: &NodeState, nodes: &mut Vec<SubgameNode>) -> u32 {
    let player = state.position as usize;
    let call_amount = state.to_call.min(state.stacks[player]);
    let mut new_stacks = state.stacks.clone();
    new_stacks[player] -= call_amount;
    let new_pot = state.pot + call_amount;

    // If opponent is all-in after the call, go to showdown.
    let opponent = 1 - state.position;
    if new_stacks[opponent as usize] == 0 || new_stacks[player] == 0 {
        return push_showdown(new_pot, &new_stacks, nodes);
    }

    advance_street_or_boundary(ctx, new_pot, &new_stacks, state.street, nodes)
}

fn push_showdown(pot: u32, stacks: &[u32], nodes: &mut Vec<SubgameNode>) -> u32 {
    #[allow(clippy::cast_possible_truncation)]
    let idx = nodes.len() as u32;
    nodes.push(SubgameNode::Terminal {
        is_fold: false,
        fold_player: 0,
        pot,
        stacks: stacks.to_vec(),
    });
    idx
}

/// Bet or raise: increase pot, set `to_call` for opponent.
fn apply_bet(
    ctx: &BuildContext,
    state: &NodeState,
    size_idx: u32,
    nodes: &mut Vec<SubgameNode>,
) -> u32 {
    let player = state.position as usize;
    let effective_stack = state.stacks[player].saturating_sub(state.to_call);
    let bet_portion = resolve_bet_amount(size_idx, state.pot, effective_stack, ctx.bet_sizes);
    let total = state.to_call + bet_portion;
    let actual = total.min(state.stacks[player]);

    let mut new_stacks = state.stacks.clone();
    new_stacks[player] -= actual;
    let new_pot = state.pot + actual;
    let new_to_call = actual.saturating_sub(state.to_call);

    // If opponent is already all-in, no further action needed.
    let opponent = (1 - state.position) as usize;
    if new_stacks[opponent] == 0 {
        return push_showdown(new_pot, &new_stacks, nodes);
    }

    let child = NodeState {
        position: 1 - state.position,
        pot: new_pot,
        stacks: new_stacks,
        to_call: new_to_call,
        street: state.street,
        street_bets: state.street_bets + 1,
        street_actions: state.street_actions + 1,
    };
    build_recursive(ctx, &child, nodes)
}

/// Resolve a bet size index to a chip amount using pot-fraction bet sizes.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn resolve_bet_amount(idx: u32, pot: u32, effective_stack: u32, bet_sizes: &[f32]) -> u32 {
    if idx == ALL_IN {
        return effective_stack;
    }
    // Default to pot-size if index is out of range (should not happen in practice).
    let fraction = bet_sizes.get(idx as usize).copied().unwrap_or(1.0);
    let size = (f64::from(pot) * f64::from(fraction)).round() as u32;
    size.min(effective_stack)
}

/// Advance to the next street, or insert a depth boundary if limited.
fn advance_street_or_boundary(
    ctx: &BuildContext,
    pot: u32,
    stacks: &[u32],
    current_street: u8,
    nodes: &mut Vec<SubgameNode>,
) -> u32 {
    // River: action completed → showdown
    if current_street == 3 {
        return push_showdown(pot, stacks, nodes);
    }

    let next_street = current_street + 1;
    let streets_played = next_street - ctx.starting_street;

    // Check depth limit
    if let Some(limit) = ctx.depth_limit {
        #[allow(clippy::cast_possible_truncation)]
        if streets_played as usize >= limit {
            return push_boundary(pot, stacks, nodes);
        }
    }

    // Continue to next street: position 0 acts first postflop.
    let child = NodeState {
        position: 0,
        pot,
        stacks: stacks.to_vec(),
        to_call: 0,
        street: next_street,
        street_bets: 0,
        street_actions: 0,
    };
    build_recursive(ctx, &child, nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::{Card, Suit, Value};
    use test_macros::timed_test;

    fn make_river_board() -> Vec<Card> {
        vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Four, Suit::Club),
            Card::new(Value::Ten, Suit::Club),
        ]
    }

    fn make_flop_board() -> Vec<Card> {
        vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Diamond),
        ]
    }

    #[timed_test]
    fn river_tree_has_no_depth_boundaries() {
        let tree = SubgameTreeBuilder::new()
            .board(&make_river_board())
            .bet_sizes(&[0.5, 1.0, 2.0])
            .pot(100)
            .stacks(&[200, 200])
            .build();
        let boundaries = tree
            .nodes
            .iter()
            .filter(|n| matches!(n, SubgameNode::DepthBoundary { .. }))
            .count();
        assert_eq!(boundaries, 0, "river tree should have no depth boundaries");
    }

    #[timed_test]
    fn depth_limited_flop_tree_has_boundaries() {
        let tree = SubgameTreeBuilder::new()
            .board(&make_flop_board())
            .bet_sizes(&[0.5, 1.0])
            .pot(100)
            .stacks(&[200, 200])
            .depth_limit(1)
            .build();
        let boundaries = tree
            .nodes
            .iter()
            .filter(|n| matches!(n, SubgameNode::DepthBoundary { .. }))
            .count();
        assert!(
            boundaries > 0,
            "depth-limited flop should have boundary nodes"
        );
    }

    #[timed_test]
    fn tree_root_is_decision() {
        let tree = SubgameTreeBuilder::new()
            .board(&make_river_board())
            .bet_sizes(&[1.0])
            .pot(100)
            .stacks(&[200, 200])
            .build();
        assert!(matches!(tree.nodes[0], SubgameNode::Decision { .. }));
    }

    #[timed_test]
    fn fold_child_is_terminal() {
        let tree = SubgameTreeBuilder::new()
            .board(&make_river_board())
            .bet_sizes(&[1.0])
            .pot(100)
            .stacks(&[200, 200])
            .build();
        // After a bet, opponent can fold
        if let SubgameNode::Decision {
            children, actions, ..
        } = &tree.nodes[0]
        {
            // Root: check or bet. Find bet child.
            let bet_idx = actions
                .iter()
                .position(|a| matches!(a, Action::Bet(_)))
                .expect("should have bet action");
            let bet_child = children[bet_idx] as usize;
            if let SubgameNode::Decision {
                children: c2,
                actions: a2,
                ..
            } = &tree.nodes[bet_child]
            {
                let fold_idx = a2
                    .iter()
                    .position(|a| *a == Action::Fold)
                    .expect("should have fold facing bet");
                let fold_child = c2[fold_idx] as usize;
                assert!(matches!(
                    tree.nodes[fold_child],
                    SubgameNode::Terminal { is_fold: true, .. }
                ));
            }
        }
    }

    #[timed_test]
    fn subgame_hands_excludes_board() {
        let board = make_river_board();
        let hands = SubgameHands::enumerate(&board);
        assert_eq!(
            hands.combos.len(),
            1081,
            "C(47,2) = 1081 for 5-card board"
        );
    }

    #[timed_test]
    fn subgame_hands_flop_count() {
        let board = make_flop_board();
        let hands = SubgameHands::enumerate(&board);
        assert_eq!(
            hands.combos.len(),
            1176,
            "C(49,2) = 1176 for 3-card board"
        );
    }
}
