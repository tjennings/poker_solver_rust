/// Postflop tree templates used by the preflop solver.
///
/// Each preflop terminal (non-all-in showdown) maps to a `PotType` which
/// selects a postflop tree template covering flop → turn → river betting.
/// The tree is purely structural: board and hand data are provided at solve time.
///
/// Trees are SPR-aware: the stack-to-pot ratio constrains bet sizing and tree
/// depth. Low SPR (e.g., 4bet pots) produces shove-or-fold trees; high SPR
/// (e.g., limped pots) produces full multi-street trees.
use crate::abstraction::Street;
use thiserror::Error;

use super::postflop_model::PostflopModelConfig;

/// Classification of a preflop pot that determines which postflop tree to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PotType {
    /// Unraised pot (~4 chips). BB is in-position.
    Limped,
    /// Single-raised pot (~7-10 chips). Raiser is in-position.
    Raised,
    /// 3-bet pot (~20-30 chips).
    ThreeBet,
    /// 4-bet-plus pot (~50+ chips).
    FourBetPlus,
}

/// Actions available at a postflop decision node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PostflopAction {
    Check,
    Fold,
    Call,
    /// Bet as a fraction of the current pot.
    Bet(f32),
    /// Raise as a fraction of the current pot.
    Raise(f32),
    /// Shove all remaining chips.
    AllIn,
}

/// How a postflop terminal was reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostflopTerminalType {
    /// A player folded; `folder` is 0 (OOP) or 1 (IP).
    Fold { folder: u8 },
    /// Hand went to showdown.
    Showdown,
    /// Both players committed all chips (stacks exhausted).
    AllIn,
}

/// A node in the postflop game tree.
#[derive(Debug, Clone)]
pub enum PostflopNode {
    /// A player decision point.
    Decision {
        /// 0 = OOP, 1 = IP.
        position: u8,
        children: Vec<u32>,
        action_labels: Vec<PostflopAction>,
    },
    /// A street transition (chance node). Weights are filled at solve time.
    Chance {
        street: Street,
        children: Vec<u32>,
        /// Transition probabilities — one per transition cluster.
        weights: Vec<f64>,
    },
    /// Terminal node reached by fold, showdown, or all-in.
    Terminal {
        terminal_type: PostflopTerminalType,
        /// Pot size relative to the initial preflop pot (for scaling payoffs).
        pot_fraction: f64,
    },
}

/// Error types for postflop tree construction.
#[derive(Debug, Error)]
pub enum PostflopTreeError {
    #[error("bet_sizes must be non-empty")]
    EmptyBetSizes,
    #[error("num_turn_transitions must be > 0")]
    ZeroTurnTransitions,
    #[error("num_river_transitions must be > 0")]
    ZeroRiverTransitions,
}

/// A complete postflop game tree template for one pot type.
#[derive(Debug, Clone)]
pub struct PostflopTree {
    pub nodes: Vec<PostflopNode>,
    pub pot_type: PotType,
    /// Stack-to-pot ratio used to constrain this tree.
    pub spr: f64,
}

// ─── Public API ──────────────────────────────────────────────────────────────

impl PotType {
    /// Classify a preflop terminal by pot size and number of raises.
    ///
    /// `raise_count` is the number of raises (0 = limped, 1 = opened, 2 = 3-bet, ≥3 = 4-bet+).
    #[must_use]
    pub fn from_raise_count(raise_count: u8) -> Self {
        match raise_count {
            0 => Self::Limped,
            1 => Self::Raised,
            2 => Self::ThreeBet,
            _ => Self::FourBetPlus,
        }
    }

    /// Returns the IP player index (0 = OOP / first to act, 1 = IP / last to act).
    ///
    /// For HU: in a limped pot the BB (index 1 in our scheme, acts last preflop)
    /// has position postflop. For a raised pot the raiser has position.
    /// This method returns the default; solver integration overrides as needed.
    #[must_use]
    pub const fn default_ip_player(&self) -> u8 {
        match self {
            // BB acts last preflop when limped → IP postflop
            Self::Limped => 1,
            // Raiser (SB in HU) is IP postflop in most configurations
            Self::Raised | Self::ThreeBet | Self::FourBetPlus => 0,
        }
    }

    /// Default SPR for this pot type (25BB HU game).
    ///
    /// These are typical values; the integration layer can override with
    /// exact values computed from the preflop action sequence.
    #[must_use]
    pub const fn default_spr(&self) -> f64 {
        match self {
            Self::Limped => 12.0,
            Self::Raised => 4.5,
            Self::ThreeBet => 1.75,
            Self::FourBetPlus => 0.75,
        }
    }
}

impl PostflopTree {
    /// Build a postflop tree template constrained by `spr` (stack-to-pot ratio).
    ///
    /// Each player starts with `spr` chips (relative to the normalized pot of 1.0).
    /// Bet sizes that would exceed remaining stack are replaced with all-in.
    /// When both players are all-in, remaining streets are skipped.
    ///
    /// # Errors
    ///
    /// Returns an error if `config` has empty `bet_sizes` or zero transition counts.
    pub fn build(
        pot_type: PotType,
        config: &PostflopModelConfig,
        spr: f64,
    ) -> Result<Self, PostflopTreeError> {
        validate_config(config)?;
        let mut nodes: Vec<PostflopNode> = Vec::new();
        let ip = pot_type.default_ip_player();
        let oop = 1 - ip;
        build_flop_subtree(&mut nodes, oop, ip, config, 1.0, spr);
        Ok(Self {
            nodes,
            pot_type,
            spr,
        })
    }

    /// Returns the root node index (always 0).
    #[must_use]
    pub const fn root(&self) -> u32 {
        0
    }

    /// Returns the total number of nodes in the tree.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of terminal nodes.
    #[must_use]
    pub fn terminal_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| matches!(n, PostflopNode::Terminal { .. }))
            .count()
    }

    /// Returns the number of chance (street transition) nodes.
    #[must_use]
    pub fn chance_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| matches!(n, PostflopNode::Chance { .. }))
            .count()
    }

    /// Returns the number of all-in terminal nodes.
    #[must_use]
    pub fn allin_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| {
                matches!(
                    n,
                    PostflopNode::Terminal {
                        terminal_type: PostflopTerminalType::AllIn,
                        ..
                    }
                )
            })
            .count()
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

fn validate_config(config: &PostflopModelConfig) -> Result<(), PostflopTreeError> {
    if config.bet_sizes.is_empty() {
        return Err(PostflopTreeError::EmptyBetSizes);
    }
    if config.num_turn_transitions == 0 {
        return Err(PostflopTreeError::ZeroTurnTransitions);
    }
    if config.num_river_transitions == 0 {
        return Err(PostflopTreeError::ZeroRiverTransitions);
    }
    Ok(())
}

/// Allocates a placeholder node and returns its index.
#[allow(clippy::cast_possible_truncation)]
fn alloc_node(nodes: &mut Vec<PostflopNode>) -> u32 {
    // Safe: game trees never exceed u32::MAX nodes
    let idx = nodes.len() as u32;
    nodes.push(PostflopNode::Terminal {
        terminal_type: PostflopTerminalType::Showdown,
        pot_fraction: 0.0,
    });
    idx
}

/// Overwrites the node at `idx`.
fn set_node(nodes: &mut [PostflopNode], idx: u32, node: PostflopNode) {
    nodes[idx as usize] = node;
}

/// Builds opening actions with SPR-aware bet capping.
///
/// Bets that exceed `remaining_stack` are replaced with a single `AllIn`.
fn capped_opening_actions(
    bet_sizes: &[f32],
    pot_fraction: f64,
    remaining_stack: f64,
) -> Vec<PostflopAction> {
    let mut actions = vec![PostflopAction::Check];
    if remaining_stack <= 0.0 {
        return actions;
    }
    let mut has_allin = false;
    for &size in bet_sizes {
        let bet_amount = pot_fraction * f64::from(size);
        if bet_amount >= remaining_stack {
            if !has_allin {
                actions.push(PostflopAction::AllIn);
                has_allin = true;
            }
        } else {
            actions.push(PostflopAction::Bet(size));
        }
    }
    // If all bet sizes are affordable but remaining_stack is finite and no all-in
    // was generated, that's fine — normal betting.
    actions
}

/// Builds facing-bet actions with SPR-aware raise capping.
///
/// Raises that exceed remaining stack (after calling) are replaced with `AllIn`.
fn capped_facing_bet_actions(
    bet_sizes: &[f32],
    raises_remaining: u8,
    new_pot: f64,
    actor_stack: f64,
    call_cost: f64,
) -> Vec<PostflopAction> {
    let mut actions = vec![PostflopAction::Fold, PostflopAction::Call];
    if raises_remaining > 0 {
        let stack_after_call = actor_stack - call_cost;
        if stack_after_call > 0.0 {
            let mut has_allin = false;
            for &size in bet_sizes {
                let raise_amount = new_pot * f64::from(size);
                if raise_amount >= stack_after_call {
                    if !has_allin {
                        actions.push(PostflopAction::AllIn);
                        has_allin = true;
                    }
                } else {
                    actions.push(PostflopAction::Raise(size));
                }
            }
        }
    }
    actions
}

/// Builds a single terminal node, returning its index.
#[allow(clippy::cast_possible_truncation)]
fn build_terminal(
    nodes: &mut Vec<PostflopNode>,
    terminal_type: PostflopTerminalType,
    pot_fraction: f64,
) -> u32 {
    // Safe: game trees never exceed u32::MAX nodes
    let idx = nodes.len() as u32;
    nodes.push(PostflopNode::Terminal {
        terminal_type,
        pot_fraction,
    });
    idx
}

/// Builds the subtree starting with the first actor's opening decision on a street.
#[allow(clippy::too_many_arguments)]
fn build_opening_decision(
    nodes: &mut Vec<PostflopNode>,
    actor: u8,
    opponent: u8,
    config: &PostflopModelConfig,
    pot_fraction: f64,
    street: Street,
    remaining_stack: f64,
) -> u32 {
    let actions = capped_opening_actions(&config.bet_sizes, pot_fraction, remaining_stack);
    let idx = alloc_node(nodes);
    let children: Vec<u32> = actions
        .iter()
        .map(|action| {
            build_after_opening_action(
                nodes,
                actor,
                opponent,
                config,
                pot_fraction,
                street,
                *action,
                remaining_stack,
            )
        })
        .collect();
    set_node(
        nodes,
        idx,
        PostflopNode::Decision {
            position: actor,
            children,
            action_labels: actions,
        },
    );
    idx
}

/// Builds the subtree after the opening player acts.
#[allow(clippy::too_many_arguments)]
fn build_after_opening_action(
    nodes: &mut Vec<PostflopNode>,
    actor: u8,
    opponent: u8,
    config: &PostflopModelConfig,
    pot_fraction: f64,
    street: Street,
    action: PostflopAction,
    remaining_stack: f64,
) -> u32 {
    match action {
        PostflopAction::Check => {
            build_check_response(
                nodes, opponent, actor, config, pot_fraction, street, remaining_stack,
            )
        }
        PostflopAction::Bet(frac) => {
            let bet_amount = pot_fraction * f64::from(frac);
            let new_pot = pot_fraction + bet_amount;
            let actor_stack = remaining_stack - bet_amount;
            build_facing_bet_decision(
                nodes,
                opponent,
                actor,
                config,
                pot_fraction,
                new_pot,
                street,
                config.raises_per_street,
                remaining_stack, // opponent hasn't bet yet
                actor_stack,
            )
        }
        PostflopAction::AllIn => {
            let bet_amount = remaining_stack;
            let new_pot = pot_fraction + bet_amount;
            // Actor is all-in; opponent can only fold or call (no raises)
            build_facing_bet_decision(
                nodes,
                opponent,
                actor,
                config,
                pot_fraction,
                new_pot,
                street,
                0, // no raises against all-in
                remaining_stack, // opponent still has full stack
                0.0,            // actor is all-in
            )
        }
        _ => unreachable!("opener can only check, bet, or go all-in"),
    }
}

/// Builds the opponent's response node after opener checks.
#[allow(clippy::too_many_arguments)]
fn build_check_response(
    nodes: &mut Vec<PostflopNode>,
    actor: u8,
    opponent: u8,
    config: &PostflopModelConfig,
    pot_fraction: f64,
    street: Street,
    remaining_stack: f64,
) -> u32 {
    let actions = capped_opening_actions(&config.bet_sizes, pot_fraction, remaining_stack);
    let idx = alloc_node(nodes);
    let children: Vec<u32> = actions
        .iter()
        .map(|action| match action {
            PostflopAction::Check => {
                build_street_end(nodes, config, pot_fraction, street, remaining_stack, actor, opponent)
            }
            PostflopAction::Bet(frac) => {
                let bet_amount = pot_fraction * f64::from(*frac);
                let new_pot = pot_fraction + bet_amount;
                let actor_stack = remaining_stack - bet_amount;
                build_facing_bet_decision(
                    nodes,
                    opponent,
                    actor,
                    config,
                    pot_fraction,
                    new_pot,
                    street,
                    config.raises_per_street,
                    remaining_stack, // opponent hasn't bet
                    actor_stack,
                )
            }
            PostflopAction::AllIn => {
                let bet_amount = remaining_stack;
                let new_pot = pot_fraction + bet_amount;
                build_facing_bet_decision(
                    nodes,
                    opponent,
                    actor,
                    config,
                    pot_fraction,
                    new_pot,
                    street,
                    0,
                    remaining_stack,
                    0.0,
                )
            }
            _ => unreachable!("check-response can only check, bet, or go all-in"),
        })
        .collect();
    set_node(
        nodes,
        idx,
        PostflopNode::Decision {
            position: actor,
            children,
            action_labels: actions,
        },
    );
    idx
}

/// Builds a decision node for a player facing a bet.
#[allow(clippy::too_many_arguments)]
fn build_facing_bet_decision(
    nodes: &mut Vec<PostflopNode>,
    actor: u8,
    bettor: u8,
    config: &PostflopModelConfig,
    prev_pot: f64,
    new_pot: f64,
    street: Street,
    raises_remaining: u8,
    actor_stack: f64,
    bettor_stack: f64,
) -> u32 {
    let call_cost = new_pot - prev_pot;
    let actions =
        capped_facing_bet_actions(&config.bet_sizes, raises_remaining, new_pot, actor_stack, call_cost);
    let idx = alloc_node(nodes);
    let children: Vec<u32> = actions
        .iter()
        .map(|action| {
            build_after_facing_bet(
                nodes,
                actor,
                bettor,
                config,
                prev_pot,
                new_pot,
                street,
                raises_remaining,
                *action,
                actor_stack,
                bettor_stack,
            )
        })
        .collect();
    set_node(
        nodes,
        idx,
        PostflopNode::Decision {
            position: actor,
            children,
            action_labels: actions,
        },
    );
    idx
}

/// Builds the child after a player responds to a bet.
#[allow(clippy::too_many_arguments)]
fn build_after_facing_bet(
    nodes: &mut Vec<PostflopNode>,
    actor: u8,
    bettor: u8,
    config: &PostflopModelConfig,
    prev_pot: f64,
    new_pot: f64,
    street: Street,
    raises_remaining: u8,
    action: PostflopAction,
    actor_stack: f64,
    bettor_stack: f64,
) -> u32 {
    let call_cost = new_pot - prev_pot;
    match action {
        PostflopAction::Fold => {
            build_terminal(nodes, PostflopTerminalType::Fold { folder: actor }, new_pot)
        }
        PostflopAction::Call => {
            // After calling, both players have equal remaining stacks
            let remaining_after_call = actor_stack - call_cost;
            build_street_end(
                nodes,
                config,
                new_pot,
                street,
                remaining_after_call,
                actor,
                bettor,
            )
        }
        PostflopAction::Raise(frac) => {
            let raise_amount = new_pot * f64::from(frac);
            let raised_pot = new_pot + raise_amount;
            let actor_after_raise = actor_stack - call_cost - raise_amount;
            build_facing_bet_decision(
                nodes,
                bettor,
                actor,
                config,
                new_pot,
                raised_pot,
                street,
                raises_remaining - 1,
                bettor_stack,
                actor_after_raise,
            )
        }
        PostflopAction::AllIn => {
            // Actor shoves remaining chips as a raise
            let raise_amount = actor_stack - call_cost;
            let raised_pot = new_pot + raise_amount;
            // Actor is all-in; bettor can only fold or call
            build_facing_bet_decision(
                nodes,
                bettor,
                actor,
                config,
                new_pot,
                raised_pot,
                street,
                0, // no more raises
                bettor_stack,
                0.0, // actor is all-in
            )
        }
        _ => unreachable!("facing-bet actor can only fold, call, raise, or go all-in"),
    }
}

/// Builds the end-of-street node: either a chance node (flop/turn) or terminal (river).
///
/// If `remaining_stack` is zero (both players all-in), skips remaining streets
/// and goes directly to an `AllIn` terminal.
#[allow(clippy::too_many_arguments)]
fn build_street_end(
    nodes: &mut Vec<PostflopNode>,
    config: &PostflopModelConfig,
    pot_fraction: f64,
    street: Street,
    remaining_stack: f64,
    oop: u8,
    ip: u8,
) -> u32 {
    // Both players all-in — no more decisions possible
    if remaining_stack <= 0.0 {
        return build_terminal(nodes, PostflopTerminalType::AllIn, pot_fraction);
    }
    match street {
        Street::Flop => {
            build_chance_node(nodes, config, pot_fraction, Street::Turn, remaining_stack, oop, ip)
        }
        Street::Turn => {
            build_chance_node(nodes, config, pot_fraction, Street::River, remaining_stack, oop, ip)
        }
        Street::River => build_terminal(nodes, PostflopTerminalType::Showdown, pot_fraction),
        Street::Preflop => unreachable!("postflop tree starts at flop"),
    }
}

/// Builds a chance node transitioning to `next_street`, then the next street's subtree.
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
fn build_chance_node(
    nodes: &mut Vec<PostflopNode>,
    config: &PostflopModelConfig,
    pot_fraction: f64,
    next_street: Street,
    remaining_stack: f64,
    oop: u8,
    ip: u8,
) -> u32 {
    let num_transitions = transition_count(config, next_street);
    // Safe: game trees never exceed u32::MAX nodes
    let idx = nodes.len() as u32;
    #[allow(clippy::cast_precision_loss)]
    let uniform_weight = 1.0_f64 / num_transitions as f64;
    // Push placeholder; children are built next
    nodes.push(PostflopNode::Chance {
        street: next_street,
        children: vec![],
        weights: vec![uniform_weight; num_transitions],
    });
    let children: Vec<u32> = (0..num_transitions)
        .map(|_| {
            build_opening_decision(nodes, oop, ip, config, pot_fraction, next_street, remaining_stack)
        })
        .collect();
    if let PostflopNode::Chance {
        children: ref mut slot,
        ..
    } = nodes[idx as usize]
    {
        *slot = children;
    }
    idx
}

fn transition_count(config: &PostflopModelConfig, street: Street) -> usize {
    match street {
        Street::Turn => config.num_turn_transitions as usize,
        Street::River => config.num_river_transitions as usize,
        _ => 1,
    }
}

/// Builds the flop subtree (OOP acts first postflop).
fn build_flop_subtree(
    nodes: &mut Vec<PostflopNode>,
    oop: u8,
    ip: u8,
    config: &PostflopModelConfig,
    pot_fraction: f64,
    remaining_stack: f64,
) {
    build_opening_decision(nodes, oop, ip, config, pot_fraction, Street::Flop, remaining_stack);
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    /// Large SPR that doesn't constrain the tree — matches legacy behavior.
    const DEEP_SPR: f64 = 100.0;

    fn fast_config() -> PostflopModelConfig {
        PostflopModelConfig {
            bet_sizes: vec![0.5, 1.0],
            raises_per_street: 1,
            num_turn_transitions: 2,
            num_river_transitions: 2,
            ..PostflopModelConfig::fast()
        }
    }

    // ── PotType ──────────────────────────────────────────────────────────────

    #[timed_test]
    fn pot_type_from_zero_raises_is_limped() {
        assert_eq!(PotType::from_raise_count(0), PotType::Limped);
    }

    #[timed_test]
    fn pot_type_from_one_raise_is_raised() {
        assert_eq!(PotType::from_raise_count(1), PotType::Raised);
    }

    #[timed_test]
    fn pot_type_from_two_raises_is_three_bet() {
        assert_eq!(PotType::from_raise_count(2), PotType::ThreeBet);
    }

    #[timed_test]
    fn pot_type_from_three_raises_is_four_bet_plus() {
        assert_eq!(PotType::from_raise_count(3), PotType::FourBetPlus);
    }

    #[timed_test]
    fn pot_type_from_many_raises_is_four_bet_plus() {
        assert_eq!(PotType::from_raise_count(10), PotType::FourBetPlus);
    }

    #[timed_test]
    fn default_spr_values() {
        assert!((PotType::Limped.default_spr() - 12.0).abs() < f64::EPSILON);
        assert!((PotType::Raised.default_spr() - 4.5).abs() < f64::EPSILON);
        assert!((PotType::ThreeBet.default_spr() - 1.75).abs() < f64::EPSILON);
        assert!((PotType::FourBetPlus.default_spr() - 0.75).abs() < f64::EPSILON);
    }

    // ── Tree construction (deep SPR — legacy behavior) ───────────────────────

    #[timed_test]
    fn tree_builds_without_error() {
        let tree = PostflopTree::build(PotType::Raised, &fast_config(), DEEP_SPR);
        assert!(tree.is_ok());
    }

    #[timed_test]
    fn tree_has_nodes() {
        let tree = PostflopTree::build(PotType::Raised, &fast_config(), DEEP_SPR).unwrap();
        assert!(tree.node_count() > 0, "expected nodes, got 0");
    }

    #[timed_test]
    fn tree_root_is_zero() {
        let tree = PostflopTree::build(PotType::Raised, &fast_config(), DEEP_SPR).unwrap();
        assert_eq!(tree.root(), 0);
    }

    #[timed_test]
    fn tree_has_terminals() {
        let tree = PostflopTree::build(PotType::Raised, &fast_config(), DEEP_SPR).unwrap();
        assert!(tree.terminal_count() > 0, "expected terminal nodes");
    }

    #[timed_test]
    fn tree_has_chance_nodes() {
        let tree = PostflopTree::build(PotType::Raised, &fast_config(), DEEP_SPR).unwrap();
        assert!(
            tree.chance_count() > 0,
            "expected chance nodes for turn and river"
        );
    }

    #[timed_test]
    fn tree_root_is_decision_node() {
        let tree = PostflopTree::build(PotType::Raised, &fast_config(), DEEP_SPR).unwrap();
        assert!(
            matches!(tree.nodes[0], PostflopNode::Decision { .. }),
            "root must be a Decision node"
        );
    }

    #[timed_test]
    fn decision_children_are_valid_indices() {
        let tree = PostflopTree::build(PotType::Raised, &fast_config(), DEEP_SPR).unwrap();
        let n = tree.node_count() as u32;
        for node in &tree.nodes {
            if let PostflopNode::Decision { children, .. } = node {
                for &c in children {
                    assert!(c < n, "child index {c} out of bounds (tree has {n} nodes)");
                }
            }
        }
    }

    #[timed_test]
    fn chance_children_are_valid_indices() {
        let tree = PostflopTree::build(PotType::Raised, &fast_config(), DEEP_SPR).unwrap();
        let n = tree.node_count() as u32;
        for node in &tree.nodes {
            if let PostflopNode::Chance { children, .. } = node {
                for &c in children {
                    assert!(c < n, "chance child index {c} out of bounds");
                }
            }
        }
    }

    #[timed_test]
    fn chance_weights_sum_to_one() {
        let tree = PostflopTree::build(PotType::Raised, &fast_config(), DEEP_SPR).unwrap();
        for node in &tree.nodes {
            if let PostflopNode::Chance { weights, .. } = node {
                let sum: f64 = weights.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-9,
                    "chance weights should sum to 1.0, got {sum}"
                );
            }
        }
    }

    #[timed_test]
    fn error_on_empty_bet_sizes() {
        let mut cfg = fast_config();
        cfg.bet_sizes = vec![];
        let result = PostflopTree::build(PotType::Raised, &cfg, DEEP_SPR);
        assert!(matches!(result, Err(PostflopTreeError::EmptyBetSizes)));
    }

    #[timed_test]
    fn error_on_zero_turn_transitions() {
        let mut cfg = fast_config();
        cfg.num_turn_transitions = 0;
        let result = PostflopTree::build(PotType::Raised, &cfg, DEEP_SPR);
        assert!(matches!(
            result,
            Err(PostflopTreeError::ZeroTurnTransitions)
        ));
    }

    #[timed_test]
    fn error_on_zero_river_transitions() {
        let mut cfg = fast_config();
        cfg.num_river_transitions = 0;
        let result = PostflopTree::build(PotType::Raised, &cfg, DEEP_SPR);
        assert!(matches!(
            result,
            Err(PostflopTreeError::ZeroRiverTransitions)
        ));
    }

    #[timed_test]
    fn all_pot_types_build_successfully() {
        let cfg = fast_config();
        for pot_type in [
            PotType::Limped,
            PotType::Raised,
            PotType::ThreeBet,
            PotType::FourBetPlus,
        ] {
            let result = PostflopTree::build(pot_type, &cfg, DEEP_SPR);
            assert!(result.is_ok(), "failed to build tree for {pot_type:?}");
        }
    }

    #[timed_test]
    fn more_transitions_means_more_nodes() {
        let mut cfg_few = fast_config();
        cfg_few.num_turn_transitions = 1;
        cfg_few.num_river_transitions = 1;

        let mut cfg_many = fast_config();
        cfg_many.num_turn_transitions = 4;
        cfg_many.num_river_transitions = 4;

        let few = PostflopTree::build(PotType::Raised, &cfg_few, DEEP_SPR).unwrap();
        let many = PostflopTree::build(PotType::Raised, &cfg_many, DEEP_SPR).unwrap();
        assert!(
            many.node_count() > few.node_count(),
            "more transitions ({}) should yield more nodes than fewer ({})",
            many.node_count(),
            few.node_count()
        );
    }

    #[timed_test]
    fn spr_stored_on_tree() {
        let tree = PostflopTree::build(PotType::Raised, &fast_config(), 4.5).unwrap();
        assert!((tree.spr - 4.5).abs() < f64::EPSILON);
    }

    // ── SPR-aware tree shape ─────────────────────────────────────────────────

    #[timed_test]
    fn different_sprs_produce_different_tree_shapes() {
        let cfg = fast_config();
        let deep = PostflopTree::build(PotType::Raised, &cfg, DEEP_SPR).unwrap();
        let medium = PostflopTree::build(PotType::Raised, &cfg, 4.5).unwrap();
        let shallow = PostflopTree::build(PotType::Raised, &cfg, 0.75).unwrap();

        // Shallower SPR should produce fewer nodes (less branching, more all-in)
        assert!(
            deep.node_count() >= medium.node_count(),
            "deep ({}) should have >= nodes than medium ({})",
            deep.node_count(),
            medium.node_count()
        );
        assert!(
            medium.node_count() > shallow.node_count(),
            "medium ({}) should have more nodes than shallow ({})",
            medium.node_count(),
            shallow.node_count()
        );
    }

    #[timed_test]
    fn shallow_spr_has_allin_terminals() {
        let cfg = fast_config();
        let tree = PostflopTree::build(PotType::Raised, &cfg, 0.75).unwrap();
        assert!(
            tree.allin_count() > 0,
            "shallow SPR tree should have all-in terminals"
        );
    }

    #[timed_test]
    fn deep_spr_has_no_allin_terminals() {
        let cfg = fast_config();
        let tree = PostflopTree::build(PotType::Raised, &cfg, DEEP_SPR).unwrap();
        assert_eq!(
            tree.allin_count(),
            0,
            "deep SPR tree should have no all-in terminals"
        );
    }

    #[timed_test]
    fn shallow_spr_fewer_chance_nodes() {
        let cfg = fast_config();
        let deep = PostflopTree::build(PotType::Raised, &cfg, DEEP_SPR).unwrap();
        let shallow = PostflopTree::build(PotType::Raised, &cfg, 0.75).unwrap();
        assert!(
            shallow.chance_count() < deep.chance_count(),
            "shallow SPR ({}) should have fewer chance nodes than deep ({})",
            shallow.chance_count(),
            deep.chance_count()
        );
    }

    #[timed_test]
    fn very_shallow_spr_is_shove_or_fold() {
        // SPR = 0.3: all bets (0.5, 1.0) exceed stack → only Check/AllIn
        let cfg = fast_config();
        let tree = PostflopTree::build(PotType::Raised, &cfg, 0.3).unwrap();
        // Root should be a decision with Check and AllIn
        if let PostflopNode::Decision {
            action_labels, ..
        } = &tree.nodes[0]
        {
            assert!(
                action_labels.contains(&PostflopAction::Check),
                "root should have Check"
            );
            assert!(
                action_labels.contains(&PostflopAction::AllIn),
                "root should have AllIn"
            );
            assert!(
                !action_labels.iter().any(|a| matches!(a, PostflopAction::Bet(_))),
                "root should not have regular bets with SPR=0.3"
            );
        } else {
            panic!("root must be a Decision node");
        }
    }

    #[timed_test]
    fn medium_spr_has_mixed_actions() {
        // SPR = 2.0: Bet(0.5) affordable (cost 0.5 < 2.0), Bet(1.0) affordable (1.0 < 2.0)
        let cfg = fast_config();
        let tree = PostflopTree::build(PotType::Raised, &cfg, 2.0).unwrap();
        if let PostflopNode::Decision {
            action_labels, ..
        } = &tree.nodes[0]
        {
            assert!(action_labels.contains(&PostflopAction::Check));
            assert!(action_labels.contains(&PostflopAction::Bet(0.5)));
            assert!(action_labels.contains(&PostflopAction::Bet(1.0)));
        } else {
            panic!("root must be a Decision node");
        }
    }

    #[timed_test]
    fn allin_followed_by_call_yields_allin_terminal() {
        // Build a shallow tree where all-in + call should produce AllIn terminal
        let cfg = PostflopModelConfig {
            bet_sizes: vec![1.0],
            raises_per_street: 0,
            num_turn_transitions: 1,
            num_river_transitions: 1,
            ..PostflopModelConfig::fast()
        };
        let tree = PostflopTree::build(PotType::Raised, &cfg, 0.5).unwrap();
        // Verify at least one AllIn terminal exists
        let has_allin_terminal = tree.nodes.iter().any(|n| {
            matches!(
                n,
                PostflopNode::Terminal {
                    terminal_type: PostflopTerminalType::AllIn,
                    ..
                }
            )
        });
        assert!(has_allin_terminal, "should have AllIn terminal from shove+call");
    }

    #[timed_test]
    fn all_pot_types_with_default_spr_build_successfully() {
        let cfg = fast_config();
        for pot_type in [
            PotType::Limped,
            PotType::Raised,
            PotType::ThreeBet,
            PotType::FourBetPlus,
        ] {
            let spr = pot_type.default_spr();
            let result = PostflopTree::build(pot_type, &cfg, spr);
            assert!(
                result.is_ok(),
                "failed to build tree for {pot_type:?} with SPR={spr}"
            );
        }
    }

    #[timed_test]
    fn default_spr_trees_differ_across_pot_types() {
        let cfg = fast_config();
        let limped = PostflopTree::build(PotType::Limped, &cfg, PotType::Limped.default_spr()).unwrap();
        let raised = PostflopTree::build(PotType::Raised, &cfg, PotType::Raised.default_spr()).unwrap();
        let three_bet =
            PostflopTree::build(PotType::ThreeBet, &cfg, PotType::ThreeBet.default_spr()).unwrap();
        let four_bet =
            PostflopTree::build(PotType::FourBetPlus, &cfg, PotType::FourBetPlus.default_spr())
                .unwrap();

        // Trees with different SPRs should differ in node count
        assert_ne!(
            limped.node_count(),
            four_bet.node_count(),
            "limped and 4bet+ trees should differ (SPR {} vs {})",
            limped.spr,
            four_bet.spr,
        );
        // Deeper SPR → more nodes
        assert!(
            limped.node_count() > raised.node_count(),
            "limped ({}, SPR={}) should have more nodes than raised ({}, SPR={})",
            limped.node_count(),
            limped.spr,
            raised.node_count(),
            raised.spr,
        );
        assert!(
            raised.node_count() > three_bet.node_count(),
            "raised ({}, SPR={}) should have more nodes than 3bet ({}, SPR={})",
            raised.node_count(),
            raised.spr,
            three_bet.node_count(),
            three_bet.spr,
        );
        assert!(
            three_bet.node_count() > four_bet.node_count(),
            "3bet ({}, SPR={}) should have more nodes than 4bet+ ({}, SPR={})",
            three_bet.node_count(),
            three_bet.spr,
            four_bet.node_count(),
            four_bet.spr,
        );
    }

    #[timed_test]
    fn structural_invariants_hold_at_all_sprs() {
        let cfg = fast_config();
        for spr in [0.3, 0.75, 1.0, 1.75, 4.5, 12.0, DEEP_SPR] {
            let tree = PostflopTree::build(PotType::Raised, &cfg, spr).unwrap();
            let n = tree.node_count() as u32;
            // All children are valid indices
            for node in &tree.nodes {
                match node {
                    PostflopNode::Decision { children, .. } => {
                        for &c in children {
                            assert!(c < n, "SPR={spr}: child {c} out of bounds ({n} nodes)");
                        }
                    }
                    PostflopNode::Chance { children, weights, .. } => {
                        for &c in children {
                            assert!(c < n, "SPR={spr}: chance child {c} out of bounds");
                        }
                        let sum: f64 = weights.iter().sum();
                        assert!(
                            (sum - 1.0).abs() < 1e-9,
                            "SPR={spr}: chance weights sum to {sum}"
                        );
                    }
                    PostflopNode::Terminal { .. } => {}
                }
            }
        }
    }
}
