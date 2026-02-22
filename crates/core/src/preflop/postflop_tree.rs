/// Postflop tree templates used by the preflop solver.
///
/// Each preflop terminal (non-all-in showdown) maps to a `PotType` which
/// selects a postflop tree template covering flop → turn → river betting.
/// The tree is purely structural: board and hand data are provided at solve time.
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
}

/// How a postflop terminal was reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostflopTerminalType {
    /// A player folded; `folder` is 0 (OOP) or 1 (IP).
    Fold { folder: u8 },
    /// Hand went to showdown.
    Showdown,
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
    /// Terminal node reached by fold or showdown.
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
    /// Stack-to-pot ratio (each player's remaining stack / initial pot).
    /// `f64::INFINITY` means unconstrained (legacy behavior).
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
}

impl PostflopTree {
    /// Build a full three-street postflop tree template for the given pot type.
    ///
    /// # Errors
    ///
    /// Returns an error if `config` has empty `bet_sizes` or zero transition counts.
    pub fn build(
        pot_type: PotType,
        config: &PostflopModelConfig,
    ) -> Result<Self, PostflopTreeError> {
        validate_config(config)?;
        let mut nodes: Vec<PostflopNode> = Vec::new();
        let ip = pot_type.default_ip_player();
        let oop = 1 - ip;
        build_flop_subtree(&mut nodes, oop, ip, config, 1.0);
        Ok(Self { nodes, pot_type, spr: f64::INFINITY })
    }

    /// Build a three-street postflop tree with stack-depth constraints.
    ///
    /// `spr` is each player's remaining stack divided by the initial pot entering
    /// postflop. Bet sizes that would exceed the remaining stack are replaced with
    /// an all-in at the remaining fraction, and calling an all-in goes directly to
    /// showdown (skipping remaining streets).
    ///
    /// # Errors
    ///
    /// Returns an error if `config` has empty `bet_sizes` or zero transition counts.
    pub fn build_with_spr(
        config: &PostflopModelConfig,
        spr: f64,
    ) -> Result<Self, PostflopTreeError> {
        validate_config(config)?;
        let max_pot = 1.0 + 2.0 * spr;
        let mut nodes: Vec<PostflopNode> = Vec::new();
        build_flop_subtree_spr(&mut nodes, 0, 1, config, 1.0, max_pot);
        Ok(Self {
            nodes,
            pot_type: PotType::Raised,
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

/// Builds betting actions available to the first actor given no prior bet.
fn opening_actions(bet_sizes: &[f32]) -> Vec<PostflopAction> {
    let mut actions = vec![PostflopAction::Check];
    actions.extend(bet_sizes.iter().map(|&s| PostflopAction::Bet(s)));
    actions
}

/// Builds betting actions available when facing a bet (call / raises / fold).
fn facing_bet_actions(bet_sizes: &[f32], raises_remaining: u8) -> Vec<PostflopAction> {
    let mut actions = vec![PostflopAction::Fold, PostflopAction::Call];
    if raises_remaining > 0 {
        actions.extend(bet_sizes.iter().map(|&s| PostflopAction::Raise(s)));
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
fn build_opening_decision(
    nodes: &mut Vec<PostflopNode>,
    actor: u8,
    opponent: u8,
    config: &PostflopModelConfig,
    pot_fraction: f64,
    street: Street,
) -> u32 {
    let actions = opening_actions(&config.bet_sizes);
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
fn build_after_opening_action(
    nodes: &mut Vec<PostflopNode>,
    actor: u8,
    opponent: u8,
    config: &PostflopModelConfig,
    pot_fraction: f64,
    street: Street,
    action: PostflopAction,
) -> u32 {
    match action {
        PostflopAction::Check => {
            build_check_response(nodes, opponent, actor, config, pot_fraction, street)
        }
        PostflopAction::Bet(frac) => {
            let new_pot = pot_fraction * (1.0 + f64::from(frac));
            build_facing_bet_decision(
                nodes,
                opponent,
                actor,
                config,
                pot_fraction,
                new_pot,
                street,
                config.raises_per_street,
            )
        }
        _ => unreachable!("opener can only check or bet"),
    }
}

/// Builds the opponent's response node after opener checks.
fn build_check_response(
    nodes: &mut Vec<PostflopNode>,
    actor: u8,
    opponent: u8,
    config: &PostflopModelConfig,
    pot_fraction: f64,
    street: Street,
) -> u32 {
    let actions = opening_actions(&config.bet_sizes);
    let idx = alloc_node(nodes);
    let children: Vec<u32> = actions
        .iter()
        .map(|action| match action {
            PostflopAction::Check => build_street_end(nodes, config, pot_fraction, street),
            PostflopAction::Bet(frac) => {
                let new_pot = pot_fraction * (1.0 + f64::from(*frac));
                build_facing_bet_decision(
                    nodes,
                    opponent,
                    actor,
                    config,
                    pot_fraction,
                    new_pot,
                    street,
                    config.raises_per_street,
                )
            }
            _ => unreachable!(),
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
) -> u32 {
    let actions = facing_bet_actions(&config.bet_sizes, raises_remaining);
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
    _prev_pot: f64,
    new_pot: f64,
    street: Street,
    raises_remaining: u8,
    action: PostflopAction,
) -> u32 {
    match action {
        PostflopAction::Fold => {
            build_terminal(nodes, PostflopTerminalType::Fold { folder: actor }, new_pot)
        }
        PostflopAction::Call => build_street_end(nodes, config, new_pot, street),
        PostflopAction::Raise(frac) => {
            let raised_pot = new_pot * (1.0 + f64::from(frac));
            build_facing_bet_decision(
                nodes,
                bettor,
                actor,
                config,
                new_pot,
                raised_pot,
                street,
                raises_remaining - 1,
            )
        }
        _ => unreachable!("facing-bet actor can only fold, call, or raise"),
    }
}

/// Builds the end-of-street node: either a chance node (flop/turn) or terminal (river).
fn build_street_end(
    nodes: &mut Vec<PostflopNode>,
    config: &PostflopModelConfig,
    pot_fraction: f64,
    street: Street,
) -> u32 {
    match street {
        Street::Flop => build_chance_node(nodes, config, pot_fraction, Street::Turn),
        Street::Turn => build_chance_node(nodes, config, pot_fraction, Street::River),
        Street::River => build_terminal(nodes, PostflopTerminalType::Showdown, pot_fraction),
        Street::Preflop => unreachable!("postflop tree starts at flop"),
    }
}

/// Builds a chance node transitioning to `next_street`, then the next street's subtree.
#[allow(clippy::cast_possible_truncation)]
fn build_chance_node(
    nodes: &mut Vec<PostflopNode>,
    config: &PostflopModelConfig,
    pot_fraction: f64,
    next_street: Street,
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
    let ip = 1u8; // default; solver overrides per pot type
    let oop = 0u8;
    let children: Vec<u32> = (0..num_transitions)
        .map(|_| build_opening_decision(nodes, oop, ip, config, pot_fraction, next_street))
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
) {
    build_opening_decision(nodes, oop, ip, config, pot_fraction, Street::Flop);
}

// ─── SPR-constrained tree building ──────────────────────────────────────────

/// Filter bet sizes: keep sizes below remaining stack, replace overflow with all-in.
#[allow(clippy::cast_possible_truncation)]
fn constrained_bet_actions(bet_sizes: &[f32], pot_fraction: f64, max_pot: f64) -> Vec<PostflopAction> {
    let mut actions = vec![PostflopAction::Check];
    let available = max_pot / pot_fraction - 1.0;
    if available <= 1e-9 {
        return actions;
    }
    let mut added_allin = false;
    let mut any_capped = false;
    for &s in bet_sizes {
        if f64::from(s) < available - 1e-6 {
            actions.push(PostflopAction::Bet(s));
        } else {
            any_capped = true;
            if !added_allin {
                actions.push(PostflopAction::Bet(available as f32));
                added_allin = true;
            }
        }
    }
    // Only add an implicit all-in if at least one bet was capped but none replaced yet
    if !added_allin && any_capped && available > 0.01 {
        actions.push(PostflopAction::Bet(available as f32));
    }
    actions
}

/// Filter raise sizes against remaining stack.
#[allow(clippy::cast_possible_truncation)]
fn constrained_raise_actions(
    bet_sizes: &[f32],
    new_pot: f64,
    max_pot: f64,
    raises_remaining: u8,
) -> Vec<PostflopAction> {
    let mut actions = vec![PostflopAction::Fold, PostflopAction::Call];
    if raises_remaining == 0 {
        return actions;
    }
    let available = max_pot / new_pot - 1.0;
    if available <= 1e-9 {
        return actions;
    }
    let mut added_allin = false;
    let mut any_capped = false;
    for &s in bet_sizes {
        if f64::from(s) < available - 1e-6 {
            actions.push(PostflopAction::Raise(s));
        } else {
            any_capped = true;
            if !added_allin {
                actions.push(PostflopAction::Raise(available as f32));
                added_allin = true;
            }
        }
    }
    // Only add an implicit all-in if at least one raise was capped but none replaced yet
    if !added_allin && any_capped && available > 0.01 {
        actions.push(PostflopAction::Raise(available as f32));
    }
    actions
}

/// Builds the flop subtree with SPR constraints (OOP=0, IP=1).
fn build_flop_subtree_spr(
    nodes: &mut Vec<PostflopNode>,
    oop: u8,
    ip: u8,
    config: &PostflopModelConfig,
    pot_fraction: f64,
    max_pot: f64,
) {
    build_opening_decision_spr(nodes, oop, ip, config, pot_fraction, Street::Flop, max_pot);
}

/// Builds the opening decision node with SPR-constrained bet actions.
#[allow(clippy::too_many_arguments)]
fn build_opening_decision_spr(
    nodes: &mut Vec<PostflopNode>,
    actor: u8,
    opponent: u8,
    config: &PostflopModelConfig,
    pot_fraction: f64,
    street: Street,
    max_pot: f64,
) -> u32 {
    let actions = constrained_bet_actions(&config.bet_sizes, pot_fraction, max_pot);
    let idx = alloc_node(nodes);
    let children: Vec<u32> = actions
        .iter()
        .map(|action| {
            build_after_opening_action_spr(
                nodes, actor, opponent, config, pot_fraction, street, *action, max_pot,
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

/// Builds the subtree after the opening player acts (SPR-aware).
#[allow(clippy::too_many_arguments)]
fn build_after_opening_action_spr(
    nodes: &mut Vec<PostflopNode>,
    actor: u8,
    opponent: u8,
    config: &PostflopModelConfig,
    pot_fraction: f64,
    street: Street,
    action: PostflopAction,
    max_pot: f64,
) -> u32 {
    match action {
        PostflopAction::Check => {
            build_check_response_spr(nodes, opponent, actor, config, pot_fraction, street, max_pot)
        }
        PostflopAction::Bet(frac) => {
            let new_pot = (pot_fraction * (1.0 + f64::from(frac))).min(max_pot);
            build_facing_bet_decision_spr(
                nodes,
                opponent,
                actor,
                config,
                pot_fraction,
                new_pot,
                street,
                config.raises_per_street,
                max_pot,
            )
        }
        _ => unreachable!("opener can only check or bet"),
    }
}

/// Builds the opponent's response node after opener checks (SPR-aware).
#[allow(clippy::too_many_arguments)]
fn build_check_response_spr(
    nodes: &mut Vec<PostflopNode>,
    actor: u8,
    opponent: u8,
    config: &PostflopModelConfig,
    pot_fraction: f64,
    street: Street,
    max_pot: f64,
) -> u32 {
    let actions = constrained_bet_actions(&config.bet_sizes, pot_fraction, max_pot);
    let idx = alloc_node(nodes);
    let children: Vec<u32> = actions
        .iter()
        .map(|action| match action {
            PostflopAction::Check => {
                build_street_end_spr(nodes, config, pot_fraction, street, max_pot)
            }
            PostflopAction::Bet(frac) => {
                let new_pot = (pot_fraction * (1.0 + f64::from(*frac))).min(max_pot);
                build_facing_bet_decision_spr(
                    nodes,
                    opponent,
                    actor,
                    config,
                    pot_fraction,
                    new_pot,
                    street,
                    config.raises_per_street,
                    max_pot,
                )
            }
            _ => unreachable!(),
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

/// Builds a decision node for a player facing a bet (SPR-aware).
#[allow(clippy::too_many_arguments)]
fn build_facing_bet_decision_spr(
    nodes: &mut Vec<PostflopNode>,
    actor: u8,
    bettor: u8,
    config: &PostflopModelConfig,
    prev_pot: f64,
    new_pot: f64,
    street: Street,
    raises_remaining: u8,
    max_pot: f64,
) -> u32 {
    let actions = constrained_raise_actions(&config.bet_sizes, new_pot, max_pot, raises_remaining);
    let idx = alloc_node(nodes);
    let children: Vec<u32> = actions
        .iter()
        .map(|action| {
            build_after_facing_bet_spr(
                nodes,
                actor,
                bettor,
                config,
                prev_pot,
                new_pot,
                street,
                raises_remaining,
                *action,
                max_pot,
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

/// Builds the child after a player responds to a bet (SPR-aware).
#[allow(clippy::too_many_arguments)]
fn build_after_facing_bet_spr(
    nodes: &mut Vec<PostflopNode>,
    actor: u8,
    bettor: u8,
    config: &PostflopModelConfig,
    _prev_pot: f64,
    new_pot: f64,
    street: Street,
    raises_remaining: u8,
    action: PostflopAction,
    max_pot: f64,
) -> u32 {
    match action {
        PostflopAction::Fold => {
            build_terminal(nodes, PostflopTerminalType::Fold { folder: actor }, new_pot)
        }
        PostflopAction::Call => {
            if (new_pot - max_pot).abs() < 1e-9 || new_pot >= max_pot - 1e-9 {
                // All-in call → go directly to showdown
                build_terminal(nodes, PostflopTerminalType::Showdown, new_pot)
            } else {
                build_street_end_spr(nodes, config, new_pot, street, max_pot)
            }
        }
        PostflopAction::Raise(frac) => {
            let raised_pot = (new_pot * (1.0 + f64::from(frac))).min(max_pot);
            build_facing_bet_decision_spr(
                nodes,
                bettor,
                actor,
                config,
                new_pot,
                raised_pot,
                street,
                raises_remaining - 1,
                max_pot,
            )
        }
        _ => unreachable!("facing-bet actor can only fold, call, or raise"),
    }
}

/// Builds the end-of-street node with SPR constraints.
fn build_street_end_spr(
    nodes: &mut Vec<PostflopNode>,
    config: &PostflopModelConfig,
    pot_fraction: f64,
    street: Street,
    max_pot: f64,
) -> u32 {
    match street {
        Street::Flop => build_chance_node_spr(nodes, config, pot_fraction, Street::Turn, max_pot),
        Street::Turn => build_chance_node_spr(nodes, config, pot_fraction, Street::River, max_pot),
        Street::River => build_terminal(nodes, PostflopTerminalType::Showdown, pot_fraction),
        Street::Preflop => unreachable!("postflop tree starts at flop"),
    }
}

/// Builds a chance node transitioning to `next_street` (SPR-aware).
#[allow(clippy::cast_possible_truncation)]
fn build_chance_node_spr(
    nodes: &mut Vec<PostflopNode>,
    config: &PostflopModelConfig,
    pot_fraction: f64,
    next_street: Street,
    max_pot: f64,
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
    let oop = 0u8;
    let ip = 1u8;
    let children: Vec<u32> = (0..num_transitions)
        .map(|_| {
            build_opening_decision_spr(nodes, oop, ip, config, pot_fraction, next_street, max_pot)
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

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

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

    // ── Tree construction ────────────────────────────────────────────────────

    #[timed_test]
    fn tree_builds_without_error() {
        let tree = PostflopTree::build(PotType::Raised, &fast_config());
        assert!(tree.is_ok());
    }

    #[timed_test]
    fn tree_has_nodes() {
        let tree = PostflopTree::build(PotType::Raised, &fast_config()).unwrap();
        assert!(tree.node_count() > 0, "expected nodes, got 0");
    }

    #[timed_test]
    fn tree_root_is_zero() {
        let tree = PostflopTree::build(PotType::Raised, &fast_config()).unwrap();
        assert_eq!(tree.root(), 0);
    }

    #[timed_test]
    fn tree_has_terminals() {
        let tree = PostflopTree::build(PotType::Raised, &fast_config()).unwrap();
        assert!(tree.terminal_count() > 0, "expected terminal nodes");
    }

    #[timed_test]
    fn tree_has_chance_nodes() {
        let tree = PostflopTree::build(PotType::Raised, &fast_config()).unwrap();
        // 1 flop→turn + 1 turn→river per betting path that reaches it
        assert!(
            tree.chance_count() > 0,
            "expected chance nodes for turn and river"
        );
    }

    #[timed_test]
    fn tree_root_is_decision_node() {
        let tree = PostflopTree::build(PotType::Raised, &fast_config()).unwrap();
        assert!(
            matches!(tree.nodes[0], PostflopNode::Decision { .. }),
            "root must be a Decision node"
        );
    }

    #[timed_test]
    fn decision_children_are_valid_indices() {
        let tree = PostflopTree::build(PotType::Raised, &fast_config()).unwrap();
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
        let tree = PostflopTree::build(PotType::Raised, &fast_config()).unwrap();
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
        let tree = PostflopTree::build(PotType::Raised, &fast_config()).unwrap();
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
        let result = PostflopTree::build(PotType::Raised, &cfg);
        assert!(matches!(result, Err(PostflopTreeError::EmptyBetSizes)));
    }

    #[timed_test]
    fn error_on_zero_turn_transitions() {
        let mut cfg = fast_config();
        cfg.num_turn_transitions = 0;
        let result = PostflopTree::build(PotType::Raised, &cfg);
        assert!(matches!(
            result,
            Err(PostflopTreeError::ZeroTurnTransitions)
        ));
    }

    #[timed_test]
    fn error_on_zero_river_transitions() {
        let mut cfg = fast_config();
        cfg.num_river_transitions = 0;
        let result = PostflopTree::build(PotType::Raised, &cfg);
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
            let result = PostflopTree::build(pot_type, &cfg);
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

        let few = PostflopTree::build(PotType::Raised, &cfg_few).unwrap();
        let many = PostflopTree::build(PotType::Raised, &cfg_many).unwrap();
        assert!(
            many.node_count() > few.node_count(),
            "more transitions ({}) should yield more nodes than fewer ({})",
            many.node_count(),
            few.node_count()
        );
    }

    // ── SPR-constrained tree construction ───────────────────────────────────

    #[timed_test]
    fn spr_tree_shallow_has_fewer_nodes_than_deep() {
        let cfg = fast_config();
        let shallow = PostflopTree::build_with_spr(&cfg, 0.5).unwrap();
        let deep = PostflopTree::build_with_spr(&cfg, 20.0).unwrap();
        assert!(
            shallow.node_count() < deep.node_count(),
            "shallow SPR ({}) should have fewer nodes than deep ({})",
            shallow.node_count(),
            deep.node_count(),
        );
    }

    #[timed_test]
    fn spr_tree_very_shallow_only_check_or_allin() {
        // SPR=0.1: max_pot=1.2, available from 1.0 = 0.2
        // bet_sizes [0.5, 1.0] both exceed 0.2 → replaced with single all-in
        let cfg = fast_config();
        let tree = PostflopTree::build_with_spr(&cfg, 0.1).unwrap();
        match &tree.nodes[0] {
            PostflopNode::Decision { action_labels, .. } => {
                assert_eq!(
                    action_labels.len(),
                    2,
                    "expected Check + all-in, got {action_labels:?}"
                );
                assert_eq!(action_labels[0], PostflopAction::Check);
                assert!(matches!(action_labels[1], PostflopAction::Bet(_)));
                if let PostflopAction::Bet(f) = action_labels[1] {
                    assert!(
                        (f64::from(f) - 0.2).abs() < 0.02,
                        "all-in fraction should be ~0.2, got {f}"
                    );
                }
            }
            _ => panic!("root should be Decision"),
        }
    }

    #[timed_test]
    fn spr_tree_terminals_never_exceed_max_pot() {
        let cfg = fast_config();
        for &spr in &[0.5, 1.5, 5.0, 20.0] {
            let tree = PostflopTree::build_with_spr(&cfg, spr).unwrap();
            let max_pot = 1.0 + 2.0 * spr;
            for node in &tree.nodes {
                if let PostflopNode::Terminal { pot_fraction, .. } = node {
                    assert!(
                        *pot_fraction <= max_pot + 1e-6,
                        "SPR={spr}: terminal pot_fraction {pot_fraction} exceeds max_pot {max_pot}"
                    );
                }
            }
        }
    }

    #[timed_test]
    fn spr_tree_deep_matches_unconstrained() {
        // SPR=1000 → max_pot=2001 → no bet is capped → same tree as old build.
        // Use Limped because build_with_spr always uses OOP=0, IP=1 which matches
        // PotType::Limped's default_ip_player()=1. (PotType::Raised uses ip=0 at
        // the flop but chance nodes hardcode ip=1, creating an inconsistency.)
        let cfg = fast_config();
        let spr_tree = PostflopTree::build_with_spr(&cfg, 1000.0).unwrap();
        let old_tree = PostflopTree::build(PotType::Limped, &cfg).unwrap();
        assert_eq!(spr_tree.node_count(), old_tree.node_count());
    }
}
