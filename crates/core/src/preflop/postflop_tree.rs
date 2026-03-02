/// Postflop tree templates used by the preflop solver.
///
/// Each preflop terminal (non-all-in showdown) maps to a `PotType` which
/// selects a postflop tree template covering flop → turn → river betting.
/// The tree is purely structural: board and hand data are provided at solve time.
use crate::abstraction::Street;
use thiserror::Error;

use super::postflop_abstraction::MAX_POSTFLOP_ACTIONS;
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
    #[error("too many bet sizes ({count}): maximum is {max} (each bet size adds an action to decision nodes)")]
    TooManyBetSizes { count: usize, max: usize },
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
    /// In HU, SB/Button is always IP postflop regardless of preflop action.
    /// Player 0 is SB/Button, player 1 is BB.
    #[must_use]
    pub const fn default_ip_player(&self) -> u8 {
        match self {
            // SB/Button is always IP postflop in HU
            Self::Limped | Self::Raised | Self::ThreeBet | Self::FourBetPlus => 0,
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
        // OOP=1 (BB), IP=0 (SB/Button) — SB is always IP postflop in HU
        build_flop_subtree_spr(&mut nodes, 1, 0, config, 1.0, max_pot);
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
    // Worst case: constrained_raise_actions = fold + call + all sizes + all-in
    let max_actions = config.bet_sizes.len() + 3;
    if max_actions > MAX_POSTFLOP_ACTIONS {
        return Err(PostflopTreeError::TooManyBetSizes {
            count: config.bet_sizes.len(),
            max: MAX_POSTFLOP_ACTIONS - 3,
        });
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
                config.max_raises_per_street,
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
                    config.max_raises_per_street,
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
    // OOP=1 (BB), IP=0 (SB/Button) — consistent with default_ip_player()
    let oop = 1u8;
    let ip = 0u8;
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

/// With imperfect-recall per-street bucketing, board transitions are captured
/// by the hand bucket assignments themselves, so each chance node has exactly
/// one branch.
fn transition_count(_config: &PostflopModelConfig, _street: Street) -> usize {
    1
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
    for &s in bet_sizes {
        if f64::from(s) < available - 1e-6 {
            actions.push(PostflopAction::Bet(s));
        } else if !added_allin {
            actions.push(PostflopAction::Bet(available as f32));
            added_allin = true;
        }
    }
    // Always include all-in if not already present and meaningfully different from largest bet
    if !added_allin && available > 0.01 {
        let largest = actions.iter().filter_map(|a| match a {
            PostflopAction::Bet(f) => Some(f64::from(*f)),
            _ => None,
        }).fold(0.0f64, f64::max);
        if available - largest > 0.05 {
            actions.push(PostflopAction::Bet(available as f32));
        }
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
    let available = max_pot / new_pot - 1.0;
    if available <= 1e-9 {
        return actions;
    }

    if raises_remaining == 0 {
        // At raise cap: only all-in shove (no sized raises)
        if available > 0.01 {
            actions.push(PostflopAction::Raise(available as f32));
        }
        return actions;
    }

    // Add sized raises that fit below all-in, skipping those within 5% of all-in
    let mut largest_sized = 0.0f64;
    for &s in bet_sizes {
        let sf = f64::from(s);
        if sf < available - 1e-6 {
            if available - sf <= 0.05 {
                continue; // too close to all-in, redundant
            }
            actions.push(PostflopAction::Raise(s));
            largest_sized = largest_sized.max(sf);
        } else {
            // This size exceeds stack — will be replaced by all-in below
        }
    }

    // Always add all-in if meaningfully different from largest sized raise (>5% pot gap)
    if available > 0.01 && (available - largest_sized > 0.05 || largest_sized == 0.0) {
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
                config.max_raises_per_street,
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
                    config.max_raises_per_street,
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
                raises_remaining.saturating_sub(1),
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
    // OOP=1 (BB), IP=0 (SB/Button) — consistent with build_with_spr
    let oop = 1u8;
    let ip = 0u8;
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
            max_raises_per_street: 1,
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
    fn all_pot_types_have_sb_as_ip() {
        for pot_type in [
            PotType::Limped,
            PotType::Raised,
            PotType::ThreeBet,
            PotType::FourBetPlus,
        ] {
            assert_eq!(
                pot_type.default_ip_player(), 0,
                "SB (player 0) should always be IP postflop in HU, but {pot_type:?} returned {}",
                pot_type.default_ip_player()
            );
        }
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
    fn single_transition_per_chance_node() {
        let cfg = fast_config();
        let tree = PostflopTree::build(PotType::Raised, &cfg).unwrap();
        // With imperfect recall, every chance node should have exactly 1 child.
        for node in &tree.nodes {
            if let PostflopNode::Chance { children, .. } = node {
                assert_eq!(
                    children.len(),
                    1,
                    "chance nodes should have exactly 1 transition with imperfect recall"
                );
            }
        }
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
    fn spr_tree_deep_has_more_nodes_than_unconstrained() {
        // SPR=1000 → max_pot=2001 → no bet is capped, but SPR tree always adds all-in.
        // Both builders use consistent OOP=1(BB), IP=0(SB) positions.
        let cfg = fast_config();
        let spr_tree = PostflopTree::build_with_spr(&cfg, 1000.0).unwrap();
        let old_tree = PostflopTree::build(PotType::Raised, &cfg).unwrap();
        assert!(
            spr_tree.node_count() > old_tree.node_count(),
            "SPR tree ({}) should have more nodes than unconstrained ({}) due to always-allin",
            spr_tree.node_count(),
            old_tree.node_count(),
        );
    }

    #[timed_test]
    fn spr_tree_deep_allin_always_present_in_bets() {
        // With deep SPR, all-in should appear alongside configured bets
        let cfg = fast_config(); // bet_sizes: [0.5, 1.0]
        let tree = PostflopTree::build_with_spr(&cfg, 20.0).unwrap();
        // Root is opening decision — should have Check, Bet(0.5), Bet(1.0), Bet(all-in)
        match &tree.nodes[0] {
            PostflopNode::Decision { action_labels, .. } => {
                let bets: Vec<f32> = action_labels.iter().filter_map(|a| match a {
                    PostflopAction::Bet(f) => Some(*f),
                    _ => None,
                }).collect();
                assert!(
                    bets.len() >= 3,
                    "expected at least 3 bets (0.5, 1.0, all-in), got {bets:?}"
                );
                // Last bet should be the all-in (largest)
                let max_bet = bets.iter().copied().fold(0.0f32, f32::max);
                assert!(
                    max_bet > 1.0,
                    "largest bet should be all-in (> 1.0x pot), got {max_bet}"
                );
            }
            _ => panic!("root should be Decision"),
        }
    }

    #[timed_test]
    fn spr_tree_allin_available_at_raise_limit() {
        // At raise cap (raises_remaining=0), all-in shove should still be available
        let cfg = PostflopModelConfig {
            bet_sizes: vec![0.5, 1.0],
            max_raises_per_street: 1,
            ..PostflopModelConfig::fast()
        };
        let tree = PostflopTree::build_with_spr(&cfg, 5.0).unwrap();
        // Walk the tree to find a node at the raise limit
        // After: OOP bets, IP raises (1 raise used), OOP faces raise at limit
        let mut found_allin_at_cap = false;
        for node in &tree.nodes {
            if let PostflopNode::Decision { action_labels, .. } = node {
                let has_fold = action_labels.contains(&PostflopAction::Fold);
                let has_call = action_labels.contains(&PostflopAction::Call);
                let raises: Vec<&PostflopAction> = action_labels.iter()
                    .filter(|a| matches!(a, PostflopAction::Raise(_)))
                    .collect();
                // A node with Fold + Call + exactly 1 raise (all-in only) = raise cap node
                if has_fold && has_call && raises.len() == 1 {
                    found_allin_at_cap = true;
                    break;
                }
            }
        }
        assert!(
            found_allin_at_cap,
            "should find at least one node with Fold+Call+All-in at raise cap"
        );
    }

    /// Recursively walk the SPR tree, tracking pot, and verify every decision node
    /// with remaining chips includes an all-in action.
    fn assert_allin_at_every_node(
        nodes: &[PostflopNode],
        idx: u32,
        pot: f64,
        max_pot: f64,
        spr: f64,
        violations: &mut Vec<String>,
    ) {
        match &nodes[idx as usize] {
            PostflopNode::Terminal { .. } => {}
            PostflopNode::Chance { children, .. } => {
                for &child in children {
                    assert_allin_at_every_node(nodes, child, pot, max_pot, spr, violations);
                }
            }
            PostflopNode::Decision { action_labels, children, .. } => {
                let is_opening = action_labels.iter().any(|a| *a == PostflopAction::Check);
                let available = max_pot / pot - 1.0;

                if available > 0.01 {
                    if is_opening {
                        // Opening node: must have an all-in Bet
                        let has_allin_bet = action_labels.iter().any(|a| match a {
                            PostflopAction::Bet(f) => (f64::from(*f) - available).abs() < 0.02,
                            _ => false,
                        });
                        if !has_allin_bet {
                            let bets: Vec<f32> = action_labels.iter().filter_map(|a| match a {
                                PostflopAction::Bet(f) => Some(*f),
                                _ => None,
                            }).collect();
                            violations.push(format!(
                                "SPR={spr} node {idx} (opening): available={available:.3}, \
                                 no all-in bet (~{available:.2}), bets={bets:?}"
                            ));
                        }
                    } else {
                        // Facing-bet node: must have an all-in Raise
                        let has_allin_raise = action_labels.iter().any(|a| match a {
                            PostflopAction::Raise(f) => (f64::from(*f) - available).abs() < 0.02,
                            _ => false,
                        });
                        if !has_allin_raise {
                            let raises: Vec<f32> = action_labels.iter().filter_map(|a| match a {
                                PostflopAction::Raise(f) => Some(*f),
                                _ => None,
                            }).collect();
                            violations.push(format!(
                                "SPR={spr} node {idx} (facing bet): available={available:.3}, \
                                 no all-in raise (~{available:.2}), raises={raises:?}"
                            ));
                        }
                    }
                }

                // Recurse into children, computing new pot for each action
                for (action, &child) in action_labels.iter().zip(children.iter()) {
                    let child_pot = match action {
                        PostflopAction::Check => pot,
                        PostflopAction::Bet(f) | PostflopAction::Raise(f) => {
                            (pot * (1.0 + f64::from(*f))).min(max_pot)
                        }
                        PostflopAction::Call => pot, // caller matches bet; pot stays at new_pot
                        PostflopAction::Fold => pot,
                    };
                    assert_allin_at_every_node(nodes, child, child_pot, max_pot, spr, violations);
                }
            }
        }
    }

    #[timed_test]
    fn spr_tree_every_decision_has_allin() {
        for &spr in &[0.5, 2.0, 5.0, 10.0, 20.0] {
            let cfg = fast_config();
            let tree = PostflopTree::build_with_spr(&cfg, spr).unwrap();
            let max_pot = 1.0 + 2.0 * spr;
            let mut violations = Vec::new();
            assert_allin_at_every_node(&tree.nodes, 0, 1.0, max_pot, spr, &mut violations);
            assert!(
                violations.is_empty(),
                "All-in missing at {} node(s):\n{}",
                violations.len(),
                violations.join("\n")
            );
        }
    }

}
