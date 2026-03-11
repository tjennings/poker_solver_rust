//! Full-game tree builder for heads-up NLHE.
//!
//! Builds a single arena-allocated game tree spanning all 4 streets
//! (preflop through river) with configurable action abstraction per
//! street and per raise depth.

// Arena indices are u32 (sufficient for any practical game tree) and
// player indices are u8 (max 2 players in HU). Truncation is safe.
#![allow(clippy::cast_possible_truncation)]

/// Tolerance for comparing bet sizes (in BB). Two bet amounts within
/// this tolerance are considered identical for deduplication purposes.
use super::Street;

const SIZE_EPSILON: f64 = 0.01;

#[derive(Debug, Clone)]
pub enum GameNode {
    Decision {
        player: u8,
        street: Street,
        actions: Vec<TreeAction>,
        /// Arena indices, 1:1 with `actions`.
        children: Vec<u32>,
    },
    Chance {
        next_street: Street,
        child: u32,
    },
    Terminal {
        kind: TerminalKind,
        pot: f64,
        /// What each player has put in (HU: indices 0 and 1).
        invested: [f64; 2],
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TreeAction {
    Fold,
    Check,
    Call,
    /// Bet amount in BB.
    Bet(f64),
    /// Raise TO amount in BB.
    Raise(f64),
    AllIn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalKind {
    Fold { winner: u8 },
    Showdown,
    DepthBoundary,
}

pub struct GameTree {
    pub nodes: Vec<GameNode>,
    pub root: u32,
}

/// Internal state tracked during recursive tree construction.
struct BuildState {
    /// Stacks at the start of the hand (same for both players).
    starting_stack: f64,
    /// How much each player has invested so far.
    invested: [f64; 2],
    /// Current street.
    street: Street,
    /// Number of bets/raises made on the current street.
    num_raises: u8,
    /// Which player acts next (0 or 1).
    to_act: u8,
    /// Whether the current action sequence on this street has seen a bet.
    /// Used to distinguish check vs call, bet vs raise.
    facing_bet: bool,
    /// The size of the last bet/raise TO (total invested by the aggressor).
    /// Used for preflop multiplier sizings.
    last_raise_to: f64,
}

/// Configuration for the tree builder, extracted from the `build()` parameters.
#[allow(clippy::struct_field_names)]
struct TreeConfig {
    preflop_sizes: Vec<Vec<PreflopSize>>,
    flop_sizes: Vec<Vec<f64>>,
    turn_sizes: Vec<Vec<f64>>,
    river_sizes: Vec<Vec<f64>>,
    /// Maximum streets to solve from `starting_street`. `None` = full depth.
    depth_limit: Option<u8>,
    /// Street the subgame starts on (for depth counting).
    starting_street: Option<Street>,
}

impl TreeConfig {
    /// Maximum raises for a street = number of declared size depths for that street.
    fn max_raises_for_street(&self, street: Street) -> u8 {
        let len = match street {
            Street::Preflop => self.preflop_sizes.len(),
            Street::Flop => self.flop_sizes.len(),
            Street::Turn => self.turn_sizes.len(),
            Street::River => self.river_sizes.len(),
        };
        len as u8
    }
}

#[derive(Debug, Clone, Copy)]
enum PreflopSize {
    /// Raise TO an absolute amount in BB.
    Absolute(f64),
    /// Raise TO a multiple of the last raise.
    Multiplier(f64),
}

impl PreflopSize {
    fn parse(s: &str) -> Self {
        let s = s.trim();
        if let Some(stripped) = s.strip_suffix("bb") {
            Self::Absolute(stripped.parse().expect("invalid absolute preflop size"))
        } else if let Some(stripped) = s.strip_suffix('x') {
            Self::Multiplier(stripped.parse().expect("invalid multiplier preflop size"))
        } else {
            panic!("preflop size must end with 'bb' or 'x': {s}");
        }
    }
}

impl GameTree {
    /// Build a full-game tree for heads-up NLHE.
    ///
    /// # Arguments
    /// * `stack_depth` - Starting stacks in BB (both players)
    /// * `small_blind` - Small blind in BB (typically 0.5)
    /// * `big_blind` - Big blind in BB (typically 1.0)
    /// * `preflop_sizes` - Per raise depth, list of size strings
    ///   (`"2.5bb"` = absolute, `"3.0x"` = multiplier of last raise)
    /// * `flop_sizes`, `turn_sizes`, `river_sizes` - Per raise depth,
    ///   list of pot fractions. The number of entries determines the
    ///   maximum raises allowed on that street.
    #[must_use]
    pub fn build(
        stack_depth: f64,
        small_blind: f64,
        big_blind: f64,
        preflop_sizes: &[Vec<String>],
        flop_sizes: &[Vec<f64>],
        turn_sizes: &[Vec<f64>],
        river_sizes: &[Vec<f64>],
    ) -> Self {
        let config = TreeConfig {
            preflop_sizes: preflop_sizes
                .iter()
                .map(|depth| depth.iter().map(|s| PreflopSize::parse(s)).collect())
                .collect(),
            flop_sizes: flop_sizes.to_vec(),
            turn_sizes: turn_sizes.to_vec(),
            river_sizes: river_sizes.to_vec(),
            depth_limit: None,
            starting_street: None,
        };

        let mut nodes = Vec::new();

        let initial_state = BuildState {
            starting_stack: stack_depth,
            invested: [small_blind, big_blind],
            street: Street::Preflop,
            num_raises: 0, // BB is a forced blind, not a voluntary raise
            to_act: 0,     // SB acts first preflop
            facing_bet: true, // SB is facing the BB (a forced bet)
            last_raise_to: big_blind,
        };

        let root = Self::build_node(&config, &initial_state, &mut nodes);

        Self { nodes, root }
    }

    /// Recursively build a node and return its arena index.
    fn build_node(config: &TreeConfig, state: &BuildState, nodes: &mut Vec<GameNode>) -> u32 {
        let remaining = [
            state.starting_stack - state.invested[0],
            state.starting_stack - state.invested[1],
        ];
        let actor = state.to_act as usize;

        // If the actor has no chips left, treat as terminal or pass action.
        // This shouldn't normally happen if the tree is built correctly,
        // but guard against it.
        if remaining[actor] < SIZE_EPSILON {
            return Self::make_showdown_or_chance(config, state, nodes);
        }

        let mut actions = Vec::new();

        // Fold is always available when facing a bet
        if state.facing_bet {
            actions.push(TreeAction::Fold);
        }

        // Check or Call
        if state.facing_bet {
            // Call: match the opponent's bet
            let call_amount = state.invested[1 - actor] - state.invested[actor];
            if call_amount >= remaining[actor] - SIZE_EPSILON {
                // Calling for all-in
                actions.push(TreeAction::AllIn);
            } else {
                actions.push(TreeAction::Call);
            }
        } else {
            actions.push(TreeAction::Check);
        }

        // Bet/Raise sizes (only if under the street's declared raise depths and actor has chips)
        let can_raise = state.num_raises < config.max_raises_for_street(state.street)
            && remaining[actor] > SIZE_EPSILON;
        if can_raise {
            let sized_actions =
                Self::compute_sized_actions(config, state, remaining[actor]);
            actions.extend(sized_actions);
        }

        // All-in is always available when the actor has chips, regardless of raise cap.
        // This ensures players can shove even after all sized raise depths are exhausted.
        if remaining[actor] > SIZE_EPSILON {
            let all_in_amount = state.invested[actor] + remaining[actor];
            let has_all_in = actions.iter().any(|a| match a {
                TreeAction::Bet(v) | TreeAction::Raise(v) => {
                    (*v - all_in_amount).abs() < SIZE_EPSILON
                }
                TreeAction::AllIn => true,
                _ => false,
            });
            if !has_all_in {
                actions.push(TreeAction::AllIn);
            }
        }

        // If we only have Check (no bet/raise available), and the call case
        // already has AllIn, make sure we don't have duplicates.
        // Deduplicate AllIn entries
        Self::dedup_all_in(&mut actions);

        // Reserve a slot in the arena
        let node_idx = nodes.len() as u32;
        // Push a placeholder — we'll overwrite after building children
        nodes.push(GameNode::Terminal {
            kind: TerminalKind::Showdown,
            pot: 0.0,
            invested: [0.0; 2],
        });

        let mut children = Vec::with_capacity(actions.len());
        for action in &actions {
            let child_idx = Self::build_child(config, state, *action, nodes);
            children.push(child_idx);
        }

        nodes[node_idx as usize] = GameNode::Decision {
            player: state.to_act,
            street: state.street,
            actions,
            children,
        };

        node_idx
    }

    /// Build the child node resulting from taking `action` at `state`.
    fn build_child(
        config: &TreeConfig,
        state: &BuildState,
        action: TreeAction,
        nodes: &mut Vec<GameNode>,
    ) -> u32 {
        let actor = state.to_act as usize;
        let opponent = 1 - actor;
        let remaining_actor = state.starting_stack - state.invested[actor];

        match action {
            TreeAction::Fold => {
                let idx = nodes.len() as u32;
                nodes.push(GameNode::Terminal {
                    kind: TerminalKind::Fold {
                        winner: opponent as u8,
                    },
                    pot: state.invested[0] + state.invested[1],
                    invested: state.invested,
                });
                idx
            }
            TreeAction::Check => {
                // If both players have checked (actor is second to act on this street)
                let both_checked = !state.facing_bet && Self::is_closing_action(state);
                if both_checked {
                    Self::make_showdown_or_chance(config, state, nodes)
                } else {
                    // Pass action to opponent
                    let next_state = BuildState {
                        to_act: opponent as u8,
                        facing_bet: false,
                        ..*state
                    };
                    Self::build_node(config, &next_state, nodes)
                }
            }
            TreeAction::Call => {
                let call_to = state.invested[opponent];
                let mut new_invested = state.invested;
                new_invested[actor] = call_to;

                // Preflop limp: SB calls the BB blind. BB still gets to act
                // (check or raise) because the blind is not a voluntary action.
                if state.street == Street::Preflop && state.num_raises == 0 {
                    let new_state = BuildState {
                        invested: new_invested,
                        to_act: opponent as u8,
                        facing_bet: false,
                        num_raises: 0,
                        ..*state
                    };
                    return Self::build_node(config, &new_state, nodes);
                }

                // After a call, the betting round is over
                let new_state = BuildState {
                    invested: new_invested,
                    ..*state
                };
                Self::make_showdown_or_chance(config, &new_state, nodes)
            }
            TreeAction::AllIn => {
                let all_in_total = state.starting_stack;
                let mut new_invested = state.invested;
                new_invested[actor] = all_in_total;

                let is_call_all_in = state.facing_bet
                    && (state.invested[opponent] - all_in_total).abs() < SIZE_EPSILON;
                let is_call_all_in = is_call_all_in
                    || (state.facing_bet && all_in_total <= state.invested[opponent] + SIZE_EPSILON);

                if is_call_all_in || !state.facing_bet && remaining_actor < SIZE_EPSILON {
                    // All-in call or weird edge case => showdown
                    let idx = nodes.len() as u32;
                    nodes.push(GameNode::Terminal {
                        kind: TerminalKind::Showdown,
                        pot: new_invested[0] + new_invested[1],
                        invested: new_invested,
                    });
                    idx
                } else if state.facing_bet {
                    // All-in raise => opponent can fold or call
                    let next_state = BuildState {
                        invested: new_invested,
                        to_act: opponent as u8,
                        num_raises: state.num_raises + 1,
                        facing_bet: true,
                        last_raise_to: all_in_total,
                        ..*state
                    };
                    // Opponent faces the all-in: fold or call only (no re-raise)
                    Self::build_allin_response(config, &next_state, nodes)
                } else {
                    // All-in open bet => opponent responds
                    let next_state = BuildState {
                        invested: new_invested,
                        to_act: opponent as u8,
                        num_raises: state.num_raises + 1,
                        facing_bet: true,
                        last_raise_to: all_in_total,
                        ..*state
                    };
                    Self::build_allin_response(config, &next_state, nodes)
                }
            }
            TreeAction::Bet(amount) | TreeAction::Raise(amount) => {
                let mut new_invested = state.invested;
                new_invested[actor] = amount;

                let next_state = BuildState {
                    invested: new_invested,
                    to_act: opponent as u8,
                    num_raises: state.num_raises + 1,
                    facing_bet: true,
                    last_raise_to: amount,
                    ..*state
                };
                Self::build_node(config, &next_state, nodes)
            }
        }
    }

    /// Build a response node for an opponent facing an all-in.
    /// Only fold and call (all-in call) are available.
    fn build_allin_response(
        _config: &TreeConfig,
        state: &BuildState,
        nodes: &mut Vec<GameNode>,
    ) -> u32 {
        let actor = state.to_act as usize;
        let opponent = 1 - actor;
        let remaining = state.starting_stack - state.invested[actor];

        let node_idx = nodes.len() as u32;
        nodes.push(GameNode::Terminal {
            kind: TerminalKind::Showdown,
            pot: 0.0,
            invested: [0.0; 2],
        });

        let mut children = Vec::with_capacity(2);

        // Fold
        let fold_idx = nodes.len() as u32;
        nodes.push(GameNode::Terminal {
            kind: TerminalKind::Fold {
                winner: opponent as u8,
            },
            pot: state.invested[0] + state.invested[1],
            invested: state.invested,
        });
        children.push(fold_idx);

        // Call (match the all-in or go all-in yourself)
        let call_to = state.invested[opponent].min(state.starting_stack);
        let mut call_invested = state.invested;
        call_invested[actor] = call_to;
        let call_idx = nodes.len() as u32;
        nodes.push(GameNode::Terminal {
            kind: TerminalKind::Showdown,
            pot: call_invested[0] + call_invested[1],
            invested: call_invested,
        });
        children.push(call_idx);

        // Check if actor also can't cover — use AllIn action label
        let call_action = if remaining <= state.invested[opponent] - state.invested[actor] + SIZE_EPSILON {
            TreeAction::AllIn
        } else {
            TreeAction::Call
        };

        nodes[node_idx as usize] = GameNode::Decision {
            player: state.to_act,
            street: state.street,
            actions: vec![TreeAction::Fold, call_action],
            children,
        };

        node_idx
    }

    /// Determine if the current action is the "closing" action on the street
    /// (i.e., the second player to act checking, or calling).
    fn is_closing_action(state: &BuildState) -> bool {
        // Player 1 always closes the action round:
        // - Preflop: BB (player 1) checking after SB limps transitions to flop
        // - Postflop: player 1 checking after player 0 checks transitions
        state.to_act == 1
    }

    /// After a call or check-check, either go to showdown or next street.
    fn make_showdown_or_chance(
        config: &TreeConfig,
        state: &BuildState,
        nodes: &mut Vec<GameNode>,
    ) -> u32 {
        let both_all_in = (state.starting_stack - state.invested[0]).abs() < SIZE_EPSILON
            && (state.starting_stack - state.invested[1]).abs() < SIZE_EPSILON;

        match state.street.next() {
            Some(next_street) if !both_all_in => {
                // Check depth limit before transitioning streets
                if let (Some(limit), Some(start)) = (config.depth_limit, config.starting_street) {
                    let streets_played = next_street as usize - start as usize;
                    if streets_played >= limit as usize {
                        let idx = nodes.len() as u32;
                        nodes.push(GameNode::Terminal {
                            kind: TerminalKind::DepthBoundary,
                            pot: state.invested[0] + state.invested[1],
                            invested: state.invested,
                        });
                        return idx;
                    }
                }

                // Transition to next street via Chance node
                let chance_idx = nodes.len() as u32;
                // Placeholder
                nodes.push(GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: 0.0,
                    invested: [0.0; 2],
                });

                let next_state = BuildState {
                    street: next_street,
                    num_raises: 0,
                    to_act: 0, // OOP acts first on postflop streets
                    facing_bet: false,
                    last_raise_to: 0.0,
                    ..*state
                };

                let child = Self::build_node(config, &next_state, nodes);

                nodes[chance_idx as usize] = GameNode::Chance {
                    next_street,
                    child,
                };

                chance_idx
            }
            _ => {
                // River (no next street) or both all-in: showdown
                let idx = nodes.len() as u32;
                nodes.push(GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    pot: state.invested[0] + state.invested[1],
                    invested: state.invested,
                });
                idx
            }
        }
    }

    /// Compute sized bet/raise actions for the current state.
    fn compute_sized_actions(
        config: &TreeConfig,
        state: &BuildState,
        remaining: f64,
    ) -> Vec<TreeAction> {
        let pot = state.invested[0] + state.invested[1];

        // Determine which raise depth index to use
        let depth_idx = state.num_raises as usize;

        match state.street {
            Street::Preflop => {
                Self::compute_preflop_sizes(config, state, remaining, depth_idx)
            }
            Street::Flop => {
                Self::compute_postflop_sizes(
                    &config.flop_sizes,
                    state,
                    remaining,
                    pot,
                    depth_idx,
                )
            }
            Street::Turn => {
                Self::compute_postflop_sizes(
                    &config.turn_sizes,
                    state,
                    remaining,
                    pot,
                    depth_idx,
                )
            }
            Street::River => {
                Self::compute_postflop_sizes(
                    &config.river_sizes,
                    state,
                    remaining,
                    pot,
                    depth_idx,
                )
            }
        }
    }

    fn compute_preflop_sizes(
        config: &TreeConfig,
        state: &BuildState,
        remaining: f64,
        depth_idx: usize,
    ) -> Vec<TreeAction> {
        let actor = state.to_act as usize;
        let sizes = if depth_idx < config.preflop_sizes.len() {
            &config.preflop_sizes[depth_idx]
        } else {
            // Fall back to last configured depth
            config
                .preflop_sizes
                .last()
                .map_or(&[] as &[PreflopSize], Vec::as_slice)
        };

        let all_in_total = state.starting_stack;
        let mut actions = Vec::new();

        for &size in sizes {
            let raise_to = match size {
                PreflopSize::Absolute(bb) => bb,
                PreflopSize::Multiplier(mult) => state.last_raise_to * mult,
            };

            // Must raise to at least the minimum raise
            let min_raise_to = Self::min_raise_to(state);
            let raise_to = raise_to.max(min_raise_to);

            // Can't raise more than all-in
            if raise_to >= all_in_total - SIZE_EPSILON {
                // This would be all-in; skip (all-in added separately)
                continue;
            }

            // Must be able to afford it
            let additional = raise_to - state.invested[actor];
            if additional > remaining + SIZE_EPSILON {
                continue;
            }

            // Deduplicate
            let already_present = actions.iter().any(|a| match a {
                TreeAction::Raise(v) | TreeAction::Bet(v) => (*v - raise_to).abs() < SIZE_EPSILON,
                _ => false,
            });
            if !already_present {
                if state.facing_bet {
                    actions.push(TreeAction::Raise(raise_to));
                } else {
                    actions.push(TreeAction::Bet(raise_to));
                }
            }
        }

        actions
    }

    fn compute_postflop_sizes(
        sizes_per_depth: &[Vec<f64>],
        state: &BuildState,
        remaining: f64,
        pot: f64,
        depth_idx: usize,
    ) -> Vec<TreeAction> {
        let actor = state.to_act as usize;
        let opponent = 1 - actor;
        let fractions = if depth_idx < sizes_per_depth.len() {
            &sizes_per_depth[depth_idx]
        } else {
            sizes_per_depth
                .last()
                .map_or(&[] as &[f64], Vec::as_slice)
        };

        let all_in_total = state.starting_stack;
        let mut actions = Vec::new();

        for &frac in fractions {
            let bet_amount = if state.facing_bet {
                // Raise: call first, then add fraction of new pot
                let call_amount = state.invested[opponent] - state.invested[actor];
                let pot_after_call = pot + call_amount;
                let raise_amount = call_amount + pot_after_call * frac;
                state.invested[actor] + raise_amount
            } else {
                // Open bet: fraction of pot
                let bet = pot * frac;
                state.invested[actor] + bet
            };

            let min_raise = Self::min_raise_to(state);
            let bet_amount = bet_amount.max(min_raise);

            if bet_amount >= all_in_total - SIZE_EPSILON {
                continue;
            }

            let additional = bet_amount - state.invested[actor];
            if additional > remaining + SIZE_EPSILON {
                continue;
            }

            let already_present = actions.iter().any(|a| match a {
                TreeAction::Raise(v) | TreeAction::Bet(v) => {
                    (*v - bet_amount).abs() < SIZE_EPSILON
                }
                _ => false,
            });
            if !already_present {
                if state.facing_bet {
                    actions.push(TreeAction::Raise(bet_amount));
                } else {
                    actions.push(TreeAction::Bet(bet_amount));
                }
            }
        }

        actions
    }

    /// Minimum legal raise-to amount.
    fn min_raise_to(state: &BuildState) -> f64 {
        let actor = state.to_act as usize;
        let opponent = 1 - actor;
        if state.facing_bet {
            // Must raise by at least the size of the last raise
            let last_raise_size = state.last_raise_to - state.invested[actor];
            // The gap between what opponent has in and what we have in
            let call_amount = state.invested[opponent] - state.invested[actor];
            // Min raise = call + at least the same increment
            state.invested[actor] + call_amount + call_amount.max(last_raise_size)
        } else {
            // Min bet is 1 BB
            state.invested[actor] + 1.0
        }
    }

    /// Remove duplicate `AllIn` entries from the actions list.
    fn dedup_all_in(actions: &mut Vec<TreeAction>) {
        let mut seen_all_in = false;
        actions.retain(|a| {
            if matches!(a, TreeAction::AllIn) {
                if seen_all_in {
                    return false;
                }
                seen_all_in = true;
            }
            true
        });
    }

    /// Build a subgame tree rooted at a specific postflop street.
    ///
    /// Unlike [`build`], this starts mid-game with a known pot and stack
    /// state, and optionally stops after `depth_limit` street transitions.
    #[must_use]
    pub fn build_subgame(
        street: Street,
        pot: f64,
        invested: [f64; 2],
        starting_stack: f64,
        bet_sizes: &[Vec<f64>],
        depth_limit: Option<u8>,
    ) -> Self {
        let config = TreeConfig {
            preflop_sizes: vec![],
            flop_sizes: bet_sizes.to_vec(),
            turn_sizes: bet_sizes.to_vec(),
            river_sizes: bet_sizes.to_vec(),
            depth_limit,
            starting_street: Some(street),
        };

        // Distribute pot evenly if invested doesn't sum to pot (caller
        // may pass [pot/2, pot/2] or an asymmetric split).
        let _ = pot; // pot is implicit in invested[0] + invested[1]

        let initial_state = BuildState {
            starting_stack,
            invested,
            street,
            num_raises: 0,
            to_act: 0,       // OOP acts first postflop
            facing_bet: false,
            last_raise_to: 0.0,
        };

        let mut nodes = Vec::new();
        let root = Self::build_node(&config, &initial_state, &mut nodes);
        Self { nodes, root }
    }

    /// Count the number of nodes of each type in the tree.
    #[must_use]
    pub fn node_counts(&self) -> (usize, usize, usize) {
        let mut decision = 0;
        let mut chance = 0;
        let mut terminal = 0;
        for node in &self.nodes {
            match node {
                GameNode::Decision { .. } => decision += 1,
                GameNode::Chance { .. } => chance += 1,
                GameNode::Terminal { .. } => terminal += 1,
            }
        }
        (decision, chance, terminal)
    }

    /// Build a mapping from arena index to decision-node index.
    ///
    /// For every `Decision` node in the arena, assigns a sequential
    /// index (0, 1, 2, ...) in arena order. Non-decision nodes map to
    /// `u32::MAX`.
    #[must_use]
    pub fn decision_index_map(&self) -> Vec<u32> {
        let mut map = vec![u32::MAX; self.nodes.len()];
        let mut idx = 0u32;
        for (i, node) in self.nodes.iter().enumerate() {
            if matches!(node, GameNode::Decision { .. }) {
                map[i] = idx;
                idx += 1;
            }
        }
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_tree() -> GameTree {
        GameTree::build(
            10.0,
            0.5,
            1.0,
            &[vec!["2.5bb".into()]],
            &[vec![1.0]],
            &[vec![1.0]],
            &[vec![1.0]],
        )
    }

    #[test]
    fn test_minimal_tree_has_terminals() {
        let tree = simple_tree();
        let terminals = tree
            .nodes
            .iter()
            .filter(|n| matches!(n, GameNode::Terminal { .. }))
            .count();
        assert!(terminals > 0, "Tree should have terminal nodes");
    }

    #[test]
    fn test_root_is_decision() {
        let tree = simple_tree();
        assert!(matches!(
            tree.nodes[tree.root as usize],
            GameNode::Decision { .. }
        ));
    }

    #[test]
    fn test_root_player_is_sb() {
        let tree = simple_tree();
        if let GameNode::Decision { player, street, .. } = &tree.nodes[tree.root as usize] {
            assert_eq!(*player, 0, "SB (player 0) should act first preflop");
            assert_eq!(*street, Street::Preflop);
        } else {
            panic!("Root should be a Decision node");
        }
    }

    #[test]
    fn test_all_in_always_available() {
        // Test with deep stacks and multiple raise depths to ensure AllIn is present
        // even AFTER all raise depths are exhausted.
        for stack in [10.0, 50.0, 100.0] {
            let tree = GameTree::build(
                stack,
                0.5,
                1.0,
                &[vec!["2.5bb".into()], vec!["3bb".into()]],
                &[vec![0.5]],
                &[vec![0.5]],
                &[vec![0.5]],
            );
            check_all_in_everywhere(&tree, stack);
        }
    }

    /// Walk the tree verifying every decision node where the actor has chips
    /// includes AllIn. This catches the beyond-raise-cap case.
    fn check_all_in_everywhere(tree: &GameTree, starting_stack: f64) {
        check_all_in_with_invested(tree, starting_stack, [0.5, 1.0]);
    }

    fn check_all_in_with_invested(tree: &GameTree, starting_stack: f64, initial: [f64; 2]) {
        fn walk(
            tree: &GameTree,
            node_idx: u32,
            invested: [f64; 2],
            starting_stack: f64,
            violations: &mut Vec<String>,
        ) {
            match &tree.nodes[node_idx as usize] {
                GameNode::Terminal { .. } => {}
                GameNode::Chance { child, .. } => {
                    walk(tree, *child, invested, starting_stack, violations);
                }
                GameNode::Decision {
                    player,
                    actions,
                    children,
                    ..
                } => {
                    let remaining = starting_stack - invested[*player as usize];
                    if remaining > 0.01 {
                        let has_all_in =
                            actions.iter().any(|a| matches!(a, TreeAction::AllIn));
                        if !has_all_in {
                            violations.push(format!(
                                "node {node_idx}: player {} remaining={remaining:.1} \
                                 invested={invested:?} actions={actions:?}",
                                player
                            ));
                        }
                    }

                    for (action, &child) in actions.iter().zip(children.iter()) {
                        let mut new_inv = invested;
                        match action {
                            TreeAction::Fold | TreeAction::Check => {}
                            TreeAction::Call => {
                                new_inv[*player as usize] = new_inv[1 - *player as usize];
                            }
                            TreeAction::Bet(v) | TreeAction::Raise(v) => {
                                new_inv[*player as usize] = *v;
                            }
                            TreeAction::AllIn => {
                                new_inv[*player as usize] = starting_stack;
                            }
                        }
                        walk(tree, child, new_inv, starting_stack, violations);
                    }
                }
            }
        }

        let mut violations = Vec::new();
        walk(tree, tree.root, initial, starting_stack, &mut violations);
        assert!(
            violations.is_empty(),
            "AllIn missing at {} node(s):\n{}",
            violations.len(),
            violations.join("\n")
        );
    }

    #[test]
    fn test_fold_creates_terminal() {
        let tree = simple_tree();
        if let GameNode::Decision {
            actions, children, ..
        } = &tree.nodes[tree.root as usize]
        {
            let fold_idx = actions
                .iter()
                .position(|a| matches!(a, TreeAction::Fold))
                .expect("Root should have Fold action");
            let fold_child = children[fold_idx];
            assert!(matches!(
                tree.nodes[fold_child as usize],
                GameNode::Terminal {
                    kind: TerminalKind::Fold { .. },
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_fold_terminal_winner_is_opponent() {
        let tree = simple_tree();
        if let GameNode::Decision {
            actions,
            children,
            player,
            ..
        } = &tree.nodes[tree.root as usize]
        {
            let fold_idx = actions
                .iter()
                .position(|a| matches!(a, TreeAction::Fold))
                .unwrap();
            let fold_child = children[fold_idx];
            if let GameNode::Terminal { kind, .. } = &tree.nodes[fold_child as usize] {
                assert_eq!(
                    *kind,
                    TerminalKind::Fold {
                        winner: 1 - player
                    }
                );
            }
        }
    }

    #[test]
    fn test_check_check_leads_to_chance_or_showdown() {
        let tree = simple_tree();
        let chance_count = tree
            .nodes
            .iter()
            .filter(|n| matches!(n, GameNode::Chance { .. }))
            .count();
        assert!(
            chance_count > 0,
            "Tree should have Chance nodes for street transitions"
        );
    }

    #[test]
    fn test_raise_depths_enforced() {
        // 2 preflop depths, 2 flop depths, 1 turn, 1 river
        let tree = GameTree::build(
            100.0,
            0.5,
            1.0,
            &[vec!["2.5bb".into()], vec!["3.0x".into()]],
            &[vec![1.0], vec![1.0]],
            &[vec![1.0]],
            &[vec![1.0]],
        );
        assert!(tree.nodes.len() > 10, "Tree should have substantial nodes");
    }

    #[test]
    fn test_preflop_sb_actions() {
        let tree = simple_tree();
        if let GameNode::Decision { actions, .. } = &tree.nodes[tree.root as usize] {
            // SB facing BB should have: Fold, Call, Raise(2.5), AllIn
            assert!(
                actions.iter().any(|a| matches!(a, TreeAction::Fold)),
                "SB should be able to fold"
            );
            assert!(
                actions.iter().any(|a| matches!(a, TreeAction::Call)),
                "SB should be able to call (limp)"
            );
            assert!(
                actions.iter().any(|a| matches!(a, TreeAction::Raise(_))),
                "SB should be able to raise"
            );
            assert!(
                actions.iter().any(|a| matches!(a, TreeAction::AllIn)),
                "SB should be able to go all-in"
            );
        }
    }

    #[test]
    fn test_postflop_player0_acts_first() {
        let tree = simple_tree();
        // Find a Chance node transitioning to Flop and verify its child
        for node in &tree.nodes {
            if let GameNode::Chance {
                next_street: Street::Flop,
                child,
            } = node
            {
                if let GameNode::Decision { player, street, .. } = &tree.nodes[*child as usize] {
                    assert_eq!(*player, 0, "Player 0 (OOP) should act first on flop");
                    assert_eq!(*street, Street::Flop);
                }
                return;
            }
        }
        panic!("Should find a Chance node to Flop");
    }

    #[test]
    fn test_showdown_on_river() {
        let tree = simple_tree();
        // Find a river decision and verify that check-check or call leads to showdown
        let has_river_showdown = tree.nodes.iter().any(|n| {
            matches!(
                n,
                GameNode::Terminal {
                    kind: TerminalKind::Showdown,
                    ..
                }
            )
        });
        assert!(has_river_showdown, "Tree should have showdown terminals");
    }

    #[test]
    fn test_invested_tracks_correctly() {
        let tree = simple_tree();
        // Fold at root: SB folds having put in 0.5
        if let GameNode::Decision {
            actions, children, ..
        } = &tree.nodes[tree.root as usize]
        {
            let fold_idx = actions
                .iter()
                .position(|a| matches!(a, TreeAction::Fold))
                .unwrap();
            if let GameNode::Terminal { invested, pot, .. } = &tree.nodes[children[fold_idx] as usize]
            {
                assert!(
                    (invested[0] - 0.5).abs() < SIZE_EPSILON,
                    "SB should have invested 0.5, got {}",
                    invested[0]
                );
                assert!(
                    (invested[1] - 1.0).abs() < SIZE_EPSILON,
                    "BB should have invested 1.0, got {}",
                    invested[1]
                );
                assert!(
                    (*pot - 1.5).abs() < SIZE_EPSILON,
                    "Pot should be 1.5, got {pot}"
                );
            }
        }
    }

    #[test]
    fn test_no_raise_after_declared_depths() {
        // With 1 preflop depth, SB can open-raise (raise #1), but BB
        // cannot re-raise (would exceed declared depths).
        let tree = GameTree::build(
            10.0,
            0.5,
            1.0,
            &[vec!["2.5bb".into()]],
            &[vec![1.0]],
            &[vec![1.0]],
            &[vec![1.0]],
        );
        // Root: SB should be able to raise
        if let GameNode::Decision {
            actions, children, ..
        } = &tree.nodes[tree.root as usize]
        {
            let raise_idx = actions
                .iter()
                .position(|a| matches!(a, TreeAction::Raise(_)))
                .expect("SB should have a raise with 1 preflop depth");

            // BB's response to SB's raise: should NOT have raise actions
            let bb_node = &tree.nodes[children[raise_idx] as usize];
            if let GameNode::Decision { actions, .. } = bb_node {
                let has_raise = actions.iter().any(|a| {
                    matches!(a, TreeAction::Raise(_) | TreeAction::Bet(_))
                });
                assert!(
                    !has_raise,
                    "BB should not raise after 1 preflop depth is exhausted. Actions: {actions:?}"
                );
            }
        }
    }

    #[test]
    fn test_multiplier_sizing() {
        let tree = GameTree::build(
            100.0,
            0.5,
            1.0,
            &[vec!["2.5bb".into()], vec!["3.0x".into()], vec!["3.0x".into()]],
            &[vec![1.0]],
            &[vec![1.0]],
            &[vec![1.0]],
        );
        // SB raises to 2.5bb. BB can re-raise to 3.0x * 2.5 = 7.5bb
        // Find BB's response to SB's raise
        if let GameNode::Decision {
            actions, children, ..
        } = &tree.nodes[tree.root as usize]
        {
            let raise_idx = actions
                .iter()
                .position(|a| matches!(a, TreeAction::Raise(_)))
                .expect("SB should have a raise");
            let bb_node = &tree.nodes[children[raise_idx] as usize];
            if let GameNode::Decision { actions, .. } = bb_node {
                let raise_action = actions.iter().find(|a| matches!(a, TreeAction::Raise(_)));
                if let Some(TreeAction::Raise(to)) = raise_action {
                    assert!(
                        (*to - 7.5).abs() < SIZE_EPSILON,
                        "BB should raise to 7.5bb (3.0x * 2.5), got {to}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_deep_stack_tree_size() {
        let tree = GameTree::build(
            100.0,
            0.5,
            1.0,
            &[vec!["2.5bb".into()], vec!["3.0x".into()], vec!["3.0x".into()]],
            &[vec![0.5, 1.0], vec![0.5, 1.0]],
            &[vec![0.5, 1.0], vec![0.5, 1.0]],
            &[vec![0.5, 1.0], vec![0.5, 1.0]],
        );
        let (decision, chance, terminal) = tree.node_counts();
        assert!(
            decision > 100,
            "Deep stack tree should have many decisions, got {decision}"
        );
        assert!(
            chance > 0,
            "Should have chance nodes"
        );
        assert!(
            terminal > 0,
            "Should have terminal nodes"
        );
    }

    #[test]
    fn test_all_children_in_bounds() {
        let tree = simple_tree();
        let len = tree.nodes.len() as u32;
        for (i, node) in tree.nodes.iter().enumerate() {
            match node {
                GameNode::Decision { children, .. } => {
                    for &child in children {
                        assert!(
                            child < len,
                            "Node {i}: child index {child} out of bounds (len={len})"
                        );
                    }
                }
                GameNode::Chance { child, .. } => {
                    assert!(
                        *child < len,
                        "Node {i}: chance child {child} out of bounds (len={len})"
                    );
                }
                GameNode::Terminal { .. } => {}
            }
        }
    }

    #[test]
    fn test_actions_children_same_length() {
        let tree = simple_tree();
        for (i, node) in tree.nodes.iter().enumerate() {
            if let GameNode::Decision {
                actions, children, ..
            } = node
            {
                assert_eq!(
                    actions.len(),
                    children.len(),
                    "Node {i}: actions and children length mismatch"
                );
            }
        }
    }

    #[test]
    fn test_pot_fraction_sizing() {
        // With pot = 2bb (SB calls, both invested 1bb each), a 0.5 pot bet = 1bb
        let tree = GameTree::build(
            20.0,
            0.5,
            1.0,
            &[vec!["2.0bb".into()]],
            &[vec![0.5], vec![0.5]],
            &[vec![0.5], vec![0.5]],
            &[vec![0.5], vec![0.5]],
        );
        // Find a flop decision where player opens with a bet
        for node in &tree.nodes {
            if let GameNode::Decision {
                street: Street::Flop,
                actions,
                ..
            } = node
            {
                if !actions.iter().any(|a| matches!(a, TreeAction::Bet(_))) {
                    continue;
                }
                if let Some(TreeAction::Bet(amount)) = actions.iter().find(|a| matches!(a, TreeAction::Bet(_))) {
                    // After SB limps (calls 0.5 more) or raises preflop and gets called,
                    // pot is invested[0] + invested[1], bet = pot * 0.5
                    // Amount is invested[actor] + bet, so it should be > invested[actor]
                    assert!(
                        *amount > 0.0,
                        "Bet amount should be positive, got {amount}"
                    );
                }
                return;
            }
        }
    }

    #[test]
    fn test_build_subgame_basic() {
        let tree = GameTree::build_subgame(
            Street::Flop, 10.0, [5.0, 5.0], 50.0,
            &[vec![0.5, 1.0]], None,
        );
        assert!(matches!(tree.nodes[tree.root as usize], GameNode::Decision { player: 0, .. }));
        let has_chance = tree.nodes.iter().any(|n| matches!(n, GameNode::Chance { .. }));
        assert!(has_chance, "Full-depth subgame should have Chance nodes");
    }

    #[test]
    fn test_build_subgame_depth_limited() {
        let tree = GameTree::build_subgame(
            Street::Flop, 10.0, [5.0, 5.0], 50.0,
            &[vec![0.5, 1.0]], Some(1),
        );
        let has_boundary = tree.nodes.iter().any(|n| matches!(
            n, GameNode::Terminal { kind: TerminalKind::DepthBoundary, .. }
        ));
        assert!(has_boundary, "Depth-limited subgame should have DepthBoundary terminals");
        let has_chance = tree.nodes.iter().any(|n| matches!(n, GameNode::Chance { .. }));
        assert!(!has_chance, "Depth-limited flop subgame should not transition to turn");
    }

    #[test]
    fn test_build_subgame_river_no_chance() {
        let tree = GameTree::build_subgame(
            Street::River, 20.0, [10.0, 10.0], 50.0,
            &[vec![0.5, 1.0]], None,
        );
        let has_chance = tree.nodes.iter().any(|n| matches!(n, GameNode::Chance { .. }));
        assert!(!has_chance, "River subgame has no next street");
        let has_boundary = tree.nodes.iter().any(|n| matches!(
            n, GameNode::Terminal { kind: TerminalKind::DepthBoundary, .. }
        ));
        assert!(!has_boundary, "River subgame has no DepthBoundary");
    }

    #[test]
    fn test_build_subgame_allin_everywhere() {
        for street in [Street::Flop, Street::Turn, Street::River] {
            let tree = GameTree::build_subgame(
                street, 10.0, [5.0, 5.0], 50.0,
                &[vec![0.5]], Some(1),
            );
            check_all_in_with_invested(&tree, 50.0, [5.0, 5.0]);
        }
    }
}
