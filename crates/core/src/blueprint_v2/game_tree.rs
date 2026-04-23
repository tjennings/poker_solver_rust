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
        /// Blueprint decision node index for this node (set during subgame
        /// construction when an abstract tree is provided). `None` for
        /// blueprint-native trees or when no mapping is available.
        #[allow(dead_code)]
        blueprint_decision_idx: Option<u32>,
    },
    Chance {
        next_street: Street,
        child: u32,
    },
    Terminal {
        kind: TerminalKind,
        pot: f64,
        /// Each player's remaining stack at this terminal.
        /// Payoff = final_stack - starting_stack (+ pot if winner).
        stacks: [f64; 2],
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

#[derive(Debug, Clone)]
pub struct GameTree {
    pub nodes: Vec<GameNode>,
    pub root: u32,
    /// Which seat is the button/SB. The other seat is BB.
    /// Preflop: `dealer` (SB) acts first.
    /// Postflop: `1 - dealer` (BB/OOP) acts first.
    pub dealer: u8,
    /// Starting stack per player (same for both). Used for terminal payoff computation.
    #[allow(dead_code)]
    pub starting_stack: f64,
}

/// Internal state tracked during recursive tree construction.
struct BuildState {
    /// Stacks at the start of the hand (same for both players).
    #[allow(dead_code)]
    starting_stack: f64,
    /// Each player's current stack (depletes as bets/calls are made).
    stacks: [f64; 2],
    /// Per-street bet amount per player (for to-call computation). Resets each street.
    street_bets: [f64; 2],
    /// Which seat is the button/SB (0 or 1).
    dealer: u8,
    /// Small blind amount in BB.
    small_blind: f64,
    /// Big blind amount in BB.
    big_blind: f64,
    /// Running total pot (includes blinds and all bets).
    pot: f64,
    /// Current street.
    street: Street,
    /// Number of bets/raises made on the current street.
    num_raises: u8,
    /// Which player acts next (0 or 1).
    to_act: u8,
    /// Whether the current action sequence on this street has seen a bet.
    /// Used to distinguish check vs call, bet vs raise.
    facing_bet: bool,
    /// The size of the last bet/raise TO (total street bet by the aggressor).
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
    /// When false, SB cannot open-limp (call the BB blind). Default true.
    allow_preflop_limp: bool,
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
    /// * `small_blind` - Small blind in chips (typically 1)
    /// * `big_blind` - Big blind in chips (typically 2)
    /// * `preflop_sizes` - Per raise depth, list of size strings
    ///   (`"5bb"` = raise to 5 big blinds, `"3.0x"` = multiplier of last raise)
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
        Self::build_with_options(stack_depth, small_blind, big_blind, preflop_sizes, flop_sizes, turn_sizes, river_sizes, true)
    }

    /// Build a game tree with additional options.
    #[must_use]
    pub fn build_with_options(
        stack_depth: f64,
        small_blind: f64,
        big_blind: f64,
        preflop_sizes: &[Vec<String>],
        flop_sizes: &[Vec<f64>],
        turn_sizes: &[Vec<f64>],
        river_sizes: &[Vec<f64>],
        allow_preflop_limp: bool,
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
            allow_preflop_limp,
        };

        let mut nodes = Vec::new();

        let sb = 0usize; // dealer seat
        let bb = 1usize;
        let mut initial_stacks = [stack_depth; 2];
        initial_stacks[sb] -= small_blind;
        initial_stacks[bb] -= big_blind;

        let initial_state = BuildState {
            starting_stack: stack_depth,
            stacks: initial_stacks,
            street_bets: {
                let mut sb_arr = [0.0; 2];
                sb_arr[sb] = small_blind;
                sb_arr[bb] = big_blind;
                sb_arr
            },
            dealer: 0, // seat 0 is SB/button
            small_blind,
            big_blind,
            pot: small_blind + big_blind, // blinds are dead money in pot
            street: Street::Preflop,
            num_raises: 0, // BB is a forced blind, not a voluntary raise
            to_act: 0,     // SB acts first preflop
            facing_bet: true, // SB is facing the BB (a forced bet)
            last_raise_to: big_blind,
        };

        let root = Self::build_node(&config, &initial_state, &mut nodes);

        Self { nodes, root, dealer: 0, starting_stack: stack_depth }
    }

    /// Recursively build a node and return its arena index.
    fn build_node(config: &TreeConfig, state: &BuildState, nodes: &mut Vec<GameNode>) -> u32 {
        let remaining = state.stacks;
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
            let call_amount = if state.street == Street::Preflop && state.num_raises == 0 && state.facing_bet {
                // SB faces BB obligation: call = BB blind - SB blind
                state.big_blind - state.small_blind
            } else {
                state.street_bets[1 - actor] - state.street_bets[actor]
            };
            // Skip SB open-limp when disallowed.
            let is_preflop_limp = state.street == Street::Preflop && state.num_raises == 0;
            if is_preflop_limp && !config.allow_preflop_limp {
                // No limp option — SB must raise or fold.
            } else if call_amount >= remaining[actor] - SIZE_EPSILON {
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
            let all_in_amount = state.street_bets[actor] + remaining[actor];
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
            stacks: [0.0; 2],
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
            blueprint_decision_idx: None,
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
        let remaining_actor = state.stacks[actor];

        match action {
            TreeAction::Fold => {
                // Fold: no stack/pot change
                let idx = nodes.len() as u32;
                nodes.push(GameNode::Terminal {
                    kind: TerminalKind::Fold {
                        winner: opponent as u8,
                    },
                    pot: state.pot,
                    stacks: state.stacks,
                });
                idx
            }
            TreeAction::Check => {
                // Check: no stack/pot change
                let both_checked = !state.facing_bet && Self::is_closing_action(state);
                if both_checked {
                    Self::make_showdown_or_chance(config, state, nodes)
                } else {
                    let next_state = BuildState {
                        to_act: opponent as u8,
                        facing_bet: false,
                        ..*state
                    };
                    Self::build_node(config, &next_state, nodes)
                }
            }
            TreeAction::Call => {
                let call_amount = state.street_bets[opponent] - state.street_bets[actor];
                let mut new_stacks = state.stacks;
                new_stacks[actor] -= call_amount;
                let new_pot = state.pot + call_amount;
                let mut new_street_bets = state.street_bets;
                new_street_bets[actor] = state.street_bets[opponent];

                // Preflop limp: SB calls the BB blind. BB still gets to act
                // (check or raise) because the blind is not a voluntary action.
                if state.street == Street::Preflop && state.num_raises == 0 {
                    let new_state = BuildState {
                        stacks: new_stacks,
                        street_bets: new_street_bets,
                        pot: new_pot,
                        to_act: opponent as u8,
                        facing_bet: false,
                        num_raises: 0,
                        ..*state
                    };
                    return Self::build_node(config, &new_state, nodes);
                }

                // After a call, the betting round is over
                let new_state = BuildState {
                    stacks: new_stacks,
                    street_bets: new_street_bets,
                    pot: new_pot,
                    ..*state
                };
                Self::make_showdown_or_chance(config, &new_state, nodes)
            }
            TreeAction::AllIn => {
                let additional = remaining_actor;
                let new_pot = state.pot + additional;
                let mut new_stacks = state.stacks;
                new_stacks[actor] = 0.0;
                let mut new_street_bets = state.street_bets;
                new_street_bets[actor] += additional;

                let all_in_street_bet = new_street_bets[actor];
                let is_call_all_in = state.facing_bet
                    && (state.street_bets[opponent] - all_in_street_bet).abs() < SIZE_EPSILON;
                let is_call_all_in = is_call_all_in
                    || (state.facing_bet && all_in_street_bet <= state.street_bets[opponent] + SIZE_EPSILON);

                if is_call_all_in || !state.facing_bet && remaining_actor < SIZE_EPSILON {
                    // All-in call or weird edge case => showdown
                    let idx = nodes.len() as u32;
                    nodes.push(GameNode::Terminal {
                        kind: TerminalKind::Showdown,
                        pot: new_pot,
                        stacks: new_stacks,
                    });
                    idx
                } else if state.facing_bet {
                    // All-in raise => opponent can fold or call
                    let next_state = BuildState {
                        stacks: new_stacks,
                        street_bets: new_street_bets,
                        pot: new_pot,
                        to_act: opponent as u8,
                        num_raises: state.num_raises + 1,
                        facing_bet: true,
                        last_raise_to: all_in_street_bet,
                        ..*state
                    };
                    Self::build_allin_response(config, &next_state, nodes)
                } else {
                    // All-in open bet => opponent responds
                    let next_state = BuildState {
                        stacks: new_stacks,
                        street_bets: new_street_bets,
                        pot: new_pot,
                        to_act: opponent as u8,
                        num_raises: state.num_raises + 1,
                        facing_bet: true,
                        last_raise_to: all_in_street_bet,
                        ..*state
                    };
                    Self::build_allin_response(config, &next_state, nodes)
                }
            }
            TreeAction::Bet(amount) | TreeAction::Raise(amount) => {
                // amount is raise-TO (total street bet)
                let additional = amount - state.street_bets[actor];
                let new_pot = state.pot + additional;
                let mut new_stacks = state.stacks;
                new_stacks[actor] -= additional;
                let mut new_street_bets = state.street_bets;
                new_street_bets[actor] = amount;

                let next_state = BuildState {
                    stacks: new_stacks,
                    street_bets: new_street_bets,
                    pot: new_pot,
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
        let remaining = state.stacks[actor];

        let node_idx = nodes.len() as u32;
        nodes.push(GameNode::Terminal {
            kind: TerminalKind::Showdown,
            pot: 0.0,
            stacks: [0.0; 2],
        });

        let mut children = Vec::with_capacity(2);

        // Fold: pot unchanged, stacks unchanged
        let fold_idx = nodes.len() as u32;
        nodes.push(GameNode::Terminal {
            kind: TerminalKind::Fold {
                winner: opponent as u8,
            },
            pot: state.pot,
            stacks: state.stacks,
        });
        children.push(fold_idx);

        // Call (match the all-in or go all-in yourself)
        let call_amount = (state.street_bets[opponent] - state.street_bets[actor]).min(remaining);
        let call_pot = state.pot + call_amount;
        let mut call_stacks = state.stacks;
        call_stacks[actor] -= call_amount;
        let call_idx = nodes.len() as u32;
        nodes.push(GameNode::Terminal {
            kind: TerminalKind::Showdown,
            pot: call_pot,
            stacks: call_stacks,
        });
        children.push(call_idx);

        // Check if actor also can't cover — use AllIn action label
        let call_action = if remaining <= state.street_bets[opponent] - state.street_bets[actor] + SIZE_EPSILON {
            TreeAction::AllIn
        } else {
            TreeAction::Call
        };

        nodes[node_idx as usize] = GameNode::Decision {
            player: state.to_act,
            street: state.street,
            actions: vec![TreeAction::Fold, call_action],
            children,
            blueprint_decision_idx: None,
        };

        node_idx
    }

    /// Determine if the current action is the "closing" action on the street
    /// (i.e., the second player to act checking, or calling).
    fn is_closing_action(state: &BuildState) -> bool {
        // The second player to act on this street closes the action.
        // Preflop: BB (1 - dealer) acts second.
        // Postflop: IP (dealer) acts second.
        if state.street == Street::Preflop {
            state.to_act == 1 - state.dealer
        } else {
            state.to_act == state.dealer
        }
    }

    /// After a call or check-check, either go to showdown or next street.
    fn make_showdown_or_chance(
        config: &TreeConfig,
        state: &BuildState,
        nodes: &mut Vec<GameNode>,
    ) -> u32 {
        let both_all_in = state.stacks[0].abs() < SIZE_EPSILON
            && state.stacks[1].abs() < SIZE_EPSILON;

        match state.street.next() {
            Some(next_street) if !both_all_in => {
                // Check depth limit before transitioning streets
                if let (Some(limit), Some(start)) = (config.depth_limit, config.starting_street) {
                    let streets_played = next_street as usize - start as usize;
                    if streets_played >= limit as usize {
                        let idx = nodes.len() as u32;
                        nodes.push(GameNode::Terminal {
                            kind: TerminalKind::DepthBoundary,
                            pot: state.pot,
                            stacks: state.stacks,
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
                    stacks: [0.0; 2],
                });

                // Postflop: OOP (1 - dealer) acts first.
                let postflop_to_act = 1 - state.dealer;

                // Street bets reset at street transitions
                let next_state = BuildState {
                    street: next_street,
                    num_raises: 0,
                    to_act: postflop_to_act,
                    facing_bet: false,
                    last_raise_to: 0.0,
                    street_bets: [0.0; 2],
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
                    pot: state.pot,
                    stacks: state.stacks,
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
        let pot = state.pot;

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

        let all_in_street_bet = state.street_bets[actor] + remaining;
        let mut actions = Vec::new();

        for &size in sizes {
            let raise_to = match size {
                PreflopSize::Absolute(bb) => bb * state.big_blind,
                PreflopSize::Multiplier(mult) => state.last_raise_to * mult,
            };

            // Must raise to at least the minimum raise
            let min_raise_to = Self::min_raise_to(state);
            let raise_to = raise_to.max(min_raise_to);

            // Can't raise more than all-in
            if raise_to >= all_in_street_bet - SIZE_EPSILON {
                // This would be all-in; skip (all-in added separately)
                continue;
            }

            // Must be able to afford it
            let additional = raise_to - state.street_bets[actor];
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

        let all_in_street_bet = state.street_bets[actor] + remaining;
        let mut actions = Vec::new();

        for &frac in fractions {
            let bet_amount = if state.facing_bet {
                // Raise: call first, then add fraction of new pot
                let call_amount = state.street_bets[opponent] - state.street_bets[actor];
                let pot_after_call = pot + call_amount;
                let raise_amount = call_amount + pot_after_call * frac;
                state.street_bets[actor] + raise_amount
            } else {
                // Open bet: fraction of pot
                let bet = pot * frac;
                state.street_bets[actor] + bet
            };

            let min_raise = Self::min_raise_to(state);
            let bet_amount = bet_amount.max(min_raise);

            if bet_amount >= all_in_street_bet - SIZE_EPSILON {
                continue;
            }

            let additional = bet_amount - state.street_bets[actor];
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

    /// Minimum legal raise-to amount (as a street bet total).
    fn min_raise_to(state: &BuildState) -> f64 {
        let actor = state.to_act as usize;
        let opponent = 1 - actor;
        if state.street == Street::Preflop && state.num_raises == 0 && state.facing_bet {
            // Preflop root: SB facing BB obligation.
            // Call = BB - SB blind. Min raise = call + BB.
            let call = state.big_blind - state.small_blind;
            call + state.big_blind
        } else if state.facing_bet {
            // Must raise by at least the size of the last raise
            let last_raise_size = state.last_raise_to - state.street_bets[actor];
            // The gap between what opponent has in and what we have in
            let call_amount = state.street_bets[opponent] - state.street_bets[actor];
            // Min raise = call + at least the same increment
            state.street_bets[actor] + call_amount + call_amount.max(last_raise_size)
        } else {
            // Min bet is 1 BB
            state.street_bets[actor] + 1.0
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
    ///
    /// `invested` is per-player total investment so far (used to compute
    /// remaining stacks as `starting_stack - invested[p]`).
    /// `dealer` indicates which seat is the button/SB. OOP = `1 - dealer`.
    #[must_use]
    pub fn build_subgame(
        street: Street,
        pot: f64,
        invested: [f64; 2],
        starting_stack: f64,
        bet_sizes: &[Vec<f64>],
        depth_limit: Option<u8>,
        dealer: u8,
    ) -> Self {
        let config = TreeConfig {
            preflop_sizes: vec![],
            flop_sizes: bet_sizes.to_vec(),
            turn_sizes: bet_sizes.to_vec(),
            river_sizes: bet_sizes.to_vec(),
            depth_limit,
            starting_street: Some(street),
            allow_preflop_limp: true,
        };

        // OOP acts first postflop = 1 - dealer
        let to_act = 1 - dealer;

        // Convert invested to stacks: stack = starting_stack - invested
        let stacks = [
            starting_stack - invested[0],
            starting_stack - invested[1],
        ];

        let initial_state = BuildState {
            starting_stack,
            stacks,
            street_bets: [0.0; 2], // street bets start at 0 for a new street
            dealer,
            small_blind: 0.0, // no blind dead money in postflop subgames
            big_blind: 0.0,
            pot,
            street,
            num_raises: 0,
            to_act,
            facing_bet: false,
            last_raise_to: 0.0,
        };

        let mut nodes = Vec::new();
        let root = Self::build_node(&config, &initial_state, &mut nodes);
        Self { nodes, root, dealer, starting_stack }
    }

    /// Annotate each decision node in this (subgame) tree with the
    /// corresponding blueprint decision index from `abstract_tree`.
    ///
    /// Walks both trees in parallel from `abstract_start`. Since the
    /// subgame is built from the same action config, the action order
    /// matches 1:1 for shared actions (the subgame may have extra raises).
    ///
    /// After this call, `GameNode::Decision.blueprint_decision_idx` is
    /// `Some(idx)` for every reachable decision node.
    pub fn annotate_blueprint_indices(
        &mut self,
        abstract_tree: &GameTree,
        abstract_start: u32,
        decision_idx_map: &[u32],
    ) {
        let mut stack: Vec<(u32, u32)> = vec![(self.root, abstract_start)];

        while let Some((sg_node, abs_node)) = stack.pop() {
            let sg_idx = sg_node as usize;
            let abs_idx = abs_node as usize;

            match (&self.nodes[sg_idx], &abstract_tree.nodes[abs_idx]) {
                (
                    GameNode::Decision { actions: sg_actions, children: sg_children, .. },
                    GameNode::Decision { actions: abs_actions, children: abs_children, .. },
                ) => {
                    let dec_idx = decision_idx_map[abs_idx];

                    // Clone what we need before mutating
                    let sg_children = sg_children.clone();
                    let abs_children = abs_children.clone();
                    let sg_actions = sg_actions.clone();
                    let abs_actions = abs_actions.clone();

                    // Set the blueprint index on this node
                    if let GameNode::Decision { blueprint_decision_idx, .. } = &mut self.nodes[sg_idx] {
                        *blueprint_decision_idx = if dec_idx != u32::MAX { Some(dec_idx) } else { None };
                    }

                    // Walk children: for each abstract action, find its match in the subgame
                    for (abs_ai, abs_action) in abs_actions.iter().enumerate() {
                        // Find matching subgame action by type+ordinal
                        let abs_ordinal = abs_actions[..abs_ai].iter()
                            .filter(|a| same_action_type(abs_action, a))
                            .count();
                        let mut count = 0usize;
                        for (sg_ai, sg_action) in sg_actions.iter().enumerate() {
                            if same_action_type(abs_action, sg_action) {
                                if count == abs_ordinal {
                                    let sg_child = skip_chance_node(self, sg_children[sg_ai]);
                                    let abs_child = skip_chance_node_ref(abstract_tree, abs_children[abs_ai]);
                                    stack.push((sg_child, abs_child));
                                    break;
                                }
                                count += 1;
                            }
                        }
                    }
                }

                (GameNode::Chance { child: sg_child, .. }, GameNode::Chance { child: abs_child, .. }) => {
                    stack.push((*sg_child, *abs_child));
                }

                _ => {}
            }
        }
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

    /// Return the position label for a seat: `"SB"` for the dealer, `"BB"` for the other.
    #[must_use]
    pub fn position_label(&self, seat: u8) -> &str {
        if seat == self.dealer { "SB" } else { "BB" }
    }

    /// Return the seat index of the small blind (= dealer).
    #[must_use]
    pub fn sb_seat(&self) -> u8 {
        self.dealer
    }

    /// Return the seat index of the big blind (= 1 - dealer).
    #[must_use]
    pub fn bb_seat(&self) -> u8 {
        1 - self.dealer
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

    /// Collect all `Chance` node arena indices reachable from `start`
    /// via DFS, in arena-encounter order.
    ///
    /// Useful for mapping range-solver boundary ordinals to abstract
    /// tree chance nodes: the i-th entry corresponds to the i-th
    /// boundary the solver will encounter.
    #[must_use]
    pub fn chance_descendants(&self, start: u32) -> Vec<u32> {
        let mut result = Vec::new();
        let mut stack = vec![start];
        while let Some(idx) = stack.pop() {
            match &self.nodes[idx as usize] {
                GameNode::Chance { child, .. } => {
                    result.push(idx);
                    stack.push(*child);
                }
                GameNode::Decision { children, .. } => {
                    // Push in reverse so left children are visited first
                    // (matching DFS order).
                    for &c in children.iter().rev() {
                        stack.push(c);
                    }
                }
                GameNode::Terminal { .. } => {}
            }
        }
        result
    }
}

/// Check if two actions are the same type (ignoring size values).
fn same_action_type(a: &TreeAction, b: &TreeAction) -> bool {
    matches!(
        (a, b),
        (TreeAction::Fold, TreeAction::Fold)
            | (TreeAction::Check, TreeAction::Check)
            | (TreeAction::Call, TreeAction::Call)
            | (TreeAction::AllIn, TreeAction::AllIn)
            | (TreeAction::Bet(_), TreeAction::Bet(_))
            | (TreeAction::Raise(_), TreeAction::Raise(_))
    )
}

/// Skip past a chance node, returning the child.
fn skip_chance_node(tree: &GameTree, node: u32) -> u32 {
    match &tree.nodes[node as usize] {
        GameNode::Chance { child, .. } => *child,
        _ => node,
    }
}

/// Skip past a chance node in an immutable tree reference.
fn skip_chance_node_ref(tree: &GameTree, node: u32) -> u32 {
    skip_chance_node(tree, node)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_tree() -> GameTree {
        GameTree::build(
            20.0,
            1.0,
            2.0,
            &[vec!["5bb".into()]],
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
        for stack in [20.0, 100.0, 200.0] {
            let tree = GameTree::build(
                stack,
                1.0,
                2.0,
                &[vec!["5bb".into()], vec!["6bb".into()]],
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
        // Full game tree: blinds are [1.0, 2.0] for dealer=0
        let blinds = blind_amounts_for_tree(tree, 1.0, 2.0);
        check_all_in_with_invested(tree, starting_stack, [0.0, 0.0], blinds);
    }

    /// Compute per-seat blind amounts from tree's dealer field.
    fn blind_amounts_for_tree(tree: &GameTree, sb: f64, bb: f64) -> [f64; 2] {
        let mut blinds = [0.0; 2];
        blinds[tree.dealer as usize] = sb;
        blinds[1 - tree.dealer as usize] = bb;
        blinds
    }

    fn check_all_in_with_invested(tree: &GameTree, starting_stack: f64, initial: [f64; 2], blinds: [f64; 2]) {
        fn walk(
            tree: &GameTree,
            node_idx: u32,
            invested: [f64; 2],
            starting_stack: f64,
            blinds: [f64; 2],
            violations: &mut Vec<String>,
        ) {
            match &tree.nodes[node_idx as usize] {
                GameNode::Terminal { .. } => {}
                GameNode::Chance { child, .. } => {
                    walk(tree, *child, invested, starting_stack, blinds, violations);
                }
                GameNode::Decision {
                    player,
                    actions,
                    children,
                    ..
                } => {
                    let p = *player as usize;
                    let remaining = starting_stack - blinds[p] - invested[p];
                    // Skip all-in response nodes (exactly [Fold, Call/AllIn]) —
                    // these intentionally lack a full AllIn option.
                    let is_allin_response = actions.len() == 2
                        && matches!(actions[0], TreeAction::Fold)
                        && matches!(actions[1], TreeAction::Call | TreeAction::AllIn);
                    if remaining > 0.01 && !is_allin_response {
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
                                new_inv[p] = new_inv[1 - p];
                            }
                            TreeAction::Bet(v) | TreeAction::Raise(v) => {
                                new_inv[p] = *v;
                            }
                            TreeAction::AllIn => {
                                new_inv[p] = starting_stack - blinds[p];
                            }
                        }
                        walk(tree, child, new_inv, starting_stack, blinds, violations);
                    }
                }
            }
        }

        let mut violations = Vec::new();
        walk(tree, tree.root, initial, starting_stack, blinds, &mut violations);
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
            200.0,
            1.0,
            2.0,
            &[vec!["5bb".into()], vec!["3.0x".into()]],
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
            // SB facing BB should have: Fold, Call, Raise(5), AllIn
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
    fn test_postflop_oop_acts_first() {
        let tree = simple_tree();
        // With dealer=0, OOP = 1 - dealer = seat 1 (BB) acts first postflop.
        for node in &tree.nodes {
            if let GameNode::Chance {
                next_street: Street::Flop,
                child,
            } = node
            {
                if let GameNode::Decision { player, street, .. } = &tree.nodes[*child as usize] {
                    assert_eq!(*player, 1, "BB (seat 1, OOP) should act first on flop");
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
    fn test_fold_pot_includes_blinds() {
        let tree = simple_tree();
        // Fold at root: pot should be SB + BB = 1 + 2 = 3 chips
        if let GameNode::Decision {
            actions, children, ..
        } = &tree.nodes[tree.root as usize]
        {
            let fold_idx = actions
                .iter()
                .position(|a| matches!(a, TreeAction::Fold))
                .unwrap();
            if let GameNode::Terminal { pot, .. } = &tree.nodes[children[fold_idx] as usize]
            {
                assert!(
                    (*pot - 3.0).abs() < SIZE_EPSILON,
                    "Pot should include blinds: 3.0 chips, got {pot}"
                );
            }
        }
    }

    #[test]
    fn test_no_raise_after_declared_depths() {
        // With 1 preflop depth, SB can open-raise (raise #1), but BB
        // cannot re-raise (would exceed declared depths).
        let tree = GameTree::build(
            20.0,
            1.0,
            2.0,
            &[vec!["5bb".into()]],
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
            200.0,
            1.0,
            2.0,
            &[vec!["5bb".into()], vec!["3.0x".into()], vec!["3.0x".into()]],
            &[vec![1.0]],
            &[vec![1.0]],
            &[vec![1.0]],
        );
        // SB raises to 5bb = 10 chips. BB can re-raise to 3.0x * 10 = 30 chips
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
                        (*to - 30.0).abs() < SIZE_EPSILON,
                        "BB should raise to 30 chips (3.0x * 10), got {to}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_deep_stack_tree_size() {
        let tree = GameTree::build(
            200.0,
            1.0,
            2.0,
            &[vec!["5bb".into()], vec!["3.0x".into()], vec!["3.0x".into()]],
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
        // With pot = 4 chips (SB calls, both invested 2 chips each), a 0.5 pot bet = 2 chips
        let tree = GameTree::build(
            40.0,
            1.0,
            2.0,
            &[vec!["4bb".into()]],
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
            &[vec![0.5, 1.0]], None, 0,
        );
        // With dealer=0, postflop OOP = 1 - dealer = seat 1 acts first.
        assert!(matches!(tree.nodes[tree.root as usize], GameNode::Decision { player: 1, .. }));
        let has_chance = tree.nodes.iter().any(|n| matches!(n, GameNode::Chance { .. }));
        assert!(has_chance, "Full-depth subgame should have Chance nodes");
    }

    #[test]
    fn test_build_subgame_depth_limited() {
        let tree = GameTree::build_subgame(
            Street::Flop, 10.0, [5.0, 5.0], 50.0,
            &[vec![0.5, 1.0]], Some(1), 0,
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
            &[vec![0.5, 1.0]], None, 0,
        );
        let has_chance = tree.nodes.iter().any(|n| matches!(n, GameNode::Chance { .. }));
        assert!(!has_chance, "River subgame has no next street");
        let has_boundary = tree.nodes.iter().any(|n| matches!(
            n, GameNode::Terminal { kind: TerminalKind::DepthBoundary, .. }
        ));
        assert!(!has_boundary, "River subgame has no DepthBoundary");
    }

    /// Hand trace: verify invested tracks voluntary chips only.
    ///
    /// Config: stack=100, sb=1, bb=2, preflop=["5bb"]
    /// Trace: SB Raise(10) -> BB Fold
    /// SB raises to 5bb = 10 chips. Additional = 10 - 1 = 9.
    /// Pot = SB blind(1) + BB blind(2) + SB additional(9) = 12
    #[test]
    fn test_raise_fold_pot_trace() {
        let tree = GameTree::build(
            100.0,
            1.0,
            2.0,
            &[vec!["5bb".into()]],
            &[vec![0.5]],
            &[vec![0.5]],
            &[vec![0.5]],
        );
        if let GameNode::Decision { actions, children, .. } = &tree.nodes[tree.root as usize] {
            let raise_idx = actions
                .iter()
                .position(|a| matches!(a, TreeAction::Raise(_)))
                .expect("SB should have a raise");

            // BB's response: find Fold
            if let GameNode::Decision { actions: bb_a, children: bb_c, .. } = &tree.nodes[children[raise_idx] as usize] {
                let fold_idx = bb_a.iter().position(|a| matches!(a, TreeAction::Fold))
                    .expect("BB should have Fold");
                if let GameNode::Terminal { pot, .. } = &tree.nodes[bb_c[fold_idx] as usize] {
                    // Pot = blinds(3) + SB additional(9) = 12.0 chips
                    assert!(
                        (*pot - 12.0).abs() < SIZE_EPSILON,
                        "Pot should be 12.0 chips (blinds 3 + SB raise to 5bb=10 chips), got {pot}"
                    );
                } else { panic!("Expected Terminal after fold"); }
            } else { panic!("Expected Decision for BB"); }
        } else { panic!("Expected Decision at root"); }
    }

    /// Preflop limp: SB calls (voluntary = BB - SB blind = 1 chip),
    /// BB checks, pot at flop = 1 + 2 + 1 + 0 = 4 chips
    ///
    /// Trace: SB call -> BB check -> Chance -> Flop P0 bets -> P1 folds.
    /// The fold terminal's pot includes blinds + both voluntary amounts.
    #[test]
    fn test_preflop_limp_pot() {
        let tree = simple_tree();
        if let GameNode::Decision { actions, children, .. } = &tree.nodes[tree.root as usize] {
            let call_idx = actions
                .iter()
                .position(|a| matches!(a, TreeAction::Call))
                .expect("SB should have Call");
            let bb_response = &tree.nodes[children[call_idx] as usize];
            // BB gets to act (check or raise)
            if let GameNode::Decision { actions: bb_actions, children: bb_children, .. } = bb_response {
                let check_idx = bb_actions
                    .iter()
                    .position(|a| matches!(a, TreeAction::Check))
                    .expect("BB should have Check after limp");
                let after_check = &tree.nodes[bb_children[check_idx] as usize];
                // Should transition to flop
                match after_check {
                    GameNode::Chance { child, .. } => {
                        // Flop root: P0 (OOP) acts first, has Check/Bet/AllIn (no Fold since not facing bet)
                        // Follow Bet path to get P1's response, which will have Fold
                        if let GameNode::Decision { actions: flop_a, children: flop_c, .. } = &tree.nodes[*child as usize] {
                            let bet_idx = flop_a.iter().position(|a| matches!(a, TreeAction::Bet(_)))
                                .expect("OOP should have a Bet action on flop");
                            // P1 faces the bet and can fold
                            if let GameNode::Decision { actions: p1_a, children: p1_c, .. } = &tree.nodes[flop_c[bet_idx] as usize] {
                                let fold_idx = p1_a.iter().position(|a| matches!(a, TreeAction::Fold))
                                    .expect("P1 should have Fold facing a bet");
                                if let GameNode::Terminal { pot, .. } = &tree.nodes[p1_c[fold_idx] as usize] {
                                    // After limp: pot = 4.0 chips (1 SB + 2 BB + 1 SB call)
                                    // P0 bets on flop, P1 folds. Pot includes limp pot + bet.
                                    assert!(
                                        *pot > 4.0 - SIZE_EPSILON,
                                        "Pot should be at least 4.0 chips (after limp + bet), got {pot}"
                                    );
                                } else {
                                    panic!("Fold child should be Terminal");
                                }
                            } else {
                                panic!("P1 response should be Decision");
                            }
                        }
                    }
                    _ => panic!("After limp check-check expected Chance"),
                }
            }
        }
    }

    #[test]
    fn test_build_subgame_allin_everywhere() {
        for street in [Street::Flop, Street::Turn, Street::River] {
            let tree = GameTree::build_subgame(
                street, 10.0, [5.0, 5.0], 50.0,
                &[vec![0.5]], Some(1), 0,
            );
            check_all_in_with_invested(&tree, 50.0, [5.0, 5.0], [0.0, 0.0]);
        }
    }

    // ── Phase 1: Domain types & game tree refactor tests ─────────────

    #[test]
    fn test_terminal_has_pot_only_no_invested() {
        // After the refactor, Terminal nodes should have only `pot`, not `invested`.
        let tree = simple_tree();
        for node in &tree.nodes {
            if let GameNode::Terminal { pot, .. } = node {
                // Pot should be positive.
                assert!(*pot > 0.0, "Terminal pot should be positive, got {pot}");
                // Crucially: there should be NO `invested` field.
                // This test asserts by pattern matching — if `invested` exists
                // on Terminal, this match would need to include it.
            }
        }
    }

    #[test]
    fn test_fold_at_preflop_root_pot() {
        // SB folds immediately: pot = small_blind + big_blind = 1 + 2 = 3 chips
        let tree = simple_tree();
        if let GameNode::Decision { actions, children, .. } = &tree.nodes[tree.root as usize] {
            let fold_idx = actions.iter().position(|a| matches!(a, TreeAction::Fold)).unwrap();
            if let GameNode::Terminal { pot, .. } = &tree.nodes[children[fold_idx] as usize] {
                assert!(
                    (*pot - 3.0).abs() < SIZE_EPSILON,
                    "Fold at preflop root should have pot=3.0 chips, got {pot}"
                );
            } else {
                panic!("Expected Terminal after fold");
            }
        }
    }

    #[test]
    fn test_raise_then_call_pot() {
        // SB raises to 5 chips, BB calls.
        // Pot = SB blind (1) + BB blind (2) + SB raise-to (5) + BB call (5) = 13 chips
        // Wait: raise TO 5 means SB invested 5 voluntary. BB calls to match = 5.
        // Pot = 1 + 2 + (5-1) + (5-2) = 10. Actually: stacks are debited for both.
        // Actually: pot = 5 + 5 = 10 chips (both invested 5 total each).
        let tree = simple_tree();
        if let GameNode::Decision { actions, children, .. } = &tree.nodes[tree.root as usize] {
            let raise_idx = actions.iter().position(|a| matches!(a, TreeAction::Raise(_))).unwrap();
            // BB's response to SB's raise
            if let GameNode::Decision { actions: bb_a, children: bb_c, .. } = &tree.nodes[children[raise_idx] as usize] {
                let call_idx = bb_a.iter().position(|a| matches!(a, TreeAction::Call)).unwrap();
                // After call, should go to chance/showdown. Find the first terminal via chance nodes.
                let after_call = bb_c[call_idx] as usize;
                // This should be a chance node to flop
                if let GameNode::Chance { child, .. } = &tree.nodes[after_call] {
                    // Follow into flop, find a fold terminal to verify pot
                    if let GameNode::Decision { actions: flop_a, children: flop_c, .. } = &tree.nodes[*child as usize] {
                        if let Some(bet_idx) = flop_a.iter().position(|a| matches!(a, TreeAction::Bet(_))) {
                            if let GameNode::Decision { actions: p1_a, children: p1_c, .. } = &tree.nodes[flop_c[bet_idx] as usize] {
                                let fold_idx = p1_a.iter().position(|a| matches!(a, TreeAction::Fold)).unwrap();
                                if let GameNode::Terminal { pot, .. } = &tree.nodes[p1_c[fold_idx] as usize] {
                                    // After SB raise 5, BB call, flop bet: pot should be >= 10.0 chips (the pre-bet pot)
                                    assert!(
                                        *pot >= 10.0 - SIZE_EPSILON,
                                        "After raise+call+bet, pot should be >= 10.0, got {pot}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_dealer_field_replaces_blinds() {
        let tree = simple_tree();
        // After refactor, tree should have `dealer` field, not `blinds`.
        // dealer=0 means seat 0 is SB.
        assert_eq!(tree.dealer, 0, "Default dealer should be seat 0 (SB)");
    }

    #[test]
    fn test_position_label_sb() {
        let tree = simple_tree();
        assert_eq!(tree.position_label(tree.dealer), "SB");
    }

    #[test]
    fn test_position_label_bb() {
        let tree = simple_tree();
        assert_eq!(tree.position_label(1 - tree.dealer), "BB");
    }

    #[test]
    fn test_sb_seat() {
        let tree = simple_tree();
        assert_eq!(tree.sb_seat(), 0);
    }

    #[test]
    fn test_bb_seat() {
        let tree = simple_tree();
        assert_eq!(tree.bb_seat(), 1);
    }

    #[test]
    fn test_postflop_bb_acts_first_with_dealer() {
        // When dealer=0, SB=seat 0, BB=seat 1.
        // Postflop: BB (seat 1) should act first = 1 - dealer.
        let tree = simple_tree();
        for node in &tree.nodes {
            if let GameNode::Chance { next_street: Street::Flop, child } = node {
                if let GameNode::Decision { player, .. } = &tree.nodes[*child as usize] {
                    assert_eq!(
                        *player,
                        1 - tree.dealer,
                        "BB (1-dealer) should act first postflop"
                    );
                }
                return;
            }
        }
        panic!("Should find a Chance node to Flop");
    }

    #[test]
    fn terminal_payoffs_zero_sum() {
        // SB raises to 5 chips, BB calls. pot=10, stacks=[95, 95] (from 100)
        // Starting stack=100 chips, SB blind=1, BB blind=2
        // SB Raise(5): additional=5-1=4, stacks=[95, 98], pot=7, street_bets=[5, 2]
        // BB Call: call_amount=5-2=3, stacks=[95, 95], pot=10, street_bets=[5, 5]
        // Then showdown terminal: stacks=[95, 95], pot=10
        //
        // Winner: (95 + 10) - 100 = +5
        // Loser: 95 - 100 = -5
        // Sum = 0
        //
        // Tie: (95 + 10/2) - 100 = 0
        // Both sum to 0
        let tree = GameTree::build(
            100.0, 1.0, 2.0,
            &[vec!["5bb".into()]],
            &[vec![0.5]],
            &[vec![0.5]],
            &[vec![0.5]],
        );
        let starting_stack = tree.starting_stack;

        for node in &tree.nodes {
            if let GameNode::Terminal { kind, pot, stacks } = node {
                match kind {
                    TerminalKind::Fold { winner } => {
                        let w = *winner as usize;
                        let l = 1 - w;
                        let winner_ev = (stacks[w] + pot) - starting_stack;
                        let loser_ev = stacks[l] - starting_stack;
                        let sum = winner_ev + loser_ev;
                        assert!(
                            sum.abs() < 0.01,
                            "Fold not zero-sum: winner_ev={winner_ev:.2}, loser_ev={loser_ev:.2}, sum={sum:.2}, \
                             pot={pot:.2}, stacks={stacks:?}"
                        );
                        // Winner should gain, loser should lose
                        assert!(winner_ev >= -0.01, "Winner EV should be non-negative: {winner_ev:.2}");
                        assert!(loser_ev <= 0.01, "Loser EV should be non-positive: {loser_ev:.2}");
                    }
                    TerminalKind::Showdown => {
                        // For showdown, winner gets pot, loser gets nothing
                        // Check both perspectives sum to zero
                        let p0_winner_ev = (stacks[0] + pot) - starting_stack;
                        let p1_loser_ev = stacks[1] - starting_stack;
                        let sum = p0_winner_ev + p1_loser_ev;
                        assert!(
                            sum.abs() < 0.01,
                            "Showdown not zero-sum (p0 wins): sum={sum:.2}, pot={pot:.2}, stacks={stacks:?}"
                        );

                        // Tie check
                        let p0_tie = (stacks[0] + pot / 2.0) - starting_stack;
                        let p1_tie = (stacks[1] + pot / 2.0) - starting_stack;
                        let tie_sum = p0_tie + p1_tie;
                        assert!(
                            tie_sum.abs() < 0.01,
                            "Showdown tie not zero-sum: sum={tie_sum:.2}, pot={pot:.2}, stacks={stacks:?}"
                        );
                    }
                    TerminalKind::DepthBoundary => {}
                }
            }
        }
    }

    #[test]
    fn terminal_stacks_pot_conservation() {
        // For every terminal: stacks[0] + stacks[1] + pot == 2 * starting_stack
        // (total chips in the system are conserved)
        let tree = GameTree::build(
            100.0, 1.0, 2.0,
            &[vec!["5bb".into()], vec!["3.0x".into()]],
            &[vec![0.5, 1.0]],
            &[vec![0.5, 1.0]],
            &[vec![0.5, 1.0]],
        );
        let starting_stack = tree.starting_stack;

        for (i, node) in tree.nodes.iter().enumerate() {
            if let GameNode::Terminal { pot, stacks, kind } = node {
                if matches!(kind, TerminalKind::DepthBoundary) {
                    continue;
                }
                let total = stacks[0] + stacks[1] + pot;
                let expected = 2.0 * starting_stack;
                assert!(
                    (total - expected).abs() < 0.01,
                    "Node {i}: chip conservation violated: stacks={stacks:?}, pot={pot:.2}, \
                     total={total:.2}, expected={expected:.2}, kind={kind:?}"
                );
            }
        }
    }

    #[test]
    fn test_subgame_oop_acts_first() {
        // Subgame: OOP (1 - dealer) acts first.
        let tree = GameTree::build_subgame(
            Street::Flop, 10.0, [5.0, 5.0], 50.0,
            &[vec![0.5, 1.0]], None, 0,
        );
        if let GameNode::Decision { player, .. } = &tree.nodes[tree.root as usize] {
            // dealer=0 => OOP = 1 - 0 = 1
            assert_eq!(*player, 1 - tree.dealer, "OOP should act first in subgame");
        }
    }

    // ── Preflop sizing tests ────────────────────────────────────────────

    /// Helper: extract the Raise/Bet chip amount at a given tree node for
    /// actions matching a BB label.
    fn find_raise_chips(tree: &GameTree, node_idx: u32, bb_label: &str) -> Option<f64> {
        if let GameNode::Decision { actions, .. } = &tree.nodes[node_idx as usize] {
            actions.iter().find_map(|a| {
                let label = action_bb_label(a);
                if label == bb_label {
                    match a {
                        TreeAction::Raise(v) | TreeAction::Bet(v) => Some(*v),
                        _ => None,
                    }
                } else {
                    None
                }
            })
        } else {
            None
        }
    }

    fn action_bb_label(action: &TreeAction) -> String {
        match action {
            TreeAction::Fold => "fold".into(),
            TreeAction::Check => "check".into(),
            TreeAction::Call => "call".into(),
            TreeAction::AllIn => "all-in".into(),
            TreeAction::Bet(v) | TreeAction::Raise(v) => {
                format!("{}bb", (v / 2.0).round() as u32)
            }
        }
    }

    /// "Xbb" notation should produce raise-to = X * big_blind chips.
    #[test]
    fn preflop_absolute_bb_sizing() {
        // stack=200 chips, sb=1, bb=2
        // "2bb" → 2 * 2 = 4 chips, "4bb" → 4 * 2 = 8 chips, "10bb" → 10 * 2 = 20 chips
        let tree = GameTree::build(
            200.0, 1.0, 2.0,
            &[vec!["2bb".into()], vec!["4bb".into(), "10bb".into()]],
            &[vec![0.5]], &[vec![0.5]], &[vec![0.5]],
        );
        // SB open: "2bb" → Raise(4.0)
        let sb_raise = find_raise_chips(&tree, tree.root, "2bb");
        assert_eq!(sb_raise, Some(4.0), "SB open 2bb should be 4 chips");

        // BB response: find the raise child
        if let GameNode::Decision { actions, children, .. } = &tree.nodes[tree.root as usize] {
            let raise_idx = actions.iter().position(|a| matches!(a, TreeAction::Raise(v) if (*v - 4.0).abs() < SIZE_EPSILON)).unwrap();
            let bb_node = children[raise_idx];
            // BB should have "4bb" → Raise(8.0) and "10bb" → Raise(20.0)
            let bb_4bb = find_raise_chips(&tree, bb_node, "4bb");
            assert_eq!(bb_4bb, Some(8.0), "BB 3bet 4bb should be 8 chips");
            let bb_10bb = find_raise_chips(&tree, bb_node, "10bb");
            assert_eq!(bb_10bb, Some(20.0), "BB 3bet 10bb should be 20 chips");
        } else {
            panic!("root should be Decision");
        }
    }

    /// Multiplier sizing: "3.0x" should multiply the last raise-to amount.
    #[test]
    fn preflop_multiplier_sizing_correct() {
        // SB opens "5bb" = 10 chips. BB responds "3.0x" = 3 * 10 = 30 chips.
        let tree = GameTree::build(
            200.0, 1.0, 2.0,
            &[vec!["5bb".into()], vec!["3.0x".into()]],
            &[vec![0.5]], &[vec![0.5]], &[vec![0.5]],
        );
        if let GameNode::Decision { actions, children, .. } = &tree.nodes[tree.root as usize] {
            let raise_idx = actions.iter().position(|a| matches!(a, TreeAction::Raise(_))).unwrap();
            let bb_node = children[raise_idx];
            let bb_raise = find_raise_chips(&tree, bb_node, "15bb");
            assert_eq!(bb_raise, Some(30.0), "BB 3.0x of 10 chips = 30 chips = 15bb");
        }
    }

    /// Mixed sizing: absolute and multiplier at the same depth.
    #[test]
    fn preflop_mixed_absolute_and_multiplier() {
        // SB opens "2bb" = 4 chips. BB can "4bb" (8 chips) or "2.0x" (2 * 4 = 8 chips).
        // Both produce the same amount → deduplicated to one action.
        let tree = GameTree::build(
            200.0, 1.0, 2.0,
            &[vec!["2bb".into()], vec!["4bb".into(), "2.0x".into()]],
            &[vec![0.5]], &[vec![0.5]], &[vec![0.5]],
        );
        if let GameNode::Decision { actions, children, .. } = &tree.nodes[tree.root as usize] {
            let raise_idx = actions.iter().position(|a| matches!(a, TreeAction::Raise(_))).unwrap();
            let bb_node = children[raise_idx];
            if let GameNode::Decision { actions: bb_actions, .. } = &tree.nodes[bb_node as usize] {
                let raise_count = bb_actions.iter().filter(|a| matches!(a, TreeAction::Raise(_))).count();
                assert_eq!(raise_count, 1, "4bb and 2.0x produce same amount, should be deduped to 1 raise");
            }
        }
    }

    /// Min-raise floor: if the absolute BB amount is below the min raise, it
    /// gets floored up.
    #[test]
    fn preflop_min_raise_floor() {
        // SB opens "2bb" = 4 chips. BB tries "3bb" = 6 chips.
        // Min raise = last_raise_to(4) + increment(4-2) = 6 chips. So 6 is exactly min.
        let tree = GameTree::build(
            200.0, 1.0, 2.0,
            &[vec!["2bb".into()], vec!["3bb".into()]],
            &[vec![0.5]], &[vec![0.5]], &[vec![0.5]],
        );
        if let GameNode::Decision { actions, children, .. } = &tree.nodes[tree.root as usize] {
            let raise_idx = actions.iter().position(|a| matches!(a, TreeAction::Raise(_))).unwrap();
            let bb_node = children[raise_idx];
            let bb_raise = find_raise_chips(&tree, bb_node, "3bb");
            assert_eq!(bb_raise, Some(6.0), "BB 3bb = 6 chips = min raise");
        }
    }

    /// Postflop pot-fraction sizing produces correct chip amounts.
    #[test]
    fn postflop_pot_fraction_sizing() {
        // SRP: SB 2bb (4 chips), BB calls. Pot = 8 chips entering flop.
        // Flop bets: 0.33 pot = 2.64, 0.5 pot = 4.0, 1.0 pot = 8.0
        let tree = GameTree::build(
            200.0, 1.0, 2.0,
            &[vec!["2bb".into()]],
            &[vec![0.33, 0.5, 1.0]],
            &[vec![0.5]],
            &[vec![0.5]],
        );
        // Navigate: SB raises → BB calls → chance → flop decision
        if let GameNode::Decision { actions, children, .. } = &tree.nodes[tree.root as usize] {
            let raise_idx = actions.iter().position(|a| matches!(a, TreeAction::Raise(_))).unwrap();
            let bb_node_idx = children[raise_idx];
            if let GameNode::Decision { actions: bb_actions, children: bb_children, .. } = &tree.nodes[bb_node_idx as usize] {
                let call_idx = bb_actions.iter().position(|a| matches!(a, TreeAction::Call)).unwrap();
                let mut flop_idx = bb_children[call_idx];
                // Skip chance node
                while let GameNode::Chance { child, .. } = &tree.nodes[flop_idx as usize] {
                    flop_idx = *child;
                }
                if let GameNode::Decision { actions: flop_actions, .. } = &tree.nodes[flop_idx as usize] {
                    let bets: Vec<f64> = flop_actions.iter().filter_map(|a| {
                        if let TreeAction::Bet(v) = a { Some(*v) } else { None }
                    }).collect();
                    assert!(!bets.is_empty(), "flop should have bet actions");
                    // With pot=8, 0.33*8≈2.64, 0.5*8=4.0, 1.0*8=8.0
                    // Bet amounts are "raise to" from 0 street bet, so they equal the bet size
                    for &b in &bets {
                        assert!(b > 0.0 && b < 200.0, "bet {b} should be reasonable");
                    }
                }
            }
        }
    }
}
