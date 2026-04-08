//! N-player game tree builder for multiplayer MCCFR.
//!
//! Builds an arena-allocated game tree for 2-8 player NLHE with
//! configurable action abstraction per street and per raise depth.

#![allow(clippy::cast_possible_truncation)]

use super::config::{ForcedBetKind, MpActionAbstractionConfig, MpGameConfig};
use super::types::{Chips, PlayerSet, Seat, Street};
use super::MAX_PLAYERS;

/// Tolerance for comparing bet sizes (in chips).
const SIZE_EPSILON: f64 = 0.01;

// ── Public types ────────────────────────────────────────────────────

/// An action in the game tree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TreeAction {
    Fold,
    Check,
    Call,
    /// Lead bet (opening into an unbet pot). Amount in chips.
    Lead(f64),
    /// Raise TO amount in chips.
    Raise(f64),
    AllIn,
}

/// A node in the arena-allocated game tree.
#[derive(Debug, Clone)]
pub enum MpGameNode {
    Decision {
        seat: Seat,
        street: Street,
        actions: Vec<TreeAction>,
        children: Vec<u32>,
    },
    Chance {
        next_street: Street,
        child: u32,
    },
    Terminal {
        kind: TerminalKind,
        pot: Chips,
        contributions: [Chips; MAX_PLAYERS],
    },
}

/// How a terminal node was reached.
#[derive(Debug, Clone)]
pub enum TerminalKind {
    LastStanding { winner: Seat },
    Showdown { active: PlayerSet },
}

/// The complete N-player game tree.
#[derive(Debug)]
pub struct MpGameTree {
    pub nodes: Vec<MpGameNode>,
    pub root: u32,
    pub num_players: u8,
    pub starting_stack: Chips,
}

// ── Internal: preflop size parsing ──────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum PreflopSize {
    Absolute(f64),
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

// ── Internal: parsed config ─────────────────────────────────────────

struct TreeBuildConfig {
    preflop_lead: Vec<PreflopSize>,
    preflop_raise: Vec<Vec<PreflopSize>>,
    flop_lead: Vec<f64>,
    flop_raise: Vec<Vec<f64>>,
    turn_lead: Vec<f64>,
    turn_raise: Vec<Vec<f64>>,
    river_lead: Vec<f64>,
    river_raise: Vec<Vec<f64>>,
}

impl TreeBuildConfig {
    fn from_action_config(ac: &MpActionAbstractionConfig) -> Self {
        Self {
            preflop_lead: parse_preflop_values(&ac.preflop.lead),
            preflop_raise: parse_preflop_raise_depths(&ac.preflop.raise),
            flop_lead: parse_f64_values(&ac.flop.lead),
            flop_raise: parse_f64_raise_depths(&ac.flop.raise),
            turn_lead: parse_f64_values(&ac.turn.lead),
            turn_raise: parse_f64_raise_depths(&ac.turn.raise),
            river_lead: parse_f64_values(&ac.river.lead),
            river_raise: parse_f64_raise_depths(&ac.river.raise),
        }
    }

    fn lead_sizes(&self, street: Street) -> LeadSizes<'_> {
        match street {
            Street::Preflop => LeadSizes::Preflop(&self.preflop_lead),
            Street::Flop => LeadSizes::Postflop(&self.flop_lead),
            Street::Turn => LeadSizes::Postflop(&self.turn_lead),
            Street::River => LeadSizes::Postflop(&self.river_lead),
        }
    }

    fn raise_sizes_at_depth(&self, street: Street, depth: usize) -> RaiseSizes<'_> {
        match street {
            Street::Preflop => RaiseSizes::Preflop(get_depth_or_last(&self.preflop_raise, depth)),
            Street::Flop => RaiseSizes::Postflop(get_depth_or_last(&self.flop_raise, depth)),
            Street::Turn => RaiseSizes::Postflop(get_depth_or_last(&self.turn_raise, depth)),
            Street::River => RaiseSizes::Postflop(get_depth_or_last(&self.river_raise, depth)),
        }
    }

    fn max_raise_depths(&self, street: Street) -> usize {
        match street {
            Street::Preflop => self.preflop_raise.len(),
            Street::Flop => self.flop_raise.len(),
            Street::Turn => self.turn_raise.len(),
            Street::River => self.river_raise.len(),
        }
    }
}

#[derive(Clone, Copy)]
enum LeadSizes<'a> {
    Preflop(&'a [PreflopSize]),
    Postflop(&'a [f64]),
}

#[derive(Clone, Copy)]
enum RaiseSizes<'a> {
    Preflop(&'a [PreflopSize]),
    Postflop(&'a [f64]),
}

fn get_depth_or_last<T>(depths: &[Vec<T>], idx: usize) -> &[T] {
    depths.get(idx).or_else(|| depths.last()).map_or(&[], Vec::as_slice)
}

// ── Internal: YAML parsing helpers ──────────────────────────────────

fn parse_preflop_values(values: &[serde_yaml::Value]) -> Vec<PreflopSize> {
    values.iter().map(|v| PreflopSize::parse(&yaml_to_string(v))).collect()
}

fn parse_preflop_raise_depths(depths: &[Vec<serde_yaml::Value>]) -> Vec<Vec<PreflopSize>> {
    depths.iter().map(|d| parse_preflop_values(d)).collect()
}

fn parse_f64_values(values: &[serde_yaml::Value]) -> Vec<f64> {
    values
        .iter()
        .map(|v| yaml_to_f64(v).expect("expected numeric size"))
        .collect()
}

fn parse_f64_raise_depths(depths: &[Vec<serde_yaml::Value>]) -> Vec<Vec<f64>> {
    depths.iter().map(|d| parse_f64_values(d)).collect()
}

fn yaml_to_string(v: &serde_yaml::Value) -> String {
    match v {
        serde_yaml::Value::String(s) => s.clone(),
        serde_yaml::Value::Number(n) => format!("{n}"),
        _ => panic!("unexpected YAML value type for size: {v:?}"),
    }
}

fn yaml_to_f64(v: &serde_yaml::Value) -> Option<f64> {
    match v {
        serde_yaml::Value::Number(n) => n.as_f64(),
        serde_yaml::Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

// ── Internal: build state ───────────────────────────────────────────

#[derive(Clone)]
struct BuildState {
    stacks: [Chips; MAX_PLAYERS],
    street_bets: [Chips; MAX_PLAYERS],
    contributions: [Chips; MAX_PLAYERS],
    active: PlayerSet,
    all_in: PlayerSet,
    acted_since_aggression: PlayerSet,
    street: Street,
    pot: Chips,
    num_players: u8,
    raise_count: u8,
    to_act: Seat,
    facing_bet: bool,
    last_raise_to: Chips,
    dealer: u8,
    big_blind_amount: Chips,
}

// ── Public build entry point ────────────────────────────────────────

impl MpGameTree {
    /// Build a game tree from game and action abstraction configs.
    #[must_use]
    pub fn build(config: &MpGameConfig, action_config: &MpActionAbstractionConfig) -> Self {
        let tree_config = TreeBuildConfig::from_action_config(action_config);
        let stack = Chips(config.stack_depth);
        let state = init_build_state(config, stack);
        let mut builder = TreeBuilder {
            config: &tree_config,
            nodes: Vec::new(),
        };
        let root = builder.build_node(&state);
        let nodes = builder.nodes;
        let decision_count = nodes.iter().filter(|n| matches!(n, MpGameNode::Decision { .. })).count();
        let chance_count = nodes.iter().filter(|n| matches!(n, MpGameNode::Chance { .. })).count();
        let terminal_count = nodes.iter().filter(|n| matches!(n, MpGameNode::Terminal { .. })).count();
        eprintln!(
            "  MP Tree: {} nodes ({} decision, {} chance, {} terminal)",
            nodes.len(), decision_count, chance_count, terminal_count,
        );
        Self {
            nodes,
            root,
            num_players: config.num_players,
            starting_stack: stack,
        }
    }
}

struct TreeBuilder<'a> {
    config: &'a TreeBuildConfig,
    nodes: Vec<MpGameNode>,
}

// ── Initialization ──────────────────────────────────────────────────

fn init_build_state(config: &MpGameConfig, stack: Chips) -> BuildState {
    let n = config.num_players;
    let mut state = BuildState {
        stacks: [Chips::ZERO; MAX_PLAYERS],
        street_bets: [Chips::ZERO; MAX_PLAYERS],
        contributions: [Chips::ZERO; MAX_PLAYERS],
        active: PlayerSet::all(n),
        all_in: PlayerSet::empty(),
        acted_since_aggression: PlayerSet::empty(),
        street: Street::Preflop,
        pot: Chips::ZERO,
        num_players: n,
        raise_count: 0,
        to_act: Seat::from_raw(0),
        facing_bet: true,
        last_raise_to: Chips(2.0),
        dealer: find_dealer(config),
        big_blind_amount: Chips(2.0),
    };

    for s in state.stacks.iter_mut().take(n as usize) {
        *s = stack;
    }

    let mut bb_seat: u8 = 1;
    let mut straddle_seat: Option<u8> = None;

    for blind in &config.blinds {
        let amt = Chips(blind.amount);
        state.apply_forced_bet(blind.seat as usize, amt);
        if blind.kind == ForcedBetKind::BigBlind {
            state.big_blind_amount = amt;
            bb_seat = blind.seat;
        }
        if blind.kind == ForcedBetKind::Straddle {
            straddle_seat = Some(blind.seat);
        }
    }

    state.last_raise_to = state.big_blind_amount;
    state.to_act = find_preflop_first_to_act(n, bb_seat, straddle_seat);
    state
}

impl BuildState {
    fn apply_forced_bet(&mut self, seat: usize, amount: Chips) {
        self.stacks[seat] -= amount;
        self.street_bets[seat] += amount;
        self.contributions[seat] += amount;
        self.pot += amount;
    }
}

fn find_preflop_first_to_act(
    num_players: u8,
    bb_seat: u8,
    straddle_seat: Option<u8>,
) -> Seat {
    let after = straddle_seat.unwrap_or(bb_seat);
    let next = (after + 1) % num_players;
    Seat::new(next, num_players)
}

fn find_dealer(config: &MpGameConfig) -> u8 {
    // In 2-player, SB = BTN = seat 0. In N-player, BTN is seat before SB.
    // Convention: first SB blind determines BTN position.
    if config.num_players == 2 {
        // Heads-up: SB is the BTN
        for blind in &config.blinds {
            if blind.kind == ForcedBetKind::SmallBlind {
                return blind.seat;
            }
        }
        0
    } else {
        // BTN is the seat before SB (clockwise)
        for blind in &config.blinds {
            if blind.kind == ForcedBetKind::SmallBlind {
                return (blind.seat + config.num_players - 1) % config.num_players;
            }
        }
        config.num_players - 1
    }
}

// ── Action generation (pure functions on BuildState) ─────────────────

fn generate_actions(config: &TreeBuildConfig, state: &BuildState) -> Vec<TreeAction> {
    let mut actions = Vec::new();

    if state.facing_bet {
        actions.push(TreeAction::Fold);
    }

    add_check_or_call(state, &mut actions);
    add_sized_actions(config, state, &mut actions);
    add_all_in_if_needed(state, &mut actions);
    dedup_all_in(&mut actions);

    actions
}

fn add_check_or_call(state: &BuildState, actions: &mut Vec<TreeAction>) {
    let seat = state.to_act;
    let remaining = state.stacks[seat.index() as usize];

    if state.facing_bet {
        let to_call = max_street_bet(state) - state.street_bets[seat.index() as usize];
        if to_call >= remaining - Chips(SIZE_EPSILON) {
            actions.push(TreeAction::AllIn);
        } else {
            actions.push(TreeAction::Call);
        }
    } else {
        actions.push(TreeAction::Check);
    }
}

fn add_sized_actions(
    config: &TreeBuildConfig,
    state: &BuildState,
    actions: &mut Vec<TreeAction>,
) {
    let depth = state.raise_count as usize;
    if depth >= config.max_raise_depths(state.street) {
        return;
    }

    let remaining = state.stacks[state.to_act.index() as usize];
    if remaining < Chips(SIZE_EPSILON) {
        return;
    }

    if state.facing_bet {
        add_raise_sizes(state, config.raise_sizes_at_depth(state.street, depth), actions);
    } else {
        add_lead_sizes(state, config.lead_sizes(state.street), actions);
    }
}

// NOTE: add_lead_sizes and add_raise_sizes share structure (match/loop/resolve)
// but differ in how amounts are computed: leads use pot*frac, raises use
// call+pot_after_call*frac. Unifying would require closures and hurt readability.
fn add_lead_sizes(state: &BuildState, sizes: LeadSizes<'_>, actions: &mut Vec<TreeAction>) {
    match sizes {
        LeadSizes::Preflop(preflop) => {
            for &size in preflop {
                try_add_lead(state, resolve_preflop_size(state, size), actions);
            }
        }
        LeadSizes::Postflop(fractions) => {
            for &frac in fractions {
                try_add_lead(state, state.pot * frac, actions);
            }
        }
    }
}

fn add_raise_sizes(state: &BuildState, sizes: RaiseSizes<'_>, actions: &mut Vec<TreeAction>) {
    match sizes {
        RaiseSizes::Preflop(preflop) => {
            for &size in preflop {
                try_add_raise(state, resolve_preflop_raise_to(state, size), actions);
            }
        }
        RaiseSizes::Postflop(fractions) => {
            for &frac in fractions {
                try_add_raise(state, compute_postflop_raise_to(state, frac), actions);
            }
        }
    }
}

fn resolve_preflop_size(state: &BuildState, size: PreflopSize) -> Chips {
    match size {
        PreflopSize::Absolute(bb) => state.big_blind_amount * bb,
        PreflopSize::Multiplier(mult) => state.last_raise_to * mult,
    }
}

fn resolve_preflop_raise_to(state: &BuildState, size: PreflopSize) -> Chips {
    resolve_preflop_size(state, size).max(min_raise_to(state))
}

fn compute_postflop_raise_to(state: &BuildState, frac: f64) -> Chips {
    let my_bet = state.street_bets[state.to_act.index() as usize];
    let call_amount = max_street_bet(state) - my_bet;
    let pot_after_call = state.pot + call_amount;
    let raise_amount = call_amount + pot_after_call * frac;
    (my_bet + raise_amount).max(min_raise_to(state))
}

fn try_add_lead(state: &BuildState, amount: Chips, actions: &mut Vec<TreeAction>) {
    let my_bet = state.street_bets[state.to_act.index() as usize];
    let raise_to = (my_bet + amount).max(min_raise_to(state));
    try_add_sized_action(state, actions, raise_to, TreeAction::Lead);
}

fn try_add_raise(state: &BuildState, raise_to: Chips, actions: &mut Vec<TreeAction>) {
    try_add_sized_action(state, actions, raise_to, TreeAction::Raise);
}

fn try_add_sized_action(
    state: &BuildState,
    actions: &mut Vec<TreeAction>,
    raise_to: Chips,
    variant: fn(f64) -> TreeAction,
) {
    let idx = state.to_act.index() as usize;
    let remaining = state.stacks[idx];
    let all_in_to = state.street_bets[idx] + remaining;
    let additional = raise_to - state.street_bets[idx];

    if raise_to >= all_in_to - Chips(SIZE_EPSILON) {
        return; // all-in added separately
    }
    if additional > remaining + Chips(SIZE_EPSILON) {
        return;
    }
    if is_size_duplicate(actions, raise_to) {
        return;
    }
    actions.push(variant(raise_to.0));
}

fn is_size_duplicate(actions: &[TreeAction], amount: Chips) -> bool {
    actions.iter().any(|a| match a {
        TreeAction::Lead(v) | TreeAction::Raise(v) => (Chips(*v) - amount).0.abs() < SIZE_EPSILON,
        _ => false,
    })
}

fn add_all_in_if_needed(state: &BuildState, actions: &mut Vec<TreeAction>) {
    let seat = state.to_act;
    let remaining = state.stacks[seat.index() as usize];
    if remaining < Chips(SIZE_EPSILON) {
        return;
    }
    let all_in_to = state.street_bets[seat.index() as usize] + remaining;
    let already = actions.iter().any(|a| match a {
        TreeAction::Lead(v) | TreeAction::Raise(v) => {
            (Chips(*v) - all_in_to).0.abs() < SIZE_EPSILON
        }
        TreeAction::AllIn => true,
        _ => false,
    });
    if !already {
        actions.push(TreeAction::AllIn);
    }
}

fn dedup_all_in(actions: &mut Vec<TreeAction>) {
    let mut seen = false;
    actions.retain(|a| {
        if matches!(a, TreeAction::AllIn) {
            if seen {
                return false;
            }
            seen = true;
        }
        true
    });
}

// ── Bet math ────────────────────────────────────────────────────────

fn max_street_bet(state: &BuildState) -> Chips {
    let n = state.num_players as usize;
    state.street_bets[..n]
        .iter()
        .copied()
        .fold(Chips::ZERO, Chips::max)
}

fn min_raise_to(state: &BuildState) -> Chips {
    let my_bet = state.street_bets[state.to_act.index() as usize];
    if state.facing_bet {
        let current_max = max_street_bet(state);
        let call_amount = current_max - my_bet;
        let raise_increment = (state.last_raise_to - my_bet).max(call_amount);
        my_bet + call_amount + raise_increment
    } else {
        my_bet + state.big_blind_amount
    }
}

// ── State query helpers ─────────────────────────────────────────────

fn next_active_non_allin(state: &BuildState, after: Seat) -> Option<Seat> {
    let n = state.num_players;
    for offset in 1..n {
        let candidate = Seat::from_raw((after.index() + offset) % n);
        if state.active.contains(candidate) && !state.all_in.contains(candidate) {
            return Some(candidate);
        }
    }
    None
}

fn count_active_non_allin(state: &BuildState) -> u8 {
    state.active.iter().filter(|s| !state.all_in.contains(*s)).count() as u8
}

fn is_bb_facing_limps(state: &BuildState, seat: Seat) -> bool {
    if state.street != Street::Preflop || state.raise_count != 0 || !state.facing_bet {
        return false;
    }
    let my_bet = state.street_bets[seat.index() as usize];
    let current_max = max_street_bet(state);
    (my_bet - current_max).0.abs() < SIZE_EPSILON
}

fn is_round_closed(state: &BuildState) -> bool {
    let current_max = max_street_bet(state);
    for seat in state.active.iter() {
        if state.all_in.contains(seat) {
            continue;
        }
        if !state.acted_since_aggression.contains(seat) {
            return false;
        }
        if (state.street_bets[seat.index() as usize] - current_max).0.abs() >= SIZE_EPSILON {
            return false;
        }
    }
    true
}

fn new_street_state(state: &BuildState, next_street: Street) -> BuildState {
    let mut next = state.clone();
    next.street = next_street;
    next.street_bets = [Chips::ZERO; MAX_PLAYERS];
    next.raise_count = 0;
    next.facing_bet = false;
    next.last_raise_to = Chips::ZERO;
    next.acted_since_aggression = PlayerSet::empty();
    next.to_act = find_postflop_first_to_act(&next);
    next
}

fn find_postflop_first_to_act(state: &BuildState) -> Seat {
    let n = state.num_players;
    for offset in 1..=n {
        let candidate = (state.dealer + offset) % n;
        let seat = Seat::from_raw(candidate);
        if state.active.contains(seat) && !state.all_in.contains(seat) {
            return seat;
        }
    }
    state.active.iter().next().unwrap_or(Seat::from_raw(0))
}

// ── TreeBuilder: recursive tree construction ────────────────────────

impl TreeBuilder<'_> {
    fn build_node(&mut self, state: &BuildState) -> u32 {
        let seat = state.to_act;
        let remaining = state.stacks[seat.index() as usize];

        if remaining < Chips(SIZE_EPSILON) && state.active.contains(seat) {
            return self.advance_past_zero_stack(state);
        }

        let state = if is_bb_facing_limps(state, seat) {
            let mut s = state.clone();
            s.facing_bet = false;
            s
        } else {
            state.clone()
        };

        let actions = generate_actions(self.config, &state);
        let node_idx = self.push_placeholder();
        let children: Vec<u32> = actions.iter().map(|a| self.build_child(&state, *a)).collect();

        self.nodes[node_idx as usize] = MpGameNode::Decision {
            seat,
            street: state.street,
            actions,
            children,
        };
        node_idx
    }

    fn push_placeholder(&mut self) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(MpGameNode::Terminal {
            kind: TerminalKind::Showdown { active: PlayerSet::empty() },
            pot: Chips::ZERO,
            contributions: [Chips::ZERO; MAX_PLAYERS],
        });
        idx
    }

    fn advance_past_zero_stack(&mut self, state: &BuildState) -> u32 {
        if let Some(next) = next_active_non_allin(state, state.to_act) {
            let mut next_state = state.clone();
            next_state.to_act = next;
            self.build_node(&next_state)
        } else {
            self.make_showdown_or_chance(state)
        }
    }

    fn build_child(&mut self, state: &BuildState, action: TreeAction) -> u32 {
        match action {
            TreeAction::Fold => self.build_fold_child(state),
            TreeAction::Check => self.build_check_child(state),
            TreeAction::Call => self.build_call_child(state),
            TreeAction::Lead(amt) | TreeAction::Raise(amt) => {
                self.apply_aggression(state, Chips(amt))
            }
            TreeAction::AllIn => self.build_allin_child(state),
        }
    }

    fn build_fold_child(&mut self, state: &BuildState) -> u32 {
        let mut next = state.clone();
        next.active.remove(state.to_act);

        if next.active.count() == 1 {
            let winner = next.active.iter().next().unwrap();
            return self.make_terminal_last_standing(state, winner);
        }
        self.advance_to_next_player(&next)
    }

    fn build_check_child(&mut self, state: &BuildState) -> u32 {
        let mut next = state.clone();
        next.acted_since_aggression.insert(state.to_act);

        if is_round_closed(&next) {
            return self.make_showdown_or_chance(&next);
        }
        self.advance_to_next_player(&next)
    }

    fn build_call_child(&mut self, state: &BuildState) -> u32 {
        let seat = state.to_act;
        let idx = seat.index() as usize;
        let current_max = max_street_bet(state);
        let call_amount = current_max - state.street_bets[idx];

        let mut next = state.clone();
        next.stacks[idx] -= call_amount;
        next.street_bets[idx] = current_max;
        next.contributions[idx] += call_amount;
        next.pot += call_amount;
        next.acted_since_aggression.insert(seat);

        if next.stacks[idx] < Chips(SIZE_EPSILON) {
            next.all_in.insert(seat);
        }
        if is_round_closed(&next) {
            return self.make_showdown_or_chance(&next);
        }
        self.advance_to_next_player(&next)
    }

    fn apply_aggression(&mut self, state: &BuildState, raise_to: Chips) -> u32 {
        let seat = state.to_act;
        let idx = seat.index() as usize;
        let additional = raise_to - state.street_bets[idx];

        let mut next = state.clone();
        next.stacks[idx] -= additional;
        next.street_bets[idx] = raise_to;
        next.contributions[idx] += additional;
        next.pot += additional;
        next.raise_count += 1;
        next.facing_bet = true;
        next.last_raise_to = raise_to;
        next.acted_since_aggression = PlayerSet::empty();
        next.acted_since_aggression.insert(seat);

        self.advance_to_next_player(&next)
    }

    fn build_allin_child(&mut self, state: &BuildState) -> u32 {
        let seat = state.to_act;
        let idx = seat.index() as usize;
        let remaining = state.stacks[idx];
        let raise_to = state.street_bets[idx] + remaining;
        let is_call_allin = state.facing_bet
            && raise_to <= max_street_bet(state) + Chips(SIZE_EPSILON);

        let mut next = state.clone();
        next.stacks[idx] = Chips::ZERO;
        next.street_bets[idx] = raise_to;
        next.contributions[idx] += remaining;
        next.pot += remaining;
        next.all_in.insert(seat);
        next.acted_since_aggression.insert(seat);

        if is_call_allin {
            if is_round_closed(&next) {
                return self.make_showdown_or_chance(&next);
            }
            return self.advance_to_next_player(&next);
        }

        next.raise_count += 1;
        next.facing_bet = true;
        next.last_raise_to = raise_to;
        next.acted_since_aggression = PlayerSet::empty();
        next.acted_since_aggression.insert(seat);

        self.advance_to_next_player(&next)
    }

    fn advance_to_next_player(&mut self, state: &BuildState) -> u32 {
        if let Some(next_seat) = next_active_non_allin(state, state.to_act) {
            let mut next = state.clone();
            next.to_act = next_seat;
            self.build_node(&next)
        } else {
            self.make_showdown_or_chance(state)
        }
    }

    fn make_showdown_or_chance(&mut self, state: &BuildState) -> u32 {
        let should_runout = count_active_non_allin(state) <= 1;

        match state.street.next() {
            Some(next_street) if !should_runout => {
                self.make_chance_node(state, next_street)
            }
            Some(next_street) if should_runout => {
                self.make_runout_chain(state, next_street)
            }
            _ => self.make_terminal_showdown(state),
        }
    }

    fn make_chance_node(&mut self, state: &BuildState, next_street: Street) -> u32 {
        let chance_idx = self.push_placeholder();
        let next = new_street_state(state, next_street);
        let child = self.build_node(&next);

        self.nodes[chance_idx as usize] = MpGameNode::Chance { next_street, child };
        chance_idx
    }

    fn make_runout_chain(&mut self, state: &BuildState, next_street: Street) -> u32 {
        let chance_idx = self.push_placeholder();
        let child = match next_street.next() {
            Some(further) => self.make_runout_chain(state, further),
            None => self.make_terminal_showdown(state),
        };

        self.nodes[chance_idx as usize] = MpGameNode::Chance { next_street, child };
        chance_idx
    }

    fn make_terminal_last_standing(&mut self, state: &BuildState, winner: Seat) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(MpGameNode::Terminal {
            kind: TerminalKind::LastStanding { winner },
            pot: state.pot,
            contributions: state.contributions,
        });
        idx
    }

    fn make_terminal_showdown(&mut self, state: &BuildState) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(MpGameNode::Terminal {
            kind: TerminalKind::Showdown { active: state.active },
            pot: state.pot,
            contributions: state.contributions,
        });
        idx
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_mp::config::{
        ForcedBet, ForcedBetKind, MpActionAbstractionConfig, MpGameConfig, MpStreetSizes,
    };
    use test_macros::timed_test;

    /// Build a minimal config for testing.
    fn test_config(num_players: u8) -> (MpGameConfig, MpActionAbstractionConfig) {
        let blinds = vec![
            ForcedBet {
                seat: 0,
                kind: ForcedBetKind::SmallBlind,
                amount: 1.0,
            },
            ForcedBet {
                seat: 1,
                kind: ForcedBetKind::BigBlind,
                amount: 2.0,
            },
        ];

        let game = MpGameConfig {
            name: format!("{num_players}-player test"),
            num_players,
            stack_depth: 100.0,
            blinds,
            rake_rate: 0.0,
            rake_cap: 0.0,
        };

        let preflop_sizes = MpStreetSizes {
            lead: vec![serde_yaml::Value::String("5bb".into())],
            raise: vec![vec![serde_yaml::Value::String("3.0x".into())]],
        };
        let postflop_sizes = MpStreetSizes {
            lead: vec![yaml_f64(0.67)],
            raise: vec![vec![yaml_f64(1.0)]],
        };

        let action = MpActionAbstractionConfig {
            preflop: preflop_sizes,
            flop: postflop_sizes.clone(),
            turn: postflop_sizes.clone(),
            river: postflop_sizes,
        };

        (game, action)
    }

    fn yaml_f64(v: f64) -> serde_yaml::Value {
        serde_yaml::Value::Number(serde_yaml::Number::from(v))
    }

    /// Build a minimal config with NO sized bets (fold/check/call/all-in only).
    fn minimal_config(num_players: u8) -> (MpGameConfig, MpActionAbstractionConfig) {
        let blinds = vec![
            ForcedBet {
                seat: 0,
                kind: ForcedBetKind::SmallBlind,
                amount: 1.0,
            },
            ForcedBet {
                seat: 1,
                kind: ForcedBetKind::BigBlind,
                amount: 2.0,
            },
        ];
        let game = MpGameConfig {
            name: format!("{num_players}-player minimal"),
            num_players,
            stack_depth: 20.0,
            blinds,
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        let empty = MpStreetSizes {
            lead: vec![],
            raise: vec![],
        };
        let action = MpActionAbstractionConfig {
            preflop: empty.clone(),
            flop: empty.clone(),
            turn: empty.clone(),
            river: empty,
        };
        (game, action)
    }

    #[timed_test]
    fn build_2_player_has_root_decision() {
        let (game, action) = test_config(2);
        let tree = MpGameTree::build(&game, &action);
        assert!(
            matches!(
                &tree.nodes[tree.root as usize],
                MpGameNode::Decision { seat, street: Street::Preflop, .. }
                if seat.index() == 0
            ),
            "Root should be a Decision node for seat 0 (SB) on preflop"
        );
    }

    #[timed_test]
    fn build_2_player_fold_is_terminal() {
        let (game, action) = test_config(2);
        let tree = MpGameTree::build(&game, &action);
        if let MpGameNode::Decision {
            actions, children, ..
        } = &tree.nodes[tree.root as usize]
        {
            let fold_idx = actions
                .iter()
                .position(|a| matches!(a, TreeAction::Fold))
                .expect("Root should have Fold");
            let child = &tree.nodes[children[fold_idx] as usize];
            assert!(
                matches!(child, MpGameNode::Terminal { kind: TerminalKind::LastStanding { winner }, .. } if winner.index() == 1),
                "Fold in 2-player should produce LastStanding for seat 1"
            );
        } else {
            panic!("Root should be Decision");
        }
    }

    #[timed_test]
    fn build_2_player_check_check_advances_street() {
        let (game, action) = test_config(2);
        let tree = MpGameTree::build(&game, &action);
        // SB calls (limp), BB checks -> should advance to flop via Chance
        let has_chance = tree.nodes.iter().any(|n| {
            matches!(
                n,
                MpGameNode::Chance {
                    next_street: Street::Flop,
                    ..
                }
            )
        });
        assert!(has_chance, "Two checks should produce a Chance node to flop");
    }

    #[timed_test]
    fn build_3_player_fold_continues_game() {
        let (game, action) = test_config(3);
        let tree = MpGameTree::build(&game, &action);
        // Root is seat 2 (left of BB). Fold should NOT be terminal (2 remain).
        if let MpGameNode::Decision {
            actions, children, ..
        } = &tree.nodes[tree.root as usize]
        {
            let fold_idx = actions
                .iter()
                .position(|a| matches!(a, TreeAction::Fold))
                .expect("Root should have Fold");
            let child = &tree.nodes[children[fold_idx] as usize];
            assert!(
                matches!(child, MpGameNode::Decision { .. }),
                "First fold in 3-player should lead to another Decision, got {child:?}"
            );
        } else {
            panic!("Root should be Decision");
        }
    }

    #[timed_test]
    fn build_3_player_two_folds_is_terminal() {
        let (game, action) = test_config(3);
        let tree = MpGameTree::build(&game, &action);
        // Seat 2 folds -> Seat 0 folds -> terminal (seat 1 wins)
        let root = &tree.nodes[tree.root as usize];
        if let MpGameNode::Decision {
            actions, children, ..
        } = root
        {
            let fold_idx = actions
                .iter()
                .position(|a| matches!(a, TreeAction::Fold))
                .unwrap();
            let second = &tree.nodes[children[fold_idx] as usize];
            if let MpGameNode::Decision {
                actions: a2,
                children: c2,
                ..
            } = second
            {
                let fold_idx2 = a2
                    .iter()
                    .position(|a| matches!(a, TreeAction::Fold))
                    .unwrap();
                let terminal = &tree.nodes[c2[fold_idx2] as usize];
                assert!(
                    matches!(terminal, MpGameNode::Terminal { kind: TerminalKind::LastStanding { .. }, .. }),
                    "Two folds in 3-player should be LastStanding terminal"
                );
            } else {
                panic!("Second node should be Decision");
            }
        } else {
            panic!("Root should be Decision");
        }
    }

    #[timed_test]
    fn build_2_player_call_preflop_goes_to_flop() {
        let (game, action) = test_config(2);
        let tree = MpGameTree::build(&game, &action);
        // In 2-player: SB calls (limp), BB checks -> Chance to flop
        // Find call action at root
        if let MpGameNode::Decision {
            actions, children, ..
        } = &tree.nodes[tree.root as usize]
        {
            let call_idx = actions
                .iter()
                .position(|a| matches!(a, TreeAction::Call))
                .expect("SB should have Call");
            // BB gets to act (check or raise)
            let bb_node = &tree.nodes[children[call_idx] as usize];
            if let MpGameNode::Decision {
                actions: bb_actions,
                children: bb_children,
                ..
            } = bb_node
            {
                let check_idx = bb_actions
                    .iter()
                    .position(|a| matches!(a, TreeAction::Check))
                    .expect("BB should have Check");
                let after_check = &tree.nodes[bb_children[check_idx] as usize];
                assert!(
                    matches!(after_check, MpGameNode::Chance { next_street: Street::Flop, .. }),
                    "SB call + BB check should advance to flop"
                );
            } else {
                panic!("BB response should be Decision");
            }
        } else {
            panic!("Root should be Decision");
        }
    }

    #[timed_test]
    fn build_2_player_tree_has_all_node_types() {
        let (game, action) = test_config(2);
        let tree = MpGameTree::build(&game, &action);
        let has_decision = tree
            .nodes
            .iter()
            .any(|n| matches!(n, MpGameNode::Decision { .. }));
        let has_chance = tree
            .nodes
            .iter()
            .any(|n| matches!(n, MpGameNode::Chance { .. }));
        let has_terminal = tree
            .nodes
            .iter()
            .any(|n| matches!(n, MpGameNode::Terminal { .. }));
        assert!(has_decision, "Tree should have Decision nodes");
        assert!(has_chance, "Tree should have Chance nodes");
        assert!(has_terminal, "Tree should have Terminal nodes");
    }

    #[timed_test]
    fn build_2_player_tree_decision_count() {
        let (game, action) = test_config(2);
        let tree = MpGameTree::build(&game, &action);
        let decision_count = tree
            .nodes
            .iter()
            .filter(|n| matches!(n, MpGameNode::Decision { .. }))
            .count();
        assert!(
            decision_count > 10,
            "2-player tree should have many decision nodes, got {decision_count}"
        );
        assert!(
            decision_count < 100_000,
            "2-player tree should not be too large, got {decision_count}"
        );
    }

    #[timed_test]
    fn build_6_player_no_panic() {
        let (game, action) = minimal_config(6);
        let tree = MpGameTree::build(&game, &action);
        assert!(tree.nodes.len() > 1, "6-player tree should have nodes");
        assert_eq!(tree.num_players, 6);
    }

    #[timed_test]
    fn build_all_in_deduplication() {
        // Use a stack where a sized bet equals the remaining stack.
        // The all-in should not appear twice.
        let blinds = vec![
            ForcedBet {
                seat: 0,
                kind: ForcedBetKind::SmallBlind,
                amount: 1.0,
            },
            ForcedBet {
                seat: 1,
                kind: ForcedBetKind::BigBlind,
                amount: 2.0,
            },
        ];
        let game = MpGameConfig {
            name: "shallow".into(),
            num_players: 2,
            // Stack = 10 chips. SB posts 1, has 9 left.
            // 5bb raise = 10 chips = all-in.
            stack_depth: 10.0,
            blinds,
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        let preflop_sizes = MpStreetSizes {
            lead: vec![serde_yaml::Value::String("5bb".into())],
            raise: vec![vec![serde_yaml::Value::String("3.0x".into())]],
        };
        let postflop_sizes = MpStreetSizes {
            lead: vec![yaml_f64(0.67)],
            raise: vec![vec![yaml_f64(1.0)]],
        };
        let action_cfg = MpActionAbstractionConfig {
            preflop: preflop_sizes,
            flop: postflop_sizes.clone(),
            turn: postflop_sizes.clone(),
            river: postflop_sizes,
        };
        let tree = MpGameTree::build(&game, &action_cfg);
        // Check root: should have exactly one AllIn action
        if let MpGameNode::Decision { actions, .. } = &tree.nodes[tree.root as usize] {
            let allin_count = actions
                .iter()
                .filter(|a| matches!(a, TreeAction::AllIn))
                .count();
            assert_eq!(
                allin_count, 1,
                "Should have exactly one AllIn action, got {allin_count}. Actions: {actions:?}"
            );
        }
    }

    #[timed_test]
    fn showdown_terminal_has_active_set() {
        let (game, action) = test_config(2);
        let tree = MpGameTree::build(&game, &action);
        let showdown = tree.nodes.iter().find(|n| {
            matches!(
                n,
                MpGameNode::Terminal {
                    kind: TerminalKind::Showdown { .. },
                    ..
                }
            )
        });
        assert!(showdown.is_some(), "Tree should have showdown terminals");
        if let Some(MpGameNode::Terminal {
            kind: TerminalKind::Showdown { active },
            ..
        }) = showdown
        {
            assert!(
                active.count() >= 2,
                "Showdown should have at least 2 active players, got {}",
                active.count()
            );
        }
    }

    #[timed_test]
    fn contributions_sum_to_pot_at_terminal() {
        let (game, action) = test_config(2);
        let tree = MpGameTree::build(&game, &action);
        for (i, node) in tree.nodes.iter().enumerate() {
            if let MpGameNode::Terminal {
                pot, contributions, ..
            } = node
            {
                let sum: Chips = contributions
                    .iter()
                    .take(tree.num_players as usize)
                    .copied()
                    .sum();
                assert!(
                    (sum.0 - pot.0).abs() < SIZE_EPSILON,
                    "Node {i}: contributions sum {sum:?} != pot {pot:?}"
                );
            }
        }
    }

    // ── Tests for refactored helpers ────────────────────────────────

    #[timed_test]
    fn get_depth_or_last_returns_idx_when_in_range() {
        let depths = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(get_depth_or_last(&depths, 0), &[1.0, 2.0]);
        assert_eq!(get_depth_or_last(&depths, 1), &[3.0, 4.0]);
    }

    #[timed_test]
    fn get_depth_or_last_falls_back_to_last() {
        let depths = vec![vec![1.0], vec![2.0, 3.0]];
        assert_eq!(get_depth_or_last(&depths, 5), &[2.0, 3.0]);
        assert_eq!(get_depth_or_last(&depths, 99), &[2.0, 3.0]);
    }

    #[timed_test]
    fn get_depth_or_last_returns_empty_when_no_depths() {
        let depths: Vec<Vec<f64>> = vec![];
        let result: &[f64] = get_depth_or_last(&depths, 0);
        assert!(result.is_empty());
    }

    #[timed_test]
    fn is_size_duplicate_detects_lead_match() {
        let actions = vec![TreeAction::Lead(10.0)];
        assert!(is_size_duplicate(&actions, Chips(10.005)));
        assert!(!is_size_duplicate(&actions, Chips(11.0)));
    }

    #[timed_test]
    fn is_size_duplicate_detects_raise_match() {
        let actions = vec![TreeAction::Raise(20.0)];
        assert!(is_size_duplicate(&actions, Chips(20.005)));
        assert!(!is_size_duplicate(&actions, Chips(21.0)));
    }

    #[timed_test]
    fn is_size_duplicate_ignores_non_sized_actions() {
        let actions = vec![TreeAction::Fold, TreeAction::Check, TreeAction::AllIn];
        assert!(!is_size_duplicate(&actions, Chips(10.0)));
    }

    #[timed_test]
    fn is_size_duplicate_empty_actions() {
        assert!(!is_size_duplicate(&[], Chips(10.0)));
    }

    #[timed_test]
    fn find_preflop_first_to_act_no_straddle() {
        // 3 players, BB at seat 1, no straddle -> first to act is seat 2
        let seat = find_preflop_first_to_act(3, 1, None);
        assert_eq!(seat.index(), 2);
    }

    #[timed_test]
    fn find_preflop_first_to_act_with_straddle() {
        // 4 players, BB at seat 1, straddle at seat 2 -> first to act is seat 3
        let seat = find_preflop_first_to_act(4, 1, Some(2));
        assert_eq!(seat.index(), 3);
    }

    #[timed_test]
    fn find_preflop_first_to_act_wraps_around() {
        // 3 players, BB at seat 2, no straddle -> first to act wraps to seat 0
        let seat = find_preflop_first_to_act(3, 2, None);
        assert_eq!(seat.index(), 0);
    }

    #[timed_test]
    fn build_state_apply_forced_bet_updates_all_fields() {
        let mut state = BuildState {
            stacks: [Chips(100.0); MAX_PLAYERS],
            street_bets: [Chips::ZERO; MAX_PLAYERS],
            contributions: [Chips::ZERO; MAX_PLAYERS],
            active: PlayerSet::all(2),
            all_in: PlayerSet::empty(),
            acted_since_aggression: PlayerSet::empty(),
            street: Street::Preflop,
            pot: Chips::ZERO,
            num_players: 2,
            raise_count: 0,
            to_act: Seat::from_raw(0),
            facing_bet: false,
            last_raise_to: Chips::ZERO,
            dealer: 0,
            big_blind_amount: Chips(2.0),
        };

        state.apply_forced_bet(0, Chips(1.0));
        assert_eq!(state.stacks[0], Chips(99.0));
        assert_eq!(state.street_bets[0], Chips(1.0));
        assert_eq!(state.contributions[0], Chips(1.0));
        assert_eq!(state.pot, Chips(1.0));

        state.apply_forced_bet(1, Chips(2.0));
        assert_eq!(state.stacks[1], Chips(98.0));
        assert_eq!(state.street_bets[1], Chips(2.0));
        assert_eq!(state.contributions[1], Chips(2.0));
        assert_eq!(state.pot, Chips(3.0));
    }

    #[timed_test]
    fn try_add_sized_action_skips_allin_equivalent() {
        let state = BuildState {
            stacks: [Chips(10.0), Chips(100.0), Chips::ZERO, Chips::ZERO,
                     Chips::ZERO, Chips::ZERO, Chips::ZERO, Chips::ZERO],
            street_bets: [Chips::ZERO; MAX_PLAYERS],
            contributions: [Chips::ZERO; MAX_PLAYERS],
            active: PlayerSet::all(2),
            all_in: PlayerSet::empty(),
            acted_since_aggression: PlayerSet::empty(),
            street: Street::Flop,
            pot: Chips(20.0),
            num_players: 2,
            raise_count: 0,
            to_act: Seat::from_raw(0),
            facing_bet: false,
            last_raise_to: Chips::ZERO,
            dealer: 0,
            big_blind_amount: Chips(2.0),
        };

        let mut actions = Vec::new();
        // raise_to of 10.0 = all-in for seat 0 (stack=10), should be skipped
        try_add_sized_action(&state, &mut actions, Chips(10.0), TreeAction::Lead);
        assert!(actions.is_empty(), "Should skip all-in equivalent size");
    }

    #[timed_test]
    fn try_add_sized_action_adds_valid_size() {
        let state = BuildState {
            stacks: [Chips(100.0), Chips(100.0), Chips::ZERO, Chips::ZERO,
                     Chips::ZERO, Chips::ZERO, Chips::ZERO, Chips::ZERO],
            street_bets: [Chips::ZERO; MAX_PLAYERS],
            contributions: [Chips::ZERO; MAX_PLAYERS],
            active: PlayerSet::all(2),
            all_in: PlayerSet::empty(),
            acted_since_aggression: PlayerSet::empty(),
            street: Street::Flop,
            pot: Chips(20.0),
            num_players: 2,
            raise_count: 0,
            to_act: Seat::from_raw(0),
            facing_bet: false,
            last_raise_to: Chips::ZERO,
            dealer: 0,
            big_blind_amount: Chips(2.0),
        };

        let mut actions = Vec::new();
        try_add_sized_action(&state, &mut actions, Chips(15.0), TreeAction::Lead);
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], TreeAction::Lead(v) if (v - 15.0).abs() < 0.01));
    }

    #[timed_test]
    fn try_add_sized_action_skips_duplicate() {
        let state = BuildState {
            stacks: [Chips(100.0), Chips(100.0), Chips::ZERO, Chips::ZERO,
                     Chips::ZERO, Chips::ZERO, Chips::ZERO, Chips::ZERO],
            street_bets: [Chips::ZERO; MAX_PLAYERS],
            contributions: [Chips::ZERO; MAX_PLAYERS],
            active: PlayerSet::all(2),
            all_in: PlayerSet::empty(),
            acted_since_aggression: PlayerSet::empty(),
            street: Street::Flop,
            pot: Chips(20.0),
            num_players: 2,
            raise_count: 0,
            to_act: Seat::from_raw(0),
            facing_bet: false,
            last_raise_to: Chips::ZERO,
            dealer: 0,
            big_blind_amount: Chips(2.0),
        };

        let mut actions = vec![TreeAction::Lead(15.0)];
        try_add_sized_action(&state, &mut actions, Chips(15.005), TreeAction::Lead);
        assert_eq!(actions.len(), 1, "Should skip near-duplicate size");
    }

    #[timed_test]
    fn straddle_config_builds_tree_without_panic() {
        let blinds = vec![
            ForcedBet { seat: 0, kind: ForcedBetKind::SmallBlind, amount: 1.0 },
            ForcedBet { seat: 1, kind: ForcedBetKind::BigBlind, amount: 2.0 },
            ForcedBet { seat: 2, kind: ForcedBetKind::Straddle, amount: 4.0 },
        ];
        let game = MpGameConfig {
            name: "straddle-test".into(),
            num_players: 4,
            stack_depth: 100.0,
            blinds,
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        let empty = MpStreetSizes { lead: vec![], raise: vec![] };
        let action = MpActionAbstractionConfig {
            preflop: empty.clone(),
            flop: empty.clone(),
            turn: empty.clone(),
            river: empty,
        };
        let tree = MpGameTree::build(&game, &action);
        // With straddle at seat 2, first to act should be seat 3
        if let MpGameNode::Decision { seat, .. } = &tree.nodes[tree.root as usize] {
            assert_eq!(seat.index(), 3, "First to act with straddle should be seat 3");
        } else {
            panic!("Root should be Decision");
        }
    }
}
