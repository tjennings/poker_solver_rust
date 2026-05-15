//! Lazy public-state MCCFR traversal for multiplayer blueprints.
//!
//! This module mirrors the eager [`super::game_tree`] betting semantics without
//! materializing the full public tree. It is the bridge between dynamic action
//! generation and [`super::sparse_storage::SparseMpStorage`].

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_arguments
)]

use rand::Rng;
use std::sync::atomic::{AtomicU64, Ordering};

use super::MAX_PLAYERS;
use super::config::{ForcedBetKind, MpActionAbstractionConfig, MpGameConfig};
use super::game_tree::{TerminalKind, TreeAction};
use super::mccfr::{PruneStats, terminal_value};
use super::sparse_storage::{MpInfosetKey, SparseMpStorage};
use super::storage::{REGRET_SCALE, STRATEGY_SCALE};
use super::types::{Chips, DealWithBuckets, PlayerSet, Seat, Street};

const SIZE_EPSILON: f64 = 0.01;
const MAX_ACTIONS: usize = 16;
const ACTION_BITS: u16 = 4;
const PACKED_ACTION_SLOTS: u16 = 32;
pub const LAZY_ACTION_STREET_COUNT: usize = 4;

static ACTION_MAX_RAISE_COUNT: [AtomicU64; LAZY_ACTION_STREET_COUNT] =
    [const { AtomicU64::new(0) }; LAZY_ACTION_STREET_COUNT];
static ACTION_OVER_CONFIG_DECISIONS: [AtomicU64; LAZY_ACTION_STREET_COUNT] =
    [const { AtomicU64::new(0) }; LAZY_ACTION_STREET_COUNT];
static ACTION_OVER_CONFIG_AGGRESSIONS: [AtomicU64; LAZY_ACTION_STREET_COUNT] =
    [const { AtomicU64::new(0) }; LAZY_ACTION_STREET_COUNT];
static ACTION_ALL_IN_AGGRESSIONS: [AtomicU64; LAZY_ACTION_STREET_COUNT] =
    [const { AtomicU64::new(0) }; LAZY_ACTION_STREET_COUNT];

/// Hysteresis settings for negative-action subtree gating during lazy traversal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NegativeActionTraversalConfig {
    pub enabled: bool,
    pub prune_below: i32,
    pub reactivate_at: i32,
}

#[derive(Debug, Clone, Copy)]
enum PreflopSize {
    Absolute(f64),
    Multiplier(f64),
}

impl PreflopSize {
    fn parse(value: &str) -> Self {
        let trimmed = value.trim();
        if let Some(stripped) = trimmed.strip_suffix("bb") {
            Self::Absolute(stripped.parse().expect("invalid absolute preflop size"))
        } else if let Some(stripped) = trimmed.strip_suffix('x') {
            Self::Multiplier(stripped.parse().expect("invalid multiplier preflop size"))
        } else {
            panic!("preflop size must end with 'bb' or 'x': {trimmed}");
        }
    }
}

struct LazyActionConfig {
    max_flop_players: Option<u8>,
    allow_preflop_limp: bool,
    preflop_lead: Vec<PreflopSize>,
    preflop_raise: Vec<Vec<PreflopSize>>,
    flop_lead: Vec<f64>,
    flop_raise: Vec<Vec<f64>>,
    turn_lead: Vec<f64>,
    turn_raise: Vec<Vec<f64>>,
    river_lead: Vec<f64>,
    river_raise: Vec<Vec<f64>>,
}

impl LazyActionConfig {
    fn from_action_config(config: &MpActionAbstractionConfig) -> Self {
        Self {
            max_flop_players: config.max_flop_players,
            allow_preflop_limp: true,
            preflop_lead: parse_preflop_values(&config.preflop.lead),
            preflop_raise: parse_preflop_raise_depths(&config.preflop.raise),
            flop_lead: parse_f64_values(&config.flop.lead),
            flop_raise: parse_f64_raise_depths(&config.flop.raise),
            turn_lead: parse_f64_values(&config.turn.lead),
            turn_raise: parse_f64_raise_depths(&config.turn.raise),
            river_lead: parse_f64_values(&config.river.lead),
            river_raise: parse_f64_raise_depths(&config.river.raise),
        }
    }

    fn from_configs(game: &MpGameConfig, config: &MpActionAbstractionConfig) -> Self {
        let mut actions = Self::from_action_config(config);
        actions.allow_preflop_limp = game.allow_preflop_limp;
        actions
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

/// Lazy action abstraction audit counters accumulated between heartbeat reads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LazyActionLimitSnapshot {
    pub max_raise_count: [u64; LAZY_ACTION_STREET_COUNT],
    pub over_config_decisions: [u64; LAZY_ACTION_STREET_COUNT],
    pub over_config_aggressions: [u64; LAZY_ACTION_STREET_COUNT],
    pub all_in_aggressions: [u64; LAZY_ACTION_STREET_COUNT],
}

/// Read and reset lazy action abstraction audit counters.
#[must_use]
pub fn take_lazy_action_limit_snapshot() -> LazyActionLimitSnapshot {
    LazyActionLimitSnapshot {
        max_raise_count: take_atomic_array(&ACTION_MAX_RAISE_COUNT),
        over_config_decisions: take_atomic_array(&ACTION_OVER_CONFIG_DECISIONS),
        over_config_aggressions: take_atomic_array(&ACTION_OVER_CONFIG_AGGRESSIONS),
        all_in_aggressions: take_atomic_array(&ACTION_ALL_IN_AGGRESSIONS),
    }
}

/// Lazy public game model used by sparse traversal.
pub struct LazyMpGame {
    actions: LazyActionConfig,
    root: LazyPublicState,
    pub num_players: u8,
    pub starting_stack: Chips,
}

impl LazyMpGame {
    /// Build a lazy public game model without materializing a tree.
    #[must_use]
    pub fn new(game: &MpGameConfig, action_config: &MpActionAbstractionConfig) -> Self {
        let stack = Chips(game.stack_depth);
        Self {
            actions: LazyActionConfig::from_configs(game, action_config),
            root: init_public_state(game, stack),
            num_players: game.num_players,
            starting_stack: stack,
        }
    }

    /// Return the root public state.
    #[must_use]
    pub const fn root_state(&self) -> LazyPublicState {
        self.root
    }

    /// Generate legal actions for an already-normalized decision state.
    #[must_use]
    pub fn actions(&self, state: &LazyPublicState) -> Vec<TreeAction> {
        generate_actions(&self.actions, state)
    }
}

/// Compact public betting state carried by lazy traversal.
#[derive(Clone, Copy)]
pub struct LazyPublicState {
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

impl LazyPublicState {
    /// Current acting seat.
    #[must_use]
    pub const fn to_act(self) -> Seat {
        self.to_act
    }

    /// Current street.
    #[must_use]
    pub const fn street(self) -> Street {
        self.street
    }

    /// Current pot in chips.
    #[must_use]
    pub const fn pot(self) -> Chips {
        self.pot
    }

    /// Number of raises/opening bets made on this street.
    #[must_use]
    pub const fn raise_count(self) -> u8 {
        self.raise_count
    }
}

#[derive(Clone, Copy, Default)]
struct LazyHistory {
    hi: u64,
    lo: u64,
    hash: u64,
    len: u16,
}

impl LazyHistory {
    fn append(self, action_idx: usize) -> Self {
        let code = (action_idx as u64) & 0xF;
        let mut next = self;
        if next.len < PACKED_ACTION_SLOTS {
            let bit = u32::from(next.len * ACTION_BITS);
            if bit < 64 {
                next.lo |= code << bit;
            } else {
                next.hi |= code << (bit - 64);
            }
        }
        next.hash = mix_history(next.hash, action_idx as u64);
        next.len = next.len.saturating_add(1);
        next
    }

    fn key(self, state: LazyPublicState, bucket: u16) -> MpInfosetKey {
        MpInfosetKey::from_street_bucket(
            state.to_act,
            state.street,
            bucket,
            self.hi,
            self.lo,
            self.hash,
            self.len,
        )
    }
}

/// A lazy public-state cursor resolved from a user-facing spot string.
///
/// This is the lazy backend equivalent of an eager public-node id: it carries
/// the current public state plus the compact action history needed to derive
/// sparse infoset keys for strategy lookup.
#[derive(Clone, Copy)]
pub struct LazyResolvedSpot {
    state: LazyPublicState,
    history: LazyHistory,
}

impl LazyResolvedSpot {
    /// Resolve the lazy root spot for a game.
    #[must_use]
    pub fn root(game: &LazyMpGame) -> Self {
        Self {
            state: game.root_state(),
            history: LazyHistory::default(),
        }
    }

    /// Current acting seat.
    #[must_use]
    pub const fn to_act(self) -> Seat {
        self.state.to_act()
    }

    /// Current street.
    #[must_use]
    pub const fn street(self) -> Street {
        self.state.street()
    }

    /// Generate legal actions at this resolved spot.
    #[must_use]
    pub fn actions(self, game: &LazyMpGame) -> Vec<TreeAction> {
        game.actions(&self.state)
    }

    /// Build the sparse infoset key for a street-local bucket at this spot.
    #[must_use]
    pub fn key_for_bucket(self, bucket: u16) -> MpInfosetKey {
        self.history.key(self.state, bucket)
    }

    /// Advance this resolved spot by one action index.
    #[must_use]
    pub fn advance(self, game: &LazyMpGame, action_idx: usize) -> Option<Self> {
        let actions = game.actions(&self.state);
        let action = *actions.get(action_idx)?;
        match normalize_node(apply_action(self.state, action)) {
            LazyNode::Decision(state) => Some(Self {
                state,
                history: self.history.append(action_idx),
            }),
            LazyNode::Terminal { .. } => None,
        }
    }
}

enum LazyNode {
    Decision(LazyPublicState),
    Terminal {
        kind: TerminalKind,
        contributions: [Chips; MAX_PLAYERS],
    },
}

/// Traverse the lazy public game with external-sampling MCCFR.
pub fn traverse_external_lazy(
    game: &LazyMpGame,
    storage: &SparseMpStorage,
    deal: &DealWithBuckets,
    traverser: Seat,
    rng: &mut impl Rng,
    rake_rate: f64,
    rake_cap: Chips,
    prune: bool,
    prune_threshold: i32,
    negative_action: NegativeActionTraversalConfig,
) -> (f64, PruneStats) {
    traverse_node(
        game,
        storage,
        deal,
        traverser,
        LazyNode::Decision(game.root_state()),
        LazyHistory::default(),
        rng,
        rake_rate,
        rake_cap,
        prune,
        prune_threshold,
        negative_action,
    )
}

fn traverse_node(
    game: &LazyMpGame,
    storage: &SparseMpStorage,
    deal: &DealWithBuckets,
    traverser: Seat,
    node: LazyNode,
    history: LazyHistory,
    rng: &mut impl Rng,
    rake_rate: f64,
    rake_cap: Chips,
    prune: bool,
    prune_threshold: i32,
    negative_action: NegativeActionTraversalConfig,
) -> (f64, PruneStats) {
    match normalize_node(node) {
        LazyNode::Terminal {
            kind,
            contributions,
        } => (
            terminal_value(
                &kind,
                &contributions,
                deal,
                traverser,
                game.num_players,
                rake_rate,
                rake_cap,
            ),
            PruneStats::default(),
        ),
        LazyNode::Decision(state) => {
            let actions = game.actions(&state);
            let bucket = deal.buckets[state.to_act.index() as usize][state.street.index()].0;
            let key = history.key(state, bucket);
            if state.to_act == traverser {
                traverse_traverser(
                    game,
                    storage,
                    deal,
                    traverser,
                    state,
                    history,
                    key,
                    &actions,
                    rng,
                    rake_rate,
                    rake_cap,
                    prune,
                    prune_threshold,
                    negative_action,
                )
            } else {
                traverse_opponent(
                    game,
                    storage,
                    deal,
                    traverser,
                    state,
                    history,
                    key,
                    &actions,
                    rng,
                    rake_rate,
                    rake_cap,
                    prune,
                    prune_threshold,
                    negative_action,
                )
            }
        }
    }
}

fn traverse_traverser(
    game: &LazyMpGame,
    storage: &SparseMpStorage,
    deal: &DealWithBuckets,
    traverser: Seat,
    state: LazyPublicState,
    history: LazyHistory,
    key: MpInfosetKey,
    actions: &[TreeAction],
    rng: &mut impl Rng,
    rake_rate: f64,
    rake_cap: Chips,
    prune: bool,
    prune_threshold: i32,
    negative_action: NegativeActionTraversalConfig,
) -> (f64, PruneStats) {
    let num_actions = actions.len();
    debug_assert!(num_actions <= MAX_ACTIONS);
    let mut strategy = [0.0; MAX_ACTIONS];
    storage.regret_matched_strategy(key, num_actions, &mut strategy);
    let mut blocked = [false; MAX_ACTIONS];
    for (action_idx, action) in actions.iter().copied().enumerate() {
        let child_history = history.append(action_idx);
        let action_is_aggressive = action_increments_raise_count(&state, action);
        blocked[action_idx] = negative_action_blocks(
            storage,
            negative_action,
            action_is_aggressive,
            key,
            action_idx,
            child_history,
        );
    }
    // A concurrent worker may block an edge after this snapshot. Keep any
    // already-started traversal value rather than replacing it with an
    // artificial zero; physical purge is deferred to the discount boundary.
    let mut eligible_strategy = [0.0; MAX_ACTIONS];
    if !mask_strategy_to_unblocked(
        &strategy[..num_actions],
        &blocked[..num_actions],
        &mut eligible_strategy[..num_actions],
    ) {
        return (0.0, PruneStats::default());
    }

    let mut values = [0.0; MAX_ACTIONS];
    let mut pruned = [false; MAX_ACTIONS];
    let mut node_value = 0.0;
    let mut prune_stats = PruneStats::default();

    for (action_idx, action) in actions.iter().copied().enumerate() {
        let child_history = history.append(action_idx);
        if blocked[action_idx] {
            storage.record_negative_action_blocked_traversal_skip();
            pruned[action_idx] = true;
            continue;
        }
        let child = apply_action(state, action);
        let child_is_terminal = matches!(&child, LazyNode::Terminal { .. });
        if should_prune_lazy(
            storage,
            prune,
            prune_threshold,
            child_is_terminal,
            key,
            action_idx,
            &mut prune_stats,
        ) {
            pruned[action_idx] = true;
            continue;
        }
        let (value, child_stats) = traverse_node(
            game,
            storage,
            deal,
            traverser,
            child,
            child_history,
            rng,
            rake_rate,
            rake_cap,
            prune,
            prune_threshold,
            negative_action,
        );
        values[action_idx] = value;
        node_value += eligible_strategy[action_idx] * value;
        prune_stats.merge(child_stats);
    }

    for (action_idx, value) in values[..num_actions].iter().enumerate() {
        if pruned[action_idx] {
            continue;
        }
        let raw = (value - node_value) * REGRET_SCALE;
        let delta = raw.clamp(f64::from(i32::MIN), f64::from(i32::MAX)).round() as i32;
        storage.add_regret(key, num_actions, action_idx, delta);
        let regret = storage.get_regret(key, action_idx);
        let child_history = history.append(action_idx);
        let action_is_aggressive = action_increments_raise_count(&state, actions[action_idx]);
        let _ = transition_negative_action_gate(
            storage,
            negative_action,
            action_is_aggressive,
            key,
            action_idx,
            regret,
            child_history,
        );
    }
    update_strategy_sums(storage, key, num_actions, &eligible_strategy);

    (node_value, prune_stats)
}

fn traverse_opponent(
    game: &LazyMpGame,
    storage: &SparseMpStorage,
    deal: &DealWithBuckets,
    traverser: Seat,
    state: LazyPublicState,
    history: LazyHistory,
    key: MpInfosetKey,
    actions: &[TreeAction],
    rng: &mut impl Rng,
    rake_rate: f64,
    rake_cap: Chips,
    prune: bool,
    prune_threshold: i32,
    negative_action: NegativeActionTraversalConfig,
) -> (f64, PruneStats) {
    let num_actions = actions.len();
    debug_assert!(num_actions <= MAX_ACTIONS);
    let mut strategy = [0.0; MAX_ACTIONS];
    storage.regret_matched_strategy(key, num_actions, &mut strategy);
    let mut blocked = [false; MAX_ACTIONS];
    for (action_idx, action) in actions.iter().copied().enumerate() {
        let child_history = history.append(action_idx);
        let action_is_aggressive = action_increments_raise_count(&state, action);
        blocked[action_idx] = negative_action_blocks(
            storage,
            negative_action,
            action_is_aggressive,
            key,
            action_idx,
            child_history,
        );
    }
    // See traverser-side note: the eligibility snapshot gates descent, but a
    // racing block after descent must not bias the sampled value to zero.
    let mut eligible_strategy = [0.0; MAX_ACTIONS];
    if !mask_strategy_to_unblocked(
        &strategy[..num_actions],
        &blocked[..num_actions],
        &mut eligible_strategy[..num_actions],
    ) {
        return (0.0, PruneStats::default());
    }
    let sampled = sample_action(&eligible_strategy[..num_actions], rng);

    update_strategy_sums(storage, key, num_actions, &eligible_strategy);

    let child_history = history.append(sampled);
    // Opponent nodes only attempt the sampled child. Use a read-only current
    // edge check here so telemetry cannot count blocked non-sampled actions or
    // perturb traversal blocking with a second gate transition.
    if negative_action.enabled
        && action_increments_raise_count(&state, actions[sampled])
        && storage.is_negative_action_edge_blocked(key, sampled)
    {
        storage.record_negative_action_blocked_traversal_skip();
    }
    let (value, stats) = traverse_node(
        game,
        storage,
        deal,
        traverser,
        apply_action(state, actions[sampled]),
        child_history,
        rng,
        rake_rate,
        rake_cap,
        prune,
        prune_threshold,
        negative_action,
    );
    (value, stats)
}

fn update_strategy_sums(
    storage: &SparseMpStorage,
    key: MpInfosetKey,
    num_actions: usize,
    strategy: &[f64; MAX_ACTIONS],
) {
    for (action_idx, prob) in strategy[..num_actions].iter().enumerate() {
        let raw = prob * STRATEGY_SCALE;
        let delta = raw.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32;
        storage.add_strategy_sum(key, num_actions, action_idx, delta);
    }
}

fn negative_action_blocks(
    storage: &SparseMpStorage,
    config: NegativeActionTraversalConfig,
    action_is_aggressive: bool,
    key: MpInfosetKey,
    action: usize,
    child_history: LazyHistory,
) -> bool {
    if !config.enabled || !action_is_aggressive {
        return false;
    }
    let regret = storage.get_regret(key, action);
    transition_negative_action_gate(
        storage,
        config,
        action_is_aggressive,
        key,
        action,
        regret,
        child_history,
    )
    .blocked
}

fn transition_negative_action_gate(
    storage: &SparseMpStorage,
    config: NegativeActionTraversalConfig,
    action_is_aggressive: bool,
    key: MpInfosetKey,
    action: usize,
    regret: i32,
    child_history: LazyHistory,
) -> super::sparse_storage::NegativeActionGateResult {
    if !config.enabled || !action_is_aggressive {
        return super::sparse_storage::NegativeActionGateResult::default();
    }
    storage.transition_negative_action_edge(
        key,
        action,
        regret,
        config.prune_below,
        config.reactivate_at,
        (child_history.hi, child_history.lo, child_history.len),
    )
}

fn should_prune_lazy(
    storage: &SparseMpStorage,
    prune: bool,
    prune_threshold: i32,
    child_is_terminal: bool,
    key: MpInfosetKey,
    action: usize,
    stats: &mut PruneStats,
) -> bool {
    if !prune || child_is_terminal {
        return false;
    }
    stats.total += 1;
    if storage.get_regret(key, action) < prune_threshold {
        stats.hits += 1;
        return true;
    }
    false
}

fn sample_action(strategy: &[f64], rng: &mut impl Rng) -> usize {
    let roll: f64 = rng.random();
    let mut cumulative = 0.0;
    for (idx, probability) in strategy.iter().enumerate() {
        cumulative += probability;
        if roll < cumulative {
            return idx;
        }
    }
    strategy.len() - 1
}

fn mask_strategy_to_unblocked(strategy: &[f64], blocked: &[bool], out: &mut [f64]) -> bool {
    debug_assert_eq!(strategy.len(), blocked.len());
    debug_assert_eq!(strategy.len(), out.len());
    out.fill(0.0);
    let total: f64 = strategy
        .iter()
        .zip(blocked)
        .filter_map(|(probability, is_blocked)| (!is_blocked).then_some(*probability))
        .sum();
    let eligible = blocked.iter().filter(|is_blocked| !**is_blocked).count();
    if eligible == 0 {
        return false;
    }
    if total > 0.0 {
        for (idx, (probability, is_blocked)) in strategy.iter().zip(blocked).enumerate() {
            if !*is_blocked {
                out[idx] = probability / total;
            }
        }
    } else {
        let uniform = 1.0 / eligible as f64;
        for (idx, is_blocked) in blocked.iter().enumerate() {
            if !*is_blocked {
                out[idx] = uniform;
            }
        }
    }
    true
}

fn normalize_node(node: LazyNode) -> LazyNode {
    let LazyNode::Decision(mut state) = node else {
        return node;
    };
    loop {
        if state.active.count() == 1 {
            let winner = state.active.iter().next().unwrap();
            return LazyNode::Terminal {
                kind: TerminalKind::LastStanding { winner },
                contributions: state.contributions,
            };
        }
        let seat = state.to_act;
        let remaining = state.stacks[seat.index() as usize];
        if remaining < Chips(SIZE_EPSILON) && state.active.contains(seat) {
            if let Some(next) = next_active_non_allin(&state, seat) {
                state.to_act = next;
                continue;
            }
            return showdown_or_next_street(state);
        }
        if is_bb_facing_limps(&state, seat) {
            state.facing_bet = false;
        }
        return LazyNode::Decision(state);
    }
}

fn apply_action(state: LazyPublicState, action: TreeAction) -> LazyNode {
    match action {
        TreeAction::Fold => fold_child(state),
        TreeAction::Check => check_child(state),
        TreeAction::Call => call_child(state),
        TreeAction::Lead(amount) | TreeAction::Raise(amount) => {
            aggression_child(state, Chips(amount))
        }
        TreeAction::AllIn => all_in_child(state),
    }
}

fn fold_child(state: LazyPublicState) -> LazyNode {
    let mut next = state;
    next.active.remove(state.to_act);

    if next.active.count() == 1 {
        let winner = next.active.iter().next().unwrap();
        return LazyNode::Terminal {
            kind: TerminalKind::LastStanding { winner },
            contributions: state.contributions,
        };
    }
    advance_to_next_player(next)
}

fn check_child(state: LazyPublicState) -> LazyNode {
    let mut next = state;
    next.acted_since_aggression.insert(state.to_act);

    if is_round_closed(&next) {
        return showdown_or_next_street(next);
    }
    advance_to_next_player(next)
}

fn call_child(state: LazyPublicState) -> LazyNode {
    let seat = state.to_act;
    let idx = seat.index() as usize;
    let current_max = max_street_bet(&state);
    let call_amount = current_max - state.street_bets[idx];

    let mut next = state;
    next.stacks[idx] -= call_amount;
    next.street_bets[idx] = current_max;
    next.contributions[idx] += call_amount;
    next.pot += call_amount;
    next.acted_since_aggression.insert(seat);

    if next.stacks[idx] < Chips(SIZE_EPSILON) {
        next.all_in.insert(seat);
    }
    if is_round_closed(&next) {
        return showdown_or_next_street(next);
    }
    advance_to_next_player(next)
}

fn aggression_child(state: LazyPublicState, raise_to: Chips) -> LazyNode {
    let seat = state.to_act;
    let idx = seat.index() as usize;
    let additional = raise_to - state.street_bets[idx];

    let mut next = state;
    next.stacks[idx] -= additional;
    next.street_bets[idx] = raise_to;
    next.contributions[idx] += additional;
    next.pot += additional;
    next.raise_count += 1;
    next.facing_bet = true;
    next.last_raise_to = raise_to;
    next.acted_since_aggression = PlayerSet::empty();
    next.acted_since_aggression.insert(seat);

    advance_to_next_player(next)
}

fn all_in_child(state: LazyPublicState) -> LazyNode {
    let seat = state.to_act;
    let idx = seat.index() as usize;
    let remaining = state.stacks[idx];
    let raise_to = state.street_bets[idx] + remaining;
    let is_call_allin =
        state.facing_bet && raise_to <= max_street_bet(&state) + Chips(SIZE_EPSILON);

    let mut next = state;
    next.stacks[idx] = Chips::ZERO;
    next.street_bets[idx] = raise_to;
    next.contributions[idx] += remaining;
    next.pot += remaining;
    next.all_in.insert(seat);
    next.acted_since_aggression.insert(seat);

    if is_call_allin {
        if is_round_closed(&next) {
            return showdown_or_next_street(next);
        }
        return advance_to_next_player(next);
    }

    next.raise_count += 1;
    next.facing_bet = true;
    next.last_raise_to = raise_to;
    next.acted_since_aggression = PlayerSet::empty();
    next.acted_since_aggression.insert(seat);

    advance_to_next_player(next)
}

fn advance_to_next_player(mut state: LazyPublicState) -> LazyNode {
    if let Some(next) = next_active_non_allin(&state, state.to_act) {
        state.to_act = next;
        LazyNode::Decision(state)
    } else {
        showdown_or_next_street(state)
    }
}

fn showdown_or_next_street(state: LazyPublicState) -> LazyNode {
    let should_runout = count_active_non_allin(&state) <= 1;
    match state.street.next() {
        Some(next_street) if !should_runout => {
            LazyNode::Decision(new_street_state(state, next_street))
        }
        Some(_) | None => LazyNode::Terminal {
            kind: TerminalKind::Showdown {
                active: state.active,
            },
            contributions: state.contributions,
        },
    }
}

fn init_public_state(config: &MpGameConfig, stack: Chips) -> LazyPublicState {
    let num_players = config.num_players;
    let mut state = LazyPublicState {
        stacks: [Chips::ZERO; MAX_PLAYERS],
        street_bets: [Chips::ZERO; MAX_PLAYERS],
        contributions: [Chips::ZERO; MAX_PLAYERS],
        active: PlayerSet::all(num_players),
        all_in: PlayerSet::empty(),
        acted_since_aggression: PlayerSet::empty(),
        street: Street::Preflop,
        pot: Chips::ZERO,
        num_players,
        raise_count: 0,
        to_act: Seat::from_raw(0),
        facing_bet: true,
        last_raise_to: Chips(2.0),
        dealer: find_dealer(config),
        big_blind_amount: Chips(2.0),
    };

    for stack_slot in state.stacks.iter_mut().take(num_players as usize) {
        *stack_slot = stack;
    }

    let mut big_blind_seat = 1;
    let mut straddle_seat = None;
    for blind in &config.blinds {
        let amount = Chips(blind.amount);
        state.apply_forced_bet(blind.seat as usize, amount);
        if blind.kind == ForcedBetKind::BigBlind {
            state.big_blind_amount = amount;
            big_blind_seat = blind.seat;
        }
        if blind.kind == ForcedBetKind::Straddle {
            straddle_seat = Some(blind.seat);
        }
    }

    state.last_raise_to = state.big_blind_amount;
    state.to_act = find_preflop_first_to_act(num_players, big_blind_seat, straddle_seat);
    state
}

impl LazyPublicState {
    fn apply_forced_bet(&mut self, seat: usize, amount: Chips) {
        self.stacks[seat] -= amount;
        self.street_bets[seat] += amount;
        self.contributions[seat] += amount;
        self.pot += amount;
    }
}

fn generate_actions(config: &LazyActionConfig, state: &LazyPublicState) -> Vec<TreeAction> {
    let mut actions = Vec::new();
    let is_unopened_preflop = is_unopened_preflop(state);

    if state.facing_bet {
        actions.push(TreeAction::Fold);
    }
    if is_unopened_preflop && state.facing_bet {
        if unopened_preflop_call_allowed(config, state) {
            actions.push(TreeAction::Call);
        }
    } else {
        add_check_or_call(config, state, &mut actions);
    }
    let suppress_new_aggression = suppresses_new_aggression(state);
    if !suppress_new_aggression {
        add_sized_actions(config, state, &mut actions);
    }
    if !is_unopened_preflop
        && !suppress_new_aggression
        && u64::from(state.raise_count) < allowed_raise_count_with_all_in(config, state.street)
    {
        add_all_in_if_needed(state, &mut actions);
    }
    dedup_all_in(&mut actions);
    record_action_limit_audit(config, state, &actions);
    actions
}

fn add_check_or_call(
    config: &LazyActionConfig,
    state: &LazyPublicState,
    actions: &mut Vec<TreeAction>,
) {
    let seat = state.to_act;
    let remaining = state.stacks[seat.index() as usize];

    if state.facing_bet {
        if !preflop_call_allowed(config.max_flop_players, state) {
            return;
        }
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

fn unopened_preflop_call_allowed(config: &LazyActionConfig, state: &LazyPublicState) -> bool {
    let actor = state.to_act.index() as usize;
    let actor_has_blind_posted = state.street_bets[actor] > Chips(SIZE_EPSILON);
    (config.allow_preflop_limp || actor_has_blind_posted)
        && preflop_call_allowed(config.max_flop_players, state)
}

fn add_sized_actions(
    config: &LazyActionConfig,
    state: &LazyPublicState,
    actions: &mut Vec<TreeAction>,
) {
    let depth = state.raise_count as usize;
    let is_preflop_open = is_unopened_preflop(state);

    let remaining = state.stacks[state.to_act.index() as usize];
    if remaining < Chips(SIZE_EPSILON) {
        return;
    }

    if is_preflop_open {
        add_lead_sizes(state, config.lead_sizes(state.street), actions);
    } else if !state.facing_bet {
        if depth >= config.max_raise_depths(state.street) {
            return;
        }
        add_lead_sizes(state, config.lead_sizes(state.street), actions);
    } else {
        let raise_depth = if state.street == Street::Preflop {
            depth.saturating_sub(1)
        } else {
            depth
        };
        if raise_depth >= config.max_raise_depths(state.street) {
            return;
        }
        add_raise_sizes(
            state,
            config.raise_sizes_at_depth(state.street, raise_depth),
            actions,
        );
    }
}

fn suppresses_new_aggression(state: &LazyPublicState) -> bool {
    state.street == Street::River && spr_bucket(*state) == 0
}

fn record_action_limit_audit(
    config: &LazyActionConfig,
    state: &LazyPublicState,
    actions: &[TreeAction],
) {
    let street_idx = state.street as usize;
    let current_count = u64::from(state.raise_count);
    let allowed_count = allowed_raise_count_with_all_in(config, state.street);
    ACTION_MAX_RAISE_COUNT[street_idx].fetch_max(current_count, Ordering::Relaxed);
    if current_count > allowed_count {
        ACTION_OVER_CONFIG_DECISIONS[street_idx].fetch_add(1, Ordering::Relaxed);
    }
    for action in actions {
        if !action_increments_raise_count(state, *action) {
            continue;
        }
        let after_count = current_count.saturating_add(1);
        ACTION_MAX_RAISE_COUNT[street_idx].fetch_max(after_count, Ordering::Relaxed);
        if after_count > allowed_count {
            ACTION_OVER_CONFIG_AGGRESSIONS[street_idx].fetch_add(1, Ordering::Relaxed);
        }
        if matches!(action, TreeAction::AllIn) {
            ACTION_ALL_IN_AGGRESSIONS[street_idx].fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn allowed_raise_count_with_all_in(config: &LazyActionConfig, street: Street) -> u64 {
    let configured = if street == Street::Preflop {
        u64::from(!config.preflop_lead.is_empty()) + config.max_raise_depths(street) as u64
    } else {
        config.max_raise_depths(street) as u64
    };
    configured.saturating_add(1)
}

fn action_increments_raise_count(state: &LazyPublicState, action: TreeAction) -> bool {
    match action {
        TreeAction::Lead(_) | TreeAction::Raise(_) => true,
        TreeAction::AllIn => {
            let idx = state.to_act.index() as usize;
            let raise_to = state.street_bets[idx] + state.stacks[idx];
            !(state.facing_bet && raise_to <= max_street_bet(state) + Chips(SIZE_EPSILON))
        }
        TreeAction::Fold | TreeAction::Check | TreeAction::Call => false,
    }
}

fn add_lead_sizes(state: &LazyPublicState, sizes: LeadSizes<'_>, actions: &mut Vec<TreeAction>) {
    match sizes {
        LeadSizes::Preflop(preflop) => {
            for &size in preflop {
                let raise_to = resolve_preflop_size(state, size).max(min_raise_to(state));
                try_add_sized_action(state, actions, raise_to, TreeAction::Lead);
            }
        }
        LeadSizes::Postflop(fractions) => {
            for &fraction in fractions {
                try_add_lead(state, state.pot * fraction, actions);
            }
        }
    }
}

fn add_raise_sizes(state: &LazyPublicState, sizes: RaiseSizes<'_>, actions: &mut Vec<TreeAction>) {
    match sizes {
        RaiseSizes::Preflop(preflop) => {
            for &size in preflop {
                try_add_raise(state, resolve_preflop_raise_to(state, size), actions);
            }
        }
        RaiseSizes::Postflop(fractions) => {
            for &fraction in fractions {
                try_add_raise(state, compute_postflop_raise_to(state, fraction), actions);
            }
        }
    }
}

fn try_add_lead(state: &LazyPublicState, amount: Chips, actions: &mut Vec<TreeAction>) {
    let my_bet = state.street_bets[state.to_act.index() as usize];
    let raise_to = (my_bet + amount).max(min_raise_to(state));
    try_add_sized_action(state, actions, raise_to, TreeAction::Lead);
}

fn try_add_raise(state: &LazyPublicState, raise_to: Chips, actions: &mut Vec<TreeAction>) {
    try_add_sized_action(state, actions, raise_to, TreeAction::Raise);
}

fn try_add_sized_action(
    state: &LazyPublicState,
    actions: &mut Vec<TreeAction>,
    raise_to: Chips,
    variant: fn(f64) -> TreeAction,
) {
    let idx = state.to_act.index() as usize;
    let remaining = state.stacks[idx];
    let all_in_to = state.street_bets[idx] + remaining;
    let additional = raise_to - state.street_bets[idx];

    if raise_to >= all_in_to - Chips(SIZE_EPSILON) {
        return;
    }
    if additional > remaining + Chips(SIZE_EPSILON) {
        return;
    }
    if is_size_duplicate(actions, raise_to) {
        return;
    }
    actions.push(variant(raise_to.0));
}

fn add_all_in_if_needed(state: &LazyPublicState, actions: &mut Vec<TreeAction>) {
    let seat = state.to_act;
    let remaining = state.stacks[seat.index() as usize];
    if remaining < Chips(SIZE_EPSILON) {
        return;
    }
    let all_in_to = state.street_bets[seat.index() as usize] + remaining;
    if state.facing_bet && all_in_to <= max_street_bet(state) + Chips(SIZE_EPSILON) {
        return;
    }
    let already = actions.iter().any(|action| match action {
        TreeAction::Lead(value) | TreeAction::Raise(value) => {
            (Chips(*value) - all_in_to).0.abs() < SIZE_EPSILON
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
    actions.retain(|action| {
        if matches!(action, TreeAction::AllIn) {
            if seen {
                return false;
            }
            seen = true;
        }
        true
    });
}

fn is_size_duplicate(actions: &[TreeAction], amount: Chips) -> bool {
    actions.iter().any(|action| match action {
        TreeAction::Lead(value) | TreeAction::Raise(value) => {
            (Chips(*value) - amount).0.abs() < SIZE_EPSILON
        }
        _ => false,
    })
}

fn resolve_preflop_size(state: &LazyPublicState, size: PreflopSize) -> Chips {
    match size {
        PreflopSize::Absolute(big_blinds) => state.big_blind_amount * big_blinds,
        PreflopSize::Multiplier(multiplier) => state.last_raise_to * multiplier,
    }
}

fn resolve_preflop_raise_to(state: &LazyPublicState, size: PreflopSize) -> Chips {
    resolve_preflop_size(state, size).max(min_raise_to(state))
}

fn compute_postflop_raise_to(state: &LazyPublicState, fraction: f64) -> Chips {
    let my_bet = state.street_bets[state.to_act.index() as usize];
    let call_amount = max_street_bet(state) - my_bet;
    let pot_after_call = state.pot + call_amount;
    let raise_amount = call_amount + pot_after_call * fraction;
    (my_bet + raise_amount).max(min_raise_to(state))
}

fn max_street_bet(state: &LazyPublicState) -> Chips {
    state.street_bets[..state.num_players as usize]
        .iter()
        .copied()
        .fold(Chips::ZERO, Chips::max)
}

fn preflop_call_allowed(max_flop_players: Option<u8>, state: &LazyPublicState) -> bool {
    let Some(max_flop_players) = max_flop_players else {
        return true;
    };
    if state.street != Street::Preflop || !state.facing_bet {
        return true;
    }

    let current_max = max_street_bet(state);
    let actor = state.to_act;
    let committed_after_call = state
        .active
        .iter()
        .filter(|seat| {
            *seat == actor
                || state.all_in.contains(*seat)
                || (state.street_bets[seat.index() as usize] - current_max)
                    .0
                    .abs()
                    < SIZE_EPSILON
        })
        .count();
    let max_flop_players = usize::from(max_flop_players);

    if preflop_call_closes_action(state, current_max) {
        committed_after_call <= max_flop_players
    } else {
        committed_after_call < max_flop_players
    }
}

fn preflop_call_closes_action(state: &LazyPublicState, current_max: Chips) -> bool {
    for seat in state.active.iter() {
        if seat == state.to_act || state.all_in.contains(seat) {
            continue;
        }
        if !state.acted_since_aggression.contains(seat) {
            return false;
        }
        if (state.street_bets[seat.index() as usize] - current_max)
            .0
            .abs()
            >= SIZE_EPSILON
        {
            return false;
        }
    }
    true
}

fn min_raise_to(state: &LazyPublicState) -> Chips {
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

fn new_street_state(state: LazyPublicState, next_street: Street) -> LazyPublicState {
    let mut next = state;
    next.street = next_street;
    next.street_bets = [Chips::ZERO; MAX_PLAYERS];
    next.raise_count = 0;
    next.facing_bet = false;
    next.last_raise_to = Chips::ZERO;
    next.acted_since_aggression = PlayerSet::empty();
    next.to_act = find_postflop_first_to_act(&next);
    next
}

fn find_postflop_first_to_act(state: &LazyPublicState) -> Seat {
    for offset in 1..=state.num_players {
        let candidate = (state.dealer + offset) % state.num_players;
        let seat = Seat::from_raw(candidate);
        if state.active.contains(seat) && !state.all_in.contains(seat) {
            return seat;
        }
    }
    state.active.iter().next().unwrap_or(Seat::from_raw(0))
}

fn next_active_non_allin(state: &LazyPublicState, after: Seat) -> Option<Seat> {
    for offset in 1..state.num_players {
        let candidate = Seat::from_raw((after.index() + offset) % state.num_players);
        if state.active.contains(candidate) && !state.all_in.contains(candidate) {
            return Some(candidate);
        }
    }
    None
}

fn count_active_non_allin(state: &LazyPublicState) -> u8 {
    state
        .active
        .iter()
        .filter(|seat| !state.all_in.contains(*seat))
        .count() as u8
}

fn is_round_closed(state: &LazyPublicState) -> bool {
    let current_max = max_street_bet(state);
    for seat in state.active.iter() {
        if state.all_in.contains(seat) {
            continue;
        }
        if !state.acted_since_aggression.contains(seat) {
            return false;
        }
        if (state.street_bets[seat.index() as usize] - current_max)
            .0
            .abs()
            >= SIZE_EPSILON
        {
            return false;
        }
    }
    true
}

fn is_unopened_preflop(state: &LazyPublicState) -> bool {
    state.street == Street::Preflop && state.raise_count == 0
}

fn is_bb_facing_limps(state: &LazyPublicState, seat: Seat) -> bool {
    if state.street != Street::Preflop || state.raise_count != 0 || !state.facing_bet {
        return false;
    }
    let my_bet = state.street_bets[seat.index() as usize];
    let current_max = max_street_bet(state);
    (my_bet - current_max).0.abs() < SIZE_EPSILON
}

fn spr_bucket(state: LazyPublicState) -> u8 {
    let active_stack = state
        .active
        .iter()
        .filter(|seat| !state.all_in.contains(*seat))
        .map(|seat| state.stacks[seat.index() as usize].0)
        .filter(|stack| *stack > SIZE_EPSILON)
        .fold(f64::INFINITY, f64::min);
    if !active_stack.is_finite() || state.pot.0 <= SIZE_EPSILON {
        return 31;
    }
    let spr = (active_stack / state.pot.0).max(0.0);
    let bucket = (spr.log2().max(0.0) * 4.0).round().clamp(0.0, 31.0);
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    {
        bucket as u8
    }
}

fn find_preflop_first_to_act(num_players: u8, big_blind_seat: u8, straddle: Option<u8>) -> Seat {
    Seat::new(
        (straddle.unwrap_or(big_blind_seat) + 1) % num_players,
        num_players,
    )
}

fn find_dealer(config: &MpGameConfig) -> u8 {
    if config.num_players == 2 {
        return config
            .blinds
            .iter()
            .find(|blind| blind.kind == ForcedBetKind::SmallBlind)
            .map_or(0, |blind| blind.seat);
    }
    config
        .blinds
        .iter()
        .find(|blind| blind.kind == ForcedBetKind::SmallBlind)
        .map_or(config.num_players - 1, |blind| {
            (blind.seat + config.num_players - 1) % config.num_players
        })
}

fn parse_preflop_values(values: &[serde_yaml::Value]) -> Vec<PreflopSize> {
    values
        .iter()
        .map(|value| PreflopSize::parse(&yaml_to_string(value)))
        .collect()
}

fn parse_preflop_raise_depths(depths: &[Vec<serde_yaml::Value>]) -> Vec<Vec<PreflopSize>> {
    depths
        .iter()
        .map(|depth| parse_preflop_values(depth))
        .collect()
}

fn parse_f64_values(values: &[serde_yaml::Value]) -> Vec<f64> {
    values
        .iter()
        .map(|value| yaml_to_f64(value).expect("expected numeric size"))
        .collect()
}

fn parse_f64_raise_depths(depths: &[Vec<serde_yaml::Value>]) -> Vec<Vec<f64>> {
    depths.iter().map(|depth| parse_f64_values(depth)).collect()
}

fn yaml_to_string(value: &serde_yaml::Value) -> String {
    match value {
        serde_yaml::Value::String(value) => value.clone(),
        serde_yaml::Value::Number(number) => format!("{number}"),
        _ => panic!("unexpected YAML value type for size: {value:?}"),
    }
}

fn yaml_to_f64(value: &serde_yaml::Value) -> Option<f64> {
    match value {
        serde_yaml::Value::Number(number) => number.as_f64(),
        serde_yaml::Value::String(value) => value.parse().ok(),
        _ => None,
    }
}

fn get_depth_or_last<T>(depths: &[Vec<T>], idx: usize) -> &[T] {
    depths
        .get(idx)
        .or_else(|| depths.last())
        .map_or(&[], Vec::as_slice)
}

fn mix_history(hash: u64, value: u64) -> u64 {
    let mixed = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (hash ^ mixed)
        .wrapping_mul(0xBF58_476D_1CE4_E5B9)
        .rotate_left(27)
}

fn take_atomic_array<const N: usize>(atoms: &[AtomicU64; N]) -> [u64; N] {
    std::array::from_fn(|idx| atoms[idx].swap(0, Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use std::sync::{Mutex, MutexGuard};
    use test_macros::timed_test;

    use super::*;
    use crate::blueprint_mp::config::{ForcedBet, MpStreetSizes, MpStreetSizes as StreetSizes};
    use crate::blueprint_mp::mccfr::sample_deal;
    use crate::blueprint_mp::types::{Bucket, Deal};

    static LAZY_ACTION_LIMIT_AUDIT_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn reset_lazy_action_limit_audit_for_test() -> MutexGuard<'static, ()> {
        let guard = LAZY_ACTION_LIMIT_AUDIT_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _ = take_lazy_action_limit_snapshot();
        guard
    }

    fn game_config(num_players: u8, stack_depth: f64) -> MpGameConfig {
        MpGameConfig {
            name: "lazy test".to_string(),
            num_players,
            stack_depth,
            allow_preflop_limp: true,
            blinds: vec![
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
            ],
            rake_rate: 0.0,
            rake_cap: 0.0,
        }
    }

    fn action_config() -> MpActionAbstractionConfig {
        let preflop = MpStreetSizes {
            lead: vec![serde_yaml::Value::String("2bb".to_string())],
            raise: vec![
                vec![serde_yaml::Value::String("2.0x".to_string())],
                vec![serde_yaml::Value::String("2.0x".to_string())],
            ],
        };
        let postflop = StreetSizes {
            lead: vec![serde_yaml::Value::Number(serde_yaml::Number::from(1))],
            raise: vec![vec![serde_yaml::Value::Number(serde_yaml::Number::from(1))]],
        };
        MpActionAbstractionConfig {
            max_flop_players: None,
            preflop,
            flop: postflop.clone(),
            turn: postflop.clone(),
            river: postflop,
        }
    }

    fn lazy_cap_action_config(max_flop_players: Option<u8>) -> LazyActionConfig {
        let mut action = action_config();
        action.max_flop_players = max_flop_players;
        LazyActionConfig::from_action_config(&action)
    }

    fn lazy_open_one_caller_state(active_seats: &[u8], to_act: u8) -> LazyPublicState {
        let mut street_bets = [Chips::ZERO; MAX_PLAYERS];
        street_bets[0] = Chips(10.0);
        street_bets[1] = Chips(10.0);
        street_bets[4] = Chips(1.0);
        street_bets[5] = Chips(2.0);

        let mut stacks = [Chips::ZERO; MAX_PLAYERS];
        let mut contributions = [Chips::ZERO; MAX_PLAYERS];
        for idx in 0..6 {
            stacks[idx] = Chips(30.0) - street_bets[idx];
            contributions[idx] = street_bets[idx];
        }

        let mut active = PlayerSet::empty();
        for &seat in active_seats {
            active.insert(Seat::from_raw(seat));
        }

        let mut acted_since_aggression = PlayerSet::empty();
        acted_since_aggression.insert(Seat::from_raw(0));
        acted_since_aggression.insert(Seat::from_raw(1));
        let pot = Chips(contributions.iter().map(|chips| chips.0).sum());

        LazyPublicState {
            stacks,
            street_bets,
            contributions,
            active,
            all_in: PlayerSet::empty(),
            acted_since_aggression,
            street: Street::Preflop,
            pot,
            num_players: 6,
            raise_count: 1,
            to_act: Seat::from_raw(to_act),
            facing_bet: true,
            last_raise_to: Chips(10.0),
            dealer: 3,
            big_blind_amount: Chips(2.0),
        }
    }

    fn test_buckets(deal: &Deal, bucket_counts: [u16; 4]) -> DealWithBuckets {
        let mut buckets = [[Bucket(0); 4]; MAX_PLAYERS];
        for (seat, seat_buckets) in buckets
            .iter_mut()
            .enumerate()
            .take(deal.num_players as usize)
        {
            for (street, bucket) in seat_buckets.iter_mut().enumerate() {
                *bucket = Bucket((seat as u16 + street as u16) % bucket_counts[street]);
            }
        }
        DealWithBuckets {
            deal: deal.clone(),
            buckets,
        }
    }

    fn river_spr_zero_state(facing_bet: bool, to_call: f64, remaining: f64) -> LazyPublicState {
        let mut active = PlayerSet::empty();
        active.insert(Seat::from_raw(0));
        active.insert(Seat::from_raw(1));
        let mut street_bets = [Chips::ZERO; MAX_PLAYERS];
        street_bets[1] = Chips(to_call);
        let mut stacks = [Chips::ZERO; MAX_PLAYERS];
        stacks[0] = Chips(remaining);
        stacks[1] = Chips(remaining);
        LazyPublicState {
            stacks,
            street_bets,
            contributions: [Chips::ZERO; MAX_PLAYERS],
            active,
            all_in: PlayerSet::empty(),
            acted_since_aggression: PlayerSet::empty(),
            street: Street::River,
            pot: Chips(100.0),
            num_players: 2,
            raise_count: if facing_bet { 1 } else { 0 },
            to_act: Seat::from_raw(0),
            facing_bet,
            last_raise_to: Chips(to_call),
            dealer: 0,
            big_blind_amount: Chips(2.0),
        }
    }

    fn postflop_audit_state(street: Street, facing_bet: bool, raise_count: u8) -> LazyPublicState {
        let mut active = PlayerSet::empty();
        active.insert(Seat::from_raw(0));
        active.insert(Seat::from_raw(1));
        let mut street_bets = [Chips::ZERO; MAX_PLAYERS];
        if facing_bet {
            street_bets[1] = Chips(10.0);
        }
        let mut stacks = [Chips::ZERO; MAX_PLAYERS];
        stacks[0] = Chips(50.0);
        stacks[1] = Chips(50.0);
        LazyPublicState {
            stacks,
            street_bets,
            contributions: [Chips::ZERO; MAX_PLAYERS],
            active,
            all_in: PlayerSet::empty(),
            acted_since_aggression: PlayerSet::empty(),
            street,
            pot: Chips(100.0),
            num_players: 2,
            raise_count,
            to_act: Seat::from_raw(0),
            facing_bet,
            last_raise_to: Chips(10.0),
            dealer: 0,
            big_blind_amount: Chips(2.0),
        }
    }

    #[timed_test]
    fn lazy_root_generates_open_actions_without_building_tree() {
        let game = LazyMpGame::new(&game_config(6, 200.0), &action_config());

        let root = game.root_state();
        let actions = game.actions(&root);

        assert_eq!(root.to_act(), Seat::from_raw(2));
        assert_eq!(root.street(), Street::Preflop);
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, TreeAction::Fold))
        );
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, TreeAction::Call))
        );
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, TreeAction::Lead(amount) if (*amount - 4.0).abs() < SIZE_EPSILON))
        );
    }

    #[timed_test]
    fn lazy_root_honors_no_limp_config() {
        let mut game_config = game_config(6, 200.0);
        game_config.allow_preflop_limp = false;
        let game = LazyMpGame::new(&game_config, &action_config());

        let mut state = game.root_state();
        for position in ["UTG", "HJ", "CO", "BTN"] {
            let actions = game.actions(&state);
            assert!(
                !actions
                    .iter()
                    .any(|action| matches!(action, TreeAction::Call)),
                "{position} should not include a cold limp when limps are disabled, got {actions:?}"
            );
            assert!(
                actions
                    .iter()
                    .any(|action| matches!(action, TreeAction::Lead(amount) if (*amount - 4.0).abs() < SIZE_EPSILON)),
                "{position} should keep configured opens when limps are disabled, got {actions:?}"
            );

            let LazyNode::Decision(next_state) =
                normalize_node(apply_action(state, TreeAction::Fold))
            else {
                panic!("{position} fold should route to next unopened preflop decision");
            };
            state = next_state;
        }

        let sb_actions = game.actions(&state);
        assert!(
            sb_actions
                .iter()
                .any(|action| matches!(action, TreeAction::Call)),
            "SB should keep completion/call when limps are disabled, got {sb_actions:?}"
        );
        let LazyNode::Decision(bb_state) = normalize_node(apply_action(state, TreeAction::Call))
        else {
            panic!("SB completion should route to BB decision");
        };
        let bb_actions = game.actions(&bb_state);
        assert!(
            bb_actions
                .iter()
                .any(|action| matches!(action, TreeAction::Check)),
            "BB should be able to check after SB completion, got {bb_actions:?}"
        );
    }

    #[timed_test]
    fn lazy_preflop_flop_player_cap_removes_non_closing_overcall_but_keeps_closing_call() {
        let config = lazy_cap_action_config(Some(3));
        let sb_non_closing = lazy_open_one_caller_state(&[0, 1, 4, 5], 4);
        let bb_closing = lazy_open_one_caller_state(&[0, 1, 5], 5);

        let sb_actions = generate_actions(&config, &sb_non_closing);
        let bb_actions = generate_actions(&config, &bb_closing);

        assert!(
            !sb_actions
                .iter()
                .any(|action| matches!(action, TreeAction::Call)),
            "SB non-closing overcall should be capped after opener plus caller: {sb_actions:?}"
        );
        assert!(
            bb_actions
                .iter()
                .any(|action| matches!(action, TreeAction::Call)),
            "BB closing call should remain legal at the cap: {bb_actions:?}"
        );
    }

    #[timed_test]
    fn lazy_preflop_flop_player_cap_none_preserves_non_closing_call() {
        let config = lazy_cap_action_config(None);
        let sb_non_closing = lazy_open_one_caller_state(&[0, 1, 4, 5], 4);

        let actions = generate_actions(&config, &sb_non_closing);

        assert!(
            actions
                .iter()
                .any(|action| matches!(action, TreeAction::Call)),
            "default uncapped config should preserve SB non-closing call: {actions:?}"
        );
    }

    #[timed_test]
    fn lazy_river_spr_zero_suppresses_new_aggression_when_unopened() {
        let game = LazyMpGame::new(&game_config(2, 200.0), &action_config());
        let state = river_spr_zero_state(false, 0.0, 50.0);

        let actions = game.actions(&state);

        assert_eq!(actions, vec![TreeAction::Check]);
    }

    #[timed_test]
    fn lazy_river_spr_zero_suppresses_all_in_raise_when_facing_bet() {
        let game = LazyMpGame::new(&game_config(2, 200.0), &action_config());
        let state = river_spr_zero_state(true, 10.0, 50.0);

        let actions = game.actions(&state);

        assert_eq!(actions, vec![TreeAction::Fold, TreeAction::Call]);
    }

    #[timed_test]
    fn lazy_river_spr_zero_keeps_all_in_call_when_call_closes_stack() {
        let game = LazyMpGame::new(&game_config(2, 200.0), &action_config());
        let state = river_spr_zero_state(true, 50.0, 50.0);

        let actions = game.actions(&state);

        assert_eq!(actions, vec![TreeAction::Fold, TreeAction::AllIn]);
    }

    #[timed_test]
    fn lazy_action_limit_audit_allows_one_all_in_aggression_past_raise_rows() {
        let game = LazyMpGame::new(&game_config(2, 200.0), &action_config());
        let _audit_guard = reset_lazy_action_limit_audit_for_test();
        let state = postflop_audit_state(Street::Flop, true, 1);

        let actions = game.actions(&state);
        let snapshot = take_lazy_action_limit_snapshot();

        assert!(
            actions
                .iter()
                .any(|action| matches!(action, TreeAction::AllIn))
        );
        assert_eq!(snapshot.max_raise_count[Street::Flop as usize], 2);
        assert_eq!(snapshot.all_in_aggressions[Street::Flop as usize], 1);
        assert_eq!(snapshot.over_config_decisions[Street::Flop as usize], 0);
        assert_eq!(snapshot.over_config_aggressions[Street::Flop as usize], 0);
    }

    #[timed_test]
    fn lazy_action_limit_audit_flags_decisions_past_allowed_extra_all_in() {
        let game = LazyMpGame::new(&game_config(2, 200.0), &action_config());
        let _audit_guard = reset_lazy_action_limit_audit_for_test();
        let state = postflop_audit_state(Street::Flop, false, 3);

        let actions = game.actions(&state);
        let snapshot = take_lazy_action_limit_snapshot();

        assert_eq!(actions, vec![TreeAction::Check]);
        assert_eq!(snapshot.max_raise_count[Street::Flop as usize], 3);
        assert_eq!(snapshot.over_config_decisions[Street::Flop as usize], 1);
        assert_eq!(snapshot.over_config_aggressions[Street::Flop as usize], 0);
    }

    #[timed_test]
    fn lazy_second_preflop_raise_row_is_reachable() {
        let game = LazyMpGame::new(&game_config(6, 200.0), &action_config());
        let root = game.root_state();
        let open = game
            .actions(&root)
            .into_iter()
            .find(|action| matches!(action, TreeAction::Lead(_)))
            .unwrap();
        let LazyNode::Decision(facing_open) = normalize_node(apply_action(root, open)) else {
            panic!("open should advance to next decision");
        };
        let three_bet = game
            .actions(&facing_open)
            .into_iter()
            .find(|action| matches!(action, TreeAction::Raise(_)))
            .unwrap();
        let LazyNode::Decision(facing_three_bet) =
            normalize_node(apply_action(facing_open, three_bet))
        else {
            panic!("3-bet should advance to next decision");
        };

        let actions = game.actions(&facing_three_bet);

        assert_eq!(facing_three_bet.raise_count(), 2);
        assert!(
            actions
                .iter()
                .any(|action| matches!(action, TreeAction::Raise(_))),
            "second preflop raise row should still offer a 4-bet action: {actions:?}"
        );
    }

    #[timed_test]
    fn lazy_traversal_updates_sparse_storage() {
        let game = LazyMpGame::new(&game_config(2, 20.0), &action_config());
        let storage = SparseMpStorage::with_shards(8);
        let mut rng = SmallRng::seed_from_u64(42);
        let deal = sample_deal(2, &mut rng);
        let buckets = test_buckets(&deal, [10, 10, 10, 10]);

        let (value, _stats) = traverse_external_lazy(
            &game,
            &storage,
            &buckets,
            Seat::from_raw(0),
            &mut rng,
            0.0,
            Chips::ZERO,
            false,
            -250,
            NegativeActionTraversalConfig::default(),
        );

        assert!(value.is_finite());
        assert!(storage.entry_count() > 0);
    }

    #[timed_test]
    fn lazy_opponent_traversal_updates_full_average_strategy_vector() {
        let game = LazyMpGame::new(&game_config(2, 20.0), &action_config());
        let storage = SparseMpStorage::with_shards(8);
        let mut rng = SmallRng::seed_from_u64(42);
        let deal = sample_deal(2, &mut rng);
        let buckets = test_buckets(&deal, [10, 10, 10, 10]);
        let root = game.root_state();
        let actions = game.actions(&root);
        let bucket = buckets.buckets[root.to_act.index() as usize][root.street.index()].0;
        let key = LazyHistory::default().key(root, bucket);

        let (value, _stats) = traverse_external_lazy(
            &game,
            &storage,
            &buckets,
            Seat::from_raw(1),
            &mut rng,
            0.0,
            Chips::ZERO,
            false,
            -250,
            NegativeActionTraversalConfig::default(),
        );

        assert!(value.is_finite());
        for action_idx in 0..actions.len() {
            assert!(
                storage.get_strategy_sum(key, action_idx) > 0,
                "opponent average update should record every root action"
            );
        }
    }

    #[timed_test]
    fn negative_action_masked_strategy_falls_back_to_uniform_eligible_actions() {
        let strategy = [1.0, 0.0, 0.0, 0.0];
        let blocked = [true, false, true, false];
        let mut masked = [42.0; 4];

        assert!(mask_strategy_to_unblocked(&strategy, &blocked, &mut masked));

        assert_eq!(masked, [0.0, 0.5, 0.0, 0.5]);
    }

    #[timed_test]
    fn negative_action_masked_strategy_renormalizes_remaining_mass() {
        let strategy = [0.7, 0.2, 0.1];
        let blocked = [true, false, false];
        let mut masked = [0.0; 3];

        assert!(mask_strategy_to_unblocked(&strategy, &blocked, &mut masked));

        assert!((masked[0] - 0.0).abs() < 1e-12);
        assert!((masked[1] - (2.0 / 3.0)).abs() < 1e-12);
        assert!((masked[2] - (1.0 / 3.0)).abs() < 1e-12);
    }

    #[timed_test]
    fn lazy_pruning_ignores_terminal_children() {
        let storage = SparseMpStorage::with_shards(8);
        let key =
            MpInfosetKey::from_street_bucket(Seat::from_raw(0), Street::Preflop, 0, 0, 0, 0, 0);
        let mut stats = PruneStats::default();

        storage.add_regret(key, 2, 0, -10_000);

        assert!(!should_prune_lazy(
            &storage, true, -1, true, key, 0, &mut stats
        ));
        assert_eq!(stats.total, 0);
        assert_eq!(stats.hits, 0);
    }

    #[timed_test]
    fn lazy_pruning_skips_negative_nonterminal_actions() {
        let storage = SparseMpStorage::with_shards(8);
        let key =
            MpInfosetKey::from_street_bucket(Seat::from_raw(0), Street::Preflop, 0, 0, 0, 0, 0);
        let mut stats = PruneStats::default();

        storage.add_regret(key, 2, 1, -10_000);

        assert!(should_prune_lazy(
            &storage, true, -1, false, key, 1, &mut stats
        ));
        assert_eq!(stats.total, 1);
        assert_eq!(stats.hits, 1);
    }

    #[timed_test]
    fn negative_action_gate_disabled_preserves_default_traversal_storage_behavior() {
        let game = LazyMpGame::new(&game_config(2, 20.0), &action_config());
        let storage = SparseMpStorage::with_shards(8);
        let root = game.root_state();
        let root_history = LazyHistory::default();
        let root_key = root_history.key(root, 10);
        let action_idx = 0;
        let child_history = root_history.append(action_idx);
        let descendant_history = child_history.append(0);
        let descendant_key = descendant_history.key(root, 10);

        storage.add_regret(root_key, game.actions(&root).len(), action_idx, -2);
        storage.add_regret(descendant_key, 2, 0, 17);

        assert!(!negative_action_blocks(
            &storage,
            NegativeActionTraversalConfig {
                enabled: false,
                prune_below: -1,
                reactivate_at: 0,
            },
            true,
            root_key,
            action_idx,
            child_history
        ));

        assert!(!storage.is_negative_action_edge_blocked(root_key, action_idx));
        assert_eq!(storage.negative_action_blocked_edge_count(), 0);
        assert_eq!(storage.get_regret(descendant_key, 0), 17);
        assert_eq!(
            storage.negative_action_telemetry(),
            crate::blueprint_mp::sparse_storage::SparseNegativeActionTelemetry::default()
        );
    }

    #[timed_test]
    fn negative_action_gate_ignores_passive_edges() {
        let game = LazyMpGame::new(&game_config(2, 20.0), &action_config());
        let storage = SparseMpStorage::with_shards(8);
        let root = game.root_state();
        let actions = game.actions(&root);
        let root_history = LazyHistory::default();
        let root_key = root_history.key(root, 10);
        let action_idx = actions
            .iter()
            .position(|action| !action_increments_raise_count(&root, *action))
            .expect("root should have a passive action");
        let child_history = root_history.append(action_idx);
        let descendant_history = child_history.append(0);
        let descendant_key = descendant_history.key(root, 10);

        storage.add_regret(root_key, actions.len(), action_idx, -200);
        storage.add_regret(descendant_key, 2, 0, 17);

        assert!(!negative_action_blocks(
            &storage,
            NegativeActionTraversalConfig {
                enabled: true,
                prune_below: -1,
                reactivate_at: 0,
            },
            false,
            root_key,
            action_idx,
            child_history
        ));

        assert!(!storage.is_negative_action_edge_blocked(root_key, action_idx));
        assert_eq!(storage.negative_action_blocked_edge_count(), 0);
        assert_eq!(storage.get_regret(descendant_key, 0), 17);
    }

    #[timed_test]
    fn lazy_negative_action_gate_blocks_aggressive_edge_without_traversal_time_purge() {
        let game = LazyMpGame::new(&game_config(2, 20.0), &action_config());
        let storage = SparseMpStorage::with_shards(8);
        let mut rng = SmallRng::seed_from_u64(43);
        let deal = sample_deal(2, &mut rng);
        let buckets = test_buckets(&deal, [10, 10, 10, 10]);
        let root = game.root_state();
        let actions = game.actions(&root);
        let root_bucket = buckets.buckets[root.to_act.index() as usize][root.street.index()].0;
        let root_history = LazyHistory::default();
        let root_key = root_history.key(root, root_bucket);
        let action_idx = actions
            .iter()
            .position(|action| action_increments_raise_count(&root, *action))
            .expect("root should have an aggressive action");
        let child_history = root_history.append(action_idx);
        let descendant_history = child_history.append(0);
        let descendant_key = descendant_history.key(root, root_bucket);

        storage.add_regret(root_key, actions.len(), action_idx, -2);
        storage.add_regret(descendant_key, 2, 0, 17);
        assert_eq!(storage.get_regret(descendant_key, 0), 17);

        let (value, _stats) = traverse_external_lazy(
            &game,
            &storage,
            &buckets,
            root.to_act,
            &mut rng,
            0.0,
            Chips::ZERO,
            false,
            -250,
            NegativeActionTraversalConfig {
                enabled: true,
                prune_below: -1,
                reactivate_at: 0,
            },
        );

        assert!(value.is_finite());
        assert!(storage.is_negative_action_edge_blocked(root_key, action_idx));
        assert_eq!(storage.get_regret(descendant_key, 0), 17);
        assert_eq!(
            storage.negative_action_telemetry().blocked_traversal_skips,
            1
        );
    }

    #[timed_test]
    fn lazy_negative_action_opponent_mask_does_not_count_non_sampled_skips() {
        let game = LazyMpGame::new(&game_config(2, 20.0), &action_config());
        let storage = SparseMpStorage::with_shards(8);
        let mut rng = SmallRng::seed_from_u64(44);
        let deal = sample_deal(2, &mut rng);
        let buckets = test_buckets(&deal, [10, 10, 10, 10]);
        let root = game.root_state();
        let root_bucket = buckets.buckets[root.to_act.index() as usize][root.street.index()].0;
        let root_history = LazyHistory::default();
        let root_key = root_history.key(root, root_bucket);
        let actions = game.actions(&root);
        let action_idx = actions
            .iter()
            .position(|action| action_increments_raise_count(&root, *action))
            .expect("root should have an aggressive action");

        storage.add_regret(root_key, actions.len(), action_idx, -2);

        let (value, _stats) = traverse_external_lazy(
            &game,
            &storage,
            &buckets,
            Seat::from_raw((root.to_act.index() + 1) % root.num_players),
            &mut rng,
            0.0,
            Chips::ZERO,
            false,
            -250,
            NegativeActionTraversalConfig {
                enabled: true,
                prune_below: -1,
                reactivate_at: 0,
            },
        );

        assert!(value.is_finite());
        assert!(storage.is_negative_action_edge_blocked(root_key, action_idx));
        assert_eq!(
            storage.negative_action_telemetry().blocked_traversal_skips,
            0
        );
    }
}
