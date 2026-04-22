//! Per-boundary trace logging for the hybrid solver.
//!
//! Emits TXT files capturing what the cfvnet boundary evaluator sees and
//! produces at specific boundaries, enabling post-hoc debugging of
//! distribution-shift on 4bet-pot ranges.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use range_solver::Action;
use range_solver::card::index_to_card_pair;
use range_solver::interface::GameNode;

// ---------------------------------------------------------------------------
// Hand name formatting
// ---------------------------------------------------------------------------

const RANK_CHARS: [char; 13] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];

/// Map a card pair to a canonical hand name (e.g. "AA", "AKs", "72o").
///
/// Card encoding: `rank = card / 4`, `suit = card % 4`.
/// Rank 0 = Deuce, rank 12 = Ace.
pub fn canonical_hand_name(c1: u8, c2: u8) -> String {
    let r1 = c1 / 4;
    let r2 = c2 / 4;
    let s1 = c1 % 4;
    let s2 = c2 % 4;
    let (hi, lo) = if r1 >= r2 { (r1, r2) } else { (r2, r1) };
    let hi_ch = RANK_CHARS[hi as usize];
    let lo_ch = RANK_CHARS[lo as usize];
    if hi == lo {
        format!("{hi_ch}{lo_ch}")
    } else if s1 == s2 {
        format!("{hi_ch}{lo_ch}s")
    } else {
        format!("{hi_ch}{lo_ch}o")
    }
}

// ---------------------------------------------------------------------------
// Range / CFV aggregation (1326 -> 169 hand classes)
// ---------------------------------------------------------------------------

/// Per-hand-class aggregate: canonical name, combo count, total combos, mean value.
struct HandClassAgg {
    name: String,
    total_combos: usize,
    nonzero_combos: usize,
    weight_sum: f64,
}

/// Group the 1326-element vector by canonical hand name.
///
/// Returns aggregates sorted by descending absolute mean value.
fn aggregate_1326_by_hand(values: &[f32], threshold: f32) -> Vec<HandClassAgg> {
    use std::collections::HashMap;
    assert_eq!(values.len(), 1326);

    let mut map: HashMap<String, (usize, usize, f64)> = HashMap::new();
    for idx in 0..1326 {
        let (c1, c2) = index_to_card_pair(idx);
        let name = canonical_hand_name(c1, c2);
        let entry = map.entry(name).or_insert((0, 0, 0.0));
        entry.0 += 1;
        if values[idx].abs() > threshold {
            entry.1 += 1;
            entry.2 += values[idx] as f64;
        }
    }

    let mut aggs: Vec<HandClassAgg> = map
        .into_iter()
        .map(|(name, (total, nonzero, sum))| HandClassAgg {
            name,
            total_combos: total,
            nonzero_combos: nonzero,
            weight_sum: sum,
        })
        .collect();

    aggs.sort_by(|a, b| {
        let mean_a = if a.nonzero_combos > 0 {
            (a.weight_sum / a.nonzero_combos as f64).abs()
        } else {
            0.0
        };
        let mean_b = if b.nonzero_combos > 0 {
            (b.weight_sum / b.nonzero_combos as f64).abs()
        } else {
            0.0
        };
        mean_b.partial_cmp(&mean_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    aggs
}

/// Format a 1326-element range vector as a compact hand-class string.
///
/// Output: `"AA:6/6:1.00 KK:6/6:0.98 ..."` sorted by descending mean weight.
/// Only includes hand classes with mean weight > 0.001.
pub fn format_range_as_hands(range_1326: &[f32]) -> String {
    let aggs = aggregate_1326_by_hand(range_1326, 0.001);
    let parts: Vec<String> = aggs
        .iter()
        .filter(|a| a.nonzero_combos > 0)
        .map(|a| {
            let mean = a.weight_sum / a.nonzero_combos as f64;
            format!("{}:{}/{}:{:.2}", a.name, a.nonzero_combos, a.total_combos, mean)
        })
        .collect();
    parts.join(" ")
}

/// Format a 1326-element CFV vector as a compact hand-class string.
///
/// Output: `"AA:+3.45 KK:+2.10 ..."` sorted by descending absolute mean CFV.
/// Only includes hand classes with at least one non-zero combo.
/// CFV values are in chip units.
pub fn format_cfvs_as_hands(cfvs_1326: &[f32]) -> String {
    let aggs = aggregate_1326_by_hand(cfvs_1326, 0.001);
    let parts: Vec<String> = aggs
        .iter()
        .filter(|a| a.nonzero_combos > 0)
        .map(|a| {
            let mean = a.weight_sum / a.nonzero_combos as f64;
            if mean >= 0.0 {
                format!("{}:+{:.2}", a.name, mean)
            } else {
                format!("{}:{:.2}", a.name, mean)
            }
        })
        .collect();
    parts.join(" ")
}

// ---------------------------------------------------------------------------
// Spot-string helpers
// ---------------------------------------------------------------------------

/// Convert a range-solver `Action` into the lowercased label that Tauri's
/// `GameSession::encode_spot` / `load_spot` expects.
///
/// Labels: `check`, `call`, `fold`, `{bb}bb`, `all-in`.
/// Chip amounts are halved (1 chip = 0.5 BB) and rounded.
fn format_postflop_action(action: &Action) -> String {
    match action {
        Action::Check => "check".to_string(),
        Action::Call => "call".to_string(),
        Action::Fold => "fold".to_string(),
        Action::Bet(chips) | Action::Raise(chips) => {
            let bb = (*chips as f64 / 2.0).round() as i64;
            format!("{bb}bb")
        }
        Action::AllIn(_) => "all-in".to_string(),
        _ => "?".to_string(),
    }
}

/// Format a sequence of `Action`s (root-to-boundary) as the postflop portion
/// of a Tauri spot string.
///
/// `Action::Chance(card)` inserts a `|{card}|` separator between streets.
/// Player actions alternate starting with OOP (BB) at the beginning of each
/// street.
fn format_action_path_as_spot_suffix(actions: &[Action]) -> String {
    use range_solver::card::card_to_string;

    let mut parts: Vec<String> = Vec::new();
    let mut current_street: Vec<String> = Vec::new();
    // OOP (BB) = 0, IP (SB) = 1; OOP acts first on each street.
    let mut player_to_act: usize = 0;

    for action in actions {
        match action {
            Action::Chance(card) => {
                // Flush current street's actions
                if !current_street.is_empty() {
                    parts.push(current_street.join(","));
                    current_street.clear();
                }
                // Emit the dealt card
                let card_str = card_to_string(*card).unwrap_or_else(|_| "??".to_string());
                parts.push(card_str);
                // Reset to OOP acting first on new street
                player_to_act = 0;
            }
            Action::None => {}
            action => {
                let pos = if player_to_act == 0 { "bb" } else { "sb" };
                let label = format_postflop_action(action);
                current_street.push(format!("{pos}:{label}"));
                // Alternate player after each action
                player_to_act ^= 1;
            }
        }
    }

    // Flush remaining street actions
    if !current_street.is_empty() {
        parts.push(current_street.join(","));
    }

    parts.join("|")
}

/// Build the postflop spot-string suffix for every depth boundary in ordinal
/// order. Uses a DFS from the root to discover the `Action` path leading to
/// each boundary node.
///
/// Returns one `String` per boundary ordinal. If the walk fails to reach a
/// boundary (should not happen), the entry is `"<error: unreachable>"`.
pub fn build_boundary_spot_paths(game: &range_solver::PostFlopGame) -> Vec<String> {
    let boundary_indices = game.boundary_node_indices();
    let n = boundary_indices.len();
    if n == 0 {
        return Vec::new();
    }

    // Map boundary arena index → ordinal for O(1) lookup.
    let mut index_to_ord = std::collections::HashMap::with_capacity(n);
    for (ord, &idx) in boundary_indices.iter().enumerate() {
        index_to_ord.insert(idx, ord);
    }

    let mut paths = vec!["<error: unreachable>".to_string(); n];

    // DFS walk: (node_arena_index, action_history_so_far)
    let mut stack: Vec<(usize, Vec<Action>)> = vec![(0, Vec::new())];
    while let Some((node_idx, history)) = stack.pop() {
        if let Some(&ord) = index_to_ord.get(&node_idx) {
            paths[ord] = format_action_path_as_spot_suffix(&history);
        }
        let children = game.child_indices(node_idx);
        for child_idx in children {
            let child_node = game.node_at(child_idx);
            let mut child_history = history.clone();
            child_history.push(child_node.prev_action());
            drop(child_node);
            stack.push((child_idx, child_history));
        }
    }

    paths
}

/// Combine the user-supplied preflop spot prefix with a postflop path suffix.
///
/// If the postflop suffix is empty, returns the prefix unchanged.
pub fn assemble_full_spot(preflop_prefix: &str, postflop_suffix: &str) -> String {
    if postflop_suffix.is_empty() {
        return preflop_prefix.to_string();
    }
    format!("{preflop_prefix}|{postflop_suffix}")
}

// ---------------------------------------------------------------------------
// Preceding-decision map
// ---------------------------------------------------------------------------

/// For each depth boundary, find the nearest ancestor decision node and its
/// action labels.
///
/// Returns a map from `boundary_ordinal -> (decision_node_index, action_labels)`.
/// Built via a single DFS from root, tracking the most recent decision node
/// on each path.
pub fn build_preceding_decision_map(
    game: &range_solver::PostFlopGame,
) -> HashMap<usize, (usize, Vec<String>)> {
    let boundary_indices = game.boundary_node_indices();
    let n = boundary_indices.len();
    if n == 0 {
        return HashMap::new();
    }

    let mut index_to_ord = HashMap::with_capacity(n);
    for (ord, &idx) in boundary_indices.iter().enumerate() {
        index_to_ord.insert(idx, ord);
    }

    let mut result = HashMap::with_capacity(n);

    // DFS: (node_index, most_recent_decision_idx_or_None)
    let mut stack: Vec<(usize, Option<usize>)> = vec![(0, None)];
    while let Some((node_idx, last_decision)) = stack.pop() {
        let node = game.node_at(node_idx);
        let is_decision = !node.is_terminal() && !node.is_chance();

        let current_decision = if is_decision {
            Some(node_idx)
        } else {
            last_decision
        };

        // If this is a boundary node, record the preceding decision.
        if let Some(&ord) = index_to_ord.get(&node_idx) {
            if let Some(dec_idx) = last_decision {
                let dec_node = game.node_at(dec_idx);
                let actions: Vec<String> = game
                    .child_indices(dec_idx)
                    .iter()
                    .map(|&ci| {
                        let child = game.node_at(ci);
                        format_postflop_action(&child.prev_action())
                    })
                    .collect();
                drop(dec_node);
                result.insert(ord, (dec_idx, actions));
            }
        }

        let children = game.child_indices(node_idx);
        drop(node);
        for child_idx in children {
            stack.push((child_idx, current_decision));
        }
    }

    result
}

// ---------------------------------------------------------------------------
// TraceConfig (CLI-level configuration, parsed in main.rs)
// ---------------------------------------------------------------------------

/// Configuration for boundary tracing, passed from CLI to compare_solve::run.
pub struct TraceConfig {
    /// `None` means no tracing. `Some("all")` or `Some("0,42,100")`.
    pub boundaries: Option<String>,
    /// `"last"`, `"all"`, or `"0,9"`.
    pub iters_str: String,
    /// Directory to write trace files.
    pub dir: PathBuf,
}

impl TraceConfig {
    /// Build a `BoundaryTracer` from this config, resolving "last" with `total_iters`.
    ///
    /// Returns `None` if tracing is disabled (no --trace-boundaries flag).
    pub fn into_tracer(self, total_iters: u32) -> Option<BoundaryTracer> {
        let ord_filter = TraceFilter::parse(self.boundaries.as_deref());
        if matches!(ord_filter, TraceFilter::None) {
            return None;
        }
        let iter_filter = TraceFilter::parse_iters(&self.iters_str, total_iters);
        Some(BoundaryTracer::new(ord_filter, iter_filter, self.dir))
    }
}

// ---------------------------------------------------------------------------
// TraceFilter
// ---------------------------------------------------------------------------

/// Filter for selecting which ordinals or iterations to trace.
#[derive(Debug)]
pub enum TraceFilter {
    /// No tracing.
    None,
    /// Trace all values.
    All,
    /// Trace only specific values.
    Set(HashSet<usize>),
    /// Trace only the last iteration (value = max_iter - 1).
    Last(usize),
}

impl TraceFilter {
    /// Parse from an optional CLI string.
    ///
    /// `None` -> `TraceFilter::None`, `"all"` -> `All`, `"0,42,100"` -> `Set`.
    pub fn parse(input: Option<&str>) -> Self {
        match input {
            Option::None => TraceFilter::None,
            Some("all") => TraceFilter::All,
            Some(csv) => {
                let set: HashSet<usize> = csv
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                if set.is_empty() {
                    TraceFilter::None
                } else {
                    TraceFilter::Set(set)
                }
            }
        }
    }

    /// Parse iteration filter, with "last" support.
    ///
    /// `"last"` -> `Last(total_iters - 1)`, `"all"` -> `All`, `"0,9"` -> `Set`.
    pub fn parse_iters(input: &str, total_iters: u32) -> Self {
        match input {
            "last" => TraceFilter::Last(total_iters.saturating_sub(1) as usize),
            "all" => TraceFilter::All,
            csv => {
                let set: HashSet<usize> = csv
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                if set.is_empty() {
                    TraceFilter::None
                } else {
                    TraceFilter::Set(set)
                }
            }
        }
    }

    /// Returns true if the given value passes this filter.
    pub fn matches(&self, value: usize) -> bool {
        match self {
            TraceFilter::None => false,
            TraceFilter::All => true,
            TraceFilter::Set(s) => s.contains(&value),
            TraceFilter::Last(v) => value == *v,
        }
    }
}

// ---------------------------------------------------------------------------
// BoundaryTraceEvent
// ---------------------------------------------------------------------------

/// Raw data for a single boundary trace event.
pub struct BoundaryTraceEvent {
    pub board: String,
    pub pot: i32,
    pub stack: f64,
    /// Full spot string for navigating to this boundary in the Tauri UI.
    /// `None` if spot generation failed or was not configured.
    pub spot: Option<String>,
    pub oop_range_1326: Vec<f32>,
    pub ip_range_1326: Vec<f32>,
    pub oop_cfvs_1326: Vec<f32>,
    pub ip_cfvs_1326: Vec<f32>,
    pub strategy_at_prev: Option<StrategyAtPrevDecision>,
}

/// Strategy at the preceding decision node.
pub struct StrategyAtPrevDecision {
    pub node_idx: usize,
    /// Player to act: 0 = OOP, 1 = IP.
    pub player: usize,
    pub actions: Vec<String>,
    /// Per-hand-class strategy: `(hand_name, [prob_per_action])`, sorted by
    /// descending reach weight.
    pub by_hand: Vec<(String, Vec<f32>)>,
}

// ---------------------------------------------------------------------------
// TXT formatting
// ---------------------------------------------------------------------------

/// Count the number of combos with weight >= 0.001 in a 1326-element vector.
pub fn count_nonzero_combos(range: &[f32]) -> usize {
    range.iter().filter(|&&w| w.abs() >= 0.001).count()
}

/// Format a single boundary trace event as a human-readable TXT record.
pub fn format_trace_txt(iter: u32, boundary_ord: usize, event: &BoundaryTraceEvent) -> String {
    let mut out = String::with_capacity(4096);

    // Header line
    let spr = if event.pot > 0 {
        event.stack / event.pot as f64
    } else {
        0.0
    };
    let _ = writeln!(
        out,
        "[iter={iter} boundary={boundary_ord} board={} pot={} stack={} spr={spr:.2}]",
        event.board,
        event.pot,
        event.stack as i64,
    );

    // Spot line (only if present)
    if let Some(ref spot) = event.spot {
        let _ = writeln!(out, "Spot: {spot}");
    }

    // Ranges
    let oop_combos = count_nonzero_combos(&event.oop_range_1326);
    let ip_combos = count_nonzero_combos(&event.ip_range_1326);
    let _ = writeln!(
        out,
        "OOP range ({oop_combos} combos): {}",
        format_range_as_hands(&event.oop_range_1326)
    );
    let _ = writeln!(
        out,
        "IP range ({ip_combos} combos):  {}",
        format_range_as_hands(&event.ip_range_1326)
    );

    // CFVs
    let _ = writeln!(
        out,
        "OOP CFVs (chips): {}",
        format_cfvs_as_hands(&event.oop_cfvs_1326)
    );
    let _ = writeln!(
        out,
        "IP CFVs  (chips): {}",
        format_cfvs_as_hands(&event.ip_cfvs_1326)
    );

    // Strategy at preceding decision
    if let Some(ref s) = event.strategy_at_prev {
        let player_label = if s.player == 0 { "OOP" } else { "IP" };
        let _ = writeln!(
            out,
            "Strategy at preceding decision (node #{}, {player_label} to act):",
            s.node_idx,
        );
        let _ = writeln!(out, "  Actions: [{}]", s.actions.join(", "));
        // Find max hand-name length for alignment
        let max_name = s.by_hand.iter().map(|(n, _)| n.len()).max().unwrap_or(2);
        for (name, probs) in &s.by_hand {
            let probs_str: Vec<String> = probs.iter().map(|p| format!("{p:.2}")).collect();
            let padded_name = format!("{name}:");
            let _ = writeln!(
                out,
                "  {padded_name:width$} [{probs}]",
                width = max_name + 2,
                probs = probs_str.join(", "),
            );
        }
    }

    // Separator
    let _ = writeln!(out, "---");
    out
}

// ---------------------------------------------------------------------------
// BoundaryTracer
// ---------------------------------------------------------------------------

/// Thread-safe tracer that writes per-boundary TXT files.
pub struct BoundaryTracer {
    enabled_ordinals: TraceFilter,
    enabled_iters: TraceFilter,
    trace_dir: PathBuf,
    handles: Mutex<HashMap<usize, std::io::BufWriter<std::fs::File>>>,
}

impl BoundaryTracer {
    pub fn new(ords: TraceFilter, iters: TraceFilter, dir: PathBuf) -> Self {
        Self {
            enabled_ordinals: ords,
            enabled_iters: iters,
            trace_dir: dir,
            handles: Mutex::new(HashMap::new()),
        }
    }

    /// Returns true if this (ord, iter) pair should be traced.
    pub fn enabled(&self, ord: usize, iter: u32) -> bool {
        self.enabled_ordinals.matches(ord) && self.enabled_iters.matches(iter as usize)
    }

    /// Write a trace event for one boundary at one iteration.
    pub fn trace(&self, ord: usize, iter: u32, event: &BoundaryTraceEvent) {
        if !self.enabled(ord, iter) {
            return;
        }

        let txt = format_trace_txt(iter, ord, event);

        let mut handles = self.handles.lock().expect("tracer lock poisoned");
        let writer = handles.entry(ord).or_insert_with(|| {
            std::fs::create_dir_all(&self.trace_dir)
                .expect("failed to create trace directory");
            let path = self.trace_dir.join(format!("boundary_{ord}.txt"));
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));
            std::io::BufWriter::new(file)
        });
        write!(writer, "{txt}").expect("failed to write trace record");
        writer.flush().expect("failed to flush trace record");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_solver::Action;

    // ---------------------------------------------------------------
    // build_boundary_spot_paths integration tests
    // ---------------------------------------------------------------

    /// Build a turn-start game with depth_limit=1, so flop→turn chance
    /// creates depth boundary nodes (one per non-isomorphic turn card).
    fn make_depth_limited_turn_game() -> range_solver::PostFlopGame {
        use range_solver::card::{card_from_str, flop_from_str, NOT_DEALT};
        use range_solver::bet_size::BetSizeOptions;
        use range_solver::{CardConfig, ActionTree, TreeConfig, BoardState, PostFlopGame};

        let oop_range: range_solver::range::Range = "AA,KK".parse().unwrap();
        let ip_range: range_solver::range::Range = "QQ,JJ".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: NOT_DEALT,
            river: NOT_DEALT,
        };
        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::Flop,
            starting_pot: 100,
            effective_stack: 200,
            flop_bet_sizes: [sizes.clone(), sizes.clone()],
            turn_bet_sizes: [sizes.clone(), sizes.clone()],
            river_bet_sizes: [sizes.clone(), sizes],
            depth_limit: Some(1), // boundary at turn deal
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
        game.allocate_memory(false);
        game
    }

    #[test]
    fn boundary_spot_paths_has_correct_count() {
        let game = make_depth_limited_turn_game();
        let n = game.num_boundary_nodes();
        assert!(n > 0, "should have boundary nodes with depth_limit=1");
        let paths = build_boundary_spot_paths(&game);
        assert_eq!(paths.len(), n);
    }

    #[test]
    fn boundary_spot_paths_differ_for_different_turns() {
        let game = make_depth_limited_turn_game();
        let paths = build_boundary_spot_paths(&game);
        assert!(paths.len() >= 2, "need at least 2 boundary nodes");
        // Not all paths should be identical — different turn cards
        // create different chance actions in the history.
        let unique: std::collections::HashSet<&String> = paths.iter().collect();
        assert!(
            unique.len() > 1,
            "boundary paths should differ across turn cards, got all identical: {:?}",
            &paths[..2.min(paths.len())]
        );
    }

    #[test]
    fn boundary_spot_paths_contain_card_separator() {
        let game = make_depth_limited_turn_game();
        let paths = build_boundary_spot_paths(&game);
        // With depth_limit=1 from flop, boundaries are at turn deal.
        // The path from root to boundary crosses through flop actions
        // then a Chance(card). The formatted path should contain a '|'
        // separating the flop actions from the turn card.
        let has_separator = paths.iter().any(|p| p.contains('|'));
        assert!(
            has_separator,
            "at least some boundary paths should contain '|' card separator, got: {:?}",
            &paths[..2.min(paths.len())]
        );
    }

    #[test]
    fn assemble_spot_prepends_preflop() {
        let preflop = "sb:2bb,bb:10bb,sb:22bb,bb:call|Jd9d7d";
        let postflop = "bb:check,sb:22bb,bb:call|3c";
        let result = assemble_full_spot(preflop, postflop);
        assert_eq!(
            result,
            "sb:2bb,bb:10bb,sb:22bb,bb:call|Jd9d7d|bb:check,sb:22bb,bb:call|3c"
        );
    }

    #[test]
    fn assemble_spot_with_empty_postflop() {
        let preflop = "sb:2bb,bb:call|Jd9d7d";
        let result = assemble_full_spot(preflop, "");
        assert_eq!(result, "sb:2bb,bb:call|Jd9d7d");
    }

    // ---------------------------------------------------------------
    // format_action_path_as_spot_suffix tests
    // ---------------------------------------------------------------

    #[test]
    fn spot_suffix_single_check() {
        // OOP (BB) checks at flop root
        let actions = vec![Action::Check];
        let result = format_action_path_as_spot_suffix(&actions);
        assert_eq!(result, "bb:check");
    }

    #[test]
    fn spot_suffix_check_bet_call() {
        let actions = vec![Action::Check, Action::Bet(44), Action::Call];
        let result = format_action_path_as_spot_suffix(&actions);
        assert_eq!(result, "bb:check,sb:22bb,bb:call");
    }

    #[test]
    fn spot_suffix_with_chance_card() {
        // OOP check, IP check, then deal turn card 12 (5c), then OOP check
        // card 12: rank=12/4=3 -> '5', suit=12%4=0 -> 'c'
        let actions = vec![
            Action::Check,
            Action::Check,
            Action::Chance(12),
            Action::Check,
        ];
        let result = format_action_path_as_spot_suffix(&actions);
        assert_eq!(result, "bb:check,sb:check|5c|bb:check");
    }

    #[test]
    fn spot_suffix_allin() {
        let actions = vec![Action::Bet(50), Action::AllIn(200)];
        let result = format_action_path_as_spot_suffix(&actions);
        assert_eq!(result, "bb:25bb,sb:all-in");
    }

    #[test]
    fn spot_suffix_empty_actions() {
        let actions: Vec<Action> = vec![];
        let result = format_action_path_as_spot_suffix(&actions);
        assert_eq!(result, "");
    }

    #[test]
    fn spot_suffix_two_streets_with_chance() {
        // flop: bb check, sb bet, bb call
        // turn deal: card 20 (7c)
        // card 20: rank=20/4=5 -> '7', suit=20%4=0 -> 'c'
        // turn: bb check, sb all-in, bb fold
        let actions = vec![
            Action::Check,
            Action::Bet(44),
            Action::Call,
            Action::Chance(20),
            Action::Check,
            Action::AllIn(200),
            Action::Fold,
        ];
        let result = format_action_path_as_spot_suffix(&actions);
        assert_eq!(result, "bb:check,sb:22bb,bb:call|7c|bb:check,sb:all-in,bb:fold");
    }

    // ---------------------------------------------------------------
    // format_postflop_action tests
    // ---------------------------------------------------------------

    #[test]
    fn format_action_check() {
        assert_eq!(format_postflop_action(&Action::Check), "check");
    }

    #[test]
    fn format_action_call() {
        assert_eq!(format_postflop_action(&Action::Call), "call");
    }

    #[test]
    fn format_action_fold() {
        assert_eq!(format_postflop_action(&Action::Fold), "fold");
    }

    #[test]
    fn format_action_bet_even_chips() {
        // 44 chips = 22 BB
        assert_eq!(format_postflop_action(&Action::Bet(44)), "22bb");
    }

    #[test]
    fn format_action_bet_odd_chips() {
        // 33 chips = 16.5 BB — should show decimal
        assert_eq!(format_postflop_action(&Action::Bet(33)), "17bb");
    }

    #[test]
    fn format_action_raise() {
        // 100 chips = 50 BB
        assert_eq!(format_postflop_action(&Action::Raise(100)), "50bb");
    }

    #[test]
    fn format_action_allin() {
        assert_eq!(format_postflop_action(&Action::AllIn(200)), "all-in");
    }

    #[test]
    fn format_action_none_returns_unknown() {
        assert_eq!(format_postflop_action(&Action::None), "?");
    }

    // ---------------------------------------------------------------
    // canonical_hand_name tests
    // ---------------------------------------------------------------

    #[test]
    fn canonical_hand_name_pair_aces() {
        // Two aces of different suits: Ac=48, Ad=49
        assert_eq!(canonical_hand_name(48, 49), "AA");
    }

    #[test]
    fn canonical_hand_name_pair_deuces() {
        // 2c=0, 2d=1
        assert_eq!(canonical_hand_name(0, 1), "22");
    }

    #[test]
    fn canonical_hand_name_suited_ak() {
        // Ac=48, Kc=44 (both clubs)
        assert_eq!(canonical_hand_name(48, 44), "AKs");
    }

    #[test]
    fn canonical_hand_name_offsuit_ak() {
        // Ac=48, Kd=45 (different suits)
        assert_eq!(canonical_hand_name(48, 45), "AKo");
    }

    #[test]
    fn canonical_hand_name_72o() {
        // 7c=20, 2d=1 (different suits)
        assert_eq!(canonical_hand_name(20, 1), "72o");
    }

    #[test]
    fn canonical_hand_name_order_independent() {
        // Should give same result regardless of card order
        assert_eq!(canonical_hand_name(48, 44), canonical_hand_name(44, 48));
        assert_eq!(canonical_hand_name(0, 1), canonical_hand_name(1, 0));
    }

    #[test]
    fn canonical_hand_name_suited_98() {
        // 9h=30, 8h=26 (both hearts, suit=2)
        assert_eq!(canonical_hand_name(30, 26), "98s");
    }

    #[test]
    fn canonical_hand_name_pair_kings() {
        // Kc=44, Ks=47
        assert_eq!(canonical_hand_name(44, 47), "KK");
    }

    // ---------------------------------------------------------------
    // format_range_as_hands tests
    // ---------------------------------------------------------------

    #[test]
    fn format_range_uniform_has_169_entries() {
        let range = vec![1.0f32; 1326];
        let result = format_range_as_hands(&range);
        let entries: Vec<&str> = result.split_whitespace().collect();
        assert_eq!(entries.len(), 169, "expected 169 hand classes, got {}", entries.len());
    }

    #[test]
    fn format_range_uniform_pairs_have_6_combos() {
        let range = vec![1.0f32; 1326];
        let result = format_range_as_hands(&range);
        assert!(result.contains("AA:6/6:1.00"), "result={result}");
        assert!(result.contains("22:6/6:1.00"), "result={result}");
    }

    #[test]
    fn format_range_uniform_suited_have_4_combos() {
        let range = vec![1.0f32; 1326];
        let result = format_range_as_hands(&range);
        assert!(result.contains("AKs:4/4:1.00"), "result={result}");
    }

    #[test]
    fn format_range_uniform_offsuit_have_12_combos() {
        let range = vec![1.0f32; 1326];
        let result = format_range_as_hands(&range);
        assert!(result.contains("AKo:12/12:1.00"), "result={result}");
    }

    #[test]
    fn format_range_empty_returns_empty_string() {
        let range = vec![0.0f32; 1326];
        let result = format_range_as_hands(&range);
        assert!(result.is_empty(), "expected empty, got: {result}");
    }

    #[test]
    fn format_range_sorted_by_descending_weight() {
        let mut range = vec![0.5f32; 1326];
        use range_solver::card::card_pair_to_index;
        let aa_pairs = [(48,49),(48,50),(48,51),(49,50),(49,51),(50,51)];
        for (c1, c2) in &aa_pairs {
            range[card_pair_to_index(*c1, *c2)] = 1.0;
        }
        let result = format_range_as_hands(&range);
        let aa_pos = result.find("AA:").expect("AA not found");
        let kk_pos = result.find("KK:").expect("KK not found");
        assert!(aa_pos < kk_pos, "AA ({aa_pos}) should appear before KK ({kk_pos})");
    }

    // ---------------------------------------------------------------
    // format_cfvs_as_hands tests
    // ---------------------------------------------------------------

    #[test]
    fn format_cfvs_uniform_range() {
        let cfvs = vec![10.0f32; 1326];
        let result = format_cfvs_as_hands(&cfvs);
        assert!(result.contains("AA:+10.00"), "result={result}");
    }

    #[test]
    fn format_cfvs_negative_values() {
        let cfvs = vec![-5.0f32; 1326];
        let result = format_cfvs_as_hands(&cfvs);
        assert!(result.contains("AA:-5.00"), "result={result}");
    }

    #[test]
    fn format_cfvs_all_zero_returns_empty() {
        let cfvs = vec![0.0f32; 1326];
        let result = format_cfvs_as_hands(&cfvs);
        assert!(result.is_empty(), "expected empty, got: {result}");
    }

    // ---------------------------------------------------------------
    // TraceFilter tests
    // ---------------------------------------------------------------

    #[test]
    fn trace_filter_none_matches_nothing() {
        let f = TraceFilter::None;
        assert!(!f.matches(0));
        assert!(!f.matches(42));
    }

    #[test]
    fn trace_filter_all_matches_everything() {
        let f = TraceFilter::All;
        assert!(f.matches(0));
        assert!(f.matches(999));
    }

    #[test]
    fn trace_filter_set_matches_only_members() {
        let f = TraceFilter::Set(HashSet::from([0, 42, 100]));
        assert!(f.matches(0));
        assert!(f.matches(42));
        assert!(f.matches(100));
        assert!(!f.matches(1));
        assert!(!f.matches(99));
    }

    #[test]
    fn trace_filter_last_stores_max() {
        let f = TraceFilter::Last(99);
        assert!(!f.matches(0));
        assert!(!f.matches(98));
        assert!(f.matches(99));
    }

    #[test]
    fn parse_trace_filter_none_from_none() {
        let f = TraceFilter::parse(None);
        assert!(matches!(f, TraceFilter::None));
    }

    #[test]
    fn parse_trace_filter_all() {
        let f = TraceFilter::parse(Some("all"));
        assert!(matches!(f, TraceFilter::All));
    }

    #[test]
    fn parse_trace_filter_csv() {
        let f = TraceFilter::parse(Some("0,42,100"));
        match f {
            TraceFilter::Set(s) => {
                assert!(s.contains(&0));
                assert!(s.contains(&42));
                assert!(s.contains(&100));
                assert_eq!(s.len(), 3);
            }
            _ => panic!("expected Set, got {f:?}"),
        }
    }

    #[test]
    fn parse_trace_iters_last() {
        let f = TraceFilter::parse_iters("last", 200);
        assert!(matches!(f, TraceFilter::Last(199)));
    }

    #[test]
    fn parse_trace_iters_all() {
        let f = TraceFilter::parse_iters("all", 200);
        assert!(matches!(f, TraceFilter::All));
    }

    #[test]
    fn parse_trace_iters_csv() {
        let f = TraceFilter::parse_iters("0,9", 200);
        match f {
            TraceFilter::Set(s) => {
                assert!(s.contains(&0));
                assert!(s.contains(&9));
                assert_eq!(s.len(), 2);
            }
            _ => panic!("expected Set, got {f:?}"),
        }
    }

    // ---------------------------------------------------------------
    // TraceConfig tests
    // ---------------------------------------------------------------

    #[test]
    fn trace_config_into_tracer_none_when_no_boundaries() {
        let config = TraceConfig {
            boundaries: None,
            iters_str: "last".to_string(),
            dir: PathBuf::from("/tmp/test"),
        };
        assert!(config.into_tracer(100).is_none());
    }

    #[test]
    fn trace_config_into_tracer_some_when_boundaries_set() {
        let config = TraceConfig {
            boundaries: Some("0,42".to_string()),
            iters_str: "last".to_string(),
            dir: PathBuf::from("/tmp/test"),
        };
        let tracer = config.into_tracer(100);
        assert!(tracer.is_some());
        let tracer = tracer.unwrap();
        assert!(tracer.enabled(0, 99));
        assert!(!tracer.enabled(0, 98));
        assert!(tracer.enabled(42, 99));
        assert!(!tracer.enabled(1, 99));
    }

    // ---------------------------------------------------------------
    // BoundaryTracer tests
    // ---------------------------------------------------------------

    #[test]
    fn tracer_enabled_respects_both_filters() {
        let tracer = BoundaryTracer::new(
            TraceFilter::Set(HashSet::from([0, 42])),
            TraceFilter::Set(HashSet::from([5, 9])),
            PathBuf::from("/tmp/test_traces"),
        );
        assert!(tracer.enabled(0, 5));
        assert!(tracer.enabled(42, 9));
        assert!(!tracer.enabled(0, 3));
        assert!(!tracer.enabled(1, 5));
        assert!(!tracer.enabled(99, 99));
    }

    #[test]
    fn tracer_disabled_when_ordinals_none() {
        let tracer = BoundaryTracer::new(
            TraceFilter::None,
            TraceFilter::All,
            PathBuf::from("/tmp/test_traces"),
        );
        assert!(!tracer.enabled(0, 0));
    }

    // ---------------------------------------------------------------
    // format_trace_txt tests
    // ---------------------------------------------------------------

    #[test]
    fn format_trace_txt_header_line() {
        let event = BoundaryTraceEvent {
            board: "JdTh9dQc".to_string(),
            pot: 88,
            stack: 78.0,
            spot: Some("sb:2bb,bb:10bb,sb:22bb,bb:call|Jd9d7d|bb:check,sb:bet44%,bb:call|3c".to_string()),
            oop_range_1326: vec![1.0; 1326],
            ip_range_1326: vec![0.5; 1326],
            oop_cfvs_1326: vec![5.0; 1326],
            ip_cfvs_1326: vec![-3.0; 1326],
            strategy_at_prev: None,
        };
        let txt = format_trace_txt(50, 42, &event);
        let first_line = txt.lines().next().unwrap();
        assert!(
            first_line.starts_with("[iter=50 boundary=42 board=JdTh9dQc pot=88 stack=78 spr=0.89]"),
            "header mismatch: {first_line}"
        );
    }

    #[test]
    fn format_trace_txt_spot_line() {
        let event = BoundaryTraceEvent {
            board: "JdTh9dQc".to_string(),
            pot: 88,
            stack: 78.0,
            spot: Some("sb:2bb,bb:call|Jd9d7d|bb:check,sb:bet44%,bb:call|3c".to_string()),
            oop_range_1326: vec![1.0; 1326],
            ip_range_1326: vec![0.5; 1326],
            oop_cfvs_1326: vec![5.0; 1326],
            ip_cfvs_1326: vec![-3.0; 1326],
            strategy_at_prev: None,
        };
        let txt = format_trace_txt(0, 0, &event);
        let lines: Vec<&str> = txt.lines().collect();
        assert!(
            lines[1].starts_with("Spot: sb:2bb,bb:call|Jd9d7d|bb:check,sb:bet44%,bb:call|3c"),
            "spot line mismatch: {}", lines[1]
        );
    }

    #[test]
    fn format_trace_txt_range_combo_count() {
        let mut oop_range = vec![0.0f32; 1326];
        // Set exactly 10 combos to nonzero
        for i in 0..10 {
            oop_range[i] = 1.0;
        }
        let event = BoundaryTraceEvent {
            board: "AhKd2c".to_string(),
            pot: 100,
            stack: 50.0,
            spot: None,
            oop_range_1326: oop_range,
            ip_range_1326: vec![0.5; 1326],
            oop_cfvs_1326: vec![0.0; 1326],
            ip_cfvs_1326: vec![0.0; 1326],
            strategy_at_prev: None,
        };
        let txt = format_trace_txt(0, 0, &event);
        assert!(
            txt.contains("OOP range (10 combos)"),
            "should show 10 combos, got:\n{txt}"
        );
    }

    #[test]
    fn format_trace_txt_spr_calculation() {
        let event = BoundaryTraceEvent {
            board: "AhKd2c".to_string(),
            pot: 100,
            stack: 200.0,
            spot: None,
            oop_range_1326: vec![1.0; 1326],
            ip_range_1326: vec![1.0; 1326],
            oop_cfvs_1326: vec![0.0; 1326],
            ip_cfvs_1326: vec![0.0; 1326],
            strategy_at_prev: None,
        };
        let txt = format_trace_txt(0, 0, &event);
        // spr = stack / pot = 200 / 100 = 2.00
        assert!(txt.contains("spr=2.00"), "spr should be 2.00, got:\n{txt}");
    }

    #[test]
    fn format_trace_txt_with_strategy() {
        let event = BoundaryTraceEvent {
            board: "AhKd2c".to_string(),
            pot: 100,
            stack: 50.0,
            spot: None,
            oop_range_1326: vec![1.0; 1326],
            ip_range_1326: vec![1.0; 1326],
            oop_cfvs_1326: vec![0.0; 1326],
            ip_cfvs_1326: vec![0.0; 1326],
            strategy_at_prev: Some(StrategyAtPrevDecision {
                node_idx: 1234,
                player: 0,
                actions: vec!["check".to_string(), "bet33%".to_string(), "allin".to_string()],
                by_hand: vec![
                    ("AA".to_string(), vec![0.0, 0.30, 0.70]),
                    ("KK".to_string(), vec![0.20, 0.80, 0.00]),
                ],
            }),
        };
        let txt = format_trace_txt(0, 0, &event);
        assert!(
            txt.contains("Strategy at preceding decision (node #1234, OOP to act):"),
            "strategy header missing, got:\n{txt}"
        );
        assert!(
            txt.contains("Actions: [check, bet33%, allin]"),
            "actions line missing, got:\n{txt}"
        );
        assert!(
            txt.contains("AA:  [0.00, 0.30, 0.70]"),
            "AA strategy line missing, got:\n{txt}"
        );
    }

    #[test]
    fn format_trace_txt_ends_with_separator() {
        let event = BoundaryTraceEvent {
            board: "AhKd2c".to_string(),
            pot: 100,
            stack: 50.0,
            spot: None,
            oop_range_1326: vec![0.0; 1326],
            ip_range_1326: vec![0.0; 1326],
            oop_cfvs_1326: vec![0.0; 1326],
            ip_cfvs_1326: vec![0.0; 1326],
            strategy_at_prev: None,
        };
        let txt = format_trace_txt(0, 0, &event);
        assert!(txt.ends_with("---\n"), "should end with --- separator");
    }

    #[test]
    fn format_trace_txt_no_spot_line_when_none() {
        let event = BoundaryTraceEvent {
            board: "AhKd2c".to_string(),
            pot: 100,
            stack: 50.0,
            spot: None,
            oop_range_1326: vec![0.0; 1326],
            ip_range_1326: vec![0.0; 1326],
            oop_cfvs_1326: vec![0.0; 1326],
            ip_cfvs_1326: vec![0.0; 1326],
            strategy_at_prev: None,
        };
        let txt = format_trace_txt(0, 0, &event);
        assert!(!txt.contains("Spot:"), "should not have Spot line when None");
    }

    // ---------------------------------------------------------------
    // count_nonzero_combos test
    // ---------------------------------------------------------------

    #[test]
    fn count_nonzero_combos_counts_correctly() {
        let mut range = vec![0.0f32; 1326];
        range[0] = 1.0;
        range[5] = 0.5;
        range[100] = 0.001;
        // 0.0009 is below threshold, should not count
        range[200] = 0.0009;
        assert_eq!(count_nonzero_combos(&range), 3);
    }

    // ---------------------------------------------------------------
    // tracer writes .txt files
    // ---------------------------------------------------------------

    #[test]
    fn tracer_writes_txt_file() {
        let dir = std::env::temp_dir().join("boundary_trace_test_txt");
        let _ = std::fs::remove_dir_all(&dir);
        let tracer = BoundaryTracer::new(
            TraceFilter::All,
            TraceFilter::All,
            dir.clone(),
        );
        let event = BoundaryTraceEvent {
            board: "Jd9d7dQc".to_string(),
            pot: 88,
            stack: 78.0,
            spot: None,
            oop_range_1326: vec![1.0; 1326],
            ip_range_1326: vec![0.5; 1326],
            oop_cfvs_1326: vec![5.0; 1326],
            ip_cfvs_1326: vec![-3.0; 1326],
            strategy_at_prev: None,
        };
        tracer.trace(42, 0, &event);
        tracer.trace(42, 1, &event);

        let path = dir.join("boundary_42.txt");
        assert!(path.exists(), "trace file should exist as .txt");
        let content = std::fs::read_to_string(&path).unwrap();
        // Two records, each ending with "---\n"
        let separators: Vec<_> = content.match_indices("---").collect();
        assert_eq!(separators.len(), 2, "expected 2 records, got {}", separators.len());

        // First record should have iter=0, second iter=1
        assert!(content.contains("[iter=0 boundary=42"), "first record header");
        assert!(content.contains("[iter=1 boundary=42"), "second record header");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tracer_skips_disabled_ordinals() {
        let dir = std::env::temp_dir().join("boundary_trace_test_skip_txt");
        let _ = std::fs::remove_dir_all(&dir);
        let tracer = BoundaryTracer::new(
            TraceFilter::Set(HashSet::from([10])),
            TraceFilter::All,
            dir.clone(),
        );
        let event = BoundaryTraceEvent {
            board: "Ah2c3d".to_string(),
            pot: 100,
            stack: 50.0,
            spot: None,
            oop_range_1326: vec![1.0; 1326],
            ip_range_1326: vec![1.0; 1326],
            oop_cfvs_1326: vec![0.0; 1326],
            ip_cfvs_1326: vec![0.0; 1326],
            strategy_at_prev: None,
        };
        tracer.trace(5, 0, &event);
        let path = dir.join("boundary_5.txt");
        assert!(!path.exists(), "file should not exist for disabled ordinal");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---------------------------------------------------------------
    // build_preceding_decision_map tests
    // ---------------------------------------------------------------

    #[test]
    fn preceding_decision_map_has_entry_per_boundary() {
        let game = make_depth_limited_turn_game();
        let n = game.num_boundary_nodes();
        assert!(n > 0);
        let map = build_preceding_decision_map(&game);
        assert_eq!(map.len(), n);
    }

    #[test]
    fn preceding_decision_map_node_is_decision() {
        let game = make_depth_limited_turn_game();
        let map = build_preceding_decision_map(&game);
        for &(node_idx, ref actions) in map.values() {
            let node = game.node_at(node_idx);
            assert!(
                !node.is_terminal() && !node.is_chance(),
                "preceding node must be a decision node"
            );
            assert!(!actions.is_empty(), "should have at least one action label");
        }
    }
}
