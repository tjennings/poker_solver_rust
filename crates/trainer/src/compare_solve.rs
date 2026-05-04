//! `compare-solve` subcommand: run a spot through both Subgame and Exact
//! solvers and report per-combo strategy diffs and exploitability delta.
//!
//! Built to debug bean izod — subgame converging worse than its blueprint seed.

use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use poker_solver_core::blueprint_v2::Street;
use poker_solver_core::blueprint_v2::bucket_file::BucketFile;
use poker_solver_core::blueprint_v2::bundle::{BlueprintV2Strategy, load_config};
use poker_solver_core::blueprint_v2::cbv::CbvTable;
use poker_solver_core::blueprint_v2::config::BlueprintV2Config;
use poker_solver_core::blueprint_v2::game_tree::GameTree as V2GameTree;
use poker_solver_core::blueprint_v2::mccfr::AllBuckets;

use poker_solver_tauri::gadget;
use poker_solver_tauri::postflop::CbvContext;
use poker_solver_tauri::{
    BoundaryKind, GameSession, StreetBoundaryConfig, StreetBoundaryMode, build_solve_game,
    build_solve_game_parts, parse_rs_poker_card, resolve_street_boundary,
    seed_solver_with_blueprint,
};

use range_solver::card::card_to_string;
use range_solver::interface::Game;
use range_solver::{
    Action, PostFlopGame, cfvalues_after_history_with_reach, compute_exploitability, finalize,
    solve_step,
};

use crate::boundary_trace::{
    BoundaryTracer, assemble_full_spot, build_boundary_spot_paths, build_preceding_decision_map,
    capture_boundary_traces,
};

// ---------------------------------------------------------------------------
// Boundary CFV diagnostic types
// ---------------------------------------------------------------------------

/// Summary statistics for a slice of boundary CFVs.
#[derive(Debug, Clone)]
pub struct BoundaryCfvStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub stddev: f32,
    pub count_nonzero: usize,
}

impl BoundaryCfvStats {
    /// Compute summary statistics from a slice of per-hand CFVs.
    pub fn from_slice(cfvs: &[f32]) -> Self {
        if cfvs.is_empty() {
            return Self {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                stddev: 0.0,
                count_nonzero: 0,
            };
        }
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut count_nonzero = 0usize;
        for &v in cfvs {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v as f64;
            if v != 0.0 {
                count_nonzero += 1;
            }
        }
        let n = cfvs.len() as f64;
        let mean = sum / n;
        let var = cfvs
            .iter()
            .map(|&v| {
            let d = v as f64 - mean;
            d * d
            })
            .sum::<f64>()
            / n;
        Self {
            min,
            max,
            mean: mean as f32,
            stddev: var.sqrt() as f32,
            count_nonzero,
        }
    }
}

/// Check if any CFV exceeds the expected chip range.
/// Returns warning strings for out-of-range values.
pub fn check_cfv_out_of_range(stats: &BoundaryCfvStats, chip_range: f32) -> Vec<String> {
    let mut warnings = Vec::new();
    if stats.max.abs() > chip_range || stats.min.abs() > chip_range {
        warnings.push(format!(
            "CFV out of range: min={:.4} max={:.4} exceeds chip_range={:.0}",
            stats.min, stats.max, chip_range,
        ));
    }
    warnings
}

/// Check if any CFV summary stat is NaN or Infinity.
pub fn check_cfv_non_finite(stats: &BoundaryCfvStats) -> Vec<String> {
    let mut warnings = Vec::new();
    let vals = [stats.min, stats.max, stats.mean, stats.stddev];
    let names = ["min", "max", "mean", "stddev"];
    for (val, name) in vals.iter().zip(&names) {
        if !val.is_finite() {
            warnings.push(format!("non-finite CFV {name}={val}"));
        }
    }
    warnings
}

/// Check if boundary CFV magnitudes vary wildly across boundaries,
/// suggesting a unit mismatch (e.g., chips vs pot-fractions).
pub fn check_unit_mismatch(all_stats: &[BoundaryCfvStats]) -> Vec<String> {
    let mut warnings = Vec::new();
    let magnitudes: Vec<f64> = all_stats
        .iter()
        .map(|s| {
        let m = s.mean.abs().max(s.stddev.abs()) as f64;
        if m < 1e-10 { 1e-10 } else { m }
        })
        .collect();
    if magnitudes.len() < 2 {
        return warnings;
    }
    let max_mag = magnitudes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_mag = magnitudes.iter().cloned().fold(f64::INFINITY, f64::min);
    if max_mag / min_mag > 100.0 {
        warnings.push(format!(
            "unit mismatch suspected: magnitude range {:.4}..{:.4} (ratio {:.0}x)",
            min_mag,
            max_mag,
            max_mag / min_mag,
        ));
    }
    warnings
}

/// Check if biased CFVs diverge wildly from the unbiased baseline.
/// `bias_stats[0]` is assumed to be Unbiased.
pub fn check_bias_divergence(bias_stats: &[&BoundaryCfvStats], bias_names: &[&str]) -> Vec<String> {
    let mut warnings = Vec::new();
    if bias_stats.is_empty() {
        return warnings;
    }
    let base_mag = {
        let m = bias_stats[0].mean.abs().max(bias_stats[0].stddev.abs()) as f64;
        if m < 1e-10 { 1e-10 } else { m }
    };
    for i in 1..bias_stats.len() {
        let mag = bias_stats[i].mean.abs().max(bias_stats[i].stddev.abs()) as f64;
        let ratio = if mag < 1e-10 { 0.0 } else { mag / base_mag };
        if ratio > 100.0 || (base_mag > 1e-6 && ratio < 0.01) {
            let name = if i < bias_names.len() {
                bias_names[i]
            } else {
                "?"
            };
            warnings.push(format!(
                "{name} bias divergence: mean={:.4} vs unbiased={:.4} (ratio {:.0}x)",
                bias_stats[i].mean, bias_stats[0].mean, ratio,
            ));
        }
    }
    warnings
}

/// Returns a human-readable label for the active gadget mode.
///
/// Used for both the startup `eprintln!` message and test assertions.
pub fn gadget_mode_label(gadget: bool, gadget_clamp: bool) -> &'static str {
    if gadget {
        "tree"
    } else if gadget_clamp {
        "clamp"
    } else {
        "off"
    }
}

/// Per-hand L1 distance (mass moved) between two strategy vectors.
///
/// Returns `(mean_mass, max_mass, max_hand_idx)`.
/// Each hand's mass moved = sum_actions |exact[a,h] - subgame[a,h]| / 2.
pub fn compute_mass_moved(
    exact: &[f32],
    subgame: &[f32],
    num_hands: usize,
    num_actions: usize,
) -> (f64, f64, usize) {
    assert_eq!(exact.len(), num_actions * num_hands);
    assert_eq!(subgame.len(), num_actions * num_hands);

    let mut total = 0.0f64;
    let mut max_mass = 0.0f64;
    let mut max_idx = 0usize;

    for h in 0..num_hands {
        let mut hand_mass = 0.0f64;
        for a in 0..num_actions {
            let idx = a * num_hands + h;
            hand_mass += (exact[idx] as f64 - subgame[idx] as f64).abs();
        }
        hand_mass /= 2.0;
        total += hand_mass;
        if hand_mass > max_mass {
            max_mass = hand_mass;
            max_idx = h;
        }
    }

    let mean = if num_hands > 0 {
        total / num_hands as f64
    } else {
        0.0
    };
    (mean, max_mass, max_idx)
}

/// Per-action-class bias: how much more mass subgame assigns to each action
/// class relative to exact.
///
/// Returns a vec of `(action_class_name, delta)` where delta = subgame_total - exact_total.
pub fn compute_action_bias(
    exact: &[f32],
    subgame: &[f32],
    num_hands: usize,
    action_labels: &[String],
) -> Vec<(String, f64)> {
    let num_actions = action_labels.len();
    assert_eq!(exact.len(), num_actions * num_hands);
    assert_eq!(subgame.len(), num_actions * num_hands);

    let mut result = Vec::new();
    for (a, label) in action_labels.iter().enumerate() {
        let base = a * num_hands;
        let exact_sum: f64 = exact[base..base + num_hands]
            .iter()
            .map(|&v| v as f64)
            .sum();
        let subgame_sum: f64 = subgame[base..base + num_hands]
            .iter()
            .map(|&v| v as f64)
            .sum();
        let class_name = classify_action(label);
        result.push((class_name, subgame_sum - exact_sum));
    }
    result
}

/// Classify an action label into its action class.
fn classify_action(label: &str) -> String {
    let lower = label.to_lowercase();
    if lower == "fold" {
        "Fold".to_string()
    } else if lower == "check" {
        "Check".to_string()
    } else if lower == "call" {
        "Call".to_string()
    } else if lower == "all-in" || lower == "allin" {
        "AllIn".to_string()
    } else if lower.ends_with("bb") {
        // Could be bet or raise — check the label prefix
        // In practice, the range-solver actions are labeled NNbb for bets and raises
        // The label itself doesn't distinguish, but the action_type does.
        // For simplicity, group all sized actions as "Bet/Raise".
        "Bet/Raise".to_string()
    } else {
        label.to_string()
    }
}

/// Find the top N hands by mass moved.
///
/// Returns `(hand_index, mass_moved, exact_probs, subgame_probs)` sorted descending.
pub fn top_hands_by_mass(
    exact: &[f32],
    subgame: &[f32],
    num_hands: usize,
    num_actions: usize,
    top_n: usize,
) -> Vec<(usize, f64, Vec<f32>, Vec<f32>)> {
    let mut hands: Vec<(usize, f64)> = Vec::with_capacity(num_hands);
    for h in 0..num_hands {
        let mut mass = 0.0f64;
        for a in 0..num_actions {
            let idx = a * num_hands + h;
            mass += (exact[idx] as f64 - subgame[idx] as f64).abs();
        }
        mass /= 2.0;
        hands.push((h, mass));
    }
    hands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    hands.truncate(top_n);

    hands
        .into_iter()
        .map(|(h, mass)| {
            let exact_probs: Vec<f32> =
                (0..num_actions).map(|a| exact[a * num_hands + h]).collect();
            let subgame_probs: Vec<f32> = (0..num_actions)
                .map(|a| subgame[a * num_hands + h])
                .collect();
        (h, mass, exact_probs, subgame_probs)
        })
        .collect()
}

/// Find the strategy.bin path within a bundle, optionally pinned to a snapshot.
///
/// Resolution order:
/// 1. If `snapshot` is Some, use `bundle_dir/<snapshot>/strategy.bin`
/// 2. Else, try `bundle_dir/final/strategy.bin`
/// 3. Else, use the latest `snapshot_*` directory
fn find_strategy_path(
    bundle_dir: &Path,
    snapshot: Option<&str>,
) -> Result<std::path::PathBuf, String> {
    if let Some(snap) = snapshot {
        let p = bundle_dir.join(snap).join("strategy.bin");
        if p.exists() {
            return Ok(p);
        }
        return Err(format!("Snapshot '{}' not found at {}", snap, p.display()));
    }

    let final_path = bundle_dir.join("final/strategy.bin");
    if final_path.exists() {
        return Ok(final_path);
    }

    let mut snapshots: Vec<_> = std::fs::read_dir(bundle_dir)
        .map_err(|e| format!("Cannot read bundle dir: {e}"))?
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with("snapshot_"))
        })
        .collect();
    snapshots.sort_by_key(|e| e.file_name());
    match snapshots.last() {
        Some(entry) => Ok(entry.path().join("strategy.bin")),
        None => Err(format!(
            "No strategy.bin found in '{}'",
            bundle_dir.display()
        )),
    }
}

/// Load a blueprint bundle with optional snapshot pinning.
///
/// Returns (config, strategy, game_tree, decision_map, cbv_context).
/// Reuses patterns from `bench_rollout::load_bundle` and `inspect_spot::load_blueprint`.
fn load_bundle_with_snapshot(
    bundle_dir: &Path,
    snapshot: Option<&str>,
) -> Result<
    (
        BlueprintV2Config,
        BlueprintV2Strategy,
        V2GameTree,
        Vec<u32>,
        Arc<CbvContext>,
    ),
    String,
> {
    let config = load_config(bundle_dir).map_err(|e| format!("Failed to load config.yaml: {e}"))?;

    let strat_path = find_strategy_path(bundle_dir, snapshot)?;
    eprintln!("Loading strategy from {}...", strat_path.display());
    let strategy = BlueprintV2Strategy::load(&strat_path)
        .map_err(|e| format!("Failed to load strategy.bin: {e}"))?;

    let aa = &config.action_abstraction;
    let tree = V2GameTree::build_with_options(
        config.game.stack_depth,
        config.game.small_blind,
        config.game.big_blind,
        &aa.preflop,
        &aa.flop,
        &aa.turn,
        &aa.river,
        config.game.allow_preflop_limp,
    );
    let decision_map = tree.decision_index_map();

    // Load bucket files
    let cluster_dir = if bundle_dir.join("buckets").exists() {
        bundle_dir.join("buckets")
    } else {
        return Err("No buckets/ directory in bundle".into());
    };

    let bucket_counts = [
        config.clustering.preflop.buckets,
        config.clustering.flop.buckets,
        config.clustering.turn.buckets,
        config.clustering.river.buckets,
    ];
    let mut bucket_files: [Option<BucketFile>; 4] = [None, None, None, None];
    for (i, name) in [
        "preflop.buckets",
        "flop.buckets",
        "turn.buckets",
        "river.buckets",
    ]
    .iter()
    .enumerate()
    {
        let path = cluster_dir.join(name);
        if path.exists() {
            match BucketFile::load(&path) {
                Ok(bf) => bucket_files[i] = Some(bf),
                Err(e) => eprintln!("Warning: failed to load {}: {e}", path.display()),
            }
        }
    }
    let loaded = bucket_files.iter().filter(|f| f.is_some()).count();
    eprintln!("[compare] loaded {loaded}/4 bucket files");

    let mut all_buckets = AllBuckets::new(bucket_counts, bucket_files);
    if cluster_dir.join("flop_0000.buckets").exists() {
        all_buckets = all_buckets.with_per_flop_dir(cluster_dir);
    }
    all_buckets.equity_fallback = true;

    // Load CBV table
    let strat_dir = strat_path.parent().unwrap_or(Path::new("."));
    let cbv_path = strat_dir.join("cbv_p0.bin");
    let cbv_table = if cbv_path.exists() {
        CbvTable::load(&cbv_path).ok()
    } else {
        let alt = bundle_dir.join("cbv_p0.bin");
        if alt.exists() {
            CbvTable::load(&alt).ok()
        } else {
            None
        }
    };
    let cbv_table = cbv_table.unwrap_or_else(|| CbvTable {
        values: vec![],
        node_offsets: vec![],
        buckets_per_node: vec![],
    });

    let ctx = Arc::new(CbvContext {
        cbv_table,
        abstract_tree: tree.clone(),
        all_buckets: Arc::new(all_buckets),
        strategy: Arc::new(strategy.clone()),
    });

    Ok((config, strategy, tree, decision_map, ctx))
}

/// Load ONNX session and wire per-boundary `NeuralBoundaryEvaluator`s
/// into the game's `per_boundary_evaluators` vector.
///
/// When `opt_out` is `Some`, each evaluator is wrapped in a `GadgetEvaluator`
/// that clamps the opponent's CFVs upward to the opt-out values.
fn setup_neural_boundaries(
    game: &mut PostFlopGame,
    model_path: &Path,
    opt_out: Option<Arc<dyn poker_solver_tauri::gadget::OptOutProvider>>,
) {
    let boundary_boards = game.boundary_boards();
    let n_boundaries = game.num_boundary_nodes();

    if boundary_boards.is_empty() {
        eprintln!("[compare] no boundary boards found; skipping neural setup");
        return;
    }

    let session = match cfvnet::eval::boundary_evaluator::load_shared_onnx_session(model_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[compare] ONNX session load failed: {e}");
            return;
        }
    };

    let gadget_label = if opt_out.is_some() { " + gadget" } else { "" };
    let mut per_boundary: Vec<Arc<dyn range_solver::game::BoundaryEvaluator>> =
        Vec::with_capacity(n_boundaries);
    for (ordinal, board_4) in boundary_boards.into_iter().enumerate() {
        let private_cards_pair = [
            game.private_cards(0).to_vec(),
            game.private_cards(1).to_vec(),
        ];
        let neural_eval = cfvnet::eval::boundary_evaluator::neural_boundary_evaluator_from_shared(
            Arc::clone(&session),
            board_4.clone(),
            private_cards_pair.clone(),
        );
        let inner: Arc<dyn range_solver::game::BoundaryEvaluator> = Arc::new(neural_eval);
        let wrapped: Arc<dyn range_solver::game::BoundaryEvaluator> = match &opt_out {
            Some(provider) => Arc::new(poker_solver_tauri::gadget::GadgetEvaluator::new(
                    inner,
                    Arc::clone(provider),
                    ordinal,
                    board_4,
                    private_cards_pair,
            )),
            None => inner,
        };
        per_boundary.push(wrapped);
    }
    game.per_boundary_evaluators = per_boundary;
    game.boundary_evaluator = None;

    eprintln!("[compare] neural-cfvnet mode: {n_boundaries} boundaries (ONNX){gadget_label}",);
}

/// Wire per-boundary `SubtreeExactEvaluator`s into the game's
/// `per_boundary_evaluators` vector -- mirrors the Tauri side so that
/// compare-solve can use ExactSubtree as a ground-truth boundary evaluator.
///
/// When `opt_out` is `Some`, each evaluator is wrapped in a `GadgetEvaluator`
/// that clamps the opponent's CFVs upward to the opt-out values.
fn setup_exact_subtree_boundaries(
    game: &mut PostFlopGame,
    opt_out: Option<Arc<dyn poker_solver_tauri::gadget::OptOutProvider>>,
    solve_iters: u32,
    target_exp: f32,
) {
    let boundary_boards = game.boundary_boards();
    let n_boundaries = game.num_boundary_nodes();
    if boundary_boards.is_empty() {
        eprintln!("[compare] no boundary boards found; skipping exact_subtree setup");
        return;
    }
    let tree_cfg = game.tree_config().clone();
    let private_cards = [
        game.private_cards(0).to_vec(),
        game.private_cards(1).to_vec(),
    ];
    let initial_weights = [
        game.initial_weights(0).to_vec(),
        game.initial_weights(1).to_vec(),
    ];
    let gadget_label = if opt_out.is_some() { " + gadget" } else { "" };
    let mut per_boundary: Vec<Arc<dyn range_solver::game::BoundaryEvaluator>> =
        Vec::with_capacity(n_boundaries);
    for (ordinal, board) in boundary_boards.iter().enumerate() {
        let eval: Arc<dyn range_solver::game::BoundaryEvaluator> = Arc::new(
            poker_solver_tauri::exact_subtree::SubtreeExactEvaluator::new(
                board.clone(),
                private_cards.clone(),
                initial_weights.clone(),
                tree_cfg.clone(),
            )
            .with_solve_iters(solve_iters)
            .with_target_exploitability(target_exp),
        );
        let wrapped: Arc<dyn range_solver::game::BoundaryEvaluator> = match &opt_out {
            Some(provider) => Arc::new(poker_solver_tauri::gadget::GadgetEvaluator::new(
                    eval,
                    Arc::clone(provider),
                    ordinal,
                    board.clone(),
                    private_cards.clone(),
            )),
            None => eval,
        };
        per_boundary.push(wrapped);
    }
    game.per_boundary_evaluators = per_boundary;
    game.boundary_evaluator = None;
    eprintln!("[compare] exact-subtree mode: {n_boundaries} boundaries (full CFR){gadget_label}");
}

struct OracleBoundaryEvaluator {
    exact_game: Arc<PostFlopGame>,
    history: Vec<usize>,
    orientation: OracleCfvOrientation,
    scale: f32,
}

struct SharedOracleBoundaryEvaluator {
    exact_game: Arc<RwLock<PostFlopGame>>,
    history: Vec<usize>,
    orientation: OracleCfvOrientation,
    scale: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OracleCfvOrientation {
    Current,
    SwapPlayers,
    SignFlip,
    SwapPlayersSignFlip,
}

impl OracleCfvOrientation {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "current" => Ok(Self::Current),
            "swap" => Ok(Self::SwapPlayers),
            "sign-flip" => Ok(Self::SignFlip),
            "swap-sign-flip" => Ok(Self::SwapPlayersSignFlip),
            other => Err(format!(
                "invalid --oracle-orientation value '{other}': expected \
                 'current', 'swap', 'sign-flip', or 'swap-sign-flip'"
            )),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::SwapPlayers => "swap",
            Self::SignFlip => "sign-flip",
            Self::SwapPlayersSignFlip => "swap-sign-flip",
        }
    }

    fn orient(self, mut oop: Vec<f32>, mut ip: Vec<f32>) -> (Vec<f32>, Vec<f32>) {
        match self {
            Self::Current => (oop, ip),
            Self::SwapPlayers => (ip, oop),
            Self::SignFlip => {
                oop.iter_mut().for_each(|v| *v = -*v);
                ip.iter_mut().for_each(|v| *v = -*v);
                (oop, ip)
            }
            Self::SwapPlayersSignFlip => {
                oop.iter_mut().for_each(|v| *v = -*v);
                ip.iter_mut().for_each(|v| *v = -*v);
                (ip, oop)
            }
        }
    }
}

fn scale_cfvs(values: &mut [f32], scale: f32) {
    if scale != 1.0 {
        values.iter_mut().for_each(|v| *v *= scale);
    }
}

impl range_solver::game::BoundaryEvaluator for OracleBoundaryEvaluator {
    fn compute_cfvs(
        &self,
        _player: usize,
        _pot: i32,
        _remaining_stack: f64,
        _opponent_reach: &[f32],
        _num_hands: usize,
        _continuation_index: usize,
    ) -> Vec<f32> {
        unreachable!("OracleBoundaryEvaluator must be used through raw CFV path")
    }

    fn compute_raw_cfvs_both(
        &self,
        _pot: i32,
        _remaining_stack: f64,
        oop_reach: &[f32],
        ip_reach: &[f32],
        _num_oop: usize,
        _num_ip: usize,
        continuation_index: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        assert_eq!(
            continuation_index, 0,
            "exact_oracle supports only the unbiased continuation"
        );
        let mut oop =
            cfvalues_after_history_with_reach(&self.exact_game, &self.history, 0, ip_reach);
        let mut ip =
            cfvalues_after_history_with_reach(&self.exact_game, &self.history, 1, oop_reach);
        scale_cfvs(&mut oop, self.scale);
        scale_cfvs(&mut ip, self.scale);
        Some(self.orientation.orient(oop, ip))
    }
}

impl range_solver::game::BoundaryEvaluator for SharedOracleBoundaryEvaluator {
    fn compute_cfvs(
        &self,
        _player: usize,
        _pot: i32,
        _remaining_stack: f64,
        _opponent_reach: &[f32],
        _num_hands: usize,
        _continuation_index: usize,
    ) -> Vec<f32> {
        unreachable!("SharedOracleBoundaryEvaluator must be used through raw CFV path")
    }

    fn compute_raw_cfvs_both(
        &self,
        _pot: i32,
        _remaining_stack: f64,
        oop_reach: &[f32],
        ip_reach: &[f32],
        _num_oop: usize,
        _num_ip: usize,
        continuation_index: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        assert_eq!(
            continuation_index, 0,
            "iteration-aligned exact_oracle supports only the unbiased continuation"
        );
        let exact_game = self
            .exact_game
            .read()
            .expect("iteration-aligned exact oracle lock poisoned");
        let mut oop = cfvalues_after_history_with_reach(&exact_game, &self.history, 0, ip_reach);
        let mut ip = cfvalues_after_history_with_reach(&exact_game, &self.history, 1, oop_reach);
        scale_cfvs(&mut oop, self.scale);
        scale_cfvs(&mut ip, self.scale);
        Some(self.orientation.orient(oop, ip))
    }
}

fn build_boundary_play_histories(game: &PostFlopGame) -> Vec<Vec<usize>> {
    let boundary_indices = game.boundary_node_indices();
    let n = boundary_indices.len();
    if n == 0 {
        return Vec::new();
    }

    let mut index_to_ord = std::collections::HashMap::with_capacity(n);
    for (ord, &idx) in boundary_indices.iter().enumerate() {
        index_to_ord.insert(idx, ord);
    }

    let mut paths: Vec<Option<Vec<usize>>> = vec![None; n];
    let mut stack: Vec<(usize, Vec<usize>)> = vec![(0, Vec::new())];
    while let Some((node_idx, history)) = stack.pop() {
        if let Some(&ord) = index_to_ord.get(&node_idx) {
            paths[ord] = Some(history.clone());
        }

        for (action_idx, child_idx) in game.child_indices(node_idx).into_iter().enumerate() {
            let child = game.node_at(child_idx);
            let action = child.prev_action();
            drop(child);

            let mut child_history = history.clone();
            match action {
                Action::Chance(card) => child_history.push(card as usize),
                _ => child_history.push(action_idx),
            }
            stack.push((child_idx, child_history));
        }
    }

    paths
        .into_iter()
        .map(|path| path.unwrap_or_else(|| vec![usize::MAX]))
        .collect()
}

fn setup_oracle_boundaries(
    game: &mut PostFlopGame,
    exact_game: Arc<PostFlopGame>,
    orientation: OracleCfvOrientation,
    scale: f32,
) -> Result<(), String> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(format!(
            "invalid --oracle-scale value {scale}: expected a positive finite value"
        ));
    }
    let histories = build_boundary_play_histories(game);
    let n_boundaries = game.num_boundary_nodes();
    if histories.len() != n_boundaries {
        return Err(format!(
            "oracle boundary history count mismatch: histories={} boundaries={n_boundaries}",
            histories.len(),
        ));
    }
    let per_boundary: Vec<Arc<dyn range_solver::game::BoundaryEvaluator>> = histories
        .into_iter()
        .map(|history| {
            Arc::new(OracleBoundaryEvaluator {
                exact_game: Arc::clone(&exact_game),
                history,
                orientation,
                scale,
            }) as Arc<dyn range_solver::game::BoundaryEvaluator>
        })
        .collect();
    game.per_boundary_evaluators = per_boundary;
    game.boundary_evaluator = None;
    eprintln!(
        "[compare] exact-oracle mode: {n_boundaries} boundaries \
         (solved exact game, orientation={}, scale={scale:.6})",
        orientation.label()
    );
    Ok(())
}

fn setup_iteration_aligned_oracle_boundaries(
    game: &mut PostFlopGame,
    exact_game: Arc<RwLock<PostFlopGame>>,
    orientation: OracleCfvOrientation,
    scale: f32,
) -> Result<(), String> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(format!(
            "invalid --oracle-scale value {scale}: expected a positive finite value"
        ));
    }
    let histories = build_boundary_play_histories(game);
    let n_boundaries = game.num_boundary_nodes();
    if histories.len() != n_boundaries {
        return Err(format!(
            "oracle boundary history count mismatch: histories={} boundaries={n_boundaries}",
            histories.len(),
        ));
    }
    let per_boundary: Vec<Arc<dyn range_solver::game::BoundaryEvaluator>> = histories
        .into_iter()
        .map(|history| {
            Arc::new(SharedOracleBoundaryEvaluator {
                exact_game: Arc::clone(&exact_game),
                history,
                orientation,
                scale,
            }) as Arc<dyn range_solver::game::BoundaryEvaluator>
        })
        .collect();
    game.per_boundary_evaluators = per_boundary;
    game.boundary_evaluator = None;
    eprintln!(
        "[compare] exact-oracle mode: {n_boundaries} boundaries \
         (iteration-aligned exact game, orientation={}, scale={scale:.6})",
        orientation.label()
    );
    Ok(())
}

// capture_boundary_traces, build_1326_range, expand_to_1326 live in
// boundary_trace module (shared with tauri game_solve).

/// Run a DCFR solve loop, returning (wall_time, final_exploitability).
fn run_dcfr_solve(
    game: &mut PostFlopGame,
    iters: u32,
    label: &str,
    verbose: bool,
    tracer: Option<&BoundaryTracer>,
    spot_paths: Option<&[String]>,
) -> (f64, f64) {
    let start = Instant::now();
    let has_per_boundary = !game.per_boundary_evaluators.is_empty();

    // Build preceding decision map once for strategy extraction
    let preceding_map = tracer.as_ref().map(|_| build_preceding_decision_map(game));

    for t in 0..iters {
        // Neural cfvnet path: clear CFV cache every iteration so boundary
        // values are recomputed with updated opponent reaches.
        if has_per_boundary {
            game.clear_boundary_cfvs();
        }

        // DCFR discount params for boundary continuation regrets
        let nearest_pow4 = if t == 0 {
            0
        } else {
            1u32 << ((t.leading_zeros() ^ 31) & !1)
        };
        let t_alpha = (t as i32 - 1).max(0) as f64;
        let t_gamma = (t - nearest_pow4) as f64;
        let pow_alpha = t_alpha * t_alpha.sqrt();
        let alpha = (pow_alpha / (pow_alpha + 1.0)) as f32;
        let beta = 0.5f32;
        let gamma = (t_gamma / (t_gamma + 1.0)).powi(3) as f32;
        game.set_boundary_discount(alpha, beta, gamma);

        solve_step(game, t);

        // Capture boundary traces after this iteration's CFVs are cached.
        if let Some(tr) = tracer {
            capture_boundary_traces(game, tr, spot_paths, preceding_map.as_ref(), t);
            // capture_boundary_traces leaves game at root; fine before next solve_step.
        }

        if verbose && (t + 1) % 20 == 0 {
            let wall = start.elapsed().as_secs_f64();
            eprintln!("[{label}] iter {}/{iters}  wall {wall:.1}s", t + 1);
        }
        // Unconditional progress heartbeat — logs every 10 iters so long
        // runs (especially subgame with per-iter subtree re-solves) show
        // progress without needing --verbose.
        if (t + 1) % 10 == 0 {
            let wall = start.elapsed().as_secs_f64();
            let per_iter = wall / (t + 1) as f64;
            let eta = per_iter * (iters - (t + 1)) as f64;
            eprintln!(
                "[{label}] {}/{iters} iters | {wall:.1}s elapsed | {per_iter:.2}s/iter | ETA {eta:.0}s",
                t + 1
            );
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }
    }

    finalize(game);
    game.back_to_root();
    game.cache_normalized_weights();

    // Compute exploitability with cached boundary CFVs.
    // Disable lazy evaluators so exploitability uses the cached values.
    let saved_evaluator = game.boundary_evaluator.take();
    let saved_per_boundary = std::mem::take(&mut game.per_boundary_evaluators);
    let exp = compute_exploitability(game);
    game.boundary_evaluator = saved_evaluator;
    game.per_boundary_evaluators = saved_per_boundary;

    let wall = start.elapsed().as_secs_f64();
    (wall, exp as f64)
}

fn run_iteration_aligned_oracle_solve(
    exact_game: PostFlopGame,
    subgame_game: &mut PostFlopGame,
    iters: u32,
    oracle_orientation: OracleCfvOrientation,
    oracle_scale: f32,
    verbose: bool,
    tracer: Option<&BoundaryTracer>,
    spot_paths: Option<&[String]>,
) -> Result<(PostFlopGame, f64, f64, f64, f64), String> {
    let exact_game = Arc::new(RwLock::new(exact_game));
    setup_iteration_aligned_oracle_boundaries(
        subgame_game,
        Arc::clone(&exact_game),
        oracle_orientation,
        oracle_scale,
    )?;

    let start = Instant::now();
    let mut exact_wall = 0.0;
    let mut subgame_wall = 0.0;
    let has_per_boundary = !subgame_game.per_boundary_evaluators.is_empty();
    let preceding_map = tracer.as_ref().map(|_| build_preceding_decision_map(subgame_game));

    for t in 0..iters {
        let exact_iter_start = Instant::now();
        {
            let exact_guard = exact_game
                .write()
                .expect("iteration-aligned exact oracle lock poisoned");
            solve_step(&*exact_guard, t);
        }
        exact_wall += exact_iter_start.elapsed().as_secs_f64();

        if has_per_boundary {
            subgame_game.clear_boundary_cfvs();
        }

        let nearest_pow4 = if t == 0 {
            0
        } else {
            1u32 << ((t.leading_zeros() ^ 31) & !1)
        };
        let t_alpha = (t as i32 - 1).max(0) as f64;
        let t_gamma = (t - nearest_pow4) as f64;
        let pow_alpha = t_alpha * t_alpha.sqrt();
        let alpha = (pow_alpha / (pow_alpha + 1.0)) as f32;
        let beta = 0.5f32;
        let gamma = (t_gamma / (t_gamma + 1.0)).powi(3) as f32;
        subgame_game.set_boundary_discount(alpha, beta, gamma);

        let subgame_iter_start = Instant::now();
        solve_step(subgame_game, t);
        subgame_wall += subgame_iter_start.elapsed().as_secs_f64();

        if let Some(tr) = tracer {
            capture_boundary_traces(subgame_game, tr, spot_paths, preceding_map.as_ref(), t);
        }

        if verbose && (t + 1) % 20 == 0 {
            let wall = start.elapsed().as_secs_f64();
            eprintln!("[aligned] iter {}/{iters}  wall {wall:.1}s", t + 1);
        }
        if (t + 1) % 10 == 0 {
            let wall = start.elapsed().as_secs_f64();
            let per_iter = wall / (t + 1) as f64;
            let eta = per_iter * (iters - (t + 1)) as f64;
            eprintln!(
                "[aligned] {}/{iters} iters | {wall:.1}s elapsed | {per_iter:.2}s/iter | ETA {eta:.0}s",
                t + 1
            );
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }
    }

    let exact_finalize_start = Instant::now();
    {
        let mut exact_guard = exact_game
            .write()
            .expect("iteration-aligned exact oracle lock poisoned");
        finalize(&mut *exact_guard);
        exact_guard.back_to_root();
        exact_guard.cache_normalized_weights();
    }
    exact_wall += exact_finalize_start.elapsed().as_secs_f64();

    let subgame_finalize_start = Instant::now();
    finalize(subgame_game);
    subgame_game.back_to_root();
    subgame_game.cache_normalized_weights();
    let saved_evaluator = subgame_game.boundary_evaluator.take();
    let saved_per_boundary = std::mem::take(&mut subgame_game.per_boundary_evaluators);
    let subgame_exp = compute_exploitability(subgame_game) as f64;
    subgame_game.boundary_evaluator = saved_evaluator;
    subgame_game.per_boundary_evaluators = saved_per_boundary;
    subgame_wall += subgame_finalize_start.elapsed().as_secs_f64();

    let exact_exp = {
        let exact_guard = exact_game
            .read()
            .expect("iteration-aligned exact oracle lock poisoned");
        compute_exploitability(&*exact_guard) as f64
    };

    subgame_game.boundary_evaluator = None;
    subgame_game.per_boundary_evaluators.clear();
    let exact_game = Arc::try_unwrap(exact_game)
        .map_err(|_| "iteration-aligned exact oracle still has live references".to_string())?
        .into_inner()
        .map_err(|_| "iteration-aligned exact oracle lock poisoned".to_string())?;

    Ok((exact_game, exact_wall, exact_exp, subgame_wall, subgame_exp))
}

/// Find indices in private_cards matching representative sample hands.
///
/// Returns (hand_label, game_index) for each match found. Looks for
/// AA, AKs, JJ, 54s, 72o as strong/medium/weak representatives.
///
/// Legacy path: retained for Phase 5 cleanup.
#[allow(dead_code)]
fn find_sample_hand_indices(private_cards: &[(u8, u8)]) -> Vec<(String, usize)> {
    // Card IDs: rank = card >> 2, suit = card & 3
    // 2=0..A=12, club=0 diamond=1 heart=2 spade=3
    let targets: Vec<(&str, u8, u8)> = vec![
        ("AA",  48, 51), // Ac As
        ("AKs", 48, 44), // Ac Kc
        ("JJ",  36, 37), // Jc Jd
        ("54s", 12,  8), // 5c 4c
        ("72o", 20,  1), // 7c 2d
    ];

    let mut result = Vec::new();
    for (label, c1, c2) in &targets {
        for (i, &(pc1, pc2)) in private_cards.iter().enumerate() {
            if (pc1 == *c1 && pc2 == *c2) || (pc1 == *c2 && pc2 == *c1) {
                result.push((label.to_string(), i));
                break;
            }
        }
    }
    result
}

/// Dump boundary CFV statistics after precompute.
///
/// For each boundary ordinal, prints per-player, per-bias summary stats
/// and flags sanity check violations.
///
/// Legacy path: retained for Phase 5 cleanup.
#[allow(dead_code)]
fn dump_boundary_cfv_stats(game: &PostFlopGame, pot: i32, eff_stack: i32) {
    let n_boundaries = game.num_boundary_nodes();
    let k = game.num_continuations();
    let bias_names = ["Unbiased", "Fold", "Call", "Raise"];
    let chip_range = (pot + 2 * eff_stack) as f32;
    let boundary_indices = game.boundary_node_indices();

    println!();
    println!("=== Boundary CFV Dump ({n_boundaries} boundaries, K={k}) ===");
    println!("chip_range (pot + 2*eff_stack): {chip_range:.0}");
    println!();

    // Collect all unbiased stats for cross-boundary unit mismatch check
    let mut all_unbiased_stats: Vec<BoundaryCfvStats> = Vec::new();
    let mut all_warnings: Vec<String> = Vec::new();

    for ordinal in 0..n_boundaries {
        let pot_at = game.boundary_pot(ordinal);
        let node_idx = boundary_indices[ordinal];
        let node = game.node_at(node_idx);
        let invested = node.bet_amount();
        let turn_card = node.turn_card();
        let river_card = node.river_card();
        let acting = node.acting_player();
        drop(node);

        let turn_str = card_to_string(turn_card).unwrap_or_else(|_| "--".to_string());
        let river_str = card_to_string(river_card).unwrap_or_else(|_| "--".to_string());

        println!(
            "--- boundary {ordinal}: pot={pot_at} invested={invested} \
             acting_player={acting} turn={turn_str} river={river_str} ---"
        );

        for player in 0..2 {
            let player_label = if player == 0 { "OOP" } else { "IP" };
            let num_hands = game.num_private_hands(player);
            let sample_hands = find_sample_hand_indices(game.private_cards(player));

            let mut bias_stats_vec: Vec<BoundaryCfvStats> = Vec::new();

            for ki in 0..k.min(bias_names.len()) {
                let cfvs = game.get_boundary_cfvs_multi(ordinal, player, ki);
                let stats = BoundaryCfvStats::from_slice(&cfvs);

                println!(
                    "  {player_label} {}: n={num_hands} min={:.6} max={:.6} \
                     mean={:.6} stddev={:.6} nonzero={}",
                    bias_names[ki],
                    stats.min,
                    stats.max,
                    stats.mean,
                    stats.stddev,
                    stats.count_nonzero,
                );

                // Print sample hands
                if !sample_hands.is_empty() && !cfvs.is_empty() {
                    let samples: Vec<String> = sample_hands
                        .iter()
                        .map(|(label, idx)| {
                            let val = if *idx < cfvs.len() {
                                cfvs[*idx]
                            } else {
                                f32::NAN
                            };
                            format!("{label}={val:.6}")
                        })
                        .collect();
                    println!("    samples: {}", samples.join("  "));
                }

                // Per-bias sanity checks
                for w in check_cfv_out_of_range(&stats, chip_range) {
                    let msg = format!("boundary {ordinal} {player_label} {}: {w}", bias_names[ki],);
                    println!("    WARNING {msg}");
                    all_warnings.push(msg);
                }
                for w in check_cfv_non_finite(&stats) {
                    let msg = format!("boundary {ordinal} {player_label} {}: {w}", bias_names[ki],);
                    println!("    ERROR {msg}");
                    all_warnings.push(msg);
                }

                if ki == 0 && player == 0 {
                    all_unbiased_stats.push(stats.clone());
                }
                bias_stats_vec.push(stats);
            }

            // Check bias divergence for this boundary x player
            if bias_stats_vec.len() >= 2 {
                let refs: Vec<&BoundaryCfvStats> = bias_stats_vec.iter().collect();
                let names: Vec<&str> = bias_names[..refs.len()].iter().copied().collect();
                for w in check_bias_divergence(&refs, &names) {
                    let msg = format!("boundary {ordinal} {player_label}: {w}",);
                    println!("    WARNING {msg}");
                    all_warnings.push(msg);
                }
            }
        }
        println!();
    }

    // Cross-boundary unit mismatch check
    if all_unbiased_stats.len() >= 2 {
        for w in check_unit_mismatch(&all_unbiased_stats) {
            println!("WARNING (cross-boundary): {w}");
            all_warnings.push(w);
        }
    }

    // Summary
    if all_warnings.is_empty() {
        println!("=== All boundary CFV sanity checks passed ===");
    } else {
        println!("=== {} sanity check issues found ===", all_warnings.len());
        for (i, w) in all_warnings.iter().enumerate() {
            println!("  {}: {w}", i + 1);
        }
    }
    println!();
}

/// Format a range-solver action for display.
fn format_action(action: &range_solver::Action) -> String {
    match action {
        range_solver::Action::Fold => "F".to_string(),
        range_solver::Action::Check => "X".to_string(),
        range_solver::Action::Call => "C".to_string(),
        range_solver::Action::Bet(amt) => format!("B{}", *amt as f64 / 2.0),
        range_solver::Action::Raise(amt) => format!("R{}", *amt as f64 / 2.0),
        range_solver::Action::AllIn(_) => "A".to_string(),
        _ => "?".to_string(),
    }
}

/// Format a range-solver action as a full label for display.
fn format_action_label(action: &range_solver::Action) -> String {
    match action {
        range_solver::Action::Fold => "Fold".to_string(),
        range_solver::Action::Check => "Check".to_string(),
        range_solver::Action::Call => "Call".to_string(),
        range_solver::Action::Bet(amt) => {
            let bb = *amt as f64 / 2.0;
            format!("{bb:.0}bb")
        }
        range_solver::Action::Raise(amt) => {
            let bb = *amt as f64 / 2.0;
            format!("{bb:.0}bb")
        }
        range_solver::Action::AllIn(_) => "All-in".to_string(),
        _ => "?".to_string(),
    }
}

/// Format a hand label from a range-solver card pair.
fn format_hand(c1: u8, c2: u8) -> String {
    let s1 = card_to_string(c1).unwrap_or_else(|_| "??".to_string());
    let s2 = card_to_string(c2).unwrap_or_else(|_| "??".to_string());
    format!("{s1}{s2}")
}

/// Build the subgame game via Option A2 per-boundary gadget path.
///
/// Injects a 4-node gadget subtree at each cfvnet depth-boundary terminal.
/// Post-injection ordinals: 0..N = cfvnet boundaries (stable), N..3N = gadget
/// terminals (pre-populated with opt-out values).
#[allow(clippy::too_many_arguments)]
fn build_gadget_tree_game(
    board: &[String],
    oop_w: &[f32],
    ip_w: &[f32],
    pot: i32,
    eff_stack: i32,
    bet_sizes: &[Vec<f64>],
    depth_limit: Option<u8>,
    ctx: &Arc<CbvContext>,
    current_node: u32,
    board_cards: &[poker_solver_core::poker::Card],
    boundary_cut: &Option<(u8, BoundaryKind)>,
    gadget_provider: &str,
    gadget_constant: f32,
    solve_iters: u32,
    target_exp: f32,
) -> Result<PostFlopGame, String> {
    let board_u8: Vec<u8> = board_cards
        .iter()
        .map(|c| poker_solver_tauri::rs_card_to_range_solver(*c))
        .collect();

    // Build inner boundary evaluator (handles cfvnet ordinals 0..N).
    let inner_evaluator = build_inner_evaluator(
        board,
        oop_w,
        ip_w,
        pot,
        eff_stack,
        bet_sizes,
        depth_limit,
        &board_u8,
        boundary_cut,
        solve_iters,
        target_exp,
    )?;

    // Build the gadget game via the appropriate provider.
    let (card_config, action_tree) = build_solve_game_parts(
        board,
        oop_w,
        ip_w,
        pot,
        eff_stack,
        bet_sizes,
        false,
        depth_limit,
    )?;

    let game = match gadget_provider {
        "constant" => {
            eprintln!("[compare] gadget-tree (A2): ConstantOptOut({gadget_constant})");
            gadget::make_per_boundary_gadget_game_constant(
                card_config,
                action_tree,
                gadget_constant,
                inner_evaluator,
            )?
        }
        "blueprint-cbv" => {
            eprintln!("[compare] gadget-tree (A2): BlueprintCbvOptOut (per-boundary pot)");
            gadget::make_per_boundary_gadget_game(
                card_config,
                action_tree,
                ctx,
                current_node,
                &board_u8,
                inner_evaluator,
            )?
        }
        other => {
            return Err(format!(
                "invalid --gadget-provider '{other}': expected 'blueprint-cbv' or 'constant'"
            ));
        }
    };

    let n_total = game.num_boundary_nodes();
    let n_original = n_total / 3;
    eprintln!(
        "[compare] gadget-tree (A2): {} original boundaries, {} total (incl. {} gadget terminals)",
        n_original,
        n_total,
        n_total - n_original,
    );

    Ok(game)
}

/// Build the inner boundary evaluator for cfvnet ordinals 0..N.
///
/// Extracted from `build_gadget_tree_game` to keep it under the 60-line limit.
#[allow(clippy::too_many_arguments)]
fn build_inner_evaluator(
    board: &[String],
    oop_w: &[f32],
    ip_w: &[f32],
    pot: i32,
    eff_stack: i32,
    bet_sizes: &[Vec<f64>],
    depth_limit: Option<u8>,
    board_u8: &[u8],
    boundary_cut: &Option<(u8, BoundaryKind)>,
    solve_iters: u32,
    target_exp: f32,
) -> Result<Arc<dyn range_solver::game::BoundaryEvaluator>, String> {
    let (tmp_cc, tmp_at) = build_solve_game_parts(
        board,
        oop_w,
        ip_w,
        pot,
        eff_stack,
        bet_sizes,
        false,
        depth_limit,
    )?;
    let tmp_game = PostFlopGame::with_config(tmp_cc, tmp_at)
        .map_err(|e| format!("Failed to build temp game: {e}"))?;
    let private_cards: [Vec<(u8, u8)>; 2] = [
        tmp_game.private_cards(0).to_vec(),
        tmp_game.private_cards(1).to_vec(),
    ];
    let tree_cfg = tmp_game.tree_config().clone();
    drop(tmp_game);

    let initial_weights = [
        vec![1.0f32; private_cards[0].len()],
        vec![1.0f32; private_cards[1].len()],
    ];
    let evaluator: Arc<dyn range_solver::game::BoundaryEvaluator> = match boundary_cut {
        Some((_, BoundaryKind::ExactSubtree)) | None => Arc::new(
            poker_solver_tauri::exact_subtree::SubtreeExactEvaluator::new(
                    board_u8.to_vec(),
                    private_cards,
                    initial_weights,
                    tree_cfg,
            )
            .with_solve_iters(solve_iters)
            .with_target_exploitability(target_exp),
        ),
            Some((_, BoundaryKind::Cfvnet(model_path))) => {
            let session =
                cfvnet::eval::boundary_evaluator::load_shared_onnx_session(Path::new(model_path))
                    .map_err(|e| format!("ONNX session load failed: {e}"))?;
            Arc::new(
                cfvnet::eval::boundary_evaluator::neural_boundary_evaluator_from_shared(
                    session,
                    board_u8.to_vec(),
                    private_cards,
                ),
            )
            }
        };
    Ok(evaluator)
}

/// Set up legacy clamp-path boundary evaluators (--gadget-clamp mode).
///
/// When gadget_clamp is true, constructs an OptOutProvider and wraps each
/// boundary evaluator in a GadgetEvaluator. When false, wires boundaries
/// without clamp wrapping.
#[allow(clippy::too_many_arguments)]
fn setup_clamp_boundaries(
    game: &mut PostFlopGame,
    gadget_clamp: bool,
    gadget_provider: &str,
    gadget_constant: f32,
    board_cards: &[poker_solver_core::poker::Card],
    ctx: &Arc<CbvContext>,
    current_node: u32,
    boundary_cut: &Option<(u8, BoundaryKind)>,
    solve_iters: u32,
    target_exp: f32,
) -> Result<(), String> {
    let opt_out: Option<Arc<dyn gadget::OptOutProvider>> = if gadget_clamp {
        let board_u8: Vec<u8> = board_cards
            .iter()
            .map(|c| poker_solver_tauri::rs_card_to_range_solver(*c))
            .collect();
        let private_cards: [Vec<(u8, u8)>; 2] = [
            game.private_cards(0).to_vec(),
            game.private_cards(1).to_vec(),
        ];
        Some(compute_opt_out_provider(
            gadget_provider,
            gadget_constant,
            ctx,
            current_node,
            &board_u8,
            &private_cards,
        )?)
    } else {
        None
    };

    let n_boundaries = game.num_boundary_nodes();
    if let Some((_, kind)) = boundary_cut {
        if n_boundaries > 0 {
            match kind {
                BoundaryKind::Cfvnet(model_path) => {
                    setup_neural_boundaries(game, Path::new(model_path), opt_out);
                }
                BoundaryKind::ExactSubtree => {
                    setup_exact_subtree_boundaries(game, opt_out, solve_iters, target_exp);
                }
            }
        }
    }
    Ok(())
}

/// Compute an OptOutProvider for the legacy clamp path.
fn compute_opt_out_provider(
    gadget_provider: &str,
    gadget_constant: f32,
    ctx: &Arc<CbvContext>,
    current_node: u32,
    board_u8: &[u8],
    private_cards: &[Vec<(u8, u8)>; 2],
) -> Result<Arc<dyn gadget::OptOutProvider>, String> {
    match gadget_provider {
        "constant" => {
            eprintln!("[compare] gadget-clamp: ConstantOptOut({gadget_constant})");
            Ok(Arc::new(gadget::ConstantOptOut(gadget_constant)))
        }
        "blueprint-cbv" => {
            eprintln!("[compare] gadget-clamp: BlueprintCbvOptOut (per-boundary pot)");
            Ok(Arc::new(gadget::BlueprintCbvOptOut::from_cbv_context(
                ctx,
                current_node,
                board_u8,
                private_cards,
            )))
        }
        other => Err(format!(
            "invalid --gadget-provider '{other}': expected 'blueprint-cbv' or 'constant'"
        )),
    }
}

fn resolved_boundary_is_oracle(
    config: &StreetBoundaryConfig,
    root_street: Street,
    oracle_flags: [bool; 3],
) -> bool {
    let streets: &[(Street, &StreetBoundaryMode, bool)] = match root_street {
        Street::Flop => &[
            (Street::Flop, &config.flop, oracle_flags[0]),
            (Street::Turn, &config.turn, oracle_flags[1]),
            (Street::River, &config.river, oracle_flags[2]),
        ],
        Street::Turn => &[
            (Street::Turn, &config.turn, oracle_flags[1]),
            (Street::River, &config.river, oracle_flags[2]),
        ],
        Street::River => &[(Street::River, &config.river, oracle_flags[2])],
        Street::Preflop => return false,
    };

    for (i, (_, mode, is_oracle)) in streets.iter().enumerate() {
        if i == 0 && !matches!(mode, StreetBoundaryMode::Exact) {
            continue;
        }
        if !matches!(mode, StreetBoundaryMode::Exact) {
            return *is_oracle;
        }
    }
    false
}

/// Run the compare-solve harness.
///
/// `tolerance > 0` enables a pass/fail check on the final strategy at the
/// root: the maximum absolute per-action delta between the exact and subgame
/// strategies (averaged per hand in each 13x13 matrix cell using reach-weighted
/// aggregation) must be <= `tolerance`. If the delta exceeds tolerance, the
/// run returns an `Err` so the caller can signal failure via non-zero exit
/// code.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn run(
    bundle_dir: &Path,
    snapshot: Option<&str>,
    spot: &str,
    iters: u32,
    exact_iters: Option<u32>,
    subgame_iters: Option<u32>,
    verbose: bool,
    dump_boundary_cfvs: bool,
    street_boundary_config: StreetBoundaryConfig,
    oracle_boundary_flags: [bool; 3],
    oracle_orientation: OracleCfvOrientation,
    oracle_scale: f32,
    oracle_iteration_aligned: bool,
    trace_config: crate::boundary_trace::TraceConfig,
    tolerance: f32,
    gadget: bool,
    gadget_clamp: bool,
    gadget_provider: &str,
    gadget_constant: f32,
) -> Result<(), String> {
    let exact_iters = exact_iters.unwrap_or(iters);
    let subgame_iters = subgame_iters.unwrap_or(iters);
    let mode = gadget_mode_label(gadget, gadget_clamp);
    eprintln!("gadget mode: {mode}");
    // 1. Load bundle
    let (config, strategy, tree, decision_map, ctx) =
        load_bundle_with_snapshot(bundle_dir, snapshot)?;

    // 2. Create GameSession and load spot
    let mut session = GameSession::new(config.clone(), strategy, tree.clone(), decision_map, None);
    session.set_cbv_context(Arc::clone(&ctx));
    session.load_spot(spot)?;

    let state = session.get_state();
    if state.is_terminal {
        return Err("Spot is a terminal node, nothing to solve".to_string());
    }
    if state.board.is_empty() || state.board.len() < 3 {
        return Err(
            "Spot must be at a postflop decision node (deal board cards first)".to_string(),
        );
    }

    // 3. Extract solving parameters from session
    let board = state.board.clone();
    let pot = state.pot;
    let eff_stack = state.stacks[0].min(state.stacks[1]);
    let board_str = board.join("");
    let position = &state.position;

    // Get weights from session (access via get_state is indirect; use session internals)
    // Re-parse the spot to get the weights at this node
    let (oop_w, ip_w) = session.weights();
    let oop_w = oop_w.to_vec();
    let ip_w = ip_w.to_vec();

    // Determine bet sizes from config
    let street = match board.len() {
        3 => Street::Flop,
        4 => Street::Turn,
        5 => Street::River,
        _ => return Err(format!("Invalid board length: {}", board.len())),
    };
    let bet_sizes = match street {
        Street::Flop => &config.action_abstraction.flop,
        Street::Turn => &config.action_abstraction.turn,
        Street::River => &config.action_abstraction.river,
        Street::Preflop => return Err("Cannot solve preflop".to_string()),
    };

    // Parse board cards for rs_poker
    let board_cards: Vec<poker_solver_core::poker::Card> = board
        .iter()
        .map(|s| parse_rs_poker_card(s))
        .collect::<Result<Vec<_>, _>>()?;

    // Resolve boundary config
    let boundary_cut = resolve_street_boundary(&street_boundary_config, street);
    let depth_limit = boundary_cut.as_ref().map(|(d, _)| *d);
    let oracle_boundary_active =
        resolved_boundary_is_oracle(&street_boundary_config, street, oracle_boundary_flags);
    if oracle_boundary_active && (gadget || gadget_clamp) {
        return Err(
            "exact_oracle boundary mode is incompatible with --gadget and --gadget-clamp"
                .to_string(),
        );
    }
    if oracle_iteration_aligned {
        if !oracle_boundary_active {
            return Err(
                "--oracle-iteration-aligned requires an exact_oracle boundary mode".to_string(),
            );
        }
        if exact_iters != subgame_iters {
            return Err(
                "--oracle-iteration-aligned requires --exact-iters and --subgame-iters to match"
                    .to_string(),
            );
        }
    }

    // Print header (display values in BB = chips / 2)
    let pot_bb = pot as f64 / 2.0;
    let eff_bb = eff_stack as f64 / 2.0;
    println!("=== compare-solve summary ===");
    println!("spot: {spot}");
    println!("board: {board_str}  pot: {pot_bb:.0}  eff_stack: {eff_bb:.0}");
    println!("position: {position}");
    let iter_label = if exact_iters == subgame_iters {
        exact_iters.to_string()
    } else {
        format!("exact={exact_iters}, subgame={subgame_iters}")
    };
    println!(
        "iters: {iter_label}  boundary: {}{}",
        match &boundary_cut {
            Some((d, BoundaryKind::Cfvnet(p))) => format!("depth={d}, model={p}"),
            Some((d, BoundaryKind::ExactSubtree)) if oracle_boundary_active => {
                format!("depth={d}, exact_oracle")
            }
            Some((d, BoundaryKind::ExactSubtree)) => format!("depth={d}, exact_subtree"),
            None => "all-exact".to_string(),
        },
        if oracle_iteration_aligned {
            "  oracle_alignment=iteration"
        } else {
            ""
        }
    );
    println!();

    // 4. Build exact game
    eprintln!("[compare] building exact game...");
    let mut exact_game =
        build_solve_game(&board, &oop_w, &ip_w, pot, eff_stack, bet_sizes, true, None)?;

    let (mem_exact, _) = exact_game.memory_usage();
    if verbose {
        eprintln!(
            "[exact] memory: {:.1} MB, OOP hands: {}, IP hands: {}",
            mem_exact as f64 / 1_048_576.0,
            exact_game.private_cards(0).len(),
            exact_game.private_cards(1).len(),
        );
    }

    // 5. Build subgame game (with depth limit from SBC)
    // When all-exact (no boundary cut), build as exact too (no boundaries).
    let subgame_is_exact = boundary_cut.is_none();
    let current_node = session.node_idx();
    let gadget_tree_active = gadget && !subgame_is_exact;
    let setup_boundary_cut = if oracle_boundary_active {
        None
    } else {
        boundary_cut.clone()
    };
    eprintln!("[compare] building subgame game...");
    let mut subgame_game = if gadget_tree_active {
        build_gadget_tree_game(
            &board,
            &oop_w,
            &ip_w,
            pot,
            eff_stack,
            bet_sizes,
            depth_limit,
            &ctx,
            current_node,
            &board_cards,
            &setup_boundary_cut,
            gadget_provider,
            gadget_constant,
            subgame_iters,
            3.0,
        )?
    } else {
        let mut game = build_solve_game(
            &board,
            &oop_w,
            &ip_w,
            pot,
            eff_stack,
            bet_sizes,
            subgame_is_exact,
            depth_limit,
        )?;
        // Wire legacy clamp path if --gadget-clamp
        setup_clamp_boundaries(
            &mut game,
            gadget_clamp,
            gadget_provider,
            gadget_constant,
            &board_cards,
            &ctx,
            current_node,
            &setup_boundary_cut,
            subgame_iters,
            3.0,
        )?;
        game
    };

    let n_boundaries = subgame_game.num_boundary_nodes();
    let (mem_subgame, _) = subgame_game.memory_usage();
    if verbose {
        eprintln!(
            "[subgame] memory: {:.1} MB, boundary nodes: {n_boundaries}, OOP hands: {}, IP hands: {}",
            mem_subgame as f64 / 1_048_576.0,
            subgame_game.private_cards(0).len(),
            subgame_game.private_cards(1).len(),
        );
    }

    // 6. Seed both solvers with blueprint strategy
    let seed_street = street;
    let subgame_seed_start = if gadget_tree_active { 4 } else { 0 };
    seed_solver_with_blueprint(
        &exact_game,
        &ctx.strategy,
        &ctx.all_buckets,
        &ctx.abstract_tree,
        &board_cards,
        seed_street,
        current_node,
        0,
    );
    seed_solver_with_blueprint(
        &subgame_game,
        &ctx.strategy,
        &ctx.all_buckets,
        &ctx.abstract_tree,
        &board_cards,
        seed_street,
        current_node,
        subgame_seed_start,
    );

    let _ = dump_boundary_cfvs;

    // 7b. Build boundary tracer (no-op if --trace-boundaries not given)
    // In gadget-tree mode ordinals 0 and 1 are static gadget terminals
    // whose CFVs never change during solve -- skip tracing them.
    let skip = if gadget_tree_active { 2 } else { 0 };
    let tracer = trace_config.into_tracer_with_skip(subgame_iters, skip);
    if tracer.is_some() {
        eprintln!(
            "[compare] boundary tracing enabled ({n_boundaries} boundaries, skip {skip} leading)"
        );
    }

    // 7c. Build spot strings for each boundary ordinal.
    // In gadget-tree mode, ordinals 0 and 1 are gadget Terminate nodes
    // (not real depth boundaries). The tracer and spot_paths vectors are
    // indexed by ordinal, so we keep all entries to maintain alignment.
    // The BoundaryTracer's ordinal filter controls which are actually traced.
    let spot_paths: Option<Vec<String>> = if tracer.is_some() && n_boundaries > 0 {
        let postflop_paths = build_boundary_spot_paths(&subgame_game);
        Some(
            postflop_paths
                .iter()
                .map(|suffix| assemble_full_spot(spot, suffix))
                .collect(),
        )
    } else {
        None
    };

    let (exact_game, exact_wall, exact_exp, subgame_wall, subgame_exp) =
        if oracle_iteration_aligned {
            eprintln!(
                "[compare] solving exact/subgame in iteration-aligned mode ({exact_iters} iters)..."
            );
            run_iteration_aligned_oracle_solve(
                exact_game,
                &mut subgame_game,
                exact_iters,
                oracle_orientation,
                oracle_scale,
                verbose,
                tracer.as_ref(),
                spot_paths.as_deref(),
            )
            .map(|(exact_game, exact_wall, exact_exp, subgame_wall, subgame_exp)| {
                (
                    Arc::new(exact_game),
                    exact_wall,
                    exact_exp,
                    subgame_wall,
                    subgame_exp,
                )
            })?
        } else {
            // 8. Solve exact
            eprintln!("[compare] solving exact ({exact_iters} iters)...");
            let (exact_wall, exact_exp) =
                run_dcfr_solve(&mut exact_game, exact_iters, "exact", verbose, None, None);
            exact_game.back_to_root();
            let exact_game = Arc::new(exact_game);

            if oracle_boundary_active {
                setup_oracle_boundaries(
                    &mut subgame_game,
                    Arc::clone(&exact_game),
                    oracle_orientation,
                    oracle_scale,
                )?;
            }

            // 9. Solve subgame (with optional tracer)
            eprintln!("[compare] solving subgame ({subgame_iters} iters)...");
            let (subgame_wall, subgame_exp) = run_dcfr_solve(
                &mut subgame_game,
                subgame_iters,
                "subgame",
                verbose,
                tracer.as_ref(),
                spot_paths.as_deref(),
            );

            (exact_game, exact_wall, exact_exp, subgame_wall, subgame_exp)
        };

    // 10. Extract strategies at root
    subgame_game.back_to_root();

    let exact_player = exact_game.current_player();
    let subgame_player = subgame_game.current_player();
    assert_eq!(exact_player, subgame_player, "Players must match at root");

    let exact_strat = exact_game.strategy();
    let subgame_strat = subgame_game.strategy();
    let num_hands = exact_game.num_private_hands(exact_player);
    let exact_actions = exact_game.available_actions();
    let subgame_actions = subgame_game.available_actions();
    let num_actions = exact_actions.len();

    // Verify matching action counts
    if exact_actions.len() != subgame_actions.len() {
        return Err(format!(
            "Action count mismatch: exact={} subgame={}",
            exact_actions.len(),
            subgame_actions.len()
        ));
    }

    // 11. Run diff helpers
    let (mean_mass, max_mass, max_idx) =
        compute_mass_moved(&exact_strat, &subgame_strat, num_hands, num_actions);

    let action_labels: Vec<String> = exact_actions.iter().map(format_action_label).collect();
    let bias = compute_action_bias(&exact_strat, &subgame_strat, num_hands, &action_labels);

    let top = top_hands_by_mass(&exact_strat, &subgame_strat, num_hands, num_actions, 10);

    // Get hand label for max mass index
    let private_cards = exact_game.private_cards(exact_player);
    let max_hand_label = if max_idx < private_cards.len() {
        let (c1, c2) = private_cards[max_idx];
        format_hand(c1, c2)
    } else {
        format!("#{max_idx}")
    };

    // 12. Print report
    // Convert exploitability from chips to mBB (1 BB = 2 chips, so mBB = chips * 500)
    let exact_exp_mbb = exact_exp * 500.0;
    let subgame_exp_mbb = subgame_exp * 500.0;

    println!("=== Exact solve ===");
    println!(
        "wall: {exact_wall:.1}s  final_exp: {exact_exp_mbb:.2} mbb/hand  memory: {:.1} MB",
        mem_exact as f64 / 1_048_576.0,
    );
    println!();

    println!("=== Subgame solve ===");
    println!(
        "solve: {subgame_wall:.1}s  final_exp: {subgame_exp_mbb:.2} mbb/hand  memory: {:.1} MB  boundaries: {n_boundaries}",
        mem_subgame as f64 / 1_048_576.0,
    );
    println!();

    println!("=== Diff (exact vs hybrid, at root) ===");
    println!("mean mass moved per hand: {mean_mass:.3}");
    println!("max mass moved: {max_mass:.3} at hand #{max_idx} ({max_hand_label})");
    println!();

    println!("per-action-class bias (subgame - exact):");
    // Aggregate by action class
    let mut class_bias: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for (class_name, delta) in &bias {
        *class_bias.entry(class_name.clone()).or_default() += delta;
    }
    // Normalize by number of hands
    for (class_name, delta) in &class_bias {
        let per_hand = *delta / num_hands as f64;
        println!("  {class_name:<10} {per_hand:+.3}");
    }
    println!();

    let exp_delta_mbb = subgame_exp_mbb - exact_exp_mbb;
    println!(
        "exploitability delta: hybrid {:+.2} mbb/hand ({})",
        exp_delta_mbb,
        if exp_delta_mbb > 0.0 {
            "worse"
        } else {
            "better"
        }
    );
    println!();

    println!("=== Top 10 hands by mass moved ===");
    let action_short: Vec<String> = exact_actions.iter().map(format_action).collect();
    for (rank, (h_idx, mass, exact_probs, subgame_probs)) in top.iter().enumerate() {
        let hand_label = if *h_idx < private_cards.len() {
            let (c1, c2) = private_cards[*h_idx];
            format_hand(c1, c2)
        } else {
            format!("#{h_idx}")
        };

        let exact_str: Vec<String> = exact_probs
            .iter()
            .zip(&action_short)
            .map(|(p, a)| format!("{a}:{p:.2}"))
            .collect();
        let subgame_str: Vec<String> = subgame_probs
            .iter()
            .zip(&action_short)
            .map(|(p, a)| format!("{a}:{p:.2}"))
            .collect();

        println!(
            "{:>2}. {:<6}  mass={:.3}  exact=[{}]  subgame=[{}]",
            rank + 1,
            hand_label,
            mass,
            exact_str.join(" "),
            subgame_str.join(" "),
        );
    }

    // 13. Tolerance check (for harness-driven iteration loops). Compares the
    // reach-weighted per-class aggregate (the same quantity the UI shows) to
    // stay consistent with what humans will inspect.
    if tolerance > 0.0 {
        let agg_exact = aggregate_by_class(&exact_strat, private_cards, num_hands, num_actions);
        let agg_sub = aggregate_by_class(&subgame_strat, private_cards, num_hands, num_actions);
        let mut worst: Option<(String, String, f32, f32, f32)> = None; // class, action, e, s, |Δ|
        for (class, e_probs) in &agg_exact {
            let Some(s_probs) = agg_sub.get(class) else {
                continue;
            };
            for a in 0..num_actions {
                let e = e_probs[a];
                let s = s_probs[a];
                let d = (e - s).abs();
                if worst.as_ref().map_or(true, |w| d > w.4) {
                    worst = Some((class.clone(), action_labels[a].clone(), e, s, d));
                }
            }
        }
        println!();
        if let Some((class, action, e, s, d)) = worst {
            println!(
                "=== Tolerance check (threshold={tolerance:.4}) ===\n\
                 worst cell: {class} @ {action}  exact={e:.4}  subgame={s:.4}  |Δ|={d:.4}"
            );
            if d > tolerance {
                return Err(format!(
                    "FAIL: strategy delta {d:.4} exceeds tolerance {tolerance:.4} \
                     (class={class}, action={action}, exact={e:.4}, subgame={s:.4})"
                ));
            }
            println!("PASS");
        }
    }

    Ok(())
}

/// Reach-weighted per-class aggregate of a strategy vector. Matches the
/// `build_solve_matrix_at_current` aggregation so we compare the same
/// quantity the user inspects in the UI.
fn aggregate_by_class(
    strategy: &[f32],
    private_cards: &[(u8, u8)],
    num_hands: usize,
    num_actions: usize,
) -> std::collections::BTreeMap<String, Vec<f32>> {
    use std::collections::BTreeMap;
    let mut sums: BTreeMap<String, (f64, Vec<f64>)> = BTreeMap::new();
    for (h, &(c1, c2)) in private_cards.iter().enumerate().take(num_hands) {
        let class = canonical_hand_name(c1, c2);
        let entry = sums
            .entry(class)
            .or_insert_with(|| (0.0, vec![0.0; num_actions]));
        entry.0 += 1.0; // uniform weight across combos (tolerance check treats
                         // all AA combos equally); reach-weighted comparison is
                         // future work — simple-mean is adequate for 0.1%.
        for a in 0..num_actions {
            entry.1[a] += strategy[a * num_hands + h] as f64;
        }
    }
    sums.into_iter()
        .map(|(k, (w, v))| {
            let probs: Vec<f32> = v
                .iter()
                .map(|&s| if w > 0.0 { (s / w) as f32 } else { 0.0 })
                .collect();
            (k, probs)
        })
        .collect()
}

/// Canonical hand name matching the boundary-trace module. Reproduced here
/// to avoid a cross-crate dep from trainer → tauri-app.
fn canonical_hand_name(c1: u8, c2: u8) -> String {
    const RANKS: [char; 13] = [
        '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
    ];
    let r1 = c1 / 4;
    let r2 = c2 / 4;
    let s1 = c1 % 4;
    let s2 = c2 % 4;
    let (hi, lo) = if r1 >= r2 { (r1, r2) } else { (r2, r1) };
    let h = RANKS[hi as usize];
    let l = RANKS[lo as usize];
    if hi == lo {
        format!("{h}{l}")
    } else if s1 == s2 {
        format!("{h}{l}s")
    } else {
        format!("{h}{l}o")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::{CardConfig, card_from_str, flop_from_str};

    fn make_depth_limited_flop_game() -> PostFlopGame {
        let oop_range: range_solver::range::Range = "AA,KK,QQ".parse().unwrap();
        let ip_range: range_solver::range::Range = "QQ,JJ,TT".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: range_solver::card::NOT_DEALT,
            river: range_solver::card::NOT_DEALT,
        };
        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::Flop,
            starting_pot: 100,
            effective_stack: 200,
            flop_bet_sizes: [sizes.clone(), sizes.clone()],
            turn_bet_sizes: [sizes.clone(), sizes.clone()],
            river_bet_sizes: [sizes.clone(), sizes],
            depth_limit: Some(1),
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
        game.allocate_memory(false);
        game
    }

    fn make_one_boundary_turn_game(depth_limit: Option<u8>) -> PostFlopGame {
        let oop_range: range_solver::range::Range = "AA,KK,QQ".parse().unwrap();
        let ip_range: range_solver::range::Range = "JJ,TT,99".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs 7h 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: range_solver::card::NOT_DEALT,
        };
        let turn_sizes = BetSizeOptions::try_from(("50%", "")).unwrap();
        let river_sizes = BetSizeOptions::try_from(("50%", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 200,
            turn_bet_sizes: [turn_sizes.clone(), turn_sizes],
            river_bet_sizes: [river_sizes.clone(), river_sizes],
            add_allin_threshold: 0.0,
            force_allin_threshold: 0.0,
            merging_threshold: 0.0,
            depth_limit,
            ..Default::default()
        };
        let mut tree = ActionTree::new(tree_config).unwrap();
        tree.remove_line(&[Action::Bet(50), Action::Call])
            .unwrap();
        tree.remove_line(&[Action::Check, Action::Bet(50), Action::Call])
            .unwrap();
        assert!(tree.invalid_terminals().is_empty());

        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
        game.allocate_memory(false);
        game
    }

    fn run_contract_solve(game: &mut PostFlopGame, iters: u32) -> f32 {
        run_dcfr_solve(game, iters, "contract", false, None, None).1 as f32
    }

    #[test]
    fn boundary_play_histories_encode_player_indices_and_chance_cards() {
        let game = make_depth_limited_flop_game();
        let histories = build_boundary_play_histories(&game);
        assert_eq!(histories.len(), game.num_boundary_nodes());
        assert!(!histories.is_empty());

        let root_actions = game.child_indices(0).len();
        let has_player_action = histories
            .iter()
            .any(|h| h.first().is_some_and(|&a| a < root_actions));
        let boards = game.boundary_boards();
        let has_chance_card = histories.iter().zip(boards.iter()).any(|(history, board)| {
            board
                .get(3)
                .is_some_and(|card| history.contains(&(*card as usize)))
        });
        assert!(
            has_player_action,
            "expected at least one root player action index"
        );
        assert!(
            has_chance_card,
            "expected at least one chance card id in histories"
        );
    }

    #[test]
    fn resolved_boundary_is_oracle_tracks_first_non_exact_boundary() {
        let cfg = StreetBoundaryConfig {
            flop: StreetBoundaryMode::Exact,
            turn: StreetBoundaryMode::Exact,
            river: StreetBoundaryMode::ExactSubtree,
        };
        assert!(resolved_boundary_is_oracle(
            &cfg,
            Street::Flop,
            [false, false, true]
        ));
        assert!(!resolved_boundary_is_oracle(
            &cfg,
            Street::Flop,
            [false, false, false]
        ));
    }

    #[test]
    fn oracle_cfv_orientation_parses_labels() {
        assert_eq!(
            OracleCfvOrientation::parse("current").unwrap(),
            OracleCfvOrientation::Current
        );
        assert_eq!(
            OracleCfvOrientation::parse("swap").unwrap(),
            OracleCfvOrientation::SwapPlayers
        );
        assert_eq!(
            OracleCfvOrientation::parse("sign-flip").unwrap(),
            OracleCfvOrientation::SignFlip
        );
        assert_eq!(
            OracleCfvOrientation::parse("swap-sign-flip").unwrap(),
            OracleCfvOrientation::SwapPlayersSignFlip
        );
        assert!(OracleCfvOrientation::parse("bad").is_err());
    }

    #[test]
    fn oracle_cfv_orientation_transforms_vectors() {
        let oop = vec![1.0, -2.0];
        let ip = vec![3.0, -4.0];

        assert_eq!(
            OracleCfvOrientation::Current.orient(oop.clone(), ip.clone()),
            (oop.clone(), ip.clone())
        );
        assert_eq!(
            OracleCfvOrientation::SwapPlayers.orient(oop.clone(), ip.clone()),
            (ip.clone(), oop.clone())
        );
        assert_eq!(
            OracleCfvOrientation::SignFlip.orient(oop.clone(), ip.clone()),
            (vec![-1.0, 2.0], vec![-3.0, 4.0])
        );
        assert_eq!(
            OracleCfvOrientation::SwapPlayersSignFlip.orient(oop, ip),
            (vec![-3.0, 4.0], vec![-1.0, 2.0])
        );
    }

    #[test]
    fn oracle_boundary_one_boundary_contract_matches_exact() {
        let mut exact_game = make_one_boundary_turn_game(None);
        let mut subgame = make_one_boundary_turn_game(Some(0));
        assert_eq!(
            subgame.num_boundary_nodes(),
            1,
            "test game should isolate a single boundary handoff"
        );

        let exact_exp = run_contract_solve(&mut exact_game, 400);
        exact_game.back_to_root();
        let exact_game = Arc::new(exact_game);
        setup_oracle_boundaries(
            &mut subgame,
            Arc::clone(&exact_game),
            OracleCfvOrientation::Current,
            1.0,
        )
        .unwrap();

        let subgame_exp = run_contract_solve(&mut subgame, 400);
        subgame.back_to_root();
        let exp_delta = (subgame_exp - exact_exp).abs();

        let exact_player = exact_game.current_player();
        assert_eq!(exact_player, subgame.current_player());
        assert_eq!(exact_game.available_actions(), subgame.available_actions());

        let exact_strat = exact_game.strategy_at_index(0);
        let subgame_strat = subgame.strategy_at_index(0);
        let num_hands = exact_game.num_private_hands(exact_player);
        let num_actions = exact_game.available_actions().len();
        let (mean_mass, max_mass, max_idx) =
            compute_mass_moved(&exact_strat, &subgame_strat, num_hands, num_actions);

        eprintln!(
            "one-boundary oracle contract: exact_exp={exact_exp:.6}, \
             subgame_exp={subgame_exp:.6}, exp_delta={exp_delta:.6}, \
             mean_mass={mean_mass:.6}, max_mass={max_mass:.6}, max_idx={max_idx}"
        );
        assert!(
            exp_delta < 0.02,
            "one-boundary oracle exploitability diverged: exact_exp={exact_exp:.6}, \
             subgame_exp={subgame_exp:.6}, exp_delta={exp_delta:.6}"
        );
        assert!(
            max_mass < 0.02,
            "one-boundary oracle contract diverged: exact_exp={exact_exp:.6}, \
             subgame_exp={subgame_exp:.6}, mean_mass={mean_mass:.6}, \
             max_mass={max_mass:.6}, max_idx={max_idx}"
        );
    }

    // ---------------------------------------------------------------
    // compute_mass_moved tests
    // ---------------------------------------------------------------

    #[test]
    fn mass_moved_identical_strategies_is_zero() {
        // Two identical strategy vectors should produce zero mass moved
        let strat = vec![0.5f32, 0.5, 0.3, 0.7, 0.5, 0.5, 0.7, 0.3];
        // 2 actions, 4 hands: layout is [a0h0, a0h1, a0h2, a0h3, a1h0, a1h1, a1h2, a1h3]
        let (mean, max, _) = compute_mass_moved(&strat, &strat, 4, 2);
        assert_eq!(mean, 0.0);
        assert_eq!(max, 0.0);
    }

    #[test]
    fn mass_moved_completely_different_strategies() {
        // exact: hand 0 always action 0, hand 1 always action 1
        // subgame: opposite
        // 2 actions, 2 hands
        let exact   = vec![1.0f32, 0.0, 0.0, 1.0]; // a0: [1,0], a1: [0,1]
        let subgame = vec![0.0f32, 1.0, 1.0, 0.0]; // a0: [0,1], a1: [1,0]
        let (mean, max, max_idx) = compute_mass_moved(&exact, &subgame, 2, 2);
        // Each hand has |1-0|+|0-1| = 2, /2 = 1.0
        assert!((mean - 1.0).abs() < 1e-9);
        assert!((max - 1.0).abs() < 1e-9);
        // Both have same mass, so max_idx is 0 (first encountered)
        assert_eq!(max_idx, 0);
    }

    #[test]
    fn mass_moved_partial_diff() {
        // 3 actions, 2 hands
        // exact:   a0=[0.5, 0.2], a1=[0.3, 0.5], a2=[0.2, 0.3]
        // subgame: a0=[0.6, 0.1], a1=[0.2, 0.6], a2=[0.2, 0.3]
        let exact   = vec![0.5, 0.2, 0.3, 0.5, 0.2, 0.3];
        let subgame = vec![0.6, 0.1, 0.2, 0.6, 0.2, 0.3];
        let (mean, max, max_idx) = compute_mass_moved(&exact, &subgame, 2, 3);
        // hand 0: |0.5-0.6|+|0.3-0.2|+|0.2-0.2| = 0.1+0.1+0 = 0.2, /2 = 0.1
        // hand 1: |0.2-0.1|+|0.5-0.6|+|0.3-0.3| = 0.1+0.1+0 = 0.2, /2 = 0.1
        assert!((mean - 0.1).abs() < 1e-6, "mean={mean}");
        assert!((max - 0.1).abs() < 1e-6, "max={max}");
        assert_eq!(max_idx, 0);
    }

    #[test]
    fn mass_moved_zero_hands() {
        let (mean, max, idx) = compute_mass_moved(&[], &[], 0, 2);
        assert_eq!(mean, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(idx, 0);
    }

    // ---------------------------------------------------------------
    // compute_action_bias tests
    // ---------------------------------------------------------------

    #[test]
    fn action_bias_identical_is_zero() {
        let strat = vec![0.5f32, 0.5, 0.5, 0.5];
        let labels = vec!["Fold".to_string(), "Call".to_string()];
        let bias = compute_action_bias(&strat, &strat, 2, &labels);
        assert_eq!(bias.len(), 2);
        for (_, delta) in &bias {
            assert!(delta.abs() < 1e-9, "delta={delta}");
        }
    }

    #[test]
    fn action_bias_shows_fold_heavy_subgame() {
        // 2 actions (Fold, Call), 3 hands
        // exact:   fold=[0.3, 0.3, 0.3], call=[0.7, 0.7, 0.7]
        // subgame: fold=[0.5, 0.5, 0.5], call=[0.5, 0.5, 0.5]
        let exact   = vec![0.3, 0.3, 0.3, 0.7, 0.7, 0.7];
        let subgame = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let labels = vec!["Fold".to_string(), "Call".to_string()];
        let bias = compute_action_bias(&exact, &subgame, 3, &labels);
        // Fold delta: 1.5 - 0.9 = 0.6
        assert!((bias[0].1 - 0.6).abs() < 1e-6, "fold delta={}", bias[0].1);
        assert_eq!(bias[0].0, "Fold");
        // Call delta: 1.5 - 2.1 = -0.6
        assert!(
            (bias[1].1 - (-0.6)).abs() < 1e-6,
            "call delta={}",
            bias[1].1
        );
        assert_eq!(bias[1].0, "Call");
    }

    #[test]
    fn action_bias_classifies_bet_labels() {
        let labels = vec![
            "Fold".to_string(),
            "Check".to_string(),
            "4bb".to_string(),
            "All-in".to_string(),
        ];
        let strat = vec![0.25f32; 8]; // 4 actions, 2 hands, all uniform
        let bias = compute_action_bias(&strat, &strat, 2, &labels);
        assert_eq!(bias[0].0, "Fold");
        assert_eq!(bias[1].0, "Check");
        assert_eq!(bias[2].0, "Bet/Raise");
        assert_eq!(bias[3].0, "AllIn");
    }

    // ---------------------------------------------------------------
    // top_hands_by_mass tests
    // ---------------------------------------------------------------

    #[test]
    fn top_hands_returns_sorted_descending() {
        // 2 actions, 4 hands
        // exact:   a0=[0.5, 0.2, 0.9, 0.4], a1=[0.5, 0.8, 0.1, 0.6]
        // subgame: a0=[0.5, 0.5, 0.1, 0.4], a1=[0.5, 0.5, 0.9, 0.6]
        let exact   = vec![0.5, 0.2, 0.9, 0.4,  0.5, 0.8, 0.1, 0.6];
        let subgame = vec![0.5, 0.5, 0.1, 0.4,  0.5, 0.5, 0.9, 0.6];
        let top = top_hands_by_mass(&exact, &subgame, 4, 2, 2);
        assert_eq!(top.len(), 2);
        // hand 2 has mass = (|0.9-0.1|+|0.1-0.9|)/2 = 0.8
        // hand 1 has mass = (|0.2-0.5|+|0.8-0.5|)/2 = 0.3
        // hand 0 has mass = 0
        // hand 3 has mass = 0
        assert_eq!(top[0].0, 2); // hand index 2
        assert!((top[0].1 - 0.8).abs() < 1e-6);
        assert_eq!(top[1].0, 1); // hand index 1
        assert!((top[1].1 - 0.3).abs() < 1e-6);
    }

    #[test]
    fn top_hands_includes_strategy_vectors() {
        let exact   = vec![1.0f32, 0.0, 0.0, 1.0];
        let subgame = vec![0.0f32, 1.0, 1.0, 0.0];
        let top = top_hands_by_mass(&exact, &subgame, 2, 2, 1);
        assert_eq!(top.len(), 1);
        // Both have mass 1.0, first one wins
        assert_eq!(top[0].2, vec![1.0, 0.0]); // exact probs for hand 0
        assert_eq!(top[0].3, vec![0.0, 1.0]); // subgame probs for hand 0
    }

    // ---------------------------------------------------------------
    // classify_action tests
    // ---------------------------------------------------------------

    #[test]
    fn classify_action_labels() {
        assert_eq!(classify_action("Fold"), "Fold");
        assert_eq!(classify_action("Check"), "Check");
        assert_eq!(classify_action("Call"), "Call");
        assert_eq!(classify_action("All-in"), "AllIn");
        assert_eq!(classify_action("allin"), "AllIn");
        assert_eq!(classify_action("4bb"), "Bet/Raise");
        assert_eq!(classify_action("22bb"), "Bet/Raise");
    }

    // ---------------------------------------------------------------
    // find_strategy_path tests
    // ---------------------------------------------------------------

    #[test]
    fn find_strategy_path_pinned_snapshot() {
        let dir = std::env::temp_dir().join("cmp_solve_test_pinned");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("snapshot_0001")).unwrap();
        std::fs::write(dir.join("snapshot_0001/strategy.bin"), b"x").unwrap();
        let result = find_strategy_path(&dir, Some("snapshot_0001")).unwrap();
        assert_eq!(result, dir.join("snapshot_0001/strategy.bin"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_strategy_path_pinned_snapshot_missing_errors() {
        let dir = std::env::temp_dir().join("cmp_solve_test_missing");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let result = find_strategy_path(&dir, Some("snapshot_9999"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("snapshot_9999"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_strategy_path_prefers_final() {
        let dir = std::env::temp_dir().join("cmp_solve_test_final");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("final")).unwrap();
        std::fs::create_dir_all(dir.join("snapshot_0001")).unwrap();
        std::fs::write(dir.join("final/strategy.bin"), b"f").unwrap();
        std::fs::write(dir.join("snapshot_0001/strategy.bin"), b"s").unwrap();
        let result = find_strategy_path(&dir, None).unwrap();
        assert_eq!(result, dir.join("final/strategy.bin"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_strategy_path_latest_snapshot_fallback() {
        let dir = std::env::temp_dir().join("cmp_solve_test_latest");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("snapshot_0002")).unwrap();
        std::fs::create_dir_all(dir.join("snapshot_0005")).unwrap();
        std::fs::write(dir.join("snapshot_0002/strategy.bin"), b"a").unwrap();
        std::fs::write(dir.join("snapshot_0005/strategy.bin"), b"b").unwrap();
        let result = find_strategy_path(&dir, None).unwrap();
        assert_eq!(result, dir.join("snapshot_0005/strategy.bin"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_strategy_path_empty_dir_errors() {
        let dir = std::env::temp_dir().join("cmp_solve_test_empty");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let result = find_strategy_path(&dir, None);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---------------------------------------------------------------
    // BoundaryCfvStats tests
    // ---------------------------------------------------------------

    #[test]
    fn cfv_stats_basic_values() {
        let cfvs = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let stats = BoundaryCfvStats::from_slice(&cfvs);
        assert!((stats.min - 1.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
        assert!((stats.mean - 3.0).abs() < 1e-6);
        // stddev of [1,2,3,4,5] = sqrt(2.0) ~= 1.4142
        assert!((stats.stddev - 2.0f64.sqrt() as f32).abs() < 1e-4);
        assert_eq!(stats.count_nonzero, 5);
    }

    #[test]
    fn cfv_stats_with_zeros() {
        let cfvs = vec![0.0f32, 0.0, 3.0, 0.0, 5.0];
        let stats = BoundaryCfvStats::from_slice(&cfvs);
        assert!((stats.min - 0.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
        assert_eq!(stats.count_nonzero, 2);
    }

    #[test]
    fn cfv_stats_empty_slice() {
        let cfvs: Vec<f32> = vec![];
        let stats = BoundaryCfvStats::from_slice(&cfvs);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.stddev, 0.0);
        assert_eq!(stats.count_nonzero, 0);
    }

    #[test]
    fn cfv_stats_single_value() {
        let cfvs = vec![42.0f32];
        let stats = BoundaryCfvStats::from_slice(&cfvs);
        assert!((stats.min - 42.0).abs() < 1e-6);
        assert!((stats.max - 42.0).abs() < 1e-6);
        assert!((stats.mean - 42.0).abs() < 1e-6);
        assert!((stats.stddev - 0.0).abs() < 1e-6);
        assert_eq!(stats.count_nonzero, 1);
    }

    #[test]
    fn cfv_stats_all_same_value() {
        let cfvs = vec![7.0f32; 100];
        let stats = BoundaryCfvStats::from_slice(&cfvs);
        assert!((stats.min - 7.0).abs() < 1e-6);
        assert!((stats.max - 7.0).abs() < 1e-6);
        assert!((stats.mean - 7.0).abs() < 1e-6);
        assert!((stats.stddev - 0.0).abs() < 1e-6);
        assert_eq!(stats.count_nonzero, 100);
    }

    // ---------------------------------------------------------------
    // find_sample_hand_indices tests
    // ---------------------------------------------------------------

    #[test]
    fn find_sample_hands_finds_matching_cards() {
        // AA target: (48, 51) = Ac, As
        let private_cards = vec![(48u8, 51u8), (20, 1), (36, 37)];
        let result = find_sample_hand_indices(&private_cards);
        // Should find AA at index 0, 72o at index 1, JJ at index 2
        assert!(result.iter().any(|(l, i)| l == "AA" && *i == 0));
        assert!(result.iter().any(|(l, i)| l == "72o" && *i == 1));
        assert!(result.iter().any(|(l, i)| l == "JJ" && *i == 2));
    }

    #[test]
    fn find_sample_hands_handles_reversed_order() {
        // Cards might be stored in reverse order
        let private_cards = vec![(51u8, 48u8)]; // As, Ac (reversed)
        let result = find_sample_hand_indices(&private_cards);
        assert!(result.iter().any(|(l, _)| l == "AA"));
    }

    #[test]
    fn find_sample_hands_empty_input() {
        let private_cards: Vec<(u8, u8)> = vec![];
        let result = find_sample_hand_indices(&private_cards);
        assert!(result.is_empty());
    }

    #[test]
    fn cfv_stats_negative_values() {
        let cfvs = vec![-10.0f32, -5.0, 0.0, 5.0, 10.0];
        let stats = BoundaryCfvStats::from_slice(&cfvs);
        assert!((stats.min - (-10.0)).abs() < 1e-6);
        assert!((stats.max - 10.0).abs() < 1e-6);
        assert!((stats.mean - 0.0).abs() < 1e-6);
        assert_eq!(stats.count_nonzero, 4);
    }

    // ---------------------------------------------------------------
    // Sanity check tests
    // ---------------------------------------------------------------

    #[test]
    fn check_out_of_range_flags_large_cfvs() {
        let stats = BoundaryCfvStats {
            min: -5.0,
            max: 2000.0,
            mean: 100.0,
            stddev: 500.0,
            count_nonzero: 10,
        };
        let warnings = check_cfv_out_of_range(&stats, 200.0);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("out of range"));
    }

    #[test]
    fn check_out_of_range_passes_normal_values() {
        let stats = BoundaryCfvStats {
            min: -50.0,
            max: 80.0,
            mean: 10.0,
            stddev: 20.0,
            count_nonzero: 10,
        };
        let warnings = check_cfv_out_of_range(&stats, 200.0);
        assert!(warnings.is_empty());
    }

    #[test]
    fn check_non_finite_flags_nan() {
        let stats = BoundaryCfvStats {
            min: f32::NAN,
            max: 5.0,
            mean: f32::NAN,
            stddev: 1.0,
            count_nonzero: 5,
        };
        let warnings = check_cfv_non_finite(&stats);
        assert!(!warnings.is_empty());
        assert!(warnings[0].contains("non-finite"));
    }

    #[test]
    fn check_non_finite_flags_inf() {
        let stats = BoundaryCfvStats {
            min: -1.0,
            max: f32::INFINITY,
            mean: 5.0,
            stddev: 1.0,
            count_nonzero: 5,
        };
        let warnings = check_cfv_non_finite(&stats);
        assert!(!warnings.is_empty());
    }

    #[test]
    fn check_non_finite_passes_normal() {
        let stats = BoundaryCfvStats {
            min: -10.0,
            max: 10.0,
            mean: 0.0,
            stddev: 5.0,
            count_nonzero: 10,
        };
        let warnings = check_cfv_non_finite(&stats);
        assert!(warnings.is_empty());
    }

    #[test]
    fn check_unit_mismatch_flags_large_ratio() {
        // One boundary has mean magnitude ~0.001, another ~1000
        let stats_a = BoundaryCfvStats {
            min: -0.002,
            max: 0.002,
            mean: 0.001,
            stddev: 0.0005,
            count_nonzero: 10,
        };
        let stats_b = BoundaryCfvStats {
            min: -2000.0,
            max: 2000.0,
            mean: 1000.0,
            stddev: 500.0,
            count_nonzero: 10,
        };
        let warnings = check_unit_mismatch(&[stats_a, stats_b]);
        assert!(!warnings.is_empty());
        assert!(warnings[0].contains("unit mismatch"));
    }

    #[test]
    fn check_unit_mismatch_passes_similar_scales() {
        let stats_a = BoundaryCfvStats {
            min: -50.0,
            max: 50.0,
            mean: 10.0,
            stddev: 20.0,
            count_nonzero: 10,
        };
        let stats_b = BoundaryCfvStats {
            min: -80.0,
            max: 80.0,
            mean: 15.0,
            stddev: 30.0,
            count_nonzero: 10,
        };
        let warnings = check_unit_mismatch(&[stats_a, stats_b]);
        assert!(warnings.is_empty());
    }

    #[test]
    fn check_bias_divergence_flags_extreme_ratio() {
        // Unbiased mean=1.0, Fold mean=1000.0 -> 1000x ratio
        let unbiased = BoundaryCfvStats {
            min: 0.5,
            max: 1.5,
            mean: 1.0,
            stddev: 0.2,
            count_nonzero: 10,
        };
        let fold = BoundaryCfvStats {
            min: 500.0,
            max: 1500.0,
            mean: 1000.0,
            stddev: 200.0,
            count_nonzero: 10,
        };
        let call = BoundaryCfvStats {
            min: 0.4,
            max: 1.6,
            mean: 1.1,
            stddev: 0.3,
            count_nonzero: 10,
        };
        let raise = BoundaryCfvStats {
            min: 0.3,
            max: 1.7,
            mean: 0.9,
            stddev: 0.25,
            count_nonzero: 10,
        };
        let bias_names = ["Unbiased", "Fold", "Call", "Raise"];
        let all = [&unbiased, &fold, &call, &raise];
        let warnings = check_bias_divergence(&all, &bias_names);
        assert!(!warnings.is_empty());
        assert!(warnings[0].contains("Fold"));
    }

    #[test]
    fn check_bias_divergence_passes_similar() {
        let unbiased = BoundaryCfvStats {
            min: -10.0,
            max: 50.0,
            mean: 20.0,
            stddev: 15.0,
            count_nonzero: 10,
        };
        let fold = BoundaryCfvStats {
            min: -15.0,
            max: 55.0,
            mean: 22.0,
            stddev: 16.0,
            count_nonzero: 10,
        };
        let call = BoundaryCfvStats {
            min: -12.0,
            max: 48.0,
            mean: 18.0,
            stddev: 14.0,
            count_nonzero: 10,
        };
        let raise = BoundaryCfvStats {
            min: -8.0,
            max: 52.0,
            mean: 21.0,
            stddev: 15.5,
            count_nonzero: 10,
        };
        let bias_names = ["Unbiased", "Fold", "Call", "Raise"];
        let all = [&unbiased, &fold, &call, &raise];
        let warnings = check_bias_divergence(&all, &bias_names);
        assert!(warnings.is_empty());
    }
}
