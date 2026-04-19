//! `compare-solve` subcommand: run a spot through both Subgame and Exact
//! solvers and report per-combo strategy diffs and exploitability delta.
//!
//! Built to debug bean izod — subgame converging worse than its blueprint seed.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use poker_solver_core::blueprint_v2::bundle::{load_config, BlueprintV2Strategy};
use poker_solver_core::blueprint_v2::bucket_file::BucketFile;
use poker_solver_core::blueprint_v2::cbv::CbvTable;
use poker_solver_core::blueprint_v2::config::BlueprintV2Config;
use poker_solver_core::blueprint_v2::continuation::BiasType;
use poker_solver_core::blueprint_v2::game_tree::{
    GameNode as V2GameNode, GameTree as V2GameTree,
};
use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
use poker_solver_core::blueprint_v2::{LeafEvaluator, Street};
use poker_solver_core::hands::all_hands;

use poker_solver_tauri::{
    GameSession, SolveBoundaryEvaluator, build_solve_game,
    parse_rs_poker_card, range_solver_to_rs_card, seed_solver_with_blueprint,
};
use poker_solver_tauri::postflop::{CbvContext, RolloutLeafEvaluator};

use range_solver::card::card_to_string;
use range_solver::interface::Game;
use range_solver::{PostFlopGame, solve_step, finalize, compute_exploitability};

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

    let mean = if num_hands > 0 { total / num_hands as f64 } else { 0.0 };
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
        let exact_sum: f64 = exact[base..base + num_hands].iter().map(|&v| v as f64).sum();
        let subgame_sum: f64 = subgame[base..base + num_hands].iter().map(|&v| v as f64).sum();
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

    hands.into_iter().map(|(h, mass)| {
        let exact_probs: Vec<f32> = (0..num_actions).map(|a| exact[a * num_hands + h]).collect();
        let subgame_probs: Vec<f32> = (0..num_actions).map(|a| subgame[a * num_hands + h]).collect();
        (h, mass, exact_probs, subgame_probs)
    }).collect()
}

/// Find the strategy.bin path within a bundle, optionally pinned to a snapshot.
///
/// Resolution order:
/// 1. If `snapshot` is Some, use `bundle_dir/<snapshot>/strategy.bin`
/// 2. Else, try `bundle_dir/final/strategy.bin`
/// 3. Else, use the latest `snapshot_*` directory
fn find_strategy_path(bundle_dir: &Path, snapshot: Option<&str>) -> Result<std::path::PathBuf, String> {
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
) -> Result<(BlueprintV2Config, BlueprintV2Strategy, V2GameTree, Vec<u32>, Arc<CbvContext>), String> {
    let config = load_config(bundle_dir)
        .map_err(|e| format!("Failed to load config.yaml: {e}"))?;

    let strat_path = find_strategy_path(bundle_dir, snapshot)?;
    eprintln!("Loading strategy from {}...", strat_path.display());
    let strategy = BlueprintV2Strategy::load(&strat_path)
        .map_err(|e| format!("Failed to load strategy.bin: {e}"))?;

    let aa = &config.action_abstraction;
    let tree = V2GameTree::build_with_options(
        config.game.stack_depth,
        config.game.small_blind,
        config.game.big_blind,
        &aa.preflop, &aa.flop, &aa.turn, &aa.river,
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
    for (i, name) in ["preflop.buckets", "flop.buckets", "turn.buckets", "river.buckets"]
        .iter().enumerate()
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
        if alt.exists() { CbvTable::load(&alt).ok() } else { None }
    };
    let cbv_table = cbv_table.unwrap_or_else(|| CbvTable {
        values: vec![], node_offsets: vec![], buckets_per_node: vec![],
    });

    let ctx = Arc::new(CbvContext {
        cbv_table,
        abstract_tree: tree.clone(),
        all_buckets: Arc::new(all_buckets),
        strategy: Arc::new(strategy.clone()),
    });

    Ok((config, strategy, tree, decision_map, ctx))
}

/// Find the abstract_node_idx for boundary rollouts (first Chance node from current position).
fn find_boundary_node(tree: &V2GameTree, node_idx: u32) -> u32 {
    let mut idx = node_idx;
    for _ in 0..100 {
        match &tree.nodes[idx as usize] {
            V2GameNode::Chance { .. } => return idx,
            V2GameNode::Decision { children, .. } => {
                if let Some(&first_child) = children.first() {
                    idx = first_child;
                } else {
                    break;
                }
            }
            V2GameNode::Terminal { .. } => break,
        }
    }
    node_idx
}

/// Set up boundary evaluator and precompute multi-continuation CFVs for a subgame game.
#[allow(clippy::too_many_arguments)]
fn setup_subgame_boundaries(
    game: &mut PostFlopGame,
    ctx: &CbvContext,
    board_cards: &[poker_solver_core::poker::Card],
    abstract_node_idx: u32,
    eff_stack: i32,
    pot: i32,
    enumerate_depth: Option<u8>,
    opponent_samples: Option<u32>,
    verbose: bool,
) {
    let n_boundaries = game.num_boundary_nodes();
    if n_boundaries == 0 {
        return;
    }

    // Build combos in rollout ordering
    let mut combos: Vec<[poker_solver_core::poker::Card; 2]> = Vec::new();
    for hand in all_hands() {
        for (c0, c1) in hand.combos() {
            if board_cards.iter().any(|b| *b == c0 || *b == c1) {
                continue;
            }
            combos.push([c0, c1]);
        }
    }

    let build_map = |player: usize, combos: &[[poker_solver_core::poker::Card; 2]]| -> Vec<usize> {
        game.private_cards(player).iter().map(|&(c1, c2)| {
            let rs_c1 = range_solver_to_rs_card(c1);
            let rs_c2 = range_solver_to_rs_card(c2);
            combos.iter().position(|c|
                (c[0] == rs_c1 && c[1] == rs_c2) || (c[0] == rs_c2 && c[1] == rs_c1)
            ).unwrap_or(usize::MAX)
        }).collect()
    };
    let map0 = build_map(0, &combos);
    let map1 = build_map(1, &combos);

    let bias_factor = 10.0;
    let num_rollouts = 3;
    let opp_samples = opponent_samples.unwrap_or(8);
    let starting_stack = f64::from(eff_stack) + f64::from(pot) / 2.0;

    let mut rollout = RolloutLeafEvaluator::new(
        Arc::clone(&ctx.strategy),
        Arc::new(ctx.abstract_tree.clone()),
        Arc::clone(&ctx.all_buckets),
        abstract_node_idx,
        BiasType::Unbiased,
        bias_factor,
        num_rollouts,
        opp_samples,
        starting_stack,
        f64::from(pot),
    );
    if let Some(depth) = enumerate_depth {
        rollout.enumerate_decision_depth = depth;
    }

    let combos_for_precomp = combos.clone();
    let board_cards_for_precomp = board_cards.to_vec();
    let game_to_combo_for_precomp = [map0.clone(), map1.clone()];

    game.boundary_evaluator = Some(Arc::new(SolveBoundaryEvaluator {
        private_cards: [
            game.private_cards(0).to_vec(),
            game.private_cards(1).to_vec(),
        ],
        board_cards: board_cards.to_vec(),
        eff_stack: f64::from(eff_stack),
        rollout: Some(rollout),
        combos,
        game_to_combo: [map0, map1],
    }));

    // Precompute multi-valued boundary CFVs (K=4 continuations).
    let k = game.boundary_evaluator.as_ref().map_or(1, |e| e.num_continuations());
    if k > 1 {
        game.init_multi_continuation(k);
        let biases = [BiasType::Unbiased, BiasType::Fold, BiasType::Call, BiasType::Raise];
        let precomp_counter = Arc::new(AtomicU64::new(0));
        let start = Instant::now();

        for ordinal in 0..n_boundaries {
            let pot_at_boundary = game.boundary_pot(ordinal);

            for player in 0..2 {
                let num_hands = game.num_private_hands(player);

                for (ki, &bias) in biases.iter().enumerate().take(k) {
                    let mut eval = RolloutLeafEvaluator::new(
                        Arc::clone(&ctx.strategy),
                        Arc::new(ctx.abstract_tree.clone()),
                        Arc::clone(&ctx.all_buckets),
                        abstract_node_idx,
                        bias,
                        bias_factor,
                        num_rollouts,
                        opp_samples,
                        starting_stack,
                        f64::from(pot),
                    );
                    if let Some(depth) = enumerate_depth {
                        eval.enumerate_decision_depth = depth;
                    }
                    eval.call_counter = Arc::clone(&precomp_counter);

                    let opp = player ^ 1;
                    let mut opp_combo_reach = vec![0.0f64; combos_for_precomp.len()];
                    let opp_weights = game.initial_weights(opp);
                    for (gi, &ci) in game_to_combo_for_precomp[opp].iter().enumerate() {
                        if ci < opp_combo_reach.len() && gi < opp_weights.len() {
                            opp_combo_reach[ci] = opp_weights[gi] as f64;
                        }
                    }
                    let mut hero_combo_reach = vec![0.0f64; combos_for_precomp.len()];
                    let hero_weights = game.initial_weights(player);
                    for (gi, &ci) in game_to_combo_for_precomp[player].iter().enumerate() {
                        if ci < hero_combo_reach.len() && gi < hero_weights.len() {
                            hero_combo_reach[ci] = hero_weights[gi] as f64;
                        }
                    }

                    let requests = vec![(pot_at_boundary as f64, 0.0, player as u8)];
                    let results = eval.evaluate_boundaries(
                        &combos_for_precomp, &board_cards_for_precomp,
                        &hero_combo_reach, &opp_combo_reach, &requests,
                    );
                    let combo_cfvs = results.into_iter().next().unwrap_or_default();

                    let hero_map = &game_to_combo_for_precomp[player];
                    let mut cfvs = vec![0.0f32; num_hands];
                    for (game_idx, &combo_idx) in hero_map.iter().enumerate() {
                        if combo_idx < combo_cfvs.len() && game_idx < cfvs.len() {
                            cfvs[game_idx] = combo_cfvs[combo_idx] as f32;
                        }
                    }
                    game.set_boundary_cfvs_multi(ordinal, player, ki, cfvs);
                }
            }
        }

        let elapsed = start.elapsed();
        eprintln!(
            "[compare] precomputed {} boundaries x {} continuations x 2 players in {:.1}s",
            n_boundaries, k, elapsed.as_secs_f64()
        );
        if verbose {
            eprintln!("[compare] precompute rollout calls: {}", precomp_counter.load(Ordering::Relaxed));
        }
    }
}

/// Run a DCFR solve loop, returning (wall_time, final_exploitability).
fn run_dcfr_solve(
    game: &mut PostFlopGame,
    iters: u32,
    label: &str,
    verbose: bool,
) -> (f64, f64) {
    let start = Instant::now();
    for t in 0..iters {
        // DCFR discount params for boundary continuation regrets
        let nearest_pow4 = if t == 0 { 0 } else { 1u32 << ((t.leading_zeros() ^ 31) & !1) };
        let t_alpha = (t as i32 - 1).max(0) as f64;
        let t_gamma = (t - nearest_pow4) as f64;
        let pow_alpha = t_alpha * t_alpha.sqrt();
        let alpha = (pow_alpha / (pow_alpha + 1.0)) as f32;
        let beta = 0.5f32;
        let gamma = (t_gamma / (t_gamma + 1.0)).powi(3) as f32;
        game.set_boundary_discount(alpha, beta, gamma);

        solve_step(game, t);

        if verbose && (t + 1) % 20 == 0 {
            let wall = start.elapsed().as_secs_f64();
            eprintln!("[{label}] iter {}/{iters}  wall {wall:.1}s", t + 1);
        }
    }

    finalize(game);
    game.back_to_root();
    game.cache_normalized_weights();

    // Compute exploitability with cached boundary CFVs (disable lazy evaluator).
    let saved_evaluator = game.boundary_evaluator.take();
    let exp = compute_exploitability(game);
    game.boundary_evaluator = saved_evaluator;

    let wall = start.elapsed().as_secs_f64();
    (wall, exp as f64)
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

/// Run the compare-solve harness.
#[allow(clippy::too_many_lines)]
pub fn run(
    bundle_dir: &Path,
    snapshot: Option<&str>,
    spot: &str,
    iters: u32,
    enumerate_depth: Option<u8>,
    opponent_samples: Option<u32>,
    verbose: bool,
) -> Result<(), String> {
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
        return Err("Spot must be at a postflop decision node (deal board cards first)".to_string());
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

    // Find abstract_node_idx for boundary rollouts
    let abstract_node_idx = find_boundary_node(&tree, session.node_idx());

    // Parse board cards for rs_poker
    let board_cards: Vec<poker_solver_core::poker::Card> = board.iter()
        .map(|s| parse_rs_poker_card(s))
        .collect::<Result<Vec<_>, _>>()?;

    // Print header (display values in BB = chips / 2)
    let pot_bb = pot as f64 / 2.0;
    let eff_bb = eff_stack as f64 / 2.0;
    println!("=== compare-solve summary ===");
    println!("spot: {spot}");
    println!("board: {board_str}  pot: {pot_bb:.0}  eff_stack: {eff_bb:.0}");
    println!("position: {position}");
    println!(
        "iters: {iters}  subgame enumerate_depth: {}  opp_samples: {}",
        enumerate_depth.unwrap_or(2),
        opponent_samples.unwrap_or(8)
    );
    println!();

    // 4. Build exact game
    eprintln!("[compare] building exact game...");
    let mut exact_game = build_solve_game(&board, &oop_w, &ip_w, pot, eff_stack, bet_sizes, true)?;

    let (mem_exact, _) = exact_game.memory_usage();
    if verbose {
        eprintln!(
            "[exact] memory: {:.1} MB, OOP hands: {}, IP hands: {}",
            mem_exact as f64 / 1_048_576.0,
            exact_game.private_cards(0).len(),
            exact_game.private_cards(1).len(),
        );
    }

    // 5. Build subgame game
    eprintln!("[compare] building subgame game...");
    let mut subgame_game = build_solve_game(&board, &oop_w, &ip_w, pot, eff_stack, bet_sizes, false)?;

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
    let current_node = session.node_idx();
    seed_solver_with_blueprint(
        &exact_game, &ctx.strategy, &ctx.all_buckets, &ctx.abstract_tree,
        &board_cards, seed_street, current_node,
    );
    seed_solver_with_blueprint(
        &subgame_game, &ctx.strategy, &ctx.all_buckets, &ctx.abstract_tree,
        &board_cards, seed_street, current_node,
    );

    // 7. Precompute subgame boundaries
    eprintln!("[compare] precomputing subgame boundaries...");
    let precomp_start = Instant::now();
    setup_subgame_boundaries(
        &mut subgame_game, &ctx, &board_cards,
        abstract_node_idx, eff_stack, pot,
        enumerate_depth, opponent_samples, verbose,
    );
    let precomp_wall = precomp_start.elapsed().as_secs_f64();

    // 8. Solve exact
    eprintln!("[compare] solving exact ({iters} iters)...");
    let (exact_wall, exact_exp) = run_dcfr_solve(&mut exact_game, iters, "exact", verbose);

    // 9. Solve subgame
    eprintln!("[compare] solving subgame ({iters} iters)...");
    let (subgame_wall, subgame_exp) = run_dcfr_solve(&mut subgame_game, iters, "subgame", verbose);

    // 10. Extract strategies at root
    exact_game.back_to_root();
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
            exact_actions.len(), subgame_actions.len()
        ));
    }

    // 11. Run diff helpers
    let (mean_mass, max_mass, max_idx) = compute_mass_moved(&exact_strat, &subgame_strat, num_hands, num_actions);

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
    println!("wall: {exact_wall:.1}s  final_exp: {exact_exp_mbb:.2} mbb/hand");
    println!();

    println!("=== Subgame solve ===");
    println!(
        "precompute: {precomp_wall:.1}s  solve: {subgame_wall:.1}s  wall: {:.1}s  final_exp: {subgame_exp_mbb:.2} mbb/hand",
        precomp_wall + subgame_wall,
    );
    println!();

    println!("=== Diff (exact vs subgame, at root) ===");
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
        "exploitability delta: subgame {:+.2} mbb/hand ({})",
        exp_delta_mbb,
        if exp_delta_mbb > 0.0 { "worse" } else { "better" }
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

        let exact_str: Vec<String> = exact_probs.iter().zip(&action_short)
            .map(|(p, a)| format!("{a}:{p:.2}"))
            .collect();
        let subgame_str: Vec<String> = subgame_probs.iter().zip(&action_short)
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

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!((bias[1].1 - (-0.6)).abs() < 1e-6, "call delta={}", bias[1].1);
        assert_eq!(bias[1].0, "Call");
    }

    #[test]
    fn action_bias_classifies_bet_labels() {
        let labels = vec!["Fold".to_string(), "Check".to_string(), "4bb".to_string(), "All-in".to_string()];
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
}
