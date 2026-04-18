//! `bench-rollout` subcommand: reproducible benchmark for rollout throughput.
//!
//! Loads a blueprint bundle and sets up a canonical flop scenario, then drives
//! the rollout evaluator directly in a tight loop for a bounded wall time.
//! Reports total rollout hands, elapsed time, and throughput.
//!
//! Does **not** run DCFR — this benchmarks the rollout path only.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use poker_solver_core::blueprint_v2::bucket_file::BucketFile;
use poker_solver_core::blueprint_v2::bundle::{load_config, BlueprintV2Strategy};
use poker_solver_core::blueprint_v2::cbv::CbvTable;
use poker_solver_core::blueprint_v2::game_tree::{GameNode, GameTree};
use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
use poker_solver_core::blueprint_v2::Street;
use poker_solver_tauri::postflop::{
    build_rollout_evaluator, CbvContext, RolloutBenchContext,
};
use range_solver::range::Range;

/// Default OOP range for bench/validate scenarios.
pub(crate) const DEFAULT_OOP_RANGE: &str = "TT+,AQs+,AKo";
/// Default IP range for bench/validate scenarios.
pub(crate) const DEFAULT_IP_RANGE: &str = "JJ+,AKs,AQs";

/// Result of a benchmark run.
struct BenchResult {
    elapsed: Duration,
    hands: u64,
    iterations: u32,
}

impl BenchResult {
    fn hands_per_sec(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.hands as f64 / secs
    }
}

/// Drive the rollout evaluator directly for `duration`, counting hands via the
/// atomic counter.  Each iteration calls `rollout_chip_values_with_state` once
/// over all combos.
fn bench_loop(
    bench_ctx: &RolloutBenchContext,
    counter: &Arc<AtomicU64>,
    duration: Duration,
) -> BenchResult {
    counter.store(0, Ordering::Relaxed);
    let traverser: u8 = 0; // OOP
    let boundary_pot = 100.0;
    let boundary_invested = [50.0, 50.0];

    let start = Instant::now();
    let mut iterations: u32 = 0;
    while start.elapsed() < duration {
        let _values = bench_ctx.evaluator.rollout_chip_values_with_state(
            &bench_ctx.combos,
            &bench_ctx.board,
            &bench_ctx.oop_range,
            &bench_ctx.ip_range,
            traverser,
            boundary_pot,
            boundary_invested,
        );
        iterations += 1;
    }
    let elapsed = start.elapsed();
    let hands = counter.load(Ordering::Relaxed);
    BenchResult {
        elapsed,
        hands,
        iterations,
    }
}

/// Format the benchmark result line.
///
/// Output format:
/// `{elapsed:>6.1}s elapsed, {iterations} calls, {ms_per_call:.1} ms/call ({calls_per_sec:.2} calls/s), {hands} hands, {hands_per_sec:.0} hands/s`
fn format_result(elapsed_secs: f64, iterations: u32, hands: u64, hands_per_sec: f64) -> String {
    let (ms_per_call_str, calls_per_sec) = if iterations == 0 {
        ("--".to_string(), 0.0)
    } else {
        let ms = elapsed_secs / iterations as f64 * 1000.0;
        let cps = iterations as f64 / elapsed_secs;
        (format!("{ms:.1}"), cps)
    };
    format!(
        "{elapsed_secs:>6.1}s elapsed, {iterations} calls, {ms_per_call_str} ms/call ({calls_per_sec:.2} calls/s), {hands} hands, {hands_per_sec:.0} hands/s",
    )
}

/// Find the first flop decision node via BFS through the abstract tree.
///
/// Walks preflop actions until it hits a Chance node transitioning to Flop,
/// then returns the child (the first flop decision node index).
pub(crate) fn find_first_flop_node(tree: &GameTree) -> Option<u32> {
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(tree.root);
    while let Some(idx) = queue.pop_front() {
        match &tree.nodes[idx as usize] {
            GameNode::Chance {
                next_street: Street::Flop,
                child,
            } => return Some(*child),
            GameNode::Decision { children, .. } => {
                for &c in children {
                    queue.push_back(c);
                }
            }
            GameNode::Chance { child, .. } => {
                queue.push_back(*child);
            }
            GameNode::Terminal { .. } => {}
        }
    }
    None
}

/// Load a blueprint bundle and construct the `CbvContext` needed for rollout.
///
/// Reuses the loading pattern from `inspect_spot::load_blueprint`.
pub(crate) fn load_bundle(
    bundle_dir: &Path,
) -> Result<(Arc<CbvContext>, GameTree), String> {
    let config = load_config(bundle_dir)
        .map_err(|e| format!("Failed to load config.yaml: {e}"))?;

    // Find strategy.bin
    let strat_path = if bundle_dir.join("final/strategy.bin").exists() {
        bundle_dir.join("final/strategy.bin")
    } else {
        // Find latest snapshot
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
            Some(entry) => entry.path().join("strategy.bin"),
            None => {
                return Err(format!(
                    "No strategy.bin found in '{}'",
                    bundle_dir.display()
                ))
            }
        }
    };

    eprintln!("Loading strategy from {}...", strat_path.display());
    let strategy = BlueprintV2Strategy::load(&strat_path)
        .map_err(|e| format!("Failed to load strategy.bin: {e}"))?;

    let aa = &config.action_abstraction;
    let tree = GameTree::build(
        config.game.stack_depth,
        config.game.small_blind,
        config.game.big_blind,
        &aa.preflop,
        &aa.flop,
        &aa.turn,
        &aa.river,
    );

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
    eprintln!("[bench] loaded {loaded}/4 bucket files");

    let mut all_buckets = AllBuckets::new(bucket_counts, bucket_files);
    let per_flop = cluster_dir.join("flop_0000.buckets");
    if per_flop.exists() {
        all_buckets = all_buckets.with_per_flop_dir(cluster_dir);
    }
    // Enable equity fallback for boards not in bucket files.
    // Global bucket files may not cover every possible turn/river runout,
    // so rollouts would panic without this.
    all_buckets.equity_fallback = true;

    // Load CBV table if present (optional for rollout)
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
        strategy: Arc::new(strategy),
    });

    Ok((ctx, tree))
}

/// Build the rollout evaluator for the bench scenario.
fn build_bench_scenario(
    board_cards: &[poker_solver_core::poker::Card],
    ctx: &CbvContext,
    flop_node: u32,
    pot_f: f64,
    starting_stack: f64,
    counter: &Arc<AtomicU64>,
    enumerate_depth: Option<u8>,
    opponent_samples: Option<u32>,
) -> Result<RolloutBenchContext, String> {
    let oop_range: Range = DEFAULT_OOP_RANGE
        .parse()
        .map_err(|e| format!("Bad OOP range: {e}"))?;
    let ip_range: Range = DEFAULT_IP_RANGE
        .parse()
        .map_err(|e| format!("Bad IP range: {e}"))?;

    let mut bench_ctx = build_rollout_evaluator(
        board_cards, ctx, flop_node, pot_f, starting_stack,
        Some(Arc::clone(counter)), Some(&oop_range), Some(&ip_range),
    );
    if let Some(depth) = enumerate_depth {
        bench_ctx.evaluator.enumerate_decision_depth = depth;
    }
    if let Some(opp_samples) = opponent_samples {
        bench_ctx.evaluator.num_opponent_samples = opp_samples;
    }
    Ok(bench_ctx)
}

/// Run the bench-rollout command.
///
/// Loads the bundle, builds a rollout evaluator for a canonical flop scenario,
/// and drives it directly for the specified duration.  The hand counter tracks
/// rollout terminals reached; the result is printed as hands/sec.
///
/// Does **not** run DCFR — this benchmarks the rollout path only.
pub fn run(
    bundle_dir: &Path,
    duration_secs: u64,
    board_str: &str,
    pot: u32,
    stacks: u32,
    enumerate_depth: Option<u8>,
    opponent_samples: Option<u32>,
) -> Result<(), String> {
    let board_cards = parse_board_cards(board_str)?;
    if board_cards.len() != 3 {
        return Err(format!(
            "Bench requires a flop (3 cards) for boundary rollouts, got {} cards",
            board_cards.len()
        ));
    }

    let (ctx, tree) = load_bundle(bundle_dir)?;
    let flop_node = find_first_flop_node(&tree)
        .ok_or("No flop decision node found in abstract tree")?;
    eprintln!("[bench] abstract flop node index: {flop_node}");

    let pot_f = f64::from(pot);
    let starting_stack = f64::from(stacks) + pot_f / 2.0;
    eprintln!("[bench] board={board_str} pot={pot} stacks={stacks} duration={duration_secs}s");

    let counter = Arc::new(AtomicU64::new(0));
    let bench_ctx = build_bench_scenario(
        &board_cards, &ctx, flop_node, pot_f, starting_stack,
        &counter, enumerate_depth, opponent_samples,
    )?;

    eprintln!(
        "[bench] {} combos, enumerate_depth={}, opp_samples={}, starting rollout benchmark...",
        bench_ctx.combos.len(),
        bench_ctx.evaluator.enumerate_decision_depth,
        bench_ctx.evaluator.num_opponent_samples,
    );
    let result = bench_loop(&bench_ctx, &counter, Duration::from_secs(duration_secs));

    let output = format_result(
        result.elapsed.as_secs_f64(),
        result.iterations,
        result.hands,
        result.hands_per_sec(),
    );
    println!("{output}");

    Ok(())
}

/// Re-export `parse_board_cards` from the tauri crate for use by validate_rollout.
pub(crate) use poker_solver_tauri::postflop::parse_board_cards;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_result_computes_hands_per_sec() {
        let result = BenchResult {
            elapsed: std::time::Duration::from_secs(2),
            hands: 1000,
            iterations: 50,
        };
        let hps = result.hands_per_sec();
        assert!((hps - 500.0).abs() < 1.0, "expected ~500, got {hps}");
    }

    #[test]
    fn bench_result_hands_per_sec_zero_elapsed() {
        let result = BenchResult {
            elapsed: std::time::Duration::from_secs(0),
            hands: 100,
            iterations: 5,
        };
        assert_eq!(result.hands_per_sec(), 0.0);
    }

    #[test]
    fn format_result_shows_elapsed_hands_and_rate() {
        let output = format_result(12.5, 50, 125_000, 10_000.0);
        assert!(output.contains("12.5s elapsed"), "missing elapsed: {output}");
        assert!(output.contains("125000 hands"), "missing hands: {output}");
        assert!(output.contains("10000 hands/s"), "missing rate: {output}");
    }

    #[test]
    fn format_result_includes_iterations_and_per_call_latency() {
        // 10 calls in 5.0s => 2.0 ms/call is wrong, it's 500.0 ms/call; 2.00 calls/s
        let output = format_result(5.0, 10, 50_000, 10_000.0);
        assert!(output.contains("10 calls"), "missing calls: {output}");
        assert!(
            output.contains("500.0 ms/call"),
            "missing ms/call: {output}"
        );
        assert!(
            output.contains("2.00 calls/s"),
            "missing calls/s: {output}"
        );
    }

    #[test]
    fn format_result_zero_iterations_no_divide_by_zero() {
        let output = format_result(5.0, 0, 0, 0.0);
        assert!(
            output.contains("-- ms/call"),
            "missing placeholder ms/call: {output}"
        );
        assert!(
            output.contains("0.00 calls/s"),
            "missing zero calls/s: {output}"
        );
    }

    #[test]
    fn format_result_zero_hands() {
        let output = format_result(5.0, 10, 0, 0.0);
        assert!(output.contains("0 hands"), "missing zero hands: {output}");
        assert!(output.contains("0 hands/s"), "missing zero rate: {output}");
    }

    #[test]
    fn format_result_fractional_rate() {
        let output = format_result(1.0, 5, 50, 50.0);
        assert!(output.contains("50 hands/s"), "missing rate: {output}");
    }

    #[test]
    fn format_result_single_iteration() {
        let output = format_result(2.0, 1, 100, 50.0);
        assert!(
            output.contains("2000.0 ms/call"),
            "missing ms/call: {output}"
        );
        assert!(
            output.contains("0.50 calls/s"),
            "missing calls/s: {output}"
        );
    }

    #[test]
    fn parse_board_cards_valid_flop() {
        let cards = parse_board_cards("Ks7h2c").unwrap();
        assert_eq!(cards.len(), 3);
    }

    #[test]
    fn parse_board_cards_invalid() {
        assert!(parse_board_cards("XX").is_err());
    }

    #[test]
    fn parse_board_cards_turn() {
        let cards = parse_board_cards("Ks7h2c4d").unwrap();
        assert_eq!(cards.len(), 4);
    }

    #[test]
    fn parse_board_cards_too_few() {
        assert!(parse_board_cards("Ks").is_err());
    }

    #[test]
    fn parse_board_cards_odd_length() {
        assert!(parse_board_cards("Ks7").is_err());
    }

    #[test]
    fn parse_board_cards_too_many() {
        assert!(parse_board_cards("Ks7h2c4d3sAh").is_err());
    }

    #[test]
    fn find_first_flop_node_in_standard_tree() {
        let tree = GameTree::build(
            100.0, 1.0, 2.0,
            &[vec!["5bb".to_string()]],
            &[vec![0.5]],
            &[vec![0.5]],
            &[vec![0.5]],
        );
        let flop_node = find_first_flop_node(&tree);
        assert!(flop_node.is_some(), "should find a flop node");

        // The node should be a Decision on the Flop street.
        if let GameNode::Decision { street, .. } = &tree.nodes[flop_node.unwrap() as usize] {
            assert_eq!(*street, Street::Flop);
        } else {
            panic!("Expected Decision node at flop");
        }
    }
}
