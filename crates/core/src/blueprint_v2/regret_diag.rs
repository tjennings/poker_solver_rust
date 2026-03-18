//! Diagnostic tool for dumping regret trees to a log file.
//!
//! Writes detailed regret/strategy information at each decision node
//! along a game tree path, to a timestamped log file for offline analysis.

use std::fmt::Write as FmtWrite;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use super::game_tree::{GameNode, GameTree, TreeAction};
use super::storage::RegretStorage;

/// Dump the preflop regret tree (first 2 levels of decision nodes).
pub fn dump_preflop_regrets(
    path: &Path,
    tree: &GameTree,
    storage: &dyn RegretStorage,
    bucket_labels: &[&str],
    buckets: &[u16],
) {
    let mut file = match open_log(path) {
        Some(f) => f,
        None => return,
    };

    let timestamp = epoch_secs();
    let _ = writeln!(file, "\n=== Preflop Regret Dump @ t={timestamp} ===");

    dump_node(
        &mut file, tree, storage, tree.root, 0,
        bucket_labels, buckets, &mut String::new(), 2,
    );
}

/// Dump the full game tree for a specific spot — preflop through postflop.
///
/// `preflop_storage` is the global BlueprintStorage for preflop nodes.
/// `postflop_storage` is the per-flop CompactStorage for this flop.
/// `max_depth` controls how deep to recurse (e.g., 6 = preflop + flop + turn actions).
pub fn dump_full_spot(
    path: &Path,
    tree: &GameTree,
    preflop_storage: &dyn RegretStorage,
    postflop_storage: &dyn RegretStorage,
    spot_name: &str,
    bucket_labels: &[&str],
    preflop_buckets: &[u16],
    postflop_buckets: &[u16],
    max_depth: usize,
) {
    let mut file = match open_log(path) {
        Some(f) => f,
        None => return,
    };

    let timestamp = epoch_secs();
    let _ = writeln!(file, "\n=== Full Spot Dump: {spot_name} @ t={timestamp} ===");

    dump_node_split(
        &mut file, tree,
        preflop_storage, postflop_storage,
        tree.root, 0,
        bucket_labels, preflop_buckets, postflop_buckets,
        &mut String::new(), max_depth,
    );
}

/// Dump a summary of regret stats across ALL per-flop storages.
/// Shows min/max/mean regret across all slots to detect saturation or wrapping.
pub fn dump_regret_stats(
    path: &Path,
    flop_storages: &[super::compact_storage::CompactStorage],
    label: &str,
) {
    let mut file = match open_log(path) {
        Some(f) => f,
        None => return,
    };

    let timestamp = epoch_secs();
    let _ = writeln!(file, "\n=== Per-Flop Regret Stats: {label} @ t={timestamp} ===");

    let mut global_min = i32::MAX;
    let mut global_max = i32::MIN;
    let mut global_sum = 0i64;
    let mut global_count = 0u64;
    let mut saturated_pos = 0u64;
    let mut saturated_neg = 0u64;
    let mut zero_count = 0u64;

    for (i, cs) in flop_storages.iter().enumerate() {
        let mut flop_min = i32::MAX;
        let mut flop_max = i32::MIN;

        for atom in &cs.regrets {
            let v = atom.load(std::sync::atomic::Ordering::Relaxed);
            flop_min = flop_min.min(v);
            flop_max = flop_max.max(v);
            global_sum += i64::from(v);
            global_count += 1;
            if v == i32::MAX { saturated_pos += 1; }
            if v == i32::MIN { saturated_neg += 1; }
            if v == 0 { zero_count += 1; }
        }

        global_min = global_min.min(flop_min);
        global_max = global_max.max(flop_max);

        // Print a few sample flops
        if i < 3 || i == flop_storages.len() / 2 || i == flop_storages.len() - 1 {
            let _ = writeln!(file, "  flop_{i:04}: min={flop_min:>7} max={flop_max:>7}");
        }
    }

    let mean = if global_count > 0 { global_sum as f64 / global_count as f64 } else { 0.0 };
    let _ = writeln!(file, "  ---");
    let _ = writeln!(file, "  Global: min={global_min} max={global_max} mean={mean:.1}");
    let _ = writeln!(file, "  Slots: {global_count} total, {zero_count} zero ({:.1}%)",
        zero_count as f64 / global_count as f64 * 100.0);
    let _ = writeln!(file, "  Saturated: +MAX={saturated_pos} -MIN={saturated_neg}");
}

// ── Internal helpers ─────────────────────────────────────────────────

fn open_log(path: &Path) -> Option<File> {
    OpenOptions::new().create(true).append(true).open(path).ok()
}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Dump a single-storage node tree (preflop only).
fn dump_node(
    file: &mut File,
    tree: &GameTree,
    storage: &dyn RegretStorage,
    node_idx: u32,
    depth: usize,
    bucket_labels: &[&str],
    buckets: &[u16],
    path_str: &mut String,
    max_depth: usize,
) {
    if depth > max_depth { return; }

    match &tree.nodes[node_idx as usize] {
        GameNode::Decision { player, street, actions, children } => {
            let indent = "  ".repeat(depth);
            let action_names: Vec<String> = actions.iter().map(format_action).collect();
            let _ = writeln!(file,
                "{indent}[node={node_idx} P{player} {street:?}] {path_str}");
            let _ = writeln!(file, "{indent}  Actions: {}", action_names.join(" | "));

            print_buckets(file, storage, node_idx, &indent, bucket_labels, buckets);

            for (a, &child_idx) in children.iter().enumerate() {
                let old_len = path_str.len();
                if !path_str.is_empty() { path_str.push_str(" → "); }
                path_str.push_str(&action_names[a]);
                dump_node(file, tree, storage, child_idx, depth + 1,
                    bucket_labels, buckets, path_str, max_depth);
                path_str.truncate(old_len);
            }
        }
        GameNode::Chance { next_street, child } => {
            let indent = "  ".repeat(depth);
            let _ = writeln!(file, "{indent}[Chance → {next_street:?}]");
        }
        GameNode::Terminal { kind, pot, invested } => {
            let indent = "  ".repeat(depth);
            let _ = writeln!(file, "{indent}[Terminal {kind:?} pot={pot:.1}]");
        }
    }
}

/// Dump a split-storage node tree (preflop + postflop).
fn dump_node_split(
    file: &mut File,
    tree: &GameTree,
    preflop_storage: &dyn RegretStorage,
    postflop_storage: &dyn RegretStorage,
    node_idx: u32,
    depth: usize,
    bucket_labels: &[&str],
    preflop_buckets: &[u16],
    postflop_buckets: &[u16],
    path_str: &mut String,
    max_depth: usize,
) {
    if depth > max_depth { return; }

    match &tree.nodes[node_idx as usize] {
        GameNode::Decision { player, street, actions, children } => {
            let indent = "  ".repeat(depth);
            let action_names: Vec<String> = actions.iter().map(format_action).collect();
            let _ = writeln!(file,
                "{indent}[node={node_idx} P{player} {street:?}] {path_str}");
            let _ = writeln!(file, "{indent}  Actions: {}", action_names.join(" | "));

            let (storage, buckets): (&dyn RegretStorage, &[u16]) =
                if *street == super::Street::Preflop {
                    (preflop_storage, preflop_buckets)
                } else {
                    (postflop_storage, postflop_buckets)
                };

            print_buckets(file, storage, node_idx, &indent, bucket_labels, buckets);

            for (a, &child_idx) in children.iter().enumerate() {
                let old_len = path_str.len();
                if !path_str.is_empty() { path_str.push_str(" → "); }
                path_str.push_str(&action_names[a]);
                dump_node_split(file, tree, preflop_storage, postflop_storage,
                    child_idx, depth + 1, bucket_labels,
                    preflop_buckets, postflop_buckets, path_str, max_depth);
                path_str.truncate(old_len);
            }
        }
        GameNode::Chance { next_street, .. } => {
            let indent = "  ".repeat(depth);
            let _ = writeln!(file, "{indent}[Chance → {next_street:?}]");
        }
        GameNode::Terminal { kind, pot, .. } => {
            let indent = "  ".repeat(depth);
            let _ = writeln!(file, "{indent}[Terminal {kind:?} pot={pot:.1}]");
        }
    }
}

fn print_buckets(
    file: &mut File,
    storage: &dyn RegretStorage,
    node_idx: u32,
    indent: &str,
    bucket_labels: &[&str],
    buckets: &[u16],
) {
    let num_actions = storage.num_actions(node_idx);
    if num_actions == 0 { return; }

    for (label, &bucket) in bucket_labels.iter().zip(buckets.iter()) {
        let mut strat_buf = vec![0.0_f64; num_actions];
        storage.current_strategy_into(node_idx, bucket, &mut strat_buf);

        let mut regret_str = String::new();
        let mut strat_str = String::new();
        for a in 0..num_actions {
            let r = storage.get_regret(node_idx, bucket, a);
            let _ = write!(regret_str, "{r:>8}");
            let _ = write!(strat_str, "{:>7.1}%", strat_buf[a] * 100.0);
        }

        let _ = writeln!(file, "{indent}  {label:>5} b={bucket:>3} R[{regret_str}] S[{strat_str}]");
    }
}

fn format_action(action: &TreeAction) -> String {
    match action {
        TreeAction::Fold => "Fold".into(),
        TreeAction::Check => "Check".into(),
        TreeAction::Call => "Call".into(),
        TreeAction::Bet(x) => format!("Bet{x:.1}"),
        TreeAction::Raise(x) => format!("R{x:.1}"),
        TreeAction::AllIn => "AllIn".into(),
    }
}
