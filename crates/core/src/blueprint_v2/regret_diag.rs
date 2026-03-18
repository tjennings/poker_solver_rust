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
use super::Street;

/// Dump the regret tree from the root, following all preflop actions
/// and optionally descending into postflop for a specific action path.
///
/// Writes to the given log file path (appends).
pub fn dump_preflop_regrets(
    path: &Path,
    tree: &GameTree,
    storage: &dyn RegretStorage,
    bucket_labels: &[&str],  // e.g., ["AA", "AKs", "72o"]
    buckets: &[u16],         // corresponding bucket indices
) {
    let mut file = match OpenOptions::new().create(true).append(true).open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Warning: cannot open regret log {}: {e}", path.display());
            return;
        }
    };

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let _ = writeln!(file, "\n=== Regret Dump @ t={timestamp} ===");

    dump_node_recursive(
        &mut file, tree, storage, tree.root, 0, bucket_labels, buckets,
        &mut String::new(), 2, // max_depth: preflop + first postflop level
    );
}

/// Recursively dump decision nodes up to max_depth.
fn dump_node_recursive(
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
    if depth > max_depth {
        return;
    }

    match &tree.nodes[node_idx as usize] {
        GameNode::Decision {
            player,
            street,
            actions,
            children,
        } => {
            let indent = "  ".repeat(depth);
            let _ = writeln!(
                file,
                "{indent}Decision node={node_idx} player={player} street={street:?} actions={} path=[{path_str}]",
                actions.len()
            );

            // Print action labels
            let action_names: Vec<String> = actions.iter().map(format_action).collect();
            let _ = writeln!(file, "{indent}  Actions: {}", action_names.join(", "));

            // For each tracked bucket/hand, print regrets and strategy
            for (label, &bucket) in bucket_labels.iter().zip(buckets.iter()) {
                let num_actions = storage.num_actions(node_idx);
                let mut strat_buf = vec![0.0_f64; num_actions];
                storage.current_strategy_into(node_idx, bucket, &mut strat_buf);

                let mut regret_str = String::new();
                let mut strat_str = String::new();
                for a in 0..num_actions {
                    let r = storage.get_regret(node_idx, bucket, a);
                    let s = strat_buf[a];
                    let _ = write!(regret_str, "{:>8}", r);
                    let _ = write!(strat_str, "{:>8.1}%", s * 100.0);
                }

                let _ = writeln!(file, "{indent}  {label:>5} bkt={bucket:>3} regret=[{regret_str}] strat=[{strat_str}]");
            }

            // Recurse into children
            for (a, &child_idx) in children.iter().enumerate() {
                let old_len = path_str.len();
                if !path_str.is_empty() {
                    path_str.push_str(", ");
                }
                path_str.push_str(&action_names[a]);
                dump_node_recursive(
                    file, tree, storage, child_idx, depth + 1,
                    bucket_labels, buckets, path_str, max_depth,
                );
                path_str.truncate(old_len);
            }
        }

        GameNode::Chance { child, next_street } => {
            let indent = "  ".repeat(depth);
            let _ = writeln!(file, "{indent}Chance → {next_street:?}");
            // Don't recurse past chance nodes for now
        }

        GameNode::Terminal { kind, pot, invested } => {
            let indent = "  ".repeat(depth);
            let _ = writeln!(file, "{indent}Terminal {kind:?} pot={pot:.1} invested=[{:.1}, {:.1}]",
                invested[0], invested[1]);
        }
    }
}

fn format_action(action: &TreeAction) -> String {
    match action {
        TreeAction::Fold => "Fold".to_string(),
        TreeAction::Check => "Check".to_string(),
        TreeAction::Call => "Call".to_string(),
        TreeAction::Bet(x) => format!("Bet {x:.1}"),
        TreeAction::Raise(x) => format!("Raise {x:.1}"),
        TreeAction::AllIn => "AllIn".to_string(),
    }
}
