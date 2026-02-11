//! Game tree analyzer CLI subcommand.
//!
//! Two modes:
//! - **Deal tree**: Walk a concrete deal from a trained bundle, showing strategy
//!   probabilities at each node with path-weighted EV.
//! - **Key describe**: Translate an info set key between hex and human-readable formats.

use std::path::Path;

use poker_solver_core::Game;
use poker_solver_core::blueprint::{AbstractionModeConfig, BlueprintStrategy, StrategyBundle};
use poker_solver_core::game::{AbstractionMode, HunlPostflop, PostflopConfig};
use poker_solver_core::info_key::{canonical_hand_index_from_str, describe_key};
use poker_solver_core::tree::{self, EvSummary, TreeConfig, TreeNode};

/// Run the tree analyzer.
///
/// # Errors
///
/// Returns an error if the bundle cannot be loaded or the key is invalid.
pub fn run_tree(
    bundle_path: &Path,
    depth: usize,
    min_prob: f32,
    seed: u64,
    key_input: Option<&str>,
    hand_filter: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let bundle = StrategyBundle::load(bundle_path)?;
    let bet_sizes = &bundle.config.game.bet_sizes;
    let mode_str = abstraction_mode_str(bundle.config.abstraction_mode);

    if let Some(key_str) = key_input {
        return run_key_describe(key_str, bet_sizes, mode_str, &bundle.blueprint);
    }

    run_deal_tree(&bundle, depth, min_prob, seed, hand_filter)
}

fn abstraction_mode_str(mode: AbstractionModeConfig) -> &'static str {
    match mode {
        AbstractionModeConfig::Ehs2 => "ehs2",
        AbstractionModeConfig::HandClass => "hand_class",
        AbstractionModeConfig::HandClassV2 => "hand_class_v2",
    }
}

fn run_key_describe(
    input: &str,
    bet_sizes: &[f32],
    mode: &str,
    blueprint: &BlueprintStrategy,
) -> Result<(), Box<dyn std::error::Error>> {
    let desc = describe_key(input, bet_sizes, mode, Some(blueprint))?;

    println!("Key:     0x{:016X}", desc.raw);
    println!("Compose: {}", desc.compose_string());
    println!();
    println!("  Street:  {}", capitalize(desc.street_label));
    println!("  Hand:    {} (bits: {:#010x})", desc.hand_label, desc.hand_bits);
    println!("  SPR:     {}", desc.spr_bucket);

    if desc.action_labels.is_empty() {
        println!("  Actions: (none)");
    } else {
        println!("  Actions: [{}]", desc.action_labels.join(", "));
    }
    println!();

    match &desc.strategy {
        Some(probs) => {
            let labels = build_strategy_labels(probs.len(), bet_sizes, &desc.action_codes);
            let parts: Vec<String> = labels
                .iter()
                .zip(probs.iter())
                .map(|(label, &p)| format!("{label} {p:.2}"))
                .collect();
            println!("  Strategy: {}", parts.join(" | "));
        }
        None => println!("  Strategy: (not found in blueprint)"),
    }

    Ok(())
}

fn run_deal_tree(
    bundle: &StrategyBundle,
    depth: usize,
    min_prob: f32,
    seed: u64,
    hand_filter: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = &bundle.config;
    let abstraction_mode = build_abstraction_mode(config);
    let deal_count = 1000;

    let mut game = HunlPostflop::new(config.game.clone(), abstraction_mode, deal_count);
    game.set_seed(seed);
    let states = game.initial_states();

    // Filter by hand if requested
    let state = if let Some(hand_str) = hand_filter {
        let target_idx = canonical_hand_index_from_str(hand_str)
            .ok_or_else(|| format!("invalid hand: {hand_str}"))?;
        states
            .iter()
            .find(|s| {
                poker_solver_core::info_key::canonical_hand_index(s.p1_holding)
                    == target_idx
            })
            .ok_or_else(|| format!("no deal found for hand {hand_str} in {deal_count} deals"))?
    } else {
        states
            .first()
            .ok_or("no deals generated")?
    };

    // Print deal header
    print_deal_header(state, &config.game);

    let tree_config = TreeConfig {
        max_depth: depth,
        min_prob,
        bet_sizes: config.game.bet_sizes.clone(),
    };

    let root = tree::build_tree(
        &game,
        state,
        &bundle.blueprint,
        &tree_config,
        "Root".into(),
        1.0,
        1.0,
        0,
    );

    render_tree(&root, &config.game.bet_sizes);
    println!();

    let summary = tree::compute_deal_ev(&root);
    render_ev_summary(&summary, min_prob);

    Ok(())
}

fn build_abstraction_mode(config: &poker_solver_core::blueprint::BundleConfig) -> Option<AbstractionMode> {
    match config.abstraction_mode {
        AbstractionModeConfig::HandClass => Some(AbstractionMode::HandClass),
        AbstractionModeConfig::HandClassV2 => Some(AbstractionMode::HandClassV2 {
            strength_bits: config.strength_bits,
            equity_bits: config.equity_bits,
        }),
        AbstractionModeConfig::Ehs2 => {
            // EHS2 mode requires card abstraction — skip for now (just use raw keys)
            None
        }
    }
}

fn print_deal_header(
    state: &poker_solver_core::game::PostflopState,
    config: &PostflopConfig,
) {
    let p1 = format!("{}{}", state.p1_holding[0], state.p1_holding[1]);
    let p2 = format!("{}{}", state.p2_holding[0], state.p2_holding[1]);

    let board_str = if let Some(board) = &state.full_board {
        let flop = format!("{} {} {}", board[0], board[1], board[2]);
        let turn = format!("{}", board[3]);
        let river = format!("{}", board[4]);
        format!("[{flop} | {turn} | {river}]")
    } else {
        "(no board)".into()
    };

    let pot_bb = f64::from(state.pot) / 2.0;
    let s1_bb = f64::from(state.stacks[0]) / 2.0;
    let s2_bb = f64::from(state.stacks[1]) / 2.0;

    println!("Deal: P1=[{p1}] P2=[{p2}]  Board={board_str}");
    println!(
        "      Stack depth: {} BB  Pot: {pot_bb:.1} BB  Stacks: [{s1_bb:.1}, {s2_bb:.1}] BB",
        config.stack_depth
    );
    println!();
}

// ---------------------------------------------------------------------------
// Tree rendering
// ---------------------------------------------------------------------------

const STREET_NAMES: [&str; 4] = ["Preflop", "Flop", "Turn", "River"];

/// Render the tree as ASCII with Unicode box-drawing characters.
pub fn render_tree(root: &TreeNode, _bet_sizes: &[f32]) {
    render_node(root, &[], true, 0);
}

fn render_node(
    node: &TreeNode,
    prefix: &[bool], // true = last child at this level
    is_root: bool,
    parent_street: u8,
) {
    let indent = build_indent(prefix);

    if is_root {
        let street = STREET_NAMES[node.street as usize];
        let player = player_label(node.player);
        let strategy_flag = if node.has_strategy { "" } else { " (no data)" };
        println!("{street} ({player}) \u{2014} {}{strategy_flag}", node.hand_desc);
    } else {
        let connector = if *prefix.last().unwrap_or(&true) {
            "\u{2514}\u{2500}\u{2500}"
        } else {
            "\u{251C}\u{2500}\u{2500}"
        };

        // Terminal annotation
        if let Some(ref terminal) = node.terminal {
            println!(
                "{indent}{connector} {} ({:.0}%) \u{2500}\u{2500} [{}]",
                node.action_label,
                node.probability * 100.0,
                terminal.description
            );
            return;
        }

        println!(
            "{indent}{connector} {} ({:.0}%)",
            node.action_label,
            node.probability * 100.0,
        );

        // Street transition header
        if node.street != parent_street && !node.children.is_empty() {
            let child_indent = build_child_indent(prefix);
            let street = STREET_NAMES[node.street as usize];
            let player = player_label(node.player);
            let strategy_flag = if node.has_strategy { "" } else { " (no data)" };
            println!("{child_indent}{street} ({player}) \u{2014} {}{strategy_flag}", node.hand_desc);
        }
    }

    if node.children.is_empty() && node.terminal.is_none() && !is_root {
        let child_indent = build_child_indent(prefix);
        println!("{child_indent}...");
        return;
    }

    for (i, child) in node.children.iter().enumerate() {
        let is_last = i == node.children.len() - 1;
        let mut child_prefix = prefix.to_vec();
        if !is_root {
            child_prefix.push(*prefix.last().unwrap_or(&true));
        }
        // Replace last entry with whether this child is last
        if !child_prefix.is_empty() {
            *child_prefix.last_mut().unwrap() = is_last;
        } else {
            child_prefix.push(is_last);
        }
        render_node(child, &child_prefix, false, node.street);
    }
}

fn build_indent(prefix: &[bool]) -> String {
    if prefix.is_empty() {
        return String::new();
    }
    // All levels except the last determine continuation lines
    prefix[..prefix.len() - 1]
        .iter()
        .map(|&is_last| if is_last { "    " } else { "\u{2502}   " })
        .collect()
}

fn build_child_indent(prefix: &[bool]) -> String {
    prefix
        .iter()
        .map(|&is_last| if is_last { "    " } else { "\u{2502}   " })
        .collect()
}

fn player_label(player: Option<u8>) -> &'static str {
    match player {
        Some(0) => "P1 to act",
        Some(1) => "P2 to act",
        _ => "terminal",
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

// ---------------------------------------------------------------------------
// EV summary rendering
// ---------------------------------------------------------------------------

fn render_ev_summary(summary: &EvSummary, min_prob: f32) {
    println!("\u{2500}\u{2500}\u{2500} EV Summary (P1 perspective) \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
    println!();
    println!("  {:<50} {:>8} {:>10}", "Path", "Reach%", "EV (BB)");
    println!("  {}", "\u{2500}".repeat(70));

    for (path, reach, ev) in &summary.terminals {
        let truncated = if path.len() > 48 {
            format!("{}...", &path[..45])
        } else {
            path.clone()
        };
        println!(
            "  {:<50} {:>7.1}% {:>+10.2}",
            truncated,
            reach * 100.0,
            ev
        );
    }

    println!();
    println!(
        "  Coverage: {:.1}% of probability mass (min_prob={min_prob})",
        summary.total_reach * 100.0
    );
    println!("  Weighted EV: {:+.3} BB", summary.total_ev_bb);
}

// ---------------------------------------------------------------------------
// Strategy label helpers
// ---------------------------------------------------------------------------

/// Build action labels for a strategy vector based on the action history.
///
/// If we're at the first action on a street (no prior actions), labels are
/// Check + Bet sizes + All-in. Otherwise Fold + Call + Raise sizes + All-in.
fn build_strategy_labels(
    num_probs: usize,
    bet_sizes: &[f32],
    action_codes: &[u8],
) -> Vec<String> {
    // Determine if this is a first-to-act or facing-bet scenario
    let facing_bet = action_codes.last().is_some_and(|&c| matches!(c, 4..=15));

    if facing_bet {
        // Fold, Call, Raise sizes..., Raise All-In
        let mut labels = vec!["Fold".into(), "Call".into()];
        for &size in bet_sizes {
            if labels.len() >= num_probs {
                break;
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let pct = (size * 100.0) as u32;
            labels.push(format!("Raise {pct}%"));
        }
        if labels.len() < num_probs {
            labels.push("Raise All-In".into());
        }
        labels.truncate(num_probs);
        labels
    } else {
        // Check, Bet sizes..., Bet All-In
        let mut labels = vec!["Check".into()];
        for &size in bet_sizes {
            if labels.len() >= num_probs {
                break;
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let pct = (size * 100.0) as u32;
            labels.push(format!("Bet {pct}%"));
        }
        if labels.len() < num_probs {
            labels.push("Bet All-In".into());
        }
        labels.truncate(num_probs);
        labels
    }
}
