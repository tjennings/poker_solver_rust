use std::fmt::Write as FmtWrite;
use std::io::Write;
use std::path::PathBuf;

use poker_solver_core::preflop::{
    PostflopAction, PostflopNode, PostflopTerminalType, PostflopTree, PotType,
};

use crate::PostflopSolveConfig;

/// Dump the postflop tree as indented text or Graphviz DOT.
pub fn run(
    dot: bool,
    output: Option<PathBuf>,
    config: PathBuf,
    spr: f64,
    pot_type: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let yaml = std::fs::read_to_string(&config)?;
    let pf_solve: PostflopSolveConfig = serde_yaml::from_str(&yaml)?;

    let pot = parse_pot_type(&pot_type)?;
    let mut tree = PostflopTree::build_with_spr(&pf_solve.postflop_model, spr)?;
    tree.pot_type = pot;

    let mut writer: Box<dyn Write> = match output {
        Some(path) => Box::new(std::fs::File::create(path)?),
        None => Box::new(std::io::stdout().lock()),
    };

    if dot {
        dump_dot(&tree, &mut writer)?;
    } else {
        dump_text(&tree, &mut writer)?;
    }

    Ok(())
}

// ─── Text output ─────────────────────────────────────────────────────────────

fn dump_text(tree: &PostflopTree, w: &mut dyn Write) -> std::io::Result<()> {
    writeln!(
        w,
        "PostflopTree  pot={:?}  spr={:.1}  nodes={}",
        tree.pot_type,
        tree.spr,
        tree.nodes.len()
    )?;
    if !tree.nodes.is_empty() {
        write_node_text(tree, 0, "", true, w)?;
    }
    Ok(())
}

fn write_node_text(
    tree: &PostflopTree,
    idx: u32,
    prefix: &str,
    is_last: bool,
    w: &mut dyn Write,
) -> std::io::Result<()> {
    let connector = if prefix.is_empty() {
        ""
    } else if is_last {
        "└── "
    } else {
        "├── "
    };
    let child_prefix = if prefix.is_empty() {
        String::new()
    } else if is_last {
        format!("{prefix}    ")
    } else {
        format!("{prefix}│   ")
    };

    let node = &tree.nodes[idx as usize];
    let label = node_text_label(idx, node);
    writeln!(w, "{prefix}{connector}{label}")?;

    match node {
        PostflopNode::Decision {
            children,
            action_labels,
            ..
        } => {
            for (i, (&child_idx, action)) in
                children.iter().zip(action_labels.iter()).enumerate()
            {
                let last = i + 1 == children.len();
                let act = action_label(action);
                writeln!(w, "{child_prefix}{} {act} →", if last { "└──" } else { "├──" })?;
                let next_prefix = if last {
                    format!("{child_prefix}    ")
                } else {
                    format!("{child_prefix}│   ")
                };
                write_node_text(tree, child_idx, &next_prefix, true, w)?;
            }
        }
        PostflopNode::Chance { children, .. } => {
            for (i, &child_idx) in children.iter().enumerate() {
                let last = i + 1 == children.len();
                write_node_text(tree, child_idx, &child_prefix, last, w)?;
            }
        }
        PostflopNode::Terminal { .. } => {}
    }

    Ok(())
}

fn node_text_label(idx: u32, node: &PostflopNode) -> String {
    match node {
        PostflopNode::Decision { position, children, .. } => {
            format!(
                "[{}] Decision {} ({} actions)",
                idx,
                pos_name(*position),
                children.len()
            )
        }
        PostflopNode::Chance { street, children, .. } => {
            format!(
                "[{}] Chance {:?} ({} children)",
                idx,
                street,
                children.len()
            )
        }
        PostflopNode::Terminal {
            terminal_type,
            pot_fraction,
        } => {
            let tt = match terminal_type {
                PostflopTerminalType::Fold { folder } => {
                    format!("Fold({})", pos_name(*folder))
                }
                PostflopTerminalType::Showdown => "Showdown".to_string(),
            };
            format!("[{}] Terminal {} pot={:.2}", idx, tt, pot_fraction)
        }
    }
}

// ─── DOT output ──────────────────────────────────────────────────────────────

fn dump_dot(tree: &PostflopTree, w: &mut dyn Write) -> std::io::Result<()> {
    let mut buf = String::with_capacity(4096);
    writeln!(buf, "digraph PostflopTree {{").unwrap();
    writeln!(buf, "  rankdir=TB;").unwrap();
    writeln!(buf, "  node [fontname=\"Courier\" fontsize=10];").unwrap();

    for (i, node) in tree.nodes.iter().enumerate() {
        let (label, shape) = dot_node_attrs(i as u32, node);
        writeln!(buf, "  N{i} [label=\"{label}\" shape={shape}];").unwrap();
    }

    for (i, node) in tree.nodes.iter().enumerate() {
        match node {
            PostflopNode::Decision {
                children,
                action_labels,
                ..
            } => {
                for (&child, action) in children.iter().zip(action_labels.iter()) {
                    let act = action_label(action);
                    writeln!(buf, "  N{i} -> N{child} [label=\"{act}\"];").unwrap();
                }
            }
            PostflopNode::Chance { children, .. } => {
                for &child in children {
                    writeln!(buf, "  N{i} -> N{child};").unwrap();
                }
            }
            PostflopNode::Terminal { .. } => {}
        }
    }

    writeln!(buf, "}}").unwrap();
    w.write_all(buf.as_bytes())
}

fn dot_node_attrs(_idx: u32, node: &PostflopNode) -> (String, &'static str) {
    match node {
        PostflopNode::Decision { position, .. } => {
            (format!("Decision\\n{}", pos_name(*position)), "box")
        }
        PostflopNode::Chance { street, .. } => {
            (format!("Chance\\n{:?}", street), "diamond")
        }
        PostflopNode::Terminal {
            terminal_type,
            pot_fraction,
        } => {
            let tt = match terminal_type {
                PostflopTerminalType::Fold { folder } => {
                    format!("Fold({})", pos_name(*folder))
                }
                PostflopTerminalType::Showdown => "Showdown".to_string(),
            };
            (format!("{}\\npot={:.2}", tt, pot_fraction), "ellipse")
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn action_label(a: &PostflopAction) -> String {
    match a {
        PostflopAction::Check => "Check".to_string(),
        PostflopAction::Fold => "Fold".to_string(),
        PostflopAction::Call => "Call".to_string(),
        PostflopAction::Bet(f) => format!("Bet({:.2})", f),
        PostflopAction::Raise(f) => format!("Raise({:.2})", f),
    }
}

fn pos_name(p: u8) -> &'static str {
    match p {
        0 => "IP",
        1 => "OOP",
        _ => "??",
    }
}

fn parse_pot_type(s: &str) -> Result<PotType, Box<dyn std::error::Error>> {
    match s.to_ascii_lowercase().as_str() {
        "limped" => Ok(PotType::Limped),
        "raised" => Ok(PotType::Raised),
        "3bet" | "threebet" | "three_bet" => Ok(PotType::ThreeBet),
        "4bet" | "4bet+" | "fourbet" | "fourbetplus" | "four_bet_plus" => Ok(PotType::FourBetPlus),
        _ => Err(format!("unknown pot type: {s:?} (expected: limped, raised, 3bet, 4bet)").into()),
    }
}
