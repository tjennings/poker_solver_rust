//! Trace LHE strategy evolution across SD-CFR checkpoints.
//!
//! For each spot (player + hand + optional preceding action), loads every
//! checkpoint's model buffers, computes the average strategy, and prints
//! a table showing how fold/call/raise probabilities evolve over training.

use std::error::Error;
use std::path::{Path, PathBuf};

use poker_solver_core::game::{Action, Game, LimitHoldem, LimitHoldemConfig, Player};
use poker_solver_core::poker::{Card, Suit, Value};
use poker_solver_deep_cfr::eval::ExplicitPolicy;
use poker_solver_deep_cfr::lhe_encoder::LheEncoder;
use poker_solver_deep_cfr::model_buffer::ModelBuffer;
use poker_solver_deep_cfr::traverse::StateEncoder;

use rand::SeedableRng;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// Spot representation
// ---------------------------------------------------------------------------

/// A specific decision point to trace across checkpoints.
#[derive(Debug, Clone)]
pub struct Spot {
    /// Display label (e.g. "SB AA", "BB.R AKs")
    pub label: String,
    /// Which player is deciding
    pub player: Player,
    /// Concrete hole cards for the hand
    pub hand: [Card; 2],
    /// Action to apply before querying (None = root)
    pub preceding: Option<Action>,
}

/// Strategy + raw advantages for one spot at one checkpoint.
#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub iteration: u32,
    pub fold: f32,
    pub call: f32,
    pub raise: f32,
    pub net_count: usize,
    pub raw_advantages: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Rank ordering (shared with lhe_viz)
// ---------------------------------------------------------------------------

const RANK_ORDER: [Value; 13] = [
    Value::Ace,
    Value::King,
    Value::Queen,
    Value::Jack,
    Value::Ten,
    Value::Nine,
    Value::Eight,
    Value::Seven,
    Value::Six,
    Value::Five,
    Value::Four,
    Value::Three,
    Value::Two,
];

fn rank_from_char(c: char) -> Option<usize> {
    match c {
        'A' => Some(0),
        'K' => Some(1),
        'Q' => Some(2),
        'J' => Some(3),
        'T' => Some(4),
        '9' => Some(5),
        '8' => Some(6),
        '7' => Some(7),
        '6' => Some(8),
        '5' => Some(9),
        '4' => Some(10),
        '3' => Some(11),
        '2' => Some(12),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse a hand string like "AA", "AKs", "72o" into (row, col) indices.
pub fn parse_hand(s: &str) -> Result<(usize, usize), Box<dyn Error>> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 2 || chars.len() > 3 {
        return Err(format!("invalid hand notation: '{s}'").into());
    }

    let r1 = rank_from_char(chars[0]).ok_or_else(|| format!("invalid rank: '{}'", chars[0]))?;
    let r2 = rank_from_char(chars[1]).ok_or_else(|| format!("invalid rank: '{}'", chars[1]))?;

    if chars.len() == 2 {
        // Pair (e.g. "AA") — must be same rank
        if r1 != r2 {
            return Err(format!("two-char hand '{s}' must be a pair (e.g. 'AA')").into());
        }
        return Ok((r1, r2));
    }

    match chars[2] {
        's' => {
            // Suited: upper triangle (row < col)
            let (row, col) = if r1 < r2 { (r1, r2) } else { (r2, r1) };
            Ok((row, col))
        }
        'o' => {
            // Offsuit: lower triangle (row > col)
            let (row, col) = if r1 > r2 { (r1, r2) } else { (r2, r1) };
            Ok((row, col))
        }
        c => Err(format!("invalid hand suffix: '{c}' (expected 's' or 'o')").into()),
    }
}

/// Parse a spot string like "SB AA", "BB.R AKs", "BB.C JTs".
pub fn parse_spot(s: &str) -> Result<Spot, Box<dyn Error>> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() != 2 {
        return Err(format!("spot must be '<position> <hand>', got: '{s}'").into());
    }

    let pos = parts[0];
    let hand_str = parts[1];
    let (row, col) = parse_hand(hand_str)?;
    let hand = make_hole_cards(row, col);

    let (player, preceding) = match pos {
        "SB" => (Player::Player1, None),
        "BB.R" => (Player::Player2, Some(Action::Raise(0))),
        "BB.C" => (Player::Player2, Some(Action::Call)),
        _ => return Err(format!("unknown position: '{pos}' (expected SB, BB.R, BB.C)").into()),
    };

    Ok(Spot {
        label: s.to_string(),
        player,
        hand,
        preceding,
    })
}

/// Parse spots from a YAML file.
pub fn parse_spots_file(path: &Path) -> Result<Vec<Spot>, Box<dyn Error>> {
    let yaml = std::fs::read_to_string(path)?;
    let entries: Vec<SpotEntry> = serde_yaml::from_str(&yaml)?;
    entries.iter().map(|e| {
        let notation = format!("{} {}", e.player, e.hand);
        parse_spot(&notation)
    }).collect()
}

#[derive(serde::Deserialize)]
struct SpotEntry {
    player: String,
    hand: String,
}

// ---------------------------------------------------------------------------
// Hole card construction (same logic as lhe_viz)
// ---------------------------------------------------------------------------

fn make_hole_cards(row: usize, col: usize) -> [Card; 2] {
    let rank1 = RANK_ORDER[row];
    let rank2 = RANK_ORDER[col];
    let is_suited = row < col;

    let (suit1, suit2) = if is_suited {
        (Suit::Spade, Suit::Spade)
    } else {
        (Suit::Spade, Suit::Heart)
    };

    [Card::new(rank1, suit1), Card::new(rank2, suit2)]
}

// ---------------------------------------------------------------------------
// Checkpoint scanning
// ---------------------------------------------------------------------------

/// Find all `lhe_checkpoint_N` directories under `dir`, sorted by N.
pub fn scan_checkpoints(dir: &Path) -> Result<Vec<(u32, PathBuf)>, Box<dyn Error>> {
    let mut checkpoints = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if let Some(rest) = name_str.strip_prefix("lhe_checkpoint_") {
            if let Ok(n) = rest.parse::<u32>() {
                checkpoints.push((n, entry.path()));
            }
        }
    }
    checkpoints.sort_by_key(|(n, _)| *n);
    Ok(checkpoints)
}

// ---------------------------------------------------------------------------
// Per-checkpoint probing
// ---------------------------------------------------------------------------

/// Probe a single checkpoint for all spots.
fn probe_checkpoint(
    checkpoint_path: &Path,
    iteration: u32,
    spots: &[Spot],
    lhe_config: &LimitHoldemConfig,
    num_actions: usize,
    hidden_dim: usize,
    board_samples: usize,
    seed: u64,
) -> Result<Vec<ProbeResult>, Box<dyn Error>> {
    let p1_buf = ModelBuffer::load_from_file(&checkpoint_path.join("p1.bin"))?;
    let p2_buf = ModelBuffer::load_from_file(&checkpoint_path.join("p2.bin"))?;

    let device = candle_core::Device::Cpu;
    let policies = [
        ExplicitPolicy::from_buffer(&p1_buf, num_actions, hidden_dim, &device)?,
        ExplicitPolicy::from_buffer(&p2_buf, num_actions, hidden_dim, &device)?,
    ];

    let net_counts = [p1_buf.len(), p2_buf.len()];
    let encoder = LheEncoder::new();
    let game = LimitHoldem::new(lhe_config.clone(), 1, seed);

    spots
        .iter()
        .map(|spot| {
            probe_spot(&game, &encoder, &policies, spot, lhe_config, board_samples, seed, iteration, &net_counts)
        })
        .collect()
}

fn probe_spot(
    game: &LimitHoldem,
    encoder: &LheEncoder,
    policies: &[ExplicitPolicy; 2],
    spot: &Spot,
    lhe_config: &LimitHoldemConfig,
    board_samples: usize,
    seed: u64,
    iteration: u32,
    net_counts: &[usize; 2],
) -> Result<ProbeResult, Box<dyn Error>> {
    let hole = spot.hand;
    let deck = build_remaining_deck(&hole);
    let mut rng = StdRng::seed_from_u64(seed);

    let mut fold_sum = 0.0f64;
    let mut call_sum = 0.0f64;
    let mut raise_sum = 0.0f64;
    let mut last_raw = Vec::new();
    let mut count = 0usize;

    let pi = player_index(spot.player);

    for _ in 0..board_samples {
        let board = sample_board(&deck, &mut rng);
        let state = build_state(hole, board, spot.player, lhe_config);

        let state = match spot.preceding {
            Some(action) => game.next_state(&state, action),
            None => state,
        };

        let features = encoder.encode(&state, spot.player);
        let probs = match policies[pi].strategy(&features) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let actions = game.actions(&state);
        let (f, c, r) = classify_action_probs(&actions, &probs);
        fold_sum += f64::from(f);
        call_sum += f64::from(c);
        raise_sum += f64::from(r);
        count += 1;

        // Capture raw advantages from last board sample
        if let Ok(raw) = policies[pi].latest_raw_advantages(&features) {
            last_raw = raw;
        }
    }

    let n = count.max(1) as f64;
    Ok(ProbeResult {
        iteration,
        fold: (fold_sum / n) as f32,
        call: (call_sum / n) as f32,
        raise: (raise_sum / n) as f32,
        net_count: net_counts[pi],
        raw_advantages: last_raw,
    })
}

// ---------------------------------------------------------------------------
// Shared helpers (same patterns as lhe_viz)
// ---------------------------------------------------------------------------

fn build_remaining_deck(hole: &[Card; 2]) -> Vec<Card> {
    poker_solver_core::poker::full_deck()
        .into_iter()
        .filter(|c| *c != hole[0] && *c != hole[1])
        .collect()
}

fn sample_board(deck: &[Card], rng: &mut StdRng) -> [Card; 5] {
    let mut pool = deck.to_vec();
    pool.shuffle(rng);
    [pool[0], pool[1], pool[2], pool[3], pool[4]]
}

fn build_state(
    hole: [Card; 2],
    board: [Card; 5],
    target_player: Player,
    config: &LimitHoldemConfig,
) -> poker_solver_core::game::LimitHoldemState {
    let dummy_opp = pick_dummy_opponent(&hole, &board);
    match target_player {
        Player::Player1 => {
            poker_solver_core::game::LimitHoldemState::new_preflop(hole, dummy_opp, board, config.stack_depth)
        }
        Player::Player2 => {
            poker_solver_core::game::LimitHoldemState::new_preflop(dummy_opp, hole, board, config.stack_depth)
        }
    }
}

fn pick_dummy_opponent(hole: &[Card; 2], board: &[Card; 5]) -> [Card; 2] {
    let used: std::collections::HashSet<Card> = hole.iter().chain(board.iter()).copied().collect();
    let mut dummy = Vec::with_capacity(2);
    for card in poker_solver_core::poker::full_deck() {
        if !used.contains(&card) {
            dummy.push(card);
            if dummy.len() == 2 {
                break;
            }
        }
    }
    [dummy[0], dummy[1]]
}

const fn player_index(player: Player) -> usize {
    match player {
        Player::Player1 => 0,
        Player::Player2 => 1,
    }
}

fn classify_action_probs(actions: &[Action], probs: &[f32]) -> (f32, f32, f32) {
    let mut fold = 0.0f32;
    let mut call = 0.0f32;
    let mut raise = 0.0f32;

    for (i, &action) in actions.iter().enumerate() {
        let p = if i < probs.len() { probs[i] } else { 0.0 };
        match action {
            Action::Fold => fold += p,
            Action::Check | Action::Call => call += p,
            Action::Bet(_) | Action::Raise(_) => raise += p,
        }
    }
    (fold, call, raise)
}

// ---------------------------------------------------------------------------
// Table rendering
// ---------------------------------------------------------------------------

const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

fn render_table(spot: &Spot, results: &[ProbeResult]) {
    let action_label = match spot.preceding {
        None => "Fold/Call/Raise",
        Some(Action::Raise(_)) => "Fold/Call/Raise",
        Some(Action::Call) => "Check/Raise",
        _ => "Fold/Call/Raise",
    };

    println!("\n{BOLD}=== {} ({}) ==={RESET}", spot.label, action_label);
    println!(
        "{:>5} | {:>6} | {:>6} | {:>6} | {:>4} | Raw Adv (latest net)",
        "Iter", "Fold", "Call", "Raise", "Nets"
    );
    println!("------|--------|--------|--------|------|---------------------");

    for r in results {
        let raw_str = if r.raw_advantages.is_empty() {
            "n/a".to_string()
        } else {
            format!(
                "[{}]",
                r.raw_advantages
                    .iter()
                    .map(|v| format!("{v:>6.2}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };

        println!(
            "{:>5} | {:>6.3} | {:>6.3} | {:>6.3} | {:>4} | {}",
            r.iteration, r.fold, r.call, r.raise, r.net_count, raw_str
        );
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the trace-lhe diagnostic across all checkpoints.
#[allow(clippy::too_many_arguments)]
pub fn run_trace_lhe(
    dir: &Path,
    spot_strings: &[String],
    spots_file: Option<&Path>,
    every: u32,
    board_samples: usize,
    num_actions_override: Option<usize>,
    hidden_dim_override: Option<usize>,
    seed_override: Option<u64>,
    stack_depth_override: Option<u32>,
    num_streets_override: Option<u8>,
) -> Result<(), Box<dyn Error>> {
    // Collect spots from CLI args and/or file
    let mut spots: Vec<Spot> = spot_strings
        .iter()
        .map(|s| parse_spot(s))
        .collect::<Result<Vec<_>, _>>()?;

    if let Some(path) = spots_file {
        spots.extend(parse_spots_file(path)?);
    }

    if spots.is_empty() {
        return Err("no spots specified (use --spot or --spots-file)".into());
    }

    // Load training config for defaults
    let config_path = dir.join("training_config.yaml");
    let saved_config = if config_path.exists() {
        let yaml = std::fs::read_to_string(&config_path)?;
        Some(serde_yaml::from_str::<super::SdCfrTrainingConfig>(&yaml)?)
    } else {
        None
    };

    let num_actions = num_actions_override
        .or(saved_config.as_ref().map(|c| c.network.num_actions))
        .unwrap_or(3);
    let hidden_dim = hidden_dim_override
        .or(saved_config.as_ref().map(|c| c.network.hidden_dim))
        .unwrap_or(64);
    let seed = seed_override
        .or(saved_config.as_ref().map(|c| c.deals.seed))
        .unwrap_or(42);
    let stack_depth = stack_depth_override
        .or(saved_config.as_ref().and_then(|c| c.lhe_game.as_ref().map(|g| g.stack_depth)))
        .unwrap_or(20);
    let num_streets = num_streets_override
        .or(saved_config.as_ref().and_then(|c| c.lhe_game.as_ref().map(|g| g.num_streets)))
        .unwrap_or(4);

    let lhe_config = LimitHoldemConfig {
        stack_depth,
        num_streets,
        ..Default::default()
    };

    // Scan checkpoints
    let all_checkpoints = scan_checkpoints(dir)?;
    if all_checkpoints.is_empty() {
        return Err(format!("no lhe_checkpoint_N directories found in {}", dir.display()).into());
    }

    let checkpoints: Vec<_> = all_checkpoints
        .into_iter()
        .filter(|(n, _)| every <= 1 || *n % every == 0 || *n == 1)
        .collect();

    println!("=== LHE Strategy Trace ===");
    println!("  Directory: {}", dir.display());
    println!("  Checkpoints: {}", checkpoints.len());
    println!("  Spots: {}", spots.len());
    println!("  Board samples: {board_samples}");
    println!("  Network: num_actions={num_actions}, hidden_dim={hidden_dim}");
    println!("  LHE: stack_depth={stack_depth} BB, streets={num_streets}");

    // Collect results: results[spot_idx][checkpoint_idx]
    let mut all_results: Vec<Vec<ProbeResult>> = vec![Vec::new(); spots.len()];

    for (iter_num, cp_path) in &checkpoints {
        let results = probe_checkpoint(
            cp_path,
            *iter_num,
            &spots,
            &lhe_config,
            num_actions,
            hidden_dim,
            board_samples,
            seed,
        )?;
        for (i, result) in results.into_iter().enumerate() {
            all_results[i].push(result);
        }
    }

    // Render tables
    for (spot, results) in spots.iter().zip(all_results.iter()) {
        render_table(spot, results);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_spot_sb_pair() {
        let spot = parse_spot("SB AA").unwrap();
        assert_eq!(spot.player, Player::Player1);
        assert!(spot.preceding.is_none());
        assert_eq!(spot.hand[0].value, Value::Ace);
        assert_eq!(spot.hand[1].value, Value::Ace);
        assert_ne!(spot.hand[0].suit, spot.hand[1].suit);
    }

    #[test]
    fn parse_spot_bb_raise() {
        let spot = parse_spot("BB.R AKs").unwrap();
        assert_eq!(spot.player, Player::Player2);
        assert_eq!(spot.preceding, Some(Action::Raise(0)));
        assert_eq!(spot.hand[0].value, Value::Ace);
        assert_eq!(spot.hand[1].value, Value::King);
        assert_eq!(spot.hand[0].suit, spot.hand[1].suit); // suited
    }

    #[test]
    fn parse_spot_bb_call() {
        let spot = parse_spot("BB.C JTs").unwrap();
        assert_eq!(spot.player, Player::Player2);
        assert_eq!(spot.preceding, Some(Action::Call));
        assert_eq!(spot.hand[0].suit, spot.hand[1].suit); // suited
    }

    #[test]
    fn parse_hand_pairs() {
        assert_eq!(parse_hand("AA").unwrap(), (0, 0));
        assert_eq!(parse_hand("KK").unwrap(), (1, 1));
        assert_eq!(parse_hand("22").unwrap(), (12, 12));
    }

    #[test]
    fn parse_hand_suited() {
        // "AKs" → row=0 (A), col=1 (K) — upper triangle
        assert_eq!(parse_hand("AKs").unwrap(), (0, 1));
        // "76s" → row=7 (7), col=8 (6) — upper triangle (7 < 8 in index)
        // Wait: 7 is index 7, 6 is index 8, so (7, 8)
        assert_eq!(parse_hand("76s").unwrap(), (7, 8));
    }

    #[test]
    fn parse_hand_offsuit() {
        // "AKo" → row=1 (K), col=0 (A) — lower triangle (row > col)
        assert_eq!(parse_hand("AKo").unwrap(), (1, 0));
        // "72o" → row=12 (2), col=7 (7) — lower triangle
        assert_eq!(parse_hand("72o").unwrap(), (12, 7));
    }

    #[test]
    fn parse_hand_invalid() {
        assert!(parse_hand("X").is_err());
        assert!(parse_hand("AK").is_err()); // two different ranks, no suffix
        assert!(parse_hand("AKx").is_err());
    }

    #[test]
    fn parse_spot_invalid_position() {
        assert!(parse_spot("UTG AA").is_err());
    }

    #[test]
    fn scan_checkpoints_finds_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();

        // Create some checkpoint dirs
        std::fs::create_dir(base.join("lhe_checkpoint_1")).unwrap();
        std::fs::create_dir(base.join("lhe_checkpoint_5")).unwrap();
        std::fs::create_dir(base.join("lhe_checkpoint_10")).unwrap();
        std::fs::create_dir(base.join("other_dir")).unwrap();

        let result = scan_checkpoints(base).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].0, 1);
        assert_eq!(result[1].0, 5);
        assert_eq!(result[2].0, 10);
    }

    #[test]
    fn scan_checkpoints_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let result = scan_checkpoints(tmp.path()).unwrap();
        assert!(result.is_empty());
    }
}
