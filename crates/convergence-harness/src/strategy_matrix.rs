use range_solver::action_tree::Action;
use range_solver::card::card_to_string;
use range_solver::PostFlopGame;

// Card encoding: card_id = 4 * rank + suit
// rank: 2=0, 3=1, ..., A=12
// suit: club=0, diamond=1, heart=2, spade=3
//
// Matrix layout: rows/cols 0=Ace, 1=King, ..., 12=Two
// Pairs on diagonal, suited above (row < col), offsuit below (row > col)

const RANK_LABELS: [&str; 13] = [
    "A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2",
];

const CELL_W: usize = 9;

/// Extract rank (0=Two..12=Ace) from card_id.
fn card_rank(card: u8) -> usize {
    (card / 4) as usize
}

/// Convert card rank (0=Two..12=Ace) to matrix index (0=Ace..12=Two).
fn matrix_index(rank: usize) -> usize {
    12 - rank
}

/// Map a combo (two card IDs) to its 13x13 matrix cell (row, col).
/// Pairs on diagonal, suited above diagonal, offsuit below.
fn combo_to_cell(c1: u8, c2: u8) -> (usize, usize) {
    let r1 = card_rank(c1);
    let r2 = card_rank(c2);
    let s1 = c1 % 4;
    let s2 = c2 % 4;
    let (hi, lo) = if r1 >= r2 { (r1, r2) } else { (r2, r1) };
    let mi_hi = matrix_index(hi);
    let mi_lo = matrix_index(lo);
    // Pair and suited: hi on row, lo on col (on or above diagonal)
    // Offsuit: lo on row, hi on col (below diagonal)
    if hi == lo || s1 == s2 {
        (mi_hi, mi_lo)
    } else {
        (mi_lo, mi_hi)
    }
}

/// Return the hand label for a matrix cell (e.g. "AKs", "AKo", "AA").
fn cell_label(row: usize, col: usize) -> String {
    let (hi, lo) = if row <= col { (row, col) } else { (col, row) };
    let suffix = if row == col {
        ""
    } else if row < col {
        "s"
    } else {
        "o"
    };
    format!("{}{}{}", RANK_LABELS[hi], RANK_LABELS[lo], suffix)
}

/// Map an Action to an RGB color tuple.
fn action_color(action: &Action, pot: i32) -> (u8, u8, u8) {
    match action {
        Action::AllIn(_) => (255, 193, 7),    // gold
        Action::Raise(amt) | Action::Bet(amt) => {
            let frac = if pot > 0 {
                *amt as f32 / pot as f32
            } else {
                1.0
            };
            if frac > 1.5 {
                (183, 28, 28) // dark red
            } else if frac > 0.75 {
                (211, 47, 47) // red
            } else if frac > 0.4 {
                (255, 152, 0) // orange
            } else {
                (255, 183, 77) // light orange
            }
        }
        Action::Call => (76, 175, 80),  // green
        Action::Check => (76, 175, 80), // green
        Action::Fold => (68, 114, 196), // blue
        _ => (80, 80, 80),             // grey for None/Chance
    }
}

/// Map an Action to a short label string.
fn action_label(action: &Action, pot: i32) -> String {
    match action {
        Action::Fold => "F".into(),
        Action::Check => "X".into(),
        Action::Call => "C".into(),
        Action::AllIn(_) => "AI".into(),
        Action::Bet(amt) => {
            if pot > 0 {
                let pct = (*amt as f32 / pot as f32 * 100.0).round() as i32;
                format!("B{}%", pct)
            } else {
                format!("B{}", amt)
            }
        }
        Action::Raise(amt) => {
            if pot > 0 {
                let pct = (*amt as f32 / pot as f32 * 100.0).round() as i32;
                format!("R{}%", pct)
            } else {
                format!("R{}", amt)
            }
        }
        _ => "?".into(),
    }
}

/// Aggression rank: higher = more aggressive (sorted descending for left-to-right).
fn aggression_rank(action: &Action) -> i32 {
    match action {
        Action::AllIn(_) => 1000,
        Action::Raise(amt) => 200 + *amt,
        Action::Bet(amt) => 100 + *amt,
        Action::Call => -100,
        Action::Check => -100,
        Action::Fold => -200,
        _ => -300,
    }
}

fn luminance(r: u8, g: u8, b: u8) -> f32 {
    0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32
}

/// Render a single matrix cell as a colored string with Unicode half-blocks.
/// `policy` is a slice of (action, weight) pairs.
fn render_cell(policy: &[(Action, f32)], label: &str, pot: i32) -> String {
    let total: f32 = policy.iter().map(|(_, w)| w).sum();
    if total <= 0.0 {
        let bg = "\x1b[48;2;50;50;50m\x1b[97m";
        return format!("{}{:^w$}\x1b[0m", bg, label, w = CELL_W);
    }

    // Sort actions: most aggressive first
    let mut sorted: Vec<(Action, f32)> = policy.to_vec();
    sorted.sort_by_key(|a| std::cmp::Reverse(aggression_rank(&a.0)));

    // Compute how many half-columns each action gets (2x resolution)
    let half_cols = CELL_W * 2;
    let mut alloc: Vec<(Action, usize)> = Vec::new();
    let mut used = 0usize;
    for (i, (action, weight)) in sorted.iter().enumerate() {
        let frac = weight / total;
        let cols = if i == sorted.len() - 1 {
            half_cols.saturating_sub(used)
        } else {
            (frac * half_cols as f32).round().min((half_cols - used) as f32) as usize
        };
        if cols > 0 {
            alloc.push((*action, cols));
            used += cols;
        }
    }

    // Build a half-column color array
    let mut half_colors: Vec<(u8, u8, u8)> = Vec::with_capacity(half_cols);
    for (action, count) in &alloc {
        let color = action_color(action, pot);
        for _ in 0..*count {
            half_colors.push(color);
        }
    }
    while half_colors.len() < half_cols {
        half_colors.push((50, 50, 50));
    }

    // Render character-by-character
    let mut out = String::new();
    let label_start = (CELL_W.saturating_sub(label.len())) / 2;
    let label_bytes: Vec<char> = label.chars().collect();

    for i in 0..CELL_W {
        let left = half_colors[i * 2];
        let right = half_colors[i * 2 + 1];
        let label_idx = i.wrapping_sub(label_start);
        let label_char = if label_idx < label_bytes.len() {
            Some(label_bytes[label_idx])
        } else {
            None
        };

        if left == right {
            let (r, g, b) = left;
            let fg = if luminance(r, g, b) > 128.0 {
                "\x1b[30m"
            } else {
                "\x1b[97m"
            };
            let ch = label_char.unwrap_or(' ');
            out.push_str(&format!(
                "\x1b[48;2;{};{};{}m{}{}\x1b[0m",
                r, g, b, fg, ch
            ));
        } else if let Some(ch) = label_char {
            let (r, g, b) = left;
            let fg = if luminance(r, g, b) > 128.0 {
                "\x1b[30m"
            } else {
                "\x1b[97m"
            };
            out.push_str(&format!(
                "\x1b[48;2;{};{};{}m{}{}\x1b[0m",
                r, g, b, fg, ch
            ));
        } else {
            let (lr, lg, lb) = left;
            let (rr, rg, rb) = right;
            out.push_str(&format!(
                "\x1b[48;2;{};{};{}m\x1b[38;2;{};{};{}m\u{2590}\x1b[0m",
                lr, lg, lb, rr, rg, rb
            ));
        }
    }

    out
}

/// Print a colored 13x13 preflop strategy matrix to stderr.
///
/// The game must be at its root node (a player decision node, not chance/terminal).
/// `player` is 0 for OOP (SB) or 1 for IP (BB).
pub fn print_strategy_matrix(game: &PostFlopGame, player: usize) {
    let strategy = game.strategy();
    let actions = game.available_actions();
    let private_cards = game.private_cards(player);
    let num_hands = private_cards.len();
    let num_actions = actions.len();
    let pot = game.tree_config().starting_pot;

    // Build 13x13 grid: each cell accumulates (action_idx -> weight_sum), count
    let mut grid: Vec<Vec<(Vec<f32>, usize)>> = vec![vec![(vec![0.0; num_actions], 0); 13]; 13];

    for (hand_idx, &(c1, c2)) in private_cards.iter().enumerate() {
        let (row, col) = combo_to_cell(c1, c2);
        let cell = &mut grid[row][col];
        for action_idx in 0..num_actions {
            let strat_val = strategy[action_idx * num_hands + hand_idx];
            cell.0[action_idx] += strat_val;
        }
        cell.1 += 1;
    }

    let flop = game.card_config().flop;
    let flop_str = flop.iter()
        .map(|&c| card_to_string(c).unwrap_or_else(|_| "??".into()))
        .collect::<Vec<_>>()
        .join(" ");

    eprintln!("\n  {} Strategy Matrix — Flop: [{}]",
        if player == 0 { "OOP (SB)" } else { "IP (BB)" },
        flop_str);
    eprintln!();

    for (row, grid_row) in grid.iter().enumerate() {
        eprint!("  ");
        for (col, (sums, count)) in grid_row.iter().enumerate() {
            let policy: Vec<(Action, f32)> = if *count > 0 {
                actions
                    .iter()
                    .enumerate()
                    .map(|(i, &a)| (a, sums[i] / *count as f32))
                    .filter(|(_, w)| *w > 0.001)
                    .collect()
            } else {
                vec![]
            };
            let label = cell_label(row, col);
            if col < 12 {
                eprint!("{} ", render_cell(&policy, &label, pot));
            } else {
                eprint!("{}", render_cell(&policy, &label, pot));
            }
        }
        eprintln!("\x1b[0m");
    }

    // Legend
    let mut seen_actions: Vec<Action> = Vec::new();
    for grid_row in &grid {
        for (sums, count) in grid_row {
            if *count > 0 {
                for (i, &a) in actions.iter().enumerate() {
                    if sums[i] / *count as f32 > 0.001 && !seen_actions.contains(&a) {
                        seen_actions.push(a);
                    }
                }
            }
        }
    }
    seen_actions.sort_by_key(|a| std::cmp::Reverse(aggression_rank(a)));

    if !seen_actions.is_empty() {
        eprintln!();
        eprint!("  ");
        for action in &seen_actions {
            let (r, g, b) = action_color(action, pot);
            let fg = if luminance(r, g, b) > 128.0 {
                "\x1b[30m"
            } else {
                "\x1b[97m"
            };
            let lbl = action_label(action, pot);
            eprint!(
                "\x1b[48;2;{};{};{}m{} {:^5} \x1b[0m ",
                r, g, b, fg, lbl
            );
        }
        eprintln!();
    }
    eprintln!();
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── combo_to_cell tests ──

    #[test]
    fn test_combo_to_cell_pair_aa() {
        // Ac=48, Ad=49 → both rank 12 (Ace) → matrix (0,0)
        assert_eq!(combo_to_cell(48, 49), (0, 0));
    }

    #[test]
    fn test_combo_to_cell_pair_22() {
        // 2c=0, 2d=1 → both rank 0 (Two) → matrix (12,12)
        assert_eq!(combo_to_cell(0, 1), (12, 12));
    }

    #[test]
    fn test_combo_to_cell_suited_aks() {
        // Ac=48, Kc=44 → rank 12, rank 11, same suit → suited → (0, 1)
        assert_eq!(combo_to_cell(48, 44), (0, 1));
    }

    #[test]
    fn test_combo_to_cell_offsuit_ako() {
        // Ac=48, Kd=45 → rank 12, rank 11, diff suit → offsuit → (1, 0)
        assert_eq!(combo_to_cell(48, 45), (1, 0));
    }

    #[test]
    fn test_combo_to_cell_suited_order_independent() {
        // AKs should give same result regardless of card order
        assert_eq!(combo_to_cell(48, 44), combo_to_cell(44, 48));
    }

    #[test]
    fn test_combo_to_cell_offsuit_order_independent() {
        // AKo should give same result regardless of card order
        assert_eq!(combo_to_cell(48, 45), combo_to_cell(45, 48));
    }

    #[test]
    fn test_combo_to_cell_pair_kk() {
        // Kc=44, Kd=45 → rank 11 → matrix (1,1)
        assert_eq!(combo_to_cell(44, 45), (1, 1));
    }

    #[test]
    fn test_combo_to_cell_suited_small_cards() {
        // 3c=4, 2c=0 → rank 1, rank 0, same suit → suited → (11, 12)
        assert_eq!(combo_to_cell(4, 0), (11, 12));
    }

    #[test]
    fn test_combo_to_cell_offsuit_small_cards() {
        // 3c=4, 2d=1 → rank 1, rank 0, diff suit → offsuit → (12, 11)
        assert_eq!(combo_to_cell(4, 1), (12, 11));
    }

    // ── action_color tests ──

    #[test]
    fn test_action_color_fold_returns_blue() {
        assert_eq!(action_color(&Action::Fold, 100), (68, 114, 196));
    }

    #[test]
    fn test_action_color_check_returns_green() {
        assert_eq!(action_color(&Action::Check, 100), (76, 175, 80));
    }

    #[test]
    fn test_action_color_call_returns_green() {
        assert_eq!(action_color(&Action::Call, 100), (76, 175, 80));
    }

    #[test]
    fn test_action_color_allin_returns_gold() {
        assert_eq!(action_color(&Action::AllIn(200), 100), (255, 193, 7));
    }

    #[test]
    fn test_action_color_large_bet_returns_dark_red() {
        // Bet 200 into pot 100 = 200% pot → > 1.5 → dark red
        assert_eq!(action_color(&Action::Bet(200), 100), (183, 28, 28));
    }

    #[test]
    fn test_action_color_medium_bet_returns_red() {
        // Bet 100 into pot 100 = 100% pot → > 0.75 → red
        assert_eq!(action_color(&Action::Bet(100), 100), (211, 47, 47));
    }

    #[test]
    fn test_action_color_small_bet_returns_orange() {
        // Bet 50 into pot 100 = 50% pot → > 0.4 → orange
        assert_eq!(action_color(&Action::Bet(50), 100), (255, 152, 0));
    }

    #[test]
    fn test_action_color_tiny_bet_returns_light_orange() {
        // Bet 30 into pot 100 = 30% pot → ≤ 0.4 → light orange
        assert_eq!(action_color(&Action::Bet(30), 100), (255, 183, 77));
    }

    #[test]
    fn test_action_color_raise_uses_same_spectrum() {
        // Raise 200 into pot 100 → same as Bet with same fraction
        assert_eq!(
            action_color(&Action::Raise(200), 100),
            action_color(&Action::Bet(200), 100)
        );
    }

    // ── action_label tests ──

    #[test]
    fn test_action_label_fold() {
        assert_eq!(action_label(&Action::Fold, 100), "F");
    }

    #[test]
    fn test_action_label_check() {
        assert_eq!(action_label(&Action::Check, 100), "X");
    }

    #[test]
    fn test_action_label_call() {
        assert_eq!(action_label(&Action::Call, 100), "C");
    }

    #[test]
    fn test_action_label_allin() {
        assert_eq!(action_label(&Action::AllIn(200), 100), "AI");
    }

    #[test]
    fn test_action_label_bet_pot_relative() {
        // Bet 67 into pot 100 = 67%
        assert_eq!(action_label(&Action::Bet(67), 100), "B67%");
    }

    #[test]
    fn test_action_label_raise_pot_relative() {
        assert_eq!(action_label(&Action::Raise(150), 100), "R150%");
    }

    #[test]
    fn test_action_label_bet_zero_pot() {
        // Edge case: pot is 0 → fall back to raw amount
        assert_eq!(action_label(&Action::Bet(50), 0), "B50");
    }

    // ── cell_label tests ──

    #[test]
    fn test_cell_label_pair() {
        assert_eq!(cell_label(0, 0), "AA");
        assert_eq!(cell_label(12, 12), "22");
    }

    #[test]
    fn test_cell_label_suited() {
        assert_eq!(cell_label(0, 1), "AKs");
    }

    #[test]
    fn test_cell_label_offsuit() {
        assert_eq!(cell_label(1, 0), "AKo");
    }

    // ── render_cell tests ──

    #[test]
    fn test_render_cell_empty_policy() {
        let result = render_cell(&[], "AA", 100);
        assert!(result.contains("AA"));
        assert!(result.contains("\x1b[0m")); // has reset code
    }

    #[test]
    fn test_render_cell_single_action() {
        let policy = vec![(Action::Fold, 1.0)];
        let result = render_cell(&policy, "AKo", 100);
        let stripped = strip_ansi(&result);
        assert!(
            stripped.contains("AKo"),
            "Expected 'AKo' in stripped output: {:?}",
            stripped
        );
        assert!(result.contains("\x1b[0m"));
    }

    #[test]
    fn test_render_cell_multiple_actions() {
        let policy = vec![(Action::Fold, 0.3), (Action::Call, 0.7)];
        let result = render_cell(&policy, "TT", 100);
        let stripped = strip_ansi(&result);
        assert!(
            stripped.contains("TT"),
            "Expected 'TT' in stripped output: {:?}",
            stripped
        );
    }

    #[test]
    fn test_render_cell_width_is_cell_w() {
        // Strip ANSI codes and check character count
        let policy = vec![(Action::Fold, 1.0)];
        let result = render_cell(&policy, "AA", 100);
        let stripped = strip_ansi(&result);
        assert_eq!(
            stripped.chars().count(),
            CELL_W,
            "Cell should be {} characters wide, got {}",
            CELL_W,
            stripped.chars().count()
        );
    }

    // ── aggression_rank tests ──

    #[test]
    fn test_aggression_rank_ordering() {
        assert!(aggression_rank(&Action::AllIn(100)) > aggression_rank(&Action::Raise(50)));
        assert!(aggression_rank(&Action::Raise(50)) > aggression_rank(&Action::Bet(50)));
        assert!(aggression_rank(&Action::Bet(50)) > aggression_rank(&Action::Call));
        assert!(aggression_rank(&Action::Call) > aggression_rank(&Action::Fold));
    }

    // ── luminance tests ──

    #[test]
    fn test_luminance_black() {
        assert!((luminance(0, 0, 0) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_luminance_white() {
        assert!((luminance(255, 255, 255) - 255.0).abs() < 0.01);
    }

    // ── helpers ──

    /// Strip ANSI escape codes from a string.
    fn strip_ansi(s: &str) -> String {
        let mut result = String::new();
        let mut in_escape = false;
        for ch in s.chars() {
            if ch == '\x1b' {
                in_escape = true;
            } else if in_escape {
                if ch.is_ascii_alphabetic() {
                    in_escape = false;
                }
            } else {
                result.push(ch);
            }
        }
        result
    }
}
