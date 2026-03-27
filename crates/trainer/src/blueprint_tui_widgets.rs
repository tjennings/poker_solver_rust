//! Hand-grid widget for the Blueprint V2 TUI dashboard.
//!
//! Renders a 13x13 grid of poker starting hands with color-coded cells
//! reflecting the dominant action from the current strategy snapshot.

use ratatui::prelude::*;
use ratatui::widgets::Widget;

// ─── CellStrategy ────────────────────────────────────────────────────────────

/// Per-cell strategy data: a distribution over named actions.
#[derive(Debug, Clone, Default)]
pub struct CellStrategy {
    /// `(action_name, frequency)` pairs.
    pub actions: Vec<(String, f32)>,
    /// Mean chip EV for this hand group (big blinds).
    pub ev: Option<f32>,
}

impl CellStrategy {
    /// Returns `(action_name, frequency)` of the most-frequent action.
    ///
    /// Ties are broken by position (first wins). Returns `("", 0.0)` when the
    /// action list is empty.
    pub fn dominant_action(&self) -> (&str, f32) {
        self.actions
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(name, freq)| (name.as_str(), *freq))
            .unwrap_or(("", 0.0))
    }

    /// `true` if L1 distance from `previous` is below `threshold`.
    ///
    /// When the action vectors differ in length the comparison is
    /// automatically "not converged".
    pub fn is_converged(&self, previous: &CellStrategy, threshold: f32) -> bool {
        if self.actions.len() != previous.actions.len() {
            return false;
        }
        let delta: f32 = self
            .actions
            .iter()
            .zip(previous.actions.iter())
            .map(|((_, a), (_, b))| (a - b).abs())
            .sum();
        delta < threshold
    }
}

// ─── HandGridState ───────────────────────────────────────────────────────────

/// Full state for rendering a 13x13 hand grid.
#[derive(Debug, Clone)]
pub struct HandGridState {
    pub cells: [[CellStrategy; 13]; 13],
    /// Previous strategy snapshot for convergence detection.
    pub prev_cells: Option<[[CellStrategy; 13]; 13]>,
    pub scenario_name: String,
    pub action_path: Vec<String>,
    pub board_display: Option<String>,
    pub cluster_id: Option<u32>,
    pub street_label: String,
    pub iteration_at_snapshot: u64,
}

// ─── Color helpers ───────────────────────────────────────────────────────────

/// Map an action name to a base colour.
///
/// Fold = grey, Check = blue, Call = green,
/// Bets/Raises = light red → dark red (scaled by bet size), All-in = darkest red.
/// Map an action name to a colour. For bet/raise, `rank` is the position
/// among bet/raise actions (0 = largest) and `total_bets` is how many
/// bet/raise actions exist. This spreads the red gradient evenly.
pub fn action_color(name: &str) -> Color {
    action_color_ranked(name, 0, 1)
}

pub fn action_color_ranked(name: &str, rank: usize, total_bets: usize) -> Color {
    let lower = name.to_ascii_lowercase();
    if lower == "fold" {
        return Color::Rgb(140, 140, 140);
    }
    if lower == "check" {
        return Color::Rgb(80, 140, 220);
    }
    if lower == "call" {
        return Color::Rgb(60, 179, 113);
    }
    if lower.starts_with("bet") || lower.starts_with("raise") {
        // Spread evenly across the red gradient based on rank among bets.
        // rank 0 = largest bet (darkest), rank total-1 = smallest (lightest).
        let t = if total_bets <= 1 {
            0.5
        } else {
            rank as f64 / (total_bets - 1) as f64
        };
        // t=0 → darkest (largest bet), t=1 → lightest (smallest bet)
        let r = (140.0 + 115.0 * t) as u8; // 140 → 255
        let g = (30.0 + 90.0 * t) as u8;   // 30 → 120
        let b = (30.0 + 70.0 * t) as u8;   // 30 → 100
        return Color::Rgb(r, g, b);
    }
    if lower == "all-in" || lower == "allin" {
        return Color::Rgb(180, 30, 180); // purple for all-in
    }
    Color::DarkGray
}

// ─── Widget ──────────────────────────────────────────────────────────────────

/// Ratatui widget that paints the 13x13 hand grid.
pub struct HandGridWidget<'a> {
    pub state: &'a HandGridState,
}

impl Widget for &HandGridWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        use poker_solver_core::hands::CanonicalHand;

        // 3-row cells: row 0-1 = stacked color bar, row 2 = hand label.
        let cell_h: u16 = 3;
        if area.height < 4 || area.width < 13 {
            return;
        }

        // ── Title row ────────────────────────────────────────────────
        let title = format!(
            " {} | {} ",
            self.state.scenario_name, self.state.street_label,
        );
        buf.set_string(area.x, area.y, &title, Style::default().bold());

        // ── Grid rows ───────────────────────────────────────────────
        // Last column of each cell is a vertical border (black or blue).
        let cell_w = cell_width(area.width);
        let content_w = cell_w.saturating_sub(1).max(2);
        let grid_x0 = area.x;
        let grid_y0 = area.y + 1;
        for r in 0..13u16 {
            let y0 = grid_y0 + r * cell_h;       // color bar row 1
            let y1 = y0 + 1;                      // color bar row 2
            let y2 = y0 + 2;                      // hand label row
            if y0 >= area.y + area.height {
                break;
            }

            for c in 0..13u16 {
                let cell = &self.state.cells[r as usize][c as usize];
                let x = grid_x0 + c * cell_w;
                if x + cell_w > area.x + area.width {
                    continue;
                }

                // Determine border color: blue if strategy changed, black otherwise.
                let border_color = if let Some(ref prev) = self.state.prev_cells {
                    if !cell.is_converged(&prev[r as usize][c as usize], 0.01) {
                        Color::Rgb(60, 120, 255)
                    } else {
                        Color::Rgb(0, 0, 0)
                    }
                } else {
                    Color::Rgb(0, 0, 0)
                };
                let border_style = Style::default().bg(border_color);

                // Rows 0-1: stacked color bar (doubled height)
                for &y in &[y0, y1] {
                    if y >= area.y + area.height {
                        break;
                    }
                    if cell.actions.is_empty() {
                        let style = Style::default().bg(Color::Rgb(40, 40, 40));
                        for dx in 0..content_w {
                            buf.set_string(x + dx, y, " ", style);
                        }
                    } else {
                        render_color_bar(buf, x, y, content_w, &cell.actions);
                    }
                    buf.set_string(x + content_w, y, " ", border_style);
                }

                // Row 2: hand label + EV (black bg) + remainder = color bar
                if y2 < area.y + area.height {
                    let hand_name =
                        CanonicalHand::from_matrix_position(r as usize, c as usize)
                            .map_or_else(|| "   ".to_string(), |h| format!("{h}"));
                    let hand_label = if let Some(ev) = cell.ev {
                        // Convert chips to BB for display
                        let ev_bb = ev / 2.0;
                        let ev_bb = if ev_bb.abs() < 0.005 { 0.0_f32 } else { ev_bb };
                        let sign = if ev_bb > 0.0 { "+" } else { "" };
                        format!("{hand_name} {sign}{ev_bb:.2}")
                    } else {
                        hand_name
                    };
                    let label_len = hand_label.len().min(content_w as usize);
                    let label_style =
                        Style::default().bg(Color::Rgb(0, 0, 0)).fg(Color::White);
                    buf.set_string(x, y2, &hand_label[..label_len], label_style);

                    // Fill remainder with the color bar pattern
                    let remainder_start = label_len as u16;
                    if remainder_start < content_w {
                        if cell.actions.is_empty() {
                            let style = Style::default().bg(Color::Rgb(40, 40, 40));
                            for dx in remainder_start..content_w {
                                buf.set_string(x + dx, y2, " ", style);
                            }
                        } else {
                            // Render full color bar then we already wrote the label
                            // over the first chars — just render the tail portion.
                            render_color_bar_offset(
                                buf, x, y2, content_w, &cell.actions, remainder_start,
                            );
                        }
                    }
                    buf.set_string(x + content_w, y2, " ", border_style);
                }
            }
        }

        // ── Footer ──────────────────────────────────────────────────
        let footer_y = grid_y0 + 13 * cell_h;
        if footer_y < area.y + area.height {
            let board = self
                .state
                .board_display
                .as_deref()
                .unwrap_or("--");
            let cluster = self
                .state
                .cluster_id
                .map_or_else(|| "--".to_string(), |id| format!("c{id}"));
            let path = if self.state.action_path.is_empty() {
                "root".to_string()
            } else {
                self.state.action_path.join(" > ")
            };
            let info = format!(
                "board:{board}  cluster:{cluster}  iter:{}  path:{path}",
                self.state.iteration_at_snapshot,
            );
            buf.set_string(area.x, footer_y, &info, Style::default().dim());
        }

        // ── Legend row ──────────────────────────────────────────────
        let legend_y = footer_y + 1;
        if legend_y < area.y + area.height {
            let mut seen = Vec::new();
            for row in &self.state.cells {
                for cell in row {
                    for (name, _) in &cell.actions {
                        if !seen.iter().any(|s: &String| s == name) {
                            seen.push(name.clone());
                        }
                    }
                }
            }
            seen.sort_by_key(|name| action_sort_key(name));

            let total_bets = seen.iter().filter(|n| is_bet_or_raise(n)).count();
            let mut bet_rank = 0;
            let mut x = area.x;
            for label in &seen {
                let color = if is_bet_or_raise(label) {
                    let c = action_color_ranked(label, bet_rank, total_bets);
                    bet_rank += 1;
                    c
                } else {
                    action_color(label)
                };
                let style = Style::default().bg(color).fg(Color::Black);
                let display = label.to_ascii_uppercase();
                buf.set_string(x, legend_y, format!(" {display} "), style);
                x += display.len() as u16 + 3;
            }
        }
    }
}

fn is_bet_or_raise(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.starts_with("bet") || lower.starts_with("raise")
}

/// Sort key for display ordering: all-in, raises large→small, call/check, fold.
fn action_sort_key(name: &str) -> (u8, i64) {
    let lower = name.to_ascii_lowercase();
    if lower == "all-in" || lower == "allin" {
        (0, 0) // all-in first
    } else if lower.starts_with("bet") || lower.starts_with("raise") {
        let size: f64 = lower
            .trim_start_matches("bet")
            .trim_start_matches("raise")
            .trim()
            .parse()
            .unwrap_or(0.0);
        // Negate so larger sizes sort first
        (1, (-size * 1000.0) as i64)
    } else if lower == "call" || lower == "check" {
        (2, 0)
    } else if lower == "fold" {
        (3, 0)
    } else {
        (4, 0)
    }
}

/// Render a stacked color bar for a cell's action distribution.
///
/// Each character position maps to a fraction of the cell width. Actions
/// are drawn left-to-right: all-in, raises large→small, call/check, fold.
fn render_color_bar(buf: &mut Buffer, x: u16, y: u16, w: u16, actions: &[(String, f32)]) {
    let total: f32 = actions.iter().map(|(_, f)| f).sum();
    if total <= 0.0 {
        let style = Style::default().bg(Color::Rgb(40, 40, 40));
        for dx in 0..w {
            buf.set_string(x + dx, y, " ", style);
        }
        return;
    }

    let mut sorted: Vec<_> = actions.to_vec();
    sorted.sort_by_key(|(name, _)| action_sort_key(name));

    // Count bet/raise actions and assign each a rank for the color gradient.
    let bet_ranks: Vec<Option<(usize, usize)>> = {
        let total_bets = sorted.iter().filter(|(n, _)| is_bet_or_raise(n)).count();
        let mut rank = 0;
        sorted.iter().map(|(n, _)| {
            if is_bet_or_raise(n) {
                let r = rank;
                rank += 1;
                Some((r, total_bets))
            } else {
                None
            }
        }).collect()
    };

    let mut col = 0_u16;
    let mut cumulative = 0.0_f32;
    for (i, (name, freq)) in sorted.iter().enumerate() {
        cumulative += freq / total;
        let end = if i == sorted.len() - 1 {
            w
        } else {
            (cumulative * w as f32).round() as u16
        };
        let color = if let Some((rank, total_bets)) = bet_ranks[i] {
            action_color_ranked(name, rank, total_bets)
        } else {
            action_color(name)
        };
        let style = Style::default().bg(color);
        while col < end && col < w {
            buf.set_string(x + col, y, " ", style);
            col += 1;
        }
    }
}

/// Like `render_color_bar` but only writes characters from column `skip` onward.
/// Used for the label row where the first few chars are the hand name on black.
fn render_color_bar_offset(
    buf: &mut Buffer,
    x: u16,
    y: u16,
    w: u16,
    actions: &[(String, f32)],
    skip: u16,
) {
    let total: f32 = actions.iter().map(|(_, f)| f).sum();
    if total <= 0.0 {
        let style = Style::default().bg(Color::Rgb(40, 40, 40));
        for dx in skip..w {
            buf.set_string(x + dx, y, " ", style);
        }
        return;
    }

    let mut sorted: Vec<_> = actions.to_vec();
    sorted.sort_by_key(|(name, _)| action_sort_key(name));

    let bet_ranks: Vec<Option<(usize, usize)>> = {
        let total_bets = sorted.iter().filter(|(n, _)| is_bet_or_raise(n)).count();
        let mut rank = 0;
        sorted
            .iter()
            .map(|(n, _)| {
                if is_bet_or_raise(n) {
                    let r = rank;
                    rank += 1;
                    Some((r, total_bets))
                } else {
                    None
                }
            })
            .collect()
    };

    let mut col = 0_u16;
    let mut cumulative = 0.0_f32;
    for (i, (name, freq)) in sorted.iter().enumerate() {
        cumulative += freq / total;
        let end = if i == sorted.len() - 1 {
            w
        } else {
            (cumulative * w as f32).round() as u16
        };
        let color = if let Some((rank, total_bets)) = bet_ranks[i] {
            action_color_ranked(name, rank, total_bets)
        } else {
            action_color(name)
        };
        let style = Style::default().bg(color);
        while col < end && col < w {
            if col >= skip {
                buf.set_string(x + col, y, " ", style);
            }
            col += 1;
        }
    }
}

/// Get the color for the dominant action, respecting bet/raise rank gradient.
fn dominant_action_color(dom_name: &str, actions: &[(String, f32)]) -> Color {
    if is_bet_or_raise(dom_name) {
        let mut sorted_names: Vec<_> = actions
            .iter()
            .filter(|(n, _)| is_bet_or_raise(n))
            .map(|(n, _)| n.clone())
            .collect();
        sorted_names.sort_by_key(|n| action_sort_key(n));
        let total_bets = sorted_names.len();
        let rank = sorted_names.iter().position(|n| n == dom_name).unwrap_or(0);
        action_color_ranked(dom_name, rank, total_bets)
    } else {
        action_color(dom_name)
    }
}

/// Calculate per-cell character width from the available area width.
fn cell_width(area_w: u16) -> u16 {
    // No axis labels — full width divided among 13 columns.
    (area_w / 13).max(3)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use test_macros::timed_test;

    /// Build a mock grid where AA = 80% raise, 72o = 90% fold, rest = empty.
    fn mock_grid_state() -> HandGridState {
        // INVARIANT: Default::default() for [[CellStrategy; 13]; 13] gives
        // empty action vecs in every cell.
        let cells: [[CellStrategy; 13]; 13] = std::array::from_fn(|_| {
            std::array::from_fn(|_| CellStrategy::default())
        });
        let mut state = HandGridState {
            cells,
            prev_cells: None,
            scenario_name: "UTG open".to_string(),
            action_path: vec!["raise".to_string()],
            board_display: Some("Ah Kd 7c".to_string()),
            cluster_id: Some(42),
            street_label: "Flop".to_string(),
            iteration_at_snapshot: 5000,
        };

        // AA is row=0, col=0 (pair on diagonal)
        state.cells[0][0] = CellStrategy {
            actions: vec![
                ("raise 1.0".to_string(), 0.80),
                ("fold".to_string(), 0.10),
                ("call".to_string(), 0.10),
            ],
            ev: Some(5.1),
        };

        // 72o: row 6 (7), col 12 (2) — below diagonal = offsuit
        state.cells[6][12] = CellStrategy {
            actions: vec![
                ("fold".to_string(), 0.90),
                ("call".to_string(), 0.05),
                ("raise 1.0".to_string(), 0.05),
            ],
            ev: Some(-2.3),
        };

        state
    }

    #[timed_test(10)]
    fn action_color_mapping() {
        assert_eq!(action_color("fold"), Color::Rgb(140, 140, 140));
        assert_eq!(action_color("check"), Color::Rgb(80, 140, 220));
        assert_eq!(action_color("call"), Color::Rgb(60, 179, 113));
        // Bets/raises: smaller size → lighter red
        assert!(matches!(action_color("bet 0.3"), Color::Rgb(..)));
        assert!(matches!(action_color("raise 1.0"), Color::Rgb(..)));
        // All-in is darkest
        assert_eq!(action_color("all-in"), Color::Rgb(180, 30, 180));
    }

    #[timed_test(10)]
    fn dominant_action_picks_highest() {
        let cell = CellStrategy {
            actions: vec![
                ("fold".to_string(), 0.2),
                ("call".to_string(), 0.7),
                ("raise 1.0".to_string(), 0.1),
            ],
            ev: None,
        };
        let (name, freq) = cell.dominant_action();
        assert_eq!(name, "call");
        assert!((freq - 0.7).abs() < 1e-6);
    }

    #[timed_test(10)]
    fn convergence_stable() {
        let current = CellStrategy {
            actions: vec![
                ("fold".to_string(), 0.50),
                ("call".to_string(), 0.50),
            ],
            ev: None,
        };
        let previous = CellStrategy {
            actions: vec![
                ("fold".to_string(), 0.51),
                ("call".to_string(), 0.49),
            ],
            ev: None,
        };
        assert!(current.is_converged(&previous, 0.05));
    }

    #[timed_test(10)]
    fn convergence_moving() {
        let current = CellStrategy {
            actions: vec![
                ("fold".to_string(), 0.70),
                ("call".to_string(), 0.30),
            ],
            ev: None,
        };
        let previous = CellStrategy {
            actions: vec![
                ("fold".to_string(), 0.40),
                ("call".to_string(), 0.60),
            ],
            ev: None,
        };
        assert!(!current.is_converged(&previous, 0.05));
    }

    #[timed_test(10)]
    fn widget_renders_without_panic() {
        let state = mock_grid_state();
        let widget = HandGridWidget { state: &state };
        let backend = TestBackend::new(80, 45);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                frame.render_widget(&widget, frame.area());
            })
            .unwrap();
    }

    #[timed_test(10)]
    fn widget_renders_convergence_border() {
        // Create grid with prev_cells that are very similar -> converged
        // cells should render blue border on changed cells.
        let mut state = mock_grid_state();
        let mut prev = state.cells.clone();
        // Slightly perturb prev to make them converged (delta < 0.01)
        if let Some(action) = prev[0][0].actions.get_mut(0) {
            action.1 += 0.005;
        }
        state.prev_cells = Some(prev);

        let backend = TestBackend::new(160, 45);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                let area = frame.area();
                frame.render_widget(&HandGridWidget { state: &state }, area);
            })
            .unwrap();
        // No panic = success
    }
}
