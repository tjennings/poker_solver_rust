//! Hand-grid widget for the Blueprint V2 TUI dashboard.
//!
//! Renders a 13x13 grid of poker starting hands with color-coded cells
//! reflecting the dominant action from the current strategy snapshot.

use ratatui::prelude::*;
use ratatui::widgets::Widget;

/// Rank labels for rows/columns: A down to 2.
const RANK_LABELS: [&str; 13] = [
    "A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2",
];

// ─── CellStrategy ────────────────────────────────────────────────────────────

/// Per-cell strategy data: a distribution over named actions.
#[derive(Debug, Clone, Default)]
pub struct CellStrategy {
    /// `(action_name, frequency)` pairs.
    pub actions: Vec<(String, f32)>,
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
    pub scenario_name: String,
    pub action_path: Vec<String>,
    pub board_display: Option<String>,
    pub cluster_id: Option<u32>,
    pub street_label: String,
    pub iteration_at_snapshot: u64,
}

// ─── Color helpers ───────────────────────────────────────────────────────────

/// Map an action name to a base colour.
pub fn action_color(name: &str) -> Color {
    let lower = name.to_ascii_lowercase();
    if lower == "fold" {
        return Color::Rgb(180, 60, 60);
    }
    if lower == "check" {
        return Color::Rgb(100, 149, 237);
    }
    if lower == "call" {
        return Color::Rgb(60, 179, 113);
    }
    if lower.starts_with("bet") || lower.starts_with("raise") {
        let digit = lower.chars().last().and_then(|c| c.to_digit(10));
        return match digit {
            Some(0) => Color::Rgb(230, 190, 50),
            Some(1) => Color::Rgb(230, 150, 40),
            Some(2) => Color::Rgb(220, 120, 40),
            Some(3) => Color::Rgb(200, 80, 120),
            _ => Color::Rgb(180, 60, 180),
        };
    }
    if lower.contains("all") || lower.contains("ai") {
        return Color::Rgb(255, 255, 255);
    }
    Color::DarkGray
}

/// Scale a colour's brightness by `dominance` (the dominant action's
/// frequency).  Maps the range `[0.33, 1.0]` to brightness `[40%, 100%]`.
/// Values outside the range are clamped.
pub fn dim_color(color: Color, dominance: f32) -> Color {
    let factor = if dominance <= 0.33 {
        0.4
    } else if dominance >= 1.0 {
        1.0
    } else {
        // Linear interpolation: 0.33 → 0.4, 1.0 → 1.0
        0.4 + (dominance - 0.33) / (1.0 - 0.33) * 0.6
    };

    match color {
        Color::Rgb(r, g, b) => {
            let scale = |v: u8| (f32::from(v) * factor).round().min(255.0) as u8;
            Color::Rgb(scale(r), scale(g), scale(b))
        }
        other => other,
    }
}

// ─── Widget ──────────────────────────────────────────────────────────────────

/// Ratatui widget that paints the 13x13 hand grid.
pub struct HandGridWidget<'a> {
    pub state: &'a HandGridState,
}

impl Widget for &HandGridWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Layout: 2 header rows, 13 grid rows, 2 footer rows = 17 minimum.
        if area.height < 17 || area.width < 15 {
            return; // too small to render
        }

        // ── Title row ────────────────────────────────────────────────
        let title = format!(
            " {} | {} ",
            self.state.scenario_name, self.state.street_label,
        );
        buf.set_string(area.x, area.y, &title, Style::default().bold());

        // ── Column labels ────────────────────────────────────────────
        let col_y = area.y + 1;
        let cell_w = cell_width(area.width);
        let grid_x0 = area.x + 3; // space for row label + " "
        for (c, label) in RANK_LABELS.iter().enumerate() {
            let x = grid_x0 + (c as u16) * cell_w;
            if x < area.x + area.width {
                buf.set_string(x, col_y, label, Style::default().dim());
            }
        }

        // ── Grid rows ───────────────────────────────────────────────
        let grid_y0 = area.y + 2;
        for r in 0..13u16 {
            let y = grid_y0 + r;
            if y >= area.y + area.height {
                break;
            }

            // Row label
            buf.set_string(area.x, y, RANK_LABELS[r as usize], Style::default().dim());

            for c in 0..13u16 {
                let cell = &self.state.cells[r as usize][c as usize];
                let (action, freq) = cell.dominant_action();
                let base = action_color(action);
                let bg = dim_color(base, freq);
                let pct = (freq * 100.0).round() as u32;
                let label = if pct > 0 {
                    format!("{pct:>3}")
                } else {
                    "   ".to_string()
                };

                let x = grid_x0 + c * cell_w;
                if x + cell_w <= area.x + area.width {
                    let style = Style::default().bg(bg).fg(Color::Black);
                    buf.set_string(x, y, &label, style);
                }
            }
        }

        // ── Footer ──────────────────────────────────────────────────
        let footer_y = grid_y0 + 13;
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
            let legend_items: &[(&str, Color)] = &[
                ("FOLD", action_color("fold")),
                ("CHK", action_color("check")),
                ("CALL", action_color("call")),
                ("BET-S", action_color("bet0")),
                ("BET-M", action_color("bet2")),
                ("BET-L", action_color("bet4")),
            ];
            let mut x = area.x;
            for (label, color) in legend_items {
                let style = Style::default().bg(*color).fg(Color::Black);
                buf.set_string(x, legend_y, format!(" {label} "), style);
                x += label.len() as u16 + 3; // label + padding
            }
        }
    }
}

/// Calculate per-cell character width from the available area width.
fn cell_width(area_w: u16) -> u16 {
    // 3 chars reserved for row label, remainder divided among 13 columns.
    let usable = area_w.saturating_sub(3);
    (usable / 13).max(3)
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
                ("raise0".to_string(), 0.80),
                ("fold".to_string(), 0.10),
                ("call".to_string(), 0.10),
            ],
        };

        // 72o: row 6 (7), col 12 (2) — below diagonal = offsuit
        state.cells[6][12] = CellStrategy {
            actions: vec![
                ("fold".to_string(), 0.90),
                ("call".to_string(), 0.05),
                ("raise0".to_string(), 0.05),
            ],
        };

        state
    }

    #[timed_test(10)]
    fn action_color_mapping() {
        assert_eq!(action_color("fold"), Color::Rgb(180, 60, 60));
        assert_eq!(action_color("check"), Color::Rgb(100, 149, 237));
        assert_eq!(action_color("call"), Color::Rgb(60, 179, 113));
        assert_eq!(action_color("raise0"), Color::Rgb(230, 190, 50));
    }

    #[timed_test(10)]
    fn dominant_action_picks_highest() {
        let cell = CellStrategy {
            actions: vec![
                ("fold".to_string(), 0.2),
                ("call".to_string(), 0.7),
                ("raise0".to_string(), 0.1),
            ],
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
        };
        let previous = CellStrategy {
            actions: vec![
                ("fold".to_string(), 0.51),
                ("call".to_string(), 0.49),
            ],
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
        };
        let previous = CellStrategy {
            actions: vec![
                ("fold".to_string(), 0.40),
                ("call".to_string(), 0.60),
            ],
        };
        assert!(!current.is_converged(&previous, 0.05));
    }

    #[timed_test(10)]
    fn widget_renders_without_panic() {
        let state = mock_grid_state();
        let widget = HandGridWidget { state: &state };
        let backend = TestBackend::new(80, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                frame.render_widget(&widget, frame.area());
            })
            .unwrap();
    }
}
