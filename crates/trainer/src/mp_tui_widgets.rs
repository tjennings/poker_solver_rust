//! Compact half-block 13x13 grid widget for N-player TUI.
//!
//! Renders a 13x13 hand matrix using Unicode half-block characters
//! for sub-character color resolution. Designed to fit 6 grids on one screen.

use ratatui::prelude::*;
use ratatui::widgets::Widget;

use crate::blueprint_tui_widgets::{CellStrategy, HandGridState, action_color_ranked};

/// Width of one cell in characters.
pub const COMPACT_CELL_W: u16 = 4;

/// Total width of one compact grid (rank label column + 13 cells).
pub const fn compact_grid_width() -> u16 {
    2 + 13 * COMPACT_CELL_W // 2 for rank label + pad
}

/// Total height of one compact grid (title + header + 13 data rows).
pub const fn compact_grid_height() -> u16 {
    15 // 1 title + 1 column header + 13 data rows
}

/// How many grids fit side by side.
pub fn compute_grids_per_row(terminal_width: u16, grid_width: u16) -> u16 {
    (terminal_width / grid_width).max(1)
}

const RANKS: [char; 13] = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];
const HALF_BLOCK: &str = "\u{2590}";

// ---- Aggression ranking ----

/// Returns an aggression score for sorting (higher = more aggressive).
fn aggression_score(name: &str) -> (u8, i64) {
    let lower = name.to_ascii_lowercase();
    if lower == "all-in" || lower == "allin" {
        return (0, 0);
    }
    if lower.starts_with("raise") {
        let size = parse_action_size(&lower, "raise");
        return (1, (-size * 1000.0) as i64);
    }
    if lower.starts_with("bet") {
        let size = parse_action_size(&lower, "bet");
        return (2, (-size * 1000.0) as i64);
    }
    if lower == "call" {
        return (3, 0);
    }
    if lower == "check" {
        return (4, 0);
    }
    if lower == "fold" {
        return (5, 0);
    }
    (6, 0)
}

fn parse_action_size(lower: &str, prefix: &str) -> f64 {
    lower
        .trim_start_matches(prefix)
        .trim()
        .trim_end_matches('%')
        .parse()
        .unwrap_or(0.0)
}

/// Rank actions by aggression (most aggressive first).
pub fn rank_actions(actions: &[(String, f32)]) -> Vec<(&str, f32)> {
    let mut sorted: Vec<(&str, f32)> = actions
        .iter()
        .map(|(name, freq)| (name.as_str(), *freq))
        .collect();
    sorted.sort_by(|a, b| aggression_score(a.0).cmp(&aggression_score(b.0)));
    sorted
}

// ---- Half-block color generation ----

/// Convert a cell's action distribution into `slots` color values,
/// proportional to action frequencies.
pub fn cell_to_half_blocks(cell: &CellStrategy, slots: usize) -> Vec<Color> {
    if cell.actions.is_empty() {
        return vec![Color::Rgb(40, 40, 40); slots];
    }
    let sorted = rank_actions(&cell.actions);
    let total_bets = sorted.iter().filter(|(n, _)| is_bet_raise(n)).count();
    let mut bet_rank = 0;
    let mut colors = Vec::with_capacity(slots);
    let mut remaining = slots;
    for (i, &(name, freq)) in sorted.iter().enumerate() {
        let count = if i == sorted.len() - 1 {
            remaining
        } else {
            ((freq * slots as f32).round() as usize).min(remaining)
        };
        let color = if is_bet_raise(name) {
            let c = action_color_ranked(name, bet_rank, total_bets);
            bet_rank += 1;
            c
        } else {
            action_color_ranked(name, 0, 1)
        };
        for _ in 0..count {
            colors.push(color);
        }
        remaining -= count;
    }
    colors
}

fn is_bet_raise(name: &str) -> bool {
    let l = name.to_ascii_lowercase();
    l.starts_with("bet") || l.starts_with("raise")
}

// ---- Widget ----

/// Compact 13x13 grid widget using half-block rendering.
pub struct CompactGridWidget<'a> {
    pub state: &'a HandGridState,
}

impl Widget for &CompactGridWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.height < 3 || area.width < compact_grid_width() {
            return;
        }
        render_title(buf, area, &self.state.scenario_name);
        render_header(buf, area);
        for r in 0..13u16 {
            let y = area.y + 2 + r;
            if y >= area.y + area.height {
                break;
            }
            render_row(buf, area.x, y, r as usize, &self.state.cells[r as usize]);
        }
    }
}

fn render_title(buf: &mut Buffer, area: Rect, name: &str) {
    let style = Style::default().add_modifier(Modifier::BOLD);
    let max_len = area.width as usize;
    let display = if name.len() > max_len { &name[..max_len] } else { name };
    buf.set_string(area.x, area.y, display, style);
}

fn render_header(buf: &mut Buffer, area: Rect) {
    let style = Style::default().fg(Color::DarkGray);
    let y = area.y + 1;
    // 2-char offset for row label column
    for (i, &rank) in RANKS.iter().enumerate() {
        let x = area.x + 2 + (i as u16) * COMPACT_CELL_W;
        if x >= area.x + area.width {
            break;
        }
        buf.set_string(x, y, &format!(" {rank}  ")[..COMPACT_CELL_W as usize], style);
    }
}

fn render_row(buf: &mut Buffer, x0: u16, y: u16, row_idx: usize, cells: &[CellStrategy; 13]) {
    let label_style = Style::default().fg(Color::DarkGray);
    buf.set_string(x0, y, &RANKS[row_idx].to_string(), label_style);
    for c in 0..13 {
        let cx = x0 + 2 + (c as u16) * COMPACT_CELL_W;
        render_cell(buf, cx, y, &cells[c]);
    }
}

fn render_cell(buf: &mut Buffer, x: u16, y: u16, cell: &CellStrategy) {
    let colors = cell_to_half_blocks(cell, (COMPACT_CELL_W as usize) * 2);
    for i in 0..COMPACT_CELL_W as usize {
        let left = colors[i * 2];
        let right = colors[i * 2 + 1];
        let style = Style::default().bg(left).fg(right);
        buf.set_string(x + i as u16, y, HALF_BLOCK, style);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_tui_widgets::CellStrategy;
    use test_macros::timed_test;

    #[timed_test]
    fn compact_grid_dimensions() {
        assert_eq!(COMPACT_CELL_W, 4);
        assert_eq!(compact_grid_width(), 54); // 2 + 13*4
        assert_eq!(compact_grid_height(), 15);
    }

    #[timed_test]
    fn grids_per_row_180_cols() {
        assert_eq!(compute_grids_per_row(180, compact_grid_width()), 3);
    }

    #[timed_test]
    fn grids_per_row_120_cols() {
        assert_eq!(compute_grids_per_row(120, compact_grid_width()), 2);
    }

    #[timed_test]
    fn grids_per_row_narrow() {
        assert_eq!(compute_grids_per_row(30, compact_grid_width()), 1);
    }

    #[timed_test]
    fn half_blocks_empty_cell() {
        let cell = CellStrategy::default();
        let blocks = cell_to_half_blocks(&cell, 8);
        assert_eq!(blocks.len(), 8);
        assert!(blocks.iter().all(|&c| c == Color::Rgb(40, 40, 40)));
    }

    #[timed_test]
    fn half_blocks_single_action() {
        let cell = CellStrategy {
            actions: vec![("fold".into(), 1.0)],
            ev: None,
        };
        let blocks = cell_to_half_blocks(&cell, 8);
        assert_eq!(blocks.len(), 8);
        assert!(blocks.iter().all(|&c| c == Color::Rgb(140, 140, 140)));
    }

    #[timed_test]
    fn half_blocks_two_actions_split() {
        let cell = CellStrategy {
            actions: vec![("fold".into(), 0.5), ("call".into(), 0.5)],
            ev: None,
        };
        let blocks = cell_to_half_blocks(&cell, 8);
        assert_eq!(blocks.len(), 8);
        // After ranking: call is more aggressive than fold, so call first
        // call=green gets 4 slots, fold=gray gets 4 slots
        let call_color = Color::Rgb(60, 179, 113);
        let fold_color = Color::Rgb(140, 140, 140);
        assert_eq!(blocks[..4], vec![call_color; 4]);
        assert_eq!(blocks[4..], vec![fold_color; 4]);
    }

    #[timed_test]
    fn half_blocks_respects_slot_count() {
        let cell = CellStrategy {
            actions: vec![("fold".into(), 1.0)],
            ev: None,
        };
        assert_eq!(cell_to_half_blocks(&cell, 4).len(), 4);
        assert_eq!(cell_to_half_blocks(&cell, 16).len(), 16);
    }

    #[timed_test]
    fn half_blocks_last_action_gets_remainder() {
        // 3 actions with freqs that don't divide evenly into 8 slots
        let cell = CellStrategy {
            actions: vec![
                ("fold".into(), 0.33),
                ("call".into(), 0.34),
                ("bet 50%".into(), 0.33),
            ],
            ev: None,
        };
        let blocks = cell_to_half_blocks(&cell, 8);
        assert_eq!(blocks.len(), 8);
    }

    #[timed_test]
    fn rank_actions_sorts_by_aggression() {
        let actions = vec![
            ("fold".into(), 0.3_f32),
            ("call".into(), 0.3),
            ("bet 67%".into(), 0.4),
        ];
        let ranked = rank_actions(&actions);
        assert!(ranked[0].0.starts_with("bet"));
        assert_eq!(ranked[1].0, "call");
        assert_eq!(ranked[2].0, "fold");
    }

    #[timed_test]
    fn rank_actions_allin_most_aggressive() {
        let actions = vec![
            ("fold".into(), 0.2_f32),
            ("all-in".into(), 0.3),
            ("bet 100%".into(), 0.5),
        ];
        let ranked = rank_actions(&actions);
        assert_eq!(ranked[0].0, "all-in");
        assert!(ranked[1].0.starts_with("bet"));
        assert_eq!(ranked[2].0, "fold");
    }

    #[timed_test]
    fn rank_actions_empty() {
        let actions: Vec<(String, f32)> = vec![];
        let ranked = rank_actions(&actions);
        assert!(ranked.is_empty());
    }

    #[timed_test]
    fn rank_actions_raises_before_bets() {
        let actions = vec![
            ("bet 50%".into(), 0.5_f32),
            ("raise 100%".into(), 0.5),
        ];
        let ranked = rank_actions(&actions);
        // raise > bet in aggression
        assert!(ranked[0].0.starts_with("raise"));
        assert!(ranked[1].0.starts_with("bet"));
    }

    #[timed_test]
    fn rank_actions_check_before_fold() {
        let actions = vec![
            ("fold".into(), 0.5_f32),
            ("check".into(), 0.5),
        ];
        let ranked = rank_actions(&actions);
        assert_eq!(ranked[0].0, "check");
        assert_eq!(ranked[1].0, "fold");
    }

    #[timed_test]
    fn compact_widget_renders_without_panic() {
        let state = mock_compact_state();
        let widget = CompactGridWidget { state: &state };
        let area = Rect::new(0, 0, 60, 20);
        let mut buf = Buffer::empty(area);
        (&widget).render(area, &mut buf);
    }

    #[timed_test]
    fn compact_widget_renders_title() {
        let state = mock_compact_state();
        let widget = CompactGridWidget { state: &state };
        let area = Rect::new(0, 0, 60, 20);
        let mut buf = Buffer::empty(area);
        (&widget).render(area, &mut buf);

        let mut title_row = String::new();
        for x in 0..area.width {
            title_row.push(
                buf.cell((x, 0)).unwrap().symbol().chars().next().unwrap_or(' '),
            );
        }
        assert!(
            title_row.contains("UTG open"),
            "title row should contain scenario name, got: {title_row}"
        );
    }

    #[timed_test]
    fn compact_widget_renders_column_headers() {
        let state = mock_compact_state();
        let widget = CompactGridWidget { state: &state };
        let area = Rect::new(0, 0, 60, 20);
        let mut buf = Buffer::empty(area);
        (&widget).render(area, &mut buf);

        let mut header_row = String::new();
        for x in 0..area.width {
            header_row.push(
                buf.cell((x, 1)).unwrap().symbol().chars().next().unwrap_or(' '),
            );
        }
        assert!(
            header_row.contains('A') && header_row.contains('K') && header_row.contains('2'),
            "header row should contain rank labels, got: {header_row}"
        );
    }

    #[timed_test]
    fn compact_widget_renders_row_labels() {
        let state = mock_compact_state();
        let widget = CompactGridWidget { state: &state };
        let area = Rect::new(0, 0, 60, 20);
        let mut buf = Buffer::empty(area);
        (&widget).render(area, &mut buf);

        // Row labels start at y=2 (after title+header), x=0
        let first_label = buf.cell((0, 2)).unwrap().symbol().chars().next().unwrap_or(' ');
        assert_eq!(first_label, 'A', "first row label should be 'A'");
        let last_label = buf.cell((0, 14)).unwrap().symbol().chars().next().unwrap_or(' ');
        assert_eq!(last_label, '2', "last row label should be '2'");
    }

    #[timed_test]
    fn compact_widget_too_small_does_not_panic() {
        let state = mock_compact_state();
        let widget = CompactGridWidget { state: &state };
        let area = Rect::new(0, 0, 5, 3);
        let mut buf = Buffer::empty(area);
        (&widget).render(area, &mut buf);
    }

    #[timed_test]
    fn compact_widget_half_blocks_in_cells() {
        let state = mock_compact_state();
        let widget = CompactGridWidget { state: &state };
        let area = Rect::new(0, 0, 60, 20);
        let mut buf = Buffer::empty(area);
        (&widget).render(area, &mut buf);

        // Cell (0,0) = AA has actions, starts at x=2, y=2
        // Should contain half-block characters
        let cell_x = 2;
        let cell_y = 2;
        let mut found_half_block = false;
        for dx in 0..COMPACT_CELL_W {
            let sym = buf.cell((cell_x + dx, cell_y)).unwrap().symbol();
            if sym == HALF_BLOCK {
                found_half_block = true;
            }
        }
        assert!(found_half_block, "cell with actions should contain half-block chars");
    }

    fn mock_compact_state() -> HandGridState {
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
            error_message: None,
        };
        state.cells[0][0] = CellStrategy {
            actions: vec![
                ("raise 1.0".to_string(), 0.80),
                ("fold".to_string(), 0.10),
                ("call".to_string(), 0.10),
            ],
            ev: Some(5.1),
        };
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
}
