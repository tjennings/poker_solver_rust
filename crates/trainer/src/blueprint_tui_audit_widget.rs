//! Ratatui widget for the regret audit panel.

use ratatui::prelude::*;
use ratatui::widgets::Tabs;

use crate::blueprint_tui_audit::{AuditSnapshot, Trend};
use crate::blueprint_tui_config::PlayerLabel;
use poker_solver_core::blueprint_v2::Street;

/// Static metadata for one audit entry (set at startup, never changes).
#[derive(Debug, Clone)]
pub struct AuditMeta {
    pub name: String,
    pub hand: String,
    pub player: PlayerLabel,
    pub bucket_trail: Vec<(Street, u16)>,
    pub action_labels: Vec<String>,
    pub error: Option<String>,
}

/// Full state for rendering the audit panel.
pub struct AuditPanelState {
    pub metas: Vec<AuditMeta>,
    pub snapshots: Vec<AuditSnapshot>,
    pub active_tab: usize,
    pub iteration: u64,
}

impl AuditPanelState {
    /// Advance to the next tab, wrapping around to the first.
    pub fn next_tab(&mut self) {
        if !self.metas.is_empty() {
            self.active_tab = (self.active_tab + 1) % self.metas.len();
        }
    }

    /// Move to the previous tab, wrapping around to the last.
    pub fn prev_tab(&mut self) {
        if !self.metas.is_empty() {
            self.active_tab = (self.active_tab + self.metas.len() - 1) % self.metas.len();
        }
    }
}

pub struct AuditPanelWidget<'a> {
    pub state: &'a AuditPanelState,
}

/// Format an iteration count with K/M suffixes.
fn format_iteration(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Two-letter street abbreviation.
fn street_abbrev(s: Street) -> &'static str {
    match s {
        Street::Preflop => "pf",
        Street::Flop => "fl",
        Street::Turn => "tn",
        Street::River => "rv",
    }
}

impl Widget for &AuditPanelWidget<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if self.state.metas.is_empty() {
            return;
        }

        let mut y = area.y;
        let x = area.x;
        let w = area.width;

        // 1. Tab bar
        let titles: Vec<Line<'_>> = self
            .state
            .metas
            .iter()
            .map(|m| Line::from(m.name.as_str()))
            .collect();
        let tabs = Tabs::new(titles)
            .select(self.state.active_tab)
            .highlight_style(Style::default().fg(Color::Cyan).bold())
            .divider("|");
        let tab_area = Rect::new(x, y, w, 1);
        tabs.render(tab_area, buf);
        y += 1;

        let idx = self.state.active_tab;
        let meta = &self.state.metas[idx];
        let snap = &self.state.snapshots[idx];

        // 2. Error state
        if let Some(ref err) = meta.error {
            if y < area.y + area.height {
                let style = Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD);
                let text = err.chars().take(w as usize).collect::<String>();
                buf.set_string(x, y, &text, style);
            }
            return;
        }

        // 3. Header
        if y < area.y + area.height {
            let player_str = match meta.player {
                PlayerLabel::Sb => "SB",
                PlayerLabel::Bb => "BB",
            };
            let header = format!(
                "Hand: {}  Player: {}  Iter: {}",
                meta.hand,
                player_str,
                format_iteration(self.state.iteration),
            );
            buf.set_string(x, y, &header, Style::default().fg(Color::White));
            y += 1;
        }

        // 4. Bucket trail
        if y < area.y + area.height {
            let trail: Vec<String> = meta
                .bucket_trail
                .iter()
                .map(|(s, b)| format!("{}:{}", street_abbrev(*s), b))
                .collect();
            let trail_text = format!("Buckets: {}", trail.join(" \u{2192} "));
            buf.set_string(x, y, &trail_text, Style::default().fg(Color::DarkGray));
            y += 1;
        }

        // Blank line
        y += 1;

        // 5. Table header
        if y < area.y + area.height {
            let header = format!(
                "{:<12} {:>8} {:>8}  {}",
                "Action", "Regret", "\u{0394}/tick", "Trend"
            );
            let style = Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD);
            buf.set_string(x, y, &header, style);
            y += 1;
        }

        // 6. Table rows
        for (i, label) in meta.action_labels.iter().enumerate() {
            if y >= area.y + area.height {
                break;
            }

            let truncated: String = label.chars().take(11).collect();
            let regret = snap.regrets.get(i).copied().unwrap_or(0.0);
            let delta = snap.deltas.get(i).copied().unwrap_or(0.0);
            let trend = snap.trends.get(i).copied().unwrap_or(Trend::Flat);

            // Action label (white)
            buf.set_string(x, y, format!("{truncated:<12}"), Style::default().fg(Color::White));

            // Regret value (colored)
            let regret_color = if regret > 0.0 {
                Color::Green
            } else if regret < 0.0 {
                Color::Red
            } else {
                Color::Gray
            };
            buf.set_string(
                x + 12,
                y,
                format!("{regret:>8.1}"),
                Style::default().fg(regret_color),
            );

            // Delta (gray)
            buf.set_string(
                x + 21,
                y,
                format!("{delta:>+8.1}"),
                Style::default().fg(Color::Gray),
            );

            // Trend arrow
            let (arrow, arrow_color) = match trend {
                Trend::Up => ("\u{2191}", Color::Green),
                Trend::Down => ("\u{2193}", Color::Red),
                Trend::Flat => ("\u{2192}", Color::Gray),
            };
            buf.set_string(x + 31, y, arrow, Style::default().fg(arrow_color));

            y += 1;
        }

        // Blank line
        y += 1;

        // 7. Strategy line
        if y < area.y + area.height {
            let parts: Vec<String> = meta
                .action_labels
                .iter()
                .zip(snap.strategy.iter())
                .map(|(label, &prob)| {
                    let abbrev = label.chars().next().unwrap_or('?');
                    format!("{}:{}%", abbrev, (prob * 100.0).round() as u64)
                })
                .collect();
            let strategy_text = format!("Strategy: {}", parts.join(" "));
            buf.set_string(x, y, &strategy_text, Style::default().fg(Color::Cyan));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use test_macros::timed_test;

    fn mock_panel_state() -> AuditPanelState {
        AuditPanelState {
            metas: vec![AuditMeta {
                name: "AKo SB open".to_string(),
                hand: "AKo".to_string(),
                player: PlayerLabel::Sb,
                bucket_trail: vec![(Street::Preflop, 3)],
                action_labels: vec!["fold".into(), "call".into(), "raise 5bb".into()],
                error: None,
            }],
            snapshots: vec![AuditSnapshot {
                regrets: vec![-1.2, 0.3, 0.9],
                deltas: vec![-0.03, 0.02, 0.01],
                trends: vec![Trend::Down, Trend::Up, Trend::Up],
                strategy: vec![0.0, 0.25, 0.75],
            }],
            active_tab: 0,
            iteration: 1_200_000,
        }
    }

    #[timed_test(10)]
    fn audit_panel_renders_without_panic() {
        let state = mock_panel_state();
        let widget = AuditPanelWidget { state: &state };
        let backend = TestBackend::new(50, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                frame.render_widget(&widget, frame.area());
            })
            .unwrap();
    }

    #[timed_test(10)]
    fn audit_panel_tab_switching() {
        let mut state = mock_panel_state();
        state.metas.push(AuditMeta {
            name: "TT 3bet".into(),
            hand: "TT".into(),
            player: PlayerLabel::Sb,
            bucket_trail: vec![(Street::Preflop, 5)],
            action_labels: vec!["fold".into(), "call".into()],
            error: None,
        });
        state.snapshots.push(AuditSnapshot {
            regrets: vec![0.0, 0.0],
            deltas: vec![0.0, 0.0],
            trends: vec![Trend::Flat, Trend::Flat],
            strategy: vec![0.5, 0.5],
        });
        assert_eq!(state.active_tab, 0);
        state.next_tab();
        assert_eq!(state.active_tab, 1);
        state.next_tab();
        assert_eq!(state.active_tab, 0);
        state.prev_tab();
        assert_eq!(state.active_tab, 1);
    }

    #[timed_test(10)]
    fn audit_panel_error_renders() {
        let state = AuditPanelState {
            metas: vec![AuditMeta {
                name: "bad".into(),
                hand: "AKo".into(),
                player: PlayerLabel::Sb,
                bucket_trail: vec![],
                action_labels: vec![],
                error: Some("Spot failed to resolve".into()),
            }],
            snapshots: vec![AuditSnapshot {
                regrets: vec![],
                deltas: vec![],
                trends: vec![],
                strategy: vec![],
            }],
            active_tab: 0,
            iteration: 0,
        };
        let widget = AuditPanelWidget { state: &state };
        let backend = TestBackend::new(50, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                frame.render_widget(&widget, frame.area());
            })
            .unwrap();
    }

    #[timed_test(10)]
    fn empty_audits_renders_nothing() {
        let state = AuditPanelState {
            metas: vec![],
            snapshots: vec![],
            active_tab: 0,
            iteration: 0,
        };
        let widget = AuditPanelWidget { state: &state };
        let backend = TestBackend::new(50, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                frame.render_widget(&widget, frame.area());
            })
            .unwrap();
    }
}
