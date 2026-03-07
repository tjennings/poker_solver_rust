//! Main TUI application for the Blueprint V2 training dashboard.
//!
//! Provides a split-panel layout: left side shows training telemetry
//! (iterations, throughput sparkline, strategy delta sparkline, leaf
//! movement sparkline), right side shows tabbed 13x13 hand grids for
//! monitored scenarios.

use std::io;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::backend::CrosstermBackend;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Sparkline, Tabs};

use crate::blueprint_tui_config::TelemetryConfig;
use crate::blueprint_tui_metrics::BlueprintTuiMetrics;
use crate::blueprint_tui_widgets::{HandGridState, HandGridWidget};

/// Maximum sparkline history length.
const SPARKLINE_HISTORY: usize = 120;

/// A scenario resolved to a specific game-tree node with its hand grid.
pub struct ResolvedScenario {
    pub name: String,
    pub node_idx: u32,
    pub grid: HandGridState,
}

/// Application state owned by the TUI render thread.
pub struct BlueprintTuiApp {
    metrics: Arc<BlueprintTuiMetrics>,
    scenarios: Vec<ResolvedScenario>,
    active_tab: usize,
    telemetry_config: TelemetryConfig,
    // Throughput sparkline data
    iter_per_sec_history: Vec<u64>,
    prev_iterations: u64,
    peak_iter_per_sec: u64,
    refresh_rate_hz: f64,
    // Strategy delta sparkline data
    delta_history: Vec<u64>,
    // Leaf movement sparkline data
    leaf_movement_history: Vec<u64>,
    // Max negative regret sparkline data (stored as abs value × 1000 for sparkline)
    min_regret_history: Vec<u64>,
    // Current prune fraction (0.0–1.0)
    prune_fraction: f64,
    // Random scenario carousel tracking
    has_random_tab: bool,
}

impl BlueprintTuiApp {
    pub fn new(
        metrics: Arc<BlueprintTuiMetrics>,
        scenarios: Vec<ResolvedScenario>,
        telemetry_config: TelemetryConfig,
        refresh_rate_hz: f64,
    ) -> Self {
        Self {
            metrics,
            scenarios,
            active_tab: 0,
            telemetry_config,
            iter_per_sec_history: Vec::with_capacity(SPARKLINE_HISTORY),
            prev_iterations: 0,
            peak_iter_per_sec: 0,
            refresh_rate_hz,
            delta_history: Vec::with_capacity(SPARKLINE_HISTORY),
            leaf_movement_history: Vec::with_capacity(SPARKLINE_HISTORY),
            min_regret_history: Vec::with_capacity(SPARKLINE_HISTORY),
            prune_fraction: 0.0,
            has_random_tab: false,
        }
    }

    /// Sample iteration counter, compute delta, update sparkline histories.
    pub fn tick(&mut self) {
        let sparkline_max = self.telemetry_config.sparkline_window.min(SPARKLINE_HISTORY);

        // Throughput
        let current = self.metrics.iterations.load(Ordering::Relaxed);
        let delta = current.saturating_sub(self.prev_iterations);
        let ips = if self.refresh_rate_hz > 0.0 {
            (delta as f64 * self.refresh_rate_hz) as u64
        } else {
            delta
        };
        push_bounded(&mut self.iter_per_sec_history, ips, sparkline_max);
        if ips > self.peak_iter_per_sec {
            self.peak_iter_per_sec = ips;
        }
        self.prev_iterations = current;

        // Strategy delta sparkline: read all new values
        {
            let mut hist = self
                .metrics
                .strategy_delta_history
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            for &v in hist.iter() {
                push_bounded(
                    &mut self.delta_history,
                    (v * 10000.0) as u64,
                    sparkline_max,
                );
            }
            hist.clear();
        }

        // Leaf movement sparkline: read all new values
        {
            let mut hist = self
                .metrics
                .leaf_movement_history
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            for &v in hist.iter() {
                push_bounded(
                    &mut self.leaf_movement_history,
                    (v * 100.0) as u64,
                    sparkline_max,
                );
            }
            hist.clear();
        }

        // Min regret sparkline: read all new values
        {
            let mut hist = self
                .metrics
                .min_regret_history
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            for &v in hist.iter() {
                // Store absolute value scaled up for sparkline (regret is negative)
                push_bounded(
                    &mut self.min_regret_history,
                    (v.abs() * 1000.0) as u64,
                    sparkline_max,
                );
            }
            hist.clear();
        }

        // Prune fraction
        {
            let pf = self.metrics.prune_fraction.lock().unwrap_or_else(|e| e.into_inner());
            self.prune_fraction = *pf;
        }

        // Strategy grid refresh from trainer
        {
            let mut grids = self
                .metrics
                .strategy_grids
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            for (i, grid_opt) in grids.iter_mut().enumerate() {
                if let Some(new_grid) = grid_opt.take()
                    && i < self.scenarios.len()
                {
                    let old = std::mem::replace(&mut self.scenarios[i].grid.cells, new_grid);
                    self.scenarios[i].grid.prev_cells = Some(old);
                    self.scenarios[i].grid.iteration_at_snapshot =
                        self.metrics.iterations.load(Ordering::Relaxed);
                }
            }
        }

        // Random scenario carousel
        {
            let mut rs = self
                .metrics
                .random_scenario
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(state) = rs.take() {
                let scenario_name = state.name.clone();
                let grid_state = HandGridState {
                    cells: state.grid,
                    prev_cells: None,
                    scenario_name: scenario_name.clone(),
                    action_path: vec![],
                    board_display: if state.board_display.is_empty() {
                        None
                    } else {
                        Some(state.board_display)
                    },
                    cluster_id: None,
                    street_label: state.street_label,
                    iteration_at_snapshot: self.metrics.iterations.load(Ordering::Relaxed),
                };

                let scenario = ResolvedScenario {
                    name: scenario_name,
                    node_idx: state.node_idx,
                    grid: grid_state,
                };

                if self.has_random_tab {
                    if let Some(last) = self.scenarios.last_mut() {
                        *last = scenario;
                    }
                } else {
                    self.scenarios.push(scenario);
                    self.has_random_tab = true;
                }
            }
        }
    }

    pub fn next_tab(&mut self) {
        if !self.scenarios.is_empty() {
            self.active_tab = (self.active_tab + 1) % self.scenarios.len();
        }
    }

    pub fn prev_tab(&mut self) {
        if !self.scenarios.is_empty() {
            self.active_tab = (self.active_tab + self.scenarios.len() - 1) % self.scenarios.len();
        }
    }

    /// Render the full dashboard.
    pub fn render(&self, frame: &mut Frame) {
        let area = frame.area();

        // Top-level horizontal split: left 45%, right 55%.
        let h_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
            .split(area);

        self.render_left_panel(frame, h_chunks[0]);
        self.render_right_panel(frame, h_chunks[1]);
    }

    fn render_left_panel(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Iterations counter
                Constraint::Length(1), // Progress gauge
                Constraint::Length(1), // Runtime + ETA
                Constraint::Length(3), // Throughput sparkline
                Constraint::Length(3), // Strategy delta sparkline
                Constraint::Length(3), // Leaf movement sparkline
                Constraint::Length(3), // Max negative regret sparkline
                Constraint::Length(1), // Actions pruned bar
                Constraint::Min(0),   // Spacer
                Constraint::Length(1), // Hotkeys footer
            ])
            .split(area);

        self.render_iterations(frame, chunks[0]);
        self.render_progress_gauge(frame, chunks[1]);
        self.render_runtime(frame, chunks[2]);
        self.render_sparkline(frame, chunks[3]);
        self.render_strategy_delta(frame, chunks[4]);
        self.render_leaf_movement(frame, chunks[5]);
        self.render_min_regret(frame, chunks[6]);
        self.render_prune_bar(frame, chunks[7]);
        self.render_hotkeys(frame, chunks[9]);
    }

    fn render_iterations(&self, frame: &mut Frame, area: Rect) {
        let current = self.metrics.iterations.load(Ordering::Relaxed);
        let text = match self.metrics.target_iterations {
            Some(target) => {
                format!(
                    "Iterations: {} / {}",
                    format_count(current),
                    format_count(target),
                )
            }
            None => format!("Iterations: {}", format_count(current)),
        };
        let p = Paragraph::new(text).style(Style::default().fg(Color::White).bold());
        frame.render_widget(p, area);
    }

    fn render_progress_gauge(&self, frame: &mut Frame, area: Rect) {
        let current = self.metrics.iterations.load(Ordering::Relaxed);
        let ratio = self
            .metrics
            .target_iterations
            .map(|t| if t > 0 { current as f64 / t as f64 } else { 0.0 })
            .unwrap_or(0.0);
        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(Color::Cyan))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(format!("{:.1}%", ratio * 100.0));
        frame.render_widget(gauge, area);
    }

    fn render_runtime(&self, frame: &mut Frame, area: Rect) {
        let elapsed = self.metrics.elapsed_secs();
        let current = self.metrics.iterations.load(Ordering::Relaxed);
        let paused = self.metrics.is_paused();

        let eta_str = self
            .metrics
            .target_iterations
            .map(|t| format_eta(current, t, elapsed))
            .unwrap_or_else(|| "--".to_string());

        let pause_marker = if paused { "  [PAUSED]" } else { "" };
        let text = format!(
            "Runtime: {}  ETA: {}{pause_marker}",
            format_duration(elapsed),
            eta_str,
        );
        let style = if paused {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default().fg(Color::White)
        };
        frame.render_widget(Paragraph::new(text).style(style), area);
    }

    fn render_sparkline(&self, frame: &mut Frame, area: Rect) {
        let current = self.iter_per_sec_history.last().copied().unwrap_or(0);
        let title = format!(
            "Throughput: {} it/s  (peak {})",
            format_count(current),
            format_count(self.peak_iter_per_sec),
        );
        let sparkline = Sparkline::default()
            .block(Block::default().title(title).borders(Borders::NONE))
            .data(&self.iter_per_sec_history)
            .style(Style::default().fg(Color::Cyan));
        frame.render_widget(sparkline, area);
    }

    fn render_strategy_delta(&self, frame: &mut Frame, area: Rect) {
        let latest = self
            .delta_history
            .last()
            .map(|&v| v as f64 / 10000.0)
            .unwrap_or(0.0);
        let title = format!("Strategy delta: {latest:.6}");
        let sparkline = Sparkline::default()
            .block(Block::default().title(title).borders(Borders::NONE))
            .data(&self.delta_history)
            .style(Style::default().fg(Color::Yellow));
        frame.render_widget(sparkline, area);
    }

    fn render_leaf_movement(&self, frame: &mut Frame, area: Rect) {
        let latest = self
            .leaf_movement_history
            .last()
            .map(|&v| v as f64)
            .unwrap_or(0.0);
        let title = format!("Leaves moving (>1%): {latest:.1}%");
        let sparkline = Sparkline::default()
            .block(Block::default().title(title).borders(Borders::NONE))
            .data(&self.leaf_movement_history)
            .style(Style::default().fg(Color::Green));
        frame.render_widget(sparkline, area);
    }

    fn render_min_regret(&self, frame: &mut Frame, area: Rect) {
        // Display value: stored as abs(regret) × 1000 for sparkline resolution,
        // convert back to the original (already ÷1000 from storage scaling).
        let latest = self
            .min_regret_history
            .last()
            .map(|&v| v as f64 / 1000.0)
            .unwrap_or(0.0);
        let title = format!("Max neg regret: -{latest:.1}");
        let sparkline = Sparkline::default()
            .block(Block::default().title(title).borders(Borders::NONE))
            .data(&self.min_regret_history)
            .style(Style::default().fg(Color::Red));
        frame.render_widget(sparkline, area);
    }

    fn render_prune_bar(&self, frame: &mut Frame, area: Rect) {
        let pct = self.prune_fraction * 100.0;
        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(Color::Magenta))
            .ratio(self.prune_fraction.clamp(0.0, 1.0))
            .label(format!("Actions pruned: {pct:.1}%"));
        frame.render_widget(gauge, area);
    }

    fn render_hotkeys(&self, frame: &mut Frame, area: Rect) {
        let text = "[p]ause [s]napshot [?]help [q]uit";
        frame.render_widget(
            Paragraph::new(text).style(Style::default().fg(Color::DarkGray)),
            area,
        );
    }

    fn render_right_panel(&self, frame: &mut Frame, area: Rect) {
        if self.scenarios.is_empty() {
            let p = Paragraph::new("No scenarios configured")
                .style(Style::default().fg(Color::DarkGray));
            frame.render_widget(p, area);
            return;
        }

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(0)])
            .split(area);

        // Tabs
        let titles: Vec<Line<'_>> = self
            .scenarios
            .iter()
            .map(|s| Line::from(s.name.as_str()))
            .collect();
        let tabs = Tabs::new(titles)
            .select(self.active_tab)
            .highlight_style(Style::default().fg(Color::Cyan).bold())
            .divider("|");
        frame.render_widget(tabs, chunks[0]);

        // Active scenario grid
        let sc = &self.scenarios[self.active_tab];
        let widget = HandGridWidget { state: &sc.grid };
        frame.render_widget(&widget, chunks[1]);
    }
}

/// Spawn the TUI on a dedicated thread. Returns a join handle.
pub fn run_blueprint_tui(
    metrics: Arc<BlueprintTuiMetrics>,
    scenarios: Vec<ResolvedScenario>,
    telemetry_config: TelemetryConfig,
    refresh_interval: Duration,
    refresh_rate_hz: f64,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        if let Err(e) = run_tui_inner(
            metrics,
            scenarios,
            telemetry_config,
            refresh_interval,
            refresh_rate_hz,
        ) {
            eprintln!("Blueprint TUI error: {e}");
        }
    })
}

fn run_tui_inner(
    metrics: Arc<BlueprintTuiMetrics>,
    scenarios: Vec<ResolvedScenario>,
    telemetry_config: TelemetryConfig,
    refresh_interval: Duration,
    refresh_rate_hz: f64,
) -> io::Result<()> {
    enable_raw_mode()?;
    let mut stderr = io::stderr();
    execute!(stderr, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stderr);
    let mut terminal = ratatui::Terminal::new(backend)?;

    let mut app = BlueprintTuiApp::new(
        Arc::clone(&metrics),
        scenarios,
        telemetry_config,
        refresh_rate_hz,
    );

    loop {
        app.tick();
        terminal.draw(|frame| app.render(frame))?;

        if event::poll(refresh_interval)?
            && let Event::Key(key) = event::read()?
        {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => {
                    metrics.quit_requested.store(true, Ordering::Relaxed);
                    break;
                }
                KeyCode::Right | KeyCode::Tab => app.next_tab(),
                KeyCode::Left | KeyCode::BackTab => app.prev_tab(),
                KeyCode::Char('p') => metrics.toggle_pause(),
                KeyCode::Char('s') => metrics.request_snapshot(),
                _ => {}
            }
        }

        if metrics.quit_requested.load(Ordering::Relaxed) {
            // Final render then exit.
            app.tick();
            terminal.draw(|frame| app.render(frame))?;
            std::thread::sleep(Duration::from_secs(1));
            break;
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

// -- Helpers --

fn push_bounded(buf: &mut Vec<u64>, value: u64, max_len: usize) {
    if buf.len() >= max_len {
        buf.remove(0);
    }
    buf.push(value);
}

fn format_duration(secs: f64) -> String {
    let total = secs as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}h {m:02}m")
    } else if m > 0 {
        format!("{m}m {s:02}s")
    } else {
        format!("{s}s")
    }
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn format_eta(current: u64, target: u64, elapsed_secs: f64) -> String {
    if target == 0 || current == 0 || elapsed_secs < 1.0 {
        return "calculating...".to_string();
    }
    let ratio = current as f64 / target as f64;
    if ratio >= 1.0 {
        return "done".to_string();
    }
    let total_est = elapsed_secs / ratio;
    let remaining = (total_est - elapsed_secs).max(0.0);
    format_duration(remaining)
}

// -- Tests --

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_tui_config::TelemetryConfig;
    use crate::blueprint_tui_widgets::{CellStrategy, HandGridState};
    use test_macros::timed_test;

    fn make_metrics() -> Arc<BlueprintTuiMetrics> {
        Arc::new(BlueprintTuiMetrics::new(Some(1_000_000)))
    }

    fn make_scenario(name: &str) -> ResolvedScenario {
        let cells: [[CellStrategy; 13]; 13] =
            std::array::from_fn(|_| std::array::from_fn(|_| CellStrategy::default()));
        ResolvedScenario {
            name: name.to_string(),
            node_idx: 0,
            grid: HandGridState {
                cells,
                prev_cells: None,
                scenario_name: name.to_string(),
                action_path: vec![],
                board_display: None,
                cluster_id: None,
                street_label: "Preflop".to_string(),
                iteration_at_snapshot: 0,
            },
        }
    }

    #[timed_test(10)]
    fn app_renders_without_panic() {
        let metrics = make_metrics();
        let app = BlueprintTuiApp::new(
            metrics,
            vec![],
            TelemetryConfig::default(),
            4.0,
        );
        let backend = ratatui::backend::TestBackend::new(160, 50);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
    }

    #[timed_test(10)]
    fn tab_switching() {
        let metrics = make_metrics();
        let scenarios = vec![make_scenario("A"), make_scenario("B")];
        let mut app = BlueprintTuiApp::new(
            metrics,
            scenarios,
            TelemetryConfig::default(),
            4.0,
        );

        assert_eq!(app.active_tab, 0);
        app.next_tab();
        assert_eq!(app.active_tab, 1);
        app.next_tab();
        assert_eq!(app.active_tab, 0); // wraps
        app.prev_tab();
        assert_eq!(app.active_tab, 1); // wraps backward
    }

    #[timed_test(10)]
    fn format_helpers() {
        // format_duration
        assert_eq!(format_duration(5.0), "5s");
        assert_eq!(format_duration(90.0), "1m 30s");
        assert_eq!(format_duration(3661.0), "1h 01m");

        // format_count
        assert_eq!(format_count(500), "500");
        assert_eq!(format_count(1_500), "1.5K");
        assert_eq!(format_count(2_500_000), "2.5M");

        // format_eta
        assert_eq!(format_eta(0, 1000, 10.0), "calculating...");
        assert_eq!(format_eta(1000, 1000, 60.0), "done");
        // 500/1000 = 50%, elapsed=60s, total_est=120s, remaining=60s
        assert_eq!(format_eta(500, 1000, 60.0), "1m 00s");
    }
}
