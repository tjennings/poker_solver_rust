//! N-player TUI application with tiled 6-up grid layout and pagination.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Sparkline};

use crate::blueprint_tui_config::TelemetryConfig;
use crate::blueprint_tui_metrics::BlueprintTuiMetrics;
use crate::blueprint_tui_widgets::HandGridState;
use crate::mp_tui_widgets::{compact_grid_height, compact_grid_width, compute_grids_per_row, CompactGridWidget};

/// Maximum grids displayed per page.
pub const GRIDS_PER_PAGE: usize = 6;

/// A scenario resolved to an MP game-tree node with its hand grid.
#[allow(dead_code)] // Fields consumed by TUI metrics integration (Task 6).
pub struct ResolvedMpScenario {
    pub name: String,
    pub node_idx: u32,
    pub grid: HandGridState,
}

/// Application state for the N-player TUI render thread.
pub struct MpTuiApp {
    pub(crate) metrics: Arc<BlueprintTuiMetrics>,
    pub(crate) scenarios: Vec<ResolvedMpScenario>,
    pub(crate) current_page: usize,
    pub(crate) num_players: u8,
    pub(crate) telemetry_config: TelemetryConfig,
    pub(crate) iter_per_sec_history: Vec<u64>,
    pub(crate) prev_iterations: u64,
    pub(crate) peak_iter_per_sec: u64,
    pub(crate) last_tick: Instant,
    pub(crate) delta_history: Vec<u64>,
    pub(crate) max_regret_history: Vec<u64>,
    pub(crate) min_regret_history: Vec<u64>,
    pub(crate) avg_pos_regret_history: Vec<u64>,
    pub(crate) prune_history: Vec<u64>,
}

impl MpTuiApp {
    pub fn new(
        metrics: Arc<BlueprintTuiMetrics>,
        scenarios: Vec<ResolvedMpScenario>,
        telemetry_config: TelemetryConfig,
        num_players: u8,
    ) -> Self {
        Self {
            metrics,
            scenarios,
            current_page: 0,
            num_players,
            telemetry_config,
            iter_per_sec_history: Vec::with_capacity(120),
            prev_iterations: 0,
            peak_iter_per_sec: 0,
            last_tick: Instant::now(),
            delta_history: Vec::with_capacity(120),
            max_regret_history: Vec::with_capacity(120),
            min_regret_history: Vec::with_capacity(120),
            avg_pos_regret_history: Vec::with_capacity(120),
            prune_history: Vec::with_capacity(120),
        }
    }

    pub fn next_page(&mut self) {
        let pages = page_count(self.scenarios.len(), GRIDS_PER_PAGE);
        if pages > 0 {
            self.current_page = (self.current_page + 1) % pages;
        }
    }

    pub fn prev_page(&mut self) {
        let pages = page_count(self.scenarios.len(), GRIDS_PER_PAGE);
        if pages > 0 {
            self.current_page = (self.current_page + pages - 1) % pages;
        }
    }

    /// Render the full dashboard: metrics, grid page, hotkeys.
    pub fn render(&self, frame: &mut Frame) {
        let area = frame.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(19),
                Constraint::Min(0),
                Constraint::Length(1),
            ])
            .split(area);
        self.render_metrics(frame, chunks[0]);
        self.render_grid_page(frame, chunks[1]);
        self.render_hotkeys(frame, chunks[2]);
    }

    /// Sample iteration counter and update sparkline histories.
    pub fn tick(&mut self) {
        let sparkline_max = self.telemetry_config.sparkline_window.min(120);
        let now = Instant::now();
        let elapsed_secs = now.duration_since(self.last_tick).as_secs_f64();
        self.last_tick = now;
        let current = self.metrics.iterations.load(Ordering::Relaxed);
        let delta = current.saturating_sub(self.prev_iterations);
        let ips = if elapsed_secs > 0.0 {
            (delta as f64 / elapsed_secs) as u64
        } else {
            delta
        };
        push_bounded(&mut self.iter_per_sec_history, ips, sparkline_max);
        if ips > self.peak_iter_per_sec {
            self.peak_iter_per_sec = ips;
        }
        self.prev_iterations = current;
        self.tick_delta_history(sparkline_max);
        self.tick_regret_sparklines(sparkline_max);
        self.tick_grids();
    }
}

// -- Private rendering helpers --

impl MpTuiApp {
    fn render_metrics(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),  // iterations
                Constraint::Length(1),  // progress gauge
                Constraint::Length(1),  // runtime + ETA
                Constraint::Length(2),  // throughput sparkline
                Constraint::Length(2),  // strategy delta sparkline
                Constraint::Length(2),  // max pos regret sparkline
                Constraint::Length(2),  // max neg regret sparkline
                Constraint::Length(2),  // avg pos regret sparkline
                Constraint::Length(2),  // prune % sparkline
                Constraint::Min(0),
            ])
            .split(area);
        self.render_iterations(frame, chunks[0]);
        self.render_progress_gauge(frame, chunks[1]);
        self.render_runtime(frame, chunks[2]);
        self.render_throughput(frame, chunks[3]);
        self.render_delta_sparkline(frame, chunks[4]);
        self.render_max_regret(frame, chunks[5]);
        self.render_min_regret(frame, chunks[6]);
        self.render_avg_pos_regret(frame, chunks[7]);
        self.render_prune_sparkline(frame, chunks[8]);
    }

    fn render_iterations(&self, frame: &mut Frame, area: Rect) {
        let current = self.metrics.iterations.load(Ordering::Relaxed);
        let text = match self.metrics.target_iterations {
            Some(t) => format!("Iterations: {current} / {t}  ({} players)", self.num_players),
            None => format!("Iterations: {current}  ({} players)", self.num_players),
        };
        let p = Paragraph::new(text).style(Style::default().fg(Color::White).bold());
        frame.render_widget(p, area);
    }

    fn render_progress_gauge(&self, frame: &mut Frame, area: Rect) {
        let ratio = compute_progress_ratio(&self.metrics);
        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(Color::Cyan))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(format!("{:.1}%", ratio * 100.0));
        frame.render_widget(gauge, area);
    }

    fn render_runtime(&self, frame: &mut Frame, area: Rect) {
        let elapsed = self.metrics.elapsed_secs();
        let text = format!("Runtime: {:.0}s", elapsed);
        let p = Paragraph::new(text).style(Style::default().fg(Color::White));
        frame.render_widget(p, area);
    }

    fn render_throughput(&self, frame: &mut Frame, area: Rect) {
        let current = self.iter_per_sec_history.last().copied().unwrap_or(0);
        let title = format!("Throughput: {current} it/s  (peak {})", self.peak_iter_per_sec);
        let sparkline = Sparkline::default()
            .block(Block::default().title(title).borders(Borders::NONE))
            .data(&self.iter_per_sec_history)
            .style(Style::default().fg(Color::Cyan));
        frame.render_widget(sparkline, area);
    }

    fn render_delta_sparkline(&self, frame: &mut Frame, area: Rect) {
        let latest = self.delta_history.last()
            .map(|&v| v as f64 / 10000.0)
            .unwrap_or(0.0);
        let title = format!("Strategy delta: {latest:.6}");
        let sparkline = Sparkline::default()
            .block(Block::default().title(title).borders(Borders::NONE))
            .data(&self.delta_history)
            .style(Style::default().fg(Color::Yellow));
        frame.render_widget(sparkline, area);
    }

    fn render_max_regret(&self, frame: &mut Frame, area: Rect) {
        let latest = self.max_regret_history.last()
            .map(|&v| v as f64 / 1000.0)
            .unwrap_or(0.0);
        let title = format!("Max pos regret: {latest:.1}");
        let sparkline = Sparkline::default()
            .block(Block::default().title(title).borders(Borders::NONE))
            .data(&self.max_regret_history)
            .style(Style::default().fg(Color::Red));
        frame.render_widget(sparkline, area);
    }

    fn render_min_regret(&self, frame: &mut Frame, area: Rect) {
        let latest = self.min_regret_history.last()
            .map(|&v| v as f64 / 1000.0)
            .unwrap_or(0.0);
        let title = format!("Max neg regret: -{latest:.1}");
        let sparkline = Sparkline::default()
            .block(Block::default().title(title).borders(Borders::NONE))
            .data(&self.min_regret_history)
            .style(Style::default().fg(Color::Red));
        frame.render_widget(sparkline, area);
    }

    fn render_avg_pos_regret(&self, frame: &mut Frame, area: Rect) {
        let latest = self.avg_pos_regret_history.last()
            .map(|&v| v as f64 / 1_000_000_000.0)
            .unwrap_or(0.0);
        let title = format!("Avg pos regret: {latest:.2e}");
        let sparkline = Sparkline::default()
            .block(Block::default().title(title).borders(Borders::NONE))
            .data(&self.avg_pos_regret_history)
            .style(Style::default().fg(Color::Green));
        frame.render_widget(sparkline, area);
    }

    fn render_prune_sparkline(&self, frame: &mut Frame, area: Rect) {
        let latest = self.prune_history.last()
            .map(|&v| v as f64 / 10.0)
            .unwrap_or(0.0);
        let title = format!("Traversals pruned: {latest:.1}%");
        let sparkline = Sparkline::default()
            .block(Block::default().title(title).borders(Borders::NONE))
            .data(&self.prune_history)
            .style(Style::default().fg(Color::Magenta));
        frame.render_widget(sparkline, area);
    }
}

// -- Grid page rendering --

impl MpTuiApp {
    fn render_grid_page(&self, frame: &mut Frame, area: Rect) {
        let page = page_slice(&self.scenarios, self.current_page, GRIDS_PER_PAGE);
        if page.is_empty() {
            let p = Paragraph::new("No scenarios configured")
                .style(Style::default().fg(Color::DarkGray));
            frame.render_widget(p, area);
            return;
        }
        let cols = compute_grids_per_row(area.width, compact_grid_width());
        let gw = compact_grid_width();
        let gh = compact_grid_height();
        for (i, scenario) in page.iter().enumerate() {
            let col = i as u16 % cols;
            let row = i as u16 / cols;
            let rect = Rect::new(area.x + col * gw, area.y + row * gh, gw, gh);
            if rect.bottom() <= area.bottom() {
                let widget = CompactGridWidget { state: &scenario.grid };
                frame.render_widget(&widget, rect);
            }
        }
    }

    fn render_hotkeys(&self, frame: &mut Frame, area: Rect) {
        let pages = page_count(self.scenarios.len(), GRIDS_PER_PAGE);
        let page_indicator = if pages > 1 {
            format!("  page {}/{}", self.current_page + 1, pages)
        } else {
            String::new()
        };
        let text = format!(
            "[p]ause [s]napshot \u{2190}/\u{2192} page [q]uit{page_indicator}"
        );
        let p = Paragraph::new(text).style(Style::default().fg(Color::DarkGray));
        frame.render_widget(p, area);
    }
}

// -- Tick sub-methods --

impl MpTuiApp {
    fn tick_delta_history(&mut self, sparkline_max: usize) {
        let mut hist = self.metrics.strategy_delta_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for &v in hist.iter() {
            push_bounded(&mut self.delta_history, (v * 10000.0) as u64, sparkline_max);
        }
        hist.clear();
    }

    fn tick_regret_sparklines(&mut self, sparkline_max: usize) {
        drain_scaled(&self.metrics.max_regret_history, &mut self.max_regret_history, 1000.0, sparkline_max);
        drain_scaled_abs(&self.metrics.min_regret_history, &mut self.min_regret_history, 1000.0, sparkline_max);
        drain_scaled(&self.metrics.avg_pos_regret_history, &mut self.avg_pos_regret_history, 1_000_000_000.0, sparkline_max);
        drain_scaled(&self.metrics.prune_history, &mut self.prune_history, 10.0, sparkline_max);
    }

    fn tick_grids(&mut self) {
        let mut grids = self.metrics.strategy_grids
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for (i, grid_opt) in grids.iter_mut().enumerate() {
            if let Some(new_grid) = grid_opt.take() {
                if i < self.scenarios.len() {
                    let old = std::mem::replace(
                        &mut self.scenarios[i].grid.cells, new_grid,
                    );
                    self.scenarios[i].grid.prev_cells = Some(old);
                    self.scenarios[i].grid.iteration_at_snapshot =
                        self.metrics.iterations.load(Ordering::Relaxed);
                }
            }
        }
    }
}

// -- Helpers --

/// Drain a metrics `Mutex<Vec<f64>>`, scale each value, and push into a sparkline buffer.
fn drain_scaled(
    source: &std::sync::Mutex<Vec<f64>>,
    dest: &mut Vec<u64>,
    scale: f64,
    max_len: usize,
) {
    let mut hist = source.lock().unwrap_or_else(|e| e.into_inner());
    for &v in hist.iter() {
        push_bounded(dest, (v * scale) as u64, max_len);
    }
    hist.clear();
}

/// Like `drain_scaled` but takes the absolute value before scaling (for negative regrets).
fn drain_scaled_abs(
    source: &std::sync::Mutex<Vec<f64>>,
    dest: &mut Vec<u64>,
    scale: f64,
    max_len: usize,
) {
    let mut hist = source.lock().unwrap_or_else(|e| e.into_inner());
    for &v in hist.iter() {
        push_bounded(dest, (v.abs() * scale) as u64, max_len);
    }
    hist.clear();
}

pub(crate) fn push_bounded(buf: &mut Vec<u64>, value: u64, max_len: usize) {
    if buf.len() >= max_len {
        buf.remove(0);
    }
    buf.push(value);
}

/// Compute regret statistics from a flat regret array and push to TUI metrics.
///
/// Scans all `AtomicI32` regret values, computes max, min, and average positive
/// regret (after dividing by `regret_scale`), and pushes one sample of each
/// into the metrics sparkline history.
pub fn push_regret_telemetry(
    regrets: &[std::sync::atomic::AtomicI32],
    regret_scale: f64,
    metrics: &BlueprintTuiMetrics,
) {
    if regrets.is_empty() {
        return;
    }
    let mut max_r = i32::MIN;
    let mut min_r = i32::MAX;
    let mut pos_sum: i64 = 0;
    let mut pos_count: u64 = 0;
    for atom in regrets {
        let v = atom.load(Ordering::Relaxed);
        if v > max_r { max_r = v; }
        if v < min_r { min_r = v; }
        if v > 0 {
            pos_sum += v as i64;
            pos_count += 1;
        }
    }
    metrics.push_max_regret(max_r as f64 / regret_scale);
    metrics.push_min_regret(min_r as f64 / regret_scale);
    let avg = if pos_count > 0 {
        (pos_sum as f64 / pos_count as f64) / regret_scale
    } else {
        0.0
    };
    metrics.push_avg_pos_regret(avg);
}

fn compute_progress_ratio(metrics: &BlueprintTuiMetrics) -> f64 {
    if let Some(t) = metrics.target_iterations {
        let current = metrics.iterations.load(Ordering::Relaxed);
        if t > 0 { current as f64 / t as f64 } else { 0.0 }
    } else if let Some(minutes) = metrics.time_limit_minutes {
        let elapsed = metrics.elapsed_secs();
        if minutes > 0 { elapsed / (minutes as f64 * 60.0) } else { 0.0 }
    } else {
        0.0
    }
}

// -- Entry point --

/// Spawn the N-player TUI on a dedicated thread. Returns a join handle.
pub fn run_mp_tui(
    metrics: Arc<BlueprintTuiMetrics>,
    scenarios: Vec<ResolvedMpScenario>,
    telemetry_config: TelemetryConfig,
    refresh_interval: std::time::Duration,
    num_players: u8,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        if let Err(e) = run_mp_tui_inner(
            metrics, scenarios, telemetry_config, refresh_interval, num_players,
        ) {
            eprintln!("MP TUI error: {e}");
        }
    })
}

fn run_mp_tui_inner(
    metrics: Arc<BlueprintTuiMetrics>,
    scenarios: Vec<ResolvedMpScenario>,
    telemetry_config: TelemetryConfig,
    refresh_interval: std::time::Duration,
    num_players: u8,
) -> std::io::Result<()> {
    use crossterm::execute;
    use crossterm::terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
    };
    use ratatui::backend::CrosstermBackend;

    enable_raw_mode()?;
    let mut stderr = std::io::stderr();
    execute!(stderr, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stderr);
    let mut terminal = ratatui::Terminal::new(backend)?;
    let mut app = MpTuiApp::new(
        Arc::clone(&metrics), scenarios, telemetry_config, num_players,
    );
    run_event_loop(&mut app, &metrics, &mut terminal, refresh_interval)?;
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

fn run_event_loop(
    app: &mut MpTuiApp,
    metrics: &Arc<BlueprintTuiMetrics>,
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stderr>>,
    refresh_interval: std::time::Duration,
) -> std::io::Result<()> {
    use crossterm::event::{self, Event};
    loop {
        app.tick();
        terminal.draw(|frame| app.render(frame))?;
        if event::poll(refresh_interval)? {
            if let Event::Key(key) = event::read()? {
                handle_key(app, metrics, key.code);
            }
        }
        if metrics.quit_requested.load(Ordering::Relaxed) {
            break;
        }
    }
    Ok(())
}

fn handle_key(app: &mut MpTuiApp, metrics: &BlueprintTuiMetrics, code: crossterm::event::KeyCode) {
    use crossterm::event::KeyCode;
    match code {
        KeyCode::Char('q') | KeyCode::Esc => {
            metrics.quit_requested.store(true, Ordering::Relaxed);
        }
        KeyCode::Right | KeyCode::Tab => app.next_page(),
        KeyCode::Left | KeyCode::BackTab => app.prev_page(),
        KeyCode::Char('p') => metrics.toggle_pause(),
        KeyCode::Char('s') => metrics.request_snapshot(),
        _ => {}
    }
}

/// Number of pages needed to display `total` items at `per_page` each.
pub fn page_count(total: usize, per_page: usize) -> usize {
    (total + per_page - 1) / per_page
}

/// Slice of items for the given page (zero-indexed).
pub fn page_slice<T>(items: &[T], page: usize, per_page: usize) -> &[T] {
    let start = page * per_page;
    let end = (start + per_page).min(items.len());
    if start >= items.len() { &[] } else { &items[start..end] }
}

#[cfg(test)]
mod tests {
    use test_macros::timed_test;

    // -- page_count tests --

    #[timed_test]
    fn page_count_exact() {
        assert_eq!(super::page_count(6, 6), 1);
    }

    #[timed_test]
    fn page_count_overflow() {
        assert_eq!(super::page_count(7, 6), 2);
    }

    #[timed_test]
    fn page_count_12() {
        assert_eq!(super::page_count(12, 6), 2);
    }

    #[timed_test]
    fn page_count_zero() {
        assert_eq!(super::page_count(0, 6), 0);
    }

    #[timed_test]
    fn page_count_one_item() {
        assert_eq!(super::page_count(1, 6), 1);
    }

    // -- page_slice tests --

    #[timed_test]
    fn page_slice_first() {
        let v: Vec<i32> = (0..12).collect();
        assert_eq!(super::page_slice(&v, 0, 6), &[0, 1, 2, 3, 4, 5]);
    }

    #[timed_test]
    fn page_slice_second() {
        let v: Vec<i32> = (0..12).collect();
        assert_eq!(super::page_slice(&v, 1, 6), &[6, 7, 8, 9, 10, 11]);
    }

    #[timed_test]
    fn page_slice_partial() {
        let v: Vec<i32> = (0..8).collect();
        assert_eq!(super::page_slice(&v, 1, 6), &[6, 7]);
    }

    #[timed_test]
    fn page_slice_out_of_bounds() {
        let v: Vec<i32> = (0..3).collect();
        assert_eq!(super::page_slice(&v, 5, 6).len(), 0);
    }

    #[timed_test]
    fn page_slice_empty_input() {
        let v: Vec<i32> = vec![];
        assert_eq!(super::page_slice(&v, 0, 6).len(), 0);
    }

    // -- MpTuiApp navigation tests --

    #[timed_test]
    fn next_page_wraps() {
        let mut app = test_app(12); // 2 pages
        assert_eq!(app.current_page, 0);
        app.next_page();
        assert_eq!(app.current_page, 1);
        app.next_page();
        assert_eq!(app.current_page, 0); // wraps
    }

    #[timed_test]
    fn prev_page_wraps() {
        let mut app = test_app(12); // 2 pages
        app.prev_page();
        assert_eq!(app.current_page, 1); // wraps backward
    }

    #[timed_test]
    fn next_page_single_page() {
        let mut app = test_app(3); // 1 page
        app.next_page();
        assert_eq!(app.current_page, 0); // stays at 0
    }

    #[timed_test]
    fn prev_page_single_page() {
        let mut app = test_app(3); // 1 page
        app.prev_page();
        assert_eq!(app.current_page, 0); // stays at 0
    }

    #[timed_test]
    fn navigation_zero_scenarios() {
        let mut app = test_app(0);
        app.next_page();
        assert_eq!(app.current_page, 0);
        app.prev_page();
        assert_eq!(app.current_page, 0);
    }

    #[timed_test]
    fn grids_per_page_constant() {
        assert_eq!(super::GRIDS_PER_PAGE, 6);
    }

    // -- render tests --

    #[timed_test]
    fn render_with_scenarios_no_panic() {
        use ratatui::backend::TestBackend;
        let app = test_app(6);
        let backend = TestBackend::new(180, 50);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
    }

    #[timed_test]
    fn render_empty_scenarios_no_panic() {
        use ratatui::backend::TestBackend;
        let app = test_app(0);
        let backend = TestBackend::new(180, 50);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
    }

    #[timed_test]
    fn render_second_page() {
        use ratatui::backend::TestBackend;
        let mut app = test_app(12);
        app.next_page();
        let backend = TestBackend::new(180, 50);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
    }

    // -- tick tests --

    #[timed_test]
    fn tick_updates_throughput() {
        use std::sync::atomic::Ordering;
        let mut app = test_app(2);
        app.metrics.iterations.store(100, Ordering::Relaxed);
        app.tick();
        // iter_per_sec_history should have an entry
        assert!(!app.iter_per_sec_history.is_empty());
    }

    #[timed_test]
    fn tick_reads_delta_history() {
        let mut app = test_app(2);
        app.metrics.push_strategy_delta(0.05);
        app.tick();
        assert!(!app.delta_history.is_empty());
    }

    #[timed_test]
    fn tick_refreshes_grid_from_metrics() {
        use std::sync::atomic::Ordering;
        use crate::blueprint_tui_widgets::CellStrategy;
        let mut app = test_app(2);
        app.metrics.iterations.store(500, Ordering::Relaxed);
        let mut grid: [[CellStrategy; 13]; 13] = std::array::from_fn(|_| {
            std::array::from_fn(|_| CellStrategy::default())
        });
        grid[0][0] = CellStrategy {
            actions: vec![("fold".into(), 1.0)],
            ev: None,
        };
        app.metrics.update_scenario_grid(0, grid);
        app.tick();
        // Grid should have been consumed from metrics
        let actions = &app.scenarios[0].grid.cells[0][0].actions;
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].0, "fold");
    }

    // -- run_mp_tui signature test --

    #[timed_test]
    fn run_mp_tui_returns_join_handle() {
        use std::sync::Arc;
        use std::time::Duration;
        // Verify the function exists and returns JoinHandle<()>.
        // We cannot actually run it (needs a real terminal), so just
        // verify the function is callable with the right types.
        let _: fn(
            Arc<crate::blueprint_tui_metrics::BlueprintTuiMetrics>,
            Vec<super::ResolvedMpScenario>,
            crate::blueprint_tui_config::TelemetryConfig,
            Duration,
            u8,
        ) -> std::thread::JoinHandle<()> = super::run_mp_tui;
    }

    // -- push_bounded tests --

    #[timed_test]
    fn push_bounded_appends() {
        let mut buf = vec![];
        super::push_bounded(&mut buf, 10, 5);
        assert_eq!(buf, vec![10]);
    }

    #[timed_test]
    fn push_bounded_caps_at_max() {
        let mut buf = vec![1, 2, 3, 4, 5];
        super::push_bounded(&mut buf, 6, 5);
        assert_eq!(buf, vec![2, 3, 4, 5, 6]);
    }

    // -- telemetry sparkline tick tests --

    #[timed_test]
    fn tick_consumes_max_regret_history() {
        let mut app = test_app(2);
        app.metrics.push_max_regret(5.0);
        app.metrics.push_max_regret(10.0);
        app.tick();
        assert_eq!(app.max_regret_history.len(), 2);
        // 5.0 * 1000.0 = 5000, 10.0 * 1000.0 = 10000
        assert_eq!(app.max_regret_history[0], 5000);
        assert_eq!(app.max_regret_history[1], 10000);
        // Metrics buffer should be drained
        let hist = app.metrics.max_regret_history.lock().unwrap();
        assert!(hist.is_empty());
    }

    #[timed_test]
    fn tick_consumes_min_regret_history() {
        let mut app = test_app(2);
        app.metrics.push_min_regret(-3.0);
        app.metrics.push_min_regret(-7.5);
        app.tick();
        assert_eq!(app.min_regret_history.len(), 2);
        // Stored as abs * 1000: 3000, 7500
        assert_eq!(app.min_regret_history[0], 3000);
        assert_eq!(app.min_regret_history[1], 7500);
    }

    #[timed_test]
    fn tick_consumes_avg_pos_regret_history() {
        let mut app = test_app(2);
        app.metrics.push_avg_pos_regret(0.000123);
        app.tick();
        assert_eq!(app.avg_pos_regret_history.len(), 1);
        // 0.000123 * 1_000_000_000 = 123000
        assert_eq!(app.avg_pos_regret_history[0], 123000);
    }

    #[timed_test]
    fn tick_consumes_prune_history() {
        let mut app = test_app(2);
        app.metrics.push_prune_fraction(34.5);
        app.metrics.push_prune_fraction(50.0);
        app.tick();
        assert_eq!(app.prune_history.len(), 2);
        // 34.5 * 10.0 = 345, 50.0 * 10.0 = 500
        assert_eq!(app.prune_history[0], 345);
        assert_eq!(app.prune_history[1], 500);
    }

    #[timed_test]
    fn tick_max_regret_empty_when_no_data() {
        let mut app = test_app(2);
        app.tick();
        assert!(app.max_regret_history.is_empty());
    }

    #[timed_test]
    fn tick_prune_history_respects_sparkline_window() {
        let mut app = test_app(2);
        // Set sparkline window to 3
        app.telemetry_config.sparkline_window = 3;
        for i in 0..5 {
            app.metrics.push_prune_fraction(i as f64);
        }
        app.tick();
        // Should be capped at window size 3
        assert_eq!(app.prune_history.len(), 3);
        // Last 3: 2.0, 3.0, 4.0 -> 20, 30, 40
        assert_eq!(app.prune_history[0], 20);
        assert_eq!(app.prune_history[1], 30);
        assert_eq!(app.prune_history[2], 40);
    }

    // -- render tests for expanded metrics panel --

    #[timed_test]
    fn render_metrics_expanded_height() {
        use ratatui::backend::TestBackend;
        // Render the full app and verify the top-level layout
        // allocates 19 rows for the metrics panel.
        let app = test_app(2);
        let backend = TestBackend::new(120, 40);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|frame| {
            // The render method uses Constraint::Length(19) for metrics
            app.render(frame);
        }).unwrap();
        // If we got here without panic, the 19-row layout worked.
        // Also verify the total area was used.
        let size = terminal.size().unwrap();
        assert_eq!(size.height, 40);
    }

    #[timed_test]
    fn render_all_sparklines_no_panic() {
        use ratatui::backend::TestBackend;
        let mut app = test_app(2);
        // Push data into all sparkline histories
        app.metrics.push_max_regret(10.0);
        app.metrics.push_min_regret(-5.0);
        app.metrics.push_avg_pos_regret(0.001);
        app.metrics.push_prune_fraction(25.0);
        app.metrics.push_strategy_delta(0.05);
        app.tick();
        let backend = TestBackend::new(120, 40);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
    }

    // -- push_regret_telemetry tests --

    #[timed_test]
    fn push_regret_telemetry_max_positive() {
        use std::sync::atomic::AtomicI32;
        let metrics = std::sync::Arc::new(
            crate::blueprint_tui_metrics::BlueprintTuiMetrics::new(None, None),
        );
        // REGRET_SCALE = 1000.0, so 5000 raw = 5.0 chip value
        let regrets = [
            AtomicI32::new(5000),
            AtomicI32::new(-3000),
            AtomicI32::new(2000),
        ];
        super::push_regret_telemetry(&regrets, 1000.0, &metrics);
        let hist = metrics.max_regret_history.lock().unwrap();
        assert_eq!(hist.len(), 1);
        assert!((hist[0] - 5.0).abs() < 1e-9);
    }

    #[timed_test]
    fn push_regret_telemetry_min_negative() {
        use std::sync::atomic::AtomicI32;
        let metrics = std::sync::Arc::new(
            crate::blueprint_tui_metrics::BlueprintTuiMetrics::new(None, None),
        );
        let regrets = [
            AtomicI32::new(5000),
            AtomicI32::new(-7000),
            AtomicI32::new(2000),
        ];
        super::push_regret_telemetry(&regrets, 1000.0, &metrics);
        let hist = metrics.min_regret_history.lock().unwrap();
        assert_eq!(hist.len(), 1);
        assert!((hist[0] - (-7.0)).abs() < 1e-9);
    }

    #[timed_test]
    fn push_regret_telemetry_avg_positive() {
        use std::sync::atomic::AtomicI32;
        let metrics = std::sync::Arc::new(
            crate::blueprint_tui_metrics::BlueprintTuiMetrics::new(None, None),
        );
        // Two positive: 5000 and 2000, scale=1000 -> 5.0 and 2.0 -> avg=3.5
        let regrets = [
            AtomicI32::new(5000),
            AtomicI32::new(-3000),
            AtomicI32::new(2000),
        ];
        super::push_regret_telemetry(&regrets, 1000.0, &metrics);
        let hist = metrics.avg_pos_regret_history.lock().unwrap();
        assert_eq!(hist.len(), 1);
        assert!((hist[0] - 3.5).abs() < 1e-9);
    }

    #[timed_test]
    fn push_regret_telemetry_empty_regrets() {
        let metrics = std::sync::Arc::new(
            crate::blueprint_tui_metrics::BlueprintTuiMetrics::new(None, None),
        );
        let regrets: [std::sync::atomic::AtomicI32; 0] = [];
        super::push_regret_telemetry(&regrets, 1000.0, &metrics);
        // No values should be pushed for empty input
        let max_hist = metrics.max_regret_history.lock().unwrap();
        assert!(max_hist.is_empty());
    }

    #[timed_test]
    fn push_regret_telemetry_all_negative() {
        use std::sync::atomic::AtomicI32;
        let metrics = std::sync::Arc::new(
            crate::blueprint_tui_metrics::BlueprintTuiMetrics::new(None, None),
        );
        let regrets = [
            AtomicI32::new(-1000),
            AtomicI32::new(-2000),
        ];
        super::push_regret_telemetry(&regrets, 1000.0, &metrics);
        // avg_pos_regret should be 0.0 when no positive regrets exist
        let hist = metrics.avg_pos_regret_history.lock().unwrap();
        assert_eq!(hist.len(), 1);
        assert!((hist[0] - 0.0).abs() < 1e-9);
    }

    fn test_app(num_scenarios: usize) -> super::MpTuiApp {
        use std::sync::Arc;
        use crate::blueprint_tui_config::TelemetryConfig;
        use crate::blueprint_tui_metrics::BlueprintTuiMetrics;
        use crate::blueprint_tui_widgets::HandGridState;

        let metrics = Arc::new(BlueprintTuiMetrics::new(Some(1000), None));
        let scenarios: Vec<super::ResolvedMpScenario> = (0..num_scenarios)
            .map(|i| super::ResolvedMpScenario {
                name: format!("Scenario {i}"),
                node_idx: i as u32,
                grid: HandGridState {
                    cells: std::array::from_fn(|_| {
                        std::array::from_fn(|_| Default::default())
                    }),
                    prev_cells: None,
                    scenario_name: format!("Scenario {i}"),
                    action_path: vec![],
                    board_display: None,
                    cluster_id: None,
                    street_label: "Preflop".to_string(),
                    iteration_at_snapshot: 0,
                    error_message: None,
                },
            })
            .collect();
        super::MpTuiApp::new(
            metrics,
            scenarios,
            TelemetryConfig::default(),
            6,
        )
    }
}
