//! Full-screen TUI dashboard for solve-postflop progress monitoring.
//!
//! Runs on a dedicated thread, sampling shared atomic counters from the
//! solver and rendering sparklines, gauges, and per-flop convergence data.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::backend::CrosstermBackend;
use ratatui::prelude::*;
use poker_solver_core::preflop::SolverCounters;
use ratatui::widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Sparkline};

use crate::tui_metrics::TuiMetrics;

/// Maximum number of data points kept in each sparkline history.
const SPARKLINE_HISTORY: usize = 60;

/// Application state owned by the TUI thread.
pub struct TuiApp {
    metrics: Arc<TuiMetrics>,
    counters: Arc<SolverCounters>,
    start_time: Instant,

    // Sparkline histories
    traversals_per_sec: Vec<u64>,
    total_traversals: Vec<u64>,
    pct_traversals_pruned: Vec<u64>,
    pct_actions_pruned: Vec<u64>,

    // Previous tick counters for computing deltas
    prev_traversal_count: u64,
    prev_pruned_traversal_count: u64,
    prev_total_action_slots: u64,
    prev_pruned_action_slots: u64,

    // Refresh interval for rate calculation
    refresh_secs: f64,
}

impl TuiApp {
    pub fn new(metrics: Arc<TuiMetrics>, counters: Arc<SolverCounters>, refresh_secs: f64) -> Self {
        Self {
            metrics,
            counters,
            start_time: Instant::now(),
            traversals_per_sec: Vec::with_capacity(SPARKLINE_HISTORY),
            total_traversals: Vec::with_capacity(SPARKLINE_HISTORY),
            pct_traversals_pruned: Vec::with_capacity(SPARKLINE_HISTORY),
            pct_actions_pruned: Vec::with_capacity(SPARKLINE_HISTORY),
            prev_traversal_count: 0,
            prev_pruned_traversal_count: 0,
            prev_total_action_slots: 0,
            prev_pruned_action_slots: 0,
            refresh_secs,
        }
    }

    /// Sample metrics and update sparkline histories.
    pub fn tick(&mut self) {
        let t = self.counters.traversal_count.load(Ordering::Relaxed);
        let pt = self.counters.pruned_traversal_count.load(Ordering::Relaxed);
        let ta = self.counters.total_action_slots.load(Ordering::Relaxed);
        let pa = self.counters.pruned_action_slots.load(Ordering::Relaxed);

        let dt = t.saturating_sub(self.prev_traversal_count);
        let dpt = pt.saturating_sub(self.prev_pruned_traversal_count);
        let dta = ta.saturating_sub(self.prev_total_action_slots);
        let dpa = pa.saturating_sub(self.prev_pruned_action_slots);

        // Traversals per second (scale by refresh interval)
        let tps = if self.refresh_secs > 0.0 {
            (dt as f64 / self.refresh_secs) as u64
        } else {
            dt
        };
        push_bounded(&mut self.traversals_per_sec, tps, SPARKLINE_HISTORY);

        // Total traversals (cumulative count for this SPR)
        push_bounded(&mut self.total_traversals, t, SPARKLINE_HISTORY);

        // Pruning percentages (0-100)
        let pct_trav = if dt > 0 { (dpt * 100) / dt } else { 0 };
        push_bounded(&mut self.pct_traversals_pruned, pct_trav, SPARKLINE_HISTORY);
        let pct_act = if dta > 0 { (dpa * 100) / dta } else { 0 };
        push_bounded(&mut self.pct_actions_pruned, pct_act, SPARKLINE_HISTORY);

        self.prev_traversal_count = t;
        self.prev_pruned_traversal_count = pt;
        self.prev_total_action_slots = ta;
        self.prev_pruned_action_slots = pa;
    }

    /// Render the full dashboard to the given frame.
    pub fn render(&self, frame: &mut Frame) {
        let area = frame.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // SPR gauge
                Constraint::Length(1), // Flop gauge
                Constraint::Length(1), // spacer
                Constraint::Length(2), // traversals/sec sparkline
                Constraint::Length(2), // remaining sparkline
                Constraint::Length(2), // % traversals pruned
                Constraint::Length(2), // % actions pruned
                Constraint::Length(1), // section header
                Constraint::Min(3),   // active flops list
                Constraint::Length(1), // footer
            ])
            .split(area);

        self.render_spr_gauge(frame, chunks[0]);
        self.render_flop_gauge(frame, chunks[1]);
        self.render_sparkline_row(
            frame,
            chunks[3],
            "Traversals/sec",
            &self.traversals_per_sec,
            Color::Cyan,
        );
        self.render_sparkline_row(
            frame,
            chunks[4],
            "Total traversals",
            &self.total_traversals,
            Color::Yellow,
        );
        self.render_sparkline_row(
            frame,
            chunks[5],
            "% Trav pruned",
            &self.pct_traversals_pruned,
            Color::Green,
        );
        self.render_sparkline_row(
            frame,
            chunks[6],
            "% Actions pruned",
            &self.pct_actions_pruned,
            Color::Green,
        );
        self.render_active_flops_header(frame, chunks[7]);
        self.render_active_flops(frame, chunks[8]);
        self.render_footer(frame, chunks[9]);
    }

    fn render_spr_gauge(&self, frame: &mut Frame, area: Rect) {
        let idx = self.metrics.current_spr.load(Ordering::Relaxed);
        let total = self.metrics.total_sprs.load(Ordering::Relaxed);
        let ratio = if total > 0 {
            idx as f64 / total as f64
        } else {
            0.0
        };
        let elapsed = format_duration(self.start_time.elapsed().as_secs_f64());
        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(Color::Cyan))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(format!("SPR Progress: {idx}/{total}    {elapsed}"));
        frame.render_widget(gauge, area);
    }

    fn render_flop_gauge(&self, frame: &mut Frame, area: Rect) {
        let done = self.metrics.flops_completed.load(Ordering::Relaxed);
        let total = self.metrics.total_flops.load(Ordering::Relaxed);
        let ratio = if total > 0 {
            done as f64 / total as f64
        } else {
            0.0
        };
        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(Color::Green))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(format!("Flop Progress: {done}/{total}"));
        frame.render_widget(gauge, area);
    }

    fn render_sparkline_row(
        &self,
        frame: &mut Frame,
        area: Rect,
        label: &str,
        data: &[u64],
        color: Color,
    ) {
        let current = data.last().copied().unwrap_or(0);
        let formatted = format_count(current);
        let title = format!("{label}: {formatted}");
        let sparkline = Sparkline::default()
            .block(Block::default().title(title))
            .data(data)
            .style(Style::default().fg(color));
        frame.render_widget(sparkline, area);
    }

    fn render_active_flops_header(&self, frame: &mut Frame, area: Rect) {
        let header =
            Paragraph::new("── Active Flops ──").style(Style::default().fg(Color::Yellow));
        frame.render_widget(header, area);
    }

    fn render_active_flops(&self, frame: &mut Frame, area: Rect) {
        let mut items: Vec<(usize, String)> = Vec::new();
        for entry in self.metrics.flop_states.iter() {
            let name = entry.key();
            let state = entry.value();
            let latest_expl = state.exploitability_history.last().copied().unwrap_or(0.0);
            // Pre-compute min/max once for this flop's history.
            let hist = &state.exploitability_history;
            let (hist_min, hist_max) = hist_min_max(hist);
            // Build mini sparkline from history (last 20 values)
            let spark_text: String = hist
                .iter()
                .rev()
                .take(20)
                .rev()
                .map(|&v| sparkline_char(v, hist_min, hist_max))
                .collect();
            let line = format!(
                "{:<8} expl {} {:>8.1} mBB/h  iter {}/{}",
                name, spark_text, latest_expl, state.iteration, state.max_iterations,
            );
            items.push((state.iteration, line));
        }
        // Sort by iteration progress descending (most-progressed first)
        items.sort_by(|a, b| b.0.cmp(&a.0));
        let mut list_items: Vec<ListItem> = items
            .into_iter()
            .map(|(_, text)| ListItem::new(text))
            .collect();
        // Truncate to visible rows, showing "+N more" if clipped
        let visible_rows = area.height as usize;
        if list_items.len() > visible_rows && visible_rows > 0 {
            let hidden = list_items.len() - (visible_rows - 1);
            list_items.truncate(visible_rows - 1);
            list_items.push(ListItem::new(format!("  ... +{hidden} more flops")));
        }
        let list = List::new(list_items).block(Block::default().borders(Borders::NONE));
        frame.render_widget(list, area);
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let elapsed_secs = self.start_time.elapsed().as_secs_f64();
        let flops_done = self.metrics.flops_completed.load(Ordering::Relaxed);
        let flops_total = self.metrics.total_flops.load(Ordering::Relaxed);
        let spr_idx = self.metrics.current_spr.load(Ordering::Relaxed);
        let spr_total = self.metrics.total_sprs.load(Ordering::Relaxed);

        // Overall progress: completed SPRs + fraction of current SPR
        let overall_ratio = if spr_total > 0 {
            let spr_frac = if flops_total > 0 {
                flops_done as f64 / flops_total as f64
            } else {
                0.0
            };
            (spr_idx as f64 + spr_frac) / spr_total as f64
        } else {
            0.0
        };

        let eta = if overall_ratio > 0.01 {
            let total_est = elapsed_secs / overall_ratio;
            format_duration((total_est - elapsed_secs).max(0.0))
        } else {
            "calculating...".to_string()
        };

        let footer = Paragraph::new(format!(
            "Elapsed: {}    ETA: {}",
            format_duration(elapsed_secs),
            eta,
        ));
        frame.render_widget(footer, area);
    }
}

/// Run the TUI on a dedicated thread. Returns a join handle.
/// The TUI exits when `done` is set to `true` or user presses 'q'.
pub fn run_tui(
    metrics: Arc<TuiMetrics>,
    counters: Arc<SolverCounters>,
    refresh_interval: Duration,
    done: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    let refresh_secs = refresh_interval.as_secs_f64();
    std::thread::spawn(move || {
        if let Err(e) = run_tui_inner(metrics, counters, refresh_interval, refresh_secs, done) {
            eprintln!("TUI error: {e}");
        }
    })
}

fn run_tui_inner(
    metrics: Arc<TuiMetrics>,
    counters: Arc<SolverCounters>,
    refresh_interval: Duration,
    refresh_secs: f64,
    done: Arc<AtomicBool>,
) -> io::Result<()> {
    enable_raw_mode()?;
    let mut stderr = io::stderr();
    execute!(stderr, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stderr);
    let mut terminal = ratatui::Terminal::new(backend)?;

    let mut app = TuiApp::new(metrics, counters, refresh_secs);

    loop {
        app.tick();
        terminal.draw(|frame| app.render(frame))?;

        if event::poll(refresh_interval)? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }

        if done.load(Ordering::Relaxed) {
            // Final render
            app.tick();
            terminal.draw(|frame| app.render(frame))?;
            std::thread::sleep(Duration::from_secs(2));
            break;
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────

fn push_bounded(buf: &mut Vec<u64>, value: u64, max_len: usize) {
    if buf.len() >= max_len {
        buf.remove(0);
    }
    buf.push(value);
}

fn format_duration(secs: f64) -> String {
    let h = (secs / 3600.0) as u64;
    let m = ((secs % 3600.0) / 60.0) as u64;
    let s = (secs % 60.0) as u64;
    format!("{h:02}:{m:02}:{s:02}")
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

fn hist_min_max(history: &[f64]) -> (f64, f64) {
    let min = history.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (min, max)
}

fn sparkline_char(value: f64, min: f64, max: f64) -> char {
    let range = (max - min).max(1e-9);
    let normalized = ((value - min) / range * 7.0) as usize;
    const CHARS: [char; 8] = ['\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}', '\u{2588}'];
    CHARS[normalized.min(7)]
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui_metrics::TuiMetrics;

    fn make_counters() -> Arc<SolverCounters> {
        Arc::new(SolverCounters::default())
    }

    #[test]
    fn tui_app_renders_without_panic() {
        let metrics = Arc::new(TuiMetrics::new(3, 200));
        let counters = make_counters();
        let mut app = TuiApp::new(Arc::clone(&metrics), Arc::clone(&counters), 1.0);
        counters.traversal_count.fetch_add(1000, Ordering::Relaxed);
        app.tick();
        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
    }

    #[test]
    fn tick_computes_deltas() {
        let metrics = Arc::new(TuiMetrics::new(1, 10));
        let counters = make_counters();
        let mut app = TuiApp::new(Arc::clone(&metrics), Arc::clone(&counters), 1.0);

        counters.traversal_count.fetch_add(500, Ordering::Relaxed);
        app.tick();
        assert_eq!(app.traversals_per_sec.last(), Some(&500));

        counters.traversal_count.fetch_add(300, Ordering::Relaxed);
        app.tick();
        assert_eq!(app.traversals_per_sec.last(), Some(&300));
    }

    #[test]
    fn push_bounded_caps_at_max() {
        let mut buf = Vec::new();
        for i in 0..100 {
            push_bounded(&mut buf, i, 60);
        }
        assert_eq!(buf.len(), 60);
        assert_eq!(*buf.first().unwrap(), 40);
        assert_eq!(*buf.last().unwrap(), 99);
    }

    #[test]
    fn format_count_scales() {
        assert_eq!(format_count(500), "500");
        assert_eq!(format_count(1_500), "1.5K");
        assert_eq!(format_count(2_500_000), "2.5M");
    }

    #[test]
    fn format_duration_works() {
        assert_eq!(format_duration(0.0), "00:00:00");
        assert_eq!(format_duration(3661.0), "01:01:01");
        assert_eq!(format_duration(90.0), "00:01:30");
    }
}
