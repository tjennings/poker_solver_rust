# Blueprint Training TUI — Design Document

**Date**: 2026-03-06
**Status**: Approved

## Overview

A Ratatui-based terminal dashboard for monitoring Blueprint V2 (Pluribus-style) training in real time. Vertical split layout: left panel shows telemetry and progress, right panel shows configurable strategy grids as colored 13x13 hand matrices.

Replaces the current text-based `print_every_minutes` output as the default interactive mode. Text output available via `--no-tui` flag.

## Architecture

Extends the existing postflop TUI infrastructure in `crates/trainer/src/`. New files:

| File | Purpose |
|-|-|
| `blueprint_tui.rs` | Main app: layout, render loop, input handling |
| `blueprint_tui_metrics.rs` | Shared atomic metrics struct polled by render thread |
| `blueprint_tui_widgets.rs` | Ratatui widgets: hand grid, sparkline, progress bar |
| `blueprint_tui_config.rs` | YAML-driven TUI config (scenarios, intervals, layout) |

The `train-blueprint` command spawns the TUI on a dedicated thread, passing an `Arc<BlueprintTuiMetrics>`. The training loop writes to atomics; the render thread reads them at its refresh rate. No mutexes on the hot path.

The existing `lhe_viz.rs` hand matrix data model (`HandMatrix`, `HandStrategy`) is extracted into a shared module. A new `HandGridWidget` consumes it as a Ratatui `StatefulWidget`. The old stdout-based renderer stays for `--no-tui` fallback.

## Data Flow

```
BlueprintTrainer (training thread)
  |
  +-- every iteration ----------> Arc<BlueprintTuiMetrics>
  |    . iterations_completed (AtomicU64)
  |    . deals_traversed (AtomicU64)
  |    . elapsed_ns (AtomicU64)
  |
  +-- every strategy_delta_interval --> metrics.strategy_snapshots
  |    . per-scenario strategy vectors (Mutex<Vec<f32>>)
  |    . delta from previous snapshot (atomic)
  |
  +-- every exploitability_interval --> metrics.exploitability_history
  |    . spawns exploitability on background thread
  |    . pushes result to Mutex<VecDeque<(timestamp, f64)>>
  |
  +-- every random_hold_minutes ------> metrics.random_scenario
       . picks new random game node
       . writes scenario descriptor (Mutex<ScenarioState>)

BlueprintTuiApp (render thread, ~4 fps)
  |
  +-- reads atomics (lock-free) for counters
  +-- reads Mutex snapshots for strategy grids + exploitability
  +-- computes derived: iter/sec, ETA, delta trends
  +-- renders via Ratatui
```

- **Hot path is lock-free** -- iteration counters are plain atomics
- **Mutexes only for infrequent bulk data** -- strategy snapshots and exploitability update every 30s+
- **Exploitability runs on a third thread** -- expensive, doesn't block training or rendering
- **ETA** -- based on iterations/elapsed extrapolated to target, or wall-clock limit

One iteration = one sampled deal (4 hole cards + 5 board), traversed twice (once per player) via external-sampling MCCFR through the full game tree (preflop through river).

## Config Design

New `tui` section in the Blueprint V2 YAML:

```yaml
tui:
  enabled: true
  refresh_rate_ms: 250

  telemetry:
    exploitability_interval_minutes: 5
    strategy_delta_interval_seconds: 30
    sparkline_window: 60  # data points in iter/sec graph

  scenarios:
    - name: "SB Open"
      player: SB
      actions: []

    - name: "BB vs 2.5x"
      player: BB
      actions: [raise-0]

    - name: "SB Cbet Kh7s2d"
      player: SB
      actions: [raise-0, call]
      board: [Kh, 7s, 2d]
      street: flop

    - name: "BB Face Turn Bet"
      player: BB
      actions: [raise-0, call, bet-2, call, bet-1]
      board: [Kh, 7s, 2d, Tc]
      street: turn

  random_scenario:
    enabled: true
    hold_minutes: 3
    pool: [preflop, flop, turn]
```

- Actions use the same encoding as info keys (raise-0, bet-2, call, check, fold)
- `board` is optional: absent = preflop, present = triggers cluster resolution
- `pool` controls which streets the random scenario samples from
- `--no-tui` falls back to current `print_every_minutes` text output

## Left Panel -- Telemetry

```
+-- Training Progress --------------------+
| Iterations: 1,234,567 / 10,000,000      |
| ██████████████░░░░░░░░░░░░░░  12.3%     |
| Runtime: 2h 34m    ETA: 18h 26m         |
|                                          |
| Throughput (iter/sec)                    |
| ▃▅▇▆▅▇█▇▆▅▃▅▇▆▅▇█▇▆▅▃▅▇▆▅  48,231 ^P  |
|                                          |
| Exploitability (mBB/hand)                |
| 250+                                     |
| 200+ \                                   |
| 150+  \                                  |
| 100+   \___                              |
|  50+       \____\___                      |
|    +------------------------ 2h 34m      |
| Last: 47.2 mBB  delta-strat: 0.0031     |
|                                          |
| [p]ause  [s]napshot  [e]xploitability    |
+------------------------------------------+
```

- **Progress bar**: Ratatui `Gauge`, iteration-based or time-based per config
- **Sparkline**: Ratatui `Sparkline`, rolling window of iter/sec. ^P marks peak throughput.
- **Exploitability chart**: Ratatui `Chart` with line `Dataset`, auto-scaling Y-axis, wall-time X-axis
- **Strategy delta**: average absolute change in strategy weights since last snapshot
- **Hotkey bar**: footer showing available commands

## Right Panel -- Tabbed Strategy Grids

```
+-- Scenarios ----------------------------+
| [SB Open] [BB vs 2.5x] [Cbet K72]  >   |
|                                          |
|  SB Open -- Preflop                      |
|                                          |
|    A  K  Q  J  T  9  8  7  6  5  4  ... |
|  A 92 87 85 83 80 71 68 55 52 49 44 ... |
|  K 82 91 78 74 66 58 47 40 37 35 32 ... |
|  Q 79 72 88 70 63 51 42 36 33 31 28 ... |
|  ...                                     |
|                                          |
|  Legend: FOLD CALL BET-S BET-M BET-L AI  |
|  Board: --         Cluster: --           |
|  Iter: 1,234,567   Actions: [root]       |
+------------------------------------------+
```

- **Tab bar**: Ratatui `Tabs`, Left/Right arrow keys to switch. Last tab is random (cyan/teal border).
- **Hand grid**: Custom `StatefulWidget` (13x13). Cell color = dominant action. Text = dominant action %.
- **Context footer**: board cards, cluster ID, iteration count, action path
- **Random tab**: Same grid + countdown timer ("Next in 1m 42s")

### Hand grid state

```rust
struct HandGridState {
    /// 169 entries (13x13), each a Vec of (action_name, frequency)
    cells: [[Vec<(String, f32)>; 13]; 13],
    scenario_name: String,
    action_path: Vec<String>,
    board: Option<Vec<Card>>,
    cluster_id: Option<u32>,
    street: Street,
}
```

## Color Scheme

### Action Colors

| Action | Color | Rationale |
|-|-|-|
| Fold | Red (soft) | Universal "stop" |
| Check | Light blue | Passive, neutral |
| Call | Green | Proceed |
| Bet (small) | Yellow | Mild aggression |
| Bet (medium) | Orange | Moderate aggression |
| Bet (large/pot+) | Magenta | Heavy aggression |
| All-in | Bright white on red | Maximum action |
| Raise (any) | Same tier as equivalent bet size | Consistent aggression scale |

### Rendering Rules

- Cell background = color of highest-frequency action
- If no action exceeds 60%, blend top two colors
- Text inside cell = dominant action percentage (e.g. `72`)
- Early training (near-uniform): show dim version of whichever action leads, even by 1%. A 35/33/32 fold/call/raise cell shows as dim red with `35` -- signal visible from noise immediately
- **Convergence overlay**: cells where strategy delta < threshold over last two snapshots get a bright border. Unconverged cells have no border. Grid "lights up" as hands stabilize.

### Random Tab Accent

Cyan/teal border and countdown timer in same accent color to distinguish from fixed scenarios.

## Input Handling

| Key | Action |
|-|-|
| `q` / `Ctrl-C` | Quit gracefully (finish iteration, write final snapshot) |
| Left/Right | Switch scenario tabs |
| `p` | Pause/resume training |
| `s` | Trigger immediate snapshot write |
| `e` | Trigger immediate exploitability calculation |
| `?` | Toggle help overlay |

- **Pause**: shared `AtomicBool`, training loop checks at top of each iteration, spins on condvar until unpaused
- **Snapshot/exploitability triggers**: shared `AtomicBool` flags checked in `check_timed_actions()`

## Terminal Size

- Tab-based layout: right panel shows one scenario at a time, tabs to cycle
- Graceful degradation: if terminal is very narrow, left panel stacks vertically instead of side-by-side
- No minimum size requirement, but best experience at 160x50+

## Additional Features

1. **Color intensity scaling** -- auto-scale so polarization is visible even when strategies are near-uniform early on
2. **Convergence indicator per cell** -- bright border on stabilized cells, visual sense of which hands converge first
3. **Peak throughput marker** -- annotate peak iter/sec on sparkline to detect performance degradation

## Testing Strategy

- **Unit tests**: HandGridWidget rendering with mock data, config parsing, metric computation (ETA, rates)
- **Integration test**: Spin up BlueprintTrainer with TUI metrics, run 100 iterations, assert metrics are populated and strategy grids non-empty
- **No visual regression tests** -- Ratatui rendering requires manual verification
