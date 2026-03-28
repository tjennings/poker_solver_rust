# Regret Audit Panel — Design Document

**Date**: 2026-03-27
**Status**: Approved

## Overview

Add a TUI panel for auditing per-hand regret accumulation at specific spots during blueprint training. This is a debugging/validation tool for verifying that MCCFR regret propagation and bucketing are working correctly.

## Motivation

Currently the TUI shows only aggregate regret statistics (max/min/avg sparklines). There's no way to inspect regret values for a specific hand at a specific decision point. This makes it hard to:

- Verify regret signs/magnitudes are intuitive for known spots
- Catch bugs in regret propagation (e.g. regret not updating, wrong sign)
- Detect bucket misassignment (hand landing in wrong equity bucket)

## Approach: Polled Snapshot

Each TUI tick, read current regret values directly from `BlueprintStorage` for audited `(node_idx, bucket, action)` tuples. No changes to the MCCFR hot path — all reads are from existing `AtomicI32` values.

**Alternatives considered:**
- **Callback instrumented**: Hook into MCCFR traversal for per-iteration data. Rejected — adds branch to performance-critical hot path.
- **Hybrid (polled + periodic deep sample)**: Adds complexity for marginal gain over pure polling.

## Layout

Horizontal split in the upper portion of the TUI:

```
┌─ Left (60%) ─────────────────┬─ Right (40%) ──────────────────┐
│ Iterations: 1.2M / 10M       │ ┌─ Regret Audit ─────────────┐ │
│ ████████████████░░░░░░ 12.0%  │ │ [AKo@SB Open] | [T9s@Flop] │ │
│ Runtime: 2h 15m  ETA: 16h 30m│ │                              │ │
│ Throughput: 850 it/s (peak 1K)│ │ Hand: AKo  Player: SB       │ │
│ Strategy delta: 0.001234      │ │ Buckets: pf:3 → f:12 → t:8  │ │
│ Leaves moving: 12.3%          │ │ Iter: 1.2M                   │ │
│ Max pos regret: 45.2          │ │                              │ │
│ Max neg regret: -23.1         │ │  Action   Regret  Δ    Trend │ │
│ Avg pos regret: 1.23e-4       │ │  fold     -1.20  -0.03  ↓   │ │
│ Traversals pruned: 8.1%       │ │  call     +0.30  +0.02  ↑   │ │
│                               │ │  raise5bb +0.90  +0.01  ↑   │ │
│                               │ │                              │ │
│                               │ │  Strategy: f:0% c:25% r:75% │ │
│                               │ └──────────────────────────────┘ │
├───────────────────────────────┴──────────────────────────────────┤
│ [SB Open] | [BB vs 2.5x] | [SB Cbet K72]                        │
│ ┌─ 13×13 hand grid ────────────────────────────────────────────┐ │
│ └──────────────────────────────────────────────────────────────┘ │
│ [p]ause [s]napshot [←/→]scenario [↑/↓]audit [q]uit              │
└──────────────────────────────────────────────────────────────────┘
```

- Top area splits horizontally: sparklines left (60%), audit panel right (40%)
- Audit panel has its own tab bar, navigated with `↑/↓` arrows
- Hand grids remain full-width below
- If no audits configured, right side doesn't render — sparklines take full width (no layout shift)

## YAML Configuration

New `regret_audits` list under `tui:`, sibling to `scenarios`:

```yaml
tui:
  enabled: true
  scenarios:
    - name: "SB Open"
      spot: ""
  regret_audits:
    - name: "AKo SB open"
      spot: ""
      hand: "AKo"
      player: SB
    - name: "T9s flop cbet"
      spot: "sb:2bb,bb:call|AsTd9d"
      hand: "Ts9s"
      player: SB
    - name: "TT 3bet pot"
      spot: "sb:2bb,bb:4bb"
      hand: "TT"
      player: SB
```

- `name` — label for the audit tab
- `spot` — same spot notation as scenarios (self-contained, board after `|`)
- `hand` — canonical (`AKo`, `TT`) for preflop, specific combo (`Ts9s`) for postflop where board interaction matters
- `player` — `SB` or `BB`, determines whose regrets to read

Config struct:

```rust
pub struct RegretAuditConfig {
    pub name: String,
    pub spot: String,
    pub hand: String,
    pub player: PlayerLabel,  // SB | BB
}
```

## Data Model

Each config entry resolves at startup to:

```rust
pub struct ResolvedRegretAudit {
    name: String,
    node_idx: u32,                      // resolved from spot notation
    player: u8,                         // 0 or 1
    bucket_trail: Vec<(Street, u16)>,   // [(Preflop, 3), (Flop, 12), ...]
    action_labels: Vec<String>,         // ["fold", "call", "raise 5bb"]
    num_actions: usize,
    // Updated each tick:
    regrets: Vec<f64>,                  // current cumulative regrets
    prev_regrets: Vec<f64>,             // previous tick snapshot (for raw delta)
    trend_buffer: Vec<VecDeque<f64>>,   // per-action ring buffer for smoothed trend
}
```

## Tick Logic

Runs at TUI refresh rate (e.g. every 500ms):

1. Read `storage.get_regret(node_idx, bucket, action)` for each action
2. Compute raw delta: `current - prev`
3. Push delta into per-action ring buffer (size = sparkline_window)
4. Compute smoothed trend: sign of moving average over ring buffer
5. Derive current strategy via regret matching (positive regrets normalized)
6. Store current as prev for next tick

Bucket trail is computed once at startup — walk the hand through each street's bucketing function and record `(street, bucket_id)`.

## Panel Rendering

```
┌─ Regret Audit: AKo SB open ──────────────┐
│ Hand: AKo  Player: SB  Iter: 1.2M        │
│ Buckets: pf:3 → fl:12 → tn:8 → rv:2     │
│                                           │
│  Action     Regret     Δ/tick   Trend     │
│  fold       -1200      -30      ↓         │
│  call       +300       +20      ↑         │
│  raise5bb   +900       +10      ↑         │
│                                           │
│  Strategy: f:0% c:25% r:75%              │
└───────────────────────────────────────────┘
```

- **Title**: audit name from config
- **Header**: hand, player, current iteration count
- **Bucket trail**: abbreviated street labels (`pf`, `fl`, `tn`, `rv`) — wrong bucket immediately visible
- **Table**: one row per action. Regret = raw i32/1000 from storage. Delta = change since last tick. Trend = arrow + color from smoothed moving average (green ↑, red ↓, gray →)
- **Strategy**: derived from positive regrets via regret matching, compact percentages
- **Colors**: positive regrets green, negative red, zero gray. Strategy uses existing action color scheme

## Keybindings

- `↑/↓` — cycle through audit tabs
- Updated hotkey bar: `[p]ause [s]napshot [←/→]scenario [↑/↓]audit [q]uit`

## Error Handling

- Invalid spot notation: show error in audit panel (like scenario error handling)
- Hand doesn't resolve to a bucket: show diagnostic message
- No regret_audits configured: right column doesn't render, layout unchanged
