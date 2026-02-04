# Poker Solver Rust + Tauri Rewrite Design

**Date:** 2026-02-04
**Goal:** Rewrite the Python poker solver in Rust using Burn for ML and Tauri for UX
**Scope:** Same CFR solver core, enhanced desktop UI with training controls and strategy exploration

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Tauri Shell                        │
│  (Window management, IPC, file dialogs, menus)      │
├─────────────────────────────────────────────────────┤
│              React + TypeScript Frontend             │
│  (Training UI, Strategy Explorer, Visualizations)   │
├─────────────────────────────────────────────────────┤
│                 Rust Core (Library)                  │
│  (CFR Engine, Game Logic, Burn ML, Config Loading)  │
└─────────────────────────────────────────────────────┘
```

### Layer Responsibilities

**Rust Core** - A library crate containing:
- Game abstractions (trait-based)
- Kuhn Poker and HUNL Preflop implementations
- CFR engine (vanilla for CPU, batched CFR+ for GPU via Burn)
- Configuration loading from YAML presets
- Exploitability calculation
- Strategy serialization/deserialization

**Tauri Commands** - Thin wrapper exposing Rust functions to the frontend:
- `start_training(game, config, params)` → spawns background task
- `stop_training()` → cancels training
- `get_training_progress()` → returns current iteration, exploitability
- `get_strategy(stack, history)` → returns action probabilities for matrix
- `load_config(path)` → parses YAML preset
- `save_strategy(path)` / `load_strategy(path)`

**React Frontend** - Single-page app with:
- Training panel (controls, progress, charts)
- Strategy explorer (matrix, game tree navigation)
- Settings/config management

---

## Rust Core Library Structure

```
src/
├── lib.rs              # Public API re-exports
├── game/
│   ├── mod.rs          # Game trait definition
│   ├── kuhn.rs         # Kuhn Poker implementation
│   └── hunl_preflop.rs # HUNL Preflop implementation
├── cfr/
│   ├── mod.rs          # Solver orchestration
│   ├── vanilla.rs      # CPU recursive CFR (reference impl)
│   ├── batched.rs      # GPU batched CFR+ via Burn
│   └── regret.rs       # Regret matching utilities
├── core/
│   ├── tensors.rs      # Game tree → tensor compilation
│   ├── exploitability.rs
│   ├── hands.rs        # 169 canonical hands
│   ├── equity.rs       # Preflop equity tables
│   └── device.rs       # Burn device selection
├── config/
│   ├── mod.rs          # Config types and loading
│   └── presets/        # YAML files (standard.yaml, etc.)
├── strategy/
│   ├── mod.rs          # Strategy types
│   └── persistence.rs  # Save/load (MessagePack)
└── error.rs            # Custom error types with thiserror
```

### Key Traits

```rust
pub trait Game: Send + Sync {
    type State: Clone + Send;

    fn initial_states(&self) -> Vec<Self::State>;
    fn is_terminal(&self, state: &Self::State) -> bool;
    fn player(&self, state: &Self::State) -> Player;
    fn actions(&self, state: &Self::State) -> Vec<Action>;
    fn next_state(&self, state: &Self::State, action: Action) -> Self::State;
    fn utility(&self, state: &Self::State, player: Player) -> f64;
    fn info_set_key(&self, state: &Self::State) -> String;
}
```

---

## Burn Integration for GPU Acceleration

### Device Abstraction

```rust
pub enum Device {
    Cpu,
    Cuda(usize),    // GPU index
    Metal,          // Apple Silicon
}

impl Device {
    pub fn auto() -> Self {
        // Prefer CUDA → Metal → CPU
    }
}
```

Supported backends:
- `burn-cuda` for NVIDIA GPUs
- `burn-wgpu` for Metal/MPS (Apple Silicon)
- `burn-ndarray` for CPU fallback

### Tensor-Based Game Tree

```rust
pub struct CompiledGame<B: Backend> {
    pub node_player: Tensor<B, 1, Int>,      // [num_nodes]
    pub node_info_set: Tensor<B, 1, Int>,    // [num_nodes]
    pub node_num_actions: Tensor<B, 1, Int>, // [num_nodes]
    pub action_child: Tensor<B, 2, Int>,     // [num_nodes, max_actions]
    pub action_mask: Tensor<B, 2, Bool>,     // [num_nodes, max_actions]
    pub terminal_mask: Tensor<B, 1, Bool>,   // [num_nodes]
    pub terminal_utils: Tensor<B, 2>,        // [num_nodes, 2] - per player
    pub info_set_to_idx: HashMap<String, usize>,
}
```

### Batched CFR+ Core

```rust
pub struct BatchedCfr<B: Backend> {
    regret_sum: Tensor<B, 2>,     // [num_info_sets, max_actions]
    strategy_sum: Tensor<B, 2>,   // [num_info_sets, max_actions]
}

impl<B: Backend> BatchedCfr<B> {
    pub fn iterate(&mut self, game: &CompiledGame<B>, batch_size: usize) {
        // 1. Compute reach probabilities (top-down)
        // 2. Compute utilities at terminals
        // 3. Propagate counterfactual values (bottom-up)
        // 4. Update regrets with CFR+ flooring (max with 0)
        // 5. Accumulate strategy weighted by reach
    }
}
```

---

## Tauri Commands and IPC

### Training Commands

```rust
#[tauri::command]
async fn start_training(
    config_path: String,
    iterations: u64,
    batch_size: usize,
    device: String,        // "auto", "cpu", "cuda", "metal"
    stack_depths: Vec<u32>,
    state: State<AppState>,
) -> Result<(), String>;

#[tauri::command]
async fn stop_training(state: State<AppState>) -> Result<(), String>;

#[tauri::command]
fn get_training_status(state: State<AppState>) -> TrainingStatus;
```

### TrainingStatus

```rust
pub struct TrainingStatus {
    pub running: bool,
    pub current_iteration: u64,
    pub total_iterations: u64,
    pub current_stack: u32,
    pub exploitability: f64,
    pub iterations_per_second: f64,
    pub history: Vec<ProgressPoint>,  // For charting
}
```

### Strategy Exploration Commands

```rust
#[tauri::command]
fn get_strategy_matrix(
    stack: u32,
    history: Vec<String>,
    state: State<AppState>,
) -> Result<StrategyMatrix, String>;

#[tauri::command]
fn get_available_actions(
    stack: u32,
    history: Vec<String>,
    state: State<AppState>,
) -> Result<Vec<ActionInfo>, String>;
```

### StrategyMatrix

```rust
pub struct StrategyMatrix {
    pub cells: [[CellStrategy; 13]; 13],
    pub position: String,
    pub history_display: String,
}

pub struct CellStrategy {
    pub hand: String,
    pub fold: f64,
    pub call: f64,
    pub raises: Vec<(f64, f64)>,  // (size, frequency)
    pub all_in: f64,
}
```

### Tauri Events (for real-time updates)

```rust
// Rust emits progress events
app_handle.emit_all("training-progress", &status)?;
```

```typescript
// Frontend listens
listen('training-progress', (event) => {
  setProgress(event.payload);
});
```

---

## React Frontend Structure

```
src/
├── main.tsx
├── App.tsx
├── components/
│   ├── layout/
│   │   ├── Sidebar.tsx
│   │   └── Header.tsx
│   ├── training/
│   │   ├── TrainingPanel.tsx
│   │   ├── ConfigSelector.tsx
│   │   ├── ParameterForm.tsx
│   │   ├── StackSelector.tsx
│   │   ├── ProgressDisplay.tsx
│   │   └── ConvergenceChart.tsx
│   ├── explorer/
│   │   ├── ExplorerPanel.tsx
│   │   ├── HandMatrix.tsx
│   │   ├── MatrixCell.tsx
│   │   ├── ActionBlockStrip.tsx
│   │   ├── ActionBlock.tsx
│   │   └── StackSwitcher.tsx
│   └── common/
│       ├── Button.tsx
│       ├── Select.tsx
│       └── Tooltip.tsx
├── hooks/
│   ├── useTraining.ts
│   ├── useStrategy.ts
│   └── useTauri.ts
├── types/
│   └── index.ts
└── styles/
    └── index.css
```

### Main Views

1. **Training** - Configure, run, monitor training with live charts
2. **Explorer** - Navigate game tree, view hand matrix at each node
3. **Settings** - Device preferences, theme, file paths

---

## Visualizations

### Hand Matrix (13×13)

**Layout:**
- Diagonal: Pocket pairs (AA → 22)
- Upper triangle: Suited hands
- Lower triangle: Offsuit hands
- No external row/column headers

**Cell Design:**
- Hand label in upper-left corner (white text)
- Entire cell background is a stacked horizontal frequency bar
- Colors:
  - **Red shades** → Raises (lighter = smaller, darker = larger, darkest = all-in)
  - **Green** → Calls/checks
  - **Blue** → Folds

**Interactions:**
- Hover → Tooltip with exact percentages
- Click → Navigate or show action picker

### Training Analytics

**Convergence Chart:**
- X-axis: Iteration count
- Y-axis: Exploitability (log scale)
- One line per stack depth (if multi-stack)
- Updates live during training via Tauri events

**Progress Display:**
- Current iteration / total iterations
- Current stack depth
- Iterations per second
- Elapsed time

**Chart Library:** recharts

---

## Strategy Explorer Navigation

Horizontal strip of action blocks above the matrix, flowing left-to-right:

```
┌────────────┬────────────┬────────────┬────────────┐
│ SB    24.5 │ BB      24 │ SB      23 │ BB      20 │ ← Current
├────────────┼────────────┼────────────┼────────────┤
│ Fold       │ Fold       │ Fold       │ Fold       │
│ Call       │ Call       │ [Call]     │ Call       │
│ [Raise 2]  │ [Raise 5]  │ Allin 25   │ Raise 5.5  │
│ Allin 25   │ Raise 8    │            │ Raise 8.5  │
│            │ Allin 25   │            │ Allin 20   │
└────────────┴────────────┴────────────┴────────────┘
```

**Block Structure:**
- Header: Position (SB/BB) + stack remaining
- Action list: Available actions at that node
- Selected action: Highlighted (shows path taken)
- Current block: Rightmost, bordered - choices for matrix below

**Behavior:**
- Click action in current block → matrix updates, new block appears
- Click action in earlier block → rewinds to that point
- Scroll horizontally if many actions

**Special Blocks:**
- Street transitions (FLOP/TURN/RIVER) show community cards with suit colors

---

## Configuration System

### Config File Format

```yaml
name: "Standard 100BB"
stack_depths: [100]
raise_sizes: [2.5, 3, 6, 8, 10, 15, 20, 25, 50, 100]
max_bets_per_round: 4
```

### Multi-Stack Config

```yaml
name: "Multi-Stack Tournament"
stack_depths: [20, 35, 50, 75, 100]
raise_sizes: [2, 2.5, 3, 4, 5, 6, 8, 10, 15, 20]
max_bets_per_round: 3
```

### Rust Types

```rust
#[derive(Deserialize)]
pub struct Config {
    pub name: String,
    pub stack_depths: Vec<u32>,
    pub raise_sizes: Vec<f64>,
    pub max_bets_per_round: u8,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self, ConfigError>;

    pub fn get_legal_raise_sizes(
        &self,
        current_bet: f64,
        stack: f64,
        bets_this_round: u8,
    ) -> Vec<f64> {
        if bets_this_round + 1 >= self.max_bets_per_round {
            // At max bets - all-in only
            vec![stack]
        } else {
            // Filter raise_sizes to valid amounts
            self.raise_sizes
                .iter()
                .filter(|&size| *size > current_bet && *size <= stack)
                .copied()
                .collect()
        }
    }
}
```

---

## Strategy Persistence

### File Format

MessagePack (`.mpk`) - compact binary, fast serialization.

### Strategy Structure

```rust
#[derive(Serialize, Deserialize)]
pub struct TrainedStrategy {
    pub game_type: String,
    pub config_name: String,
    pub stack_depths: Vec<u32>,
    pub iterations: u64,
    pub exploitability: HashMap<u32, f64>,
    pub strategies: HashMap<u32, StackStrategy>,
}

#[derive(Serialize, Deserialize)]
pub struct StackStrategy {
    pub info_sets: HashMap<String, Vec<f64>>,
}
```

### Metadata (returned on load)

```rust
pub struct StrategyMetadata {
    pub game_type: String,
    pub config_name: String,
    pub stack_depths: Vec<u32>,
    pub iterations: u64,
    pub exploitability: HashMap<u32, f64>,
}
```

---

## Error Handling

### Custom Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum SolverError {
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Invalid game state: {0}")]
    InvalidState(String),

    #[error("Device error: {0}")]
    Device(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] rmp_serde::encode::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("File not found: {0}")]
    NotFound(PathBuf),

    #[error("Invalid YAML: {0}")]
    Parse(#[from] serde_yaml::Error),

    #[error("Invalid raise size: {0}")]
    InvalidRaiseSize(f64),

    #[error("Empty stack depths")]
    EmptyStackDepths,
}
```

### Frontend Error Display

- Toast notifications for transient errors
- Modal dialogs for critical errors
- Inline validation for form inputs

---

## Implementation Phases

### Phase 1: Rust Core Foundation
- Project setup (Cargo workspace, dependencies)
- Game trait and Kuhn Poker implementation
- Vanilla CFR (CPU, reference implementation)
- Basic tests verifying Kuhn Nash equilibrium

### Phase 2: GPU Acceleration
- Burn integration with multiple backends:
  - `burn-cuda` for NVIDIA GPUs
  - `burn-wgpu` for Metal/MPS (Apple Silicon)
  - `burn-ndarray` for CPU fallback
- Device abstraction with auto-detection (CUDA → Metal → CPU)
- Game tree tensor compilation
- Batched CFR+ implementation
- Benchmarks comparing CPU vs CUDA vs Metal

### Phase 3: HUNL Preflop
- 169 canonical hands module
- Config loading from YAML
- HUNL Preflop game implementation
- Preflop equity tables
- Exploitability calculation

### Phase 4: Tauri Shell
- Tauri project setup with React frontend
- Basic IPC commands wired up
- Training start/stop/status flow
- Strategy save/load with file dialogs

### Phase 5: Training UI
- Config selector dropdown
- Parameter form (iterations, batch size, device)
- Stack depth selection
- Progress display and convergence chart

### Phase 6: Strategy Explorer
- Hand matrix component
- Action block navigation strip
- Stack switcher
- Hover tooltips and hand details

### Phase 7: Polish
- Error handling and user feedback
- Keyboard shortcuts
- Dark/light theme
- Performance optimization

---

## Dependencies and Tech Stack

### Rust Core (Cargo.toml)

```toml
[dependencies]
# ML / Tensors
burn = { version = "0.15", features = ["wgpu", "cuda", "ndarray"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_yaml = "0.9"
rmp-serde = "1.1"

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Utilities
rayon = "1.8"
indexmap = "2.0"
```

### Tauri App (Cargo.toml)

```toml
[dependencies]
tauri = { version = "2.0", features = ["dialog"] }
poker-solver-core = { path = "../core" }
tokio = { version = "1", features = ["full"] }
```

### Frontend (package.json)

```json
{
  "dependencies": {
    "react": "^18",
    "react-dom": "^18",
    "@tauri-apps/api": "^2.0",
    "recharts": "^2.12",
    "tailwindcss": "^3.4"
  },
  "devDependencies": {
    "typescript": "^5.3",
    "vite": "^5.0",
    "@vitejs/plugin-react": "^4.2"
  }
}
```

### Project Structure

```
poker-solver/
├── Cargo.toml              # Workspace root
├── crates/
│   ├── core/               # Solver library
│   └── tauri-app/          # Tauri binary
└── frontend/               # React app
```

---

## Testing Strategy

### Unit Tests

```rust
// game/kuhn.rs
#[cfg(test)]
mod tests {
    #[test]
    fn initial_states_returns_six_deals() { ... }

    #[test]
    fn terminal_after_two_passes() { ... }

    #[test]
    fn bet_fold_gives_pot_to_bettor() { ... }
}
```

### Integration Tests

```rust
// tests/kuhn_convergence.rs
#[test]
fn kuhn_reaches_nash_equilibrium() {
    let game = KuhnPoker::new();
    let mut solver = VanillaCfr::new(&game);
    solver.train(10_000);

    let exploitability = solver.exploitability();
    assert!(exploitability < 0.001);
}

// tests/gpu_cpu_equivalence.rs
#[test]
fn batched_cfr_matches_vanilla_on_kuhn() {
    // Same strategy after N iterations on both backends
}
```

### Property-Based Tests

```rust
proptest! {
    #[test]
    fn strategy_probabilities_sum_to_one(info_set in any_info_set()) {
        let probs = solver.get_strategy(&info_set);
        let sum: f64 = probs.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pot_never_negative(actions in valid_action_sequence()) {
        let state = apply_actions(actions);
        prop_assert!(state.pot >= 0);
    }
}
```

### CI Pipeline

- `cargo test` on every PR
- `cargo clippy` with warnings as errors
- `cargo fmt --check`

---

## Summary

| Aspect | Decision |
|--------|----------|
| **Goal** | Same CFR solver core, enhanced Tauri desktop UI |
| **Frontend** | React + TypeScript |
| **ML Backend** | Burn (CUDA + Metal + CPU) |
| **UI Type** | Full workbench - training + exploration |
| **Training Controls** | Intermediate (key params exposed, config files for game setup) |
| **Visualizations** | Both hand matrix AND training analytics |
| **Multi-stack** | Yes, train and explore across multiple stack depths |
| **Hand Matrix** | 13×13 grid, hand label upper-left, full-cell frequency bars |
| **Color Scheme** | Red=raise (darker=larger), Green=call, Blue=fold |
| **Navigation** | Horizontal action block strip above matrix |
| **Config** | YAML with name, stack_depths, raise_sizes, max_bets_per_round |
| **Persistence** | MessagePack (.mpk) files |
| **Phases** | 7 phases from core → GPU → HUNL → Tauri → UI → Polish |
