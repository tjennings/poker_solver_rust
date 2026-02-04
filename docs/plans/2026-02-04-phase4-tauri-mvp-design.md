# Phase 4: Tauri Shell MVP Design

**Date:** 2026-02-04
**Goal:** Minimal Tauri app proving Rust ↔ React pipeline works end-to-end

---

## Scope

MVP with one working command that runs actual training:

- `run_kuhn_training(iterations)` - Trains MCCFR on Kuhn Poker, returns strategies
- Simple React UI with input, button, results display
- No async/background tasks (Kuhn is fast enough)
- No styling framework yet

---

## Project Structure

```
poker-solver/
├── Cargo.toml                 # Add tauri-app to workspace
├── crates/
│   ├── core/                  # Existing solver library
│   └── tauri-app/             # Tauri backend
│       ├── Cargo.toml
│       ├── src/
│       │   ├── main.rs        # Tauri entry point
│       │   └── commands.rs    # IPC command handlers
│       ├── tauri.conf.json    # Tauri config
│       └── capabilities/      # Tauri v2 permissions
└── frontend/                  # React app
    ├── package.json
    ├── vite.config.ts
    ├── tsconfig.json
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx
        └── types.ts
```

---

## IPC Command

### Rust (`commands.rs`)

```rust
use poker_solver_core::cfr::MccfrSolver;
use poker_solver_core::game::KuhnPoker;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize)]
pub struct TrainingResult {
    pub iterations: u64,
    pub strategies: HashMap<String, Vec<f64>>,
    pub elapsed_ms: u64,
}

#[tauri::command]
pub fn run_kuhn_training(iterations: u64) -> Result<TrainingResult, String> {
    let start = std::time::Instant::now();

    let game = KuhnPoker::new();
    let mut solver = MccfrSolver::new(game);
    solver.train_full(iterations);

    Ok(TrainingResult {
        iterations: solver.iterations(),
        strategies: solver.all_strategies(),
        elapsed_ms: start.elapsed().as_millis() as u64,
    })
}
```

### TypeScript

```typescript
import { invoke } from '@tauri-apps/api/core';

interface TrainingResult {
  iterations: number;
  strategies: Record<string, number[]>;
  elapsed_ms: number;
}

const result = await invoke<TrainingResult>('run_kuhn_training', {
  iterations: 1000
});
```

---

## React UI

Single-page app with:

1. Input field for iteration count
2. Train button that calls the command
3. Results display showing elapsed time and strategy table

```tsx
function App() {
  const [iterations, setIterations] = useState(1000);
  const [result, setResult] = useState<TrainingResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleTrain = async () => {
    setLoading(true);
    const res = await invoke('run_kuhn_training', { iterations });
    setResult(res);
    setLoading(false);
  };

  return (
    <div>
      <h1>Poker Solver</h1>
      <input value={iterations} onChange={...} type="number" />
      <button onClick={handleTrain} disabled={loading}>
        {loading ? 'Training...' : 'Train Kuhn Poker'}
      </button>
      {result && <StrategyTable strategies={result.strategies} />}
    </div>
  );
}
```

---

## Dependencies

### Tauri (`crates/tauri-app/Cargo.toml`)

```toml
[dependencies]
tauri = { version = "2", features = ["devtools"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
poker-solver-core = { path = "../core" }
```

### Frontend (`frontend/package.json`)

```json
{
  "dependencies": {
    "react": "^18",
    "react-dom": "^18",
    "@tauri-apps/api": "^2"
  },
  "devDependencies": {
    "typescript": "^5",
    "vite": "^5",
    "@vitejs/plugin-react": "^4"
  }
}
```

---

## Success Criteria

1. `cargo tauri dev` launches the app
2. Entering iterations and clicking "Train" calls Rust solver
3. Results display shows strategies from trained solver
4. No crashes or IPC errors
