# Rust Project Guidelines

## Issue Tracking and Task management 

**IMPORTANT**: before you do anything else, run the `beans prime` command and heed its output.


## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full solver architecture: preflop LCFR solver, postflop imperfect-recall abstraction pipeline, key control parameters, and caching.

**Keep it current:** When making changes to the solver pipeline, abstraction system, config parameters, or caching, update `docs/architecture.md` to reflect the new state.

## Training & CLI

See [`docs/training.md`](docs/training.md) for all CLI commands, config options, abstraction modes, solver backends, and postflop model presets.

**Keep it current:** When adding/changing CLI commands, config parameters, or training workflows, update `docs/training.md` to reflect the new state.

## Explorer

See [`docs/explorer.md`](docs/explorer.md) for the strategy explorer desktop app and dev server: loading strategies, browsing the game tree, API commands.

**Keep it current:** When adding/changing explorer commands, UI features, or API endpoints, update `docs/explorer.md` to reflect the new state.

## Cloud Compute

See [`docs/cloud.md`](docs/cloud.md) for the AWS compute CLI: setup, launching training jobs, monitoring, downloading models, and configuration.

**Keep it current:** When adding/changing cloud commands, AWS resources, configuration variables, or the job lifecycle, update `docs/cloud.md` to reflect the new state.

## Build & Test

```bash
cargo test                                          # all tests
cargo test -p poker-solver-core                     # core only
cargo test -p poker-solver-trainer                  # trainer only
cargo clippy                                        # lint (pedantic enabled in core)
cargo run -p poker-solver-trainer --release -- <subcommand>  # always --release for training/diag
```

**Known timer failures** (pre-existing, not bugs): `cfr/vanilla`, `cfr/exploitability`, `blueprint/subgame_cfr`, `preflop/bundle`

## Crate Map

| Crate | Purpose |
|-|-|
| `core` | Game logic, CFR solvers, preflop, hand eval, abstractions |
| `trainer` | CLI: training, diagnostics, deal generation |
| `deep-cfr` | Neural network SD-CFR (candle) |
| `gpu-cfr` | GPU sequence-form CFR (wgpu) |
| `tauri-app` | Desktop GUI exploration app |
| `devserver` | HTTP mirror of Tauri API for browser debugging |
| `test-macros` | `#[timed_test]` proc macro |

## Key Config Files

- Training configs: `sample_configurations/*.yaml`
- Agent configs: `agents/*.toml`

## Internal Units & Conventions

- SB=1, BB=2. `stack_depth` is in BB; stacks = `stack_depth * 2`
- `Action::Bet(u32)` / `Action::Raise(u32)` store index into bet_sizes, not chip amounts
- `ALL_IN = u32::MAX` sentinel for all-in actions

## Domain-Driven Design

- Model the **domain** with types that reflect domain concepts
- Use **modules** to establish bounded contexts
- Keep domain logic **pure** and separate from infrastructure (I/O, persistence, networking)
- Name types, functions, and variables using **ubiquitous language** from the domain
- Validate at the boundaries; trust internal types

## Frontend Debugging (Dev Server)

When debugging the Explorer UI, use the HTTP dev server instead of Tauri:

```bash
cargo run -p poker-solver-devserver &   # HTTP API on :3001
cd frontend && npm run dev              # Vite on :5173
# Open http://localhost:5173 in any browser
```

- The dev server mirrors all 14 Tauri exploration commands as `POST /api/{command_name}`
- The frontend auto-detects Tauri vs browser via `window.__TAURI__` and uses `fetch()` in browser mode
- File picker falls back to `window.prompt()` in browser — enter absolute paths
- Use this when you need to test UI changes without the full Tauri build cycle
- Test endpoints directly with curl: `curl -X POST http://localhost:3001/api/is_bundle_loaded -H 'Content-Type: application/json' -d '{}'`

**Key files:**
- Dev server: `crates/devserver/src/main.rs`
- Invoke wrapper: `frontend/src/invoke.ts`
- Core functions: `crates/tauri-app/src/exploration.rs` (`_core` suffix variants)

## Git Workflow
- Always use worktrees for implementing plans
- Write clear commit messages describing **what** and **why**
- Keep commits atomic: one logical change per commit
- Run tests before pushing: `cargo test`
- Use feature branches for non-trivial changes

## Implementation
- Always use agent teams to implement plans
- Always use TDD to implement features.  
- A rust architect who is responsible for the high level implementation
- A code reviewer who prodives feedback on each complete task.  Ensure tests are sufficient, ensure best rust best practices are followed. 
- Up to three developer agents who implement tasks
