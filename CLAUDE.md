# Rust Project Guidelines

## Issue Tracking and Task management 

**IMPORTANT**: before you do anything else, run the `beans prime` command and heed its output.
**REQUIRED**: Always `git commit` beans immediately after you write them.  

## **IMPORTANT** Session Behavior: Manager Mode

The default session acts as a **coordinator**, not an implementer. It must:

1. **Never write Rust code directly** — delegate all implementation to `rust-developer` agents
2. **Never review code directly** — delegate to review agents
3. **Always follow the pipeline**: research → brainstorm → plan → dispatch → review → integrate
4. **Use beans** to track every work item before dispatching

### What the manager does directly

- Clarifies requirements with the user
- Creates and decomposes plans (via skills)
- Dispatches agents and synthesizes their results
- Manages beans lifecycle
- Commits, manages git, creates PRs
- Updates CLAUDE.md and documentation

## **REQUIRED** Planning Workflow
Every time I ask for a change to the system, you MUST dispatch this workflow:

1. **Research** — dispatch `ml-researcher` for algorithm/design questions
2. **Brainstorm** — invoke `hex:brainstorming` to explore requirements 
    1. Brainstorming MUST use the ml-researcher and/or software-architect for the appropriate context.

## **REQUIRED** Software Development workflow - BEFORE you start work follow these rules

- Ensure the working tree is clean, if it isn't prompt the user to assist you in cleanup
- Run the entire test suite, it MUST complete in less than 1 minute.  If it does not, pause your current project and fix the tests

## **REQUIRED** Software Development workflow - AFTER code is complete
- Ensure the ENTIRE test suite passes. If not you MUST repair the tests
- Ensure the ENTIRE test suite runs in less than 1 minute.  If not you MUST fix it. 

## Git Workflow

- Always create a feature branch for the change
- Always use worktrees for each agent for implementing plans
- Write clear commit messages describing **what** and **why**
- Keep commits atomic: one logical change per commit
- Use feature branches for non-trivial changes

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full solver architecture: blueprint_v2 MCCFR solver, hand abstraction, range solver, and cfvnet.

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
cargo test -p range-solver                          # range solver only
cargo test -p range-solver-compare --release        # identity tests (needs --release)
cargo clippy                                        # lint (pedantic enabled in core)
cargo run -p poker-solver-trainer --release -- <subcommand>  # always --release for training/diag
```
## Crate Map

| Crate | Purpose |
|-|-|
| `core` | Blueprint_v2 MCCFR solver, CFR utilities, hand abstraction, hand eval |
| `trainer` | CLI: train-blueprint, range-solve, diagnostics |
| `range-solver` | Exact (no-abstraction) postflop DCFR solver for single-spot analysis |
| `range-solver-compare` | Identity test harness comparing range-solver against reference impl |
| `tauri-app` | Desktop GUI exploration app |
| `devserver` | HTTP mirror of Tauri API for browser debugging |
| `cfvnet` | Deep CFV network: river datagen, training (burn), evaluation |
| `test-macros` | `#[timed_test]` proc macro |

## Key Config Files

- Training configs: `sample_configurations/*.yaml`

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

## Design Context

### Users
Serious poker professionals seeking a competitive edge. They use this tool during study sessions to analyze GTO strategies, explore decision trees, and refine their game. The interface should evoke precision, analytical power, and confidence — like wielding a high-end instrument built for experts.

### Brand Personality
**Dark, focused, elite.** This is a premium tool for serious players, not a casual poker app. Every pixel should feel intentional and information-dense without being cluttered.

### Aesthetic Direction
- **Visual tone**: Dark, high-contrast, data-rich — inspired by trading terminals (Bloomberg, TradingView) and poker solvers (PioSOLVER, GTO Wizard)
- **Theme**: Dark mode only. Deep navy/purple backgrounds (#0f0f1e, #1a1a2e) with bright cyan (#00d9ff) as the primary accent
- **References**: Trading terminals for information density and professional feel; GTO Wizard/PioSOLVER for poker-domain conventions and familiar action color coding
- **Anti-references**: Casual gaming aesthetics, bright/playful color schemes, consumer-oriented "friendly" UIs

### Design Principles
1. **Information density over decoration** — Every element earns its space. Favor showing more data over whitespace or ornament.
2. **Domain-native conventions** — Use poker and solver UI conventions (action colors, matrix layouts, tree structures) that pros already know. Don't reinvent established patterns.
3. **Precision and clarity** — Numbers, probabilities, and strategies must be instantly readable. Use contrast, alignment, and tabular formatting to make data scannable.
4. **Progressive disclosure** — Show the essential view by default; reveal deeper analysis on interaction. Power is available but not overwhelming.
5. **Zero friction** — The interface should never slow down the analysis workflow. Fast interactions, minimal clicks, keyboard-friendly where possible.