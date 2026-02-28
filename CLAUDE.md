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

### Dispatch Rules

- Algorithm/design questions, game theory, poker AI → `ml-researcher` agent (read-only advisor)
- Implementation tasks → `rust-developer` agent (use worktree isolation)
- Architecture review → `software-architect` agent
- Code quality review → `idiomatic-rust-enforcer` agent
- Performance review → `rust-perf-reviewer` agent
- Plan verification → `code-reviewer` agent
- Independent tasks → dispatch in parallel via `dispatching-parallel-agents`
- Multi-step plans → use `subagent-driven-development`

### What the manager does directly

- Clarifies requirements with the user
- Creates and decomposes plans (via skills)
- Dispatches agents and synthesizes their results
- Manages beans lifecycle
- Commits, manages git, creates PRs
- Updates CLAUDE.md and documentation

## **REQUIRED** Implementation Workflow
Every time I ask for a change to the system, you MUST dispatch this workflow: 

1. **Research** — dispatch `ml-researcher` for algorithm/design questions
2. **Brainstorm** — invoke `superpowers:brainstorming` to explore requirements 
    1. Brainstorming MUST use the ml-researcher and/or software-architect for the appropriate context. 
3. **Plan** — invoke `superpowers:writing-plans` to create a structured plan 
    1. Plans MUST include a "Agent Team & Execution Order" section breaking down which agents are owning what work, and what will be done in parallel
4. **Dispatch** — `rust-developer` agents in parallel (via `superpowers:subagent-driven-development`)
5. **Review** — `software-architect` + `idiomatic-rust-enforcer` + `rust-perf-reviewer`
    1. The review process must be done TWICE. 
6. **Verify** — invoke `superpowers:verification-before-completion` before claiming done
7. **Finish** — invoke `superpowers:finishing-a-development-branch` to merge/PR

## Git Workflow

- Always use worktrees for implementing plans
- Write clear commit messages describing **what** and **why**
- Keep commits atomic: one logical change per commit
- Run tests before pushing: `cargo test`
- Use feature branches for non-trivial changes

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

## Code Search

**REQUIRED**: Use `ast-grep` for Rust code pattern searches (function calls, struct definitions, impl blocks, etc.):
- `ast-grep run --pattern '<pattern>' --lang rust .` for simple pattern matches
- `ast-grep scan --rule <rule.yml> .` for complex structural queries
- Metavariables: `$NAME` (single node), `$$$ARGS` (multiple nodes)
- **Limitation**: ast-grep cannot match Rust macro invocations (`println!`, `vec!`, `assert!`, etc.) — use `Grep` for these
- Fall back to `Grep` for non-structural text search (comments, strings, config files), macro invocations, or if ast-grep is unavailable

Examples (verified working):
```bash
ast-grep run --pattern 'impl $TRAIT for $TYPE { $$$BODY }' --lang rust .  # find trait impls
ast-grep run --pattern 'fn $NAME($$$PARAMS) -> Result<$$$RET> { $$$BODY }' --lang rust .  # find fallible functions
```

## Build & Test

```bash
cargo test                                          # all tests
cargo test -p poker-solver-core                     # core only
cargo test -p poker-solver-trainer                  # trainer only
cargo clippy                                        # lint (pedantic enabled in core)
cargo run -p poker-solver-trainer --release -- <subcommand>  # always --release for training/diag
```

**Known timer failures** (pre-existing, not bugs): `blueprint/subgame_cfr`, `preflop/bundle`

## Crate Map

| Crate | Purpose |
|-|-|
| `core` | Preflop solver, postflop pipeline, CFR utilities, hand eval, abstractions |
| `trainer` | CLI: preflop/postflop solving, diagnostics |
| `tauri-app` | Desktop GUI exploration app |
| `devserver` | HTTP mirror of Tauri API for browser debugging |
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