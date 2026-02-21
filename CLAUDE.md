# Rust Project Guidelines

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
