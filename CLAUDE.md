# Rust Project Guidelines

## Error Handling

**Never use `.unwrap()` or `.expect()` in library code.** These panic on failure and crash the program.

- Use `?` operator to propagate errors
- Return `Result<T, E>` from fallible functions
- Use `thiserror` for custom error types with meaningful context
- Reserve `.unwrap()` only for:
  - Tests (where panics are acceptable)
  - Cases proven infallible by prior validation (add a comment explaining why)

```rust
// Bad
let file = File::open(path).unwrap();

// Good
let file = File::open(path).context("failed to open config file")?;
```

## Type System & Safety

- Prefer **newtypes** over primitive types for domain concepts: `struct PlayerId(u32)` not `u32`
- Use **enums** to make illegal states unrepresentable
- Leverage the type system to enforce invariants at compile time
- Avoid `unsafe` unless absolutely necessary; document why and ensure soundness

## Immutability & Pure Functions

- **Default to immutable**: use `let` not `let mut`
- Prefer **pure functions** that take inputs and return outputs without side effects
- Isolate side effects (I/O, randomness, time) at the boundaries of the system
- Use **iterators and combinators** (`map`, `filter`, `fold`) over manual loops with mutation
- Return new values instead of mutating in place when practical

```rust
// Prefer this
fn calculate_equity(hand: &Hand, board: &Board) -> f64 { ... }

// Over this
fn calculate_equity(hand: &Hand, board: &Board, result: &mut f64) { ... }
```

## Domain-Driven Design

- Model the **domain** with types that reflect domain concepts
- Use **modules** to establish bounded contexts
- Keep domain logic **pure** and separate from infrastructure (I/O, persistence, networking)
- Name types, functions, and variables using **ubiquitous language** from the domain
- Validate at the boundaries; trust internal types

```rust
// Domain types enforce invariants
pub struct BetSize(u32);  // Always positive, validated on construction

impl BetSize {
    pub fn new(chips: u32) -> Result<Self, InvalidBetError> {
        if chips == 0 { return Err(InvalidBetError::Zero); }
        Ok(Self(chips))
    }
}
```

## Code Organization

- One concept per file; keep files focused and small
- Use `mod.rs` or `module_name.rs` to re-export public API
- Separate **data structures** from **behavior** when it aids clarity
- Group related functionality into coherent modules
- modules should expose one or more public functions as their API.  These are "orchestration" functions that internally use many small functions to accomplish their work.
- Functions must be less than 50 lines of code, preferrably less than 10 lines of code.  
- Functions must have a single responsibility, be pure (no side effects), and be easily composable.

## Testing Philosophy

### Test-Driven Development (TDD)

1. **Red**: Write a failing test that defines expected behavior
2. **Green**: Write the minimum code to make it pass
3. **Refactor**: Clean up while keeping tests green.  Eliminate duplication, keep the code DRY, evaluate the code for adhereance to best practices. 
4. **Repeate**: Write the next most obvious test, go to step 1.  

Never write implementation code without a failing test first.

### Test Characteristics

- **Fast**: Tests should run in milliseconds. Mock or stub slow dependencies.
- **Isolated**: Each test verifies one behavior. No test depends on another.
- **Repeatable**: No flakiness. Control randomness with seeded RNGs.
- **Self-documenting**: Test names describe the behavior being verified.
- **Clear-errors**: failing tests should provide as much useful data as possible to support debugging. 

```rust
#[test]
fn fold_action_ends_hand_immediately() { ... }

#[test]
fn all_in_bet_uses_entire_stack() { ... }

#[test]
fn calling_larger_bet_than_stack_goes_all_in() { ... }
```

### Test Organization

- Place unit tests in a `#[cfg(test)]` module at the bottom of each file
- Use `tests/` directory for integration tests
- Prefer **many small tests** over few large tests
- Use **proptest** for property-based testing of invariants
- Use **fuzzing** tests to test expected invariants 

### What to Test

- **Happy paths**: Normal expected behavior
- **Edge cases**: Empty inputs, boundaries, limits
- **Error cases**: Invalid inputs produce appropriate errors
- **Invariants**: Properties that must always hold (use proptest)

```rust
proptest! {
    #[test]
    fn pot_never_negative(actions in any_action_sequence()) {
        let game = simulate(actions);
        prop_assert!(game.pot() >= 0);
    }
}
```

## Performance Considerations

- **Measure before optimizing**: Use benchmarks, not intuition
- Prefer `&str` over `String` for read-only string data
- Use `Cow<'_, str>` when you might need to own or borrow
- Consider `SmallVec` or `ArrayVec` for small, fixed-size collections
- Use `#[inline]` sparingly and only with benchmark evidence

## Documentation

- Document **public API** with `///` doc comments
- Explain **why**, not **what** (the code shows what)
- Include examples in doc comments for complex functions
- Use `//` comments sparingly for non-obvious implementation details

## Dependency Management

- Prefer well-maintained crates with minimal transitive dependencies
- Pin versions in `Cargo.toml` for reproducible builds
- Audit dependencies periodically for security issues
- Feature-flag optional heavy dependencies

## Formatting & Linting

- Run `cargo fmt` before committing
- Run `cargo clippy` and address all warnings
- Configure clippy to deny warnings in CI:
  ```rust
  #![deny(clippy::all)]
  #![warn(clippy::pedantic)]
  ```

## Common Patterns

### Builder Pattern for Complex Construction

```rust
let config = ConfigBuilder::new()
    .iterations(1000)
    .learning_rate(0.001)
    .build()?;
```

### Typestate Pattern for State Machines

```rust
struct Game<S: GameState> { ... }
struct Preflop;
struct Flop;

impl Game<Preflop> {
    fn deal_flop(self) -> Game<Flop> { ... }
}
```

### From/Into for Type Conversions

```rust
impl From<RawAction> for Action {
    fn from(raw: RawAction) -> Self { ... }
}
```

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

- Write clear commit messages describing **what** and **why**
- Keep commits atomic: one logical change per commit
- Run tests before pushing: `cargo test`
- Use feature branches for non-trivial changes
