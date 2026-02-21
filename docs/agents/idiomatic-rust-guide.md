# Idiomatic Rust: Code Review Guide

> Synthesized from [Rust Design Patterns](https://rust-unofficial.github.io/patterns/) and [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).

---

## 1. Ownership & Borrowing

### Use borrowed types for function arguments
Accept `&str` not `&String`, `&[T]` not `&Vec<T>`, `&T` not `&Box<T>`. Borrowed owned types add unnecessary indirection and prevent deref coercion from working.

```rust
// Bad: rejects &str, string slices, literals
fn process(s: &String) { ... }

// Good: accepts &String, &str, slices from split(), literals
fn process(s: &str) { ... }
```

### Never clone to satisfy the borrow checker
Cloning to fix borrow errors creates independent copies that silently diverge. Instead, restructure code to fix the ownership issue, or use `Rc<T>`/`Arc<T>` when shared ownership is genuinely needed. Run `cargo clippy` to catch unnecessary clones.

### Use `mem::take()` / `mem::replace()` to move out of mutable references
When transforming enum variants with owned data, swap with a default value instead of cloning:

```rust
// Bad: clones the string unnecessarily
let name_copy = name.clone();
*e = MyEnum::B { name: name_copy };

// Good: zero-cost swap with default
*e = MyEnum::B { name: mem::take(name) };
```

### Temporary mutability — rebind to freeze
After mutable setup, rebind to an immutable binding to prevent accidental mutation:

```rust
let mut data = get_vec();
data.sort();
let data = data; // now immutable
```

## 2. Type System

### Newtypes for domain concepts
Wrap primitives in tuple structs to prevent mixing semantically different values. Zero runtime cost, catches bugs at compile time:

```rust
struct Miles(f64);
struct Kilometres(f64);
// Can't accidentally add Miles + Kilometres
```

### Use enums to make illegal states unrepresentable
Model state machines and variants with enums rather than booleans or stringly-typed fields. Prefer exhaustive matching over wildcard arms.

### Builder pattern for complex construction
Rust lacks default arguments and function overloading. Use builders for types with many optional fields:

```rust
let config = ConfigBuilder::new()
    .timeout(Duration::from_secs(30))
    .retries(3)
    .build()?;
```

### Custom traits to simplify complex bounds
When function signatures accumulate verbose trait bounds, extract a named trait:

```rust
// Bad: repeated complex bounds
fn process<G: FnMut() -> Result<T, Error>, T: Display>(g: G) { ... }

// Good: meaningful abstraction
trait Getter { type Output: Display; fn get(&mut self) -> Result<Self::Output, Error>; }
fn process<G: Getter>(g: G) { ... }
```

## 3. Error Handling

### Propagate with `?`, never `unwrap()` in library code
Reserve `.unwrap()` for tests and provably infallible cases (with a comment). Return `Result<T, E>` from fallible functions. Use `thiserror` for custom error types.

### Return consumed arguments on error
If a function takes ownership and may fail, return the value inside the error so callers can retry without cloning:

```rust
pub fn send(value: String) -> Result<(), SendError> { ... }
struct SendError(String); // caller can recover the value
```

## 4. API Design

### Naming conventions
| Pattern | Convention | Example |
|---|---|---|
| Immutable access | `as_` | `as_str()`, `as_bytes()` |
| Expensive conversion | `to_` | `to_string()`, `to_vec()` |
| Ownership transfer | `into_` | `into_inner()`, `into_bytes()` |
| Iteration | `iter()`, `iter_mut()`, `into_iter()` | -- |
| Constructors | `new()`, `with_capacity()` | -- |
| Getters | no `get_` prefix | `fn name(&self) -> &str` |

### Implement standard traits eagerly
All public types should implement, where applicable: `Debug`, `Clone`, `Default`, `PartialEq`/`Eq`, `Hash`, `Display`, `Send`/`Sync`, `From`/`Into`, `Serialize`/`Deserialize`.

### `Default` trait alongside `new()`
Always implement `Default` when a sensible zero-argument constructor exists. Use `#[derive(Default)]` when all fields support it. This enables `Option::unwrap_or_default()` and struct update syntax.

### Functions should not take out-parameters
Return values instead of writing to `&mut` output parameters:

```rust
// Bad
fn compute(input: &Data, result: &mut Output) { ... }

// Good
fn compute(input: &Data) -> Output { ... }
```

### Caller decides allocation
Accept generic iterators/slices rather than forcing specific collection types. Let callers choose where data lives.

### Only smart pointers implement `Deref`
Never use `Deref` to simulate inheritance or add methods from an inner type. It surprises users and violates trait semantics.

## 5. Patterns

### RAII guards for resource management
Use `Drop` destructors to guarantee cleanup (locks, file handles, temp files). The guard pattern ensures resources are released regardless of exit path (early return, `?`, panic).

### Strategy pattern via traits or closures
Use traits for pluggable algorithms. Rust's trait system replaces the traditional OOP strategy pattern naturally -- closures work for simple cases.

### Compose structs to satisfy the borrow checker
When the borrow checker prevents simultaneous access to different fields, decompose into smaller structs that can be borrowed independently:

```rust
// Instead of one large struct where borrowing one field locks everything:
struct Database { connection: ConnectionString, timeout: Timeout, pool: PoolSize }
// Pass individual components to functions
fn configure(conn: &ConnectionString, timeout: &mut Timeout) { ... }
```

### Visitor/Fold for heterogeneous data
Use the visitor pattern to separate traversal from operations on ASTs or complex data structures. The fold pattern creates new structures rather than mutating.

## 6. Anti-Patterns to Flag

| Anti-Pattern | Problem | Fix |
|---|---|---|
| `.clone()` to fix borrow errors | Hides ownership bugs, divergent state | Restructure ownership, use `Rc`/`Arc` |
| `#![deny(warnings)]` in source | Breaks on new compiler versions, blocks clippy | Use `RUSTFLAGS="-D warnings"` in CI only |
| `Deref` for inheritance | Implicit, surprising method resolution | Explicit delegation or trait impl |
| `unwrap()`/`expect()` in lib code | Panics crash the program | Use `?` and `Result` |
| Boolean/string arguments | Unclear call sites | Newtypes or enums |
| Wildcard match on own enums | Silently ignores new variants | Exhaustive matching |
| God structs | Borrow checker fights, unclear ownership | Decompose into focused structs |

## 7. Code Style

- **`format!` for string building** -- prefer over manual `push_str` chains for readability; use `push`/`push_str` only in hot paths with pre-allocated capacity.
- **Iterators over manual loops** -- `map`, `filter`, `fold`, `collect` express intent more clearly and compose better.
- **`Option` as iterator** -- use `.extend(option)` and `.chain(option)` instead of `if let Some` wrappers.
- **Pass variables to closures explicitly** -- use a scope block to clone/transform specific captures rather than blanket `move`.
- **`#[non_exhaustive]`** -- use on public enums/structs that may grow, but sparingly (forces wildcard arms on consumers).
- **Contain `unsafe` in small modules** -- minimize audit surface; expose safe APIs from inner unsafe modules.
- **Prefer small, focused crates** -- one responsibility per crate; enables parallel compilation and reuse.

## 8. Review Checklist

When reviewing Rust code, verify:

- [ ] No `.unwrap()` / `.expect()` in non-test code without justification comment
- [ ] Functions accept borrowed types (`&str`, `&[T]`) not borrowed owned types (`&String`, `&Vec<T>`)
- [ ] Public types implement `Debug`, `Clone`, `Default`, `PartialEq` where sensible
- [ ] Error types are meaningful (not `Box<dyn Error>` everywhere); use `thiserror`
- [ ] No `.clone()` used as borrow-checker band-aid
- [ ] Domain concepts use newtypes, not raw primitives
- [ ] `Deref` only implemented on smart pointer types
- [ ] Functions return values rather than taking `&mut` out-parameters
- [ ] Naming follows `as_`/`to_`/`into_` conventions
- [ ] `Default` implemented alongside `new()` where applicable
- [ ] Match arms are exhaustive (no unnecessary wildcards on own types)
- [ ] `unsafe` blocks are minimal, documented, and isolated in dedicated modules
- [ ] Iterator combinators preferred over manual loops with mutation
- [ ] Complex trait bounds extracted into named traits
- [ ] Structs decomposed enough to avoid borrow checker fights
- [ ] `#![deny(warnings)]` not in source (use CI env var instead)
- [ ] Functions < 50 lines; single responsibility
- [ ] No string-typed or boolean-flag arguments where enums would be clearer
