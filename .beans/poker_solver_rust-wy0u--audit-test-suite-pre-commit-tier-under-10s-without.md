---
# poker_solver_rust-wy0u
title: 'Audit test suite: pre-commit tier under 10s without losing coverage'
status: todo
type: task
priority: high
created_at: 2026-04-19T03:44:01Z
updated_at: 2026-04-19T03:44:01Z
---

Audit the full test suite, classify every test as fast-unit / slow-unit / integration, and drive the **pre-commit (non-integration) suite under 10 seconds** without reducing coverage.

## Motivation

CLAUDE.md requires `cargo test` to finish in < 60s; the current reality is ~4:10 on macOS. A separate bean (`poker_solver_rust-v55b`) covers feature-gating GPU failures. This bean targets the deeper problem: a bloated fast-path suite. The pre-commit hook should give a developer confidence in under 10 seconds so it runs on *every* commit; integration tests can stay in a separate gate (CI-only, nightly, `--release` bench, etc.).

**Primary objective:** fast pre-commit suite without losing coverage.
**Non-goal:** deleting tests we still rely on.

## Scope

All crates in the workspace:
- `core`, `trainer`, `range-solver`, `range-solver-compare`, `tauri-app`, `devserver`, `cfvnet`, `test-macros`, plus GPU crates.

Cover `#[test]`, `#[timed_test]`, integration tests (`tests/` directories), and doctests.

## Deliverables

1. **Test inventory** — a table (checked in as `docs/test-audit.md` or attached here) listing every test with:
   - crate, module path, test name
   - wall-clock duration (use `cargo test -- --nocapture --test-threads=1 -Z unstable-options --report-time` or `cargo nextest run --profile ci`)
   - classification: `fast-unit` (<50 ms), `slow-unit` (50 ms–1 s), `integration` (>1 s or hits disk/network/CUDA)
   - marker status: is it already `#[ignore]`? `#[cfg(feature=...)]`? under `tests/` dir? gated by env?
2. **Slow-test report** — every test > 100 ms ranked by duration, with a one-line hypothesis for *why* it's slow (fixture build, large solve, file IO, etc.).
3. **Per-test mitigation plan** — for each slow test, pick one of:
   - **Shrink fixture** (smaller game tree, fewer iterations, smaller runout set, `#[cfg(test)]` constants)
   - **Move to integration tier** (`tests/` directory or `#[ignore]` + nightly job, with a note on what it covers that unit tests don't)
   - **Gate behind feature flag** (e.g. `slow-tests`, `gpu`, `expensive`)
   - **Cache expensive setup** (`OnceLock`, `lazy_static` fixture) so cost is amortized across test binary
   - **Parallelize** (split large test into N cheaper assertions, or trust nextest)
   - **Delete** (only if coverage is strictly subsumed by another test)
   Every slow test must get one of these verdicts with rationale.
4. **Pre-commit command** — a single invocation (e.g. `cargo nextest run --profile=precommit -E 'not test(integration_)'` or a cargo alias) that runs the fast tier and completes in **< 10 s** on the user's macOS machine.
5. **CI split** — document what runs where: pre-commit (<10 s), push (<60 s), nightly (integration, CUDA, `range-solver-compare --release` identity tests).
6. **Coverage confirmation** — before and after the audit, run `cargo llvm-cov --workspace --summary-only` (or equivalent) on the fast tier and confirm line/region coverage does not drop. If a line is only exercised by an integration-tier test, note which tier now covers it.

## TODOs

- [ ] Decide on measurement tool (`cargo nextest` preferred for JSON output and per-test timing)
- [ ] Run full suite with timing, write inventory to `docs/test-audit.md`
- [ ] Identify and list every test > 100 ms
- [ ] For each slow test, pick a mitigation (shrink / move / gate / cache / parallelize / delete)
- [ ] Implement mitigations crate-by-crate (one PR per crate to keep diffs reviewable)
- [ ] Add a `precommit` nextest profile (or cargo alias) and wire it into the pre-commit hook
- [ ] Add a CI job spec (or document one) that runs the full integration tier on push/nightly
- [ ] Measure coverage before/after; confirm no net loss
- [ ] Update CLAUDE.md to document the three-tier gate (pre-commit <10 s, push <60 s, nightly integration)

## Known slow tests (seed list — not exhaustive)

From prior session evidence:
- `range-solver-compare` identity tests — only run with `--release`, already integration-tier
- `blueprint_mp::mccfr::tests::traverse_updates_strategy_sums` — currently failing, unrelated scoping
- `cargo test --all` full suite ≈ 4:10 wall-clock (macOS); gpu-range-solver crate has 34 failing tests on non-CUDA hosts (tracked separately under v55b)
- Per `feedback_debugging_session.md`: terminal payoff & equity matrix tests are heavy fixture builders — likely slow-unit candidates

The full audit should replace this seed list with measured numbers.

## Out of scope

- GPU / CUDA feature-gating (tracked in `poker_solver_rust-v55b`)
- Flaky-test fixes (unless uncovered during audit — file follow-up beans)
- Rewriting the blueprint MCCFR solver for testability (surgical changes only)

## Why now

Fast tests get run; slow tests get skipped. The current 4-minute suite means developers (and agents) avoid running it before commits, so regressions slip in. A <10 s gate changes the default behavior.
