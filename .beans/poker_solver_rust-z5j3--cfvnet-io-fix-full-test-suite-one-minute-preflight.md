---
# poker_solver_rust-z5j3
title: 'CFVNet IO: fix full test suite one-minute preflight'
status: todo
type: bug
priority: deferred
created_at: 2026-05-14T01:14:26Z
updated_at: 2026-05-14T01:18:09Z
parent: poker_solver_rust-8e9f
---

Repository preflight requires the full test suite to pass in under one minute. Current observed command timeout 60 cargo test exceeded 60 seconds before completion, though shown tests were passing.

- [ ] Identify slow crates/tests in full cargo test
- [ ] Split, gate, optimize, or mark long tests so standard full suite completes under 60 seconds
- [ ] Verify timeout 60 cargo test exits successfully

This blocks fully compliant implementation work under the local AGENTS.md rules.

## Deferral Note

User confirmed the suite is slow because another solve is running and asked to ignore test timing for now. Leaving this as deferred follow-up rather than blocking CFVNet IO implementation.
