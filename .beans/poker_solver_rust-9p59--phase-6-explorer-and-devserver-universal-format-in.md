---
# poker_solver_rust-9p59
title: 'Phase 6: Explorer and devserver universal format integration'
status: todo
type: task
priority: normal
created_at: 2026-06-09T18:40:56Z
updated_at: 2026-06-10T19:06:22Z
parent: poker_solver_rust-a29s
---

Teach Explorer/devserver bundle APIs about universal format metadata. Acceptance: snapshot listing reports format kind/player count; HU universal bundles load through existing views; MP bundles expose read-only bundle info and row lookup APIs before full MP browsing UI.

## Scope Note (2026-06-10 goal recap)

Explorer compatibility is THE hard requirement of the whole effort: the end state is the Explorer browsing universal bundles (including lazy sparse exports, which need real action descriptors — see the blocking bean on action identity in sparse rows). Legacy bundle browsing can be dropped once universal loading works.
