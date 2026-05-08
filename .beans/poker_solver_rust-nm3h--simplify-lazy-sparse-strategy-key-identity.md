---
# poker_solver_rust-nm3h
title: Simplify lazy sparse strategy key identity
status: in-progress
type: task
priority: high
created_at: 2026-05-08T02:21:29Z
updated_at: 2026-05-08T02:21:29Z
---

Audit and fix lazy sparse MP strategy keying so storage identity only includes dimensions needed for strategic information sets. Evaluate whether street and SPR bucket are redundant when bucket identity already encodes street/hand abstraction, then update storage, tests, telemetry/docs as needed.
