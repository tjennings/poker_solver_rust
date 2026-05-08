---
# poker_solver_rust-nm3h
title: Simplify lazy sparse strategy key identity
status: completed
type: task
priority: high
created_at: 2026-05-08T02:21:29Z
updated_at: 2026-05-08T02:28:51Z
---

Audit and fix lazy sparse MP strategy keying so storage identity only includes dimensions needed for strategic information sets. Evaluate whether street and SPR bucket are redundant when bucket identity already encodes street/hand abstraction, then update storage, tests, telemetry/docs as needed.

## Summary of Changes

- Changed lazy sparse MP infoset identity from seat + street + bucket + SPR bucket + action history to seat + street-namespaced bucket + action history.
- Encoded street into the high bits of the sparse bucket id so bucket identity implies street without a separate key field.
- Removed SPR bucket from sparse storage identity and insert attribution telemetry to avoid fragmenting strategy rows by stack/pot ratio.
- Kept action history in the key because lazy storage needs it as the public-node equivalent for CFR information sets.
- Updated tests and docs for the revised key shape.
