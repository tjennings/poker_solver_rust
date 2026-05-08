---
# poker_solver_rust-o6oc
title: Add Direct CFVNet legacy normalized conversion
status: completed
type: bug
priority: high
created_at: 2026-05-08T20:20:35Z
updated_at: 2026-05-08T20:31:39Z
---

Current Direct CFVNet checkpoint was initially assumed to output normalized chip_ev/(pot+stack) values. Follow-up audit found the Python-exported checkpoint target is scaled bcfv: `bcfv * pot / (pot + stack)`. Track the temporary compatibility mode separately from native `direct` mode for future models trained to solver bcfv units.

- [x] Verify baseline full test suite is clean and under one minute
- [x] Patch Direct evaluator conversion for current normalized checkpoint
- [x] Update tests and docs for temporary Direct compatibility
- [x] Run focused and full verification
- [x] Commit code and bean

## Summary of Changes

Added `direct_normalized_legacy` compatibility mode. It was initially wired for the current Direct checkpoint and the UI Direct CFVNet option now sends this mode; `direct` remains native bcfv for future checkpoints. Follow-up bean `poker_solver_rust-we1x` corrected the conversion after verifying the Python-exported checkpoint target is scaled bcfv, not chip_ev/(pot+stack).
