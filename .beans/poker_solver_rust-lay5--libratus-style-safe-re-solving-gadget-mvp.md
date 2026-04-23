---
# poker_solver_rust-lay5
title: Libratus-style safe re-solving gadget — MVP
status: completed
type: feature
priority: normal
created_at: 2026-04-23T19:33:39Z
updated_at: 2026-04-23T20:36:02Z
---

Parent bean for the MVP implementation of the Libratus-style safe re-solving gadget. See docs/plans/2026-04-23-deepstack-gadget.md for the validated design. See docs/progress/2026-04-22-subgame-exact-parity.md for iteration history.


## Validation Results (Iterations 10-12)

### Iter 10 -- cfvnet baseline (no gadget)
- exact_exp=77.67 mbb, subgame_exp=20932.49 mbb, worst_delta=1.0000
- cfvnet boundary produces 7x worse exploitability than exact_subtree

### Iter 11 -- BlueprintCbvOptOut gadget: BLOCKED
- PANIC: from_cbv_context passes abstract tree node ID (7916) as CbvTable boundary index; table only has 243 entries
- Root cause: Phase 1-3 bug in CBV index mapping. Unit tests use 1-node fixture so bug was masked
- Fix needed: map boundary ordinals to CBV chance-node ordinals, or make provider compute lazily per-boundary

### Iter 12 -- Sanity check (ConstantOptOut -999.0): PASS
- Results match iter 10 exactly (20932.49 mbb). Gadget wrapper infrastructure is correct
- Blocker is only the BlueprintCbvOptOut CBV index mapping

### Verdict
Gadget could NOT be tested with production BlueprintCbvOptOut due to the index bug. The constant-gadget sanity check confirms the GadgetEvaluator wrapping works correctly. The question of whether blueprint-CBV opt-out closes the gap remains OPEN.

### Follow-up
1. Fix BlueprintCbvOptOut::from_cbv_context boundary ordinal mapping (bug fix, not new feature)
2. Re-run iter 11 with fixed code
3. If gap doesn't close: bean akg3 (DeepStack-proper cfvnet retrain)
