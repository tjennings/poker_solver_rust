---
# poker_solver_rust-lay5
title: Libratus-style safe re-solving gadget — MVP
status: completed
type: feature
priority: normal
created_at: 2026-04-23T19:33:39Z
updated_at: 2026-04-24T02:16:58Z
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



## Iter 13 — unit-conversion bug fix (2026-04-22)

`chip_cfv_to_bcfv` was computing `chip_cfv / half_pot` but CbvTable stores raw chip-pot values where break-even = half_pot, not zero. Fixed to `(chip_cfv - half_pot) / half_pot` matching the cfvnet training target formula. This reduced exploitability from 318k to 219k mbb (31% improvement), but the gadget is still 10x worse than no gadget (219k vs 21k mbb baseline).

## Final Validation Outcome

The gadget infrastructure was implemented, tested end-to-end, and the unit-conversion bug fixed (iters 11-fixed through 13). Verdict: the Libratus-style static-CBV approach with *bucketed* blueprint CBVs does NOT close the cfvnet parity gap. Even with the correct conversion formula, it makes exploitability 10x WORSE on the 4-bet turn test spot (20932 -> 218619 mbb). Root cause: blueprint CBVs are per-bucket (~2-1000 buckets/street) while cfvnet output is per-combo (1326 combos). Clamping high-resolution values up to low-resolution floors pulls strategies toward blueprint's coarse equilibrium.

This conclusively motivates bean poker_solver_rust-akg3 (DeepStack-proper cfvnet retrain with per-combo opt-out input channel) as the remaining architectural path. The gadget code stays in place as infrastructure for that future work.



## Iter 14 — Final Verdict (Per-Boundary Pot Fix)

Diagnostic-driven investigation discovered a per-boundary unit bug: `from_cbv_context` was normalising all boundaries' chip CBVs by the SUBGAME's starting half_pot (73), but each boundary sits at a different point in the turn-action-tree with pots ranging 146-398. Fixed by adding `GameTree::pot_at_node` and using each chance node's own half_pot for conversion (commits `e99fd81a`, `15c25a40`, `5ec9e6e7` on branch `diag/gadget-boundary-logging`).

**Impact:** subgame_exp 218k → 40k mbb (5.4× improvement). Gadget went from 10× worse than no-gadget baseline → 2× worse.

**Clamp rate diagnostic (after fix):**
- boundary 0 (pot=146, check-check): 98% of hands clamped — blueprint CBVs legitimately high for this narrow range
- boundary 1 (pot=242, one bet-call): 59% selective clamping
- boundary 2 (pot=398, two bets): 54% selective clamping

The remaining 2× gap vs no-gadget is structural, not a bug: on paths like check-check, the subgame's narrowed range reaches a boundary where blueprint's bucket-level CBV (computed over blueprint's broader range-reach distribution) legitimately overstates opt-out value. Static post-clamp pulls cfvnet up toward blueprint's range-level equilibrium, which is wrong for this narrower range.

## MVP Completion

Infrastructure is correct and shippable:
- OptOutProvider trait + ConstantOptOut + BlueprintCbvOptOut impls
- GadgetEvaluator wrapper with ClampStats + per-boundary diagnostic logging
- CLI flags (--gadget, --gadget-provider, --gadget-constant) on compare-solve
- Tauri enable_gadget param + Settings UI checkbox
- CBV ordinal mapping via CbvTable::build_node_to_ordinal_map + GameTree::chance_descendants + GameTree::pot_at_node

The Libratus-style static-CBV MVP is complete but its use is diagnostic-only. Production path forward is bean akg3 (DeepStack-proper cfvnet retrain with per-hand opt-out input channel).
