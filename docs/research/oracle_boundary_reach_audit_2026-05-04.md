# Oracle Boundary Reach Audit (2026-05-04)

Bean: `poker_solver_rust-qcgs`

## Purpose

Test whether `exact_oracle` divergence is caused by stale or asymmetric reach vectors at depth-boundary handoff.

## Finding

The raw oracle path was caching both players' raw CFV slots on the first visit to a boundary. That first visit only guarantees that the current traverser's opponent reach has just been recorded. The traverser's own reach may still be empty for the current iteration, so the code fell back to `initial_weights`.

This is correct for the current player's raw CFV because that value depends on the opponent reach. It is not correct for the other player's raw CFV, which was being cached before that player's current opponent reach had been recorded.

## Code Path

In `PostFlopGame::evaluate_internal`, depth-boundary evaluation first stores:

- `boundary_reach[ordinal * 2 + opp] = cfreach`

The legacy `compute_cfvs_both` path then uses both seats' reaches and caches both players' normalized boundary CFVs. That path is unchanged.

The raw `compute_raw_cfvs_both` path now caches only `boundary_cfvs[ordinal * 2 + player]`, because raw CFVs are already reach-integrated for the current traverser. The other player is computed when that player visits the boundary and their current opponent reach has been captured.

## Regression Test

Added `raw_cfv_evaluator_caches_only_current_players_reach_dependent_values`.

The test uses a raw boundary evaluator whose returned OOP value depends on IP reach and whose returned IP value depends on OOP reach. It traverses OOP first, then IP, with intentionally different non-initial reaches. The evaluator must be called once per traversing player, and each side must see the current opponent reach.

This test would fail under the previous both-slot raw cache behavior because the IP slot would be prefilled during the OOP traversal using initial OOP reach.

## Canonical Result

Command:

```bash
./target/release/poker-solver-trainer compare-solve \
  --bundle ./local_data/blueprints/1k_100bb_brdcfr_v2 \
  --snapshot snapshot_0013 \
  --spot 'sb:2bb,bb:10bb,sb:22bb,bb:call|JhTh9h|bb:15bb,sb:call|7d' \
  --river-boundary exact_oracle \
  --iters 200 \
  --tolerance 1.0
```

Result after the reach-cache fix:

| Metric | Value |
| --- | ---: |
| Exact exploitability | `23.06` mbb/hand |
| Subgame exploitability | `42.85` mbb/hand |
| Exploitability delta | `+19.79` mbb/hand |
| Mean root mass moved | `0.080` |
| Max root mass moved | `0.893` |
| Worst tolerance cell | `Q7o @ All-in`, `|Δ|=0.4120` |

Per-action-class bias:

| Class | Subgame - exact |
| --- | ---: |
| Check | `-0.003` |
| Bet/Raise | `-0.020` |
| AllIn | `+0.022` |

## Interpretation

Reach-cache timing was a real root cause. Before this fix, the default `exact_oracle` scale had `subgame_exp=2860.16` mbb/hand and mean root mass moved `0.184`. After the fix, `subgame_exp=42.85` mbb/hand and mean root mass moved `0.080`.

The remaining gap is much smaller and no longer looks like a sign, unit, or first-visit reach-cache failure. The next audit should focus on how raw boundary values are injected into regret updates over iterations: especially whether exact and depth-limited solves use equivalent averaging, discounting, and chance/reach weighting at boundary terminals.
