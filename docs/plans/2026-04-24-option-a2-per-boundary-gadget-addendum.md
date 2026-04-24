# Option A2: Per-Boundary Gadget -- Design Addendum

**Status:** Shipped at merge commit `8af0d107` (2026-04-23).
**Supersedes:** Option A (root-level gadget, merged at `50799416`).
**Parent design doc:** `docs/plans/2026-04-24-option-a-deepstack-gadget-design.md`.

---

## Why we pivoted from Option A to Option A2

Option A introduced three departures from the standard CFR-D gadget
construction: (1) root-only gadget placement (two nested Decision nodes
at arena indices 0--3, above the subgame root), (2) bucketed CBV
opt-out source computed at the subgame root rather than at each depth
boundary, and (3) neutralized (zero) terminal payoffs for the
non-gadget player. A comparative review flagged that the canonical
construction in Burch, Johanson, and Bowling (2014, Section 3) places
the gadget at each depth boundary -- one per boundary per player --
not at the subgame root. The user directive was to move the gadget to
per-boundary placement and make it single-sided per traverser pass (the
opponent's gadget is visited but not regret-updated), matching the
theoretical description more closely.

## What A2 differs from A

| Dimension | Option A | Option A2 |
|-----------|----------|-----------|
| Placement | Root-only (arena 0--3) | Per-boundary (4 nodes per cfvnet boundary) |
| Activation | Always-on (both passes regret-match) | Traverser-dependent: owner's pass regret-matches; non-owner's pass is passthrough (sigma forced to (0,1)) |
| Gadget structure | Two nested Decisions at subgame root | `G_IP -> [Terminate_IP, G_OOP -> [Terminate_OOP, Follow]]` at each boundary |
| Opt-out source | `opt_out_at_subgame_root` (single-point) | `from_cbv_context` (per-boundary, per-player, per-hand, normalised by per-boundary pot) |
| Ordinal layout | Ordinals 0--1 reserved for gadget; cfvnet shifted to 2+ | Ordinals 0..N stable (cfvnet); gadget terminals at N..3N |
| Root invariant | `game.root()` returned gadget G_IP node | `game.root()` returns the real subgame root |
| Non-gadget player terminal | Zero (neutralized) | Zero (neutralized; zero-sum complement deferred) |

The traverser-disable semantics (called "Option Y" during development)
are the key behavioral change. Under Option A, both players' traverser
passes ran regret-matching at both gadget Decision nodes. Under A2,
each gadget Decision is owned by one player. On the owner's pass, it
behaves as a standard Decision node with two actions (Terminate and
Follow). On the non-owner's pass, the solver skips to Follow with no
regret or strategy-sum update, making the gadget invisible to the
non-owner's traversal. This matches the CFR-D formulation where each
player's gadget constrains only that player's counterfactual values.

## Theoretical position

**Structural placement** matches Burch, Johanson, and Bowling (2014,
Section 3): the gadget is inserted at each point where the subgame
connects to the trunk game (i.e. each depth boundary), not at an
artificial root above the subgame. Brown and Sandholm (2017, Section 3)
describe the same structural pattern for safe nested subgame solving.

**Single-sided activation** follows from the CFR-D proof structure: the
safety guarantee is local to each gadget Decision and applies only to
the gadget owner's counterfactual values. The non-owner's pass need
not interact with the gadget at all. Burch et al. (2014, Section 3,
proof of Theorem 1) state the sufficiency condition in terms of the
acting player's regret at the gadget node.

**Bucketed CBV opt-out** remains looser than the Libratus "estimate"
approach (Brown and Sandholm 2017, Section 6), which uses per-combo
unbucketed values. Our opt-out values are bucketed blueprint CBVs,
meaning hands within the same bucket share a single opt-out value.
This looseness is acceptable for safety (the guarantee holds regardless
of opt-out tightness; tighter values only reduce exploitability) but
limits how much the gadget can improve strategy quality. Tightening
via per-combo unbucketed CBVs is tracked in the rescoped bean `akg3`.

**DeepStack-Leduc reference.** The DeepStack-Leduc implementation
(`cfrd_gadget.lua`) uses an algorithmic gadget pattern (modifying the
lookahead's backward pass) rather than a structural tree modification.
Our structural approach achieves the same safety guarantee through a
different mechanism: the gadget nodes are first-class tree nodes that
the existing DCFR solver traverses without special-case backward-pass
logic.

## Citations

- Burch, N., Johanson, M., and Bowling, M. (2014). Solving
  Imperfect-Information Games Using Decomposition. *AAAI-14*, Section 3.
  https://webdocs.cs.ualberta.ca/~bowling/papers/14aaai-cfrd.pdf
- Brown, N. and Sandholm, T. (2017). Safe and Nested Subgame Solving
  for Imperfect-Information Games. *NeurIPS 2017*, Sections 3 and 6.
  https://arxiv.org/abs/1705.02955
- Moravcik, M. et al. (2017). DeepStack: Expert-Level Artificial
  Intelligence in Heads-Up No-Limit Poker. *Science* 356(6337).
  DeepStack-Leduc reference implementation:
  https://github.com/lifrordi/DeepStack-Leduc
- Schmid, M. et al. (2023). Student of Games: A Unified Learning
  Algorithm for Both Perfect and Imperfect Information Games. *Science
  Advances*. https://arxiv.org/abs/2112.03178
