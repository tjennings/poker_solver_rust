---
# poker_solver_rust-iu1u
title: 'Milestone: literature and prior-art grounding for turn-boundary CFVNet'
status: in-progress
type: feature
priority: high
created_at: 2026-05-05T02:54:52Z
updated_at: 2026-05-05T02:59:43Z
parent: poker_solver_rust-fp06
---

Research how DeepStack, Supremus, and related neural continual-resolving systems train and use counterfactual value networks at street boundaries. Feed findings into the turn-boundary model contract, dataset generator, and validation plan.\n\n## Acceptance\n\n- Primary-source notes cover DeepStack, Supremus, and relevant adjacent papers.\n- Findings identify input/output conventions, public/private range handling, rollouts, target normalization, and validation methodology.\n- Any design implications are linked back to the turn-boundary CFVNet epic and schema tasks.

## Initial research notes

- DeepStack frames the value function as a boundary counterfactual-value estimator: inputs are public state, both players' ranges, and pot size; outputs are per-player counterfactual value vectors for possible private hands. It trains separate flop and turn networks from randomly generated public situations and uses the turn network as the lower-street target for flop training. Local source: docs/papers/references/moravcik2017_deepstack.pdf. Web refs: arXiv 1701.01724, Science DOI 10.1126/science.aam6960.
- DeepStack's runtime use is iterative: CFR-D re-solving queries the value function at depth-limited leaves on each iteration because ranges change during solving. Values are normalized as fractions of pot size, and the outer network enforces the zero-sum weighted-value constraint.
- Supremus keeps the same CFVNet abstraction but strengthens it with a river network, more training data, faster GPU CFR, more iterations, larger action space, and deeper coverage. It trains bottom-up: river first, then turn, then flop, then preflop auxiliary. Local source: docs/papers/supremus.pdf. Web ref: arXiv 2007.10442.
- Supremus explicitly states the natural unbucketed input/output would be 1,326 private-hand probabilities and 1,326 values, then uses 1,000 hand buckets for tractability. Its network input is a 2,001-float vector: board encoding, both players' bucketed ranges, and pot fraction. Output is expected values for each bucket, with an external zero-sum correction.
- Brown/Sandholm/Amos depth-limited solving is the key alternative to CFVNets: use multi-valued leaf states where the opponent chooses among continuation strategies at the depth limit. This is robust but does not give the 48x inference reduction we want unless we maintain continuation strategy/value tables or another value approximator. Local source: docs/papers/references/brown2018_depth_limited.pdf. Web ref: NeurIPS 2018 paper page.
- ReBeL generalizes the idea to public belief states and learns infostate-value vectors through self-play plus search. It reinforces that the correct learned object is not a scalar state value but a vector of infostate/counterfactual values conditioned on public belief/ranges. Local source: docs/papers/ReBel.pdf. Web ref: arXiv 2007.13544.

## Design implications

- The long-term turn-boundary model should directly predict the boundary CFV vector from the 4-card turn public state, both player ranges, pot, and effective stack/commitment context, rather than enumerate river cards online.
- The dataset should be generated bottom-up from a stronger river oracle, with explicit metadata for sparse strata such as 4-bet-plus, tiny pot/high SPR, all-in pressure, and action depth.
- Validation should combine direct oracle parity, range-weighted CFV error, stratified error reports, and downstream exploitability checks. Aggregate loss alone is insufficient because Supremus and our own failures both point at sparse-strata weakness.
