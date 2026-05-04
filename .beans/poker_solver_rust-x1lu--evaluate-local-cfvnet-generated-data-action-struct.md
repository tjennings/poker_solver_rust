---
# poker_solver_rust-x1lu
title: Evaluate local CFVNet generated data action structure
status: completed
type: task
priority: normal
created_at: 2026-05-04T15:07:42Z
updated_at: 2026-05-04T15:09:58Z
---

Use the generated-data eval command and samples under local_data/cfvnet/fuzzed to inspect how many action-depth layers and bet sizes are present, then compare the observed structure to the Supremus CFVNet abstraction.

## Summary of Findings

Ran the generated-data eval command on a 10k-record sample from local_data/cfvnet/fuzzed. The command is ./target/debug/cfvnet datagen-eval -d <path>; it loaded the sample successfully and reported pot/stack/SPR/range/CFV distributions.

Important limitation: TrainingRecord persists board, pot, effective_stack, player, game_value, ranges, CFVs, and valid mask, but not the exact fuzzed bet sizes or action history. The eval command therefore validates data distributions, not the per-sample betting abstraction.

Code inspection shows current river domain datagen parses nested bet_sizes, but build_turn_game_inner maps bet_sizes[0] to first bet sizes and flattens bet_sizes[1..] into one shared raise-size pool. For boundary_net_river_datagen.yaml this means first bets are 25/50/100% plus auto all-in, while every raise depth uses 25/75% plus auto all-in. This is not Supremus's four-bucket schedule of first, second, third, remaining actions.

The sampled file showed balanced configured SPR buckets and 10k valid river records, but cannot answer how many raises occurred in training targets because that is absent from the on-disk record format.
