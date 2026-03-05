---
# poker_solver_rust-n6x2
title: Add strategy snapshot serialization at print intervals
status: in-progress
type: feature
created_at: 2026-03-05T04:03:18Z
updated_at: 2026-03-05T04:03:18Z
---

Write the full Strategies struct (strategy_sums) to numbered files at each print_every checkpoint during preflop solving. Enables inspecting convergence over time without waiting for full run completion.

## Requirements
- At each print_every interval, serialize the current average strategy to disk
- Write to output_dir/snapshots/iter_NNNN.bin (or similar)
- Must not significantly slow down training
- Should be loadable by the existing exploration tools

## Checklist
- [ ] Research Strategies struct and current serialization format
- [ ] Plan implementation
- [ ] Implement snapshot writing in solver loop
- [ ] Ensure snapshots are loadable
- [ ] Test
