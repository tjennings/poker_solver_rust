---
# poker_solver_rust-jyth
title: Restore full workspace test suite below 60 seconds
status: completed
type: bug
priority: high
created_at: 2026-08-04T18:01:38Z
updated_at: 2026-08-04T18:18:48Z
---

The mandatory pre-development baseline `/usr/bin/time -p cargo test --workspace --quiet` initially passed all tests but took 62.83 seconds, exceeding the repository hard limit. Diagnose whether this was a code regression or a measurement artifact and restore reliable evidence below 60 seconds without weakening coverage.

- [x] Reproduce and attribute the slowest critical-path tests or startup costs
- [x] Design the smallest safe runtime response
- [x] Determine that no Rust change is required after output-captured measurements
- [x] Obtain independent architecture and runtime analysis
- [x] Verify three consecutive full quiet workspace suites pass below 60 seconds

## Summary of Changes

No test code was changed. Direct streamed output introduced severe backpressure and unstable wall-time measurements. Capturing output to a temporary file produced three consecutive complete passing workspace runs at 29.82s, 30.67s, and 31.13s, safely below the 60-second gate. Profiling identified the largest harnesses for future use, but no coverage reduction was justified.
