---
# poker_solver_rust-kou7
title: Implement actual sampled river-spot datagen
status: in-progress
type: feature
priority: high
created_at: 2026-05-14T05:34:45Z
updated_at: 2026-05-14T05:34:45Z
---

Generate CFVNet river training records from actual sampled blueprint/postflop river spots rather than random boards with preflop-only ranges. Include a reasonable river action set with explicit all-in coverage and tests/docs for the new datagen mode.\n\nChecklist:\n- [ ] Trace current blueprint/range datagen APIs and choose integration point\n- [ ] Add river spot source that samples concrete reached river spots with line-conditioned ranges\n- [ ] Ensure generated river subgames use a reasonable action set including all-in\n- [ ] Add config/sample docs for the new mode\n- [ ] Add focused tests or smoke coverage\n- [ ] Run targeted verification
