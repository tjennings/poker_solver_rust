---
# poker_solver_rust-g781
title: Fix SB child matrix after root subgame solve
status: in-progress
type: bug
priority: high
created_at: 2026-05-05T18:01:56Z
updated_at: 2026-05-05T18:01:56Z
---

After solving a turn subgame at the street root, navigating BB Check to the SB action node still shows the default matrix instead of the solved subgame matrix for that game state.\n\n## TODOs\n\n- [ ] Reproduce or encode the BB-check/SB default-matrix fallback case.\n- [ ] Trace subgame/exact/blueprint matrix source selection after street-root solve.\n- [ ] Fix solved matrix lookup so child states use solved representative matrices.\n- [ ] Add regression coverage for solved root -> BB Check -> SB matrix.\n- [ ] Run targeted and full verification.\n- [ ] Merge to local main.
