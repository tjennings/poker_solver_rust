---
# poker_solver_rust-g9c4
title: Restore canonical flop transform display in Tauri explorer
status: todo
type: task
created_at: 2026-03-31T19:01:16Z
updated_at: 2026-03-31T19:01:16Z
---

When the user selects a non-canonical flop (e.g., QsTs7h), the explorer should:
1. Replace the displayed flop with its canonical equivalent (e.g., QhTh7d)
2. Show the user's original selected cards smaller underneath for reference

This UI feature previously existed — check the frontend code for remnants (search for 'canonical', 'isomorphism', 'transform', 'original flop' in frontend/src/).

The backend already canonicalizes boards internally via CanonicalBoard::from_cards() for bucket lookups, but the UI doesn't communicate the transformation to the user.
