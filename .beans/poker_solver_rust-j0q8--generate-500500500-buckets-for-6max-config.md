---
# poker_solver_rust-j0q8
title: Generate 500/500/500 buckets for 6max config
status: in-progress
type: task
priority: normal
created_at: 2026-05-20T16:52:02Z
updated_at: 2026-05-20T16:54:58Z
---

Update the 6max MP sample config from 250f/100t/20r to 500f/500t/500r bucket settings and build the corresponding bucket set under local_data/buckets.

## Progress

Updated sample_configurations/blueprint_mp_6max_250f_100t_20r.yaml to use 500/500/500 postflop buckets, cluster_path ./local_data/buckets/500f_500t_500r_v1, and snapshot output ./local_data/blueprints/500f_500t_500r_v1. inspect-mp-config confirms lazy_sparse with preflop=169, flop=500, turn=500, river=500.

## V2 path update

User requested either clearing the old folder or making the output v2. Chose the safer v2 path. Updated the MP config and the shared V2 clustering config to use local_data/buckets/500f_500t_500r_v2 and local_data/blueprints/500f_500t_500r_v2/shared_500f_500t_500r_v2 as appropriate. inspect-mp-config still confirms 169/500/500/500 with lazy_sparse.
