---
# poker_solver_rust-j0q8
title: Generate 500/500/500 buckets for 6max config
status: completed
type: task
priority: normal
created_at: 2026-05-20T16:52:02Z
updated_at: 2026-05-20T17:56:37Z
---

Update the 6max MP sample config from 250f/100t/20r to 500f/500t/500r bucket settings and build the corresponding bucket set under local_data/buckets.

## Progress

Updated sample_configurations/blueprint_mp_6max_250f_100t_20r.yaml to use 500/500/500 postflop buckets, cluster_path ./local_data/buckets/500f_500t_500r_v1, and snapshot output ./local_data/blueprints/500f_500t_500r_v1. inspect-mp-config confirms lazy_sparse with preflop=169, flop=500, turn=500, river=500.

## V2 path update

User requested either clearing the old folder or making the output v2. Chose the safer v2 path. Updated the MP config and the shared V2 clustering config to use local_data/buckets/500f_500t_500r_v2 and local_data/blueprints/500f_500t_500r_v2/shared_500f_500t_500r_v2 as appropriate. inspect-mp-config still confirms 169/500/500/500 with lazy_sparse.

## Summary of Changes

Updated sample_configurations/blueprint_mp_6max_250f_100t_20r.yaml from 250/100/20 to 500/500/500 buckets and pointed it at ./local_data/buckets/500f_500t_500r_v2. Updated sample_configurations/blueprint_v2_500f_500t_500r.yaml to generate into the same v2 bucket directory instead of reusing the partial v1 directory.

Generated the full bucket set at local_data/buckets/500f_500t_500r_v2. Files present: preflop.buckets, flop.buckets/flop.centroids, turn.buckets/turn.centroids, river.buckets/river.centroids, scorecard.json.

Verification: inspect-mp-config confirms preflop=169, flop=500, turn=500, river=500 with lazy_sparse selected. diag-clusters completed and wrote local_data/buckets/500f_500t_500r_v2/scorecard.json.
