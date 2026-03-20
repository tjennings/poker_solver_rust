# Cluster Diagnostics Baseline: 500bkt (pre-EMD fix)

**Date:** 2026-03-20
**Directory:** local_data/buckets/500bkt
**Pipeline:** L2 k-means (fastkmeans-rs) + L2 exhaustive assignment (pre-improvement)

## Bucket Sizes

| Street | Buckets | Entries | Min Size | Max Size | Mean | Std Dev |
|--------|---------|---------|----------|----------|------|---------|
| River | 500 | 178M | 2,904 | 33.3M | 356K | 1.49M |
| Turn | 500 | 21.8M | 0 | 3.4M | 43.6K | 159K |
| Flop | 500 | 2.3M | 16 | 268K | 4,654 | 12K |
| Preflop | 169 | 1,326 | 4 | 12 | 7.8 | 3.9 |

Turn has empty buckets (min=0). River and turn heavily skewed (std >> mean).

## Intra-Bucket Equity Quality

| Street | Mean Std Dev | Max Std Dev |
|--------|-------------|-------------|
| River | 0.0509 | 0.4547 |
| Turn | 0.0624 | 0.3847 |
| Flop | 0.0712 | 0.4028 |
| Preflop | 0.0000 | 0.0000 |

## Transition Consistency

| Transition | Buckets Seen | Mean EMD | Max EMD |
|------------|-------------|----------|---------|
| Flop → Turn | 494 / 500 | 26.2 | 110.3 |
| Turn → River | 415 / 500 | 23.2 | 99.9 |

Only 415/500 turn buckets observed in 30 sample boards. High max EMD (>100) indicates some buckets catch strategically unrelated hands.

## Notes

- This is the baseline before clustering quality improvements (sorted bucket IDs, EMD assignment, weighted ground distances)
- Re-cluster with the same config using the improved pipeline and compare
