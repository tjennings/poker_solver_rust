import { describe, it, expect } from 'vitest';
import { buildSolveParams, isPostflopStreet } from './strategy-tabs';
import type { StrategySource } from './strategy-tabs';

describe('StrategySource type', () => {
  it('accepts blueprint, subgame, and exact values', () => {
    const sources: StrategySource[] = ['blueprint', 'subgame', 'exact'];
    expect(sources).toHaveLength(3);
  });
});

describe('isPostflopStreet', () => {
  it('returns false for Preflop', () => {
    expect(isPostflopStreet('Preflop')).toBe(false);
  });

  it('returns true for Flop', () => {
    expect(isPostflopStreet('Flop')).toBe(true);
  });

  it('returns true for Turn', () => {
    expect(isPostflopStreet('Turn')).toBe(true);
  });

  it('returns true for River', () => {
    expect(isPostflopStreet('River')).toBe(true);
  });
});

describe('buildSolveParams', () => {
  const defaultConfig = {
    solve_iterations: 200,
    target_exploitability: 3.0,
    matrix_snapshot_interval: 10,
    rollout_bias_factor: 10.0,
    rollout_num_samples: 3,
    rollout_opponent_samples: 8,
    rollout_enumerate_depth: 2,
    range_clamp_threshold: 0.05,
  };

  it('returns mode in params', () => {
    const params = buildSolveParams('subgame', defaultConfig);
    expect(params.mode).toBe('subgame');
  });

  it('sets matrixSnapshotInterval from config for subgame mode', () => {
    const params = buildSolveParams('subgame', defaultConfig);
    expect(params.matrixSnapshotInterval).toBe(10);
  });

  it('uses config matrixSnapshotInterval for exact mode (same as subgame)', () => {
    const params = buildSolveParams('exact', defaultConfig);
    expect(params.matrixSnapshotInterval).toBe(10);
  });

  it('uses default values when config is empty', () => {
    const params = buildSolveParams('subgame', {});
    expect(params.maxIterations).toBe(200);
    expect(params.targetExploitability).toBe(3.0);
    expect(params.matrixSnapshotInterval).toBe(10);
    expect(params.rolloutBiasFactor).toBe(10.0);
    expect(params.rolloutNumSamples).toBe(3);
    expect(params.rolloutOpponentSamples).toBe(8);
    expect(params.rolloutEnumerateDepth).toBe(2);
    expect(params.rangeClampThreshold).toBe(0.05);
  });

  it('uses config values when provided', () => {
    const config = {
      solve_iterations: 500,
      target_exploitability: 1.0,
      matrix_snapshot_interval: 20,
      rollout_bias_factor: 5.0,
      rollout_num_samples: 5,
      rollout_opponent_samples: 12,
      rollout_enumerate_depth: 4,
      range_clamp_threshold: 0.1,
    };
    const params = buildSolveParams('subgame', config);
    expect(params.maxIterations).toBe(500);
    expect(params.targetExploitability).toBe(1.0);
    expect(params.matrixSnapshotInterval).toBe(20);
    expect(params.rolloutBiasFactor).toBe(5.0);
    expect(params.rolloutNumSamples).toBe(5);
    expect(params.rolloutOpponentSamples).toBe(12);
    expect(params.rolloutEnumerateDepth).toBe(4);
    expect(params.rangeClampThreshold).toBe(0.1);
  });

  it('uses matrix_snapshot_interval from config for exact mode', () => {
    const config = { matrix_snapshot_interval: 20 };
    const params = buildSolveParams('exact', config);
    expect(params.matrixSnapshotInterval).toBe(20);
  });
});
