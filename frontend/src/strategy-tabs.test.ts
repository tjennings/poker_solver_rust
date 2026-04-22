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
    range_clamp_threshold: 0.05,
    flop_boundary_mode: 'exact' as const,
    turn_boundary_mode: 'exact' as const,
    river_boundary_mode: 'exact' as const,
    flop_model_path: '',
    turn_model_path: '',
    river_model_path: '',
  };

  it('returns mode in params', () => {
    const params = buildSolveParams('subgame', defaultConfig);
    expect(params.mode).toBe('subgame');
  });

  it('sets matrixSnapshotInterval from config', () => {
    const params = buildSolveParams('subgame', defaultConfig);
    expect(params.matrixSnapshotInterval).toBe(10);
  });

  it('uses default values when config is empty', () => {
    const params = buildSolveParams('subgame', {});
    expect(params.maxIterations).toBe(200);
    expect(params.targetExploitability).toBe(3.0);
    expect(params.matrixSnapshotInterval).toBe(10);
    expect(params.rangeClampThreshold).toBe(0.05);
  });

  it('uses config values when provided', () => {
    const config = {
      solve_iterations: 500,
      target_exploitability: 1.0,
      matrix_snapshot_interval: 20,
      range_clamp_threshold: 0.1,
      flop_boundary_mode: 'exact' as const,
      turn_boundary_mode: 'exact' as const,
      river_boundary_mode: 'exact' as const,
      flop_model_path: '',
      turn_model_path: '',
      river_model_path: '',
    };
    const params = buildSolveParams('subgame', config);
    expect(params.maxIterations).toBe(500);
    expect(params.targetExploitability).toBe(1.0);
    expect(params.matrixSnapshotInterval).toBe(20);
    expect(params.rangeClampThreshold).toBe(0.1);
  });

  // streetBoundaryConfig tests

  it('builds all-exact streetBoundaryConfig when all modes are exact', () => {
    const params = buildSolveParams('subgame', defaultConfig);
    expect(params.streetBoundaryConfig).toEqual({
      flop: { mode: 'exact' },
      turn: { mode: 'exact' },
      river: { mode: 'exact' },
    });
  });

  it('builds cfvnet streetBoundaryConfig for river with model path', () => {
    const config = {
      ...defaultConfig,
      river_boundary_mode: 'cfvnet' as const,
      river_model_path: '/models/river_v2.onnx',
    };
    const params = buildSolveParams('subgame', config);
    expect(params.streetBoundaryConfig).toEqual({
      flop: { mode: 'exact' },
      turn: { mode: 'exact' },
      river: { mode: 'cfvnet', model_path: '/models/river_v2.onnx' },
    });
  });

  it('builds cfvnet streetBoundaryConfig for turn with model path', () => {
    const config = {
      ...defaultConfig,
      turn_boundary_mode: 'cfvnet' as const,
      turn_model_path: '/models/turn_v1.onnx',
    };
    const params = buildSolveParams('subgame', config);
    expect(params.streetBoundaryConfig).toEqual({
      flop: { mode: 'exact' },
      turn: { mode: 'cfvnet', model_path: '/models/turn_v1.onnx' },
      river: { mode: 'exact' },
    });
  });

  it('throws if cfvnet mode has empty model_path', () => {
    const config = {
      ...defaultConfig,
      river_boundary_mode: 'cfvnet' as const,
      river_model_path: '',
    };
    expect(() => buildSolveParams('subgame', config)).toThrow(
      'Street boundary set to cfvnet but no model_path',
    );
  });

  it('throws if cfvnet mode has empty model_path for flop', () => {
    const config = {
      ...defaultConfig,
      flop_boundary_mode: 'cfvnet' as const,
      flop_model_path: '',
    };
    expect(() => buildSolveParams('subgame', config)).toThrow(
      'Street boundary set to cfvnet but no model_path',
    );
  });

  it('does not include legacy fields in output', () => {
    const params = buildSolveParams('subgame', defaultConfig) as unknown as Record<string, unknown>;
    expect(params).not.toHaveProperty('subgameDepthLimit');
    expect(params).not.toHaveProperty('hybridRefreshInterval');
    expect(params).not.toHaveProperty('hybridSamplesPerRefresh');
    expect(params).not.toHaveProperty('rolloutBiasFactor');
    expect(params).not.toHaveProperty('rolloutNumSamples');
    expect(params).not.toHaveProperty('rolloutOpponentSamples');
    expect(params).not.toHaveProperty('rolloutEnumerateDepth');
  });

  it('uses matrix_snapshot_interval from config for exact mode', () => {
    const config = { ...defaultConfig, matrix_snapshot_interval: 20 };
    const params = buildSolveParams('exact', config);
    expect(params.matrixSnapshotInterval).toBe(20);
  });

  // Boundary tracing params

  it('defaults traceBoundaries to empty and traceIters to last', () => {
    const params = buildSolveParams('subgame', {});
    expect(params.traceBoundaries).toBe('');
    expect(params.traceIters).toBe('last');
  });

  it('passes trace config from global config', () => {
    const config = {
      ...defaultConfig,
      trace_boundaries: '0,42',
      trace_iters: 'all',
    };
    const params = buildSolveParams('subgame', config);
    expect(params.traceBoundaries).toBe('0,42');
    expect(params.traceIters).toBe('all');
  });
});
