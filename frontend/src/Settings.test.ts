import { describe, it, expect } from 'vitest';
import { hasAnyCut } from './Settings';
import type { GlobalConfig } from './types';

describe('hasAnyCut', () => {
  const allExact: GlobalConfig = {
    blueprint_dir: '',
    target_exploitability: 3.0,
    solve_iterations: 200,
    backend_url: '',
    matrix_snapshot_interval: 10,
    range_clamp_threshold: 0.05,
    flop_boundary_mode: 'exact',
    turn_boundary_mode: 'exact',
    river_boundary_mode: 'exact',
    flop_model_path: '',
    turn_model_path: '',
    river_model_path: '',
    trace_boundaries: '',
    trace_iters: 'last',
    enable_safe_resolving: false,
  };

  it('returns false when all streets are exact', () => {
    expect(hasAnyCut(allExact)).toBe(false);
  });

  it('returns true when flop is cfvnet', () => {
    expect(hasAnyCut({ ...allExact, flop_boundary_mode: 'cfvnet' })).toBe(true);
  });

  it('returns true when turn is cfvnet', () => {
    expect(hasAnyCut({ ...allExact, turn_boundary_mode: 'cfvnet' })).toBe(true);
  });

  it('returns true when river is cfvnet', () => {
    expect(hasAnyCut({ ...allExact, river_boundary_mode: 'cfvnet' })).toBe(true);
  });

  it('returns true when multiple streets are cfvnet', () => {
    expect(hasAnyCut({
      ...allExact,
      turn_boundary_mode: 'cfvnet',
      river_boundary_mode: 'cfvnet',
    })).toBe(true);
  });
});
