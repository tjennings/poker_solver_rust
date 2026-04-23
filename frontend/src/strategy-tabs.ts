import type { StreetBoundaryConfig, StreetBoundaryMode } from './types';

/** Strategy source type for tab selection */
export type StrategySource = 'blueprint' | 'subgame' | 'exact';

/** Solve mode type (subgame or exact only -- blueprint has no solve) */
export type SolveMode = 'subgame' | 'exact';

/** Returns true if the street is postflop (Flop, Turn, or River) */
export function isPostflopStreet(street: string): boolean {
  return street !== 'Preflop';
}

/** Solve parameters sent to the backend game_solve command */
export interface SolveParams {
  mode: SolveMode;
  maxIterations: number;
  targetExploitability: number;
  matrixSnapshotInterval: number;
  rangeClampThreshold: number;
  streetBoundaryConfig: StreetBoundaryConfig;
  traceBoundaries: string;
  traceIters: string;
  enableGadget: boolean;
}

function modeFromConfig(
  mode: 'exact' | 'cfvnet' | undefined,
  modelPath: string | undefined,
): StreetBoundaryMode {
  if (mode === 'cfvnet') {
    if (!modelPath) throw new Error('Street boundary set to cfvnet but no model_path');
    return { mode: 'cfvnet', model_path: modelPath };
  }
  return { mode: 'exact' };
}

/**
 * Build solve parameters from a mode and global config.
 */
export function buildSolveParams(
  mode: SolveMode,
  config: Record<string, unknown>,
): SolveParams {
  return {
    mode,
    maxIterations: (config.solve_iterations as number | undefined) ?? 200,
    targetExploitability: (config.target_exploitability as number | undefined) ?? 3.0,
    matrixSnapshotInterval: (config.matrix_snapshot_interval as number | undefined) ?? 10,
    rangeClampThreshold: (config.range_clamp_threshold as number | undefined) ?? 0.05,
    streetBoundaryConfig: {
      flop: modeFromConfig(config.flop_boundary_mode as 'exact' | 'cfvnet' | undefined, config.flop_model_path as string | undefined),
      turn: modeFromConfig(config.turn_boundary_mode as 'exact' | 'cfvnet' | undefined, config.turn_model_path as string | undefined),
      river: modeFromConfig(config.river_boundary_mode as 'exact' | 'cfvnet' | undefined, config.river_model_path as string | undefined),
    },
    traceBoundaries: (config.trace_boundaries as string | undefined) ?? '',
    traceIters: (config.trace_iters as string | undefined) ?? 'last',
    enableGadget: (config.enable_safe_resolving as boolean | undefined) ?? false,
  };
}
