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
  rolloutBiasFactor: number;
  rolloutNumSamples: number;
  rolloutOpponentSamples: number;
  rolloutEnumerateDepth: number;
  rangeClampThreshold: number;
}

/**
 * Build solve parameters from a mode and global config.
 */
export function buildSolveParams(
  mode: SolveMode,
  config: Record<string, unknown>,
): SolveParams {
  const matrixSnapshotInterval = (config.matrix_snapshot_interval as number | undefined) ?? 10;

  return {
    mode,
    maxIterations: (config.solve_iterations as number | undefined) ?? 200,
    targetExploitability: (config.target_exploitability as number | undefined) ?? 3.0,
    matrixSnapshotInterval,
    rolloutBiasFactor: (config.rollout_bias_factor as number | undefined) ?? 10.0,
    rolloutNumSamples: (config.rollout_num_samples as number | undefined) ?? 3,
    rolloutOpponentSamples: (config.rollout_opponent_samples as number | undefined) ?? 8,
    rolloutEnumerateDepth: (config.rollout_enumerate_depth as number | undefined) ?? 2,
    rangeClampThreshold: (config.range_clamp_threshold as number | undefined) ?? 0.05,
  };
}
