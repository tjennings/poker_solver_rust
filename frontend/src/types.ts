export interface TrainingResult {
  iterations: number;
  strategies: Record<string, number[]>;
  elapsed_ms: number;
}

export interface Checkpoint {
  iteration: number;
  exploitability: number;
  elapsed_ms: number;
}

export interface TrainingResultWithCheckpoints {
  checkpoints: Checkpoint[];
  strategies: Record<string, number[]>;
  total_iterations: number;
  total_elapsed_ms: number;
}
