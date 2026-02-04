export interface TrainingResult {
  iterations: number;
  strategies: Record<string, number[]>;
  elapsed_ms: number;
}
