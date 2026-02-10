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

// New types for async training
export interface TrainingProgress {
  iteration: number;
  total_iterations: number;
  exploitability: number;
  elapsed_ms: number;
  running: boolean;
}

export interface AsyncTrainingResult {
  iterations: number;
  strategies: Record<string, number[]>;
  exploitability: number;
  elapsed_ms: number;
  stopped_early: boolean;
}

export interface TrainedStrategy {
  game_type: string;
  iterations: number;
  exploitability: number;
  strategies: Record<string, number[]>;
}

// Exploration types
export interface BundleInfo {
  name: string | null;
  stack_depth: number;
  bet_sizes: number[];
  info_sets: number;
  iterations: number;
}

export interface AgentInfo {
  name: string;
  path: string;
}

export interface ActionProb {
  action: string;
  probability: number;
}

export interface MatrixCell {
  hand: string;
  suited: boolean;
  pair: boolean;
  probabilities: ActionProb[];
  filtered: boolean;
}

export interface ActionInfo {
  id: string;
  label: string;
  action_type: string;
  size_key?: string | null;
}

export interface StrategyMatrix {
  cells: MatrixCell[][];
  actions: ActionInfo[];
  street: string;
  pot: number;
  stack: number;
  to_call: number;
  stack_p1: number;
  stack_p2: number;
}

export interface ExplorationPosition {
  board: string[];
  history: string[];
  pot: number;
  stack_p1: number;
  stack_p2: number;
  to_act: number;
}

// Combo classification types
export interface ComboGroup {
  bits: number;
  class_names: string[];
  combos: string[];
  strategy: number[];
}

export interface ComboGroupInfo {
  hand: string;
  groups: ComboGroup[];
  total_combos: number;
  blocked_combos: number;
  street: string;
  pot_bucket: number;
  stack_bucket: number;
}

// Simulation types
export interface StrategySourceInfo {
  name: string;
  source_type: 'agent' | 'bundle';
  path: string;
}

export interface SimulationProgress {
  hands_played: number;
  total_hands: number;
  p1_profit_bb: number;
  current_mbbh: number;
}

export interface SimulationResult {
  hands_played: number;
  p1_profit_bb: number;
  mbbh: number;
  equity_curve: number[];
  elapsed_ms: number;
}
