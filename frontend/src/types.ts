export interface BundleInfo {
  name: string | null;
  stack_depth: number;
  bet_sizes: number[];
  info_sets: number;
  iterations: number;
  preflop_only: boolean;
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

// Board canonicalization result
export interface CanonicalizeResult {
  canonical_cards: string[];
  remapped: boolean;
  suit_map: Record<string, string> | null;
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
