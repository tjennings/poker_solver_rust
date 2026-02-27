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
  stacks: number[];
  stack_p1: number;
  stack_p2: number;
  to_act: number;
  num_players: number;
  active_players: boolean[];
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

// Villain-specific matchup EV
export interface MatchupEquity {
  villain_hand: string;
  ev_pos0: number;
  ev_pos1: number;
  ev_avg: number;
}

// Hand equity from postflop bundle
export interface HandEquity {
  ev_pos0: number;
  ev_pos1: number;
  ev_avg: number;
  ev_vs_hand: MatchupEquity | null;
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
