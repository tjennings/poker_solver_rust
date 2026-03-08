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

export interface DatasetInfo {
  name: string;
  path: string;
  kind: 'preflop' | 'postflop' | 'agent';
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
  reaching_p1: number[];
  reaching_p2: number[];
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

// Range editing types
export interface PlayerRange {
  hands: number[];           // [169] reaching probabilities
  source: 'computed' | 'edited' | 'manual';
  overrides: number[];       // indices of manually edited hands
}

export interface RangeSnapshot {
  p1_range: PlayerRange;
  p2_range: PlayerRange;
  node_index: number;
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

// Blueprint-derived config for postflop exploration
export interface BlueprintConfig {
  oop_range: string;
  ip_range: string;
  oop_weights: number[];
  ip_weights: number[];
  pot: number;
  effective_stack: number;
  oop_bet_sizes: string;
  oop_raise_sizes: string;
  ip_bet_sizes: string;
  ip_raise_sizes: string;
  blueprint_dir: string;
}

// Postflop solver types

export interface PostflopConfig {
  oop_range: string;
  ip_range: string;
  pot: number;
  effective_stack: number;
  oop_bet_sizes: string;
  oop_raise_sizes: string;
  ip_bet_sizes: string;
  ip_raise_sizes: string;
}

export interface PostflopConfigSummary {
  config: PostflopConfig;
  oop_combos: number;
  ip_combos: number;
}

export interface PostflopComboDetail {
  cards: string;
  probabilities: number[];
}

export interface PostflopMatrixCell {
  hand: string;
  suited: boolean;
  pair: boolean;
  probabilities: number[];
  combo_count: number;
  ev: number | null;
  combos: PostflopComboDetail[];
}

export interface PostflopStrategyMatrix {
  cells: PostflopMatrixCell[][];
  actions: ActionInfo[];
  player: number;
  pot: number;
  stacks: [number, number];
  board: string[];
}

export interface PostflopProgress {
  iteration: number;
  max_iterations: number;
  exploitability: number;
  is_complete: boolean;
  matrix: PostflopStrategyMatrix | null;
}

export interface PostflopStreetResult {
  filtered_oop_range: number[];
  filtered_ip_range: number[];
  pot: number;
  effective_stack: number;
}

export interface PostflopPlayResult {
  matrix: PostflopStrategyMatrix | null;
  is_terminal: boolean;
  is_chance: boolean;
  current_player: number | null;
  pot: number;
  stacks: [number, number];
}

export interface GlobalConfig {
  blueprint_dir: string;
  target_exploitability: number;
}

export interface BlueprintListEntry {
  name: string;
  path: string;
  stack_depth: number;
  has_strategy: boolean;
}

export interface PreflopRanges {
  oop_weights: number[];
  ip_weights: number[];
  pot: number;
  effective_stack: number;
  oop_bet_sizes: string;
  oop_raise_sizes: string;
  ip_bet_sizes: string;
  ip_raise_sizes: string;
}
