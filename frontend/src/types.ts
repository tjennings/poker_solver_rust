export interface BundleInfo {
  name: string | null;
  stack_depth: number;
  bet_sizes: number[];
  info_sets: number;
  iterations: number;
  preflop_only: boolean;
  rake_rate: number;
  rake_cap: number;
  snapshot_name: string | null;
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
  weight: number;
  ev?: number | null;
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
  to_act: number;
  stack_p1: number;
  stack_p2: number;
  reaching_p1: number[];
  reaching_p2: number[];
  /** Which seat is the dealer/button (SB in heads-up). */
  dealer: number;
}

export interface ExplorationPosition {
  board: string[];
  history: string[];
  pot: number;
  stacks: number[];
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

export interface GlobalConfig {
  blueprint_dir: string;
  target_exploitability: number;
  stub_range_solver?: boolean;
  solve_iterations: number;
  backend_url: string;
  rollout_bias_factor: number;
  rollout_num_samples: number;
  rollout_opponent_samples: number;
  rollout_enumerate_depth: number;
  matrix_snapshot_interval: number;
  range_clamp_threshold: number;
  subgame_depth_limit: number;
}

export interface BlueprintListEntry {
  name: string;
  path: string;
  stack_depth: number;
  has_strategy: boolean;
}

export interface SnapshotEntry {
  name: string;
  iterations: number | null;
  elapsed_minutes: number | null;
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
  rake_rate: number;
  rake_cap: number;
  abstract_node_idx: number;
}

export interface CacheInfo {
  exploitability: number;
  iterations: number;
}
