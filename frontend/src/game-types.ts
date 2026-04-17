export interface ActionRecord {
  action_id: string;
  label: string;
  position: string;  // "BB" or "SB"
  street: string;
  pot: number;
  stack: number;
  actions: GameAction[];  // all available actions at this decision point
}

export interface SolveStatus {
  iteration: number;
  max_iterations: number;
  exploitability: number;
  elapsed_secs: number;
  rollout_hands_per_sec: number;
  solver_name: string;
  is_complete: boolean;
}

export interface GameAction {
  id: string;
  label: string;
  action_type: string;  // "fold", "check", "call", "bet", "raise", "allin"
}

export interface ComboDetail {
  cards: string;
  probabilities: number[];
  weight: number;
}

export interface GameMatrixCell {
  hand: string;
  suited: boolean;
  pair: boolean;
  probabilities: number[];
  combo_count: number;
  weight: number;
  ev: number | null;
  combos: ComboDetail[];
}

export interface GameMatrix {
  cells: GameMatrixCell[][];
  actions: GameAction[];
}

export interface GameState {
  street: string;
  position: string;
  board: string[];
  pot: number;
  stacks: [number, number];
  matrix: GameMatrix | null;
  actions: GameAction[];
  action_history: ActionRecord[];
  is_terminal: boolean;
  is_chance: boolean;
  solve: SolveStatus | null;
}
