import { describe, it, expect } from 'vitest';
import type {
  ActionRecord,
  SolveStatus,
  GameAction,
  GameMatrixCell,
  GameMatrix,
  GameState,
} from './game-types';

describe('GameState types', () => {
  it('constructs a valid GameState with all fields', () => {
    const action: GameAction = {
      id: '0',
      label: 'Fold',
      action_type: 'fold',
    };

    const cell: GameMatrixCell = {
      hand: 'AKs',
      suited: true,
      pair: false,
      probabilities: [0.3, 0.7],
      combo_count: 4,
      weight: 0.85,
      ev: 1.5,
    };

    const matrix: GameMatrix = {
      cells: [[cell]],
      actions: [action],
    };

    const record: ActionRecord = {
      action_id: '0',
      label: 'Fold',
      position: 'BB',
      street: 'Preflop',
      pot: 3,
      stack: 199,
    };

    const state: GameState = {
      street: 'Preflop',
      position: 'BB',
      board: [],
      pot: 3,
      stacks: [199, 198],
      matrix,
      actions: [action],
      action_history: [record],
      is_terminal: false,
      is_chance: false,
      solve: null,
    };

    expect(state.street).toBe('Preflop');
    expect(state.stacks).toEqual([199, 198]);
    expect(state.matrix?.cells[0][0].hand).toBe('AKs');
    expect(state.actions[0].action_type).toBe('fold');
    expect(state.action_history[0].position).toBe('BB');
    expect(state.is_terminal).toBe(false);
    expect(state.is_chance).toBe(false);
    expect(state.solve).toBeNull();
  });

  it('constructs a terminal GameState without matrix', () => {
    const state: GameState = {
      street: 'Preflop',
      position: '',
      board: [],
      pot: 200,
      stacks: [100, 100],
      matrix: null,
      actions: [],
      action_history: [],
      is_terminal: true,
      is_chance: false,
      solve: null,
    };

    expect(state.is_terminal).toBe(true);
    expect(state.matrix).toBeNull();
    expect(state.actions).toHaveLength(0);
  });

  it('constructs a chance node GameState', () => {
    const state: GameState = {
      street: 'Flop',
      position: '',
      board: ['Ah', 'Kd', '7c'],
      pot: 30,
      stacks: [185, 185],
      matrix: null,
      actions: [],
      action_history: [],
      is_terminal: false,
      is_chance: true,
      solve: null,
    };

    expect(state.is_chance).toBe(true);
    expect(state.board).toHaveLength(3);
  });

  it('constructs a GameState with solve status', () => {
    const solve: SolveStatus = {
      iteration: 50,
      max_iterations: 1000,
      exploitability: 0.05,
      elapsed_secs: 2.5,
      solver_name: 'CfvSubgame',
      is_complete: false,
    };

    const state: GameState = {
      street: 'Flop',
      position: 'BB',
      board: ['Ah', 'Kd', '7c'],
      pot: 30,
      stacks: [185, 185],
      matrix: null,
      actions: [],
      action_history: [],
      is_terminal: false,
      is_chance: false,
      solve,
    };

    expect(state.solve?.iteration).toBe(50);
    expect(state.solve?.is_complete).toBe(false);
    expect(state.solve?.solver_name).toBe('CfvSubgame');
  });

  it('constructs a GameMatrixCell with null ev', () => {
    const cell: GameMatrixCell = {
      hand: 'QQ',
      suited: false,
      pair: true,
      probabilities: [0.0, 0.5, 0.5],
      combo_count: 6,
      weight: 1.0,
      ev: null,
    };

    expect(cell.ev).toBeNull();
    expect(cell.pair).toBe(true);
    expect(cell.probabilities).toHaveLength(3);
  });

  it('constructs a 13x13 matrix grid', () => {
    const makeCell = (hand: string): GameMatrixCell => ({
      hand,
      suited: false,
      pair: false,
      probabilities: [1.0],
      combo_count: 1,
      weight: 1.0,
      ev: null,
    });

    const cells: GameMatrixCell[][] = Array.from({ length: 13 }, (_, r) =>
      Array.from({ length: 13 }, (_, c) => makeCell(`${r},${c}`))
    );

    const matrix: GameMatrix = {
      cells,
      actions: [{ id: '0', label: 'Check', action_type: 'check' }],
    };

    expect(matrix.cells).toHaveLength(13);
    expect(matrix.cells[0]).toHaveLength(13);
    expect(matrix.cells[12][12].hand).toBe('12,12');
  });
});
