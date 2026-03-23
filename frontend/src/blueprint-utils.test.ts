import { describe, it, expect } from 'vitest';
import { blueprintToPostflopMatrix } from './blueprint-utils';
import { StrategyMatrix } from './types';

function makeStrategyMatrix(overrides?: Partial<StrategyMatrix>): StrategyMatrix {
  return {
    cells: [[{
      hand: 'AA',
      suited: false,
      pair: true,
      probabilities: [
        { action: 'Check', probability: 0.6 },
        { action: 'Bet 33%', probability: 0.4 },
      ],
      weight: 0.7,
    }]],
    actions: [
      { id: 'c', label: 'Check', action_type: 'check' },
      { id: 'r:0', label: 'Bet 33%', action_type: 'bet', size_key: '33%' },
    ],
    street: 'Flop',
    pot: 30,
    stack: 170,
    to_call: 0,
    to_act: 0,
    stack_p1: 170,
    stack_p2: 170,
    reaching_p1: [],
    reaching_p2: [],
    dealer: 0,
    ...overrides,
  };
}

describe('blueprintToPostflopMatrix', () => {
  it('converts probability objects to plain number arrays', () => {
    const sm = makeStrategyMatrix();
    const result = blueprintToPostflopMatrix(sm, ['Ah', '7d', '2c'], 0);

    expect(result.cells[0][0].probabilities).toEqual([0.6, 0.4]);
  });

  it('preserves hand metadata', () => {
    const sm = makeStrategyMatrix();
    const result = blueprintToPostflopMatrix(sm, ['Ah', '7d', '2c'], 0);

    expect(result.cells[0][0].hand).toBe('AA');
    expect(result.cells[0][0].suited).toBe(false);
    expect(result.cells[0][0].pair).toBe(true);
  });

  it('sets combo_count to 1 when any probability is positive', () => {
    const sm = makeStrategyMatrix();
    const result = blueprintToPostflopMatrix(sm, ['Ah', '7d', '2c'], 0);

    expect(result.cells[0][0].combo_count).toBe(1);
  });

  it('sets combo_count to 0 when all probabilities are zero', () => {
    const sm = makeStrategyMatrix({
      cells: [[{
        hand: '72o',
        suited: false,
        pair: false,
        probabilities: [
          { action: 'Check', probability: 0 },
          { action: 'Bet 33%', probability: 0 },
        ],
        weight: 0.0,
      }]],
    });
    const result = blueprintToPostflopMatrix(sm, ['Ah', '7d', '2c'], 0);

    expect(result.cells[0][0].combo_count).toBe(0);
    expect(result.cells[0][0].weight).toBe(0.0);
  });

  it('sets weight to 1.0 for reachable hands and 0.0 for unreachable', () => {
    const sm = makeStrategyMatrix();
    const result = blueprintToPostflopMatrix(sm, ['Ah', '7d', '2c'], 0);

    expect(result.cells[0][0].weight).toBe(1.0);
  });

  it('sets ev to null and combos to empty array', () => {
    const sm = makeStrategyMatrix();
    const result = blueprintToPostflopMatrix(sm, ['Ah', '7d', '2c'], 0);

    expect(result.cells[0][0].ev).toBeNull();
    expect(result.cells[0][0].combos).toEqual([]);
  });

  it('maps pot and stacks from StrategyMatrix fields', () => {
    const sm = makeStrategyMatrix({ pot: 60, stack_p1: 150, stack_p2: 140 });
    const result = blueprintToPostflopMatrix(sm, ['Ah', '7d', '2c'], 1);

    expect(result.pot).toBe(60);
    expect(result.stacks).toEqual([150, 140]);
    expect(result.player).toBe(1);
    expect(result.board).toEqual(['Ah', '7d', '2c']);
  });

  it('passes through actions unchanged', () => {
    const sm = makeStrategyMatrix();
    const result = blueprintToPostflopMatrix(sm, ['Ah', '7d', '2c'], 0);

    expect(result.actions).toBe(sm.actions);
  });

  it('uses to_act from StrategyMatrix to set player', () => {
    const sm = makeStrategyMatrix({ to_act: 1 });
    const result = blueprintToPostflopMatrix(sm, ['Ah', '7d', '2c'], sm.to_act);

    expect(result.player).toBe(1);
  });

  it('defaults to_act of 0 for OOP player', () => {
    const sm = makeStrategyMatrix({ to_act: 0 });
    const result = blueprintToPostflopMatrix(sm, ['Ah', '7d', '2c'], sm.to_act);

    expect(result.player).toBe(0);
  });
});
