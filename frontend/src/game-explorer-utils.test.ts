import { describe, it, expect } from 'vitest';
import { toMatrixCell, boardStreetLabel, expectedCardsForStreet } from './game-explorer-utils';
import type { GameMatrixCell, GameAction } from './game-types';

describe('toMatrixCell', () => {
  it('converts GameMatrixCell probabilities to ActionProb format', () => {
    const cell: GameMatrixCell = {
      hand: 'AKs',
      suited: true,
      pair: false,
      probabilities: [0.3, 0.7],
      combo_count: 4,
      weight: 0.85,
      ev: 1.5,
    };
    const actions: GameAction[] = [
      { id: '0', label: 'Fold', action_type: 'fold' },
      { id: '1', label: 'Call', action_type: 'call' },
    ];

    const result = toMatrixCell(cell, actions);

    expect(result.hand).toBe('AKs');
    expect(result.suited).toBe(true);
    expect(result.pair).toBe(false);
    expect(result.weight).toBe(0.85);
    expect(result.probabilities).toEqual([
      { action: 'Fold', probability: 0.3 },
      { action: 'Call', probability: 0.7 },
    ]);
  });

  it('preserves ev as overlayText field', () => {
    const cell: GameMatrixCell = {
      hand: 'QQ',
      suited: false,
      pair: true,
      probabilities: [1.0],
      combo_count: 6,
      weight: 1.0,
      ev: 2.5,
    };
    const actions: GameAction[] = [
      { id: '0', label: 'Check', action_type: 'check' },
    ];

    const result = toMatrixCell(cell, actions);

    expect(result.ev).toBe(2.5);
  });

  it('handles empty probabilities', () => {
    const cell: GameMatrixCell = {
      hand: '72o',
      suited: false,
      pair: false,
      probabilities: [],
      combo_count: 0,
      weight: 0.0,
      ev: null,
    };

    const result = toMatrixCell(cell, []);

    expect(result.probabilities).toEqual([]);
    expect(result.weight).toBe(0.0);
    expect(result.ev).toBeNull();
  });

  it('handles mismatched action/probability counts gracefully', () => {
    const cell: GameMatrixCell = {
      hand: 'AA',
      suited: false,
      pair: true,
      probabilities: [0.5, 0.5],
      combo_count: 6,
      weight: 1.0,
      ev: null,
    };
    const actions: GameAction[] = [
      { id: '0', label: 'Check', action_type: 'check' },
    ];

    const result = toMatrixCell(cell, actions);

    // Should have as many entries as probabilities, using action labels where available
    expect(result.probabilities).toHaveLength(2);
    expect(result.probabilities[0].action).toBe('Check');
    expect(result.probabilities[1].action).toBe('');
  });
});

describe('boardStreetLabel', () => {
  it('returns FLOP for street after Preflop', () => {
    expect(boardStreetLabel('Preflop')).toBe('FLOP');
  });

  it('returns TURN for street after Flop', () => {
    expect(boardStreetLabel('Flop')).toBe('TURN');
  });

  it('returns RIVER for street after Turn', () => {
    expect(boardStreetLabel('Turn')).toBe('RIVER');
  });

  it('returns empty string for River or unknown', () => {
    expect(boardStreetLabel('River')).toBe('');
    expect(boardStreetLabel('Unknown')).toBe('');
  });
});

describe('expectedCardsForStreet', () => {
  it('returns 3 for FLOP', () => {
    expect(expectedCardsForStreet('FLOP')).toBe(3);
  });

  it('returns 1 for TURN', () => {
    expect(expectedCardsForStreet('TURN')).toBe(1);
  });

  it('returns 1 for RIVER', () => {
    expect(expectedCardsForStreet('RIVER')).toBe(1);
  });

  it('returns 0 for unknown', () => {
    expect(expectedCardsForStreet('Unknown')).toBe(0);
  });
});
