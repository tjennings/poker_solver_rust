import type { MatrixCell } from './types';
import type { GameMatrixCell, GameAction } from './game-types';

/**
 * Convert a GameMatrixCell (backend format with number[] probabilities)
 * to a MatrixCell (existing UI format with ActionProb[] probabilities)
 * so we can reuse the existing HandCell component.
 */
export function toMatrixCell(cell: GameMatrixCell, actions: GameAction[]): MatrixCell {
  return {
    hand: cell.hand,
    suited: cell.suited,
    pair: cell.pair,
    weight: cell.weight,
    ev: cell.ev,
    probabilities: cell.probabilities.map((p, i) => ({
      action: actions[i]?.label ?? '',
      probability: p,
    })),
  };
}

/**
 * Given the current street name, return the label for the next street's
 * card picker (e.g. "Preflop" -> "FLOP").
 */
export function boardStreetLabel(currentStreet: string): string {
  switch (currentStreet) {
    case 'Preflop': return 'FLOP';
    case 'Flop': return 'TURN';
    case 'Turn': return 'RIVER';
    default: return '';
  }
}

/**
 * Number of cards expected for a given street label.
 */
export function expectedCardsForStreet(streetLabel: string): number {
  switch (streetLabel) {
    case 'FLOP': return 3;
    case 'TURN': return 1;
    case 'RIVER': return 1;
    default: return 0;
  }
}
