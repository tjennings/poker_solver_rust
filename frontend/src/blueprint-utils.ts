import { StrategyMatrix, PostflopStrategyMatrix } from './types';

/** Convert a blueprint StrategyMatrix to PostflopStrategyMatrix for rendering. */
export function blueprintToPostflopMatrix(
  sm: StrategyMatrix,
  board: string[],
  player: number,
): PostflopStrategyMatrix {
  return {
    cells: sm.cells.map(row =>
      row.map(cell => {
        const hasAction = cell.probabilities.some(ap => ap.probability > 0);
        return {
          hand: cell.hand,
          suited: cell.suited,
          pair: cell.pair,
          probabilities: cell.probabilities.map(ap => ap.probability),
          combo_count: hasAction ? 1 : 0,
          ev: null,
          combos: [],
          weight: hasAction ? 1.0 : 0.0,
        };
      }),
    ),
    actions: sm.actions,
    player,
    pot: sm.pot,
    stacks: [sm.stack_p1, sm.stack_p2],
    board,
  };
}
