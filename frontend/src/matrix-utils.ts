import { ActionInfo } from './types';

// Suit colors matching the reference image
export const SUIT_COLORS: Record<string, string> = {
  s: '#1a1a2e', // Spades - dark
  h: '#dc2626', // Hearts - red
  d: '#2563eb', // Diamonds - blue
  c: '#16a34a', // Clubs - green
};

export const SUIT_SYMBOLS: Record<string, string> = {
  s: '♠',
  h: '♥',
  d: '♦',
  c: '♣',
};

// Format a postflop EV value (in pot fractions, where 1.0 = the initial pot).
// Display as percentage of pot: +10.5% means hero profits 10.5% of the pot.
export function formatEV(potFraction: number): string {
  const pct = potFraction * 100;
  const sign = pct >= 0 ? '+' : '';
  return `${sign}${pct.toFixed(1)}%`;
}

// Order aggressive actions: sized raises (smallest → largest), then all-in last.
export function sortedBetActions(actions: ActionInfo[]): ActionInfo[] {
  const raises = actions.filter((a) => a.action_type === 'bet' || a.action_type === 'raise');
  const allins = actions.filter((a) => a.action_type === 'allin');
  return [...raises, ...allins];
}

// Return indices into the actions array in display order: fold, check/call, raises by size, all-in last.
export function displayOrderIndices(actions: ActionInfo[]): number[] {
  const passive: number[] = [];
  const raises: number[] = [];
  const allins: number[] = [];
  actions.forEach((a, i) => {
    if (a.action_type === 'fold' || a.action_type === 'check' || a.action_type === 'call') passive.push(i);
    else if (a.action_type === 'allin') allins.push(i);
    else raises.push(i);
  });
  return [...passive, ...raises, ...allins];
}

// Display order: fold, check/call, sized raises, all-in last.
export function displayOrderActions(actions: ActionInfo[]): ActionInfo[] {
  return displayOrderIndices(actions).map((i) => actions[i]);
}

// Format action labels: strip "Raise to"/"Bet" prefix, keep chip amount only.
export function formatActionLabel(action: ActionInfo): string {
  if (action.action_type === 'fold') return 'Fold';
  if (action.action_type === 'check') return 'Check';
  if (action.action_type === 'call') return action.label;
  if (action.action_type === 'allin') return 'All-in';
  return action.label.replace(/^(?:Raise(?: to)? |Bet )\s*/i, '');
}

export function getActionColor(action: ActionInfo, actions: ActionInfo[]): string {
  switch (action.action_type) {
    case 'fold':
      return 'rgba(70, 120, 200, 0.7)'; // Dusty blue
    case 'check':
    case 'call':
      return 'rgba(50, 160, 90, 0.7)'; // Muted green
    case 'bet':
    case 'raise':
    case 'allin': {
      const ordered = sortedBetActions(actions);
      const idx = ordered.findIndex((a) => a.id === action.id);
      const count = ordered.length;
      // Lightest (t=0) → darkest (t=1), all-in always darkest
      const t = count > 1 ? idx / (count - 1) : 1;
      // Interpolate from light coral (220, 120, 110) to deep crimson (160, 30, 30)
      const r = Math.round(220 - t * (220 - 160));
      const g = Math.round(120 - t * (120 - 30));
      const b = Math.round(110 - t * (110 - 30));
      return `rgba(${r}, ${g}, ${b}, 0.7)`;
    }
    default:
      return 'rgba(140, 145, 155, 0.7)'; // Soft gray
  }
}

/**
 * Map (row, col) in the 13x13 matrix to a canonical hand index (0..168).
 * Diagonal = pairs (0..12), above diagonal = suited (13..90), below = offsuit (91..168).
 */
export function matrixToHandIndex(row: number, col: number): number {
  if (row === col) {
    return row; // pair
  } else if (col > row) {
    // suited: above diagonal
    let idx = 0;
    for (let r = 0; r < row; r++) idx += (12 - r);
    idx += (col - row - 1);
    return 13 + idx;
  } else {
    // offsuit: below diagonal (row > col)
    let idx = 0;
    for (let r = 0; r < col; r++) idx += (12 - r);
    idx += (row - col - 1);
    return 91 + idx;
  }
}
