import { useState, useCallback, useMemo, useEffect } from 'react';
import { invoke } from './invoke';
import {
  BundleInfo,
  CanonicalizeResult,
  HandEquity,
  StrategyMatrix,
  ExplorationPosition,
  ActionInfo,
  MatrixCell,
  ComboGroupInfo,
} from './types';

// Suit colors matching the reference image
const SUIT_COLORS: Record<string, string> = {
  s: '#1a1a2e', // Spades - dark
  h: '#dc2626', // Hearts - red
  d: '#2563eb', // Diamonds - blue
  c: '#16a34a', // Clubs - green
};

const SUIT_SYMBOLS: Record<string, string> = {
  s: '♠',
  h: '♥',
  d: '♦',
  c: '♣',
};

// Color utilities for action probabilities.
// Bet/raise actions use a graduated red scale: lightest for the smallest
// bet size, darkest for all-in.
// Format a postflop EV value (in pot fractions, where 1.0 = the initial pot).
// Display as percentage of pot: +10.5% means hero profits 10.5% of the pot.
function formatEV(potFraction: number): string {
  const pct = potFraction * 100;
  const sign = pct >= 0 ? '+' : '';
  return `${sign}${pct.toFixed(1)}%`;
}

// Order aggressive actions: sized raises (smallest → largest), then all-in last.
function sortedBetActions(actions: ActionInfo[]): ActionInfo[] {
  const raises = actions.filter((a) => a.action_type === 'bet' || a.action_type === 'raise');
  const allins = actions.filter((a) => a.action_type === 'allin');
  return [...raises, ...allins];
}

// Return indices into the actions array in display order: fold, check/call, raises by size, all-in last.
function displayOrderIndices(actions: ActionInfo[]): number[] {
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
function displayOrderActions(actions: ActionInfo[]): ActionInfo[] {
  return displayOrderIndices(actions).map((i) => actions[i]);
}

// Format action labels: strip "Raise"/"Bet" prefix, use pot-relative sizing.
function formatActionLabel(action: ActionInfo): string {
  if (action.action_type === 'fold') return 'Fold';
  if (action.action_type === 'check') return 'Check';
  if (action.action_type === 'call') return action.label; // "Call 2" etc.
  if (action.action_type === 'allin') return 'All-in';
  // Bet/raise — use size_key to show pot-relative label
  if (action.size_key) {
    const frac = parseFloat(action.size_key);
    if (!isNaN(frac)) {
      if (frac === 1.0) return 'Pot';
      if (frac < 1.0) return `${Math.round(frac * 100)}%`;
      return `${frac}x`;
    }
  }
  return action.label;
}

function getActionColor(action: ActionInfo, actions: ActionInfo[]): string {
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
      // Interpolate from dusty rose (200, 130, 130) to wine (160, 50, 50)
      const r = Math.round(200 - t * (200 - 160));
      const g = Math.round(130 - t * (130 - 50));
      const b = Math.round(130 - t * (130 - 50));
      return `rgba(${r}, ${g}, ${b}, 0.7)`;
    }
    default:
      return 'rgba(140, 145, 155, 0.7)'; // Soft gray
  }
}

// Hand matrix cell component
function HandCell({
  cell,
  actions,
  isSelected,
  onClick,
}: {
  cell: MatrixCell;
  actions: ActionInfo[];
  isSelected: boolean;
  onClick: () => void;
}) {
  const gradientStops = useMemo(() => {
    if (cell.probabilities.length === 0) {
      return 'rgba(30, 41, 59, 1)';
    }

    const stops: string[] = [];
    let position = 0;

    // Reverse order: all-in / largest raise first (left), fold last (right)
    for (let idx = cell.probabilities.length - 1; idx >= 0; idx--) {
      const prob = cell.probabilities[idx];
      const action = actions[idx];
      if (!action || prob.probability <= 0) continue;

      const color = getActionColor(action, actions);
      const width = prob.probability * 100;
      stops.push(`${color} ${position}%`);
      stops.push(`${color} ${position + width}%`);
      position += width;
    }

    if (stops.length === 0) {
      return 'rgba(30, 41, 59, 1)';
    }

    return `linear-gradient(to right, ${stops.join(', ')})`;
  }, [cell.probabilities, actions]);

  return (
    <div
      className={`matrix-cell ${isSelected ? 'selected' : ''} ${cell.filtered ? 'filtered' : ''}`}
      style={{ background: cell.filtered ? undefined : gradientStops }}
      onClick={onClick}
    >
      <span className="cell-label">{cell.hand}</span>
    </div>
  );
}

// Detail panel showing action frequencies for a selected cell
function CellDetail({
  cell,
  actions,
}: {
  cell: MatrixCell;
  actions: ActionInfo[];
}) {
  return (
    <div className="cell-detail">
      <div className="cell-detail-header">{cell.hand}</div>
      <div className="cell-detail-actions">
        {displayOrderIndices(actions).map((idx) => {
          const prob = cell.probabilities[idx];
          const action = actions[idx];
          if (!action || !prob) return null;
          const pct = prob.probability * 100;
          return (
            <div key={action.id} className="cell-detail-row">
              <span className="cell-detail-label">{formatActionLabel(action)}</span>
              <span className="cell-detail-pct">{pct.toFixed(1)}%</span>
              <div className="cell-detail-bar-bg">
                <div
                  className="cell-detail-bar-fill"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: getActionColor(action, actions),
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Panel showing per-combo classification breakdown for a selected cell
function ComboClassPanel({
  info,
  actions,
}: {
  info: ComboGroupInfo;
  actions: ActionInfo[];
}) {
  return (
    <div className="combo-panel">
      <div className="combo-panel-header">
        {info.hand}
        <span className="combo-panel-meta">
          {info.total_combos} combo{info.total_combos !== 1 ? 's' : ''}
          {info.blocked_combos > 0 && ` (${info.blocked_combos} blocked)`}
        </span>
      </div>
      {info.groups.length === 0 && (
        <div className="combo-panel-empty">No combos on this board.</div>
      )}
      <div className="combo-panel-list">
        {info.groups.map((group) => (
          <div key={group.bits} className="combo-group">
            <div className="combo-group-tags">
              {group.class_names.length > 0
                ? group.class_names.map((name) => (
                    <span key={name} className="combo-group-tag">{name}</span>
                  ))
                : <span className="combo-group-tag empty">No class</span>
              }
              <span className="combo-group-count">
                {group.combos.length}
              </span>
            </div>
            <div className="combo-group-combos">
              {group.combos.join(', ')}
            </div>
            <div className="combo-group-bars">
              {[...group.strategy].map((_, ri) => {
                const i = group.strategy.length - 1 - ri;
                const prob = group.strategy[i];
                const action = actions[i];
                if (!action || prob <= 0) return null;
                return (
                  <div
                    key={action.id}
                    className="combo-group-bar"
                    style={{
                      width: `${prob * 100}%`,
                      backgroundColor: getActionColor(action, actions),
                    }}
                    title={`${action.label}: ${(prob * 100).toFixed(1)}%`}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Action block in the navigation strip
function ActionBlock({
  position,
  stack,
  pot,
  actions,
  selectedAction,
  onSelect,
  onHeaderClick,
  isCurrent,
}: {
  position: string;
  stack: number;
  pot: number;
  actions: ActionInfo[];
  selectedAction?: string;
  onSelect: (actionId: string) => void;
  onHeaderClick?: () => void;
  isCurrent: boolean;
}) {
  return (
    <div className={`action-block ${isCurrent ? 'current' : ''}`}>
      <div
        className={`action-block-header ${onHeaderClick ? 'clickable' : ''}`}
        onClick={onHeaderClick}
      >
        <span className="position">{position}</span>
        <span className="stack">{isNaN(stack) ? '--' : `${Math.round(stack / 2)}BB`} / {isNaN(pot) ? '--' : `${Math.round(pot / 2)}BB`}</span>
      </div>
      <div className="action-list">
        {displayOrderActions(actions).map((action) => {
          const color = getActionColor(action, actions);
          return (
            <button
              key={action.id}
              className={`action-button ${action.action_type} ${
                selectedAction === action.id ? 'selected' : ''
              }`}
              style={{ borderLeft: `3px solid ${color}` }}
              onClick={() => onSelect(action.id)}
            >
              {formatActionLabel(action)}
            </button>
          );
        })}
      </div>
    </div>
  );
}

// Large card display for street blocks
function StreetCard({ card, onClick }: { card: string | null; onClick?: () => void }) {
  if (!card) {
    return (
      <div className="street-card empty" onClick={onClick}>
        <span>?</span>
      </div>
    );
  }

  const rank = card[0]?.toUpperCase();
  const suit = card[1]?.toLowerCase();
  const bgColor = SUIT_COLORS[suit] || '#333';

  return (
    <div
      className="street-card"
      style={{ backgroundColor: bgColor }}
      onClick={onClick}
    >
      <span className="card-rank">{rank}</span>
      <span className="card-suit">{SUIT_SYMBOLS[suit] || '?'}</span>
    </div>
  );
}

const PICKER_RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];
const PICKER_SUITS = ['s', 'h', 'd', 'c'];

// Text colors for the card picker (spades white instead of dark background color)
const PICKER_COLORS: Record<string, string> = {
  s: '#fff',
  h: '#dc2626',
  d: '#2563eb',
  c: '#16a34a',
};

function CardPicker({
  expectedCards,
  deadCards,
  onConfirm,
}: {
  expectedCards: number;
  deadCards: string[];
  onConfirm: (cards: string[]) => void;
}) {
  const [selected, setSelected] = useState<string[]>([]);

  const deadSet = useMemo(() => new Set(deadCards.map((c) => c.toLowerCase())), [deadCards]);

  const handleCardClick = (card: string) => {
    if (deadSet.has(card)) return;

    setSelected((prev) => {
      if (prev.includes(card)) {
        return prev.filter((c) => c !== card);
      }
      if (prev.length >= expectedCards) return prev;
      const next = [...prev, card];
      // Auto-confirm for single card (turn/river)
      if (expectedCards === 1 && next.length === 1) {
        // Use setTimeout to avoid state update during render
        setTimeout(() => onConfirm(next), 0);
      }
      return next;
    });
  };

  return (
    <div className="card-picker">
      {PICKER_SUITS.map((suit) => (
        <div key={suit} className="card-picker-row">
          {PICKER_RANKS.map((rank) => {
            const card = `${rank}${suit}`;
            const isDead = deadSet.has(card.toLowerCase());
            const isSelected = selected.includes(card);
            return (
              <button
                key={card}
                className={`card-picker-card ${isDead ? 'dead' : ''} ${isSelected ? 'selected' : ''}`}
                style={{
                  color: PICKER_COLORS[suit] || '#eee',
                  borderColor: isSelected ? '#00d9ff' : 'transparent',
                }}
                disabled={isDead}
                onClick={() => handleCardClick(card)}
              >
                <span className="picker-rank">{rank}</span>
                <span className="picker-suit">{SUIT_SYMBOLS[suit]}</span>
              </button>
            );
          })}
        </div>
      ))}
      {expectedCards > 1 && (
        <button
          className="card-picker-confirm"
          disabled={selected.length !== expectedCards}
          onClick={() => onConfirm(selected)}
        >
          Deal {selected.length}/{expectedCards}
        </button>
      )}
    </div>
  );
}

// Street transition block (FLOP, TURN, RIVER)
function StreetBlock({
  street,
  pot,
  cards,
  expectedCards,
  onHeaderClick,
  remapInfo,
}: {
  street: string;
  pot: number;
  cards: string[];
  expectedCards: number;
  onHeaderClick?: () => void;
  remapInfo?: { original: string[]; canonical: string[]; suitMap: Record<string, string> } | null;
}) {
  const cardSlots = [];
  for (let i = 0; i < expectedCards; i++) {
    cardSlots.push(
      <StreetCard
        key={i}
        card={cards[i] || null}
      />
    );
  }

  const hasRemap = remapInfo && Object.keys(remapInfo.suitMap).length > 0;

  // Extract original cards for this street from the accumulated remap info
  const originalCards = hasRemap
    ? remapInfo!.original.slice(
        street === 'FLOP' ? 0 : street === 'TURN' ? 3 : 4,
        street === 'FLOP' ? 3 : street === 'TURN' ? 4 : 5,
      )
    : [];
  // Only show if any card actually differs
  const showOriginal = originalCards.length > 0 &&
    originalCards.some((orig, i) => orig !== cards[i]);

  return (
    <div className={`street-block ${cards.length === expectedCards ? '' : 'pending'}`}>
      <div
        className={`street-block-header ${onHeaderClick ? 'clickable' : ''}`}
        onClick={onHeaderClick}
      >
        <span className="street-name">{street}</span>
        <span className="street-pot">{pot}</span>
        {hasRemap && <span className="remap-indicator">↔</span>}
      </div>
      <div className="street-cards">
        {cardSlots}
      </div>
      {showOriginal && (
        <div className="street-cards original">
          {originalCards.map((card, i) => (
            <StreetCard key={i} card={card} />
          ))}
        </div>
      )}
    </div>
  );
}

// History item type
type HistoryItem =
  | { type: 'action'; position: string; stack: number; pot: number; actions: ActionInfo[]; selected: string }
  | { type: 'street'; street: string; pot: number; stack_p1: number; stack_p2: number; cards: string[] };

// Extract completed-street action sequences from history items.
function extractStreetHistories(items: HistoryItem[]): string[][] {
  const histories: string[][] = [];
  let current: string[] = [];

  for (const item of items) {
    if (item.type === 'action') {
      current.push(actionIdToHistoryEntry(item.selected));
    } else if (item.type === 'street') {
      if (current.length > 0) {
        histories.push(current);
        current = [];
      }
    }
  }
  // Don't push `current` — those are current-street actions already in position.history
  return histories;
}

function actionIdToHistoryEntry(selected: string): string {
  if (selected === 'call') return 'c';
  if (selected === 'check') return 'x';
  if (selected === 'fold') return 'f';
  if (selected.startsWith('bet:')) return `b:${selected.split(':')[1]}`;
  if (selected.startsWith('raise:')) return `r:${selected.split(':')[1]}`;
  return selected;
}

export default function Explorer() {
  const [bundleInfo, setBundleInfo] = useState<BundleInfo | null>(null);
  const [matrix, setMatrix] = useState<StrategyMatrix | null>(null);
  const [position, setPosition] = useState<ExplorationPosition>({
    board: [],
    history: [],
    pot: 3,
    stacks: [199, 198],
    stack_p1: 199,
    stack_p2: 198,
    to_act: 0,
    num_players: 2,
    active_players: [true, true],
  });
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [pendingStreet, setPendingStreet] = useState<{
    street: string;
    pot: number;
    stack_p1: number;
    stack_p2: number;
    expectedCards: number;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedCell, setSelectedCell] = useState<{ row: number; col: number } | null>(null);
  const [handResult, setHandResult] = useState<{ type: 'fold' | 'showdown'; pot: number } | null>(null);
  const [comboInfo, setComboInfo] = useState<ComboGroupInfo | null>(null);
  const [handEquity, setHandEquity] = useState<HandEquity | null>(null);
  const [villainHand, setVillainHand] = useState('AA');
  const [threshold, _setThreshold] = useState(2);
  const [remapInfo, setRemapInfo] = useState<{
    original: string[];
    canonical: string[];
    suitMap: Record<string, string>;
  } | null>(null);

  // Re-fetch matrix when threshold changes (if a matrix is displayed)
  useEffect(() => {
    if (!matrix) return;
    invoke<StrategyMatrix>('get_strategy_matrix', {
      position,
      threshold: threshold / 100,
      street_histories: extractStreetHistories(historyItems),
    })
      .then(setMatrix)
      .catch((e) => setError(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threshold]);

  // Fetch combo classification when a cell is selected on postflop
  useEffect(() => {
    if (!matrix || !selectedCell || position.board.length === 0) {
      setComboInfo(null);
      return;
    }
    const cell = matrix.cells[selectedCell.row]?.[selectedCell.col];
    if (!cell || cell.filtered) {
      setComboInfo(null);
      return;
    }

    invoke<ComboGroupInfo>('get_combo_classes', { position, hand: cell.hand })
      .then(setComboInfo)
      .catch((err) => {
        console.error('get_combo_classes error:', err);
        setComboInfo(null);
      });
  }, [position, matrix, selectedCell]);

  // Fetch postflop equity when a cell is selected on preflop view
  useEffect(() => {
    if (!matrix || !selectedCell || !bundleInfo?.preflop_only) {
      setHandEquity(null);
      return;
    }
    const cell = matrix.cells[selectedCell.row]?.[selectedCell.col];
    if (!cell || cell.filtered) {
      setHandEquity(null);
      return;
    }

    invoke<HandEquity | null>('get_hand_equity', {
      hand: cell.hand,
      villain_hand: villainHand || null,
    })
      .then(setHandEquity)
      .catch(() => setHandEquity(null));
  }, [matrix, selectedCell, bundleInfo, villainHand]);

  const loadSource = useCallback(
    async (path: string) => {
      setLoading(true);
      setError(null);
      try {
        const info = await invoke<BundleInfo>('load_bundle', { path });
        setBundleInfo(info);

        const sp1 = info.stack_depth * 2 - 1;
        const sp2 = info.stack_depth * 2 - 2;
        const initialPosition: ExplorationPosition = {
          board: [],
          history: [],
          pot: 3,
          stacks: [sp1, sp2],
          stack_p1: sp1,
          stack_p2: sp2,
          to_act: 0,
          num_players: 2,
          active_players: [true, true],
        };
        setPosition(initialPosition);
        setHistoryItems([]);
        setPendingStreet(null);
        setHandResult(null);
        setSelectedCell(null);
        const initialMatrix = await invoke<StrategyMatrix>('get_strategy_matrix', {
          position: initialPosition,
          threshold: threshold / 100,
        });
        setMatrix(initialMatrix);
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    },
    [threshold]
  );

  const handleLoadDataset = useCallback(async () => {
    try {
      let path: string | null = null;
      if ('__TAURI__' in window) {
        const { open } = await import('@tauri-apps/plugin-dialog');
        path = await open({ directory: true, title: 'Select Dataset Directory' });
      } else {
        path = window.prompt('Enter dataset directory path:');
      }
      if (path) {
        loadSource(path);
      }
    } catch (e) {
      setError(String(e));
    }
  }, [loadSource]);

  // Check if current betting round is complete (needs street transition)
  const checkStreetTransition = useCallback(
    (history: string[], currentStreet: string, preflopOnly: boolean): { needsTransition: boolean; nextStreet: string } => {
      if (history.length < 2) return { needsTransition: false, nextStreet: '' };

      const lastTwo = history.slice(-2);

      const isCallAfterBetOrRaise =
        lastTwo[1] === 'c' && (lastTwo[0].startsWith('r:') || lastTwo[0].startsWith('b:'));

      const isBothCheck = lastTwo[0] === 'x' && lastTwo[1] === 'x';

      const isPreflopLimp = currentStreet === 'Preflop' && lastTwo[0] === 'c' && lastTwo[1] === 'x';

      if (isCallAfterBetOrRaise || isBothCheck || isPreflopLimp) {
        if (currentStreet === 'Preflop') {
          if (preflopOnly) {
            return { needsTransition: true, nextStreet: '' };
          }
          return { needsTransition: true, nextStreet: 'FLOP' };
        }
        if (currentStreet === 'Flop') return { needsTransition: true, nextStreet: 'TURN' };
        if (currentStreet === 'Turn') return { needsTransition: true, nextStreet: 'RIVER' };
        if (currentStreet === 'River') return { needsTransition: true, nextStreet: '' };
      }

      return { needsTransition: false, nextStreet: '' };
    },
    []
  );

  const handleActionSelect = useCallback(
    async (actionId: string) => {
      if (!matrix) return;

      try {
        setLoading(true);

        let newHistory = [...position.history];
        let newToAct = position.to_act === 0 ? 1 : 0;

        let historyEntry: string;

        if (actionId === 'call') {
          historyEntry = 'c';
        } else if (actionId === 'check') {
          historyEntry = 'x';
        } else if (actionId === 'fold') {
          historyEntry = 'f';
        } else if (actionId.startsWith('bet:') || actionId.startsWith('raise:')) {
          // actionId is e.g. "bet:0", "raise:3", "bet:A" — store index in history
          const idx = actionId.split(':')[1];
          const prefix = actionId.startsWith('bet:') ? 'b' : 'r';
          historyEntry = `${prefix}:${idx}`;
        } else {
          historyEntry = actionId;
        }

        newHistory.push(historyEntry);

        // Add action to history
        const actionItem: HistoryItem = {
          type: 'action',
          position: position.to_act === 0 ? 'SB' : 'BB',
          stack: matrix.stack,
          pot: matrix.pot,
          actions: matrix.actions,
          selected: actionId,
        };
        setHistoryItems((prev) => [...prev, actionItem]);

        // Check for terminal states
        if (actionId === 'fold') {
          setPosition((prev) => ({
            ...prev,
            history: newHistory,
          }));
          setMatrix(null);
          setHandResult({ type: 'fold', pot: matrix.pot });
          return;
        }

        // Check for street transition
        const { needsTransition, nextStreet } = checkStreetTransition(
          newHistory,
          matrix.street,
          bundleInfo?.preflop_only ?? false
        );

        if (needsTransition && nextStreet) {
          // Compute pot/stacks entering the new street.
          // A call adds to_call to pot and subtracts from the caller's stack.
          const transitionPot =
            actionId === 'call' ? matrix.pot + matrix.to_call : matrix.pot;
          let sp1 = matrix.stack_p1;
          let sp2 = matrix.stack_p2;
          if (actionId === 'call') {
            if (position.to_act === 0) { sp1 -= matrix.to_call; }
            else { sp2 -= matrix.to_call; }
          }
          const expectedCards = nextStreet === 'FLOP' ? 3 : 1;
          setPendingStreet({ street: nextStreet, pot: transitionPot, stack_p1: sp1, stack_p2: sp2, expectedCards });
          setPosition((prev) => ({
            ...prev,
            history: newHistory,
          }));
          setMatrix(null);
        } else if (needsTransition && !nextStreet) {
          // Betting complete — showdown (or preflop-only terminal)
          const finalPot = actionId === 'call'
            ? matrix.pot + matrix.to_call
            : matrix.pot;
          setPosition((prev) => ({
            ...prev,
            history: newHistory,
          }));
          setMatrix(null);
          setHandResult({ type: 'showdown', pot: finalPot });
        } else {
          const newPosition: ExplorationPosition = {
            ...position,
            history: newHistory,
            to_act: newToAct as 0 | 1,
          };
          setPosition(newPosition);

          const newMatrix = await invoke<StrategyMatrix>('get_strategy_matrix', {
            position: newPosition,
            threshold: threshold / 100,
            street_histories: extractStreetHistories(historyItems),
          });
          setMatrix(newMatrix);
        }
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    },
    [matrix, position, checkStreetTransition, threshold, historyItems, bundleInfo]
  );

  const handleStreetCardsSet = useCallback(
    async (cards: string[]) => {
      if (!pendingStreet) return;

      try {
        setLoading(true);

        // Canonicalize the board cards via backend
        const result = await invoke<CanonicalizeResult>('canonicalize_board', { cards });
        const canonicalCards = result.canonical_cards;

        // Track remap info for the indicator.
        // On flop: establish if remapped. On turn/river: always accumulate if a mapping exists.
        setRemapInfo((prev) => {
          if (result.remapped && result.suit_map) {
            // This street introduced new suit substitutions
            return {
              original: [...(prev?.original ?? []), ...cards],
              canonical: [...(prev?.canonical ?? []), ...canonicalCards],
              suitMap: { ...(prev?.suitMap ?? {}), ...result.suit_map },
            };
          }
          if (prev) {
            // Mapping exists from a prior street — accumulate even if this card didn't change
            return {
              ...prev,
              original: [...prev.original, ...cards],
              canonical: [...prev.canonical, ...canonicalCards],
            };
          }
          return prev;
        });

        // Add street to history (use canonical cards)
        const streetItem: HistoryItem = {
          type: 'street',
          street: pendingStreet.street,
          pot: pendingStreet.pot,
          stack_p1: pendingStreet.stack_p1,
          stack_p2: pendingStreet.stack_p2,
          cards: canonicalCards,
        };
        setHistoryItems((prev) => [...prev, streetItem]);

        // Update board, reset history for new street.
        // Use pot/stacks from pendingStreet (computed at transition time).
        const newBoard = [...position.board, ...canonicalCards];
        const newPosition: ExplorationPosition = {
          board: newBoard,
          history: [],
          pot: pendingStreet.pot,
          stacks: [pendingStreet.stack_p1, pendingStreet.stack_p2],
          stack_p1: pendingStreet.stack_p1,
          stack_p2: pendingStreet.stack_p2,
          to_act: 0, // OOP acts first postflop
          num_players: 2,
          active_players: [true, true],
        };
        setPosition(newPosition);
        setPendingStreet(null);

        await invoke('start_bucket_computation', { board: newBoard });

        // Compute street histories: existing completed streets + the just-completed street.
        // position.history is already in short format (e.g. "r:0", "c").
        const sh = [...extractStreetHistories(historyItems), [...position.history]];

        const newMatrix = await invoke<StrategyMatrix>('get_strategy_matrix', {
          position: newPosition,
          threshold: threshold / 100,
          street_histories: sh,
        });
        setMatrix(newMatrix);
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    },
    [pendingStreet, position, threshold, historyItems]
  );

  // Rebuild position from history items up to (but not including) the given index.
  const rebuildState = useCallback(
    (index: number) => {
      const items = historyItems.slice(0, index);
      let board: string[] = [];
      let history: string[] = [];
      let streetActionCount = 0;
      // Start with preflop initial values
      let pot = 3;
      let sp1 = bundleInfo ? bundleInfo.stack_depth * 2 - 1 : 199;
      let sp2 = bundleInfo ? bundleInfo.stack_depth * 2 - 2 : 198;

      for (const item of items) {
        if (item.type === 'action') {
          let entry: string;
          if (item.selected === 'call') entry = 'c';
          else if (item.selected === 'check') entry = 'x';
          else if (item.selected === 'fold') entry = 'f';
          else if (item.selected.startsWith('bet:'))
            entry = `b:${item.selected.split(':')[1]}`;
          else if (item.selected.startsWith('raise:'))
            entry = `r:${item.selected.split(':')[1]}`;
          else entry = item.selected;
          history.push(entry);
          streetActionCount++;
        } else if (item.type === 'street') {
          board = [...board, ...item.cards];
          pot = item.pot;
          sp1 = item.stack_p1;
          sp2 = item.stack_p2;
          history = [];
          streetActionCount = 0;
        }
      }

      const pos: ExplorationPosition = {
        board,
        history,
        pot,
        stacks: [sp1, sp2],
        stack_p1: sp1,
        stack_p2: sp2,
        to_act: (streetActionCount % 2) as 0 | 1,
        num_players: 2,
        active_players: [true, true],
      };

      return { items, pos };
    },
    [historyItems, bundleInfo]
  );

  const handleHistoryRewind = useCallback(
    async (index: number) => {
      try {
        setLoading(true);
        const { items, pos } = rebuildState(index);
        setHistoryItems(items);
        setPendingStreet(null);
        setHandResult(null);
        setPosition(pos);

        const newMatrix = await invoke<StrategyMatrix>('get_strategy_matrix', {
          position: pos,
          threshold: threshold / 100,
          street_histories: extractStreetHistories(items),
        });
        setMatrix(newMatrix);
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    },
    [rebuildState, threshold]
  );

  // Rewind to a street transition, re-showing the card picker for that street.
  const handleStreetRewind = useCallback(
    (index: number) => {
      const streetItem = historyItems[index];
      if (!streetItem || streetItem.type !== 'street') return;

      // Clear remap info when rewinding to the flop (will be re-established on card pick)
      if (streetItem.street === 'FLOP') {
        setRemapInfo(null);
      }

      const { items, pos } = rebuildState(index);
      setHistoryItems(items);
      setHandResult(null);
      setMatrix(null);
      setPosition(pos);
      setPendingStreet({
        street: streetItem.street,
        pot: streetItem.pot,
        stack_p1: streetItem.stack_p1,
        stack_p2: streetItem.stack_p2,
        expectedCards: streetItem.cards.length,
      });
    },
    [historyItems, rebuildState]
  );

  const handleNewHand = useCallback(async () => {
    if (!bundleInfo) return;
    try {
      setLoading(true);
      const sp1 = bundleInfo.stack_depth * 2 - 1;
      const sp2 = bundleInfo.stack_depth * 2 - 2;
      const initialPosition: ExplorationPosition = {
        board: [],
        history: [],
        pot: 3,
        stacks: [sp1, sp2],
        stack_p1: sp1,
        stack_p2: sp2,
        to_act: 0,
        num_players: 2,
        active_players: [true, true],
      };
      setPosition(initialPosition);
      setHistoryItems([]);
      setPendingStreet(null);
      setHandResult(null);
      setSelectedCell(null);
      setComboInfo(null);
      setRemapInfo(null);
      const newMatrix = await invoke<StrategyMatrix>('get_strategy_matrix', {
        position: initialPosition,
        threshold: threshold / 100,
      });
      setMatrix(newMatrix);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [bundleInfo, threshold]);

  return (
    <div className="explorer">
      {error && <div className="error">{error}</div>}

      {bundleInfo && (
        <>
          <div className="action-strip">
            <div className="dataset-switcher" onClick={handleLoadDataset} title="Load Dataset">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                <line x1="12" y1="11" x2="12" y2="17" />
                <line x1="9" y1="14" x2="15" y2="14" />
              </svg>
            </div>
            {historyItems.map((item, idx) =>
              item.type === 'action' ? (
                <ActionBlock
                  key={idx}
                  position={item.position}
                  stack={item.stack}
                  pot={item.pot}
                  actions={item.actions}
                  selectedAction={item.selected}
                  onSelect={() => handleHistoryRewind(idx)}
                  onHeaderClick={() => handleHistoryRewind(idx)}
                  isCurrent={false}
                />
              ) : (
                <StreetBlock
                  key={idx}
                  street={item.street}
                  pot={item.pot}
                  cards={item.cards}
                  expectedCards={item.cards.length}
                  onHeaderClick={() => handleStreetRewind(idx)}
                  remapInfo={remapInfo}
                />
              )
            )}

            {pendingStreet && (
              <StreetBlock
                street={pendingStreet.street}
                pot={pendingStreet.pot}
                cards={[]}
                expectedCards={pendingStreet.expectedCards}
              />
            )}

            {matrix && !pendingStreet && (
              <ActionBlock
                position={position.to_act === 0 ? 'SB' : 'BB'}
                stack={matrix.stack}
                pot={matrix.pot}
                actions={matrix.actions}
                onSelect={handleActionSelect}
                isCurrent={true}
              />
            )}
          </div>

          {matrix && (
            <div className="matrix-container">
              <div className="matrix-with-detail">
                <div className="hand-matrix">
                  {matrix.cells.map((row, rowIdx) => (
                    <div key={rowIdx} className="matrix-row">
                      {row.map((cell, colIdx) => (
                        <HandCell
                          key={colIdx}
                          cell={cell}
                          actions={matrix.actions}
                          isSelected={selectedCell?.row === rowIdx && selectedCell?.col === colIdx}
                          onClick={() => setSelectedCell({ row: rowIdx, col: colIdx })}
                        />
                      ))}
                    </div>
                  ))}
                </div>
                <div className="detail-column">
                  {selectedCell && matrix.cells[selectedCell.row]?.[selectedCell.col] && (
                    <CellDetail
                      cell={matrix.cells[selectedCell.row][selectedCell.col]}
                      actions={matrix.actions}
                    />
                  )}
                  {handEquity && (
                    <div className="hand-equity-panel">
                      <div className="cell-detail-header">Postflop EV (% of pot)</div>
                      <div className="hand-equity-rows">
                        <div className="hand-equity-subheader">vs Range</div>
                        <div className="hand-equity-row">
                          <span className="hand-equity-label">As SB</span>
                          <span className="hand-equity-value">{formatEV(handEquity.ev_pos0)}</span>
                        </div>
                        <div className="hand-equity-row">
                          <span className="hand-equity-label">As BB</span>
                          <span className="hand-equity-value">{formatEV(handEquity.ev_pos1)}</span>
                        </div>
                        <div className="hand-equity-row hand-equity-avg">
                          <span className="hand-equity-label">Average</span>
                          <span className="hand-equity-value">{formatEV(handEquity.ev_avg)}</span>
                        </div>
                      </div>
                      <div className="hand-equity-vs-hand">
                        <div className="hand-equity-subheader">
                          vs
                          <input
                            type="text"
                            className="villain-hand-input"
                            value={villainHand}
                            onChange={(e) => setVillainHand(e.target.value.toUpperCase())}
                            placeholder="AA"
                            maxLength={3}
                          />
                        </div>
                        {handEquity.ev_vs_hand ? (
                          <div className="hand-equity-rows">
                            <div className="hand-equity-row">
                              <span className="hand-equity-label">As SB</span>
                              <span className="hand-equity-value">{formatEV(handEquity.ev_vs_hand.ev_pos0)}</span>
                            </div>
                            <div className="hand-equity-row">
                              <span className="hand-equity-label">As BB</span>
                              <span className="hand-equity-value">{formatEV(handEquity.ev_vs_hand.ev_pos1)}</span>
                            </div>
                            <div className="hand-equity-row hand-equity-avg">
                              <span className="hand-equity-label">Average</span>
                              <span className="hand-equity-value">{formatEV(handEquity.ev_vs_hand.ev_avg)}</span>
                            </div>
                          </div>
                        ) : (
                          <div className="hand-equity-row">
                            <span className="hand-equity-label" style={{ color: '#666' }}>Invalid hand</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                  {comboInfo && comboInfo.groups.length > 0 && (
                    <ComboClassPanel info={comboInfo} actions={matrix.actions} />
                  )}
                </div>
              </div>
            </div>
          )}

          {!matrix && pendingStreet && (
            <div className="card-picker-container">
              <p className="card-picker-prompt">
                Select {pendingStreet.street.toLowerCase()} card{pendingStreet.expectedCards > 1 ? 's' : ''}
              </p>
              <CardPicker
                expectedCards={pendingStreet.expectedCards}
                deadCards={position.board}
                onConfirm={handleStreetCardsSet}
              />
            </div>
          )}

          {handResult && (
            <div className="hand-complete">
              <p className="hand-complete-result">
                {handResult.type === 'fold' ? 'Player folded' : 'Showdown'} — Pot: {handResult.pot}
              </p>
              <button className="new-hand-btn" onClick={handleNewHand}>
                New Hand
              </button>
            </div>
          )}
        </>
      )}

      {!bundleInfo && !loading && (
        <div className="action-strip">
          <div className="action-block load-dataset-card" onClick={handleLoadDataset}>
            <div className="load-dataset-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                <line x1="12" y1="11" x2="12" y2="17" />
                <line x1="9" y1="14" x2="15" y2="14" />
              </svg>
            </div>
            <span className="load-dataset-label">Load Dataset</span>
          </div>
        </div>
      )}

      {loading && <div className="loading">Loading...</div>}
    </div>
  );
}
