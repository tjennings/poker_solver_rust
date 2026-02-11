import { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import {
  AgentInfo,
  BundleInfo,
  CanonicalizeResult,
  StrategyMatrix,
  ExplorationPosition,
  ActionInfo,
  MatrixCell,
  ComboGroupInfo,
} from './types';

function HamburgerMenu({
  agents,
  activeAgentName,
  loading,
  onSelectAgent,
  onLoadBundle,
}: {
  agents: AgentInfo[];
  activeAgentName: string | null;
  loading: boolean;
  onSelectAgent: (path: string) => void;
  onLoadBundle: () => void;
}) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="hamburger-menu" ref={menuRef}>
      <button className="hamburger-button" onClick={() => setOpen(!open)}>
        <span className="hamburger-icon" />
      </button>
      {open && (
        <div className="hamburger-dropdown">
          {agents.length > 0 && (
            <div className="menu-section">
              <div className="menu-section-label">Agents</div>
              {agents.map((agent) => (
                <button
                  key={agent.path}
                  className={`menu-item ${activeAgentName === agent.name ? 'active' : ''}`}
                  disabled={loading}
                  onClick={() => { onSelectAgent(agent.path); setOpen(false); }}
                >
                  {agent.name}
                </button>
              ))}
            </div>
          )}
          <div className="menu-section">
            <button
              className="menu-item"
              disabled={loading}
              onClick={() => { onLoadBundle(); setOpen(false); }}
            >
              Load Bundle...
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

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
function getActionColor(action: ActionInfo, actions: ActionInfo[]): string {
  switch (action.action_type) {
    case 'fold':
      return 'rgba(59, 130, 246, 1)'; // Blue
    case 'check':
    case 'call':
      return 'rgba(34, 197, 94, 1)'; // Green
    case 'bet':
    case 'raise':
    case 'allin': {
      const betActions = actions.filter((a) =>
        a.action_type === 'bet' || a.action_type === 'raise' || a.action_type === 'allin',
      );
      const idx = betActions.findIndex((a) => a.id === action.id);
      const count = betActions.length;
      // Lightest (t=0) → darkest (t=1)
      const t = count > 1 ? idx / (count - 1) : 1;
      // Interpolate from light red (255, 180, 180) to dark red (153, 27, 27)
      const r = Math.round(255 - t * (255 - 153));
      const g = Math.round(180 - t * (180 - 27));
      const b = Math.round(180 - t * (180 - 27));
      return `rgba(${r}, ${g}, ${b}, 1)`;
    }
    default:
      return 'rgba(156, 163, 175, 1)'; // Gray
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
        {cell.probabilities.map((prob, idx) => {
          const action = actions[idx];
          if (!action) return null;
          const pct = prob.probability * 100;
          return (
            <div key={action.id} className="cell-detail-row">
              <div className="cell-detail-bar-bg">
                <div
                  className="cell-detail-bar-fill"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: getActionColor(action, actions),
                  }}
                />
              </div>
              <span className="cell-detail-label">{action.label}</span>
              <span className="cell-detail-pct">{pct.toFixed(1)}%</span>
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
  actions,
  selectedAction,
  onSelect,
  onHeaderClick,
  isCurrent,
}: {
  position: string;
  stack: number;
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
        <span className="stack">{stack}BB</span>
      </div>
      <div className="action-list">
        {actions.map((action) => {
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
              {action.label}
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
const SUIT_DISPLAY: Record<string, string> = { s: '♠', h: '♥', d: '♦', c: '♣' };

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
  const [showRemap, setShowRemap] = useState(false);

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

  return (
    <div className={`street-block ${cards.length === expectedCards ? '' : 'pending'}`}>
      <div
        className={`street-block-header ${onHeaderClick ? 'clickable' : ''}`}
        onClick={onHeaderClick}
      >
        <span className="street-name">{street}</span>
        <span className="street-pot">{pot}</span>
        {hasRemap && (
          <span
            className="remap-indicator"
            onMouseEnter={() => setShowRemap(true)}
            onMouseLeave={() => setShowRemap(false)}
            title="Suits remapped to canonical form"
          >
            ↔
          </span>
        )}
      </div>
      <div className="street-cards">
        {cardSlots}
      </div>
      {hasRemap && showRemap && (
        <div className="remap-tooltip">
          {remapInfo!.original.map((orig, i) => (
            <span key={i} className="remap-pair">
              {orig} → {remapInfo!.canonical[i]}
              {i < remapInfo!.original.length - 1 ? '  ' : ''}
            </span>
          ))}
          <div className="remap-suits">
            {Object.entries(remapInfo!.suitMap).map(([from, to]) => (
              <span key={from}>{SUIT_DISPLAY[from] || from}→{SUIT_DISPLAY[to] || to} </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// History item type
type HistoryItem =
  | { type: 'action'; position: string; stack: number; actions: ActionInfo[]; selected: string }
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
    stack_p1: 199,
    stack_p2: 198,
    to_act: 0,
  });
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [pendingStreet, setPendingStreet] = useState<{
    street: string;
    pot: number;
    stack_p1: number;
    stack_p2: number;
    expectedCards: number;
  } | null>(null);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedCell, setSelectedCell] = useState<{ row: number; col: number } | null>(null);
  const [handResult, setHandResult] = useState<'fold' | 'showdown' | null>(null);
  const [comboInfo, setComboInfo] = useState<ComboGroupInfo | null>(null);
  const [threshold, setThreshold] = useState(2);
  const [remapInfo, setRemapInfo] = useState<{
    original: string[];
    canonical: string[];
    suitMap: Record<string, string>;
  } | null>(null);

  // Fetch available agents on mount
  useEffect(() => {
    invoke<AgentInfo[]>('list_agents')
      .then(setAgents)
      .catch(() => setAgents([]));
  }, []);

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

  const loadSource = useCallback(
    async (path: string) => {
      setLoading(true);
      setError(null);
      try {
        const info = await invoke<BundleInfo>('load_bundle', { path });
        setBundleInfo(info);

        const initialPosition: ExplorationPosition = {
          board: [],
          history: [],
          pot: 3,
          stack_p1: info.stack_depth * 2 - 1,
          stack_p2: info.stack_depth * 2 - 2,
          to_act: 0,
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

  const handleLoadAgent = useCallback(
    (agentPath: string) => {
      loadSource(agentPath);
    },
    [loadSource]
  );

  const handleLoadBundle = useCallback(async () => {
    try {
      const path = await open({
        directory: true,
        title: 'Select Strategy Bundle Directory',
      });
      if (path) {
        loadSource(path);
      }
    } catch (e) {
      setError(String(e));
    }
  }, [loadSource]);

  // Check if current betting round is complete (needs street transition)
  const checkStreetTransition = useCallback(
    (history: string[], currentStreet: string): { needsTransition: boolean; nextStreet: string } => {
      if (history.length < 2) return { needsTransition: false, nextStreet: '' };

      const lastTwo = history.slice(-2);

      // Round ends when:
      // 1. Call after bet/raise: ['b:X', 'c'] or ['r:X', 'c']
      const isCallAfterBetOrRaise =
        lastTwo[1] === 'c' && (lastTwo[0].startsWith('r:') || lastTwo[0].startsWith('b:'));

      // 2. Both players check (postflop): ['x', 'x']
      const isBothCheck = lastTwo[0] === 'x' && lastTwo[1] === 'x';

      // 3. Preflop limp: SB calls, BB checks: ['c', 'x']
      const isPreflopLimp = currentStreet === 'Preflop' && lastTwo[0] === 'c' && lastTwo[1] === 'x';

      if (isCallAfterBetOrRaise || isBothCheck || isPreflopLimp) {
        if (currentStreet === 'Preflop') return { needsTransition: true, nextStreet: 'FLOP' };
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
          setHandResult('fold');
          return;
        }

        // Check for street transition
        const { needsTransition, nextStreet } = checkStreetTransition(newHistory, matrix.street);

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
          // River betting complete — showdown
          setPosition((prev) => ({
            ...prev,
            history: newHistory,
          }));
          setMatrix(null);
          setHandResult('showdown');
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
    [matrix, position, checkStreetTransition, threshold, historyItems]
  );

  const handleStreetCardsSet = useCallback(
    async (cards: string[]) => {
      if (!pendingStreet) return;

      try {
        setLoading(true);

        // Canonicalize the board cards via backend
        const result = await invoke<CanonicalizeResult>('canonicalize_board', { cards });
        const canonicalCards = result.canonical_cards;

        // Track remap info for the indicator (flop establishes it, turn/river extends it)
        if (result.remapped && result.suit_map) {
          setRemapInfo((prev) => ({
            original: [...(prev?.original ?? []), ...cards],
            canonical: [...(prev?.canonical ?? []), ...canonicalCards],
            suitMap: { ...(prev?.suitMap ?? {}), ...result.suit_map },
          }));
        }

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
          stack_p1: pendingStreet.stack_p1,
          stack_p2: pendingStreet.stack_p2,
          to_act: 0, // OOP acts first postflop
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
        stack_p1: sp1,
        stack_p2: sp2,
        to_act: (streetActionCount % 2) as 0 | 1,
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
      const initialPosition: ExplorationPosition = {
        board: [],
        history: [],
        pot: 3,
        stack_p1: bundleInfo.stack_depth * 2 - 1,
        stack_p2: bundleInfo.stack_depth * 2 - 2,
        to_act: 0,
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
      <HamburgerMenu
        agents={agents}
        activeAgentName={bundleInfo?.name ?? null}
        loading={loading}
        onSelectAgent={handleLoadAgent}
        onLoadBundle={handleLoadBundle}
      />

      {error && <div className="error">{error}</div>}

      {bundleInfo && (
        <div className="bundle-info">
          {bundleInfo.name && <span className="bundle-name">{bundleInfo.name}</span>}
          <span>Stack: {bundleInfo.stack_depth}BB</span>
          <span>Bet sizes: {bundleInfo.bet_sizes.map((s) => `${s * 100}%`).join(', ')}</span>
          {bundleInfo.info_sets > 0 && (
            <span>Info sets: {bundleInfo.info_sets.toLocaleString()}</span>
          )}
          {bundleInfo.iterations > 0 && (
            <span>Iterations: {bundleInfo.iterations.toLocaleString()}</span>
          )}
          <span className="threshold-control">
            Filter &lt;
            <input
              type="number"
              className="threshold-input"
              value={threshold}
              onChange={(e) => setThreshold(Math.max(0, Math.min(50, Number(e.target.value))))}
              min={0}
              max={50}
              step={1}
            />
            %
          </span>
        </div>
      )}

      {bundleInfo && (
        <>
          <div className="action-strip">
            {historyItems.map((item, idx) =>
              item.type === 'action' ? (
                <ActionBlock
                  key={idx}
                  position={item.position}
                  stack={item.stack}
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
                {selectedCell && matrix.cells[selectedCell.row]?.[selectedCell.col] && (
                  <CellDetail
                    cell={matrix.cells[selectedCell.row][selectedCell.col]}
                    actions={matrix.actions}
                  />
                )}
                {comboInfo && comboInfo.groups.length > 0 && (
                  <ComboClassPanel info={comboInfo} actions={matrix.actions} />
                )}
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
                {handResult === 'fold' ? 'Player folded' : 'Showdown'} — Pot: {position.pot}
              </p>
              <button className="new-hand-btn" onClick={handleNewHand}>
                New Hand
              </button>
            </div>
          )}
        </>
      )}

      {!bundleInfo && !loading && (
        <div className="empty-state">
          {agents.length > 0 && (
            <>
              <p>Select an agent to explore rule-based strategies:</p>
              <div className="agent-cards">
                {agents.map((agent) => (
                  <button
                    key={agent.path}
                    className="agent-card"
                    onClick={() => handleLoadAgent(agent.path)}
                  >
                    {agent.name}
                  </button>
                ))}
              </div>
              <p className="or-divider">or</p>
            </>
          )}
          <p>Load a trained strategy bundle to explore HUNL strategies.</p>
        </div>
      )}

      {loading && <div className="loading">Loading...</div>}
    </div>
  );
}
