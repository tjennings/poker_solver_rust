import { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { invoke } from './invoke';
import PostflopExplorer from './PostflopExplorer';
import {
  BlueprintConfig,
  BundleInfo,
  CanonicalizeResult,
  HandEquity,
  PreflopRanges,
  StrategyMatrix,
  ExplorationPosition,
  ActionInfo,
  MatrixCell,
  ComboGroupInfo,
  PlayerRange,
  RangeSnapshot,
  BlueprintListEntry,
} from './types';
import {
  SUIT_COLORS,
  SUIT_SYMBOLS,
  formatEV,
  displayOrderIndices,
  displayOrderActions,
  formatActionLabel,
  getActionColor,
  matrixToHandIndex,
} from './matrix-utils';

// Hand matrix cell component
export function HandCell({
  cell,
  actions,
  reachWeight,
  isSelected,
  isEditing,
  isEdited,
  onClick,
  overlayText,
}: {
  cell: MatrixCell;
  actions: ActionInfo[];
  reachWeight: number;
  isSelected: boolean;
  isEditing?: boolean;
  isEdited?: boolean;
  onClick: () => void;
  overlayText?: string;
}) {
  const gradientStops = useMemo(() => {
    if (cell.probabilities.length === 0) {
      return 'rgba(30, 41, 59, 1)';
    }

    const stops: string[] = [];
    let position = 0;

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

  const barHeight = Math.max(reachWeight * 100, 0);
  const isUnreachable = reachWeight < 0.01;

  return (
    <div
      className={`matrix-cell ${isSelected ? 'selected' : ''} ${isUnreachable ? 'unreachable' : ''} ${isEditing ? 'editing' : ''} ${isEdited ? 'edited' : ''}`}
      onClick={onClick}
    >
      <div
        className="cell-bar"
        style={{
          background: isUnreachable ? undefined : gradientStops,
          height: `${barHeight}%`,
        }}
      />
      <span className="cell-label">{cell.hand}</span>
      {overlayText && <span className="hand-cell-ev">{overlayText}</span>}
    </div>
  );
}

// Detail panel showing action frequencies for a selected cell
export function CellDetail({
  cell,
  actions,
}: {
  cell: MatrixCell;
  actions: ActionInfo[];
}) {
  return (
    <div className="cell-detail">
      <div className="cell-detail-header">
        <span>{cell.hand}</span>
        <div className="cell-detail-summary-bar">
          {[...displayOrderIndices(actions)].reverse().map((idx) => {
            const prob = cell.probabilities[idx];
            const action = actions[idx];
            if (!action || !prob) return null;
            const pct = prob.probability * 100;
            if (pct < 0.1) return null;
            return (
              <div
                key={action.id}
                className="cell-detail-summary-segment"
                style={{
                  width: `${pct}%`,
                  backgroundColor: getActionColor(action, actions),
                }}
              />
            );
          })}
        </div>
      </div>
      <div className="cell-detail-actions">
        {displayOrderIndices(actions).map((idx) => {
          const prob = cell.probabilities[idx];
          const action = actions[idx];
          if (!action || !prob) return null;
          const pct = prob.probability * 100;
          return (
            <div key={action.id} className="cell-detail-row">
              <span className="cell-detail-label">{formatActionLabel(action)}</span>
              <div className="cell-detail-bar-bg">
                <div
                  className="cell-detail-bar-fill"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: getActionColor(action, actions),
                  }}
                />
              </div>
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
export function ActionBlock({
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
        <span className="stack">{isNaN(stack) ? '--' : `${+(stack / 2).toFixed(1)}BB`} / {isNaN(pot) ? '--' : `${+(pot / 2).toFixed(1)}BB`}</span>
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
export type HistoryItem =
  | { type: 'action'; position: string; stack: number; pot: number; actions: ActionInfo[]; selected: string }
  | { type: 'street'; street: string; pot: number; stack_p1: number; stack_p2: number; cards: string[] };

// Extract completed-street action sequences from history items.
function extractStreetHistories(items: HistoryItem[]): string[][] {
  const histories: string[][] = [];
  let current: string[] = [];

  for (const item of items) {
    if (item.type === 'action') {
      current.push(item.selected);
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

export default function Explorer() {
  const [bundleInfo, setBundleInfo] = useState<BundleInfo | null>(null);
  const [matrix, setMatrix] = useState<StrategyMatrix | null>(null);
  const [position, setPosition] = useState<ExplorationPosition>({
    board: [],
    history: [],
    pot: 3,
    stacks: [199, 198],
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
  const [remapInfo, setRemapInfo] = useState<{
    original: string[];
    canonical: string[];
    suitMap: Record<string, string>;
  } | null>(null);
  const [p1Range, setP1Range] = useState<PlayerRange>({
    hands: Array(169).fill(1.0),
    source: 'computed',
    overrides: [],
  });
  const [p2Range, setP2Range] = useState<PlayerRange>({
    hands: Array(169).fill(1.0),
    source: 'computed',
    overrides: [],
  });
  const [rangeSnapshots, setRangeSnapshots] = useState<RangeSnapshot[]>([]);
  const [editingPlayer, setEditingPlayer] = useState<0 | 1 | null>(null);

  const updateRangesFromMatrix = useCallback((m: StrategyMatrix) => {
    setP1Range(prev => {
      const newHands = m.reaching_p1?.length === 169 ? [...m.reaching_p1] : Array(169).fill(1.0);
      for (const idx of prev.overrides) {
        newHands[idx] = prev.hands[idx];
      }
      return {
        hands: newHands,
        source: prev.overrides.length > 0 ? 'edited' : 'computed',
        overrides: prev.overrides,
      };
    });
    setP2Range(prev => {
      const newHands = m.reaching_p2?.length === 169 ? [...m.reaching_p2] : Array(169).fill(1.0);
      for (const idx of prev.overrides) {
        newHands[idx] = prev.hands[idx];
      }
      return {
        hands: newHands,
        source: prev.overrides.length > 0 ? 'edited' : 'computed',
        overrides: prev.overrides,
      };
    });
  }, []);

  const handleCellEdit = useCallback((row: number, col: number) => {
    if (editingPlayer === null) return;
    const handIdx = matrixToHandIndex(row, col);
    const setRange = editingPlayer === 0 ? setP1Range : setP2Range;

    setRange(prev => {
      const newHands = [...prev.hands];
      // Cycle: 1.0 -> 0.5 -> 0.0 -> 1.0
      const current = newHands[handIdx];
      newHands[handIdx] = current > 0.75 ? 0.5 : current > 0.25 ? 0.0 : 1.0;
      const newOverrides = [...new Set([...prev.overrides, handIdx])];
      return {
        hands: newHands,
        source: 'edited',
        overrides: newOverrides,
      };
    });
  }, [editingPlayer]);

  // Fetch combo classification when a cell is selected on postflop
  useEffect(() => {
    if (!matrix || !selectedCell || position.board.length === 0) {
      setComboInfo(null);
      return;
    }
    const cell = matrix.cells[selectedCell.row]?.[selectedCell.col];
    if (!cell) {
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
    if (!cell) {
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

  const [showPostflop, setShowPostflop] = useState(false);
  const [showBlueprintPicker, setShowBlueprintPicker] = useState(false);
  const [blueprints, setBlueprints] = useState<BlueprintListEntry[]>([]);
  const [blueprintPostflopConfig, setBlueprintPostflopConfig] = useState<BlueprintConfig | null>(null);
  const blueprintPostflopConfigRef = useRef<BlueprintConfig | null>(null);

  const handleLoadStrategy = async () => {
    const globalConfig = JSON.parse(localStorage.getItem('global_config') || '{}');
    if (!globalConfig.blueprint_dir) {
      setError('Set Blueprint Directory in Settings first');
      return;
    }
    try {
      const list = await invoke<BlueprintListEntry[]>('list_blueprints', { dir: globalConfig.blueprint_dir });
      setBlueprints(list.filter(b => b.has_strategy));
      setShowBlueprintPicker(true);
    } catch (e) {
      setError(String(e));
    }
  };

  // Check if current betting round is complete (needs street transition)
  // Detect street transitions using action types from the V2 game tree.
  // A street ends when: call after bet/raise, check-check, or limp-check.
  const checkStreetTransition = useCallback(
    (currentStreet: string, prevActionType: string | null, actionType: string): { needsTransition: boolean; nextStreet: string } => {
      const isCallAfterBetOrRaise =
        actionType === 'call' && (prevActionType === 'bet' || prevActionType === 'raise' || prevActionType === 'allin');

      // All-in after all-in = calling the opponent's all-in (both players committed)
      const isAllinAfterAllin = actionType === 'allin' && prevActionType === 'allin';

      const isBothCheck = prevActionType === 'check' && actionType === 'check';

      const isPreflopLimp =
        currentStreet === 'Preflop' && prevActionType === 'call' && actionType === 'check';

      if (isCallAfterBetOrRaise || isAllinAfterAllin || isBothCheck || isPreflopLimp) {
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

        // Snapshot current ranges before advancing
        setRangeSnapshots(prev => [...prev, {
          p1_range: { ...p1Range, hands: [...p1Range.hands], overrides: [...p1Range.overrides] },
          p2_range: { ...p2Range, hands: [...p2Range.hands], overrides: [...p2Range.overrides] },
          node_index: historyItems.length,
        }]);

        let newHistory = [...position.history];
        let newToAct = position.to_act === 0 ? 1 : 0;

        // V2 blueprint: action IDs are array indices ("0", "1", "2", ...)
        // Pass directly to the backend.
        newHistory.push(actionId);

        // Look up the action type for transition/terminal detection.
        const selectedAction = matrix.actions.find(a => a.id === actionId);
        const actionType = selectedAction?.action_type ?? '';

        // Preflop-only fold: dead end — keep matrix visible, no action card added
        if (actionType === 'fold' && bundleInfo?.preflop_only) {
          setPosition((prev) => ({
            ...prev,
            history: newHistory,
          }));
          return;
        }

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
        if (actionType === 'fold') {
          setPosition((prev) => ({
            ...prev,
            history: newHistory,
          }));
          setMatrix(null);
          setHandResult({ type: 'fold', pot: matrix.pot });
          return;
        }

        // Check for street transition using action types.
        // Get previous action type from the last action history item.
        const prevItems = historyItems.filter(h => h.type === 'action');
        const prevActionItem = prevItems[prevItems.length - 1];
        const prevActionType = prevActionItem && prevActionItem.type === 'action'
          ? prevActionItem.actions.find(a => a.id === prevActionItem.selected)?.action_type ?? null
          : null;
        const { needsTransition, nextStreet } = checkStreetTransition(
          matrix.street,
          prevActionType,
          actionType,
        );

        if (needsTransition && nextStreet && bundleInfo?.preflop_only) {
          // Blueprint preflop → postflop transition: compute ranges, show card picker
          try {
            const ranges = await invoke<PreflopRanges>('get_preflop_ranges', {
              history: newHistory,
            });

            // Both players all-in preflop → no postflop betting possible, show showdown
            if (Math.round(ranges.effective_stack) <= 0) {
              const finalPot = Math.round(ranges.pot);
              setPosition((prev) => ({ ...prev, history: newHistory }));
              setMatrix(null);
              setHandResult({ type: 'showdown', pot: finalPot });
              return;
            }

            const FULL_RANGE = 'AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,K9s,K8s,K7s,K6s,K5s,K4s,K3s,K2s,QJs,QTs,Q9s,Q8s,Q7s,Q6s,Q5s,Q4s,Q3s,Q2s,JTs,J9s,J8s,J7s,J6s,J5s,J4s,J3s,J2s,T9s,T8s,T7s,T6s,T5s,T4s,T3s,T2s,98s,97s,96s,95s,94s,93s,92s,87s,86s,85s,84s,83s,82s,76s,75s,74s,73s,72s,65s,64s,63s,62s,54s,53s,52s,43s,42s,32s,AKo,AQo,AJo,ATo,A9o,A8o,A7o,A6o,A5o,A4o,A3o,A2o,KQo,KJo,KTo,K9o,K8o,K7o,K6o,K5o,K4o,K3o,K2o,QJo,QTo,Q9o,Q8o,Q7o,Q6o,Q5o,Q4o,Q3o,Q2o,JTo,J9o,J8o,J7o,J6o,J5o,J4o,J3o,J2o,T9o,T8o,T7o,T6o,T5o,T4o,T3o,T2o,98o,97o,96o,95o,94o,93o,92o,87o,86o,85o,84o,83o,82o,76o,75o,74o,73o,72o,65o,64o,63o,62o,54o,53o,52o,43o,42o,32o';
            const globalConfig = JSON.parse(localStorage.getItem('global_config') || '{}');
            const bpConfig: BlueprintConfig = {
              oop_range: FULL_RANGE,
              ip_range: FULL_RANGE,
              oop_weights: ranges.oop_weights,
              ip_weights: ranges.ip_weights,
              pot: Math.round(ranges.pot * 2),
              effective_stack: Math.round(ranges.effective_stack * 2),
              oop_bet_sizes: ranges.oop_bet_sizes,
              oop_raise_sizes: ranges.oop_raise_sizes,
              ip_bet_sizes: ranges.ip_bet_sizes,
              ip_raise_sizes: ranges.ip_raise_sizes,
              blueprint_dir: globalConfig.blueprint_dir || '',
              rake_rate: ranges.rake_rate,
              rake_cap: ranges.rake_cap,
              abstract_node_idx: ranges.abstract_node_idx,
            };
            setBlueprintPostflopConfig(bpConfig);
            blueprintPostflopConfigRef.current = bpConfig;
          } catch (e) {
            setError(String(e));
            return;
          }
          // Fall through to show the card picker (same as non-blueprint path)
        }

        if (needsTransition && nextStreet) {
          // Compute pot/stacks entering the new street.
          // A call adds to_call to pot and subtracts from the caller's stack.
          const isCallingAction = actionType === 'call' || (actionType === 'allin' && prevActionType === 'allin');
          const transitionPot =
            isCallingAction ? matrix.pot + matrix.to_call : matrix.pot;
          let sp1 = matrix.stack_p1;
          let sp2 = matrix.stack_p2;
          if (isCallingAction) {
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
          const isCallingAction2 = actionType === 'call' || (actionType === 'allin' && prevActionType === 'allin');
          const finalPot = isCallingAction2
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
            street_histories: extractStreetHistories(historyItems),
          });
          setMatrix(newMatrix);
          updateRangesFromMatrix(newMatrix);
        }
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    },
    [matrix, position, checkStreetTransition, historyItems, bundleInfo, p1Range, p2Range, updateRangesFromMatrix]
  );

  const handleStreetCardsSet = useCallback(
    async (cards: string[]) => {
      if (!pendingStreet) return;

      try {
        setLoading(true);

        // Canonicalize the board cards via backend
        const result = await invoke<CanonicalizeResult>('canonicalize_board', { cards });
        const canonicalCards = result.canonical_cards;

        // Blueprint preflop → postflop: transition to PostflopExplorer with board pre-filled
        if (blueprintPostflopConfigRef.current) {
          const withBoard = { ...blueprintPostflopConfigRef.current, board: canonicalCards };
          setBlueprintPostflopConfig(withBoard);
          blueprintPostflopConfigRef.current = withBoard;
          setPendingStreet(null);
          setShowPostflop(true);
          return;
        }

        // Track remap info for the indicator.
        setRemapInfo((prev) => {
          if (result.remapped && result.suit_map) {
            return {
              original: [...(prev?.original ?? []), ...cards],
              canonical: [...(prev?.canonical ?? []), ...canonicalCards],
              suitMap: { ...(prev?.suitMap ?? {}), ...result.suit_map },
            };
          }
          if (prev) {
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
        const newBoard = [...position.board, ...canonicalCards];
        const newPosition: ExplorationPosition = {
          board: newBoard,
          history: [],
          pot: pendingStreet.pot,
          stacks: [pendingStreet.stack_p1, pendingStreet.stack_p2],
          to_act: 0,
          num_players: 2,
          active_players: [true, true],
        };
        setPosition(newPosition);
        setPendingStreet(null);

        await invoke('start_bucket_computation', { board: newBoard });

        const sh = [...extractStreetHistories(historyItems), [...position.history]];

        const newMatrix = await invoke<StrategyMatrix>('get_strategy_matrix', {
          position: newPosition,
          street_histories: sh,
        });
        setMatrix(newMatrix);
        updateRangesFromMatrix(newMatrix);
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    },
    [pendingStreet, position, historyItems]
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
          history.push(item.selected);
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

        // Restore range snapshot
        const snapshot = [...rangeSnapshots].reverse().find(s => s.node_index <= index);
        if (snapshot) {
          setP1Range(snapshot.p1_range);
          setP2Range(snapshot.p2_range);
        }
        setRangeSnapshots(prev => prev.filter(s => s.node_index < index));

        const newMatrix = await invoke<StrategyMatrix>('get_strategy_matrix', {
          position: pos,
          street_histories: extractStreetHistories(items),
        });
        setMatrix(newMatrix);
        updateRangesFromMatrix(newMatrix);
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    },
    [rebuildState, rangeSnapshots, updateRangesFromMatrix]
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
      setP1Range({ hands: Array(169).fill(1.0), source: 'computed', overrides: [] });
      setP2Range({ hands: Array(169).fill(1.0), source: 'computed', overrides: [] });
      setRangeSnapshots([]);
      setEditingPlayer(null);
      const newMatrix = await invoke<StrategyMatrix>('get_strategy_matrix', {
        position: initialPosition,
      });
      setMatrix(newMatrix);
      updateRangesFromMatrix(newMatrix);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [bundleInfo, updateRangesFromMatrix]);

  return (
    <div className="explorer">
      {error && <div className="error">{error}</div>}

      {!showPostflop && (
      <div className="action-strip">
        <div className="dataset-switcher-split">
          <div className="dataset-switcher-half" onClick={handleLoadStrategy} title="Load Strategy">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
              <line x1="12" y1="11" x2="12" y2="17" />
              <line x1="9" y1="14" x2="15" y2="14" />
            </svg>
          </div>
          <div className="dataset-switcher-half" onClick={() => setShowPostflop(true)} title="Range Solve">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="2" y="2" width="20" height="20" rx="2" />
              <path d="M7 12h10M12 7v10" />
            </svg>
          </div>
        </div>

      {bundleInfo && (
        <>
          {bundleInfo.name && (
            <div className="action-block postflop-config-card" style={{ cursor: 'default' }}>
              <div className="postflop-config-label">{bundleInfo.name}</div>
              <div className="postflop-config-summary">
                {bundleInfo.stack_depth}BB
                {bundleInfo.rake_rate > 0 && ` / ${(bundleInfo.rake_rate * 100).toFixed(1)}% rake`}
              </div>
            </div>
          )}
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
        </>
      )}
      </div>
      )}

      {!showPostflop && bundleInfo && matrix && (
            <div className="range-toolbar">
              <button
                className={`range-edit-btn ${editingPlayer === 0 ? 'active' : ''}`}
                onClick={() => setEditingPlayer(editingPlayer === 0 ? null : 0)}
              >
                {editingPlayer === 0 ? 'Done' : 'Edit SB Range'}
              </button>
              <button
                className={`range-edit-btn ${editingPlayer === 1 ? 'active' : ''}`}
                onClick={() => setEditingPlayer(editingPlayer === 1 ? null : 1)}
              >
                {editingPlayer === 1 ? 'Done' : 'Edit BB Range'}
              </button>
              {(p1Range.overrides.length > 0 || p2Range.overrides.length > 0) && (
                <button
                  className="range-edit-btn reset"
                  onClick={() => {
                    setP1Range({ hands: matrix.reaching_p1?.length === 169 ? [...matrix.reaching_p1] : Array(169).fill(1.0), source: 'computed', overrides: [] });
                    setP2Range({ hands: matrix.reaching_p2?.length === 169 ? [...matrix.reaching_p2] : Array(169).fill(1.0), source: 'computed', overrides: [] });
                    setEditingPlayer(null);
                  }}
                >
                  Reset Ranges
                </button>
              )}
            </div>
          )}

      {!showPostflop && bundleInfo && matrix && (
            <div className="matrix-container">
              <div className="matrix-with-detail">
                <div className="hand-matrix">
                  {matrix.cells.map((row, rowIdx) => (
                    <div key={rowIdx} className="matrix-row">
                      {row.map((cell, colIdx) => {
                        const handIdx = matrixToHandIndex(rowIdx, colIdx);
                        const reachWeight = (() => {
                          if (editingPlayer !== null) {
                            const range = position.to_act === 0 ? p1Range : p2Range;
                            return range.hands[handIdx] ?? 1.0;
                          }
                          return position.to_act === 0
                            ? (matrix.reaching_p1?.[handIdx] ?? 1.0)
                            : (matrix.reaching_p2?.[handIdx] ?? 1.0);
                        })();
                        const isEdited = editingPlayer !== null &&
                          (position.to_act === 0 ? p1Range : p2Range).overrides.includes(handIdx);
                        return (
                          <HandCell
                            key={colIdx}
                            cell={cell}
                            actions={matrix.actions}
                            reachWeight={reachWeight}
                            isSelected={selectedCell?.row === rowIdx && selectedCell?.col === colIdx}
                            isEditing={editingPlayer !== null}
                            isEdited={isEdited}
                            onClick={() => {
                              if (editingPlayer !== null) {
                                handleCellEdit(rowIdx, colIdx);
                              } else {
                                setSelectedCell({ row: rowIdx, col: colIdx });
                              }
                            }}
                          />
                        );
                      })}
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

      {!showPostflop && bundleInfo && !matrix && pendingStreet && (
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

      {!showPostflop && bundleInfo && handResult && (
        <div className="hand-complete">
          <p className="hand-complete-result">
            {handResult.type === 'fold' ? 'Player folded' : 'Showdown'} — Pot: {handResult.pot}
          </p>
          <button className="new-hand-btn" onClick={handleNewHand}>
            New Hand
          </button>
        </div>
      )}

      {showBlueprintPicker && (
        <div className="dataset-picker-overlay" onClick={() => setShowBlueprintPicker(false)}>
        <div className="dataset-picker" onClick={e => e.stopPropagation()}>
          <div className="dataset-picker-header">
            <h3>Select Blueprint</h3>
            <button className="dataset-picker-close" onClick={() => setShowBlueprintPicker(false)}>×</button>
          </div>
          {blueprints.length === 0 ? (
            <p className="dataset-picker-empty">No blueprints found with trained strategies</p>
          ) : (
            <div className="dataset-picker-list">
              {blueprints.map((bp) => (
                <div
                  key={bp.path}
                  className="dataset-picker-item"
                  onClick={async () => {
                    setShowBlueprintPicker(false);
                    setLoading(true);
                    setError(null);
                    try {
                      const info = await invoke<BundleInfo>('load_blueprint_v2', { path: bp.path });
                      setBundleInfo(info);
                      const sp1 = info.stack_depth * 2 - 1;
                      const sp2 = info.stack_depth * 2 - 2;
                      const initialPosition: ExplorationPosition = {
                        board: [],
                        history: [],
                        pot: 3,
                        stacks: [sp1, sp2],
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
                      });
                      setMatrix(initialMatrix);
                      updateRangesFromMatrix(initialMatrix);
                    } catch (e) {
                      setError(String(e));
                    } finally {
                      setLoading(false);
                    }
                  }}
                >
                  <span className="dataset-kind-badge preflop">blueprint</span>
                  <span className="dataset-name">{bp.name} ({bp.stack_depth}BB)</span>
                </div>
              ))}
            </div>
          )}
        </div>
        </div>
      )}

      {showPostflop && (
        <PostflopExplorer
          onBack={(preflopIdx?: number) => {
            setShowPostflop(false);
            setBlueprintPostflopConfig(null);
            if (preflopIdx !== undefined) {
              handleHistoryRewind(preflopIdx);
            }
          }}
          blueprintConfig={blueprintPostflopConfig ?? undefined}
          preflopHistory={historyItems}
        />
      )}

      {loading && <div className="loading">Loading...</div>}
    </div>
  );
}
