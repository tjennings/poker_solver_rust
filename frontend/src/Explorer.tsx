import { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-dialog';
import {
  AgentInfo,
  BundleInfo,
  StrategyMatrix,
  ExplorationPosition,
  ActionInfo,
  MatrixCell,
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

// Progress event from backend
interface BucketProgressEvent {
  completed: number;
  total: number;
  board_key: string;
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

// Color utilities for action probabilities
function getActionColor(actionType: string): string {
  switch (actionType) {
    case 'fold':
      return 'rgba(59, 130, 246, 1)'; // Blue
    case 'check':
    case 'call':
      return 'rgba(34, 197, 94, 1)'; // Green
    case 'bet':
    case 'raise':
      return 'rgba(239, 68, 68, 1)'; // Red
    case 'allin':
      return 'rgba(168, 85, 247, 1)'; // Purple
    default:
      return 'rgba(156, 163, 175, 1)'; // Gray
  }
}

// Hand matrix cell component
function HandCell({
  cell,
  actions,
}: {
  cell: MatrixCell;
  actions: ActionInfo[];
}) {
  const gradientStops = useMemo(() => {
    if (cell.probabilities.length === 0) {
      return 'rgba(30, 41, 59, 1)';
    }

    const stops: string[] = [];
    let position = 0;

    cell.probabilities.forEach((prob, idx) => {
      const action = actions[idx];
      if (!action || prob.probability <= 0) return;

      const color = getActionColor(action.action_type);
      const width = prob.probability * 100;
      stops.push(`${color} ${position}%`);
      stops.push(`${color} ${position + width}%`);
      position += width;
    });

    if (stops.length === 0) {
      return 'rgba(30, 41, 59, 1)';
    }

    return `linear-gradient(to right, ${stops.join(', ')})`;
  }, [cell.probabilities, actions]);

  return (
    <div
      className="matrix-cell"
      style={{ background: gradientStops }}
      title={cell.probabilities
        .map((p, i) => `${actions[i]?.label || 'Unknown'}: ${(p.probability * 100).toFixed(1)}%`)
        .join('\n')}
    >
      <span className="cell-label">{cell.hand}</span>
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
  isCurrent,
}: {
  position: string;
  stack: number;
  actions: ActionInfo[];
  selectedAction?: string;
  onSelect: (actionId: string) => void;
  isCurrent: boolean;
}) {
  return (
    <div className={`action-block ${isCurrent ? 'current' : ''}`}>
      <div className="action-block-header">
        <span className="position">{position}</span>
        <span className="stack">{stack}BB</span>
      </div>
      <div className="action-list">
        {actions.map((action) => (
          <button
            key={action.id}
            className={`action-button ${action.action_type} ${
              selectedAction === action.id ? 'selected' : ''
            }`}
            onClick={() => onSelect(action.id)}
          >
            {action.label}
          </button>
        ))}
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

// Street transition block (FLOP, TURN, RIVER)
function StreetBlock({
  street,
  pot,
  cards,
  expectedCards,
  onCardsChange,
  isEditable,
}: {
  street: string;
  pot: number;
  cards: string[];
  expectedCards: number;
  onCardsChange: (cards: string[]) => void;
  isEditable: boolean;
}) {
  const [inputValue, setInputValue] = useState('');
  const [showInput, setShowInput] = useState(false);

  const handleSubmit = () => {
    // Parse cards - accept both "AcTh4d" and "Ac Th 4d" formats
    let parsed: string[];
    const trimmed = inputValue.trim();

    if (trimmed.includes(' ')) {
      // Space-separated format
      parsed = trimmed.split(/\s+/).filter((c) => c.length === 2);
    } else {
      // Continuous format - split every 2 characters
      parsed = [];
      for (let i = 0; i + 1 < trimmed.length; i += 2) {
        parsed.push(trimmed.slice(i, i + 2));
      }
    }

    parsed = parsed.slice(0, expectedCards);

    if (parsed.length === expectedCards) {
      onCardsChange(parsed);
      setShowInput(false);
      setInputValue('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSubmit();
    } else if (e.key === 'Escape') {
      setShowInput(false);
      setInputValue('');
    }
  };

  const cardSlots = [];
  for (let i = 0; i < expectedCards; i++) {
    cardSlots.push(
      <StreetCard
        key={i}
        card={cards[i] || null}
        onClick={isEditable ? () => setShowInput(true) : undefined}
      />
    );
  }

  return (
    <div className={`street-block ${cards.length === expectedCards ? '' : 'pending'}`}>
      <div className="street-block-header">
        <span className="street-name">{street}</span>
        <span className="street-pot">{pot}</span>
      </div>
      <div className="street-cards">
        {cardSlots}
      </div>
      {showInput && isEditable && (
        <div className="street-input">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={expectedCards === 3 ? 'Js Th 8c' : expectedCards === 1 ? '2d' : ''}
            autoFocus
          />
          <button onClick={handleSubmit}>Set</button>
        </div>
      )}
      {cards.length < expectedCards && !showInput && isEditable && (
        <button className="set-cards-btn" onClick={() => setShowInput(true)}>
          Set Cards
        </button>
      )}
    </div>
  );
}

// History item type
type HistoryItem =
  | { type: 'action'; position: string; stack: number; actions: ActionInfo[]; selected: string }
  | { type: 'street'; street: string; pot: number; cards: string[] };

export default function Explorer() {
  const [bundleInfo, setBundleInfo] = useState<BundleInfo | null>(null);
  const [matrix, setMatrix] = useState<StrategyMatrix | null>(null);
  const [position, setPosition] = useState<ExplorationPosition>({
    board: [],
    history: [],
    pot: 3,
    stack_p1: 99,
    stack_p2: 98,
    to_act: 0,
  });
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [pendingStreet, setPendingStreet] = useState<{
    street: string;
    pot: number;
    expectedCards: number;
  } | null>(null);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [computingBuckets, setComputingBuckets] = useState(false);
  const [computationProgress, setComputationProgress] = useState({ completed: 0, total: 169 });

  // Ref to track current position for use in event callbacks
  const positionRef = useRef(position);
  positionRef.current = position;

  // Listen for bucket computation progress events
  useEffect(() => {
    let unlisten: (() => void) | undefined;

    const setupListener = async () => {
      unlisten = await listen<BucketProgressEvent>('bucket-progress', async (event) => {
        const { completed, total } = event.payload;
        setComputationProgress({ completed, total });

        if (completed >= total) {
          // Computation complete - refresh the matrix
          setComputingBuckets(false);
          try {
            const newMatrix = await invoke<StrategyMatrix>('get_strategy_matrix', {
              position: positionRef.current,
            });
            setMatrix(newMatrix);
          } catch (e) {
            setError(String(e));
          }
        }
      });
    };

    setupListener();

    return () => {
      if (unlisten) {
        unlisten();
      }
    };
  }, []); // No dependencies - listener set up once, uses ref for current position

  // Fetch available agents on mount
  useEffect(() => {
    invoke<AgentInfo[]>('list_agents')
      .then(setAgents)
      .catch(() => setAgents([]));
  }, []);

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
          stack_p1: info.stack_depth - 1,
          stack_p2: info.stack_depth - 2,
          to_act: 0,
        };
        setPosition(initialPosition);
        setHistoryItems([]);
        setPendingStreet(null);

        const initialMatrix = await invoke<StrategyMatrix>('get_strategy_matrix', {
          position: initialPosition,
        });
        setMatrix(initialMatrix);
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    },
    []
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
        let newPot = position.pot;
        let newStackP1 = position.stack_p1;
        let newStackP2 = position.stack_p2;
        let newToAct = position.to_act === 0 ? 1 : 0;

        let historyEntry: string;

        if (actionId === 'call') {
          const callAmount = matrix.to_call;
          if (position.to_act === 0) {
            newStackP1 -= callAmount;
          } else {
            newStackP2 -= callAmount;
          }
          newPot += callAmount;
          historyEntry = 'c';
        } else if (actionId === 'check') {
          historyEntry = 'x';
        } else if (actionId === 'fold') {
          historyEntry = 'f';
        } else if (actionId.startsWith('bet:') || actionId.startsWith('raise:')) {
          const amount = parseInt(actionId.split(':')[1], 10);
          const prefix = actionId.startsWith('bet:') ? 'b' : 'r';
          historyEntry = `${prefix}:${amount}`;
          if (position.to_act === 0) {
            newStackP1 -= amount;
          } else {
            newStackP2 -= amount;
          }
          newPot = amount * 2 + (position.pot - matrix.to_call);
        } else {
          historyEntry = actionId;
        }

        newHistory.push(historyEntry);

        // Add action to history
        const actionItem: HistoryItem = {
          type: 'action',
          position: position.to_act === 0 ? 'SB' : 'BB',
          stack: position.to_act === 0 ? position.stack_p1 : position.stack_p2,
          actions: matrix.actions,
          selected: actionId,
        };
        setHistoryItems((prev) => [...prev, actionItem]);

        // Check for street transition
        const { needsTransition, nextStreet } = checkStreetTransition(newHistory, matrix.street);

        if (needsTransition) {
          const expectedCards = nextStreet === 'FLOP' ? 3 : 1;
          setPendingStreet({ street: nextStreet, pot: newPot, expectedCards });
          // Don't update matrix yet - wait for cards
          setPosition((prev) => ({
            ...prev,
            history: newHistory,
            pot: newPot,
            stack_p1: newStackP1,
            stack_p2: newStackP2,
          }));
          setMatrix(null);
        } else {
          const newPosition: ExplorationPosition = {
            ...position,
            history: newHistory,
            pot: newPot,
            stack_p1: newStackP1,
            stack_p2: newStackP2,
            to_act: newToAct as 0 | 1,
          };
          setPosition(newPosition);

          const newMatrix = await invoke<StrategyMatrix>('get_strategy_matrix', {
            position: newPosition,
          });
          setMatrix(newMatrix);
        }
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    },
    [matrix, position, checkStreetTransition]
  );

  const handleStreetCardsSet = useCallback(
    async (cards: string[]) => {
      if (!pendingStreet) return;

      try {
        setLoading(true);

        // Add street to history
        const streetItem: HistoryItem = {
          type: 'street',
          street: pendingStreet.street,
          pot: pendingStreet.pot,
          cards,
        };
        setHistoryItems((prev) => [...prev, streetItem]);

        // Update board
        const newBoard = [...position.board, ...cards];
        const newPosition: ExplorationPosition = {
          ...position,
          board: newBoard,
          to_act: 0, // OOP acts first postflop
        };
        setPosition(newPosition);
        setPendingStreet(null);

        // Start async bucket computation
        setComputingBuckets(true);
        setComputationProgress({ completed: 0, total: 169 });

        await invoke('start_bucket_computation', { board: newBoard });

        // Get initial matrix (will show default probabilities until computation completes)
        const newMatrix = await invoke<StrategyMatrix>('get_strategy_matrix', {
          position: newPosition,
        });
        setMatrix(newMatrix);
      } catch (e) {
        setError(String(e));
        setComputingBuckets(false);
      } finally {
        setLoading(false);
      }
    },
    [pendingStreet, position]
  );

  const handleHistoryRewind = useCallback(
    async (index: number) => {
      try {
        setLoading(true);

        // Rebuild state up to this point
        const newHistoryItems = historyItems.slice(0, index);
        let newBoard: string[] = [];
        let newHistory: string[] = [];
        let actionCount = 0;

        for (const item of newHistoryItems) {
          if (item.type === 'action') {
            // Convert selected action back to history string
            let historyEntry: string;
            if (item.selected === 'call') historyEntry = 'c';
            else if (item.selected === 'check') historyEntry = 'x';
            else if (item.selected === 'fold') historyEntry = 'f';
            else if (item.selected.startsWith('bet:'))
              historyEntry = `b:${item.selected.split(':')[1]}`;
            else if (item.selected.startsWith('raise:'))
              historyEntry = `r:${item.selected.split(':')[1]}`;
            else historyEntry = item.selected;
            newHistory.push(historyEntry);
            actionCount++;
          } else if (item.type === 'street') {
            newBoard = [...newBoard, ...item.cards];
          }
        }

        setHistoryItems(newHistoryItems);
        setPendingStreet(null);

        // Recalculate position (simplified)
        const newPosition: ExplorationPosition = {
          board: newBoard,
          history: newHistory,
          pot: 3, // Would need full replay for accurate pot
          stack_p1: bundleInfo ? bundleInfo.stack_depth - 1 : 99,
          stack_p2: bundleInfo ? bundleInfo.stack_depth - 2 : 98,
          to_act: (actionCount % 2) as 0 | 1,
        };
        setPosition(newPosition);

        const newMatrix = await invoke<StrategyMatrix>('get_strategy_matrix', {
          position: newPosition,
        });
        setMatrix(newMatrix);
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    },
    [historyItems, bundleInfo]
  );

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
                  isCurrent={false}
                />
              ) : (
                <StreetBlock
                  key={idx}
                  street={item.street}
                  pot={item.pot}
                  cards={item.cards}
                  expectedCards={item.cards.length}
                  onCardsChange={() => {}}
                  isEditable={false}
                />
              )
            )}

            {pendingStreet && (
              <StreetBlock
                street={pendingStreet.street}
                pot={pendingStreet.pot}
                cards={[]}
                expectedCards={pendingStreet.expectedCards}
                onCardsChange={handleStreetCardsSet}
                isEditable={true}
              />
            )}

            {matrix && !pendingStreet && (
              <ActionBlock
                position={position.to_act === 0 ? 'SB' : 'BB'}
                stack={position.to_act === 0 ? position.stack_p1 : position.stack_p2}
                actions={matrix.actions}
                onSelect={handleActionSelect}
                isCurrent={true}
              />
            )}
          </div>

          {matrix && (
            <div className="matrix-container">
              {computingBuckets && (
                <div className="computation-progress">
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{
                        width: `${(computationProgress.completed / computationProgress.total) * 100}%`,
                      }}
                    />
                  </div>
                  <span className="progress-text">
                    Computing hand strengths: {computationProgress.completed}/{computationProgress.total}
                  </span>
                </div>
              )}
              <div className="hand-matrix">
                {matrix.cells.map((row, rowIdx) => (
                  <div key={rowIdx} className="matrix-row">
                    {row.map((cell, colIdx) => (
                      <HandCell key={colIdx} cell={cell} actions={matrix.actions} />
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}

          {!matrix && pendingStreet && (
            <div className="waiting-for-cards">
              <p>Set the {pendingStreet.street.toLowerCase()} cards to continue</p>
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
