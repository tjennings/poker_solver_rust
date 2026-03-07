import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { invoke } from './invoke';
import {
  ActionInfo,
  MatrixCell,
  PostflopConfig,
  PostflopConfigSummary,
  PostflopActionInfo,
  PostflopStrategyMatrix,
  PostflopProgress,
  PostflopPlayResult,
  PostflopStreetResult,
} from './types';
import {
  SUIT_COLORS,
  SUIT_SYMBOLS,
  getActionColor,
  formatActionLabel,
} from './matrix-utils';
import { ActionBlock, HandCell, CellDetail } from './Explorer';

/** Convert PostflopActionInfo → ActionInfo so shared color/label utils work. */
function toActionInfo(a: PostflopActionInfo): ActionInfo {
  return {
    id: String(a.index),
    label: a.label,
    action_type: a.action_type,
    size_key: a.amount != null ? String(a.amount) : null,
  };
}

function toActionInfos(actions: PostflopActionInfo[]): ActionInfo[] {
  return actions.map(toActionInfo);
}

/** Convert postflop cell to shared MatrixCell format. */
function toMatrixCell(cell: { hand: string; suited: boolean; pair: boolean; probabilities: number[] }, actions: PostflopActionInfo[]): MatrixCell {
  return {
    hand: cell.hand,
    suited: cell.suited,
    pair: cell.pair,
    probabilities: cell.probabilities.map((p, i) => ({
      action: actions[i]?.label ?? '',
      probability: p,
    })),
  };
}

interface PostflopExplorerProps {
  onBack: () => void;
}

export default function PostflopExplorer({ onBack }: PostflopExplorerProps) {
  const [config, setConfig] = useState<PostflopConfig>({
    oop_range: 'QQ+,AKs,AKo',
    ip_range: 'TT+,AQs+,AKo',
    pot: 30,
    effective_stack: 170,
    oop_bet_sizes: '25%,33%,75%',
    oop_raise_sizes: 'a',
    ip_bet_sizes: '25%,33%,75%',
    ip_raise_sizes: 'a',
  });
  const [configSummary, setConfigSummary] = useState<PostflopConfigSummary | null>(null);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [configError, setConfigError] = useState<string | null>(null);
  const [boardInput, setBoardInput] = useState('');
  const [error, setError] = useState<string | null>(null);

  // Solve state
  const [matrix, setMatrix] = useState<PostflopStrategyMatrix | null>(null);
  const [solving, setSolving] = useState(false);
  const [progress, setProgress] = useState<PostflopProgress | null>(null);
  const [actionHistory, setActionHistory] = useState<{index: number; info: PostflopActionInfo}[]>([]);
  const pollRef = useRef<number | null>(null);

  // Navigation state
  const [showFlopPicker, setShowFlopPicker] = useState(false);
  const [showNextCardPicker, setShowNextCardPicker] = useState(false);
  const [awaitingCard, setAwaitingCard] = useState(false);
  const [terminal, setTerminal] = useState(false);
  const [selectedCell, setSelectedCell] = useState<{row: number; col: number} | null>(null);
  const [needsSolve, setNeedsSolve] = useState(false);
  // Track actions per street for close_street call
  const [streetActions, setStreetActions] = useState<number[]>([]);

  const boardCards = useMemo(() => {
    const trimmed = boardInput.trim();
    if (!trimmed) return [];
    return trimmed.split(/\s+/);
  }, [boardInput]);

  useEffect(() => {
    invoke<PostflopConfigSummary>('postflop_set_config', { config })
      .then(setConfigSummary)
      .catch((e) => setError(String(e)));
  }, []);

  const handleConfigSubmit = useCallback(async (newConfig: PostflopConfig) => {
    setConfigError(null);
    try {
      const summary = await invoke<PostflopConfigSummary>('postflop_set_config', { config: newConfig });
      setConfig(newConfig);
      setConfigSummary(summary);
      setShowConfigModal(false);
    } catch (e) {
      setConfigError(String(e));
    }
  }, []);

  // Cleanup polling on unmount
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  /** Start polling for solve progress (shared by initial solve and multi-street). */
  const startPolling = useCallback(() => {
    const poll = async () => {
      try {
        const p = await invoke<PostflopProgress>('postflop_get_progress', {});
        setProgress(p);
        if (p.matrix) setMatrix(p.matrix);
        if (p.is_complete) {
          setSolving(false);
          if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
        }
      } catch (e) {
        setError(String(e));
        setSolving(false);
        if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
      }
    };
    poll();
    pollRef.current = window.setInterval(poll, 2000);
  }, []);


  /** Reset everything for a new hand. */
  const handleReset = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    setMatrix(null);
    setSelectedCell(null);
    setNeedsSolve(false);
    setActionHistory([]);
    setStreetActions([]);
    setBoardInput('');
    setTerminal(false);
    setAwaitingCard(false);
    setProgress(null);
    setError(null);
    setSolving(false);
  }, []);

  /** Start or cancel solve for the current board. */
  const handleSolve = useCallback(() => {
    if (solving) {
      // Cancel
      handleReset();
      return;
    }
    const cards = boardInput.trim().split(/\s+/);
    if (cards.length < 3) return;
    setError(null);
    setSolving(true);
    setNeedsSolve(false);
    setMatrix(null);
    setProgress(null);
    invoke('postflop_solve_street', { board: cards })
      .then(() => startPolling())
      .catch((e) => { setError(String(e)); setSolving(false); });
  }, [solving, boardInput, handleReset, startPolling]);

  /** Navigate the solved tree by clicking an action button. */
  const handleAction = useCallback(async (actionIndex: number) => {
    if (solving) return;
    setError(null);
    setSelectedCell(null);
    try {
      const result = await invoke<PostflopPlayResult>('postflop_play_action', { action: actionIndex });

      const actionInfo = matrix?.actions[actionIndex];
      if (actionInfo) {
        setActionHistory(prev => [...prev, { index: actionIndex, info: actionInfo }]);
      }
      setStreetActions(prev => [...prev, actionIndex]);

      if (result.is_terminal) {
        setMatrix(null);
        setTerminal(true);
        return;
      }

      if (result.is_chance) {
        setMatrix(null);
        setAwaitingCard(true);
        return;
      }

      if (result.matrix) {
        setMatrix(result.matrix);
      }
    } catch (e) {
      setError(String(e));
    }
  }, [solving, matrix]);


  /** Reset everything for a new hand. */
  return (
    <div className="explorer-root">
      {error && <div className="error">{error}</div>}

      <div className="action-strip">
        {/* Split switcher: load dataset / postflop solver */}
        <div className="dataset-switcher-split">
          <div className="dataset-switcher-half" onClick={onBack} title="Load Dataset">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
              <line x1="12" y1="11" x2="12" y2="17" />
              <line x1="9" y1="14" x2="15" y2="14" />
            </svg>
          </div>
          <div className="dataset-switcher-half active" title="Postflop Solver">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="2" y="2" width="20" height="20" rx="2" />
              <path d="M7 12h10M12 7v10" />
            </svg>
          </div>
        </div>

        {/* Config card */}
        <div className="action-block postflop-config-card" onClick={() => setShowConfigModal(true)}>
          <div className="postflop-config-label">Config</div>
          <div className="postflop-config-summary">
            {config.pot} pot / {config.effective_stack} eff
          </div>
          {configSummary && (
            <div className="postflop-config-combos">
              SB: {configSummary.oop_combos} &middot; BB: {configSummary.ip_combos}
            </div>
          )}
        </div>

        {/* Flop street block */}
        <div
          className={`street-block ${boardCards.length === 3 ? '' : 'pending'}`}
          onClick={() => { if (!solving) setShowFlopPicker(true); }}
          style={{ cursor: solving ? 'default' : 'pointer' }}
        >
          <div className="street-block-header">
            <span className="street-name">FLOP</span>
          </div>
          <div className="street-cards">
            {[0, 1, 2].map((i) => {
              const card = boardCards[i];
              if (!card) return <div key={i} className="street-card empty"><span>?</span></div>;
              const rank = card[0]?.toUpperCase();
              const suit = card[1]?.toLowerCase();
              return (
                <div key={i} className="street-card" style={{ backgroundColor: SUIT_COLORS[suit] || '#333' }}>
                  <span className="card-rank">{rank}</span>
                  <span className="card-suit">{SUIT_SYMBOLS[suit] || '?'}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Action history blocks */}
        {actionHistory.map((item, i) => (
          <div key={i} className="action-block" style={{
            borderLeft: `3px solid ${getActionColor(toActionInfo(item.info), toActionInfos(matrix?.actions ?? []))}`,
          }}>
            <span style={{ fontSize: '0.85em' }}>{formatActionLabel(toActionInfo(item.info))}</span>
          </div>
        ))}

        {/* Solve button */}
        {(needsSolve || solving) && !matrix && (
          <div className={`action-block solve-block ${solving ? 'solving' : ''}`} onClick={handleSolve}>
            <span className="solve-label">{solving ? 'CANCEL' : 'SOLVE'}</span>
          </div>
        )}

        {/* Available actions */}
        {matrix && !terminal && !awaitingCard && (
          <ActionBlock
            position={matrix.player === 0 ? 'SB' : 'BB'}
            stack={matrix.stacks[matrix.player]}
            pot={matrix.pot}
            actions={toActionInfos(matrix.actions)}
            onSelect={(actionId) => !solving && handleAction(Number(actionId))}
            isCurrent={!solving}
          />
        )}

        {/* Next street card (turn/river) */}
        {awaitingCard && (
          <div className="street-block pending" style={{ cursor: 'pointer' }}
            onClick={() => setShowNextCardPicker(true)}>
            <div className="street-block-header">
              <span className="street-name">{boardCards.length === 3 ? 'TURN' : 'RIVER'}</span>
            </div>
            <div className="street-cards">
              <div className="street-card empty"><span>?</span></div>
            </div>
          </div>
        )}
      </div>

      {/* Flop card picker modal */}
      {showFlopPicker && (
        <div className="modal-overlay" onClick={() => setShowFlopPicker(false)}>
          <div className="card-picker-container" onClick={(e) => e.stopPropagation()}>
            <p className="card-picker-prompt">Select 3 flop cards</p>
            <FlopPicker
              deadCards={[]}
              onConfirm={(cards) => {
                setBoardInput(cards.join(' '));
                setShowFlopPicker(false);
                setActionHistory([]);
                setStreetActions([]);
                setMatrix(null);
                setProgress(null);
                setTerminal(false);
                setAwaitingCard(false);
                setNeedsSolve(true);
              }}
            />
          </div>
        </div>
      )}

      {/* Next card picker modal (turn/river) */}
      {showNextCardPicker && awaitingCard && (
        <div className="modal-overlay" onClick={() => setShowNextCardPicker(false)}>
          <div className="card-picker-container" onClick={(e) => e.stopPropagation()}>
          <p className="card-picker-prompt">Select {boardCards.length === 3 ? 'turn' : 'river'} card</p>
          <NextCardPicker
            deadCards={boardCards}
            onConfirm={(card) => {
              setShowNextCardPicker(false);
              setAwaitingCard(false);
              setError(null);
              invoke<PostflopStreetResult>('postflop_close_street', { action_history: streetActions })
                .then(() => {
                  const newBoard = [...boardCards, card];
                  setBoardInput(newBoard.join(' '));
                  setStreetActions([]);
                  setMatrix(null);
                  setProgress(null);
                  setNeedsSolve(true);
                })
                .catch((e) => { setError(String(e)); });
            }}
          />
          </div>
        </div>
      )}

      {/* Progress bar */}
      {solving && progress && (
        <div className="progress-bar-container">
          <div className="progress-bar-track">
            <div className="progress-bar-fill" style={{
              width: `${(progress.iteration / Math.max(progress.max_iterations, 1)) * 100}%`,
            }} />
            <span className="progress-text">
              {progress.iteration}/{progress.max_iterations} — exploit: {progress.exploitability.toExponential(2)}
            </span>
          </div>
        </div>
      )}

      {/* Player indicator */}
      {matrix && (
        <div style={{ fontSize: '0.85em', opacity: 0.6, padding: '4px 12px' }}>
          {matrix.player === 0 ? 'SB' : 'BB'} to act — Board: {matrix.board.join(' ')}
        </div>
      )}

      {/* Strategy matrix */}
      {matrix && (() => {
        const actionInfos = toActionInfos(matrix.actions);
        return (
          <div className="matrix-container">
            <div className="matrix-with-detail">
              <div className="hand-matrix">
                {matrix.cells.map((row, rowIdx) => (
                  <div key={rowIdx} className="matrix-row">
                    {row.map((cell, colIdx) => (
                      <HandCell
                        key={colIdx}
                        cell={toMatrixCell(cell, matrix.actions)}
                        actions={actionInfos}
                        reachWeight={cell.combo_count > 0 ? 1 : 0}
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
                    cell={toMatrixCell(matrix.cells[selectedCell.row][selectedCell.col], matrix.actions)}
                    actions={actionInfos}
                  />
                )}
              </div>
            </div>
          </div>
        );
      })()}

      {/* Terminal state */}
      {terminal && (
        <div style={{ padding: 24, textAlign: 'center', opacity: 0.6 }}>
          Hand complete.{' '}
          <button className="action-button" onClick={handleReset}>New Hand</button>
        </div>
      )}

      {/* Awaiting card (no matrix visible) */}
      {awaitingCard && !matrix && !solving && (
        <div style={{ padding: 24, textAlign: 'center', opacity: 0.6 }}>
          Enter the next community card above to continue.
        </div>
      )}

      {/* Config Modal */}
      {showConfigModal && (
        <ConfigModal
          config={config}
          error={configError}
          onSubmit={handleConfigSubmit}
          onClose={() => setShowConfigModal(false)}
        />
      )}
    </div>
  );
}

function ConfigModal({ config, error, onSubmit, onClose }: {
  config: PostflopConfig;
  error: string | null;
  onSubmit: (config: PostflopConfig) => void;
  onClose: () => void;
}) {
  const [draft, setDraft] = useState<PostflopConfig>({ ...config });

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h3>Game Configuration</h3>

        <label>SB Range</label>
        <textarea value={draft.oop_range}
          onChange={(e) => setDraft({ ...draft, oop_range: e.target.value })} rows={2} />

        <label>BB Range</label>
        <textarea value={draft.ip_range}
          onChange={(e) => setDraft({ ...draft, ip_range: e.target.value })} rows={2} />

        <div className="modal-row">
          <div>
            <label>Pot</label>
            <input type="number" value={draft.pot}
              onChange={(e) => setDraft({ ...draft, pot: parseInt(e.target.value) || 0 })} />
          </div>
          <div>
            <label>Effective Stack</label>
            <input type="number" value={draft.effective_stack}
              onChange={(e) => setDraft({ ...draft, effective_stack: parseInt(e.target.value) || 0 })} />
          </div>
        </div>

        <h4>Bet Sizes</h4>
        <div className="modal-row">
          <div>
            <label>SB Bet Sizes</label>
            <input value={draft.oop_bet_sizes}
              onChange={(e) => setDraft({ ...draft, oop_bet_sizes: e.target.value })} />
          </div>
          <div>
            <label>BB Bet Sizes</label>
            <input value={draft.ip_bet_sizes}
              onChange={(e) => setDraft({ ...draft, ip_bet_sizes: e.target.value })} />
          </div>
        </div>

        {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}

        <div className="modal-buttons">
          <button onClick={onClose}>Cancel</button>
          <button className="modal-primary" onClick={() => onSubmit({ ...draft, oop_raise_sizes: 'a', ip_raise_sizes: 'a' })}>Apply</button>
        </div>
      </div>
    </div>
  );
}

const PICKER_RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];
const PICKER_SUITS = ['s', 'h', 'd', 'c'];
const PICKER_COLORS: Record<string, string> = { s: '#fff', h: '#dc2626', d: '#2563eb', c: '#16a34a' };

function FlopPicker({ deadCards, onConfirm }: { deadCards: string[]; onConfirm: (cards: string[]) => void }) {
  const [selected, setSelected] = useState<string[]>([]);
  const deadSet = useMemo(() => new Set(deadCards.map((c) => c.toLowerCase())), [deadCards]);
  const confirmedRef = useRef(false);

  const handleCardClick = (card: string) => {
    if (deadSet.has(card.toLowerCase())) return;
    setSelected((prev) => {
      if (prev.includes(card)) return prev.filter((c) => c !== card);
      if (prev.length >= 3) return prev;
      const next = [...prev, card];
      if (next.length === 3 && !confirmedRef.current) {
        confirmedRef.current = true;
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
                style={{ color: PICKER_COLORS[suit] || '#eee', borderColor: isSelected ? '#00d9ff' : 'transparent' }}
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
    </div>
  );
}

function NextCardPicker({ deadCards, onConfirm }: { deadCards: string[]; onConfirm: (card: string) => void }) {
  const deadSet = useMemo(() => new Set(deadCards.map((c) => c.toLowerCase())), [deadCards]);

  return (
    <div className="card-picker">
      {PICKER_SUITS.map((suit) => (
        <div key={suit} className="card-picker-row">
          {PICKER_RANKS.map((rank) => {
            const card = `${rank}${suit}`;
            const isDead = deadSet.has(card.toLowerCase());
            return (
              <button
                key={card}
                className={`card-picker-card ${isDead ? 'dead' : ''}`}
                style={{ color: PICKER_COLORS[suit] || '#eee' }}
                disabled={isDead}
                onClick={() => onConfirm(card)}
              >
                <span className="picker-rank">{rank}</span>
                <span className="picker-suit">{SUIT_SYMBOLS[suit]}</span>
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
}

