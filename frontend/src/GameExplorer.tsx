import { useState, useCallback, useMemo, useEffect } from 'react';
import { invoke } from './invoke';
import type {
  GameState,
} from './game-types';
import { HandCell, CellDetail, ActionBlock } from './Explorer';
import { toMatrixCell } from './game-explorer-utils';
import {
  SUIT_COLORS,
  SUIT_SYMBOLS,
} from './matrix-utils';

// ── Card picker (local, since Explorer.tsx does not export it) ──────────

const PICKER_RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];
const PICKER_SUITS = ['s', 'h', 'd', 'c'];
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

  const deadSet = useMemo(
    () => new Set(deadCards.map((c) => c.toLowerCase())),
    [deadCards],
  );

  const handleCardClick = (card: string) => {
    if (deadSet.has(card.toLowerCase())) return;

    setSelected((prev) => {
      if (prev.includes(card)) {
        return prev.filter((c) => c !== card);
      }
      if (prev.length >= expectedCards) return prev;
      const next = [...prev, card];
      if (expectedCards === 1 && next.length === 1) {
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
            const isSel = selected.includes(card);
            return (
              <button
                key={card}
                className={`card-picker-card ${isDead ? 'dead' : ''} ${isSel ? 'selected' : ''}`}
                style={{
                  color: PICKER_COLORS[suit] || '#eee',
                  borderColor: isSel ? '#00d9ff' : 'transparent',
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

// (BoardCards and HistoryBreadcrumbs removed — rendered inline in action strip)

// ── Solve progress bar ──────────────────────────────────────────────────

function SolveProgress({ state }: { state: GameState }) {
  const solve = state.solve;
  if (!solve) return null;

  const pct = solve.max_iterations > 0
    ? Math.min((solve.iteration / solve.max_iterations) * 100, 100)
    : 0;

  return (
    <div className="progress-bar-container">
      <div className="progress-bar-track">
        <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
        <div className="progress-text">
          {solve.solver_name} - {solve.iteration}/{solve.max_iterations} iters
          {' '} | expl: {solve.exploitability.toFixed(3)}
          {' '} | {solve.elapsed_secs.toFixed(1)}s
          {solve.is_complete ? ' (done)' : ''}
        </div>
      </div>
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────────

export default function GameExplorer() {
  const [state, setState] = useState<GameState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedCell, setSelectedCell] = useState<{
    row: number;
    col: number;
  } | null>(null);

  const [blueprints, setBlueprints] = useState<{ name: string; path: string; stack_depth: number; latest_snapshot: string | null }[]>([]);
  const [bundleName, setBundleName] = useState<string | null>(null);
  const [solving, setSolving] = useState(false);

  // ── List available blueprints on mount ──────────────────────────

  useEffect(() => {
    const init = async () => {
      try {
        const globalConfig = JSON.parse(localStorage.getItem('global_config') || '{}');
        if (!globalConfig.blueprint_dir) {
          setError('Set Blueprint Directory in Settings first');
          return;
        }
        const list = await invoke<{ name: string; path: string; stack_depth: number; has_strategy: boolean; latest_snapshot: string | null }[]>(
          'list_blueprints', { dir: globalConfig.blueprint_dir }
        );
        setBlueprints(list.filter(b => b.has_strategy));
      } catch (e) {
        setError(String(e));
      }
    };
    init();
  }, []);

  // ── Load a specific blueprint and start game session ────────────

  const loadBlueprint = useCallback(async (path: string, name: string) => {
    try {
      setLoading(true);
      setError(null);
      setState(null);
      const info = await invoke<{ snapshot_name: string | null }>('load_blueprint_v2', { path });
      await invoke('game_new', {});
      const s = await invoke<GameState>('game_get_state', {});
      setState(s);
      setBundleName(info.snapshot_name ? `${name} (${info.snapshot_name})` : name);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  // ── Action handler ───────────────────────────────────────────────

  const playAction = useCallback(async (actionId: string) => {
    try {
      setLoading(true);
      const s = await invoke<GameState>('game_play_action', {
        actionId,
      });
      setState(s);
      setSelectedCell(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  // ── Deal card handler ────────────────────────────────────────────

  const dealCard = useCallback(async (card: string) => {
    try {
      setLoading(true);
      const s = await invoke<GameState>('game_deal_card', { card });
      setState(s);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const dealCards = useCallback(
    async (cards: string[]) => {
      try {
        setLoading(true);
        let s: GameState | null = null;
        for (const card of cards) {
          s = await invoke<GameState>('game_deal_card', { card });
        }
        if (s) setState(s);
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  // ── Back handler ─────────────────────────────────────────────────

  const goBack = useCallback(async () => {
    try {
      setLoading(true);
      const s = await invoke<GameState>('game_back', {});
      setState(s);
      setSelectedCell(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  // ── New hand ─────────────────────────────────────────────────────

  const newHand = useCallback(async () => {
    try {
      setLoading(true);
      await invoke('game_new', {});
      const s = await invoke<GameState>('game_get_state', {});
      setState(s);
      setSelectedCell(null);
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  // ── Rewind to a specific history point ───────────────────────────

  const rewindTo = useCallback(
    async (index: number) => {
      if (!state) return;
      const stepsBack = state.action_history.length - index;
      if (stepsBack <= 0) return;
      try {
        setLoading(true);
        // Call game_back once for each step to undo.
        // The backend replays from root each time, so this is correct
        // even across street boundaries.
        let s: GameState | null = null;
        for (let i = 0; i < stepsBack; i++) {
          s = await invoke<GameState>('game_back', {});
        }
        if (s) setState(s);
        setSelectedCell(null);
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    },
    [state],
  );

  // ── Derived state ────────────────────────────────────────────────

  const matrixActions = state?.matrix?.actions ?? state?.actions ?? [];
  const selectedCellData =
    selectedCell && state?.matrix
      ? state.matrix.cells[selectedCell.row]?.[selectedCell.col]
      : null;

  // Determine how many cards the next street needs
  const expectedCards = useMemo(() => {
    if (!state?.is_chance) return 0;
    const boardLen = state.board.length;
    if (boardLen === 0) return 3; // flop
    if (boardLen === 3) return 1; // turn
    if (boardLen === 4) return 1; // river
    return 0;
  }, [state]);

  const nextStreetLabel = useMemo(() => {
    if (!state?.is_chance) return '';
    const boardLen = state.board.length;
    if (boardLen === 0) return 'FLOP';
    if (boardLen === 3) return 'TURN';
    if (boardLen === 4) return 'RIVER';
    return '';
  }, [state]);

  // ── Render ───────────────────────────────────────────────────────

  if (loading) {
    return <div className="explorer"><div className="loading">Loading...</div></div>;
  }

  // Blueprint picker — shown when no game session is active
  if (!state) {
    return (
      <div className="explorer">
        {error && (
          <div className="error" onClick={() => setError(null)} style={{ cursor: 'pointer' }}>
            {error}
          </div>
        )}
        <div style={{ padding: '2rem', maxWidth: '500px', margin: '0 auto' }}>
          <h2 style={{ color: '#e2e8f0', marginBottom: '1rem', fontSize: '1.1rem' }}>Select Blueprint</h2>
          {blueprints.length === 0 ? (
            <p style={{ color: '#94a3b8' }}>No blueprints found. Set Blueprint Directory in Settings.</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {blueprints.map((bp) => (
                <button
                  key={bp.path}
                  onClick={() => loadBlueprint(bp.path, bp.name)}
                  style={{
                    padding: '0.75rem 1rem',
                    background: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: '6px',
                    color: '#e2e8f0',
                    cursor: 'pointer',
                    textAlign: 'left',
                    fontSize: '0.85rem',
                  }}
                >
                  <div style={{ fontWeight: 600 }}>{bp.name}</div>
                  <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: '2px' }}>
                    {bp.stack_depth > 0 ? `${bp.stack_depth}bb` : ''}{bp.latest_snapshot ? ` — ${bp.latest_snapshot}` : ''}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="explorer">
      {error && (
        <div className="error" onClick={() => setError(null)} style={{ cursor: 'pointer' }}>
          {error}
        </div>
      )}

      {/* Action strip: history breadcrumbs + current actions */}
      {state && (
        <div className="action-strip">
          {/* Back / New Hand / Blueprint buttons */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', flexShrink: 0 }}>
            {bundleName && (
              <button
                style={{
                  padding: '2px 6px',
                  fontSize: '0.55rem',
                  background: 'transparent',
                  border: '1px solid #334155',
                  borderRadius: '4px',
                  color: '#94a3b8',
                  cursor: 'pointer',
                }}
                onClick={() => { setState(null); setBundleName(null); }}
                title="Change blueprint"
              >
                {bundleName}
              </button>
            )}
            <button
              style={{
                padding: '4px 8px',
                fontSize: '0.65rem',
                background: '#1e293b',
                border: '1px solid #444',
                borderRadius: '4px',
                color: '#ccc',
              }}
              disabled={state.action_history.length === 0}
              onClick={goBack}
            >
              Back
            </button>
            <button
              style={{
                padding: '4px 8px',
                fontSize: '0.65rem',
                background: '#1e293b',
                border: '1px solid #444',
                borderRadius: '4px',
                color: '#ccc',
              }}
              onClick={newHand}
            >
              New
            </button>
          </div>

          {/* Action history as full ActionBlocks with street cards */}
          {(() => {
            const elems: JSX.Element[] = [];
            let prevStreet = '';
            const board = state.board;

            const streetBlock = (streetName: string, cards: string[]) => (
              <div key={`street-${streetName}`} className="street-block">
                <div className="street-block-header">
                  <span className="street-name">{streetName}</span>
                </div>
                <div className="street-cards">
                  {cards.map((card, i) => {
                    const rank = card[0]?.toUpperCase();
                    const suit = card[1]?.toLowerCase();
                    const bgColor = SUIT_COLORS[suit] || '#333';
                    return (
                      <div key={i} className="street-card" style={{ backgroundColor: bgColor }}>
                        <span className="card-rank">{rank}</span>
                        <span className="card-suit">{SUIT_SYMBOLS[suit] || '?'}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            );

            const solveBtn = (key: string) => (
              <button
                key={key}
                onClick={async () => {
                  if (solving) {
                    // Cancel
                    try {
                      setSolving(false);
                      await invoke('game_cancel_solve', {});
                      const s = await invoke<GameState>('game_get_state', {});
                      setState(s);
                    } catch (e) {
                      setError(String(e));
                    }
                  } else {
                    // Start solve
                    try {
                      setSolving(true);
                      const s = await invoke<GameState>('game_solve', {});
                      setState(s);
                    } catch (e) {
                      setError(String(e));
                      setSolving(false);
                    }
                  }
                }}
                style={{
                  padding: '0.3rem 0.4rem',
                  background: solving ? '#dc262622' : '#f59e0b22',
                  border: `2px solid ${solving ? '#dc2626' : '#f59e0b'}`,
                  borderRadius: '4px',
                  color: solving ? '#dc2626' : '#f59e0b',
                  cursor: 'pointer',
                  fontSize: '0.7rem',
                  fontWeight: 700,
                  writingMode: 'vertical-rl',
                  textOrientation: 'mixed',
                  letterSpacing: '0.15em',
                  alignSelf: 'stretch',
                  flexShrink: 0,
                  minHeight: '60px',
                }}
              >
                {solving ? 'CANCEL' : 'SOLVE'}
              </button>
            );

            state.action_history.forEach((rec, idx) => {
              // Insert street block at transitions
              if (rec.street !== prevStreet && prevStreet !== '') {
                const cards = prevStreet === 'Preflop' ? board.slice(0, 3)
                  : prevStreet === 'Flop' ? board.slice(3, 4)
                  : prevStreet === 'Turn' ? board.slice(4, 5)
                  : [];
                if (cards.length > 0) {
                  const nextStreet = rec.street;
                  elems.push(streetBlock(nextStreet, cards));
                  // Solve button after each street transition
                  elems.push(solveBtn(`solve-${prevStreet}`));
                }
              }
              prevStreet = rec.street;

              // Full ActionBlock with all available actions, selected highlighted
              elems.push(
                <ActionBlock
                  key={`action-${idx}`}
                  position={rec.position}
                  stack={rec.stack}
                  pot={rec.pot}
                  actions={rec.actions.length > 0 ? rec.actions : [{ id: rec.action_id, label: rec.label, action_type: 'bet' }]}
                  selectedAction={rec.action_id}
                  onSelect={() => rewindTo(idx)}
                  onHeaderClick={() => rewindTo(idx)}
                  isCurrent={false}
                />
              );
            });

            // Street block after last action if we transitioned to a new street
            if (prevStreet && prevStreet !== state.street && !state.is_chance) {
              const cards = prevStreet === 'Preflop' ? board.slice(0, 3)
                : prevStreet === 'Flop' ? board.slice(3, 4)
                : prevStreet === 'Turn' ? board.slice(4, 5)
                : [];
              if (cards.length > 0) {
                elems.push(streetBlock(state.street, cards));
                elems.push(solveBtn(`solve-${prevStreet}-end`));
              }
            }

            return elems;
          })()}

          {/* Card picker shown as street block when at chance node */}
          {state.is_chance && (
            <div className="street-block pending">
              <div className="street-block-header">
                <span className="street-name">{nextStreetLabel}</span>
              </div>
              <div className="street-cards">
                {Array.from({ length: expectedCards }).map((_, i) => (
                  <div key={i} className="street-card empty">
                    <span className="card-rank">?</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Current action block (non-terminal, non-chance) */}
          {!state.is_terminal && !state.is_chance && matrixActions.length > 0 && (
            <ActionBlock
              position={state.position}
              stack={state.stacks[state.position === 'SB' ? 1 : 0]}
              pot={state.pot}
              actions={matrixActions}
              onSelect={playAction}
              isCurrent={true}
            />
          )}
        </div>
      )}

      {/* Strategy matrix */}
      {state?.matrix && (
        <div className="matrix-container">
          <div className="matrix-with-detail">
            <div className="hand-matrix">
              {state.matrix.cells.map((row, rowIdx) => (
                <div key={rowIdx} className="matrix-row">
                  {row.map((cell, colIdx) => {
                    const matCell = toMatrixCell(cell, matrixActions);
                    return (
                      <HandCell
                        key={colIdx}
                        cell={matCell}
                        actions={matrixActions}
                        reachWeight={cell.weight}
                        isSelected={
                          selectedCell?.row === rowIdx &&
                          selectedCell?.col === colIdx
                        }
                        onClick={() =>
                          setSelectedCell({ row: rowIdx, col: colIdx })
                        }
                        overlayText={
                          cell.ev != null
                            ? `${cell.ev >= 0 ? '+' : ''}${(cell.ev / 2).toFixed(1)}`
                            : undefined
                        }
                      />
                    );
                  })}
                </div>
              ))}
            </div>

            {/* Detail panel — right rail */}
            <div className="detail-column" style={{ width: selectedCellData ? '300px' : '0', transition: 'width 0.15s' }}>
              {selectedCellData && (
                <>
                  <CellDetail
                    cell={toMatrixCell(selectedCellData, matrixActions)}
                    actions={matrixActions}
                  />
                  {/* Per-combo breakdown grid */}
                  {selectedCellData.combos.length > 0 && selectedCellData.combos[0].probabilities.length > 0 && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginTop: '0.5rem' }}>
                      {selectedCellData.combos.map((combo) => {
                        const r1 = combo.cards[0]?.toUpperCase();
                        const s1 = combo.cards[1]?.toLowerCase();
                        const r2 = combo.cards[2]?.toUpperCase();
                        const s2 = combo.cards[3]?.toLowerCase();
                        return (
                          <div
                            key={combo.cards}
                            style={{
                              flex: '0 0 auto',
                              minWidth: '80px',
                              padding: '0.3rem 0.4rem',
                              background: 'rgba(255,255,255,0.03)',
                              border: '1px solid #334155',
                              borderRadius: '4px',
                            }}
                          >
                            <div style={{ fontWeight: 700, fontSize: '0.8rem', marginBottom: '0.2rem' }}>
                              <span>{r1}</span>
                              <span style={{ color: SUIT_COLORS[s1] || '#fff' }}>{SUIT_SYMBOLS[s1] || ''}</span>
                              {' '}
                              <span>{r2}</span>
                              <span style={{ color: SUIT_COLORS[s2] || '#fff' }}>{SUIT_SYMBOLS[s2] || ''}</span>
                            </div>
                            {matrixActions.map((action, i) => {
                              const pct = (combo.probabilities[i] || 0) * 100;
                              if (pct < 0.5) return null;
                              return (
                                <div key={action.id} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', color: '#94a3b8' }}>
                                  <span>{action.label}</span>
                                  <span>{pct.toFixed(0)}%</span>
                                </div>
                              );
                            })}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Card picker for chance nodes */}
      {state?.is_chance && expectedCards > 0 && (
        <div className="card-picker-container">
          <p className="card-picker-prompt">
            Select {nextStreetLabel.toLowerCase()} card
            {expectedCards > 1 ? 's' : ''}
          </p>
          <CardPicker
            expectedCards={expectedCards}
            deadCards={state.board}
            onConfirm={expectedCards === 1 ? (cards) => dealCard(cards[0]) : dealCards}
          />
        </div>
      )}

      {/* Terminal state */}
      {state?.is_terminal && (
        <div className="hand-complete">
          <p className="hand-complete-result">
            Hand complete — Pot: {state.pot}
          </p>
          <button className="new-hand-btn" onClick={newHand}>
            New Hand
          </button>
        </div>
      )}

      {/* Solve progress */}
      {state && <SolveProgress state={state} />}

      {loading && <div className="loading">Loading...</div>}
    </div>
  );
}
