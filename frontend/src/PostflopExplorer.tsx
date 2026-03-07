import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { invoke } from './invoke';
import {
  ActionInfo,
  PostflopConfig,
  PostflopConfigSummary,
  PostflopActionInfo,
  PostflopMatrixCell,
  PostflopStrategyMatrix,
  PostflopProgress,
  PostflopPlayResult,
  PostflopStreetResult,
} from './types';
import {
  getActionColor,
  formatActionLabel,
} from './matrix-utils';

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

interface PostflopExplorerProps {
  onBack: () => void;
}

export default function PostflopExplorer({ onBack }: PostflopExplorerProps) {
  const [config, setConfig] = useState<PostflopConfig>({
    oop_range: 'QQ+,AKs,AKo',
    ip_range: 'JJ-66,AQs-ATs,AQo,KQs',
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
  const [awaitingCard, setAwaitingCard] = useState(false);
  const [nextCardInput, setNextCardInput] = useState('');
  const [terminal, setTerminal] = useState(false);
  // Track actions per street for close_street call
  const [streetActions, setStreetActions] = useState<number[]>([]);

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

  const handleSolve = useCallback(async () => {
    const cards = boardInput.trim().split(/\s+/);
    if (cards.length < 3 || cards.length > 5) {
      setError('Enter 3-5 cards (e.g. Ah Kd 7c)');
      return;
    }

    setError(null);
    setSolving(true);
    setActionHistory([]);
    setStreetActions([]);
    setMatrix(null);
    setProgress(null);
    setTerminal(false);
    setAwaitingCard(false);
    setNextCardInput('');

    try {
      await invoke('postflop_solve_street', { board: cards });
    } catch (e) {
      setError(String(e));
      setSolving(false);
      return;
    }

    startPolling();
  }, [boardInput, startPolling]);

  /** Navigate the solved tree by clicking an action button. */
  const handleAction = useCallback(async (actionIndex: number) => {
    if (solving) return;
    setError(null);
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

  /** Deal the next street card (turn or river). */
  const handleNextCard = useCallback(async () => {
    const card = nextCardInput.trim();
    if (!card) return;

    setError(null);
    setAwaitingCard(false);

    try {
      // Filter ranges through the completed street's action history.
      await invoke<PostflopStreetResult>('postflop_close_street', {
        action_history: streetActions,
      });

      // Build new board: current board + new card.
      const currentCards = boardInput.trim().split(/\s+/);
      const newBoard = [...currentCards, card];
      setBoardInput(newBoard.join(' '));
      setStreetActions([]);
      setNextCardInput('');

      // Solve the new street.
      setSolving(true);
      setMatrix(null);
      setProgress(null);
      await invoke('postflop_solve_street', { board: newBoard });

      startPolling();
    } catch (e) {
      setError(String(e));
      setSolving(false);
    }
  }, [nextCardInput, boardInput, streetActions, startPolling]);

  /** Reset everything for a new hand. */
  const handleReset = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    setMatrix(null);
    setActionHistory([]);
    setStreetActions([]);
    setBoardInput('');
    setTerminal(false);
    setAwaitingCard(false);
    setProgress(null);
    setError(null);
    setNextCardInput('');
    setSolving(false);
  }, []);

  return (
    <div className="explorer-root">
      {error && <div className="error">{error}</div>}

      <div className="action-strip">
        {/* Back button */}
        <div className="action-block postflop-back-btn" onClick={onBack}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </div>

        {/* Config card */}
        <div className="action-block postflop-config-card" onClick={() => setShowConfigModal(true)}>
          <div className="postflop-config-label">Config</div>
          <div className="postflop-config-summary">
            {config.pot} pot / {config.effective_stack} eff
          </div>
          {configSummary && (
            <div className="postflop-config-combos">
              OOP: {configSummary.oop_combos} &middot; IP: {configSummary.ip_combos}
            </div>
          )}
        </div>

        {/* Flop input */}
        <div className="action-block postflop-board-input">
          <div className="postflop-config-label">Flop</div>
          <input
            type="text"
            placeholder="Ah Kd 7c"
            value={boardInput}
            onChange={(e) => setBoardInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleSolve(); }}
            className="postflop-card-input"
            disabled={solving}
          />
        </div>

        {/* Action history blocks */}
        {actionHistory.map((item, i) => (
          <div key={i} className="action-block" style={{
            borderLeft: `3px solid ${getActionColor(toActionInfo(item.info), toActionInfos(matrix?.actions ?? []))}`,
          }}>
            <span style={{ fontSize: '0.85em' }}>{formatActionLabel(toActionInfo(item.info))}</span>
          </div>
        ))}

        {/* Next street card input */}
        {awaitingCard && (
          <div className="action-block postflop-board-input">
            <div className="postflop-config-label">
              {boardInput.trim().split(/\s+/).length === 3 ? 'Turn' : 'River'}
            </div>
            <input
              type="text"
              placeholder="5s"
              value={nextCardInput}
              onChange={(e) => setNextCardInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') handleNextCard(); }}
              className="postflop-card-input"
              autoFocus
            />
          </div>
        )}
      </div>

      {/* Progress bar */}
      {solving && progress && (
        <div className="progress-bar-container">
          <div className="progress-bar" style={{
            width: `${(progress.iteration / Math.max(progress.max_iterations, 1)) * 100}%`,
          }} />
          <span className="progress-text">
            {progress.iteration}/{progress.max_iterations} — exploit: {progress.exploitability.toExponential(2)}
          </span>
        </div>
      )}

      {/* Player indicator */}
      {matrix && (
        <div style={{ fontSize: '0.85em', opacity: 0.6, padding: '4px 12px' }}>
          {matrix.player === 0 ? 'OOP' : 'IP'} to act — Board: {matrix.board.join(' ')}
        </div>
      )}

      {/* Strategy matrix */}
      {matrix && (
        <div className="matrix-container">
          <div className="matrix-with-detail">
            <div className="hand-matrix">
              {matrix.cells.map((row, rowIdx) => (
                <div key={rowIdx} className="matrix-row">
                  {row.map((cell, colIdx) => (
                    <PostflopCell
                      key={colIdx}
                      cell={cell}
                      actions={matrix.actions}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Action buttons */}
          <div className="action-buttons">
            {matrix.actions.map((action) => (
              <button
                key={action.index}
                className="action-button"
                style={{ borderColor: getActionColor(toActionInfo(action), toActionInfos(matrix.actions)) }}
                onClick={() => handleAction(action.index)}
                disabled={solving}
              >
                {formatActionLabel(toActionInfo(action))}
              </button>
            ))}
          </div>
        </div>
      )}

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

        <label>OOP Range (Hero)</label>
        <textarea value={draft.oop_range}
          onChange={(e) => setDraft({ ...draft, oop_range: e.target.value })} rows={2} />

        <label>IP Range (Villain)</label>
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
            <label>OOP Bet</label>
            <input value={draft.oop_bet_sizes}
              onChange={(e) => setDraft({ ...draft, oop_bet_sizes: e.target.value })} />
          </div>
          <div>
            <label>OOP Raise</label>
            <input value={draft.oop_raise_sizes}
              onChange={(e) => setDraft({ ...draft, oop_raise_sizes: e.target.value })} />
          </div>
        </div>
        <div className="modal-row">
          <div>
            <label>IP Bet</label>
            <input value={draft.ip_bet_sizes}
              onChange={(e) => setDraft({ ...draft, ip_bet_sizes: e.target.value })} />
          </div>
          <div>
            <label>IP Raise</label>
            <input value={draft.ip_raise_sizes}
              onChange={(e) => setDraft({ ...draft, ip_raise_sizes: e.target.value })} />
          </div>
        </div>

        {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}

        <div className="modal-buttons">
          <button onClick={onClose}>Cancel</button>
          <button className="modal-primary" onClick={() => onSubmit(draft)}>Apply</button>
        </div>
      </div>
    </div>
  );
}

function PostflopCell({ cell, actions }: {
  cell: PostflopMatrixCell;
  actions: PostflopActionInfo[];
}) {
  const isUnreachable = cell.combo_count === 0;
  const actionInfos = useMemo(() => toActionInfos(actions), [actions]);

  const gradientStops = useMemo(() => {
    if (isUnreachable || cell.probabilities.length === 0) {
      return 'rgba(30, 41, 59, 1)';
    }

    const stops: string[] = [];
    let position = 0;

    // Build gradient from last action to first (aggressive on left)
    for (let idx = cell.probabilities.length - 1; idx >= 0; idx--) {
      const prob = cell.probabilities[idx];
      if (prob <= 0) continue;

      const color = getActionColor(actionInfos[idx], actionInfos);
      const width = prob * 100;
      stops.push(`${color} ${position}%`);
      stops.push(`${color} ${position + width}%`);
      position += width;
    }

    if (stops.length === 0) return 'rgba(30, 41, 59, 1)';
    return `linear-gradient(to right, ${stops.join(', ')})`;
  }, [cell.probabilities, actionInfos, isUnreachable]);

  return (
    <div className={`matrix-cell ${isUnreachable ? 'unreachable' : ''}`}>
      <div
        className="cell-bar"
        style={{
          background: isUnreachable ? undefined : gradientStops,
          height: '100%',
        }}
      />
      <span className="cell-label">{cell.hand}</span>
    </div>
  );
}
