import { useState, useCallback, useEffect } from 'react';
import { invoke } from './invoke';
import {
  PostflopConfig,
  PostflopConfigSummary,
  PostflopActionInfo,
  PostflopStrategyMatrix,
  PostflopProgress,
} from './types';
import {
  getActionColor,
  formatActionLabel,
} from './matrix-utils';

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

  // Placeholders for Tasks 10-11
  const [matrix, _setMatrix] = useState<PostflopStrategyMatrix | null>(null);
  const [_solving, _setSolving] = useState(false);
  const [_progress, _setProgress] = useState<PostflopProgress | null>(null);
  const [actionHistory, _setActionHistory] = useState<{index: number; info: PostflopActionInfo}[]>([]);

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
            className="postflop-card-input"
          />
        </div>

        {/* Action history blocks (Task 11) */}
        {actionHistory.map((item, i) => (
          <div key={i} className="action-block" style={{
            borderLeft: `3px solid ${getActionColor(item.info as any, (matrix?.actions ?? []) as any)}`,
          }}>
            <span style={{ fontSize: '0.85em' }}>{formatActionLabel(item.info as any)}</span>
          </div>
        ))}
      </div>

      {/* Matrix placeholder (Task 10) */}
      {matrix && <div>Matrix goes here</div>}

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
