import { useState, useEffect } from 'react';
import { isTauri } from './invoke';
import { useGlobalConfig } from './useGlobalConfig';

export default function Settings() {
  const { config, setConfig } = useGlobalConfig();
  const [browsing, setBrowsing] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'checking' | 'connected' | 'error'>('idle');

  useEffect(() => {
    if (!config.backend_url) {
      setConnectionStatus('idle');
      return;
    }
    setConnectionStatus('checking');
    fetch(`${config.backend_url}/health`)
      .then(res => {
        setConnectionStatus(res.ok ? 'connected' : 'error');
      })
      .catch(() => setConnectionStatus('error'));
  }, [config.backend_url]);

  const handleBrowse = async () => {
    if (!isTauri()) return;
    setBrowsing(true);
    try {
      const { open } = await import('@tauri-apps/plugin-dialog');
      const selected = await open({ directory: true, title: 'Select Blueprint Directory' });
      if (typeof selected === 'string') {
        setConfig({ blueprint_dir: selected });
      }
    } finally {
      setBrowsing(false);
    }
  };

  return (
    <div style={{ maxWidth: 520, margin: '0 auto', padding: '2rem 0' }}>
      <h2 style={{ color: '#eee', marginTop: 0, marginBottom: '1.5rem', fontSize: '1.2rem' }}>
        Settings
      </h2>

      {/* Remote Backend */}
      <div style={{ marginBottom: '1.5rem' }}>
        <label style={{ display: 'block', fontSize: '0.8rem', color: '#888', marginBottom: '0.4rem' }}>
          Remote Backend URL
        </label>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <input
            type="text"
            value={config.backend_url}
            onChange={e => setConfig({ backend_url: e.target.value })}
            placeholder="http://192.168.1.50:3001"
            style={{
              flex: 1,
              padding: '0.45rem 0.6rem',
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,255,255,0.12)',
              borderRadius: 6,
              color: '#eee',
              fontSize: '0.85rem',
              fontFamily: 'inherit',
            }}
          />
          {config.backend_url && (
            <div style={{
              width: 10,
              height: 10,
              borderRadius: '50%',
              flexShrink: 0,
              background: connectionStatus === 'connected' ? '#22c55e'
                : connectionStatus === 'error' ? '#ef4444'
                : connectionStatus === 'checking' ? '#eab308'
                : '#555',
            }} title={connectionStatus} />
          )}
        </div>
        <p style={{ fontSize: '0.7rem', color: '#555', marginTop: '0.3rem' }}>
          Connect to a remote solver backend. Leave empty for local mode.
        </p>
      </div>

      {/* Blueprint Directory */}
      <div style={{ marginBottom: '1.5rem' }}>
        <label style={{ display: 'block', fontSize: '0.8rem', color: '#888', marginBottom: '0.4rem' }}>
          Blueprint Directory
        </label>
        {isTauri() ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{
              flex: 1,
              padding: '0.45rem 0.6rem',
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: 6,
              fontSize: '0.85rem',
              color: config.blueprint_dir ? '#eee' : '#666',
              minHeight: '1.2em',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}>
              {config.blueprint_dir || 'No directory selected'}
            </div>
            <button
              onClick={handleBrowse}
              disabled={browsing}
              style={{
                padding: '0.45rem 1rem',
                fontSize: '0.85rem',
                flexShrink: 0,
              }}
            >
              Browse
            </button>
          </div>
        ) : (
          <input
            type="text"
            value={config.blueprint_dir}
            onChange={e => setConfig({ blueprint_dir: e.target.value })}
            placeholder="/path/to/blueprints"
            style={{
              width: '100%',
              boxSizing: 'border-box',
              padding: '0.45rem 0.6rem',
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,255,255,0.12)',
              borderRadius: 6,
              color: '#eee',
              fontSize: '0.85rem',
              fontFamily: 'inherit',
            }}
          />
        )}
        <p style={{ fontSize: '0.7rem', color: '#555', marginTop: '0.3rem' }}>
          Root directory where trained blueprint bundles are stored.
        </p>
      </div>

      {/* Target Exploitability */}
      <div style={{ marginBottom: '1.5rem' }}>
        <label style={{ display: 'block', fontSize: '0.8rem', color: '#888', marginBottom: '0.4rem' }}>
          Target Exploitability (% of pot)
        </label>
        <input
          type="text"
          value={config.target_exploitability}
          onChange={e => {
            const v = parseFloat(e.target.value);
            if (!isNaN(v) && v >= 0.1 && v <= 100) {
              setConfig({ target_exploitability: v });
            }
          }}
          style={{
            width: 120,
            padding: '0.45rem 0.6rem',
            background: 'rgba(255,255,255,0.06)',
            border: '1px solid rgba(255,255,255,0.12)',
            borderRadius: 6,
            color: '#eee',
            fontSize: '0.85rem',
            fontFamily: 'inherit',
          }}
        />
        <p style={{ fontSize: '0.7rem', color: '#555', marginTop: '0.3rem' }}>
          Default convergence target for postflop solves. Lower is more precise but slower.
        </p>
      </div>

      {/* Solver Dispatch Thresholds */}
      <div style={{ marginBottom: '1.5rem' }}>
        <label style={{ display: 'block', fontSize: '0.8rem', color: '#888', marginBottom: '0.4rem' }}>
          Solver Dispatch Thresholds
        </label>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <div>
            <label style={{ display: 'block', fontSize: '0.7rem', color: '#666', marginBottom: '0.2rem' }}>
              Flop
            </label>
            <input
              type="text"
              value={config.flop_combo_threshold ?? 200}
              onChange={e => {
                const v = parseInt(e.target.value);
                if (!isNaN(v) && v >= 0) setConfig({ flop_combo_threshold: v });
              }}
              style={{
                width: 80,
                padding: '0.45rem 0.6rem',
                background: 'rgba(255,255,255,0.06)',
                border: '1px solid rgba(255,255,255,0.12)',
                borderRadius: 6,
                color: '#eee',
                fontSize: '0.85rem',
                fontFamily: 'inherit',
              }}
            />
          </div>
          <div>
            <label style={{ display: 'block', fontSize: '0.7rem', color: '#666', marginBottom: '0.2rem' }}>
              Turn
            </label>
            <input
              type="text"
              value={config.turn_combo_threshold ?? 300}
              onChange={e => {
                const v = parseInt(e.target.value);
                if (!isNaN(v) && v >= 0) setConfig({ turn_combo_threshold: v });
              }}
              style={{
                width: 80,
                padding: '0.45rem 0.6rem',
                background: 'rgba(255,255,255,0.06)',
                border: '1px solid rgba(255,255,255,0.12)',
                borderRadius: 6,
                color: '#eee',
                fontSize: '0.85rem',
                fontFamily: 'inherit',
              }}
            />
          </div>
        </div>
        <p style={{ fontSize: '0.7rem', color: '#555', marginTop: '0.3rem' }}>
          Max live combos before switching to the depth-limited subgame solver. River always uses full-depth.
        </p>
      </div>

      {/* Solve Iterations */}
      <div style={{ marginBottom: '1.5rem' }}>
        <label style={{ display: 'block', fontSize: '0.8rem', color: '#888', marginBottom: '0.4rem' }}>
          Solve Iterations
        </label>
        <input
          type="text"
          value={config.solve_iterations ?? 200}
          onChange={e => {
            const v = parseInt(e.target.value);
            if (!isNaN(v) && v > 0) setConfig({ solve_iterations: v });
          }}
          style={{
            width: 80,
            padding: '0.45rem 0.6rem',
            background: 'rgba(255,255,255,0.06)',
            border: '1px solid rgba(255,255,255,0.12)',
            borderRadius: 6,
            color: '#eee',
            fontSize: '0.85rem',
            fontFamily: 'inherit',
          }}
        />
        <p style={{ fontSize: '0.7rem', color: '#555', marginTop: '0.3rem' }}>
          Number of DCFR iterations for postflop solving. More iterations = better convergence but slower.
        </p>
      </div>

      {/* Rollout Settings (depth-limited solver) */}
      <div style={{ marginBottom: '1.5rem' }}>
        <label style={{ display: 'block', fontSize: '0.8rem', color: '#888', marginBottom: '0.4rem' }}>
          Rollout Settings (depth-limited solver)
        </label>
        <div style={{ display: 'flex', gap: '0.8rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <div>
            <span style={{ fontSize: '0.7rem', color: '#666', marginRight: '0.3rem' }}>Bias Factor</span>
            <input
              type="text"
              value={config.rollout_bias_factor ?? 10}
              onChange={e => {
                const v = parseFloat(e.target.value);
                if (!isNaN(v) && v > 0) setConfig({ rollout_bias_factor: v });
              }}
              style={{
                width: 55,
                padding: '0.45rem 0.6rem',
                background: 'rgba(255,255,255,0.06)',
                border: '1px solid rgba(255,255,255,0.12)',
                borderRadius: 6,
                color: '#eee',
                fontSize: '0.85rem',
                fontFamily: 'inherit',
              }}
            />
          </div>
          <div>
            <span style={{ fontSize: '0.7rem', color: '#666', marginRight: '0.3rem' }}>Rollouts</span>
            <input
              type="text"
              value={config.rollout_num_samples ?? 3}
              onChange={e => {
                const v = parseInt(e.target.value);
                if (!isNaN(v) && v > 0) setConfig({ rollout_num_samples: v });
              }}
              style={{
                width: 45,
                padding: '0.45rem 0.6rem',
                background: 'rgba(255,255,255,0.06)',
                border: '1px solid rgba(255,255,255,0.12)',
                borderRadius: 6,
                color: '#eee',
                fontSize: '0.85rem',
                fontFamily: 'inherit',
              }}
            />
          </div>
          <div>
            <span style={{ fontSize: '0.7rem', color: '#666', marginRight: '0.3rem' }}>Eval Interval</span>
            <input
              type="text"
              value={config.leaf_eval_interval ?? 10}
              onChange={e => {
                const v = parseInt(e.target.value);
                if (!isNaN(v) && v > 0) setConfig({ leaf_eval_interval: v });
              }}
              style={{
                width: 45,
                padding: '0.45rem 0.6rem',
                background: 'rgba(255,255,255,0.06)',
                border: '1px solid rgba(255,255,255,0.12)',
                borderRadius: 6,
                color: '#eee',
                fontSize: '0.85rem',
                fontFamily: 'inherit',
              }}
            />
          </div>
          <div>
            <span style={{ fontSize: '0.7rem', color: '#666', marginRight: '0.3rem' }}>Opp. Samples</span>
            <input
              type="text"
              value={config.rollout_opponent_samples ?? 8}
              onChange={e => {
                const v = parseInt(e.target.value);
                if (!isNaN(v) && v > 0) setConfig({ rollout_opponent_samples: v });
              }}
              style={{
                width: 45,
                padding: '0.45rem 0.6rem',
                background: 'rgba(255,255,255,0.06)',
                border: '1px solid rgba(255,255,255,0.12)',
                borderRadius: 6,
                color: '#eee',
                fontSize: '0.85rem',
                fontFamily: 'inherit',
              }}
            />
          </div>
          <div>
            <span style={{ fontSize: '0.7rem', color: '#666', marginRight: '0.3rem' }}>Enum. Depth</span>
            <input
              type="text"
              value={config.rollout_enumerate_depth ?? 2}
              onChange={e => {
                const v = parseInt(e.target.value);
                if (!isNaN(v) && v >= 1 && v <= 10) setConfig({ rollout_enumerate_depth: v });
              }}
              style={{
                width: 45,
                padding: '0.45rem 0.6rem',
                background: 'rgba(255,255,255,0.06)',
                border: '1px solid rgba(255,255,255,0.12)',
                borderRadius: 6,
                color: '#eee',
                fontSize: '0.85rem',
                fontFamily: 'inherit',
              }}
            />
          </div>
        </div>
        <p style={{ fontSize: '0.7rem', color: '#555', marginTop: '0.3rem' }}>
          Bias factor multiplies fold/call/raise probabilities in continuation strategies. Rollouts = Monte Carlo samples per street transition. Eval Interval = re-evaluate leaf boundaries every N iterations. Opp. Samples = opponent hands sampled per combo at boundaries. Enum. Depth = decision levels to fully enumerate before sampling (higher = more accurate, slower).
        </p>
      </div>

      {/* Range Clamp Threshold */}
      <div style={{ marginBottom: '1.5rem' }}>
        <label style={{ display: 'block', fontSize: '0.8rem', color: '#888', marginBottom: '0.4rem' }}>
          Range Clamp Threshold
        </label>
        <input
          type="text"
          defaultValue={config.range_clamp_threshold ?? 0.05}
          onBlur={e => {
            const v = parseFloat(e.target.value);
            if (!isNaN(v) && v >= 0 && v <= 1) setConfig({ range_clamp_threshold: v });
          }}
          style={{
            width: 70,
            padding: '0.45rem 0.6rem',
            background: 'rgba(255,255,255,0.06)',
            border: '1px solid rgba(255,255,255,0.12)',
            borderRadius: 6,
            color: '#eee',
            fontSize: '0.85rem',
            fontFamily: 'inherit',
          }}
        />
        <p style={{ fontSize: '0.7rem', color: '#555', marginTop: '0.3rem' }}>
          Zero out combos with reach below this threshold before solving. Removes blueprint noise (trash hands that shouldn't be in range). 0 = no clamping.
        </p>
      </div>

      {/* Stub Range Solver */}
      <div style={{ marginBottom: '1.5rem' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', color: '#eee', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={config.stub_range_solver ?? false}
            onChange={e => setConfig({ stub_range_solver: e.target.checked })}
            style={{ accentColor: '#d4a017' }}
          />
          Stub Range Solver
        </label>
        <p style={{ fontSize: '0.7rem', color: '#555', marginTop: '0.3rem' }}>
          Replace the solver with a stub that returns random data instantly. For UI debugging only.
        </p>
      </div>
    </div>
  );
}
