import { useState, useEffect } from 'react';
import { isTauri } from './invoke';
import { useGlobalConfig } from './useGlobalConfig';
import type { GlobalConfig } from './types';

const STREETS = ['flop', 'turn', 'river'] as const;
type Street = typeof STREETS[number];

function BoundaryEvaluationSettings({
  config,
  setConfig,
}: {
  config: GlobalConfig;
  setConfig: (update: Partial<GlobalConfig>) => void;
}) {
  // Determine the first cfvnet street (if any) — later streets are disabled.
  const firstCfvnetIdx = STREETS.findIndex(
    s => config[`${s}_boundary_mode`] === 'cfvnet',
  );

  const handlePickModel = async (street: Street) => {
    if (isTauri()) {
      const { open } = await import('@tauri-apps/plugin-dialog');
      const selected = await open({
        filters: [{ name: 'ONNX', extensions: ['onnx'] }],
      });
      if (typeof selected === 'string') {
        setConfig({ [`${street}_model_path`]: selected } as Partial<GlobalConfig>);
      }
    } else {
      const path = window.prompt('Absolute path to .onnx model file');
      if (path) {
        setConfig({ [`${street}_model_path`]: path } as Partial<GlobalConfig>);
      }
    }
  };

  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <div style={{ fontSize: '0.85rem', color: '#ccc', marginBottom: '0.6rem', fontWeight: 500 }}>
        Boundary Evaluation
      </div>
      {STREETS.map((street, idx) => {
        const modeKey = `${street}_boundary_mode` as keyof GlobalConfig;
        const pathKey = `${street}_model_path` as keyof GlobalConfig;
        const mode = config[modeKey] as 'exact' | 'cfvnet';
        const modelPath = config[pathKey] as string;
        const disabled = firstCfvnetIdx !== -1 && idx > firstCfvnetIdx;

        return (
          <div key={street} style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.5rem' }}>
            <span style={{ width: 40, fontSize: '0.8rem', color: '#aaa', textTransform: 'capitalize' }}>
              {street}
            </span>
            <select
              value={mode}
              disabled={disabled}
              onChange={e => setConfig({ [modeKey]: e.target.value } as Partial<GlobalConfig>)}
              style={{
                width: 100,
                padding: '0.35rem 0.5rem',
                background: disabled ? 'rgba(255,255,255,0.02)' : 'rgba(255,255,255,0.06)',
                border: '1px solid rgba(255,255,255,0.12)',
                borderRadius: 6,
                color: disabled ? '#555' : '#eee',
                fontSize: '0.8rem',
                fontFamily: 'inherit',
                cursor: disabled ? 'not-allowed' : 'pointer',
              }}
              title={disabled ? 'Earlier boundary cut takes precedence' : undefined}
            >
              <option value="exact">Exact</option>
              <option value="cfvnet">CFVNet</option>
            </select>
            {mode === 'cfvnet' && !disabled && (
              <>
                <input
                  type="text"
                  value={modelPath}
                  readOnly
                  style={{
                    flex: 1,
                    padding: '0.35rem 0.5rem',
                    background: 'rgba(255,255,255,0.04)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: 6,
                    color: '#ccc',
                    fontSize: '0.75rem',
                    fontFamily: 'inherit',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    minWidth: 0,
                  }}
                  placeholder="No model selected"
                />
                <button
                  onClick={() => handlePickModel(street)}
                  style={{
                    padding: '0.35rem 0.7rem',
                    fontSize: '0.75rem',
                    flexShrink: 0,
                  }}
                >
                  Pick...
                </button>
              </>
            )}
          </div>
        );
      })}
      <p style={{ fontSize: '0.7rem', color: '#555', marginTop: '0.5rem' }}>
        Selecting CFVNet on a street cuts the solve at that street boundary and uses the chosen ONNX model for counterfactual values. Earlier streets must be Exact.
      </p>
    </div>
  );
}

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
          Target Exploitability
        </label>
        <input
          type="text"
          inputMode="decimal"
          defaultValue={config.target_exploitability}
          onBlur={e => {
            const v = parseFloat(e.target.value);
            if (!isNaN(v) && v > 0) {
              setConfig({ target_exploitability: v });
            } else {
              e.target.value = String(config.target_exploitability);
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

      {/* Solve Parameters */}
      <div style={{ marginBottom: '1.5rem' }}>
        <label style={{ display: 'block', fontSize: '0.8rem', color: '#888', marginBottom: '0.4rem' }}>
          Snapshot Interval
        </label>
        <input
          type="text"
          value={config.matrix_snapshot_interval ?? 10}
          onChange={e => {
            const v = parseInt(e.target.value);
            if (!isNaN(v) && v > 0) setConfig({ matrix_snapshot_interval: v });
          }}
          style={{
            width: 60,
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
          Re-snapshot the strategy matrix every N iterations for live UI updates.
        </p>
      </div>

      {/* Boundary Evaluation */}
      <BoundaryEvaluationSettings config={config} setConfig={setConfig} />

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
