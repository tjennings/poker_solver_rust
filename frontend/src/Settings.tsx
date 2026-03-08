import { useState } from 'react';
import { useGlobalConfig } from './useGlobalConfig';

const isTauri = '__TAURI__' in window;

export default function Settings() {
  const { config, setConfig } = useGlobalConfig();
  const [browsing, setBrowsing] = useState(false);

  const handleBrowse = async () => {
    if (!isTauri) return;
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

      {/* Blueprint Directory */}
      <div style={{ marginBottom: '1.5rem' }}>
        <label style={{ display: 'block', fontSize: '0.8rem', color: '#888', marginBottom: '0.4rem' }}>
          Blueprint Directory
        </label>
        {isTauri ? (
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
          type="number"
          value={config.target_exploitability}
          onChange={e => {
            const v = parseFloat(e.target.value);
            if (!isNaN(v) && v >= 0.1 && v <= 100) {
              setConfig({ target_exploitability: v });
            }
          }}
          min={0.1}
          max={100}
          step={0.1}
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
    </div>
  );
}
