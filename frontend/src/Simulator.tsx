import { useState, useEffect, useCallback } from 'react';
import { invoke } from './invoke';
import { listen } from '@tauri-apps/api/event';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import type { StrategySourceInfo, SimulationProgress, SimulationResult } from './types';

function Simulator() {
  const [sources, setSources] = useState<StrategySourceInfo[]>([]);
  const [p1Path, setP1Path] = useState('');
  const [p2Path, setP2Path] = useState('');
  const [numHands, setNumHands] = useState(10000);
  const [stackDepth, setStackDepth] = useState(100);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<SimulationProgress | null>(null);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    invoke<StrategySourceInfo[]>('list_strategy_sources')
      .then(setSources)
      .catch(e => setError(String(e)));
  }, []);

  useEffect(() => {
    const unlistenProgress = listen<SimulationProgress>('simulation-progress', (event) => {
      setProgress(event.payload);
    });

    const unlistenComplete = listen<SimulationResult>('simulation-complete', (event) => {
      setResult(event.payload);
      setRunning(false);
    });

    const unlistenError = listen<string>('simulation-error', (event) => {
      setError(event.payload);
      setRunning(false);
      setProgress(null);
    });

    return () => {
      unlistenProgress.then(f => f());
      unlistenComplete.then(f => f());
      unlistenError.then(f => f());
    };
  }, []);

  const handleStart = useCallback(async () => {
    if (!p1Path || !p2Path) {
      setError('Select both P1 and P2 sources');
      return;
    }
    setError(null);
    setResult(null);
    setProgress(null);
    setRunning(true);

    try {
      await invoke('start_simulation', {
        p1Path,
        p2Path,
        numHands,
        stackDepth,
      });
    } catch (e) {
      setError(String(e));
      setRunning(false);
    }
  }, [p1Path, p2Path, numHands, stackDepth]);

  const handleStop = useCallback(async () => {
    try {
      await invoke('stop_simulation');
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const pct = progress
    ? Math.round((progress.hands_played / progress.total_hands) * 100)
    : 0;

  const currentMbbh = progress?.current_mbbh ?? result?.mbbh;

  return (
    <div className="simulator">
      <div className="sim-config">
        <div className="sim-row">
          <label className="sim-label">P1 (Hero)</label>
          <select
            className="sim-select"
            value={p1Path}
            onChange={e => setP1Path(e.target.value)}
            disabled={running}
          >
            <option value="">Select source...</option>
            {sources.map(s => (
              <option key={s.path} value={s.path}>
                {s.name} ({s.source_type})
              </option>
            ))}
          </select>
        </div>

        <div className="sim-row">
          <label className="sim-label">P2 (Villain)</label>
          <select
            className="sim-select"
            value={p2Path}
            onChange={e => setP2Path(e.target.value)}
            disabled={running}
          >
            <option value="">Select source...</option>
            {sources.map(s => (
              <option key={s.path} value={s.path}>
                {s.name} ({s.source_type})
              </option>
            ))}
          </select>
        </div>

        <div className="sim-row">
          <label className="sim-label">Hands</label>
          <input
            type="number"
            className="sim-input"
            value={numHands}
            onChange={e => setNumHands(Number(e.target.value))}
            min={100}
            max={1000000}
            step={1000}
            disabled={running}
          />
        </div>

        <div className="sim-row">
          <label className="sim-label">Stack (BB)</label>
          <input
            type="number"
            className="sim-input"
            value={stackDepth}
            onChange={e => setStackDepth(Number(e.target.value))}
            min={10}
            max={500}
            step={10}
            disabled={running}
          />
        </div>

        <div className="sim-actions">
          {!running ? (
            <button onClick={handleStart} disabled={!p1Path || !p2Path}>
              Run Simulation
            </button>
          ) : (
            <button onClick={handleStop} className="stop-btn">
              Stop
            </button>
          )}
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {(running || result) && (
        <div className="sim-progress">
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{
                width: result
                  ? '100%'
                  : `${pct}%`,
              }}
            />
            <span className="progress-text">
              {(progress?.hands_played ?? result?.hands_played ?? 0).toLocaleString()}
              {' / '}
              {(progress?.total_hands ?? result?.hands_played ?? numHands).toLocaleString()}
              {' hands'}
            </span>
          </div>
          {currentMbbh != null && (
            <div className={`sim-live-stat ${currentMbbh >= 0 ? 'positive' : 'negative'}`}>
              {currentMbbh >= 0 ? '+' : ''}{currentMbbh.toFixed(1)} mbb/h
            </div>
          )}
        </div>
      )}

      {result && (
        <div className="sim-results">
          <div className="sim-stats">
            <div className="sim-stat">
              <span className="sim-stat-label">P1 Profit</span>
              <span className="sim-stat-value">
                {result.p1_profit_bb >= 0 ? '+' : ''}{result.p1_profit_bb.toFixed(1)} BB
              </span>
            </div>
            <div className="sim-stat">
              <span className="sim-stat-label">Hands</span>
              <span className="sim-stat-value">{result.hands_played.toLocaleString()}</span>
            </div>
            <div className="sim-stat">
              <span className="sim-stat-label">Time</span>
              <span className="sim-stat-value">{(result.elapsed_ms / 1000).toFixed(1)}s</span>
            </div>
          </div>

          {result.equity_curve.length > 1 && (
            <div className="sim-chart">
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={result.equity_curve.map((mbbh, i) => ({
                  batch: (i + 1) * 100,
                  mbbh: Math.round(mbbh * 10) / 10,
                }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis
                    dataKey="batch"
                    tick={{ fill: '#888', fontSize: 12 }}
                    label={{ value: 'Hands', position: 'bottom', fill: '#888', fontSize: 12 }}
                  />
                  <YAxis
                    tick={{ fill: '#888', fontSize: 12 }}
                    label={{ value: 'mbb/h', angle: -90, position: 'insideLeft', fill: '#888', fontSize: 12 }}
                  />
                  <Tooltip
                    contentStyle={{ background: '#16213e', border: '1px solid #333', borderRadius: 4 }}
                    labelStyle={{ color: '#888' }}
                    itemStyle={{ color: '#00d9ff' }}
                  />
                  <ReferenceLine y={0} stroke="#555" strokeDasharray="3 3" />
                  <Line
                    type="monotone"
                    dataKey="mbbh"
                    stroke="#00d9ff"
                    strokeWidth={2}
                    dot={false}
                    name="mbb/h"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {!running && !result && !error && (
        <div className="sim-empty">
          Select two strategy sources and run a simulation to compare their performance.
        </div>
      )}
    </div>
  );
}

export default Simulator;
