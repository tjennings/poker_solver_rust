import { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen, UnlistenFn } from '@tauri-apps/api/event';
import { save, open } from '@tauri-apps/plugin-dialog';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { TrainingProgress, AsyncTrainingResult, TrainedStrategy, Checkpoint } from './types';
import Explorer from './Explorer';

type View = 'training' | 'explorer';

function App() {
  const [view, setView] = useState<View>('explorer');
  const [iterations, setIterations] = useState(10000);
  const [checkpointInterval, setCheckpointInterval] = useState(500);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [strategies, setStrategies] = useState<Record<string, number[]> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<TrainingProgress | null>(null);

  // Listen for training progress events
  useEffect(() => {
    let unlisten: UnlistenFn | null = null;

    const setupListener = async () => {
      unlisten = await listen<TrainingProgress>('training-progress', (event) => {
        const data = event.payload;
        setProgress(data);

        // Add checkpoint
        setCheckpoints((prev) => {
          const last = prev[prev.length - 1];
          if (!last || last.iteration !== data.iteration) {
            return [
              ...prev,
              {
                iteration: data.iteration,
                exploitability: data.exploitability,
                elapsed_ms: data.elapsed_ms,
              },
            ];
          }
          return prev;
        });
      });
    };

    setupListener();

    return () => {
      if (unlisten) {
        unlisten();
      }
    };
  }, []);

  const handleStartTraining = useCallback(async () => {
    setLoading(true);
    setError(null);
    setCheckpoints([]);
    setStrategies(null);
    setProgress(null);

    try {
      const result = await invoke<AsyncTrainingResult>('start_training', {
        iterations,
        checkpointInterval,
      });

      setStrategies(result.strategies);

      if (result.stopped_early) {
        setError('Training stopped early');
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
      setProgress(null);
    }
  }, [iterations, checkpointInterval]);

  const handleStopTraining = useCallback(async () => {
    try {
      await invoke('stop_training');
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const handleSave = useCallback(async () => {
    try {
      const path = await save({
        filters: [{ name: 'Strategy', extensions: ['mpk'] }],
        defaultPath: 'kuhn_strategy.mpk',
      });

      if (path) {
        await invoke('save_strategy', { path });
      }
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const handleLoad = useCallback(async () => {
    try {
      const path = await open({
        filters: [{ name: 'Strategy', extensions: ['mpk'] }],
        multiple: false,
      });

      if (path) {
        const loaded = await invoke<TrainedStrategy>('load_strategy', { path });
        setStrategies(loaded.strategies);
        setCheckpoints([
          {
            iteration: loaded.iterations,
            exploitability: loaded.exploitability,
            elapsed_ms: 0,
          },
        ]);
        setError(null);
      }
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const finalExploitability = checkpoints[checkpoints.length - 1]?.exploitability;

  return (
    <div className="container">
      <header className="app-header">
        <h1>Poker Solver</h1>
        <nav className="tabs">
          <button
            className={`tab ${view === 'explorer' ? 'active' : ''}`}
            onClick={() => setView('explorer')}
          >
            Explorer
          </button>
          <button
            className={`tab ${view === 'training' ? 'active' : ''}`}
            onClick={() => setView('training')}
          >
            Training
          </button>
        </nav>
      </header>

      {view === 'explorer' ? (
        <Explorer />
      ) : (
      <>
      <div className="controls">
        <label>
          Iterations:
          <input
            type="number"
            value={iterations}
            onChange={(e) => setIterations(Number(e.target.value))}
            min={100}
            max={1000000}
            disabled={loading}
          />
        </label>
        <label>
          Checkpoint Interval:
          <input
            type="number"
            value={checkpointInterval}
            onChange={(e) => setCheckpointInterval(Number(e.target.value))}
            min={100}
            max={10000}
            disabled={loading}
          />
        </label>
        <div className="button-group">
          {loading ? (
            <button onClick={handleStopTraining} className="stop-button">
              Stop Training
            </button>
          ) : (
            <button onClick={handleStartTraining}>Train Kuhn Poker</button>
          )}
          <button onClick={handleSave} disabled={!strategies}>
            Save Strategy
          </button>
          <button onClick={handleLoad} disabled={loading}>
            Load Strategy
          </button>
        </div>
      </div>

      {loading && progress && (
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{
              width: `${(progress.iteration / progress.total_iterations) * 100}%`,
            }}
          />
          <span className="progress-text">
            {progress.iteration.toLocaleString()} / {progress.total_iterations.toLocaleString()}
          </span>
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {checkpoints.length > 0 && (
        <>
          <div className="chart-container">
            <h2>Convergence</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={checkpoints}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis
                  dataKey="iteration"
                  stroke="#888"
                  tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
                />
                <YAxis
                  stroke="#888"
                  scale="log"
                  domain={['auto', 'auto']}
                  tickFormatter={(v) => v.toFixed(3)}
                />
                <Tooltip
                  contentStyle={{ background: '#16213e', border: '1px solid #333' }}
                  formatter={(value) => [(value as number).toFixed(6), 'Exploitability']}
                  labelFormatter={(label) => `Iteration ${label}`}
                />
                <Line
                  type="monotone"
                  dataKey="exploitability"
                  stroke="#00d9ff"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="results">
            <h2>Training Results</h2>
            <div className="stats">
              <span>
                Iterations: {checkpoints[checkpoints.length - 1]?.iteration.toLocaleString()}
              </span>
              <span>Time: {checkpoints[checkpoints.length - 1]?.elapsed_ms}ms</span>
              <span>Exploitability: {finalExploitability?.toFixed(6)}</span>
            </div>
            {strategies && (
              <table>
                <thead>
                  <tr>
                    <th>Info Set</th>
                    <th>Pass/Fold</th>
                    <th>Bet/Call</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(strategies)
                    .sort(([a], [b]) => a.localeCompare(b))
                    .map(([infoSet, probs]) => (
                      <tr key={infoSet}>
                        <td>{infoSet}</td>
                        <td className="prob">{(probs[0] * 100).toFixed(1)}%</td>
                        <td className="prob">{(probs[1] * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      )}
      </>
      )}
    </div>
  );
}

export default App;
