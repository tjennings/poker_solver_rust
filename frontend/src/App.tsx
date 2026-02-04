import { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { TrainingResultWithCheckpoints } from './types';

function App() {
  const [iterations, setIterations] = useState(10000);
  const [numCheckpoints, setNumCheckpoints] = useState(20);
  const [result, setResult] = useState<TrainingResultWithCheckpoints | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleTrain = async () => {
    setLoading(true);
    setError(null);
    try {
      const checkpointInterval = Math.max(1, Math.floor(iterations / numCheckpoints));
      const res = await invoke<TrainingResultWithCheckpoints>('train_with_checkpoints', {
        totalIterations: iterations,
        checkpointInterval,
      });
      setResult(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const finalExploitability = result?.checkpoints[result.checkpoints.length - 1]?.exploitability;

  return (
    <div className="container">
      <h1>Poker Solver</h1>

      <div className="controls">
        <label>
          Iterations:
          <input
            type="number"
            value={iterations}
            onChange={(e) => setIterations(Number(e.target.value))}
            min={100}
            max={100000}
          />
        </label>
        <label>
          Checkpoints:
          <input
            type="number"
            value={numCheckpoints}
            onChange={(e) => setNumCheckpoints(Number(e.target.value))}
            min={5}
            max={100}
          />
        </label>
        <button onClick={handleTrain} disabled={loading}>
          {loading ? 'Training...' : 'Train Kuhn Poker'}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          <div className="chart-container">
            <h2>Convergence</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={result.checkpoints}>
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
              <span>Iterations: {result.total_iterations.toLocaleString()}</span>
              <span>Time: {result.total_elapsed_ms}ms</span>
              <span>Exploitability: {finalExploitability?.toFixed(6)}</span>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Info Set</th>
                  <th>Pass/Fold</th>
                  <th>Bet/Call</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(result.strategies)
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
          </div>
        </>
      )}
    </div>
  );
}

export default App;
