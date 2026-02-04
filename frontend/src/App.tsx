import { useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { TrainingResult } from './types';

function App() {
  const [iterations, setIterations] = useState(1000);
  const [result, setResult] = useState<TrainingResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleTrain = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await invoke<TrainingResult>('run_kuhn_training', {
        iterations,
      });
      setResult(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

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
            min={1}
            max={100000}
          />
        </label>
        <button onClick={handleTrain} disabled={loading}>
          {loading ? 'Training...' : 'Train Kuhn Poker'}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="results">
          <h2>Training Results</h2>
          <div className="stats">
            <span>Iterations: {result.iterations}</span>
            <span>Time: {result.elapsed_ms}ms</span>
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
      )}
    </div>
  );
}

export default App;
