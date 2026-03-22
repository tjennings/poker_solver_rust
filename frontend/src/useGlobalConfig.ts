import { useState, useCallback } from 'react';
import type { GlobalConfig } from './types';

const STORAGE_KEY = 'global_config';
const DEFAULT_CONFIG: GlobalConfig = { blueprint_dir: '', target_exploitability: 3.0, flop_combo_threshold: 200, turn_combo_threshold: 300, solve_iterations: 200, backend_url: '', rollout_bias_factor: 10.0, rollout_num_samples: 3, rollout_opponent_samples: 8, leaf_eval_interval: 10, range_clamp_threshold: 0.05 };

function loadConfig(): GlobalConfig {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? { ...DEFAULT_CONFIG, ...JSON.parse(raw) } : DEFAULT_CONFIG;
  } catch {
    return DEFAULT_CONFIG;
  }
}

export function useGlobalConfig() {
  const [config, setConfigState] = useState<GlobalConfig>(loadConfig);
  const setConfig = useCallback((update: Partial<GlobalConfig>) => {
    setConfigState(prev => {
      const next = { ...prev, ...update };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  }, []);
  return { config, setConfig };
}
