import { useState, useCallback } from 'react';
import type { GlobalConfig } from './types';

const STORAGE_KEY = 'global_config';
const DEFAULT_CONFIG: GlobalConfig = { blueprint_dir: '', target_exploitability: 3.0 };

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
