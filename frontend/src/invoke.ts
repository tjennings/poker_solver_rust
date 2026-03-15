const DEV_SERVER_URL = 'http://localhost:3001';

export function isTauri(): boolean {
  return '__TAURI__' in window || '__TAURI_INTERNALS__' in window;
}

export async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  if (isTauri()) {
    const { invoke: tauriInvoke } = await import('@tauri-apps/api/core');
    return tauriInvoke<T>(cmd, args);
  }

  const res = await fetch(`${DEV_SERVER_URL}/api/${cmd}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(args ?? {}),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text);
  }

  return res.json();
}

// ---------------------------------------------------------------------------
// GPU resolve types and invoke wrappers
// ---------------------------------------------------------------------------

export interface ModelStackConfig {
  river_layers: number;
  river_hidden: number;
  turn_layers: number;
  turn_hidden: number;
  flop_layers: number;
  flop_hidden: number;
  preflop_layers: number;
  preflop_hidden: number;
}

export interface GpuGameState {
  board: number[];
  oop_range: number[];
  ip_range: number[];
  pot: number;
  effective_stack: number;
  oop_bet_sizes: string;
  oop_raise_sizes: string;
  ip_bet_sizes: string;
  ip_raise_sizes: string;
}

export interface GpuResolveResult {
  strategy: number[];
  action_names: string[];
  evs: number[];
  iterations: number;
  player: number;
  num_hands: number;
  num_actions: number;
}

export async function loadGpuModelSet(path: string, config: ModelStackConfig): Promise<void> {
  return invoke('load_gpu_model_set', { path, config });
}

export async function gpuResolve(gameState: GpuGameState, maxIterations: number): Promise<GpuResolveResult> {
  return invoke('gpu_resolve', { game_state: gameState, max_iterations: maxIterations });
}

export async function gpuResolveProgressive(gameState: GpuGameState, checkpoints: number[]): Promise<GpuResolveResult[]> {
  return invoke('gpu_resolve_progressive', { game_state: gameState, checkpoints });
}

export async function isGpuModelLoaded(): Promise<boolean> {
  return invoke('is_gpu_model_loaded');
}
