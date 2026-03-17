const DEV_SERVER_URL = 'http://localhost:3001';
const STORAGE_KEY = 'global_config';

export function isTauri(): boolean {
  return '__TAURI__' in window || '__TAURI_INTERNALS__' in window;
}

/** Read backend_url from localStorage (avoids React dependency). */
function getBackendUrl(): string {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return '';
    return JSON.parse(raw).backend_url || '';
  } catch {
    return '';
  }
}

export function isRemoteMode(): boolean {
  return getBackendUrl() !== '';
}

export async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const backendUrl = getBackendUrl();

  // Remote mode: always use HTTP to the configured backend
  if (backendUrl) {
    return httpInvoke<T>(backendUrl, cmd, args);
  }

  // Local Tauri mode
  if (isTauri()) {
    const { invoke: tauriInvoke } = await import('@tauri-apps/api/core');
    return tauriInvoke<T>(cmd, args);
  }

  // Browser dev fallback
  return httpInvoke<T>(DEV_SERVER_URL, cmd, args);
}

async function httpInvoke<T>(baseUrl: string, cmd: string, args?: Record<string, unknown>): Promise<T> {
  const res = await fetch(`${baseUrl}/api/${cmd}`, {
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
