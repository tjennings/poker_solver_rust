import { isRemoteMode } from './invoke';

type UnlistenFn = () => void;

const STORAGE_KEY = 'global_config';

function getBackendUrl(): string {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return '';
    return JSON.parse(raw).backend_url || '';
  } catch {
    return '';
  }
}

/** Convert HTTP URL to WebSocket URL (http->ws, https->wss). */
function toWsUrl(httpUrl: string): string {
  return httpUrl.replace(/^http/, 'ws') + '/ws/events';
}

// Shared WebSocket singleton for remote mode
let sharedWs: WebSocket | null = null;
let wsListeners: Map<string, Set<(payload: unknown) => void>> = new Map();
let wsRefCount = 0;

function ensureWebSocket(): WebSocket {
  if (sharedWs && sharedWs.readyState === WebSocket.OPEN) return sharedWs;

  const url = toWsUrl(getBackendUrl());
  const ws = new WebSocket(url);

  ws.onmessage = (msg) => {
    try {
      const data = JSON.parse(msg.data);
      const listeners = wsListeners.get(data.event);
      if (listeners) {
        listeners.forEach(cb => cb(data.payload));
      }
    } catch {
      // ignore malformed messages
    }
  };

  ws.onclose = () => {
    sharedWs = null;
    // Reconnect if there are still active listeners
    if (wsRefCount > 0) {
      setTimeout(() => {
        if (wsRefCount > 0) ensureWebSocket();
      }, 1000);
    }
  };

  sharedWs = ws;
  return ws;
}

/**
 * Listen for events -- uses Tauri listen() locally, WebSocket remotely.
 * Returns an unlisten function.
 */
export async function listen<T>(
  event: string,
  handler: (payload: T) => void,
): Promise<UnlistenFn> {
  if (!isRemoteMode()) {
    // Local Tauri mode
    const { listen: tauriListen } = await import('@tauri-apps/api/event');
    return tauriListen<T>(event, (e) => handler(e.payload));
  }

  // Remote WebSocket mode
  const cb = handler as (payload: unknown) => void;
  if (!wsListeners.has(event)) {
    wsListeners.set(event, new Set());
  }
  wsListeners.get(event)!.add(cb);
  wsRefCount++;

  ensureWebSocket();

  return () => {
    const set = wsListeners.get(event);
    if (set) {
      set.delete(cb);
      if (set.size === 0) wsListeners.delete(event);
    }
    wsRefCount--;
    if (wsRefCount === 0 && sharedWs) {
      sharedWs.close();
      sharedWs = null;
    }
  };
}
