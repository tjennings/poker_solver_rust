import { useState } from 'react';
import Explorer from './Explorer';
import Simulator from './Simulator';
import Train from './Train';
import Settings from './Settings';

type View = 'explore' | 'train' | 'arena' | 'settings';

const VIEWS: { id: View; label: string; icon: JSX.Element; bottom?: boolean }[] = [
  {
    id: 'explore',
    label: 'Explore',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" />
      </svg>
    ),
  },
  {
    id: 'train',
    label: 'Train',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10 2v8L4.72 20.55a1 1 0 0 0 .93 1.45h12.7a1 1 0 0 0 .93-1.45L14 10V2" />
        <path d="M8.5 2h7" /><path d="M7 16h10" />
      </svg>
    ),
  },
  {
    id: 'arena',
    label: 'Arena',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14.5 17.5L3 6V3h3l11.5 11.5" />
        <path d="M13 19l6-6" /><path d="M16 16l4 4" /><path d="M19 21l2-2" />
        <path d="M9.5 6.5L21 18v3h-3L6.5 9.5" />
        <path d="M11 5l-6 6" /><path d="M8 8L4 4" /><path d="M5 3L3 5" />
      </svg>
    ),
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.32 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
      </svg>
    ),
    bottom: true,
  },
];

function App() {
  const [view, setView] = useState<View>('explore');

  return (
    <div className="app-shell">
      <nav className="rail">
        {VIEWS.filter(v => !v.bottom).map(v => (
          <button
            key={v.id}
            className={`rail-icon ${view === v.id ? 'active' : ''}`}
            onClick={() => setView(v.id)}
            title={v.label}
          >
            {v.icon}
          </button>
        ))}
        <div className="rail-spacer" />
        {VIEWS.filter(v => v.bottom).map(v => (
          <button
            key={v.id}
            className={`rail-icon ${view === v.id ? 'active' : ''}`}
            onClick={() => setView(v.id)}
            title={v.label}
          >
            {v.icon}
          </button>
        ))}
      </nav>
      <main className="main-pane">
        <div className="view-pane" style={{ display: view === 'explore' ? undefined : 'none' }}><Explorer /></div>
        <div className="view-pane" style={{ display: view === 'train' ? undefined : 'none' }}><Train /></div>
        <div className="view-pane" style={{ display: view === 'arena' ? undefined : 'none' }}><Simulator /></div>
        <div className="view-pane" style={{ display: view === 'settings' ? undefined : 'none' }}><Settings /></div>
      </main>
    </div>
  );
}

export default App;
