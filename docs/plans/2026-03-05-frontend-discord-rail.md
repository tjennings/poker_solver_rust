# Frontend Discord Rail Refactor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace tab-based navigation with a Discord-like left rail, modernize the theme, and simplify the Explore dataset loading flow.

**Architecture:** App.tsx becomes a shell with a 64px left rail and a right pane that renders the active view. Explorer loses its hamburger menu and empty state; instead, when no dataset is loaded, the action strip shows a single "Load Dataset" card. Two new stub views (Train, Settings) are created. CSS gets a full theme refresh with gradients and shadows.

**Tech Stack:** React 18, TypeScript, Vite, Tauri — no new dependencies.

---

### Task 1: Create stub views (Train, Settings)

**Files:**
- Create: `frontend/src/Train.tsx`
- Create: `frontend/src/Settings.tsx`

**Step 1: Create Train.tsx**

```tsx
export default function Train() {
  return (
    <div className="stub-view">
      <div className="stub-icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#555" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M9 3L5 7l4 4" /><path d="M5 7h8a5 5 0 0 1 5 5v1" />
          <path d="M15 21l4-4-4-4" /><path d="M19 17h-8a5 5 0 0 1-5-5v-1" />
        </svg>
      </div>
      <p className="stub-title">Training Configuration</p>
      <p className="stub-subtitle">Coming soon</p>
    </div>
  );
}
```

**Step 2: Create Settings.tsx**

```tsx
export default function Settings() {
  return (
    <div className="stub-view">
      <div className="stub-icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#555" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.32 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
        </svg>
      </div>
      <p className="stub-title">Settings</p>
      <p className="stub-subtitle">Coming soon</p>
    </div>
  );
}
```

**Step 3: Commit**

```bash
git add frontend/src/Train.tsx frontend/src/Settings.tsx
git commit -m "feat(ui): add stub Train and Settings views"
```

---

### Task 2: Rewrite App.tsx with left rail navigation

**Files:**
- Modify: `frontend/src/App.tsx`

**Step 1: Replace the file contents with the rail layout**

The new App.tsx should:
- Define a `View` type: `'explore' | 'train' | 'arena' | 'settings'`
- Render a left rail (`<nav className="rail">`) with 4 icon buttons
- Render a right pane (`<main className="main-pane">`) that switches views
- Use inline SVG icons (compass, flask, swords, gear)
- Settings icon gets `className="rail-spacer"` (pushed to bottom via flex)

```tsx
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
        {view === 'explore' && <Explorer />}
        {view === 'train' && <Train />}
        {view === 'arena' && <Simulator />}
        {view === 'settings' && <Settings />}
      </main>
    </div>
  );
}

export default App;
```

**Step 2: Verify build compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: no type errors

**Step 3: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat(ui): replace tab bar with Discord-like left rail navigation"
```

---

### Task 3: Modify Explorer — remove hamburger, add Load Dataset action card

**Files:**
- Modify: `frontend/src/Explorer.tsx`

**Step 1: Remove HamburgerMenu component**

Delete the entire `HamburgerMenu` function component (lines 15-76 in current file).

**Step 2: Remove HamburgerMenu usage from Explorer render**

In the `Explorer` return JSX, delete:
```tsx
<HamburgerMenu
  agents={agents}
  activeAgentName={bundleInfo?.name ?? null}
  loading={loading}
  onSelectAgent={handleLoadAgent}
  onLoadBundle={handleLoadDataset}
/>
```

**Step 3: Replace the empty state with a Load Dataset action card in the action strip**

Replace the entire `{!bundleInfo && !loading && (...)}` empty state block (agent cards, or-divider, etc.) with an action strip containing a single "Load Dataset" card:

```tsx
{!bundleInfo && !loading && (
  <div className="action-strip">
    <div className="action-block load-dataset-card" onClick={handleLoadDataset}>
      <div className="load-dataset-icon">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
          <line x1="12" y1="11" x2="12" y2="17" />
          <line x1="9" y1="14" x2="15" y2="14" />
        </svg>
      </div>
      <span className="load-dataset-label">Load Dataset</span>
    </div>
  </div>
)}
```

**Step 4: Remove the `agents` state and `useEffect` that fetches agents**

Delete:
- `const [agents, setAgents] = useState<AgentInfo[]>([]);`
- The `useEffect` that calls `invoke<AgentInfo[]>('list_agents')`
- The `handleLoadAgent` callback
- Remove `AgentInfo` from the types import if no longer used

**Step 5: Verify build compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: no type errors

**Step 6: Commit**

```bash
git add frontend/src/Explorer.tsx
git commit -m "feat(ui): remove hamburger menu, add Load Dataset action card to Explore"
```

---

### Task 4: CSS theme overhaul

**Files:**
- Modify: `frontend/src/App.css`

This is the largest task. The full CSS replacement covers:

1. **Remove** old styles: `.container`, `.view-tabs`, `.view-tab`, all `.hamburger-*` classes, `.empty-state`, `.agent-cards`, `.agent-card`, `.or-divider`
2. **Add** new layout: `.app-shell`, `.rail`, `.rail-icon`, `.rail-spacer`, `.main-pane`
3. **Add** stub view styles: `.stub-view`, `.stub-icon`, `.stub-title`, `.stub-subtitle`
4. **Add** load dataset card: `.load-dataset-card`, `.load-dataset-icon`, `.load-dataset-label`
5. **Update** theme: body gradient, card/panel gradient backgrounds with box shadows, border updates

**Step 1: Write the complete new CSS**

Replace `frontend/src/App.css` entirely. Key sections:

**Global / body:**
```css
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
  color: #eee;
  min-height: 100vh;
}
```

**App shell layout:**
```css
.app-shell {
  display: flex;
  height: 100vh;
  overflow: hidden;
}

.rail {
  width: 64px;
  flex-shrink: 0;
  background: #0a0a18;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 12px 0;
  gap: 4px;
  box-shadow: 2px 0 12px rgba(0, 0, 0, 0.4);
  z-index: 10;
}

.rail-icon {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: none;
  color: #555;
  cursor: pointer;
  transition: all 0.15s;
  position: relative;
  padding: 0;
}

.rail-icon:hover {
  color: #ccc;
  background: rgba(255, 255, 255, 0.05);
}

.rail-icon.active {
  color: #00d9ff;
  background: rgba(0, 217, 255, 0.1);
}

.rail-icon.active::before {
  content: '';
  position: absolute;
  left: -12px;
  top: 50%;
  transform: translateY(-50%);
  width: 3px;
  height: 20px;
  background: #00d9ff;
  border-radius: 0 3px 3px 0;
}

.rail-spacer {
  flex: 1;
}

.main-pane {
  flex: 1;
  overflow-y: auto;
  padding: 1.5rem 2rem;
}
```

**Updated card/panel styles** — every `.bundle-info`, `.action-strip`, `.matrix-container`, `.sim-config`, `.sim-results`, `.cell-detail`, `.combo-panel`, `.hand-equity-panel`, `.card-picker-container`, `.hand-complete` gets:
```css
background: linear-gradient(135deg, #14142a 0%, #1a1a3a 100%);
border: 1px solid rgba(255, 255, 255, 0.06);
box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
```

Replace the old flat `background: #16213e` with the gradient on these elements.

**Stub views:**
```css
.stub-view {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 60vh;
  color: #555;
}

.stub-icon { margin-bottom: 1rem; }
.stub-title { font-size: 1.1rem; color: #888; margin: 0 0 0.25rem; }
.stub-subtitle { font-size: 0.85rem; color: #555; margin: 0; }
```

**Load dataset card:**
```css
.load-dataset-card {
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  min-height: 80px;
  border: 2px dashed rgba(255, 255, 255, 0.1);
  transition: all 0.15s;
}

.load-dataset-card:hover {
  border-color: #00d9ff;
  background: rgba(0, 217, 255, 0.05);
}

.load-dataset-icon { color: #555; }
.load-dataset-card:hover .load-dataset-icon { color: #00d9ff; }
.load-dataset-label { font-size: 0.85rem; color: #888; }
.load-dataset-card:hover .load-dataset-label { color: #00d9ff; }
```

**Step 2: Remove dead CSS**

Delete all of:
- `.container`
- `.view-tabs`, `.view-tab`
- `.hamburger-menu`, `.hamburger-button`, `.hamburger-icon`, `.hamburger-dropdown`
- `.menu-section`, `.menu-section-label`, `.menu-item`
- `.empty-state`, `.agent-cards`, `.agent-card`, `.or-divider`

**Step 3: Verify the app builds**

Run: `cd frontend && npm run build`
Expected: build succeeds

**Step 4: Commit**

```bash
git add frontend/src/App.css
git commit -m "feat(ui): modernize theme with gradients, shadows, Discord rail layout"
```

---

### Task 5: Visual verification and polish

**Step 1: Start the dev server and verify layout**

Run: `cd frontend && npm run dev`

Check:
- Left rail visible with 4 icons
- Compass (Explore) active by default with cyan pill
- Clicking each icon switches the right pane
- Train and Settings show stub messages
- Explore shows the Load Dataset card in the action strip when no dataset loaded
- After loading a dataset, normal explorer behavior works (action strip, matrix, card picker)
- Arena shows the Simulator unchanged

**Step 2: Fix any visual issues found during verification**

Adjust spacing, shadows, or colors as needed.

**Step 3: Final commit**

```bash
git add -A
git commit -m "fix(ui): polish Discord rail layout and theme"
```

---

## Agent Team & Execution Order

Since this is purely frontend TypeScript/CSS work (no Rust):

| Task | Agent | Parallel? |
|-|-|-|
| Task 1: Stub views | general-purpose | Yes, with Task 2 |
| Task 2: App.tsx rail | general-purpose | Yes, with Task 1 |
| Task 3: Explorer refactor | general-purpose | After Tasks 1-2 (needs to verify imports compile) |
| Task 4: CSS overhaul | general-purpose | After Task 3 (needs final class names) |
| Task 5: Verification | general-purpose | After Task 4 |

Tasks 1 and 2 can run in parallel. Tasks 3-5 are sequential.
