import { useState } from 'react';
import Explorer from './Explorer';
import Simulator from './Simulator';

type View = 'explorer' | 'simulator';

function App() {
  const [view, setView] = useState<View>('explorer');

  return (
    <div className="container">
      <div className="view-tabs">
        <button
          className={`view-tab ${view === 'explorer' ? 'active' : ''}`}
          onClick={() => setView('explorer')}
        >
          Explorer
        </button>
        <button
          className={`view-tab ${view === 'simulator' ? 'active' : ''}`}
          onClick={() => setView('simulator')}
        >
          Simulator
        </button>
      </div>
      {view === 'explorer' ? <Explorer /> : <Simulator />}
    </div>
  );
}

export default App;
