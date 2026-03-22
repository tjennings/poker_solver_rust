import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { invoke } from './invoke';
import {
  ActionInfo,
  BlueprintConfig,
  CacheInfo,
  MatrixCell,
  PostflopConfig,
  PostflopConfigSummary,
  PostflopStrategyMatrix,
  PostflopProgress,
  PostflopPlayResult,
  PostflopStreetResult,
  StrategyMatrix,
} from './types';
import { blueprintToPostflopMatrix } from './blueprint-utils';
import {
  SUIT_COLORS,
  SUIT_SYMBOLS,
  getActionColor,
  displayOrderIndices,
  formatActionLabel,
} from './matrix-utils';
import { ActionBlock, HandCell, CellDetail, HistoryItem } from './Explorer';

/** Convert postflop cell to shared MatrixCell format. */
function toMatrixCell(cell: { hand: string; suited: boolean; pair: boolean; probabilities: number[] }, actions: ActionInfo[]): MatrixCell {
  return {
    hand: cell.hand,
    suited: cell.suited,
    pair: cell.pair,
    probabilities: cell.probabilities.map((p, i) => ({
      action: actions[i]?.label ?? '',
      probability: p,
    })),
  };
}

interface PostflopExplorerProps {
  onBack: (preflopHistoryIndex?: number) => void;
  blueprintConfig?: BlueprintConfig;
  preflopHistory?: HistoryItem[];
}

export default function PostflopExplorer({ onBack, blueprintConfig, preflopHistory }: PostflopExplorerProps) {
  const [config, setConfig] = useState<PostflopConfig>(() => {
    const gc = JSON.parse(localStorage.getItem('global_config') || '{}');
    const flopThreshold = gc.flop_combo_threshold ?? 200;
    const turnThreshold = gc.turn_combo_threshold ?? 300;
    if (blueprintConfig) {
      return {
        oop_range: blueprintConfig.oop_range,
        ip_range: blueprintConfig.ip_range,
        pot: blueprintConfig.pot,
        effective_stack: blueprintConfig.effective_stack,
        oop_bet_sizes: blueprintConfig.oop_bet_sizes,
        oop_raise_sizes: blueprintConfig.oop_raise_sizes,
        ip_bet_sizes: blueprintConfig.ip_bet_sizes,
        ip_raise_sizes: blueprintConfig.ip_raise_sizes,
        rake_rate: blueprintConfig.rake_rate,
        rake_cap: blueprintConfig.rake_cap,
        flop_combo_threshold: flopThreshold,
        turn_combo_threshold: turnThreshold,
        abstract_node_idx: blueprintConfig.abstract_node_idx,
      };
    }
    return {
      oop_range: 'QQ+,AKs,AKo',
      ip_range: 'TT+,AQs+,AKo',
      pot: 30,
      effective_stack: 170,
      oop_bet_sizes: '25%,33%,75%,a',
      oop_raise_sizes: 'a',
      ip_bet_sizes: '25%,33%,75%,a',
      ip_raise_sizes: 'a',
      rake_rate: 0,
      rake_cap: 0,
      flop_combo_threshold: flopThreshold,
      turn_combo_threshold: turnThreshold,
    };
  });
  const [configSummary, setConfigSummary] = useState<PostflopConfigSummary | null>(null);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [configError, setConfigError] = useState<string | null>(null);
  const [boardInput, setBoardInput] = useState(
    blueprintConfig?.board ? blueprintConfig.board.join(' ') : ''
  );
  const [error, setError] = useState<string | null>(null);

  // Solve state
  const [matrix, setMatrix] = useState<PostflopStrategyMatrix | null>(null);
  const [solving, setSolving] = useState(false);
  const [progress, setProgress] = useState<PostflopProgress | null>(null);
  const [actionHistory, setActionHistory] = useState<{
    selectedId: string;
    actionIndex: number;
    position: string;
    stack: number;
    pot: number;
    actions: ActionInfo[];
    streetIndex: number; // which street solve this action belongs to
  }[]>([]);
  const [currentStreetIndex, setCurrentStreetIndex] = useState(0);
  const pollRef = useRef<number | null>(null);
  const initialExplRef = useRef<number>(Infinity);
  const pendingNavRef = useRef(false);
  const configAppliedRef = useRef(false);

  // Navigation state
  const [showFlopPicker, setShowFlopPicker] = useState(false);
  const [showNextCardPicker, setShowNextCardPicker] = useState(false);
  const [awaitingCard, setAwaitingCard] = useState(false);
  const [terminal, setTerminal] = useState(false);
  const [selectedCell, setSelectedCell] = useState<{row: number; col: number} | null>(null);
  const [needsSolve, setNeedsSolve] = useState(false);
  // Track actions per street for close_street call
  const [streetActions, setStreetActions] = useState<number[]>([]);
  // Cache integration
  const [cacheInfo, setCacheInfo] = useState<CacheInfo | null>(null);
  const [loadingCache, setLoadingCache] = useState(false);
  const blockedAtChanceRef = useRef(false);
  const [, setPriorStreetActions] = useState<number[][]>([]);

  // Blueprint navigation state
  const [blueprintMode, setBlueprintMode] = useState(false);
  const [blueprintHistory, setBlueprintHistory] = useState<string[]>([]);
  const [solved, setSolved] = useState(false);
  const [blueprintFetchTrigger, setBlueprintFetchTrigger] = useState(0);

  /** Extract preflop action IDs from preflopHistory for ExplorationPosition. */
  const preflopActionIds = useMemo(() => {
    if (!preflopHistory) return [];
    return preflopHistory
      .filter((item): item is Extract<HistoryItem, { type: 'action' }> => item.type === 'action')
      .map(item => item.selected);
  }, [preflopHistory]);

  const boardCards = useMemo(() => {
    const trimmed = boardInput.trim();
    if (!trimmed) return [];
    return trimmed.split(/\s+/);
  }, [boardInput]);

  useEffect(() => {
    // Guard against HMR re-mounts calling postflop_set_config again,
    // which clears the solved game from the backend.
    if (configAppliedRef.current) return;
    configAppliedRef.current = true;

    const autoApply = async () => {
      try {
        const summary = await invoke<PostflopConfigSummary>('postflop_set_config', { config });
        setConfigSummary(summary);
        if (blueprintConfig) {
          // Set blueprint weights as filtered weights for the solve.
          // Backend returns authoritative combo counts (sum of weights).
          const result = await invoke<{ oop_combos: number; ip_combos: number }>('postflop_set_filtered_weights', {
            oop_weights: blueprintConfig.oop_weights,
            ip_weights: blueprintConfig.ip_weights,
          });
          setConfigSummary(prev => prev ? { ...prev, oop_combos: result.oop_combos, ip_combos: result.ip_combos } : prev);
          // Set cache dir if the command exists
          await invoke('postflop_set_cache_dir', { dir: blueprintConfig.blueprint_dir }).catch(() => {
            // Command may not exist yet (Task 8)
          });
        }
      } catch (e) {
        setError(String(e));
      }
    };
    autoApply();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleConfigSubmit = useCallback(async (newConfig: PostflopConfig) => {
    setConfigError(null);
    try {
      const summary = await invoke<PostflopConfigSummary>('postflop_set_config', { config: newConfig });
      setConfig(newConfig);
      setConfigSummary(summary);
      setShowConfigModal(false);
    } catch (e) {
      setConfigError(String(e));
    }
  }, []);

  // Fetch blueprint strategy when board changes and config is set.
  // Only triggers on board length changes (new street) or initial config — NOT on blueprint navigation.
  // Blueprint action navigation is handled by handleBlueprintAction/handleBlueprintNavigateBack.
  useEffect(() => {
    if (!configSummary || boardCards.length < 3 || solving || solved) return;

    let cancelled = false;
    const fetchBlueprint = async () => {
      try {
        const fullHistory = [...preflopActionIds, ...blueprintHistory];
        const sm = await invoke<StrategyMatrix>('get_strategy_matrix', {
          position: {
            board: boardCards,
            history: fullHistory,
            pot: config.pot,
            stacks: [config.effective_stack, config.effective_stack],
            to_act: 0,
            num_players: 2,
            active_players: [true, true],
          },
        });
        if (cancelled) return;
        setMatrix(blueprintToPostflopMatrix(sm, boardCards, sm.to_act));
        setBlueprintMode(true);
        setNeedsSolve(true);
      } catch (e) {
        if (!cancelled) {
          setError(String(e));
          setNeedsSolve(true);
        }
      }
    };
    fetchBlueprint();
    return () => { cancelled = true; };
  }, [boardCards.length, configSummary, blueprintFetchTrigger]); // eslint-disable-line react-hooks/exhaustive-deps

  // Cleanup polling on unmount
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  // When solve finishes while blocked at a chance node, transition to card picker
  useEffect(() => {
    if (!solving && blockedAtChanceRef.current) {
      blockedAtChanceRef.current = false;
      setTerminal(false);
      setAwaitingCard(true);
    }
  }, [solving]);

  /** Start polling for solve progress (shared by initial solve and multi-street). */
  const startPolling = useCallback(() => {
    const poll = async () => {
      try {
        const p = await invoke<PostflopProgress>('postflop_get_progress', {});
        setProgress(p);
        if (p.exploitability < 1e30 && p.exploitability > initialExplRef.current) {
          initialExplRef.current = p.exploitability;
        }
        if (p.matrix && !pendingNavRef.current) setMatrix(p.matrix);
        if (p.is_complete) {
          setSolving(false);
          setSolved(true);
          if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
        }
      } catch (e) {
        setError(String(e));
        setSolving(false);
        if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
      }
    };
    poll();
    pollRef.current = window.setInterval(poll, 500);
  }, []);


  /** Reset everything for a new hand. */
  const handleReset = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    setMatrix(null);
    setSelectedCell(null);
    setNeedsSolve(false);
    setActionHistory([]);
    setStreetActions([]);
    setCurrentStreetIndex(0);
    setBoardInput('');
    setTerminal(false);
    setAwaitingCard(false);
    setProgress(null);
    setError(null);
    setSolving(false);
    setCacheInfo(null);
    setPriorStreetActions([]);
    setLoadingCache(false);
    setSolved(false);
    setBlueprintMode(false);
    setBlueprintHistory([]);
  }, []);

  /** Start or cancel solve for the current board. */
  const handleSolve = useCallback(() => {
    if (solving) {
      // Cancel: tell backend to stop, then stop polling
      invoke('postflop_cancel_solve').catch(() => {});
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
      setSolving(false);
      setNeedsSolve(true);
      setBlueprintMode(true); // Back to blueprint mode
      setBlueprintFetchTrigger(prev => prev + 1); // Trigger re-fetch of blueprint matrix
      return;
    }
    const cards = boardInput.trim().split(/\s+/);
    if (cards.length < 3) return;
    setError(null);
    setSolving(true);
    setNeedsSolve(false);
    setBlueprintMode(false); // Switch to solver mode
    setActionHistory([]); // Reset action history for solver tree
    setBlueprintHistory([]);
    setMatrix(null);
    setProgress(null);
    initialExplRef.current = Infinity;
    const globalConfig = JSON.parse(localStorage.getItem('global_config') || '{}');
    const targetExpl = globalConfig.stub_range_solver ? 1e9 : (globalConfig.target_exploitability ?? 3.0);
    const maxIters = globalConfig.solve_iterations ?? 200;
    const biasFactor = globalConfig.rollout_bias_factor ?? 10.0;
    const numRollouts = globalConfig.rollout_num_samples ?? 3;
    const oppSamples = globalConfig.rollout_opponent_samples ?? 8;
    const leafEvalInterval = globalConfig.leaf_eval_interval ?? 10;
    const rangeClamp = globalConfig.range_clamp_threshold ?? 0.05;
    invoke('postflop_solve_street', {
      board: cards,
      target_exploitability: targetExpl,
      max_iterations: maxIters,
      rollout_bias_factor: biasFactor,
      rollout_num_samples: numRollouts,
      rollout_opponent_samples: oppSamples,
      leaf_eval_interval: leafEvalInterval,
      range_clamp_threshold: rangeClamp,
    })
      .then(() => startPolling())
      .catch((e) => { setError(String(e)); setSolving(false); });
  }, [solving, boardInput, startPolling]);

  /** Navigate the tree by clicking an action button (works during solving). */
  const handleAction = useCallback(async (actionIndex: number) => {
    setError(null);
    setSelectedCell(null);

    // Optimistically add the action card and clear the matrix immediately
    if (matrix) {
      setActionHistory(prev => [...prev, {
        selectedId: String(actionIndex),
        actionIndex,
        position: matrix.player === 0 ? 'SB' : 'BB',
        stack: matrix.stacks[matrix.player],
        pot: matrix.pot,
        actions: matrix.actions,
        streetIndex: currentStreetIndex,
      }]);
    }
    setStreetActions(prev => [...prev, actionIndex]);
    setMatrix(null);
    pendingNavRef.current = true;

    try {
      const result = await invoke<PostflopPlayResult>('postflop_play_action', { action: actionIndex });
      pendingNavRef.current = false;

      if (result.is_terminal) {
        setTerminal(true);
        return;
      }

      if (result.is_chance) {
        if (solving) {
          blockedAtChanceRef.current = true;
          setTerminal(true);
          setAwaitingCard(false);
        } else {
          setAwaitingCard(true);
        }
        return;
      }

      if (result.matrix) {
        setMatrix(result.matrix);
      }
    } catch (e) {
      // Rollback optimistic update on error
      pendingNavRef.current = false;
      setActionHistory(prev => prev.slice(0, -1));
      setStreetActions(prev => prev.slice(0, -1));
      setError(String(e));
    }
  }, [solving, matrix, currentStreetIndex]);

  /** Navigate back to a previous point by clicking a history action card. */
  const handleNavigateBack = useCallback(async (historyIndex: number) => {
    const entry = actionHistory[historyIndex];
    if (!entry) return;

    setError(null);
    setSelectedCell(null);
    setTerminal(false);
    setAwaitingCard(false);
    blockedAtChanceRef.current = false;

    // Optimistically truncate history and clear matrix
    const replayActions = actionHistory
      .slice(0, historyIndex)
      .filter(e => e.streetIndex === entry.streetIndex)
      .map(e => e.actionIndex);
    setActionHistory(prev => prev.slice(0, historyIndex));
    setStreetActions(replayActions);
    setMatrix(null);
    pendingNavRef.current = true;

    try {
      const result = await invoke<PostflopPlayResult>('postflop_navigate_to', { history: replayActions });
      pendingNavRef.current = false;

      if (result.is_terminal) {
        setTerminal(true);
        return;
      }

      if (result.is_chance) {
        if (solving) {
          blockedAtChanceRef.current = true;
          setTerminal(true);
        } else {
          setAwaitingCard(true);
        }
        return;
      }

      if (result.matrix) {
        setMatrix(result.matrix);
      }
    } catch (e) {
      pendingNavRef.current = false;
      setError(String(e));
    }
  }, [solving, actionHistory, currentStreetIndex]);

  /** Navigate the blueprint tree by clicking an action button. */
  const handleBlueprintAction = useCallback(async (actionId: string) => {
    setError(null);
    setSelectedCell(null);

    // Find the action index for the action card display
    const actionIndex = matrix?.actions.findIndex(a => a.id === actionId) ?? 0;

    // Optimistically add action card
    if (matrix) {
      setActionHistory(prev => [...prev, {
        selectedId: actionId,
        actionIndex,
        position: matrix.player === 0 ? 'SB' : 'BB',
        stack: matrix.stacks[matrix.player],
        pot: matrix.pot,
        actions: matrix.actions,
        streetIndex: currentStreetIndex,
      }]);
    }
    setMatrix(null);

    const newHistory = [...blueprintHistory, actionId];
    setBlueprintHistory(newHistory);

    try {
      const fullHistory = [...preflopActionIds, ...newHistory];
      const sm = await invoke<StrategyMatrix>('get_strategy_matrix', {
        position: {
          board: boardCards,
          history: fullHistory,
          pot: config.pot,
          stacks: [config.effective_stack, config.effective_stack],
          to_act: 0,
          num_players: 2,
          active_players: [true, true],
        },
      });

      // Detect street transition: tree advanced past a chance node
      const expectedStreet = boardCards.length === 3 ? 'Flop' : boardCards.length === 4 ? 'Turn' : 'River';
      if (sm.street !== expectedStreet) {
        // Need next card before we can show the matrix
        setAwaitingCard(true);
        return;
      }

      setMatrix(blueprintToPostflopMatrix(sm, boardCards, sm.to_act));
    } catch (e) {
      if (String(e).includes('terminal')) {
        setTerminal(true);
      } else {
        setBlueprintHistory(prev => prev.slice(0, -1));
        setActionHistory(prev => prev.slice(0, -1));
        setError(String(e));
      }
    }
  }, [blueprintHistory, matrix, boardCards, config, currentStreetIndex, preflopActionIds]);

  /** Navigate back in blueprint mode by clicking a history action card. */
  const handleBlueprintNavigateBack = useCallback(async (historyIndex: number) => {
    setError(null);
    setSelectedCell(null);
    setTerminal(false);
    setAwaitingCard(false);

    // Truncate history
    setActionHistory(prev => prev.slice(0, historyIndex));
    const newHistory = blueprintHistory.slice(0, historyIndex);
    setBlueprintHistory(newHistory);
    setMatrix(null);

    try {
      const fullHistory = [...preflopActionIds, ...newHistory];
      const sm = await invoke<StrategyMatrix>('get_strategy_matrix', {
        position: {
          board: boardCards,
          history: fullHistory,
          pot: config.pot,
          stacks: [config.effective_stack, config.effective_stack],
          to_act: 0,
          num_players: 2,
          active_players: [true, true],
        },
      });
      setMatrix(blueprintToPostflopMatrix(sm, boardCards, sm.to_act));
    } catch (e) {
      setError(String(e));
    }
  }, [blueprintHistory, boardCards, config, preflopActionIds]);

  return (
    <div className="explorer-root">
      {error && <div className="error">{error}</div>}

      <div className="action-strip">
        {/* Split switcher: load dataset / postflop solver */}
        <div className="dataset-switcher-split">
          <div className="dataset-switcher-half" onClick={() => onBack()} title="Load Strategy">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
              <line x1="12" y1="11" x2="12" y2="17" />
              <line x1="9" y1="14" x2="15" y2="14" />
            </svg>
          </div>
          <div className="dataset-switcher-half active" title="Postflop Solver">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="2" y="2" width="20" height="20" rx="2" />
              <path d="M7 12h10M12 7v10" />
            </svg>
          </div>
        </div>

        {/* Preflop action history (from blueprint explorer) */}
        {preflopHistory && preflopHistory.map((item, i) =>
          item.type === 'action' ? (
            <ActionBlock
              key={`pf-${i}`}
              position={item.position}
              stack={item.stack}
              pot={item.pot}
              actions={item.actions}
              selectedAction={item.selected}
              onSelect={() => onBack(i)}
              onHeaderClick={() => onBack(i)}
              isCurrent={false}
            />
          ) : null
        )}

        {/* Config card */}
        <div className="action-block postflop-config-card"
          onClick={blueprintConfig ? undefined : () => setShowConfigModal(true)}
          style={blueprintConfig ? { cursor: 'default' } : undefined}
        >
          <div className="postflop-config-label">{blueprintConfig ? 'Blueprint' : 'Config'}</div>
          <div className="postflop-config-summary">
            {+(config.pot / 2).toFixed(1)} pot / {+(config.effective_stack / 2).toFixed(1)} eff
            {config.rake_rate > 0 && ` / ${(config.rake_rate * 100).toFixed(1)}% rake`}
          </div>
          {configSummary && (
            <div className="postflop-config-combos">
              SB: {configSummary.oop_combos} &middot; BB: {configSummary.ip_combos}
            </div>
          )}
        </div>

        {/* Flop street block */}
        <div
          className={`street-block ${boardCards.length === 3 ? '' : 'pending'}`}
          onClick={() => { if (!solving) setShowFlopPicker(true); }}
          style={{ cursor: solving ? 'default' : 'pointer' }}
        >
          <div className="street-block-header">
            <span className="street-name">FLOP</span>
          </div>
          <div className="street-cards">
            {[0, 1, 2].map((i) => {
              const card = boardCards[i];
              if (!card) return <div key={i} className="street-card empty"><span>?</span></div>;
              const rank = card[0]?.toUpperCase();
              const suit = card[1]?.toLowerCase();
              return (
                <div key={i} className="street-card" style={{ backgroundColor: SUIT_COLORS[suit] || '#333' }}>
                  <span className="card-rank">{rank}</span>
                  <span className="card-suit">{SUIT_SYMBOLS[suit] || '?'}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Action history interleaved with street cards */}
        {(() => {
          const renderStreetCard = (idx: number, label: string) => {
            const card = boardCards[idx];
            if (!card) return null;
            const rank = card[0]?.toUpperCase();
            const suit = card[1]?.toLowerCase();
            return (
              <div key={`street-${label}`} className="street-block">
                <div className="street-block-header">
                  <span className="street-name">{label}</span>
                </div>
                <div className="street-cards">
                  <div className="street-card" style={{ backgroundColor: SUIT_COLORS[suit] || '#333' }}>
                    <span className="card-rank">{rank}</span>
                    <span className="card-suit">{SUIT_SYMBOLS[suit] || '?'}</span>
                  </div>
                </div>
              </div>
            );
          };

          const elems: React.ReactNode[] = [];

          const navigateBack = (i: number) => {
            if (blueprintMode) {
              handleBlueprintNavigateBack(i);
            } else {
              handleNavigateBack(i);
            }
          };

          // Flop (street 0) actions
          actionHistory.forEach((item, i) => {
            if (item.streetIndex !== 0) return;
            elems.push(
              <ActionBlock
                key={`ah-${i}`}
                position={item.position}
                stack={item.stack}
                pot={item.pot}
                actions={item.actions}
                selectedAction={item.selectedId}
                onSelect={() => navigateBack(i)}
                onHeaderClick={() => navigateBack(i)}
                isCurrent={false}
              />
            );
          });

          // Turn card + turn actions
          if (boardCards.length >= 4) {
            elems.push(renderStreetCard(3, 'TURN'));
            actionHistory.forEach((item, i) => {
              if (item.streetIndex !== 1) return;
              elems.push(
                <ActionBlock
                  key={`ah-${i}`}
                  position={item.position}
                  stack={item.stack}
                  pot={item.pot}
                  actions={item.actions}
                  selectedAction={item.selectedId}
                  onSelect={() => navigateBack(i)}
                  onHeaderClick={() => navigateBack(i)}
                  isCurrent={false}
                />
              );
            });
          }

          // River card + river actions
          if (boardCards.length >= 5) {
            elems.push(renderStreetCard(4, 'RIVER'));
            actionHistory.forEach((item, i) => {
              if (item.streetIndex !== 2) return;
              elems.push(
                <ActionBlock
                  key={`ah-${i}`}
                  position={item.position}
                  stack={item.stack}
                  pot={item.pot}
                  actions={item.actions}
                  selectedAction={item.selectedId}
                  onSelect={() => navigateBack(i)}
                  onHeaderClick={() => navigateBack(i)}
                  isCurrent={false}
                />
              );
            });
          }

          return elems;
        })()}

        {/* Available actions */}
        {matrix && !terminal && !awaitingCard && (
          <ActionBlock
            position={matrix.player === 0 ? 'SB' : 'BB'}
            stack={matrix.stacks[matrix.player]}
            pot={matrix.pot}
            actions={matrix.actions}
            onSelect={(actionId) => {
              if (blueprintMode) {
                handleBlueprintAction(actionId);
              } else {
                handleAction(Number(actionId));
              }
            }}
            isCurrent={true}
          />
        )}

        {/* Next street card picker */}
        {awaitingCard && (
          <div className="street-block pending" style={{ cursor: 'pointer' }}
            onClick={() => setShowNextCardPicker(true)}>
            <div className="street-block-header">
              <span className="street-name">{boardCards.length === 3 ? 'TURN' : boardCards.length === 4 ? 'RIVER' : '?'}</span>
            </div>
            <div className="street-cards">
              <div className="street-card empty"><span>?</span></div>
            </div>
          </div>
        )}

        {/* Solve button — always rightmost in strip */}
        {(needsSolve || solving || solved) && (
          <div
            className={`action-block solve-block ${solving ? 'solving' : ''} ${solved ? 'solved' : ''}`}
            onClick={solved ? undefined : handleSolve}
            style={solved ? { cursor: 'default' } : undefined}
          >
            <span className="solve-label">
              {solved ? 'SOLVED' : solving ? 'CANCEL' : 'SOLVE'}
            </span>
          </div>
        )}
      </div>

      {/* Flop card picker modal */}
      {showFlopPicker && (
        <div className="modal-overlay" onClick={() => setShowFlopPicker(false)}>
          <div className="card-picker-container" onClick={(e) => e.stopPropagation()}>
            <p className="card-picker-prompt">Select 3 flop cards</p>
            <FlopPicker
              deadCards={[]}
              onConfirm={(cards) => {
                setBoardInput(cards.join(' '));
                setShowFlopPicker(false);
                setActionHistory([]);
                setStreetActions([]);
                setPriorStreetActions([]);
                setMatrix(null);
                setProgress(null);
                setTerminal(false);
                setAwaitingCard(false);
                setCacheInfo(null);
                setSolved(false);
                setBlueprintMode(false);
                setBlueprintHistory([]);
                // Blueprint fetch will be triggered by boardCards.length change
              }}
            />
          </div>
        </div>
      )}

      {/* Next card picker modal (turn/river) */}
      {showNextCardPicker && awaitingCard && (
        <div className="modal-overlay" onClick={() => setShowNextCardPicker(false)}>
          <div className="card-picker-container" onClick={(e) => e.stopPropagation()}>
          <p className="card-picker-prompt">Select {boardCards.length === 3 ? 'turn' : 'river'} card</p>
          <NextCardPicker
            deadCards={boardCards}
            onConfirm={(card) => {
              setShowNextCardPicker(false);
              setAwaitingCard(false);
              setError(null);

              if (solved) {
                // Solved mode: use existing close_street with solver ranges
                invoke<PostflopStreetResult>('postflop_close_street', { action_history: streetActions })
                  .then(() => {
                    const newBoard = [...boardCards, card];
                    setBoardInput(newBoard.join(' '));
                    setPriorStreetActions(prev => [...prev, streetActions]);
                    setStreetActions([]);
                    setCurrentStreetIndex(prev => prev + 1);
                    setMatrix(null);
                    setProgress(null);
                    setSolved(false); // New street, not yet solved
                    setNeedsSolve(true);
                    setBlueprintMode(true); // Back to blueprint for new street
                  })
                  .catch((e) => { setError(String(e)); });
              } else {
                // Blueprint mode: update board and re-fetch blueprint strategy
                const newBoard = [...boardCards, card];
                setBoardInput(newBoard.join(' '));
                setCurrentStreetIndex(prev => prev + 1);
                setMatrix(null);
                // blueprintHistory continues — the tree already advanced past the chance node
                // The useEffect for boardCards change will re-fetch with the new board
              }
            }}
          />
          </div>
        </div>
      )}

      {/* Progress bar */}
      {solving && progress && (() => {
        const expl = progress.exploitability;
        const globalCfg = JSON.parse(localStorage.getItem('global_config') || '{}');
        const target = globalCfg.target_exploitability ?? 3.0;
        const valid = expl < 1e30;
        // Progress toward target: start from first reading, converge to target
        const startExpl = initialExplRef.current;
        let pct = 0;
        if (valid && startExpl > target) {
          pct = Math.min(1, Math.max(0, (startExpl - expl) / (startExpl - target)));
        }
        return (
          <div className="progress-bar-container">
            <div className="progress-bar-track">
              <div className="progress-bar-fill" style={{ width: `${pct * 100}%` }} />
              <span className="progress-text">
                {progress.iteration}/{progress.max_iterations} iters
                {valid && ` — ${(expl / Math.max(config.pot, 1) * 100).toFixed(1)}% pot expl`}
                {progress.elapsed_secs > 0 && ` — ${progress.elapsed_secs.toFixed(1)}s`}
                {` — ${progress.solver_name}`}
              </span>
            </div>
          </div>
        );
      })()}



      {/* Cache info */}
      {cacheInfo && matrix && !solving && (
        <div className="cache-info">
          Cached ({cacheInfo.iterations} iterations, {cacheInfo.exploitability.toFixed(1)}% pot)
          <button className="re-solve-btn" onClick={() => { setCacheInfo(null); setNeedsSolve(true); setMatrix(null); }}>
            Re-solve
          </button>
        </div>
      )}

      {/* Loading cache indicator */}
      {loadingCache && !matrix && !solving && (
        <div style={{ padding: 8, textAlign: 'center', fontSize: '0.8em', color: '#94a3b8' }}>
          Checking cache...
        </div>
      )}

      {/* Strategy matrix */}
      {matrix && (() => {
        return (
          <div className="matrix-container">
            <div className="matrix-with-detail">
              <div className="hand-matrix">
                {matrix.cells.map((row, rowIdx) => (
                  <div key={rowIdx} className="matrix-row">
                    {row.map((cell, colIdx) => (
                      <HandCell
                        key={colIdx}
                        cell={toMatrixCell(cell, matrix.actions)}
                        actions={matrix.actions}
                        reachWeight={cell.weight ?? (cell.combo_count > 0 ? 1 : 0)}
                        isSelected={selectedCell?.row === rowIdx && selectedCell?.col === colIdx}
                        onClick={() => setSelectedCell({ row: rowIdx, col: colIdx })}
                        overlayText={cell.ev != null && cell.combo_count > 0 ? cell.ev.toFixed(1) : undefined}
                      />
                    ))}
                  </div>
                ))}
              </div>
              <div className="detail-column">
                {selectedCell && matrix.cells[selectedCell.row]?.[selectedCell.col] && (<>
                  <CellDetail
                    cell={toMatrixCell(matrix.cells[selectedCell.row][selectedCell.col], matrix.actions)}
                    actions={matrix.actions}
                  />
                  <ComboBreakdown
                    combos={matrix.cells[selectedCell.row][selectedCell.col].combos}
                    actions={matrix.actions}
                  />
                </>)}
              </div>
            </div>
          </div>
        );
      })()}

      {/* Terminal state */}
      {terminal && (
        <div style={{ padding: 24, textAlign: 'center', opacity: 0.6 }}>
          Hand complete.{' '}
          <button className="action-button" onClick={handleReset}>New Hand</button>
        </div>
      )}

      {/* Awaiting card (no matrix visible) */}
      {awaitingCard && !matrix && !solving && (
        <div style={{ padding: 24, textAlign: 'center', opacity: 0.6 }}>
          Enter the next community card above to continue.
        </div>
      )}

      {/* Config Modal */}
      {showConfigModal && (
        <ConfigModal
          config={config}
          error={configError}
          onSubmit={handleConfigSubmit}
          onClose={() => setShowConfigModal(false)}
        />
      )}
    </div>
  );
}

const SUIT_ICON_COLORS: Record<string, string> = {
  s: '#ccc', h: '#dc2626', d: '#2563eb', c: '#16a34a',
};

function ComboBreakdown({ combos, actions }: {
  combos: { cards: string; probabilities: number[] }[];
  actions: ActionInfo[];
}) {
  if (!combos || combos.length === 0) return null;

  return (
    <div className="combo-breakdown">
      <div className="combo-breakdown-grid">
        {combos.map((combo) => {
          // Parse 4-char cards string e.g. "AsTh" → [A,s,T,h]
          const r1 = combo.cards[0], s1 = combo.cards[1];
          const r2 = combo.cards[2], s2 = combo.cards[3];
          return (
            <div key={combo.cards} className="combo-tile">
              <div className="combo-tile-header">
                <span className="combo-card" style={{ color: SUIT_ICON_COLORS[s1] || '#ccc' }}>
                  {r1}<span className="combo-suit">{SUIT_SYMBOLS[s1]}</span>
                </span>
                <span className="combo-card" style={{ color: SUIT_ICON_COLORS[s2] || '#ccc' }}>
                  {r2}<span className="combo-suit">{SUIT_SYMBOLS[s2]}</span>
                </span>
              </div>
              <div className="combo-tile-bar">
                {[...displayOrderIndices(actions)].reverse().map((idx) => {
                  const prob = combo.probabilities[idx];
                  if (!prob || prob <= 0) return null;
                  return (
                    <div
                      key={idx}
                      style={{
                        width: `${prob * 100}%`,
                        backgroundColor: getActionColor(actions[idx], actions),
                        height: '100%',
                      }}
                    />
                  );
                })}
              </div>
              <div className="combo-tile-actions">
                {displayOrderIndices(actions).map((idx) => {
                  const prob = combo.probabilities[idx];
                  const pct = (prob ?? 0) * 100;
                  if (pct < 0.1) return null;
                  return (
                    <div key={idx} className="combo-tile-row">
                      <span className="combo-tile-label">{formatActionLabel(actions[idx])}</span>
                      <span className="combo-tile-pct">{pct.toFixed(0)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ConfigModal({ config, error, onSubmit, onClose }: {
  config: PostflopConfig;
  error: string | null;
  onSubmit: (config: PostflopConfig) => void;
  onClose: () => void;
}) {
  const [draft, setDraft] = useState<PostflopConfig>({ ...config });

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h3>Game Configuration</h3>

        <label>SB Range</label>
        <textarea value={draft.oop_range}
          onChange={(e) => setDraft({ ...draft, oop_range: e.target.value })} rows={2} />

        <label>BB Range</label>
        <textarea value={draft.ip_range}
          onChange={(e) => setDraft({ ...draft, ip_range: e.target.value })} rows={2} />

        <div className="modal-row">
          <div>
            <label>Pot</label>
            <input type="text" value={draft.pot}
              onChange={(e) => setDraft({ ...draft, pot: parseInt(e.target.value) || 0 })} />
          </div>
          <div>
            <label>Effective Stack</label>
            <input type="text" value={draft.effective_stack}
              onChange={(e) => setDraft({ ...draft, effective_stack: parseInt(e.target.value) || 0 })} />
          </div>
        </div>

        <div className="modal-row">
          <div>
            <label>Rake Rate (%)</label>
            <input type="text"
              value={draft.rake_rate * 100}
              onChange={(e) => setDraft({ ...draft, rake_rate: parseFloat(e.target.value) / 100 || 0 })} />
          </div>
          <div>
            <label>Rake Cap</label>
            <input type="text"
              value={draft.rake_cap}
              onChange={(e) => setDraft({ ...draft, rake_cap: parseFloat(e.target.value) || 0 })} />
          </div>
        </div>

        <h4>Solver Dispatch</h4>
        <div className="modal-row">
          <div>
            <label>Flop Combo Threshold</label>
            <input type="text"
              value={draft.flop_combo_threshold}
              onChange={(e) => setDraft({ ...draft, flop_combo_threshold: parseInt(e.target.value) || 0 })} />
          </div>
          <div>
            <label>Turn Combo Threshold</label>
            <input type="text"
              value={draft.turn_combo_threshold}
              onChange={(e) => setDraft({ ...draft, turn_combo_threshold: parseInt(e.target.value) || 0 })} />
          </div>
        </div>

        <h4>Bet Sizes</h4>
        <div className="modal-row">
          <div>
            <label>SB Bet Sizes</label>
            <input value={draft.oop_bet_sizes}
              onChange={(e) => setDraft({ ...draft, oop_bet_sizes: e.target.value })} />
          </div>
          <div>
            <label>BB Bet Sizes</label>
            <input value={draft.ip_bet_sizes}
              onChange={(e) => setDraft({ ...draft, ip_bet_sizes: e.target.value })} />
          </div>
        </div>

        {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}

        <div className="modal-buttons">
          <button onClick={onClose}>Cancel</button>
          <button className="modal-primary" onClick={() => onSubmit({ ...draft, oop_raise_sizes: 'a', ip_raise_sizes: 'a' })}>Apply</button>
        </div>
      </div>
    </div>
  );
}

const PICKER_RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'];
const PICKER_SUITS = ['s', 'h', 'd', 'c'];
const PICKER_COLORS: Record<string, string> = { s: '#fff', h: '#dc2626', d: '#2563eb', c: '#16a34a' };

function FlopPicker({ deadCards, onConfirm }: { deadCards: string[]; onConfirm: (cards: string[]) => void }) {
  const [selected, setSelected] = useState<string[]>([]);
  const deadSet = useMemo(() => new Set(deadCards.map((c) => c.toLowerCase())), [deadCards]);
  const confirmedRef = useRef(false);

  const handleCardClick = (card: string) => {
    if (deadSet.has(card.toLowerCase())) return;
    setSelected((prev) => {
      if (prev.includes(card)) return prev.filter((c) => c !== card);
      if (prev.length >= 3) return prev;
      const next = [...prev, card];
      if (next.length === 3 && !confirmedRef.current) {
        confirmedRef.current = true;
        setTimeout(() => onConfirm(next), 0);
      }
      return next;
    });
  };

  return (
    <div className="card-picker">
      {PICKER_SUITS.map((suit) => (
        <div key={suit} className="card-picker-row">
          {PICKER_RANKS.map((rank) => {
            const card = `${rank}${suit}`;
            const isDead = deadSet.has(card.toLowerCase());
            const isSelected = selected.includes(card);
            return (
              <button
                key={card}
                className={`card-picker-card ${isDead ? 'dead' : ''} ${isSelected ? 'selected' : ''}`}
                style={{ color: PICKER_COLORS[suit] || '#eee', borderColor: isSelected ? '#00d9ff' : 'transparent' }}
                disabled={isDead}
                onClick={() => handleCardClick(card)}
              >
                <span className="picker-rank">{rank}</span>
                <span className="picker-suit">{SUIT_SYMBOLS[suit]}</span>
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function NextCardPicker({ deadCards, onConfirm }: { deadCards: string[]; onConfirm: (card: string) => void }) {
  const deadSet = useMemo(() => new Set(deadCards.map((c) => c.toLowerCase())), [deadCards]);

  return (
    <div className="card-picker">
      {PICKER_SUITS.map((suit) => (
        <div key={suit} className="card-picker-row">
          {PICKER_RANKS.map((rank) => {
            const card = `${rank}${suit}`;
            const isDead = deadSet.has(card.toLowerCase());
            return (
              <button
                key={card}
                className={`card-picker-card ${isDead ? 'dead' : ''}`}
                style={{ color: PICKER_COLORS[suit] || '#eee' }}
                disabled={isDead}
                onClick={() => onConfirm(card)}
              >
                <span className="picker-rank">{rank}</span>
                <span className="picker-suit">{SUIT_SYMBOLS[suit]}</span>
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
}

