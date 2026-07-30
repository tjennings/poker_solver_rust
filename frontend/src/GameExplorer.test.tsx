import { afterEach, describe, expect, it, vi } from "vitest";
import {
  CANCELLATION_POLL_ATTEMPTS,
  CANCELLATION_POLL_INTERVAL_MS,
  getComboActionRows,
  getBackendSolveGeneration,
  isUniversalMpLazyBundleName,
  isSolveStopped,
  requestSolveCancellation,
  shouldShowStrategyMatrix,
  supportsUniversalMpLazyExact,
} from "./GameExplorer";
import type { GameState } from "./game-types";

const matrix = { cells: [], actions: [] };
const solve = {
  iteration: 10,
  max_iterations: 100,
  exploitability: 1,
  elapsed_secs: 0.5,
  rollout_hands_per_sec: 0,
  solver_name: "range",
  is_complete: false,
};

function stateFor(
  street: string,
  board: string[],
  overrides: { is_terminal?: boolean; is_chance?: boolean } = {},
) {
  return {
    street,
    board,
    is_terminal: overrides.is_terminal ?? false,
    is_chance: overrides.is_chance ?? false,
  };
}

describe("Universal MP lazy Explorer capabilities", () => {
  it("identifies Universal MP lazy bundle names without changing HU names", () => {
    expect(isUniversalMpLazyBundleName("MP universal_mp_lazy (2-player)")).toBe(
      true,
    );
    expect(
      isUniversalMpLazyBundleName("bundle (2-player universal_mp_lazy)"),
    ).toBe(true);
    expect(isUniversalMpLazyBundleName("heads-up blueprint")).toBe(false);
    expect(isUniversalMpLazyBundleName(null)).toBe(false);
  });

  it("allows exact solves only at complete non-terminal postflop roots", () => {
    expect(
      supportsUniversalMpLazyExact(stateFor("Flop", ["As", "Kd", "2c"])),
    ).toBe(true);
    expect(
      supportsUniversalMpLazyExact(stateFor("Turn", ["As", "Kd", "2c", "7h"])),
    ).toBe(true);
    expect(
      supportsUniversalMpLazyExact(
        stateFor("River", ["As", "Kd", "2c", "7h", "9s"]),
      ),
    ).toBe(true);
    expect(
      supportsUniversalMpLazyExact(
        stateFor("Flop", ["As", "Kd"], { is_chance: true }),
      ),
    ).toBe(false);
    expect(supportsUniversalMpLazyExact(stateFor("Preflop", [], {}))).toBe(
      false,
    );
    expect(
      supportsUniversalMpLazyExact(
        stateFor("River", ["As", "Kd", "2c", "7h", "9s"], {
          is_terminal: true,
        }),
      ),
    ).toBe(false);
  });

  it("does not present a Blueprint matrix as an unsolved Exact result in MP-lazy mode", () => {
    const unsolvedFlop = {
      ...stateFor("Flop", ["As", "Kd", "2c"]),
      matrix,
      solve: null,
    };
    const solvedFlop = { ...unsolvedFlop, solve };
    const unsupportedRoot = {
      ...stateFor("Flop", ["As", "Kd"], { is_chance: true }),
      matrix,
      solve,
    };

    expect(
      shouldShowStrategyMatrix("universal_mp_lazy", "exact", unsolvedFlop),
    ).toBe(false);
    expect(
      shouldShowStrategyMatrix("universal_mp_lazy", "exact", solvedFlop),
    ).toBe(true);
    expect(
      shouldShowStrategyMatrix("universal_mp_lazy", "exact", unsupportedRoot),
    ).toBe(false);
    expect(
      shouldShowStrategyMatrix("universal_mp_lazy", "blueprint", unsolvedFlop),
    ).toBe(true);
    expect(shouldShowStrategyMatrix("hu", "exact", unsolvedFlop)).toBe(true);
  });
});

describe("combo action rows", () => {
  it("keeps every matrix action in order and renders zero probabilities", () => {
    const actions = [
      { id: "fold", label: "Fold", action_type: "fold" },
      { id: "call", label: "Call", action_type: "call" },
      { id: "raise", label: "Raise to 2.5", action_type: "raise" },
    ];

    const rows = getComboActionRows(actions, [0, 0.734, 0]);

    expect(rows.map((row) => row.action.id)).toEqual([
      "fold",
      "call",
      "raise",
    ]);
    expect(rows.map((row) => row.percentage)).toEqual([0, 73.4, 0]);
    expect(rows.map((row) => row.percentageLabel)).toEqual([
      "0.0%",
      "73.4%",
      "0.0%",
    ]);
    expect(
      getComboActionRows(actions, [0.5]).map((row) => row.percentage),
    ).toEqual([50, 0, 0]);
  });
});

describe("solve cancellation", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("passes the backend generation and polls until the solve stops", async () => {
    vi.useFakeTimers();
    const calls: Array<{
      command: string;
      args?: Record<string, unknown>;
    }> = [];
    const solvingState = {
      ...stateFor("River", ["As", "Kd", "2c", "7h", "9s"]),
      solve,
    } as unknown as GameState;
    const stoppedState = {
      ...solvingState,
      solve: { ...solve, is_complete: true },
    } as unknown as GameState;
    const states = [solvingState, stoppedState];
    const setState = vi.fn();
    const setError = vi.fn();
    const invokeCommand = vi.fn(
      (command: string, args?: Record<string, unknown>): Promise<unknown> => {
        calls.push({ command, args });
        return Promise.resolve(
          command === "game_cancel_solve" ? undefined : states.shift(),
        );
      },
    ) as unknown as typeof import("./invoke").invoke;

    requestSolveCancellation(
      "exact",
      4,
      17,
      (generation) => generation === 4,
      setState,
      setError,
      invokeCommand,
    );

    expect(calls).toEqual([
      { command: "game_cancel_solve", args: { mode: "exact", generation: 17 } },
    ]);

    await Promise.resolve();
    await Promise.resolve();
    expect(calls).toHaveLength(2);
    expect(calls[1]).toEqual({
      command: "game_get_state",
      args: { source: "exact" },
    });

    expect(setState).toHaveBeenCalledWith(solvingState);
    await vi.advanceTimersByTimeAsync(CANCELLATION_POLL_INTERVAL_MS);
    expect(setState).toHaveBeenCalledWith(stoppedState);
    expect(setError).not.toHaveBeenCalled();
  });

  it("bounds best-effort refreshes when the backend keeps reporting active", async () => {
    vi.useFakeTimers();
    const calls: string[] = [];
    const solvingState = {
      ...stateFor("River", ["As", "Kd", "2c", "7h", "9s"]),
      solve,
    } as unknown as GameState;
    const setState = vi.fn();
    const setError = vi.fn();
    const invokeCommand = vi.fn((command: string): Promise<unknown> => {
      calls.push(command);
      return Promise.resolve(command === "game_cancel_solve" ? undefined : solvingState);
    }) as unknown as typeof import("./invoke").invoke;

    requestSolveCancellation(
      "exact",
      4,
      17,
      (generation) => generation === 4,
      setState,
      setError,
      invokeCommand,
    );

    await Promise.resolve();
    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(
      CANCELLATION_POLL_INTERVAL_MS * CANCELLATION_POLL_ATTEMPTS,
    );

    expect(calls).toEqual([
      "game_cancel_solve",
      ...Array(CANCELLATION_POLL_ATTEMPTS).fill("game_get_state"),
    ]);
    expect(setState).toHaveBeenCalledTimes(CANCELLATION_POLL_ATTEMPTS);
    expect(setError).not.toHaveBeenCalled();
  });

  it("ignores a refresh that resolves after a newer solve owns the mode", async () => {
    const calls: string[] = [];
    let resolveRefresh!: (state: GameState) => void;
    const refresh = new Promise<GameState>((resolve) => {
      resolveRefresh = resolve;
    });
    const refreshedState = stateFor("River", []);
    const setState = vi.fn();
    const setError = vi.fn();
    const invokeCommand = vi.fn((command: string): Promise<unknown> => {
      calls.push(command);
      return command === "game_cancel_solve"
        ? Promise.resolve(undefined)
        : refresh;
    }) as unknown as typeof import("./invoke").invoke;
    let currentGeneration = 4;

    requestSolveCancellation(
      "exact",
      4,
      17,
      (generation) => generation === currentGeneration,
      setState,
      setError,
      invokeCommand,
    );

    await Promise.resolve();
    await Promise.resolve();
    expect(calls).toEqual(["game_cancel_solve", "game_get_state"]);

    currentGeneration = 5;
    resolveRefresh(refreshedState as unknown as GameState);
    await Promise.resolve();
    await Promise.resolve();

    expect(calls).toEqual(["game_cancel_solve", "game_get_state"]);
    expect(setState).not.toHaveBeenCalled();
    expect(setError).not.toHaveBeenCalled();
  });

  it("recognizes both backend generation response shapes and stopped states", () => {
    expect(getBackendSolveGeneration(17)).toBe(17);
    expect(getBackendSolveGeneration({ generation: 23 })).toBe(23);
    expect(isSolveStopped({ solve: null })).toBe(true);
    expect(isSolveStopped({ solve: { ...solve, is_complete: true } })).toBe(true);
    expect(isSolveStopped({ solve })).toBe(false);
  });
});
