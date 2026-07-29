import { describe, expect, it } from "vitest";
import {
  getComboActionRows,
  isUniversalMpLazyBundleName,
  shouldShowStrategyMatrix,
  supportsUniversalMpLazyExact,
} from "./GameExplorer";

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
