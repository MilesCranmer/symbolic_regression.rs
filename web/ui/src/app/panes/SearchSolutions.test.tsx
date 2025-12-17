import React from "react";
import { render, screen } from "@testing-library/react";
import { SearchSolutions } from "./SearchSolutions";
import { useSessionStore } from "../../state/sessionStore";
import type { SearchSnapshot, WasmSplitIndices } from "../../types/srTypes";
import { vi } from "vitest";

vi.mock("../../worker/srWorkerClient", () => {
  class SrWorkerClient {
    setHandlers() {}
    terminate() {}
    init() {}
    start() {}
    pause() {}
    reset() {}
    step() {}
    evaluate() {}
  }
  return { SrWorkerClient };
});

function setCommonState(args: { xColumns: number[] }) {
  const parsed = {
    headers: ["x", "y"],
    rows: [
      [0, 1],
      [1, 3],
      [2, 5]
    ]
  };

  const split: WasmSplitIndices = { train: [0, 1, 2], val: [] };

  const snap: SearchSnapshot = {
    total_cycles: 10,
    cycles_completed: 1,
    total_evals: 0,
    best: { id: "1", complexity: 3, loss: 0.1, cost: 0.1, equation: "x" },
    pareto_points: [{ id: "1", complexity: 3, loss: 0.1, cost: 0.1 }]
  };

  useSessionStore.setState({
    parsed,
    options: {
      has_headers: true,
      x_columns: args.xColumns,
      y_column: 1,
      weights_column: null,
      validation_fraction: 0,
      loss_kind: "mse",
      huber_delta: 1.0,

      seed: 0,
      niterations: 100,
      populations: 2,
      population_size: 16,
      ncycles_per_iteration: 10,
      maxsize: 30,
      maxdepth: 30,
      warmup_maxsize_by: 0,
      parsimony: 0,
      adaptive_parsimony_scaling: 1040,
      crossover_probability: 0.0259,
      perturbation_factor: 0.129,
      probability_negate_constant: 0.00743,
      tournament_selection_n: 15,
      tournament_selection_p: 0.982,
      alpha: 3.17,
      optimizer_nrestarts: 2,
      optimizer_probability: 0.14,
      optimizer_iterations: 8,
      optimizer_f_calls_limit: 10_000,
      fraction_replaced: 0.00036,
      fraction_replaced_hof: 0.0614,
      fraction_replaced_guesses: 0.001,
      topn: 12,

      use_frequency: true,
      use_frequency_in_tournament: true,
      skip_mutation_failures: true,
      annealing: true,
      should_optimize_constants: true,
      migration: true,
      hof_migration: true,
      use_baseline: true,
      progress: false,
      should_simplify: false,

      mutation_weights: {
        mutate_constant: 0.0346,
        mutate_operator: 0.293,
        mutate_feature: 0.1,
        swap_operands: 0.198,
        rotate_tree: 4.26,
        add_node: 2.47,
        insert_node: 0.0112,
        delete_node: 0.87,
        simplify: 0.00209,
        randomize: 0.000502,
        do_nothing: 0.273,
        optimize: 0.0,
        form_connection: 0.5,
        break_connection: 0.1
      }
    } as any,
    runtime: {
      status: "ready",
      error: null,
      split,
      snapshot: snap,
      front: [{ id: "1", complexity: 3, loss: 0.1, cost: 0.1, equation: "x" }],
      selectedId: "1",
      selectedComplexity: 3,
      evalByKey: {
        "1:train": {
          metrics: {
            n: 3,
            mse: 0.0,
            mae: 0.0,
            rmse: 0.0,
            r2: 1.0,
            corr: 1.0,
            min_abs_err: 0.0,
            max_abs_err: 0.0
          },
          yhat: [1, 3, 5]
        }
      }
    }
  });
}

describe("SearchSolutions fit plot", () => {
  it("auto mode uses 1D curve when exactly one X column is selected", () => {
    setCommonState({ xColumns: [0] });
    render(<SearchSolutions />);

    const plot = screen.getByTestId("plot-x-x");
    const payload = JSON.parse(plot.getAttribute("data-props") ?? "{}");
    const modes = (payload.data ?? []).map((t: any) => t.mode);
    expect(modes).toContain("lines");
  });

  it("auto mode falls back to parity when multiple X columns selected", () => {
    setCommonState({ xColumns: [0, 1] });
    render(<SearchSolutions />);

    const plot = screen.getByTestId("plot-x-y (actual)");
    const payload = JSON.parse(plot.getAttribute("data-props") ?? "{}");
    const modes = (payload.data ?? []).map((t: any) => t.mode);
    expect(modes).not.toContain("lines");
  });
});
