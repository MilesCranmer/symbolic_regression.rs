import "@testing-library/jest-dom/vitest";
import React from "react";
import { vi } from "vitest";

// jsdom doesn't always implement File.text() depending on version; polyfill for upload tests.
if (typeof File !== "undefined" && typeof (File.prototype as any).text !== "function") {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (File.prototype as any).text = function () {
    return new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result ?? ""));
      reader.onerror = () => reject(reader.error);
      reader.readAsText(this);
    });
  };
}

// Mock the WASM bindings so tests don't load real .wasm or depend on wasm-pack outputs.
vi.mock("../pkg/symbolic_regression_wasm.js", () => {
  const coreDefaults = {
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
    should_simplify: true,

    batching: false,

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
  };
  return {
    default: vi.fn(async () => ({})),
    builtin_operator_registry: vi.fn(async () => []),
    default_core_options: vi.fn(async () => coreDefaults),
    default_search_options: vi.fn(async () => ({
      has_headers: true,
      x_columns: null,
      y_column: null,
      weights_column: null,
      validation_fraction: 0,
      loss_kind: "mse",
      huber_delta: 1.0,

      ...coreDefaults
    }))
  };
});

// Mock Plotly React component (avoid pulling Plotly into jsdom and keep assertions simple).
vi.mock("react-plotly.js", () => {
  return {
    default: (props: any) => {
      const xTitle = props?.layout?.xaxis?.title;
      const yTitle = props?.layout?.yaxis?.title;
      const id =
        typeof xTitle === "string"
          ? `plot-x-${xTitle}`
          : typeof yTitle === "string"
            ? `plot-y-${yTitle}`
            : "plot";
      return React.createElement("div", {
        "data-testid": id,
        "data-props": JSON.stringify({ data: props.data, layout: props.layout })
      });
    }
  };
});
