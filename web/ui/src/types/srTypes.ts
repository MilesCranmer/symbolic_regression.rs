export type WasmOpInfo = {
  arity: number;
  name: string;
  display: string;
  infix: string | null;
  commutative: boolean;
  associative: boolean;
  complexity: number;
};

export type WasmMutationWeights = {
  mutate_constant: number;
  mutate_operator: number;
  mutate_feature: number;
  swap_operands: number;
  rotate_tree: number;
  add_node: number;
  insert_node: number;
  delete_node: number;
  simplify: number;
  randomize: number;
  do_nothing: number;
  optimize: number;
  form_connection: number;
  break_connection: number;
};

export type WasmSearchOptions = {
  has_headers: boolean;
  x_columns: number[] | null;
  y_column: number | null;
  weights_column: number | null;
  validation_fraction: number;
  loss_kind: string;
  huber_delta: number;

  seed: number;
  niterations: number;
  populations: number;
  population_size: number;
  ncycles_per_iteration: number;
  maxsize: number;
  maxdepth: number;
  warmup_maxsize_by: number;
  parsimony: number;
  adaptive_parsimony_scaling: number;
  crossover_probability: number;
  perturbation_factor: number;
  probability_negate_constant: number;
  tournament_selection_n: number;
  tournament_selection_p: number;
  alpha: number;
  optimizer_nrestarts: number;
  optimizer_probability: number;
  optimizer_iterations: number;
  optimizer_f_calls_limit: number;
  fraction_replaced: number;
  fraction_replaced_hof: number;
  topn: number;
  print_precision: number;

  use_frequency: boolean;
  use_frequency_in_tournament: boolean;
  skip_mutation_failures: boolean;
  annealing: boolean;
  should_optimize_constants: boolean;
  migration: boolean;
  hof_migration: boolean;
  use_baseline: boolean;
  progress: boolean;
  should_simplify: boolean;

  mutation_weights: WasmMutationWeights;
};

export type EquationPoint = {
  id: string;
  complexity: number;
  loss: number;
  cost: number;
};

export type EquationSummary = {
  id: string;
  complexity: number;
  loss: number;
  cost: number;
  /** Display-precision equation (matches `print_precision`, default 5 significant digits). */
  equation: string;
  /** Full-precision equation, suitable for copy-to-clipboard. */
  equation_full: string;
};

export type SearchSnapshot = {
  total_cycles: number;
  cycles_completed: number;
  total_evals: number;
  best: EquationSummary;
  pareto_points: EquationPoint[];
};

export type WasmSplitIndices = {
  train: number[];
  val: number[];
};

export type WasmMetrics = {
  n: number;
  mse: number;
  mae: number;
  rmse: number;
  r2: number;
  corr: number;
  min_abs_err: number;
  max_abs_err: number;
};

export type WasmEvalResult = {
  metrics: WasmMetrics;
  yhat: number[];
};
