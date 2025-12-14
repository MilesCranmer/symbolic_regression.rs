use crate::loss::{mse, LossObject};
use crate::operators::Operators;
use num_traits::Float;

#[derive(Clone, Debug)]
pub struct MutationWeights {
    pub mutate_constant: f64,
    pub mutate_operator: f64,
    pub mutate_feature: f64,
    pub swap_operands: f64,
    pub rotate_tree: f64,
    pub add_node: f64,
    pub insert_node: f64,
    pub delete_node: f64,
    pub simplify: f64,
    pub randomize: f64,
    pub do_nothing: f64,
    pub optimize: f64,
    pub form_connection: f64,
    pub break_connection: f64,
}

impl Default for MutationWeights {
    fn default() -> Self {
        // Defaults from SymbolicRegression.jl `default_options()` (>= v1.0.0 branch).
        Self {
            mutate_constant: 0.0346,
            mutate_operator: 0.293,
            mutate_feature: 0.1,
            swap_operands: 0.198,
            rotate_tree: 4.26,
            add_node: 2.47,
            insert_node: 0.0112,
            delete_node: 0.870,
            simplify: 0.00209,
            randomize: 0.000502,
            do_nothing: 0.273,
            optimize: 0.0,
            form_connection: 0.5,
            break_connection: 0.1,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum OutputStyle {
    /// Enable ANSI styles only when stderr supports it (and `NO_COLOR` is not set).
    Auto,
    /// Disable ANSI styles.
    Plain,
    /// Force ANSI styles (even when stderr is not a TTY).
    Ansi,
}

impl Default for OutputStyle {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Clone)]
pub struct Options<T: Float, const D: usize> {
    pub seed: u64,

    // Search size / structure
    pub niterations: usize,
    pub populations: usize,
    pub population_size: usize,
    pub ncycles_per_iteration: usize,

    // Operators and constraints
    pub operators: Operators<D>,
    pub maxsize: usize,
    pub maxdepth: usize,
    pub warmup_maxsize_by: f32,

    // Working with complexities / adaptive parsimony
    pub parsimony: f64,
    pub adaptive_parsimony_scaling: f64,
    pub use_frequency: bool,
    pub use_frequency_in_tournament: bool,

    // Mutations
    pub mutation_weights: MutationWeights,
    pub crossover_probability: f64,
    pub perturbation_factor: f64,
    pub probability_negate_constant: f64,
    pub skip_mutation_failures: bool,

    // Tournament selection
    pub tournament_selection_n: usize,
    pub tournament_selection_p: f32,

    // Annealing
    pub annealing: bool,
    pub alpha: f64,

    // Constant optimization
    pub optimizer_nrestarts: usize,
    pub optimizer_probability: f64,
    pub optimizer_iterations: usize,
    pub optimizer_f_calls_limit: usize,
    pub should_optimize_constants: bool,

    // Simplification (stubbed, but controls weighting)
    pub should_simplify: bool,

    // Migration (simple port)
    pub migration: bool,
    pub hof_migration: bool,
    pub fraction_replaced: f64,
    pub fraction_replaced_hof: f64,
    pub fraction_replaced_guesses: f64,
    pub topn: usize,

    // Loss
    pub loss: LossObject<T>,

    // Baseline normalization
    pub use_baseline: bool,

    // Runtime / UI
    pub progress: bool,
    pub output_style: OutputStyle,
}

impl<T: Float, const D: usize> Default for Options<T, D> {
    fn default() -> Self {
        Self {
            seed: 0,
            niterations: 10,
            populations: 31,
            population_size: 27,
            ncycles_per_iteration: 380,
            operators: Operators::new(),
            maxsize: 30,
            maxdepth: 10,
            warmup_maxsize_by: 0.0,
            parsimony: 0.0,
            adaptive_parsimony_scaling: 20.0,
            use_frequency: true,
            use_frequency_in_tournament: true,
            mutation_weights: MutationWeights::default(),
            crossover_probability: 0.0259,
            perturbation_factor: 0.129,
            probability_negate_constant: 0.00743,
            skip_mutation_failures: true,
            tournament_selection_n: 15,
            tournament_selection_p: 0.982,
            annealing: true,
            alpha: 3.17,
            optimizer_nrestarts: 2,
            optimizer_probability: 0.14,
            optimizer_iterations: 8,
            optimizer_f_calls_limit: 10_000,
            should_optimize_constants: true,
            should_simplify: false,
            migration: true,
            hof_migration: true,
            fraction_replaced: 0.00036,
            fraction_replaced_hof: 0.0614,
            fraction_replaced_guesses: 0.001,
            topn: 12,
            loss: mse::<T>(),
            use_baseline: true,
            progress: true,
            output_style: OutputStyle::Auto,
        }
    }
}
