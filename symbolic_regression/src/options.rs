use crate::loss_functions::{mse, LossObject};
use crate::operators::Operators;
use num_traits::Float;

#[rustfmt::skip]
macro_rules! sr_mutation_weights_spec {
    ($m:ident) => {
        $m! {
            mutate_constant:
                (f64, 0.0346, "mw-mutate-constant"),
            mutate_operator:
                (f64, 0.293, "mw-mutate-operator"),
            mutate_feature:
                (f64, 0.1, "mw-mutate-feature"),
            swap_operands:
                (f64, 0.198, "mw-swap-operands"),
            rotate_tree:
                (f64, 4.26, "mw-rotate-tree"),
            add_node:
                (f64, 2.47, "mw-add-node"),
            insert_node:
                (f64, 0.0112, "mw-insert-node"),
            delete_node:
                (f64, 0.870, "mw-delete-node"),
            simplify:
                (f64, 0.00209, "mw-simplify"),
            randomize:
                (f64, 0.000502, "mw-randomize"),
            do_nothing:
                (f64, 0.273, "mw-do-nothing"),
            optimize:
                (f64, 0.0, "mw-optimize"),
            form_connection:
                (f64, 0.5, "mw-form-connection"),
            break_connection:
                (f64, 0.1, "mw-break-connection"),
        }
    };
}

#[rustfmt::skip]
macro_rules! sr_options_spec {
    ($m:ident) => {
        $m! {
            values {
                seed:
                    (u64, 0, "seed"),
                niterations:
                    (usize, 100, "niterations"),
                populations:
                    (usize, 31, "populations"),
                population_size:
                    (usize, 27, "population-size"),
                ncycles_per_iteration:
                    (usize, 380, "ncycles-per-iteration"),
                batch_size:
                    (usize, 50, "batch-size"),
                complexity_of_constants:
                    (u16, 1, "complexity-of-constants"),
                complexity_of_variables:
                    (u16, 1, "complexity-of-variables"),
                maxsize:
                    (usize, 30, "maxsize"),
                maxdepth:
                    (usize, 30, "maxdepth"),
                warmup_maxsize_by: 
                    (f32, 0.0, "warmup-maxsize-by"),
                parsimony:
                    (f64, 0.0, "parsimony"),
                adaptive_parsimony_scaling:
                    (f64, 1040.0, "adaptive-parsimony-scaling"),
                crossover_probability:
                    (f64, 0.0259, "crossover-probability"),
                perturbation_factor:
                    (f64, 0.129, "perturbation-factor"),
                probability_negate_constant:
                    (f64, 0.00743, "probability-negate-constant"),
                tournament_selection_n:
                    (usize, 15, "tournament-selection-n"),
                tournament_selection_p:
                    (f32, 0.982, "tournament-selection-p"),
                alpha:
                    (f64, 3.17, "alpha"),
                optimizer_nrestarts:
                    (usize, 2, "optimizer-nrestarts"),
                optimizer_probability:
                    (f64, 0.14, "optimizer-probability"),
                optimizer_iterations:
                    (usize, 8, "optimizer-iterations"),
                optimizer_f_calls_limit:
                    (usize, 10_000, "optimizer-f-calls-limit"),
                fraction_replaced:
                    (f64, 0.00036, "fraction-replaced"),
                fraction_replaced_hof:
                    (f64, 0.0614, "fraction-replaced-hof"),
                fraction_replaced_guesses:
                    (f64, 0.001, "fraction-replaced-guesses"),
                topn:
                    (usize, 12, "topn"),
            }
            neg_flags {
                use_frequency:
                    (true, no_use_frequency, "no-use-frequency"),
                use_frequency_in_tournament:
                    (true, no_use_frequency_in_tournament, "no-use-frequency-in-tournament"),
                skip_mutation_failures:
                    (true, no_skip_mutation_failures, "no-skip-mutation-failures"),
                annealing:
                    (true, no_annealing, "no-annealing"),
                should_optimize_constants:
                    (true, no_should_optimize_constants, "no-should-optimize-constants"),
                migration:
                    (true, no_migration, "no-migration"),
                hof_migration:
                    (true, no_hof_migration, "no-hof-migration"),
                use_baseline:
                    (true, no_use_baseline, "no-use-baseline"),
                progress:
                    (true, no_progress, "no-progress"),
            }
            pos_flags {
                should_simplify:
                    (true, should_simplify, "should-simplify"),
                batching:
                    (false, batching, "batching"),
            }
        }
    };
}

#[rustfmt::skip]
macro_rules! __define_mutation_weights {
    ( $( $name:ident: ($ty:ty, $default:expr, $cli_long:literal), )* ) => {
        #[derive(Clone, Debug)]
        pub struct MutationWeights {
            $(pub $name: $ty,)*
        }

        impl Default for MutationWeights {
            fn default() -> Self {
                Self { $($name: $default,)* }
            }
        }
    };
}

sr_mutation_weights_spec!(__define_mutation_weights);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum OutputStyle {
    /// Enable ANSI styles only when stderr supports it (and `NO_COLOR` is not set).
    #[default]
    Auto,
    /// Disable ANSI styles.
    Plain,
    /// Force ANSI styles (even when stderr is not a TTY).
    Ansi,
}

macro_rules! __define_options {
    (
        values { $( $name:ident: ($ty:ty, $default:expr, $cli_long:literal), )* }
        neg_flags { $( $iname:ident: ($bdefault:expr, $cli_name:ident, $cli_blong:literal), )* }
        pos_flags { $( $pname:ident: ($pdefault:expr, $cli_pname:ident, $cli_plong:literal), )* }
    ) => {
        #[derive(Clone)]
        pub struct Options<T: Float, const D: usize> {
            $(pub $name: $ty,)*
            $(pub $iname: bool,)*
            $(pub $pname: bool,)*

            pub operators: Operators<D>,
            pub mutation_weights: MutationWeights,
            pub loss: LossObject<T>,

            pub output_style: OutputStyle,

            pub variable_complexities: Option<Vec<u16>>,
            pub operator_complexity_overrides: std::collections::HashMap<
                dynamic_expressions::operator_enum::scalar::OpId,
                u16,
            >,
            pub op_constraints: crate::check_constraints::OpConstraints<D>,
            pub nested_constraints: crate::check_constraints::NestedConstraints,
        }

        impl<T: Float, const D: usize> Default for Options<T, D> {
            fn default() -> Self {
                Self {
                    $($name: $default,)*
                    $($iname: $bdefault,)*
                    $($pname: $pdefault,)*
                    operators: Operators::new(),
                    mutation_weights: MutationWeights::default(),
                    loss: mse::<T>(),
                    output_style: OutputStyle::Auto,
                    variable_complexities: None,
                    operator_complexity_overrides: std::collections::HashMap::new(),
                    op_constraints: Default::default(),
                    nested_constraints: Default::default(),
                }
            }
        }
    };
}

sr_options_spec!(__define_options);

impl<T: Float, const D: usize> Options<T, D> {
    pub fn uses_default_complexity(&self) -> bool {
        self.complexity_of_constants == 1
            && self.complexity_of_variables == 1
            && self.variable_complexities.is_none()
            && self.operator_complexity_overrides.is_empty()
    }
}

#[cfg(feature = "cli")]
pub(crate) mod cli_args {
    use super::{MutationWeights, Options, OutputStyle};
    use clap::{Args, ValueEnum};
    use num_traits::Float;

    #[derive(Copy, Clone, Debug, ValueEnum)]
    pub enum OutputStyleCli {
        Auto,
        Plain,
        Ansi,
    }

    macro_rules! __define_mutation_weights_args {
        ( $( $name:ident: ($ty:ty, $default:expr, $cli_long:literal), )* ) => {
            #[derive(Args, Debug, Clone, Default)]
            pub struct MutationWeightsArgs {
                $(#[arg(long = $cli_long)] pub $name: Option<$ty>,)*
            }

            impl MutationWeightsArgs {
                pub fn apply_to(&self, w: &mut MutationWeights) {
                    $(if let Some(v) = self.$name { w.$name = v; })*
                }
            }
        };
    }

    sr_mutation_weights_spec!(__define_mutation_weights_args);

    macro_rules! __define_options_args {
        (
            values { $( $name:ident: ($ty:ty, $default:expr, $cli_long:literal), )* }
            neg_flags { $( $iname:ident: ($bdefault:expr, $cli_name:ident, $cli_blong:literal), )* }
            pos_flags { $( $pname:ident: ($pdefault:expr, $cli_pname:ident, $cli_plong:literal), )* }
        ) => {
            #[derive(Args, Debug, Clone, Default)]
            pub struct OptionsArgs {
                $(
                    #[arg(long = $cli_long)]
                    pub $name: Option<$ty>,
                )*

                $(
                    #[arg(long = $cli_blong)]
                    pub $cli_name: bool,
                )*

                $(
                    #[arg(long = $cli_plong)]
                    pub $cli_pname: bool,
                )*

                #[command(flatten)]
                pub mutation_weights: MutationWeightsArgs,

                #[arg(long, value_enum)]
                pub output_style: Option<OutputStyleCli>,
            }

            impl OptionsArgs {
                pub fn apply_to<T: Float, const D: usize>(&self, opt: &mut Options<T, D>) {
                    $(
                        if let Some(v) = self.$name {
                            opt.$name = v;
                        }
                    )*

                    $(
                        if self.$cli_name {
                            opt.$iname = false;
                        }
                    )*

                    $(
                        if self.$cli_pname {
                            opt.$pname = true;
                        }
                    )*

                    self.mutation_weights.apply_to(&mut opt.mutation_weights);
                    if let Some(s) = self.output_style {
                        opt.output_style = match s {
                            OutputStyleCli::Auto => OutputStyle::Auto,
                            OutputStyleCli::Plain => OutputStyle::Plain,
                            OutputStyleCli::Ansi => OutputStyle::Ansi,
                        };
                    }
                }
            }
        };
    }

    sr_options_spec!(__define_options_args);
}

#[cfg(all(test, feature = "cli"))]
mod cli_args_tests {
    use super::cli_args::OptionsArgs;
    use super::Options;

    #[test]
    fn cli_options_patch_applies() {
        let args = OptionsArgs {
            niterations: Some(123),
            no_progress: true,
            ..Default::default()
        };
        let mut opt: Options<f64, 3> = Options::default();
        args.apply_to(&mut opt);
        assert_eq!(opt.niterations, 123);
        assert!(!opt.progress);
    }
}
