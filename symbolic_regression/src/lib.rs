pub(crate) mod adaptive_parsimony;
pub(crate) mod check_constraints;
pub(crate) mod complexity;
pub(crate) mod constant_optimization;
pub(crate) mod dataset;
pub(crate) mod hall_of_fame;
pub(crate) mod loss_functions;
pub(crate) mod migration;
pub(crate) mod mutate;
pub(crate) mod operator_library;
pub mod operators;
pub(crate) mod optim;
pub(crate) mod options;
pub(crate) mod pop_member;
pub(crate) mod population;
pub mod prelude;
pub(crate) mod progress_bars;
pub(crate) mod regularized_evolution;
pub(crate) mod search_utils;
pub(crate) mod selection;
pub(crate) mod single_iteration;
pub(crate) mod warmup;

#[cfg(feature = "cli")]
pub mod cli;

#[cfg(feature = "bench")]
pub mod bench;

pub use check_constraints::{NestedConstraints, OpConstraints};
pub use complexity::compute_complexity;
pub use dataset::{Dataset, TaggedDataset};
pub use hall_of_fame::HallOfFame;
pub use loss_functions::{
    epsilon_insensitive, huber, log_cosh, lp, mae, make_loss, mse, quantile, rmse, LossKind,
};
pub use operator_library::OperatorLibrary;
pub use operators::{OperatorRegistryExt, OperatorSelectError, Operators};
pub use options::OutputStyle;
pub use options::{MutationWeights, Options};
pub use pop_member::{MemberId, PopMember};
pub use search_utils::SearchEngine;
pub use search_utils::{equation_search, SearchResult};

#[doc(hidden)]
pub use dynamic_expressions::custom_opset as __dynamic_expressions_custom_opset;

#[cfg(test)]
mod tests;
