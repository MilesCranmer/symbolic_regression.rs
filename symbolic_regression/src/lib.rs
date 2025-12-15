pub(crate) mod adaptive_parsimony;
pub(crate) mod constant_optimization;
pub(crate) mod dataset;
pub(crate) mod hall_of_fame;
pub(crate) mod loss;
pub(crate) mod member;
pub(crate) mod mutate;
pub(crate) mod operator_library;
pub(crate) mod operators;
pub(crate) mod optim;
pub(crate) mod options;
pub(crate) mod population;
pub mod prelude;
pub(crate) mod search;
pub(crate) mod selection;

#[cfg(feature = "cli")]
pub mod cli;

#[cfg(feature = "bench")]
pub mod bench;

pub use dataset::Dataset;
pub use hall_of_fame::HallOfFame;
pub use loss::mse;
pub use member::{MemberId, PopMember};
pub use operator_library::OperatorLibrary;
pub use operators::Operators;
pub use options::OutputStyle;
pub use options::{MutationWeights, Options};
pub use search::SearchEngine;
pub use search::{equation_search, SearchResult};

#[cfg(test)]
mod tests;
