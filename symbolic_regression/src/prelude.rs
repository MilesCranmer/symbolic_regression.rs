//! Convenience re-exports for examples and quickstarts.
//!
//! If you're just getting started, `use symbolic_regression::prelude::*;` should give you the
//! pieces you need to:
//! - build a [`Dataset`],
//! - configure an [`Options`] value (including an operator set),
//! - run [`equation_search`] and inspect the [`crate::HallOfFame`],
//! - define custom operators via [`op!`] / [`opset!`].

pub use dynamic_expressions::evaluate::EvalOptions;
pub use dynamic_expressions::expression::PostfixExpr;
pub use dynamic_expressions::node::PNode;
pub use dynamic_expressions::operator_enum::builtin::*;
pub use dynamic_expressions::operator_enum::presets::*;
pub use dynamic_expressions::operators::*;
pub use dynamic_expressions::strings::{print_tree, string_tree};
pub use dynamic_expressions::{Operators, eval_diff_tree_array, eval_grad_tree_array, eval_tree_array};

pub use crate::dataset::Dataset;
pub use crate::options::{MutationWeights, Options};
pub use crate::search_utils::{SearchEngine, SearchResult, equation_search};
// Re-export common `dynamic_expressions` macros so callers (and examples) don't need to depend on
// `dynamic_expressions` directly.
pub use crate::{op, opset};
