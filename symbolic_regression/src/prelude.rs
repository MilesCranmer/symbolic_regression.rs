//! Convenience re-exports for examples and quickstarts.

pub use dynamic_expressions::evaluate::EvalOptions;
pub use dynamic_expressions::expression::PostfixExpr;
pub use dynamic_expressions::node::PNode;
pub use dynamic_expressions::operator_enum::builtin::*;
pub use dynamic_expressions::operator_enum::presets::*;
pub use dynamic_expressions::operators::*;
pub use dynamic_expressions::strings::{print_tree, string_tree};
pub use dynamic_expressions::{eval_diff_tree_array, eval_grad_tree_array, eval_tree_array};

// Re-export common `dynamic_expressions` types/functions so callers (and examples) don't need to
// depend on `dynamic_expressions` directly.
pub use crate::custom_opset;
pub use crate::dataset::Dataset;
pub use crate::operators::Operators;
pub use crate::options::{MutationWeights, Options};
pub use crate::search_utils::{SearchEngine, SearchResult, equation_search};
