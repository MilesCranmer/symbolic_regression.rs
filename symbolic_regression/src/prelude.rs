//! Convenience re-exports for examples and quickstarts.

pub use crate::dataset::Dataset;
pub use crate::operators::Operators;
pub use crate::options::{MutationWeights, Options};
pub use crate::search::{equation_search, SearchResult};

// Re-export common `dynamic_expressions` types/functions so callers (and examples) don't need to
// depend on `dynamic_expressions` directly.
pub use dynamic_expressions::eval::EvalOptions;
pub use dynamic_expressions::expr::{PNode, PostfixExpr};
pub use dynamic_expressions::math::*;
pub use dynamic_expressions::operators::builtin::*;
pub use dynamic_expressions::operators::presets::*;
pub use dynamic_expressions::strings::{print_tree, string_tree, OpNames};
pub use dynamic_expressions::{eval_diff_tree_array, eval_grad_tree_array, eval_tree_array};
