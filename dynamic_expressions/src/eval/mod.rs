mod context;
mod resolve;
mod scalar_diff;
mod scalar_eval;
mod scalar_grad;

pub use context::{DiffContext, EvalContext, EvalOptions, GradContext};
pub use scalar_diff::eval_diff_tree_array;
pub use scalar_eval::eval_plan_array_into;
pub use scalar_eval::eval_tree_array;
pub use scalar_eval::eval_tree_array_into;
pub use scalar_grad::{eval_grad_tree_array, GradMatrix};
