#![deny(unsafe_op_in_unsafe_fn)]

pub mod compile;
pub mod evaluate;
pub mod evaluate_derivative;
pub mod expression;
pub mod expression_algebra;
pub mod math;
pub mod node;
pub mod node_utils;
pub mod operator_enum;
pub mod operator_registry;
pub mod simplify;
pub mod strings;
pub mod utils;

pub use num_traits;
pub use paste;

pub use crate::compile::{compile_plan, EvalPlan, Instr};
pub use crate::evaluate::{
    eval_plan_array_into, eval_tree_array, eval_tree_array_into, EvalContext, EvalOptions,
};
pub use crate::evaluate_derivative::{
    eval_diff_tree_array, eval_grad_tree_array, DiffContext, GradContext, GradMatrix,
};
pub use crate::expression::{Metadata, PostfixExpr, PostfixExpression, PostfixExpressionMut};
pub use crate::expression_algebra::{lit, Lit};
pub use crate::node::{PNode, Src};
pub use crate::node_utils::{
    count_constant_nodes, count_depth, count_nodes, has_constants, has_operators, subtree_range,
    subtree_sizes, tree_mapreduce,
};
pub use crate::simplify::{combine_operators_in_place, simplify_in_place, simplify_tree_in_place};
pub use crate::strings::{print_tree, string_tree, OpNames, StringTreeOptions};
pub use crate::utils::{compress_constants, get_scalar_constants, set_scalar_constants, ConstRef};
