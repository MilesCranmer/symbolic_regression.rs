#![deny(unsafe_op_in_unsafe_fn)]

pub mod algebra;
pub mod compile;
pub mod eval;
pub mod expr;
pub mod math;
pub mod operators;
pub mod strings;
pub mod tree;
pub mod utils;

pub use num_traits;
pub use paste;

pub use crate::algebra::{lit, Lit};
pub use crate::compile::{compile_plan, EvalPlan, Instr};
pub use crate::eval::{
    eval_diff_tree_array, eval_grad_tree_array, eval_plan_array_into, eval_tree_array,
    eval_tree_array_into, DiffContext, EvalContext, EvalOptions, GradContext, GradMatrix,
};
pub use crate::expr::{Metadata, PNode, PostfixExpr, Src};
pub use crate::expr::{PostfixExpression, PostfixExpressionMut};
pub use crate::strings::{print_tree, string_tree, OpNames, StringTreeOptions};
pub use crate::tree::{
    count_constant_nodes, count_depth, count_nodes, has_constants, has_operators, subtree_range,
    subtree_sizes, tree_mapreduce,
};
pub use crate::utils::{get_scalar_constants, set_scalar_constants, ConstRef};
