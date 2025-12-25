#![deny(unsafe_op_in_unsafe_fn)]

pub mod compile;
pub mod evaluate;
pub mod evaluate_derivative;
pub mod expression;
pub mod expression_algebra;
pub mod node;
pub mod node_utils;
pub mod operator_enum;
pub mod operator_registry;
pub mod operators;
#[cfg(feature = "proptest-utils")]
pub mod proptest_utils;
pub mod simplify;
pub mod strings;
pub mod utils;

pub use {num_traits, paste};

pub use crate::compile::{EvalPlan, Instr, compile_plan};
pub use crate::evaluate::{EvalContext, EvalOptions, eval_plan_array_into, eval_tree_array, eval_tree_array_into};
pub use crate::evaluate_derivative::{
    DiffContext, GradContext, GradMatrix, eval_diff_tree_array, eval_grad_tree_array,
};
pub use crate::expression::{Metadata, PostfixExpr, PostfixExpression, PostfixExpressionMut};
pub use crate::expression_algebra::{Lit, lit};
pub use crate::node::{PNode, Src};
pub use crate::node_utils::{
    count_constant_nodes, count_depth, count_nodes, has_constants, has_operators, subtree_range, subtree_sizes,
    tree_mapreduce,
};
pub use crate::simplify::{combine_operators_in_place, simplify_in_place, simplify_tree_in_place};
pub use crate::strings::{OpNames, StringTreeOptions, print_tree, string_tree};
pub use crate::utils::{ConstRef, compress_constants, get_scalar_constants, set_scalar_constants};
