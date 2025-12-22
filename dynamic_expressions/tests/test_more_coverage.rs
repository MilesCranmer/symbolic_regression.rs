mod common;

use common::{TestOps, make_x, var};
use dynamic_expressions::{EvalOptions, PNode, compile_plan, eval_tree_array};
use ndarray::Array2;

#[test]
fn eval_options_default_is_true_true() {
    let opts = EvalOptions::default();
    assert!(opts.check_finite);
    assert!(opts.early_exit);
}

#[test]
#[should_panic]
fn compile_plan_panics_if_n_features_exceeds_u16() {
    let nodes = vec![PNode::Var { feature: 0 }];
    let _ = compile_plan::<2>(&nodes, (u16::MAX as usize) + 1, 0);
}

#[test]
#[should_panic]
fn compile_plan_panics_if_n_consts_exceeds_u16() {
    let nodes = vec![PNode::Var { feature: 0 }];
    let _ = compile_plan::<2>(&nodes, 1, (u16::MAX as usize) + 1);
}

#[test]
fn eval_early_exit_can_trigger_inside_the_instruction_loop() {
    // cos(x1 / 0.0) should hit non-finite during the Div instruction and return early.
    let expr = dynamic_expressions::operators::cos(var(0) / 0.0);
    let (_owned, x) = make_x(1, 10);
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let (_y, ok) = eval_tree_array::<f64, TestOps, 3>(&expr, x.view(), &opts);
    assert!(!ok);
}

#[test]
fn string_helpers_cover_edge_cases() {
    // Exercise the "use provided variable name" branch and a strip_outer_parens case
    // where the outer parentheses do not enclose the whole string.
    let mut expr = var(0);
    expr.meta.variable_names = vec!["(a)(b)".to_string()];
    let s =
        dynamic_expressions::strings::string_tree(&expr, dynamic_expressions::strings::StringTreeOptions::default());
    assert_eq!(s, "(a)(b)");

    let s2 = dynamic_expressions::strings::default_string_variable(0, Some(&["x".to_string()]));
    assert_eq!(s2, "x");

    // Exercise print_tree (just assert it doesn't panic).
    let x = Array2::from_shape_vec((1, 1), vec![0.0f64]).unwrap();
    let opts = EvalOptions::default();
    let (_y, ok) = eval_tree_array::<f64, TestOps, 3>(&expr, x.view(), &opts);
    assert!(ok);
    dynamic_expressions::strings::print_tree(&expr);
}

#[test]
fn builtin_eval_apply_constant_only_early_exit_is_exercised() {
    let expr = dynamic_expressions::operators::cos(common::c(f64::NAN));
    let x = Array2::from_shape_vec((1, 1), vec![0.0f64]).unwrap();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let (_y, ok) = eval_tree_array::<f64, TestOps, 3>(&expr, x.view(), &opts);
    assert!(!ok);
}
