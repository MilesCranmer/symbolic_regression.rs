mod common;

use common::*;
use dynamic_expressions::{
    DiffContext, EvalContext, EvalOptions, GradContext, PNode, PostfixExpression, eval_diff_tree_array,
    eval_grad_tree_array, eval_tree_array,
};
use ndarray::Array2;

#[test]
fn constants_get_set_roundtrip_changes_eval() {
    let mut expr = expr_readme_like();
    let (_x_data, x) = make_x(2, 19);
    let x_view = x.view();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let (out0, ok0) = eval_tree_array::<f64, TestOps, 3>(&expr, x_view, &opts);
    assert!(ok0);

    let (mut cvals, cref): (Vec<f64>, dynamic_expressions::ConstRef) = dynamic_expressions::get_scalar_constants(&expr);
    assert_eq!(cvals.len(), 1);
    cvals[0] = 1.0; // change from 3.2 to 1.0
    dynamic_expressions::set_scalar_constants(&mut expr, &cvals, &cref);

    let (out1, ok1) = eval_tree_array::<f64, TestOps, 3>(&expr, x_view, &opts);
    assert!(ok1);
    assert_ne!(out0, out1);
}

#[test]
fn postfix_expression_trait_is_usable() {
    fn depth_via_trait<E: PostfixExpression<3>>(e: &E) -> usize {
        dynamic_expressions::count_depth(e.nodes())
    }

    let mut expr = expr_readme_like();
    expr.meta.variable_names = vec!["x1".to_string(), "x2".to_string()];
    assert_eq!(depth_via_trait(&expr), 4);
    assert_eq!(expr.meta.variable_names.len(), 2);

    fn read_all<E: PostfixExpression<3, Scalar = f64>>(e: &E) -> (usize, usize, usize) {
        (e.nodes().len(), e.consts().len(), e.meta().variable_names.len())
    }
    let (n_nodes, n_consts, n_names) = read_all(&expr);
    assert_eq!(n_nodes, expr.nodes.len());
    assert_eq!(n_consts, expr.consts.len());
    assert_eq!(n_names, expr.meta.variable_names.len());
}

#[test]
fn compile_plan_leaf_only_has_no_instrs() {
    let nodes = vec![PNode::Var { feature: 0 }];
    let plan = dynamic_expressions::compile_plan::<3>(&nodes, 1, 0);
    assert_eq!(plan.instrs.len(), 0);
    assert_eq!(plan.n_slots, 0);
}

#[test]
fn eval_root_var_and_const_cases() {
    let (x_data, x) = make_x(2, 11);
    let x_view = x.view();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let expr_var = var(1);
    let (outv, okv) = eval_tree_array::<f64, TestOps, 3>(&expr_var, x_view, &opts);
    assert!(okv);
    let expected: Vec<f64> = (0..x_view.nrows())
        .map(|row| x_data[row * x_view.ncols() + 1])
        .collect();
    assert_close_vec(&outv, &expected, 0.0);

    let expr_const = c(2.5);
    let (outc, okc) = eval_tree_array::<f64, TestOps, 3>(&expr_const, x_view, &opts);
    assert!(okc);
    assert!(outc.iter().all(|&v| v == 2.5));
}

#[test]
fn const_only_operator_fast_path_is_exercised() {
    let (_x_data, x) = make_x(2, 7);
    let x_view = x.view();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    // Add(c0, c1)
    let expr_add = c(1.0) + c(2.0);
    let (out, ok) = eval_tree_array::<f64, TestOps, 3>(&expr_add, x_view, &opts);
    assert!(ok);
    assert!(out.iter().all(|&v| v == 3.0));

    // Cos(c0)
    let expr_cos = dynamic_expressions::operators::cos(c(0.0));
    let (out, ok) = eval_tree_array::<f64, TestOps, 3>(&expr_cos, x_view, &opts);
    assert!(ok);
    assert!(out.iter().all(|&v| v == 1.0));

    // Fma(c0, c1, c2) = c0*c1 + c2
    let expr_t3 = dynamic_expressions::operators::fma(c(2.0), c(4.0), c(3.0));
    let (out, ok) = eval_tree_array::<f64, TestOps, 3>(&expr_t3, x_view, &opts);
    assert!(ok);
    assert!(out.iter().all(|&v| v == 11.0));
}

#[test]
fn early_exit_paths_return_nans_for_diff_and_grad() {
    // x1 / 0.0
    let expr = var(0) / 0.0;

    let n_rows = 9;
    let x_data = vec![1.0f64; n_rows];
    let x = Array2::from_shape_vec((n_rows, 1), x_data).unwrap();
    let x_view = x.view();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let mut dctx = DiffContext::<f64, 3>::new(n_rows);
    let (eval, der, ok) = eval_diff_tree_array::<f64, TestOps, 3>(&expr, x_view, 0, &mut dctx, &opts);
    assert!(!ok);
    assert!(eval.iter().all(|v| v.is_nan()));
    assert!(der.iter().all(|v| v.is_nan()));

    let mut gctx = GradContext::<f64, 3>::new(n_rows);
    let (eval, grad, ok) = eval_grad_tree_array::<f64, TestOps, 3>(&expr, x_view, true, &mut gctx, &opts);
    assert!(!ok);
    assert!(eval.iter().all(|v| v.is_nan()));
    assert!(grad.data.iter().all(|v| v.is_nan()));
}

#[test]
fn contexts_resize_and_reuse_scratch() {
    let n_rows = 5;
    let mut ectx = EvalContext::<f64, 3>::new(n_rows);
    ectx.ensure_scratch(2);
    assert_eq!(ectx.scratch.len(), 2);
    ectx.scratch[0].clear(); // wrong len
    ectx.ensure_scratch(2);
    assert_eq!(ectx.scratch[0].len(), n_rows);

    let mut gctx = GradContext::<f64, 3>::new(n_rows);
    gctx.ensure_scratch(2, 4);
    assert_eq!(gctx.grad_scratch[0].len(), 4 * n_rows);
    gctx.grad_scratch[0].clear();
    gctx.ensure_scratch(2, 4);
    assert_eq!(gctx.grad_scratch[0].len(), 4 * n_rows);
}

#[test]
fn reuse_contexts_hits_cached_plan_paths() {
    let expr = expr_readme_like();
    let (_x_data, x) = make_x(2, 23);
    let x_view = x.view();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    // Eval: reuse EvalContext.
    let mut ectx = EvalContext::<f64, 3>::new(x_view.nrows());
    let mut out0 = vec![0.0f64; x_view.nrows()];
    let mut out1 = vec![0.0f64; x_view.nrows()];
    let ok0 = dynamic_expressions::eval_tree_array_into::<f64, TestOps, 3>(&mut out0, &expr, x_view, &mut ectx, &opts);
    let ok1 = dynamic_expressions::eval_tree_array_into::<f64, TestOps, 3>(&mut out1, &expr, x_view, &mut ectx, &opts);
    assert!(ok0 && ok1);
    assert_eq!(out0, out1);

    // Diff: reuse DiffContext.
    let mut dctx = DiffContext::<f64, 3>::new(x_view.nrows());
    let (_e0, d0, ok0) = eval_diff_tree_array::<f64, TestOps, 3>(&expr, x_view, 0, &mut dctx, &opts);
    let (_e1, d1, ok1) = eval_diff_tree_array::<f64, TestOps, 3>(&expr, x_view, 0, &mut dctx, &opts);
    assert!(ok0 && ok1);
    assert_eq!(d0, d1);

    // Grad: reuse GradContext.
    let mut gctx = GradContext::<f64, 3>::new(x_view.nrows());
    let (_e0, g0, ok0) = eval_grad_tree_array::<f64, TestOps, 3>(&expr, x_view, true, &mut gctx, &opts);
    let (_e1, g1, ok1) = eval_grad_tree_array::<f64, TestOps, 3>(&expr, x_view, true, &mut gctx, &opts);
    assert!(ok0 && ok1);
    assert_eq!(g0.data, g1.data);
}

#[test]
fn nonfinite_root_const_branches_eval_and_diff() {
    let (_x_data, x) = make_x(1, 5);
    let x_view = x.view();

    let expr = c(f64::NAN);

    // Eval: early_exit=false
    let opts = EvalOptions {
        check_finite: true,
        early_exit: false,
    };
    let (out, ok) = eval_tree_array::<f64, TestOps, 3>(&expr, x_view, &opts);
    assert!(!ok);
    assert!(out.iter().all(|v| v.is_nan()));

    // Eval: early_exit=true
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let mut out = vec![123.0f64; x_view.nrows()];
    let mut ectx = EvalContext::<f64, 3>::new(x_view.nrows());
    let ok = dynamic_expressions::eval_tree_array_into::<f64, TestOps, 3>(&mut out, &expr, x_view, &mut ectx, &opts);
    assert!(!ok);
    assert!(out.iter().all(|&v| v == 123.0));

    // Diff: early_exit=false
    let opts = EvalOptions {
        check_finite: true,
        early_exit: false,
    };
    let mut dctx = DiffContext::<f64, 3>::new(x_view.nrows());
    let (e, d, ok) = eval_diff_tree_array::<f64, TestOps, 3>(&expr, x_view, 0, &mut dctx, &opts);
    assert!(!ok);
    assert!(e.iter().all(|v| v.is_nan()));
    assert!(d.iter().all(|&v| v == 0.0));

    // Diff: early_exit=true
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let (e, d, ok) = eval_diff_tree_array::<f64, TestOps, 3>(&expr, x_view, 0, &mut dctx, &opts);
    assert!(!ok);
    assert!(e.iter().all(|v| v.is_nan()));
    assert!(d.iter().all(|v| v.is_nan()));
}
