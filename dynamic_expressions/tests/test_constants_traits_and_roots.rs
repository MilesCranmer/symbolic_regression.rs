mod common;

use common::*;
use dynamic_expressions::operator_enum::builtin;
use dynamic_expressions::operator_enum::scalar::HasOp;
use dynamic_expressions::{
    DiffContext, EvalContext, EvalOptions, GradContext, PNode, PostfixExpr, PostfixExpression, eval_diff_tree_array,
    eval_grad_tree_array, eval_tree_array,
};
use ndarray::Array2;

#[test]
fn constants_get_set_roundtrip_changes_eval() {
    let mut expr = expr_readme_like();
    let (_x_data, x) = make_x(2, 19);
    let x = x.as_standard_layout().to_owned();
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
    let x = x.as_standard_layout().to_owned();
    let x_view = x.view();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let expr_var = var(1);
    let (outv, okv) = eval_tree_array::<f64, TestOps, 3>(&expr_var, x_view, &opts);
    assert!(okv);
    let n_rows = x_view.ncols();
    let expected: Vec<f64> = (0..n_rows).map(|row| x_data[n_rows + row]).collect();
    assert_close_vec(&outv, &expected, 0.0);

    let expr_const = c(2.5);
    let (outc, okc) = eval_tree_array::<f64, TestOps, 3>(&expr_const, x_view, &opts);
    assert!(okc);
    assert!(outc.iter().all(|&v| v == 2.5));
}

#[test]
fn const_only_operator_fast_path_is_exercised() {
    let (_x_data, x) = make_x(2, 7);
    let x = x.as_standard_layout().to_owned();
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
    let x = Array2::from_shape_vec((1, n_rows), x_data).unwrap();
    let x_view = x.view();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let mut dctx = DiffContext::<f64, 3>::new(x_view.ncols());
    let (eval, der, ok) = eval_diff_tree_array::<f64, TestOps, 3>(&expr, x_view, 0, &mut dctx, &opts);
    assert!(!ok);
    assert!(eval.iter().all(|v| v.is_nan()));
    assert!(der.iter().all(|v| v.is_nan()));

    let mut gctx = GradContext::<f64, 3>::new(x_view.ncols());
    let (eval, grad, ok) = eval_grad_tree_array::<f64, TestOps, 3>(&expr, x_view, true, &mut gctx, &opts);
    assert!(!ok);
    assert!(eval.iter().all(|v| v.is_nan()));
    assert!(grad.data.iter().all(|v| v.is_nan()));
}

#[test]
fn contexts_resize_and_reuse_scratch() {
    let n_rows = 5;
    let mut ectx = EvalContext::<f64, 3>::new(n_rows);
    let expr = expr_readme_like();
    let (_x_data, x) = make_x(2, n_rows);
    let x_view = x.view();
    ectx.setup(&expr, x_view);
    let plan_slots = ectx.plan.as_ref().unwrap().n_slots;
    let scratch = ectx.scratch.as_mut().expect("scratch allocated");
    assert_eq!(scratch.nrows(), plan_slots);
    scratch.as_slice_mut().expect("scratch contiguous").fill(0.0);
    ectx.setup(&expr, x_view);
    let scratch = ectx.scratch.as_ref().expect("scratch allocated");
    assert_eq!(scratch.ncols(), n_rows);

    let mut gctx = GradContext::<f64, 3>::new(n_rows);
    gctx.ensure_scratch(2, 4);
    assert_eq!(gctx.grad_scratch.nrows(), 2);
    assert_eq!(gctx.grad_scratch.ncols(), 4 * n_rows);
    gctx.grad_scratch
        .as_slice_mut()
        .expect("grad scratch contiguous")
        .fill(0.0);
    gctx.ensure_scratch(2, 4);
    assert_eq!(gctx.grad_scratch.ncols(), 4 * n_rows);
}

#[test]
fn reuse_contexts_hits_cached_plan_paths() {
    let expr = expr_readme_like();
    let (_x_data, x) = make_x(2, 23);
    let x = x.as_standard_layout().to_owned();
    let x_view = x.view();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    // Eval: reuse EvalContext.
    let mut ectx = EvalContext::<f64, 3>::new(x_view.ncols());
    let mut out0 = vec![0.0f64; x_view.ncols()];
    let mut out1 = vec![0.0f64; x_view.ncols()];
    let ok0 = dynamic_expressions::eval_tree_array_into::<f64, TestOps, 3>(&mut out0, &expr, x_view, &mut ectx, &opts);
    let ok1 = dynamic_expressions::eval_tree_array_into::<f64, TestOps, 3>(&mut out1, &expr, x_view, &mut ectx, &opts);
    assert!(ok0 && ok1);
    assert_eq!(out0, out1);

    // Diff: reuse DiffContext.
    let mut dctx = DiffContext::<f64, 3>::new(x_view.ncols());
    let (_e0, d0, ok0) = eval_diff_tree_array::<f64, TestOps, 3>(&expr, x_view, 0, &mut dctx, &opts);
    let (_e1, d1, ok1) = eval_diff_tree_array::<f64, TestOps, 3>(&expr, x_view, 0, &mut dctx, &opts);
    assert!(ok0 && ok1);
    assert_eq!(d0, d1);
}

#[test]
fn diff_context_reuse_recompiles_on_expr_change_same_len() {
    let expr_add = var(0) + var(1);
    let expr_mul = var(0) * var(1);
    let (_x_data, x) = make_x(2, 11);
    let x = x.as_standard_layout().to_owned();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let mut dctx = DiffContext::<f64, 3>::new(x.ncols());
    let (_e0, _d0, ok0) = eval_diff_tree_array::<f64, TestOps, 3>(&expr_add, x.view(), 0, &mut dctx, &opts);
    assert!(ok0);

    let (eval_reuse, diff_reuse, ok1) =
        eval_diff_tree_array::<f64, TestOps, 3>(&expr_mul, x.view(), 0, &mut dctx, &opts);
    assert!(ok1);

    let mut fresh = DiffContext::<f64, 3>::new(x.ncols());
    let (eval_fresh, diff_fresh, ok_fresh) =
        eval_diff_tree_array::<f64, TestOps, 3>(&expr_mul, x.view(), 0, &mut fresh, &opts);
    assert!(ok_fresh);

    assert_close_vec(&eval_reuse, &eval_fresh, 1e-12);
    assert_close_vec(&diff_reuse, &diff_fresh, 1e-12);
}

#[test]
fn grad_context_reuse_with_new_inputs_matches_fresh() {
    let expr = expr_readme_like();
    let (_x0, x0) = make_x(2, 23);
    let mut x1 = x0.to_owned();
    x1.map_inplace(|v| *v += 0.123);
    let x0 = x0.as_standard_layout().to_owned();
    let x1 = x1.as_standard_layout().to_owned();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let mut gctx = GradContext::<f64, 3>::new(x0.ncols());
    let (_e0, _g0, ok0) = eval_grad_tree_array::<f64, TestOps, 3>(&expr, x0.view(), true, &mut gctx, &opts);
    assert!(ok0);

    let (eval_reuse, grad_reuse, ok1) =
        eval_grad_tree_array::<f64, TestOps, 3>(&expr, x1.view(), true, &mut gctx, &opts);
    assert!(ok1);

    let mut fresh = GradContext::<f64, 3>::new(x1.ncols());
    let (eval_fresh, grad_fresh, ok_fresh) =
        eval_grad_tree_array::<f64, TestOps, 3>(&expr, x1.view(), true, &mut fresh, &opts);
    assert!(ok_fresh);

    assert_close_vec(&eval_reuse, &eval_fresh, 1e-12);
    assert_close_vec(&grad_reuse.data, &grad_fresh.data, 1e-12);
}

#[test]
fn grad_context_reuse_recompiles_on_expr_change_same_len() {
    let expr_add = var(0) + var(1);
    let expr_mul = var(0) * var(1);
    let (_x_data, x) = make_x(2, 11);
    let x = x.as_standard_layout().to_owned();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let mut gctx = GradContext::<f64, 3>::new(x.ncols());
    let (_e0, _g0, ok0) = eval_grad_tree_array::<f64, TestOps, 3>(&expr_add, x.view(), true, &mut gctx, &opts);
    assert!(ok0);

    let (eval_reuse, grad_reuse, ok1) =
        eval_grad_tree_array::<f64, TestOps, 3>(&expr_mul, x.view(), true, &mut gctx, &opts);
    assert!(ok1);

    let mut fresh = GradContext::<f64, 3>::new(x.ncols());
    let (eval_fresh, grad_fresh, ok_fresh) =
        eval_grad_tree_array::<f64, TestOps, 3>(&expr_mul, x.view(), true, &mut fresh, &opts);
    assert!(ok_fresh);

    assert_close_vec(&eval_reuse, &eval_fresh, 1e-12);
    assert_close_vec(&grad_reuse.data, &grad_fresh.data, 1e-12);
}

#[test]
fn grad_context_reuse_switches_variable_flag_with_equal_dir() {
    let add_op = <TestOps as HasOp<builtin::Add, 2>>::ID;
    let expr = PostfixExpr::<f64, TestOps, 3>::new(
        vec![
            PNode::Var { feature: 0 },
            PNode::Const { idx: 1 },
            PNode::Op { arity: 2, op: add_op },
        ],
        vec![0.25, 0.75],
        Default::default(),
    );

    let (_x_data, x) = make_x(2, 9);
    let x = x.as_standard_layout().to_owned();
    let n_rows = x.ncols();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let mut gctx = GradContext::<f64, 3>::new(n_rows);
    let (_e_var, grad_var, ok_var) = eval_grad_tree_array::<f64, TestOps, 3>(&expr, x.view(), true, &mut gctx, &opts);
    assert!(ok_var);
    let (_e_const, grad_const, ok_const) =
        eval_grad_tree_array::<f64, TestOps, 3>(&expr, x.view(), false, &mut gctx, &opts);
    assert!(ok_const);

    let ones = vec![1.0; n_rows];
    let zeros = vec![0.0; n_rows];
    assert_close_vec(&grad_var.data[..n_rows], &ones, 1e-12);
    assert_close_vec(&grad_var.data[n_rows..(2 * n_rows)], &zeros, 1e-12);
    assert_close_vec(&grad_const.data[..n_rows], &zeros, 1e-12);
    assert_close_vec(&grad_const.data[n_rows..(2 * n_rows)], &ones, 1e-12);
}

#[test]
fn nonfinite_root_const_branches_eval_and_diff() {
    let (_x_data, x) = make_x(1, 5);
    let x = x.as_standard_layout().to_owned();
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
    let mut out = vec![123.0f64; x_view.ncols()];
    let mut ectx = EvalContext::<f64, 3>::new(x_view.ncols());
    let ok = dynamic_expressions::eval_tree_array_into::<f64, TestOps, 3>(&mut out, &expr, x_view, &mut ectx, &opts);
    assert!(!ok);
    assert!(out.iter().all(|&v| v == 123.0));

    // Diff: early_exit=false
    let opts = EvalOptions {
        check_finite: true,
        early_exit: false,
    };
    let mut dctx = DiffContext::<f64, 3>::new(x_view.ncols());
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
