mod common;

use common::*;
use dynamic_expressions::{DiffContext, EvalOptions, GradContext, PNode, eval_diff_tree_array, eval_grad_tree_array};
use ndarray::Array2;

#[test]
fn pnode_constructors_cover_postfix_rs() {
    let v = PNode::Var { feature: 7 };
    assert!(matches!(v, PNode::Var { feature: 7 }));

    let c = PNode::Const { idx: 2 };
    assert!(matches!(c, PNode::Const { idx: 2 }));

    let u = PNode::Op { arity: 1, op: 11 };
    assert!(matches!(u, PNode::Op { arity: 1, op: 11 }));

    let b = PNode::Op { arity: 2, op: 12 };
    assert!(matches!(b, PNode::Op { arity: 2, op: 12 }));

    let o = PNode::Op { arity: 3, op: 13 };
    assert!(matches!(o, PNode::Op { arity: 3, op: 13 }));
}

#[test]
fn postfix_expression_mut_trait_is_usable() {
    fn bump_first_const<E: dynamic_expressions::PostfixExpressionMut<3, Scalar = f64>>(e: &mut E) {
        if let Some(c0) = e.consts_mut().first_mut() {
            *c0 += 1.0;
        }
        e.meta_mut().variable_names.push("x1".to_string());
        let first_is_var = matches!(e.nodes()[0], PNode::Var { .. });
        let nodes = e.nodes_mut();
        assert_eq!(matches!(nodes[0], PNode::Var { .. }), first_is_var);
    }

    let mut expr = expr_readme_like();
    assert_eq!(expr.consts[0], 3.2);
    bump_first_const(&mut expr);
    assert_eq!(expr.consts[0], 4.2);
    assert_eq!(expr.meta.variable_names.len(), 1);
}

#[test]
fn diff_root_var_and_const_branches() {
    let n_rows = 8;
    let x_data = vec![2.0f64; 2 * n_rows];
    let x = Array2::from_shape_vec((2, n_rows), x_data).unwrap();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let expr_var = var(0);
    let mut dctx0 = DiffContext::<f64, 3>::new(x.ncols());
    let (_e, d0, ok0) = eval_diff_tree_array::<f64, TestOps, 3>(&expr_var, x.view(), 0, &mut dctx0, &opts);
    assert!(ok0);
    assert!(d0.iter().all(|&v| v == 1.0));
    let mut dctx1 = DiffContext::<f64, 3>::new(x.ncols());
    let (_e, d1, ok1) = eval_diff_tree_array::<f64, TestOps, 3>(&expr_var, x.view(), 1, &mut dctx1, &opts);
    assert!(ok1);
    assert!(d1.iter().all(|&v| v == 0.0));

    let expr_const = c(1.25);
    let mut dctxc = DiffContext::<f64, 3>::new(x.ncols());
    let (_e, d, ok) = eval_diff_tree_array::<f64, TestOps, 3>(&expr_const, x.view(), 0, &mut dctxc, &opts);
    assert!(ok);
    assert!(d.iter().all(|&v| v == 0.0));
}

#[test]
fn diff_nonfinite_no_early_exit_runs_to_completion() {
    // x1 / 0.0 with early_exit=false should return complete=false but not NaN-filled vectors.
    let n_rows = 6;
    let x_data = vec![1.0f64; n_rows];
    let x = Array2::from_shape_vec((1, n_rows), x_data).unwrap();
    let expr = var(0) / 0.0;
    let opts = EvalOptions {
        check_finite: true,
        early_exit: false,
    };
    let mut dctx = DiffContext::<f64, 3>::new(x.ncols());
    let (e, d, ok) = eval_diff_tree_array::<f64, TestOps, 3>(&expr, x.view(), 0, &mut dctx, &opts);
    assert!(!ok);
    assert!(e.iter().any(|v| !v.is_finite()));
    assert!(d.iter().any(|v| !v.is_finite()));
}

#[test]
fn grad_root_var_and_const_branches() {
    let n_rows = 7;
    let x_data = vec![3.0f64; 2 * n_rows];
    let x = Array2::from_shape_vec((2, n_rows), x_data).unwrap();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    // Root Var: variable=true should have one-hot; variable=false has zero directions (no consts).
    let expr_var = var(1);
    let mut gctx0 = GradContext::<f64, 3>::new(x.ncols());
    let (_e, g, ok) = eval_grad_tree_array::<f64, TestOps, 3>(&expr_var, x.view(), true, &mut gctx0, &opts);
    assert!(ok);
    assert_eq!(g.n_dir, 2);
    assert_eq!(g.data.len(), 2 * x.ncols());
    assert!(g.data[..x.ncols()].iter().all(|&v| v == 0.0));
    assert!(g.data[x.ncols()..2 * x.ncols()].iter().all(|&v| v == 1.0));

    let mut gctx1 = GradContext::<f64, 3>::new(x.ncols());
    let (_e, g, ok) = eval_grad_tree_array::<f64, TestOps, 3>(&expr_var, x.view(), false, &mut gctx1, &opts);
    assert!(ok);
    assert_eq!(g.n_dir, 0);
    assert_eq!(g.data.len(), 0);

    // Root Const: variable=false should have ones; variable=true should have zeros.
    let expr_const = c(2.0);
    let mut gctx2 = GradContext::<f64, 3>::new(x.ncols());
    let (_e, g, ok) = eval_grad_tree_array::<f64, TestOps, 3>(&expr_const, x.view(), false, &mut gctx2, &opts);
    assert!(ok);
    assert_eq!(g.n_dir, 1);
    assert!(g.data.iter().all(|&v| v == 1.0));

    let mut gctx3 = GradContext::<f64, 3>::new(x.ncols());
    let (_e, g, ok) = eval_grad_tree_array::<f64, TestOps, 3>(&expr_const, x.view(), true, &mut gctx3, &opts);
    assert!(ok);
    assert_eq!(g.n_dir, 2);
    assert!(g.data.iter().all(|&v| v == 0.0));
}

#[test]
fn grad_root_const_nonfinite_branches() {
    let n_rows = 4;
    let x = Array2::from_shape_vec((1, n_rows), vec![0.0f64; n_rows]).unwrap();
    let expr = c(f64::NAN);

    // early_exit=false: complete=false but returns the NaN-filled eval and finite gradients (zeros or ones).
    let opts = EvalOptions {
        check_finite: true,
        early_exit: false,
    };
    let mut gctx = GradContext::<f64, 3>::new(x.ncols());
    let (e, g, ok) = eval_grad_tree_array::<f64, TestOps, 3>(&expr, x.view(), false, &mut gctx, &opts);
    assert!(!ok);
    assert!(e.iter().all(|v| v.is_nan()));
    assert!(g.data.iter().all(|&v| v == 1.0));

    // early_exit=true: immediate NaN-filled return.
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let (e, g, ok) = eval_grad_tree_array::<f64, TestOps, 3>(&expr, x.view(), false, &mut gctx, &opts);
    assert!(!ok);
    assert!(e.iter().all(|v| v.is_nan()));
    assert!(g.data.iter().all(|v| v.is_nan()));
}

#[test]
fn grad_nonfinite_no_early_exit_runs_to_completion() {
    // x1 / 0.0 with early_exit=false should return complete=false but not NaN-filled gradients.
    let n_rows = 6;
    let x = Array2::from_shape_vec((1, n_rows), vec![1.0f64; n_rows]).unwrap();
    let expr = var(0) / 0.0;
    let opts = EvalOptions {
        check_finite: true,
        early_exit: false,
    };
    let mut gctx = GradContext::<f64, 3>::new(x.ncols());
    let (e, g, ok) = eval_grad_tree_array::<f64, TestOps, 3>(&expr, x.view(), true, &mut gctx, &opts);
    assert!(!ok);
    assert!(e.iter().any(|v| !v.is_finite()));
    assert!(g.data.iter().any(|v| !v.is_finite()));
}

#[test]
fn div_finite_covers_more_scalar_grad_and_diff_branches() {
    // x1 / 1.0 (finite)
    let n_rows = 10;
    let x = Array2::from_shape_vec((1, n_rows), vec![2.0f64; n_rows]).unwrap();
    let expr = var(0) / 1.0;

    // Grad: variable true/false, early_exit true/false.
    for variable in [true, false] {
        for early_exit in [true, false] {
            let opts = EvalOptions {
                check_finite: true,
                early_exit,
            };
            let mut gctx = GradContext::<f64, 3>::new(x.ncols());
            let (_e, _g, ok) = eval_grad_tree_array::<f64, TestOps, 3>(&expr, x.view(), variable, &mut gctx, &opts);
            assert!(ok);
        }
    }

    // Diff: early_exit true/false.
    for early_exit in [true, false] {
        let opts = EvalOptions {
            check_finite: true,
            early_exit,
        };
        let mut dctx = DiffContext::<f64, 3>::new(x.ncols());
        let (_e, _d, ok) = eval_diff_tree_array::<f64, TestOps, 3>(&expr, x.view(), 0, &mut dctx, &opts);
        assert!(ok);
    }
}
