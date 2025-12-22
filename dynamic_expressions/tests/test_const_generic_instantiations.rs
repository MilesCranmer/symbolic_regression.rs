mod common;

use common::TestOps;
use dynamic_expressions::math::cos;
use dynamic_expressions::{
    eval_diff_tree_array, eval_grad_tree_array, eval_tree_array, DiffContext, EvalOptions,
    GradContext, PNode, PostfixExpr,
};
use ndarray::Array2;

fn var<const D: usize>(feature: u16) -> PostfixExpr<f64, TestOps, D> {
    PostfixExpr::new(vec![PNode::Var { feature }], vec![], Default::default())
}

fn make_x_static(n_features: usize, n_rows: usize) -> Array2<f64> {
    let mut data = vec![0.0f64; n_features * n_rows];
    for row in 0..n_rows {
        for feature in 0..n_features {
            data[row * n_features + feature] = (row as f64 + 1.0) * (feature as f64 + 1.0) * 0.01;
        }
    }
    Array2::from_shape_vec((n_rows, n_features), data).unwrap()
}

#[test]
fn eval_diff_grad_work_for_d2() {
    let x = make_x_static(2, 32);
    let expr: PostfixExpr<f64, TestOps, 2> = var::<2>(0) * cos(var::<2>(1) - 3.2);

    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let (_eval, ok) = eval_tree_array::<f64, TestOps, 2>(&expr, x.view(), &opts);
    assert!(ok);

    let mut dctx = DiffContext::<f64, 2>::new(x.nrows());
    let (_e, _d, okd) =
        eval_diff_tree_array::<f64, TestOps, 2>(&expr, x.view(), 0, &mut dctx, &opts);
    assert!(okd);

    let mut gctx = GradContext::<f64, 2>::new(x.nrows());
    let (_e, g, okg) =
        eval_grad_tree_array::<f64, TestOps, 2>(&expr, x.view(), true, &mut gctx, &opts);
    assert!(okg);
    assert_eq!(g.n_dir, x.ncols());
    assert_eq!(g.n_rows, x.nrows());
}

#[test]
fn eval_diff_grad_work_for_d1_unary_only() {
    let x = make_x_static(1, 32);
    let expr: PostfixExpr<f64, TestOps, 1> = cos(var::<1>(0));

    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let (_eval, ok) = eval_tree_array::<f64, TestOps, 1>(&expr, x.view(), &opts);
    assert!(ok);

    let mut dctx = DiffContext::<f64, 1>::new(x.nrows());
    let (_e, _d, okd) =
        eval_diff_tree_array::<f64, TestOps, 1>(&expr, x.view(), 0, &mut dctx, &opts);
    assert!(okd);

    let mut gctx = GradContext::<f64, 1>::new(x.nrows());
    let (_e, g, okg) =
        eval_grad_tree_array::<f64, TestOps, 1>(&expr, x.view(), true, &mut gctx, &opts);
    assert!(okg);
    assert_eq!(g.n_dir, x.ncols());
    assert_eq!(g.n_rows, x.nrows());
}

#[test]
fn eval_early_exit_path_is_instantiated_for_d2() {
    let x = make_x_static(1, 8);
    let nan_const: PostfixExpr<f64, TestOps, 2> = PostfixExpr::new(
        vec![PNode::Const { idx: 0 }],
        vec![f64::NAN],
        Default::default(),
    );
    let expr: PostfixExpr<f64, TestOps, 2> = cos(nan_const);

    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let (_eval, ok) = eval_tree_array::<f64, TestOps, 2>(&expr, x.view(), &opts);
    assert!(!ok);
}

#[test]
fn eval_early_exit_path_is_instantiated_for_d1() {
    let x = make_x_static(1, 8);
    let nan_const: PostfixExpr<f64, TestOps, 1> = PostfixExpr::new(
        vec![PNode::Const { idx: 0 }],
        vec![f64::NAN],
        Default::default(),
    );
    let expr: PostfixExpr<f64, TestOps, 1> = cos(nan_const);

    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let (_eval, ok) = eval_tree_array::<f64, TestOps, 1>(&expr, x.view(), &opts);
    assert!(!ok);
}
