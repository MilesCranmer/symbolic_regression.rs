use approx::assert_relative_eq;
use dynamic_expressions::utils::ZipEq;
use dynamic_expressions::{
    DiffContext, EvalOptions, GradContext, PNode, PostfixExpr, eval_diff_tree_array, eval_grad_tree_array,
    eval_tree_array,
};
use ndarray::{Array2, ArrayView2};

dynamic_expressions::opset! {
    pub struct TestOps<f64>;
    ops {
        (1, Op1) { Cos, Sin, Exp, Log, Neg, }
        (2, Op2) { Add, Sub, Mul, Div, }
        (3, Op3) { Fma, }
    }
}

#[allow(dead_code)]
pub fn var(feature: u16) -> PostfixExpr<f64, TestOps, 3> {
    PostfixExpr::new(vec![PNode::Var { feature }], vec![], Default::default())
}

#[allow(dead_code)]
pub fn c(value: f64) -> PostfixExpr<f64, TestOps, 3> {
    PostfixExpr::new(vec![PNode::Const { idx: 0 }], vec![value], Default::default())
}

#[allow(dead_code)]
pub fn make_x(n_features: usize, n_rows: usize) -> (Vec<f64>, Array2<f64>) {
    let mut data = vec![0.0f64; n_features * n_rows];
    for feature in 0..n_features {
        for row in 0..n_rows {
            data[feature * n_rows + row] = (row as f64 + 1.0) * (feature as f64 + 1.0) * 0.01;
        }
    }
    let x = Array2::from_shape_vec((n_features, n_rows), data.clone()).unwrap();
    (data, x)
}

#[allow(dead_code)]
pub fn expr_readme_like() -> PostfixExpr<f64, TestOps, 3> {
    // x1 * cos(x2 - 3.2)
    var(0) * dynamic_expressions::operators::cos(var(1) - 3.2)
}

#[allow(dead_code)]
pub fn expr_ternary() -> PostfixExpr<f64, TestOps, 3> {
    // fma(x1, x2, c0) = x1*x2 + c0
    dynamic_expressions::operators::fma(var(0), var(1), c(0.7))
}

#[allow(dead_code)]
pub fn finite_diff_dir(
    expr: &PostfixExpr<f64, TestOps, 3>,
    x_data: &[f64],
    n_features: usize,
    n_rows: usize,
    dir: usize,
    eps: f64,
) -> Vec<f64> {
    let mut plus = x_data.to_vec();
    let mut minus = x_data.to_vec();
    for row in 0..n_rows {
        plus[dir * n_rows + row] += eps;
        minus[dir * n_rows + row] -= eps;
    }
    let x_plus = Array2::from_shape_vec((n_features, n_rows), plus).unwrap();
    let x_minus = Array2::from_shape_vec((n_features, n_rows), minus).unwrap();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let (eplus, okp) = eval_tree_array::<f64, TestOps, 3>(expr, x_plus.view(), &opts);
    let (eminus, okm) = eval_tree_array::<f64, TestOps, 3>(expr, x_minus.view(), &opts);
    assert!(okp && okm);
    eplus
        .iter()
        .zip_eq(&eminus)
        .map(|(a, b)| (a - b) / (2.0 * eps))
        .collect()
}

#[allow(dead_code)]
pub fn assert_close_vec(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (&av, &bv) in a.iter().zip_eq(b) {
        assert_relative_eq!(av, bv, epsilon = tol, max_relative = tol);
    }
}

#[allow(dead_code)]
pub fn grad_matches_diffs(expr: &PostfixExpr<f64, TestOps, 3>, x: &ArrayView2<'_, f64>, variable: bool) {
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let n_rows = x.ncols();
    let n_dir = if variable { x.nrows() } else { expr.consts.len() };

    let mut gctx = GradContext::<f64, 3>::new(n_rows);
    let (_eval, grad, ok) = eval_grad_tree_array::<f64, TestOps, 3>(expr, *x, variable, &mut gctx, &opts);
    assert!(ok);
    assert_eq!(grad.n_dir, n_dir);
    assert_eq!(grad.n_rows, n_rows);

    // Compare each direction to eval_diff (variable) or to basis perturbation for constants via diff isn't supported
    // directly, so for constants we just check gradient is finite and has correct shape.
    if variable {
        for dir in 0..n_dir {
            let mut dctx = DiffContext::<f64, 3>::new(n_rows);
            let (_e, d, okd) = eval_diff_tree_array::<f64, TestOps, 3>(expr, *x, dir, &mut dctx, &opts);
            assert!(okd);
            for (&gv, dv) in grad.data[dir * n_rows..(dir + 1) * n_rows].iter().zip_eq(d) {
                assert_relative_eq!(gv, dv, epsilon = 1e-6, max_relative = 1e-6);
            }
        }
    }
}
