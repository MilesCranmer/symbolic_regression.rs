mod common;

use common::*;
use dynamic_expressions::{EvalOptions, eval_tree_array};
use rstest::rstest;

#[rstest]
#[case(true, true)]
#[case(true, false)]
#[case(false, false)]
fn eval_readme_like_matches_manual(#[case] check_finite: bool, #[case] early_exit: bool) {
    let expr = expr_readme_like();
    let (x_data, x) = make_x(2, 100);
    let opts = EvalOptions {
        check_finite,
        early_exit,
    };

    let n_rows = x.nrows();
    let n_features = x.ncols();
    let mut out = vec![0.0f64; n_rows];
    let mut ctx = dynamic_expressions::EvalContext::<f64, 3>::new(n_rows);
    let ok = dynamic_expressions::eval_tree_array_into::<f64, TestOps, 3>(&mut out, &expr, x.view(), &mut ctx, &opts);

    // manual
    let mut manual = vec![0.0f64; n_rows];
    for row in 0..n_rows {
        let x1 = x_data[row * n_features];
        let x2 = x_data[row * n_features + 1];
        manual[row] = x1 * (x2 - 3.2).cos();
    }
    if check_finite {
        assert!(ok);
    }
    assert_close_vec(&out, &manual, 1e-12);
}

#[test]
fn eval_constant_only_fill() {
    let expr = c(1.25);
    let (_x_data, x) = make_x(2, 17);
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let (out, ok) = eval_tree_array::<f64, TestOps, 3>(&expr, x.view(), &opts);
    assert!(ok);
    for v in out {
        assert_eq!(v, 1.25);
    }
}

#[test]
fn nan_detection_sets_complete_false() {
    // x1 / 0.0
    let expr = var(0) / 0.0;
    let (_x_data, x) = make_x(1, 10);
    let opts = EvalOptions {
        check_finite: true,
        early_exit: false,
    };
    let (_out, ok) = eval_tree_array::<f64, TestOps, 3>(&expr, x.view(), &opts);
    assert!(!ok);
}
