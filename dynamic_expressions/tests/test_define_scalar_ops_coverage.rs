use dynamic_expressions::operator_enum::scalar::{
    DiffKernelCtx, EvalKernelCtx, GradKernelCtx, OpId, ScalarOpSet, SrcRef,
};
use dynamic_expressions::strings::OpNames;
use dynamic_expressions::{
    eval_diff_tree_array, eval_grad_tree_array, eval_tree_array, EvalOptions, PNode, PostfixExpr,
};
use ndarray::Array2;

fn cos_eval(args: &[f64; 1]) -> f64 {
    args[0].cos()
}
fn cos_partial(args: &[f64; 1], idx: usize) -> f64 {
    match idx {
        0 => -args[0].sin(),
        _ => unreachable!(),
    }
}

fn add_eval(args: &[f64; 2]) -> f64 {
    args[0] + args[1]
}
fn add_partial(_args: &[f64; 2], idx: usize) -> f64 {
    match idx {
        0 | 1 => 1.0,
        _ => unreachable!(),
    }
}

fn sub_eval(args: &[f64; 2]) -> f64 {
    args[0] - args[1]
}
fn sub_partial(_args: &[f64; 2], idx: usize) -> f64 {
    match idx {
        0 => 1.0,
        1 => -1.0,
        _ => unreachable!(),
    }
}

fn mul_eval(args: &[f64; 2]) -> f64 {
    args[0] * args[1]
}
fn mul_partial(args: &[f64; 2], idx: usize) -> f64 {
    match idx {
        0 => args[1],
        1 => args[0],
        _ => unreachable!(),
    }
}

dynamic_expressions::define_scalar_ops! {
    pub struct FnOps<f64>;
    ops {
        (1, Op1) { Cos => (cos_eval, cos_partial), }
        (2, Op2) {
            Add => (add_eval, add_partial),
            Sub => (sub_eval, sub_partial),
            Mul => (mul_eval, mul_partial),
        }
    }
}

fn var(feature: u16) -> PostfixExpr<f64, FnOps, 2> {
    PostfixExpr::new(vec![PNode::Var { feature }], vec![], Default::default())
}

fn cos_expr(x: PostfixExpr<f64, FnOps, 2>) -> PostfixExpr<f64, FnOps, 2> {
    dynamic_expressions::expression_algebra::__apply_postfix::<f64, FnOps, 2, 1>(
        Op1::Cos as u16,
        [x],
    )
}

fn c(value: f64) -> PostfixExpr<f64, FnOps, 2> {
    PostfixExpr::new(
        vec![PNode::Const { idx: 0 }],
        vec![value],
        Default::default(),
    )
}

#[test]
fn define_scalar_ops_end_to_end_paths_are_covered() {
    // Expression: x1 * cos(x2 - 3.2)
    let expr = var(0) * cos_expr(var(1) - 3.2);

    let n_features = 2usize;
    let n_rows = 100usize;
    let mut data = vec![0.0f64; n_features * n_rows];
    for row in 0..n_rows {
        for feature in 0..n_features {
            data[row * n_features + feature] = (row as f64 + 1.0) * (feature as f64 + 1.0) * 0.001;
        }
    }
    let x = Array2::from_shape_vec((n_rows, n_features), data).unwrap();

    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };

    let (y, ok) = eval_tree_array::<f64, FnOps, 2>(&expr, x.view(), &opts);
    assert!(ok);
    assert_eq!(y.len(), n_rows);

    let mut dctx = dynamic_expressions::DiffContext::<f64, 2>::new(n_rows);
    let (_e, d, ok) = eval_diff_tree_array::<f64, FnOps, 2>(&expr, x.view(), 0, &mut dctx, &opts);
    assert!(ok);
    assert_eq!(d.len(), n_rows);

    let mut gctx = dynamic_expressions::GradContext::<f64, 2>::new(n_rows);
    let (_e, g, ok) =
        eval_grad_tree_array::<f64, FnOps, 2>(&expr, x.view(), true, &mut gctx, &opts);
    assert!(ok);
    assert_eq!(g.n_dir, n_features);
    assert_eq!(g.n_rows, n_rows);
}

#[test]
fn define_scalar_ops_constant_only_fast_path_is_exercised() {
    let expr = cos_expr(c(0.0));
    let x_data = vec![0.0f64; 2];
    let x = Array2::from_shape_vec((2, 1), x_data).unwrap();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let (y, ok) = eval_tree_array::<f64, FnOps, 2>(&expr, x.view(), &opts);
    assert!(ok);
    assert!(y.iter().all(|&v| v == 1.0));
}

#[test]
fn define_scalar_ops_constant_only_nonfinite_sets_complete_false() {
    let expr = cos_expr(c(f64::NAN));
    let x_data = vec![0.0f64; 2];
    let x = Array2::from_shape_vec((2, 1), x_data).unwrap();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: false,
    };
    let (_y, ok) = eval_tree_array::<f64, FnOps, 2>(&expr, x.view(), &opts);
    assert!(!ok);
}

#[test]
fn define_scalar_ops_nonfinite_early_exit_true_returns_false() {
    // This hits the non-const path of eval_nary and returns early from the row loop.
    let expr = var(0) * f64::NAN;
    let x_data = vec![1.0f64; 4];
    let x = Array2::from_shape_vec((4, 1), x_data).unwrap();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let (_y, ok) = eval_tree_array::<f64, FnOps, 2>(&expr, x.view(), &opts);
    assert!(!ok);
}

#[test]
fn define_scalar_ops_nonfinite_early_exit_true_constant_only_returns_false() {
    let expr = cos_expr(c(f64::NAN));
    let x_data = vec![0.0f64; 2];
    let x = Array2::from_shape_vec((2, 1), x_data).unwrap();
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let (_y, ok) = eval_tree_array::<f64, FnOps, 2>(&expr, x.view(), &opts);
    assert!(!ok);
}

#[test]
#[should_panic]
fn define_scalar_ops_eval_panics_on_unknown_id() {
    let mut out = [0.0f64];
    let args = [SrcRef::Const(0.0)];
    let opts = EvalOptions::default();
    <FnOps as ScalarOpSet<f64>>::eval(
        OpId { arity: 1, id: 999 },
        EvalKernelCtx {
            out: &mut out,
            args: &args,
            opts: &opts,
        },
    );
}

#[test]
#[should_panic]
fn define_scalar_ops_eval_panics_on_unsupported_arity() {
    let mut out = [0.0f64];
    let args = [SrcRef::Const(0.0)];
    let opts = EvalOptions::default();
    <FnOps as ScalarOpSet<f64>>::eval(
        OpId { arity: 9, id: 0 },
        EvalKernelCtx {
            out: &mut out,
            args: &args,
            opts: &opts,
        },
    );
}

#[test]
#[should_panic]
fn define_scalar_ops_diff_panics_on_unknown_id() {
    let mut out_val = [0.0f64];
    let mut out_der = [0.0f64];
    let args = [SrcRef::Const(0.0)];
    let dargs = [SrcRef::Const(0.0)];
    let opts = EvalOptions::default();
    <FnOps as ScalarOpSet<f64>>::diff(
        OpId { arity: 1, id: 999 },
        DiffKernelCtx {
            out_val: &mut out_val,
            out_der: &mut out_der,
            args: &args,
            dargs: &dargs,
            opts: &opts,
        },
    );
}

#[test]
#[should_panic]
fn define_scalar_ops_grad_panics_on_unknown_id() {
    let mut out_val = [0.0f64];
    let mut out_grad = [0.0f64];
    let args = [SrcRef::Const(0.0)];
    let arg_grads = [dynamic_expressions::operator_enum::scalar::GradRef::Zero];
    let opts = EvalOptions::default();
    <FnOps as ScalarOpSet<f64>>::grad(
        OpId { arity: 1, id: 999 },
        GradKernelCtx {
            out_val: &mut out_val,
            out_grad: &mut out_grad,
            args: &args,
            arg_grads: &arg_grads,
            n_dir: 1,
            n_rows: 1,
            opts: &opts,
        },
    );
}

#[test]
#[should_panic]
fn define_scalar_ops_diff_panics_on_unsupported_arity() {
    let mut out_val = [0.0f64];
    let mut out_der = [0.0f64];
    let args = [SrcRef::Const(0.0)];
    let dargs = [SrcRef::Const(0.0)];
    let opts = EvalOptions::default();
    <FnOps as ScalarOpSet<f64>>::diff(
        OpId { arity: 9, id: 0 },
        DiffKernelCtx {
            out_val: &mut out_val,
            out_der: &mut out_der,
            args: &args,
            dargs: &dargs,
            opts: &opts,
        },
    );
}

#[test]
#[should_panic]
fn define_scalar_ops_grad_panics_on_unsupported_arity() {
    let mut out_val = [0.0f64];
    let mut out_grad = [0.0f64];
    let args = [SrcRef::Const(0.0)];
    let arg_grads = [dynamic_expressions::operator_enum::scalar::GradRef::Zero];
    let opts = EvalOptions::default();
    <FnOps as ScalarOpSet<f64>>::grad(
        OpId { arity: 9, id: 0 },
        GradKernelCtx {
            out_val: &mut out_val,
            out_grad: &mut out_grad,
            args: &args,
            arg_grads: &arg_grads,
            n_dir: 1,
            n_rows: 1,
            opts: &opts,
        },
    );
}

#[test]
fn define_scalar_ops_op_names_unknown_paths_are_exercised() {
    assert_eq!(
        <FnOps as OpNames>::op_name(OpId { arity: 1, id: 999 }),
        "unknown_op"
    );
    assert_eq!(
        <FnOps as OpNames>::op_name(OpId { arity: 9, id: 0 }),
        "unknown_op"
    );
}
