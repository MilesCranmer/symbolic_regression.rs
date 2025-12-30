use std::hint::black_box;

use dynamic_expressions::dispatch::{GradKernelCtx, GradRef, SrcRef};
use dynamic_expressions::evaluate::kernels::{diff_nary, eval_nary, grad_nary};
use dynamic_expressions::operator_enum::builtin::Add;
use dynamic_expressions::{EvalOptions, Operator};

fn id_eval(args: &[f64; 1]) -> f64 {
    args[0]
}

fn nan_on_second_row_eval(args: &[f64; 1]) -> f64 {
    if args[0] == 2.0 { f64::NAN } else { args[0] }
}

fn one_partial(_args: &[f64; 1], _idx: usize) -> f64 {
    1.0
}

#[derive(Copy, Clone, Debug)]
struct IdUnary;

impl Operator<f64, 1> for IdUnary {
    const NAME: &'static str = "id";

    fn eval(args: &[f64; 1]) -> f64 {
        id_eval(args)
    }

    fn partial(args: &[f64; 1], idx: usize) -> f64 {
        one_partial(args, idx)
    }
}

#[derive(Copy, Clone, Debug)]
struct NanUnary;

impl Operator<f64, 1> for NanUnary {
    const NAME: &'static str = "nan_unary";

    fn eval(args: &[f64; 1]) -> f64 {
        nan_on_second_row_eval(args)
    }

    fn partial(args: &[f64; 1], idx: usize) -> f64 {
        one_partial(args, idx)
    }
}

#[test]
fn eval_nary_no_checks_returns_true() {
    let xs = [1.0, 2.0, 3.0];
    let args = [SrcRef::Slice(&xs)];
    let mut out = [0.0; 3];
    let opts = black_box(EvalOptions {
        check_finite: false,
        early_exit: false,
    });
    assert!(eval_nary::<1, f64, NanUnary>(&mut out, &args, &opts));
}

#[test]
fn eval_nary_checks_after_loop_when_no_early_exit() {
    let xs = [1.0, 2.0, 3.0];
    let args = [SrcRef::Slice(&xs)];
    let mut out = [0.0; 3];
    let opts = black_box(EvalOptions {
        check_finite: true,
        early_exit: false,
    });
    assert!(!eval_nary::<1, f64, NanUnary>(&mut out, &args, &opts));
    assert!(out[1].is_nan());
}

#[test]
fn eval_nary_const_only_check_finite_false_returns_true() {
    let args = [SrcRef::Const(1.0)];
    let mut out = [0.0; 4];
    let opts = black_box(EvalOptions {
        check_finite: false,
        early_exit: true,
    });
    assert!(eval_nary::<1, f64, IdUnary>(&mut out, &args, &opts));
}

#[test]
fn eval_nary_const_only_nonfinite_sets_complete_false() {
    let args = [SrcRef::Const(f64::NAN)];
    let mut out = [0.0; 4];
    let opts = black_box(EvalOptions {
        check_finite: true,
        early_exit: true,
    });
    assert!(!eval_nary::<1, f64, IdUnary>(&mut out, &args, &opts));
}

#[test]
fn eval_nary_checks_after_loop_when_no_early_exit_binary() {
    let xs = [1.0, f64::NAN, 3.0];
    let ys = [1.0, 2.0, 3.0];
    let args = [SrcRef::Slice(&xs), SrcRef::Slice(&ys)];
    let mut out = [0.0; 3];
    let opts = black_box(EvalOptions {
        check_finite: true,
        early_exit: false,
    });
    assert!(!eval_nary::<2, f64, Add>(&mut out, &args, &opts));
    assert!(out[1].is_nan());
}

#[test]
fn eval_nary_const_only_check_finite_false_returns_true_binary() {
    let args = [SrcRef::Const(1.0), SrcRef::Const(2.0)];
    let mut out = [0.0; 3];
    let opts = black_box(EvalOptions {
        check_finite: false,
        early_exit: true,
    });
    assert!(eval_nary::<2, f64, Add>(&mut out, &args, &opts));
}

#[test]
fn diff_nary_checks_after_loop_when_no_early_exit_binary() {
    let xs = [1.0, f64::NAN, 3.0];
    let ys = [1.0, 2.0, 3.0];
    let dxs = [1.0, 1.0, 1.0];
    let dys = [0.0, 0.0, 0.0];
    let args = [SrcRef::Slice(&xs), SrcRef::Slice(&ys)];
    let dargs = [SrcRef::Slice(&dxs), SrcRef::Slice(&dys)];
    let mut out_val = [0.0; 3];
    let mut out_der = [0.0; 3];
    let opts = black_box(EvalOptions {
        check_finite: true,
        early_exit: false,
    });
    assert!(!diff_nary::<2, f64, Add>(
        &mut out_val,
        &mut out_der,
        &args,
        &dargs,
        &opts
    ));
    assert!(out_val[1].is_nan());
}

#[test]
fn diff_nary_no_checks_returns_true() {
    let xs = [1.0, 2.0, 3.0];
    let dxs = [1.0, 1.0, 1.0];
    let args = [SrcRef::Slice(&xs)];
    let dargs = [SrcRef::Slice(&dxs)];
    let mut out_val = [0.0; 3];
    let mut out_der = [0.0; 3];
    let opts = black_box(EvalOptions {
        check_finite: false,
        early_exit: true,
    });
    assert!(diff_nary::<1, f64, NanUnary>(
        &mut out_val,
        &mut out_der,
        &args,
        &dargs,
        &opts
    ));
}

#[test]
fn diff_nary_checks_after_loop_when_no_early_exit() {
    let xs = [1.0, 2.0, 3.0];
    let dxs = [1.0, 1.0, 1.0];
    let args = [SrcRef::Slice(&xs)];
    let dargs = [SrcRef::Slice(&dxs)];
    let mut out_val = [0.0; 3];
    let mut out_der = [0.0; 3];
    let opts = black_box(EvalOptions {
        check_finite: true,
        early_exit: false,
    });
    assert!(!diff_nary::<1, f64, NanUnary>(
        &mut out_val,
        &mut out_der,
        &args,
        &dargs,
        &opts
    ));
    assert!(out_val[1].is_nan());
}

#[test]
fn diff_nary_no_checks_returns_true_binary() {
    let xs = [1.0, f64::NAN, 3.0];
    let ys = [1.0, 2.0, 3.0];
    let dxs = [1.0, 1.0, 1.0];
    let dys = [0.0, 0.0, 0.0];
    let args = [SrcRef::Slice(&xs), SrcRef::Slice(&ys)];
    let dargs = [SrcRef::Slice(&dxs), SrcRef::Slice(&dys)];
    let mut out_val = [0.0; 3];
    let mut out_der = [0.0; 3];
    let opts = black_box(EvalOptions {
        check_finite: false,
        early_exit: true,
    });
    assert!(diff_nary::<2, f64, Add>(
        &mut out_val,
        &mut out_der,
        &args,
        &dargs,
        &opts
    ));
}

#[test]
fn grad_apply_checks_after_loop_when_no_early_exit() {
    let xs = [1.0, f64::NAN];
    let ys = [1.0, 2.0];
    let args = [SrcRef::Slice(&xs), SrcRef::Slice(&ys)];
    let arg_grads = [GradRef::Zero, GradRef::Zero];
    let mut out_val = [0.0; 2];
    let mut out_grad = [0.0; 2];
    let opts = black_box(EvalOptions {
        check_finite: true,
        early_exit: false,
    });
    assert!(!grad_nary::<2, f64, Add>(GradKernelCtx {
        out_val: &mut out_val,
        out_grad: &mut out_grad,
        args: &args,
        arg_grads: &arg_grads,
        n_dir: 1,
        n_rows: 2,
        opts: &opts,
    }));
    assert!(out_val[1].is_nan());
}

#[test]
fn grad_apply_no_checks_returns_true() {
    let xs = [1.0, f64::NAN];
    let ys = [1.0, 2.0];
    let args = [SrcRef::Slice(&xs), SrcRef::Slice(&ys)];
    let arg_grads = [GradRef::Zero, GradRef::Zero];
    let mut out_val = [0.0; 2];
    let mut out_grad = [0.0; 2];
    let opts = black_box(EvalOptions {
        check_finite: false,
        early_exit: true,
    });
    assert!(grad_nary::<2, f64, Add>(GradKernelCtx {
        out_val: &mut out_val,
        out_grad: &mut out_grad,
        args: &args,
        arg_grads: &arg_grads,
        n_dir: 1,
        n_rows: 2,
        opts: &opts,
    }));
}

#[test]
fn grad_nary_no_checks_returns_true() {
    let xs = [1.0, 2.0];
    let args = [SrcRef::Slice(&xs)];
    let arg_grads = [GradRef::Zero];
    let mut out_val = [0.0; 2];
    let mut out_grad = [0.0; 2];
    let opts = black_box(EvalOptions {
        check_finite: false,
        early_exit: true,
    });
    assert!(grad_nary::<1, f64, NanUnary>(GradKernelCtx {
        out_val: &mut out_val,
        out_grad: &mut out_grad,
        args: &args,
        arg_grads: &arg_grads,
        n_dir: 1,
        n_rows: 2,
        opts: &opts,
    }));
}

#[test]
fn grad_nary_checks_after_loop_when_no_early_exit() {
    let xs = [1.0, 2.0, 3.0];
    let args = [SrcRef::Slice(&xs)];
    let arg_grads = [GradRef::Zero];
    let mut out_val = [0.0; 3];
    let mut out_grad = [0.0; 3];
    let opts = black_box(EvalOptions {
        check_finite: true,
        early_exit: false,
    });
    assert!(!grad_nary::<1, f64, NanUnary>(GradKernelCtx {
        out_val: &mut out_val,
        out_grad: &mut out_grad,
        args: &args,
        arg_grads: &arg_grads,
        n_dir: 1,
        n_rows: 3,
        opts: &opts,
    }));
    assert!(out_val[1].is_nan());
}
