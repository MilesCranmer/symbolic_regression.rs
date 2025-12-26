use dynamic_expressions::dispatch::{GradKernelCtx, GradRef, SrcRef};
use dynamic_expressions::evaluate::kernels::{diff_nary, grad_nary};
use dynamic_expressions::{EvalOptions, Operator};

fn nan_eval_1(args: &[f64; 1]) -> f64 {
    let _ = args;
    f64::NAN
}

fn zero_partial_1(args: &[f64; 1], idx: usize) -> f64 {
    let _ = args;
    let _ = idx;
    0.0
}

#[derive(Copy, Clone, Debug)]
struct NanUnary;

impl Operator<f64, 1> for NanUnary {
    const NAME: &'static str = "nan_unary";

    fn eval(args: &[f64; 1]) -> f64 {
        nan_eval_1(args)
    }

    fn partial(args: &[f64; 1], idx: usize) -> f64 {
        zero_partial_1(args, idx)
    }
}

#[test]
fn diff_nary_early_exit_writes_outputs() {
    let mut out_val = [0.0f64];
    let mut out_der = [0.0f64];
    let args = [SrcRef::Const(0.0)];
    let dargs = [SrcRef::Const(1.0)];
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let ok = diff_nary::<1, f64, NanUnary>(&mut out_val, &mut out_der, &args, &dargs, &opts);
    assert!(!ok);
    assert!(out_val[0].is_nan());
}

#[test]
fn grad_nary_early_exit_writes_outputs() {
    let mut out_val = [0.0f64];
    let mut out_grad = [0.0f64];
    let args = [SrcRef::Const(0.0)];
    let arg_grads = [GradRef::Zero];
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let ok = grad_nary::<1, f64, NanUnary>(GradKernelCtx {
        out_val: &mut out_val,
        out_grad: &mut out_grad,
        args: &args,
        arg_grads: &arg_grads,
        n_dir: 1,
        n_rows: 1,
        opts: &opts,
    });
    assert!(!ok);
    assert!(out_val[0].is_nan());
    assert!(out_grad[0].is_nan());
}
