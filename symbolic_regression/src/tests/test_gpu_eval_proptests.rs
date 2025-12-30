use std::cell::RefCell;

use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF32;
use dynamic_expressions::utils::ZipEq;
use dynamic_expressions::{EvalOptions, OperatorSet, proptest_utils};
use ndarray::{Array1, Array2};
use proptest::prelude::*;

use crate::Dataset;
use crate::gpu::{GpuClient, MAX_CONSTS, pack_expr};
use crate::pop_member::Evaluator;

const D: usize = 3;

fn ci_requires_gpu() -> bool {
    std::env::var("SYMBOLIC_REGRESSION_GPU_TEST_REQUIRED")
        .ok()
        .is_some_and(|v| v != "0")
}

fn make_dataset(n_features: usize, n_rows: usize) -> Dataset<f32> {
    let mut rng = fastrand::Rng::with_seed(0x4d1e_055e_2f77_39d1);

    let mut x = Array2::<f32>::zeros((n_features, n_rows));
    let mut y = Array1::<f32>::zeros(n_rows);
    let mut w = Array1::<f32>::zeros(n_rows);

    for row in 0..n_rows {
        for feat in 0..n_features {
            // Keep values small to reduce overflow/domain issues.
            x[(feat, row)] = rng.f32() * 2.0 - 1.0;
        }
        y[row] = rng.f32() * 2.0 - 1.0;
        w[row] = rng.f32() + 0.5;
    }

    Dataset::with_weights_and_names(x, y, Some(w), Vec::new())
}

fn proptest_cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&v| v >= 1)
        .unwrap_or(default)
}

#[test]
fn gpu_mse_matches_cpu_proptest() {
    let n_features = 4;
    let n_rows = 256;
    let dataset = make_dataset(n_features, n_rows);
    let loss_fn = crate::loss_functions::mse::<f32>();

    let gpu = match GpuClient::spawn(&dataset, 512) {
        Ok(g) => g,
        Err(err) => {
            if ci_requires_gpu() {
                panic!("GPU required but init failed: {err:?}");
            }
            return;
        }
    };

    let evaluator = RefCell::new(Evaluator::<f32, D>::new(n_rows));

    let unary_ops = [
        BuiltinOpsF32::lookup("neg").unwrap().id,
        BuiltinOpsF32::lookup("sin").unwrap().id,
        BuiltinOpsF32::lookup("cos").unwrap().id,
        BuiltinOpsF32::lookup("exp").unwrap().id,
    ];
    let binary_ops = [
        BuiltinOpsF32::lookup("add").unwrap().id,
        BuiltinOpsF32::lookup("sub").unwrap().id,
        BuiltinOpsF32::lookup("mul").unwrap().id,
    ];

    let n_consts = 4usize;
    let nodes = proptest_utils::arb_postfix_nodes(
        n_features,
        n_consts,
        unary_ops.to_vec(),
        binary_ops.to_vec(),
        Vec::new(),
        6,
        32,
        6,
    );
    let consts = prop::collection::vec(-1.0f32..1.0f32, n_consts);

    proptest!(ProptestConfig::with_cases(proptest_cases(64)), |(nodes in nodes, consts in consts)| {
        let expr = PostfixExpr::<f32, BuiltinOpsF32, D>::new(nodes, consts, Metadata::default());
        let packed = pack_expr(&expr).expect("expr should be packable");

        let cpu_loss = {
            let plan = dynamic_expressions::compile_plan(&expr.nodes, n_features, expr.consts.len());
            let mut evaluator = evaluator.borrow_mut();
            let Evaluator {
                eval_opts,
                yhat,
                scratch,
            } = &mut *evaluator;
            let ok = dynamic_expressions::eval_plan_array_into(
                yhat,
                &plan,
                &expr,
                dataset.x.view(),
                scratch,
                eval_opts,
            );
            if !ok || yhat.iter().any(|v| !v.is_finite()) {
                None
            } else {
                let loss = loss_fn.loss(yhat, dataset.y_slice(), dataset.weights_slice());
                loss.is_finite().then_some(loss)
            }
        };

        let gpu_loss = gpu.eval_mse(packed);

        match cpu_loss {
            None => prop_assert!(!gpu_loss.is_finite()),
            Some(cpu_loss) => {
                prop_assert!(gpu_loss.is_finite());
                let denom = cpu_loss.abs().max(1e-6);
                let rel = (gpu_loss - cpu_loss).abs() / denom;
                prop_assert!(rel < 1e-4, "cpu={cpu_loss} gpu={gpu_loss} rel={rel}");
            }
        }
    });
}

#[test]
fn gpu_mse_grad_matches_cpu_proptest() {
    let n_features = 3;
    let n_rows = 256;
    let dataset = make_dataset(n_features, n_rows);
    let loss_fn = crate::loss_functions::mse::<f32>();

    let gpu = match GpuClient::spawn(&dataset, 512) {
        Ok(g) => g,
        Err(err) => {
            if ci_requires_gpu() {
                panic!("GPU required but init failed: {err:?}");
            }
            return;
        }
    };

    let eval_opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let grad_ctx = RefCell::new(dynamic_expressions::GradContext::<f32, D>::new(n_rows));
    let dloss_dyhat = RefCell::new(vec![0.0f32; n_rows]);

    let unary_ops = [
        BuiltinOpsF32::lookup("neg").unwrap().id,
        BuiltinOpsF32::lookup("sin").unwrap().id,
        BuiltinOpsF32::lookup("cos").unwrap().id,
        BuiltinOpsF32::lookup("exp").unwrap().id,
    ];
    let binary_ops = [
        BuiltinOpsF32::lookup("add").unwrap().id,
        BuiltinOpsF32::lookup("sub").unwrap().id,
        BuiltinOpsF32::lookup("mul").unwrap().id,
    ];

    let n_consts = 4usize;
    let nodes = proptest_utils::arb_postfix_nodes(
        n_features,
        n_consts,
        unary_ops.to_vec(),
        binary_ops.to_vec(),
        Vec::new(),
        6,
        32,
        6,
    )
    .prop_filter("must contain at least one const", |nodes| {
        nodes.iter().any(|n| matches!(n, PNode::Const { .. }))
    });
    let consts = prop::collection::vec(-1.0f32..1.0f32, n_consts);

    proptest!(ProptestConfig::with_cases(proptest_cases(64)), |(nodes in nodes, consts in consts)| {
        let expr = PostfixExpr::<f32, BuiltinOpsF32, D>::new(nodes, consts, Metadata::default());
        let packed = pack_expr(&expr).expect("expr should be packable");

        let (cpu_loss, cpu_grad) = {
            let mut grad_ctx = grad_ctx.borrow_mut();
            let (yhat, jac, ok) = dynamic_expressions::eval_grad_tree_array(&expr, dataset.x.view(), false, &mut grad_ctx, &eval_opts);
            if !ok || yhat.iter().any(|v| !v.is_finite()) {
                (None, [f32::NAN; MAX_CONSTS])
            } else {
                let y = dataset.y_slice();
                let w = dataset.weights_slice();
                let loss = loss_fn.loss(&yhat, y, w);
                if !loss.is_finite() {
                    (None, [f32::NAN; MAX_CONSTS])
                } else {
                    let mut dloss_dyhat = dloss_dyhat.borrow_mut();
                    loss_fn.dloss_dyhat(&yhat, y, w, &mut dloss_dyhat);

                    let mut g = [0.0f32; MAX_CONSTS];
                    for (ci, gout) in g
                        .iter_mut()
                        .enumerate()
                        .take(n_consts.min(MAX_CONSTS))
                    {
                        let base = ci * n_rows;
                        let acc = dloss_dyhat
                            .iter()
                            .copied()
                            .zip_eq(jac.data[base..base + n_rows].iter().copied())
                            .fold(0.0f32, |a, (dl, dc)| a + dl * dc);
                        *gout = acc;
                    }
                    (Some(loss), g)
                }
            }
        };

        let res = gpu.eval_mse_grad(packed);

        match cpu_loss {
            None => prop_assert!(!res.loss.is_finite()),
            Some(cpu_loss) => {
                prop_assert!(res.loss.is_finite());
                let denom = cpu_loss.abs().max(1e-6);
                let rel = (res.loss - cpu_loss).abs() / denom;
                prop_assert!(rel < 1e-4, "cpu={cpu_loss} gpu={gpu_loss} rel={rel}", gpu_loss=res.loss);

                for (i, (&a, &b)) in cpu_grad
                    .iter()
                    .zip(res.grad.iter())
                    .enumerate()
                    .take(n_consts.min(MAX_CONSTS))
                {
                    prop_assert!(a.is_finite() && b.is_finite());
                    let denom = a.abs().max(1e-5);
                    let rel = (a - b).abs() / denom;
                    prop_assert!(rel < 5e-3, "i={i} cpu_grad={a} gpu_grad={b} rel={rel}");
                }
            }
        }
    });
}
