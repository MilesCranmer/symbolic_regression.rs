use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::operator_registry::OpRegistry;
use dynamic_expressions::utils::ZipEq;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::constant_optimization::{OptimizeConstantsCtx, optimize_constants};
use crate::dataset::TaggedDataset;
use crate::loss_functions::baseline_loss_from_zero_expression;
use crate::optim::{BackTracking, EvalBudget, Objective, OptimOptions, bfgs_minimize};
use crate::pop_member::Evaluator;
use crate::{Dataset, MemberId, OperatorLibrary, Options, PopMember};

const D: usize = 3;
type T = f64;
type Ops = BuiltinOpsF64;

struct QuadND<'a> {
    target: &'a [f64],
    weight: &'a [f64],
}

impl Objective for QuadND<'_> {
    fn f_only(&mut self, x: &[f64], budget: &mut EvalBudget) -> Option<f64> {
        budget.f_calls += 1;
        let mut acc = 0.0;
        for ((&xi, &ti), &wi) in x.iter().zip_eq(self.target).zip_eq(self.weight) {
            let d = xi - ti;
            acc += wi * d * d;
        }
        Some(acc)
    }

    fn fg(&mut self, x: &[f64], g_out: &mut [f64], budget: &mut EvalBudget) -> Option<f64> {
        budget.f_calls += 1;
        let mut acc = 0.0;
        for (((&xi, &ti), &wi), go) in x
            .iter()
            .zip_eq(self.target)
            .zip_eq(self.weight)
            .zip_eq(g_out.iter_mut())
        {
            let d = xi - ti;
            acc += wi * d * d;
            *go = 2.0 * wi * d;
        }
        Some(acc)
    }
}

pub fn bfgs_quadratic_n16() -> Option<(Vec<f64>, f64)> {
    let n = 16;
    let x0 = vec![0.0f64; n];
    let target: [f64; 16] = core::array::from_fn(|i| (i as f64) / 7.0 - 1.0);
    let weight: [f64; 16] = core::array::from_fn(|i| 1.0 + (i as f64) * 0.01);
    let mut obj = QuadND {
        target: &target,
        weight: &weight,
    };
    let opts = OptimOptions {
        iterations: 40,
        f_calls_limit: 0,
        g_abstol: 1e-10,
    };
    let ls = BackTracking::default();
    let res = bfgs_minimize(&x0, &mut obj, opts, ls)?;
    Some((res.minimizer, res.minimum))
}

fn build_linear_expr_for_constant_optimization() -> PostfixExpr<T, Ops, D> {
    let mul = <Ops as OpRegistry>::lookup("*").unwrap().op;
    let add = <Ops as OpRegistry>::lookup("+").unwrap().op;

    // expr: c0 * x0 + c1
    PostfixExpr::new(
        vec![
            PNode::Const { idx: 0 },
            PNode::Var { feature: 0 },
            PNode::Op {
                arity: mul.arity,
                op: mul.id,
            },
            PNode::Const { idx: 1 },
            PNode::Op {
                arity: add.arity,
                op: add.id,
            },
        ],
        vec![0.0, 0.0],
        Metadata::default(),
    )
}

pub struct ConstantOptLinearEnv {
    dataset: Dataset<T>,
    options: Options<T, D>,
}

pub fn constant_opt_linear_env() -> ConstantOptLinearEnv {
    let n_rows = 512;
    let n_features = 1;
    let x: Vec<f64> = (0..n_rows).map(|i| (i as f64) / (n_rows as f64)).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 3.0).collect();
    let dataset = Dataset::new(
        Array2::from_shape_vec((n_features, n_rows), x).unwrap(),
        Array1::from_vec(y),
    );

    let options = Options::<T, D> {
        operators: OperatorLibrary::sr_default::<Ops, D>(),
        should_optimize_constants: true,
        optimizer_iterations: 40,
        optimizer_nrestarts: 0,
        ..Default::default()
    };

    ConstantOptLinearEnv { dataset, options }
}

pub fn run_constant_opt_linear(env: &ConstantOptLinearEnv) -> (bool, f64, Vec<f64>) {
    let expr = build_linear_expr_for_constant_optimization();
    let mut member = PopMember::from_expr(MemberId(0), None, 0, expr, env.dataset.n_features);
    let mut evaluator = Evaluator::new(env.dataset.n_rows);
    let mut grad_ctx = dynamic_expressions::GradContext::new(env.dataset.n_rows);
    let baseline_loss = if env.options.use_baseline {
        baseline_loss_from_zero_expression::<T, Ops, D>(&env.dataset, env.options.loss.as_ref())
    } else {
        None
    };
    let full_dataset = TaggedDataset::new(&env.dataset, baseline_loss);
    let _ = member.evaluate(&full_dataset, &env.options, &mut evaluator);

    let mut rng = StdRng::seed_from_u64(0);
    let mut next_birth = 1000u64;

    let (improved, evals) = optimize_constants(
        &mut rng,
        &mut member,
        OptimizeConstantsCtx {
            dataset: full_dataset,
            options: &env.options,
            evaluator: &mut evaluator,
            grad_ctx: &mut grad_ctx,
            next_birth: &mut next_birth,
        },
    );

    (improved, evals, member.expr.consts.clone())
}
