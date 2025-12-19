use crate::dataset::Dataset;
use crate::dataset::TaggedDataset;
use crate::optim::{bfgs_minimize, newton_1d_minimize, BackTracking, Objective, OptimOptions};
use crate::options::Options;
use crate::pop_member::{Evaluator, PopMember};
use dynamic_expressions::operator_enum::scalar::ScalarOpSet;
use dynamic_expressions::{eval_grad_tree_array, eval_plan_array_into, EvalOptions, GradContext};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::Rng;
use rand_distr::Distribution;
use rand_distr::StandardNormal;

fn eval_loss_and_grad<T: Float, Ops, const D: usize>(
    expr: &dynamic_expressions::expression::PostfixExpr<T, Ops, D>,
    dataset: &Dataset<T>,
    options: &Options<T, D>,
    grad_ctx: &mut GradContext<T, D>,
    eval_opts: &EvalOptions,
    dloss_dyhat: &mut [T],
    grad_out: &mut [f64],
) -> Option<f64>
where
    Ops: ScalarOpSet<T>,
{
    let n_params = expr.consts.len();
    let n_rows = dataset.n_rows;
    debug_assert_eq!(grad_out.len(), n_params);
    debug_assert_eq!(dloss_dyhat.len(), n_rows);

    let x = dataset.x.view();
    let (yhat, dy_dc, ok) = eval_grad_tree_array(expr, x, false, grad_ctx, eval_opts);
    if !ok {
        return None;
    }
    if yhat.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let loss = options.loss.loss(
        &yhat,
        dataset.y.as_slice().unwrap(),
        dataset.weights.as_ref().and_then(|w| w.as_slice()),
    );
    if !loss.is_finite() {
        return None;
    }

    options.loss.dloss_dyhat(
        &yhat,
        dataset.y.as_slice().unwrap(),
        dataset.weights.as_ref().and_then(|w| w.as_slice()),
        dloss_dyhat,
    );

    for (ci, gout) in grad_out.iter_mut().enumerate() {
        if ci >= n_params {
            break;
        }
        let base = ci * n_rows;
        let acc = dloss_dyhat
            .iter()
            .copied()
            .zip(dy_dc.data[base..base + n_rows].iter().copied())
            .fold(T::zero(), |a, (dl, dc)| a + dl * dc);
        *gout = acc.to_f64().unwrap_or(f64::INFINITY);
    }

    Some(loss.to_f64().unwrap_or(f64::INFINITY))
}

fn eval_loss_only<T: Float, Ops, const D: usize>(
    plan: &dynamic_expressions::EvalPlan<D>,
    expr: &dynamic_expressions::expression::PostfixExpr<T, Ops, D>,
    dataset: &Dataset<T>,
    options: &Options<T, D>,
    evaluator: &mut Evaluator<T, D>,
    eval_opts: &EvalOptions,
) -> Option<f64>
where
    Ops: ScalarOpSet<T>,
{
    let ok = eval_plan_array_into(
        &mut evaluator.yhat,
        plan,
        expr,
        dataset.x.view(),
        &mut evaluator.scratch,
        eval_opts,
    );
    if !ok {
        return None;
    }

    let loss = options.loss.loss(
        &evaluator.yhat,
        dataset.y.as_slice().unwrap(),
        dataset.weights.as_ref().and_then(|w| w.as_slice()),
    );
    if !loss.is_finite() {
        return None;
    }
    Some(loss.to_f64().unwrap_or(f64::INFINITY))
}

struct ConstObjective<'a, T: Float, Ops, const D: usize>
where
    Ops: ScalarOpSet<T>,
{
    plan: &'a dynamic_expressions::EvalPlan<D>,
    expr: &'a mut dynamic_expressions::expression::PostfixExpr<T, Ops, D>,
    dataset: &'a Dataset<T>,
    options: &'a Options<T, D>,
    evaluator: &'a mut Evaluator<T, D>,
    grad_ctx: &'a mut GradContext<T, D>,
    eval_opts: &'a EvalOptions,
    dloss_dyhat: &'a mut [T],
}

impl<'a, T: Float + FromPrimitive + ToPrimitive, Ops, const D: usize> Objective
    for ConstObjective<'a, T, Ops, D>
where
    Ops: ScalarOpSet<T>,
{
    fn f_only(&mut self, x: &[f64], budget: &mut crate::optim::EvalBudget) -> Option<f64> {
        budget.f_calls += 1;
        for (dst, &src) in self.expr.consts.iter_mut().zip(x.iter()) {
            *dst = T::from_f64(src)?;
        }
        eval_loss_only(
            self.plan,
            self.expr,
            self.dataset,
            self.options,
            self.evaluator,
            self.eval_opts,
        )
    }

    fn fg(
        &mut self,
        x: &[f64],
        g_out: &mut [f64],
        budget: &mut crate::optim::EvalBudget,
    ) -> Option<f64> {
        budget.f_calls += 1;
        for (dst, &src) in self.expr.consts.iter_mut().zip(x.iter()) {
            *dst = T::from_f64(src)?;
        }
        eval_loss_and_grad(
            self.expr,
            self.dataset,
            self.options,
            self.grad_ctx,
            self.eval_opts,
            self.dloss_dyhat,
            g_out,
        )
    }
}

pub fn optimize_constants<T: Float + FromPrimitive + ToPrimitive, Ops, const D: usize, R: Rng>(
    rng: &mut R,
    member: &mut PopMember<T, Ops, D>,
    ctx: OptimizeConstantsCtx<'_, '_, T, D>,
) -> (bool, f64)
where
    Ops: ScalarOpSet<T>,
{
    let OptimizeConstantsCtx {
        dataset,
        options,
        evaluator,
        grad_ctx,
        next_birth,
    } = ctx;
    let dataset_ref: &Dataset<T> = dataset.data;

    if !options.should_optimize_constants {
        return (false, 0.0);
    }
    let n_params = member.expr.consts.len();
    if n_params == 0 {
        return (false, 0.0);
    }

    let orig_consts = member.expr.consts.clone();
    let orig_birth = member.birth;
    let orig_loss = member.loss;
    let orig_cost = member.cost;

    let eval_opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    grad_ctx.plan = Some(member.plan.clone());
    grad_ctx.plan_nodes_len = member.expr.nodes.len();
    grad_ctx.plan_n_consts = member.expr.consts.len();
    grad_ctx.plan_n_features = dataset_ref.n_features;

    let mut dloss_dyhat = vec![T::zero(); dataset_ref.n_rows];

    let baseline = match eval_loss_only(
        &member.plan,
        &member.expr,
        dataset_ref,
        options,
        evaluator,
        &eval_opts,
    ) {
        Some(v) => v,
        None => return (false, 0.0),
    };

    let x0: Vec<f64> = member
        .expr
        .consts
        .iter()
        .map(|v| v.to_f64().unwrap_or(0.0))
        .collect();

    let mut best_x = x0.clone();
    let mut best_f = baseline;

    let optim_opts = OptimOptions {
        iterations: options.optimizer_iterations,
        f_calls_limit: options.optimizer_f_calls_limit,
        g_abstol: 1e-8,
    };
    let ls = BackTracking::default();

    let mut n_evals: u64 = 0;

    // Main run at x0:
    {
        let mut obj = ConstObjective {
            plan: &member.plan,
            expr: &mut member.expr,
            dataset: dataset_ref,
            options,
            evaluator,
            grad_ctx,
            eval_opts: &eval_opts,
            dloss_dyhat: &mut dloss_dyhat,
        };

        if n_params == 1 {
            if let Some(res) = newton_1d_minimize(x0[0], &mut obj, optim_opts, ls) {
                n_evals = n_evals.saturating_add(res.f_calls as u64);
                if res.minimum < best_f {
                    best_f = res.minimum;
                    best_x = res.minimizer;
                }
            }
        } else if let Some(res) = bfgs_minimize(&x0, &mut obj, optim_opts, ls) {
            n_evals = n_evals.saturating_add(res.f_calls as u64);
            if res.minimum < best_f {
                best_f = res.minimum;
                best_x = res.minimizer;
            }
        }
    }

    // Restarts:
    for _ in 0..options.optimizer_nrestarts {
        let mut xt = x0.clone();
        for v in &mut xt {
            let eps: f64 = StandardNormal.sample(rng);
            *v *= 1.0 + 0.5 * eps;
        }

        let mut obj = ConstObjective {
            plan: &member.plan,
            expr: &mut member.expr,
            dataset: dataset_ref,
            options,
            evaluator,
            grad_ctx,
            eval_opts: &eval_opts,
            dloss_dyhat: &mut dloss_dyhat,
        };

        if n_params == 1 {
            if let Some(res) = newton_1d_minimize(xt[0], &mut obj, optim_opts, ls) {
                n_evals = n_evals.saturating_add(res.f_calls as u64);
                if res.minimum < best_f {
                    best_f = res.minimum;
                    best_x = res.minimizer;
                }
            }
        } else if let Some(res) = bfgs_minimize(&xt, &mut obj, optim_opts, ls) {
            n_evals = n_evals.saturating_add(res.f_calls as u64);
            if res.minimum < best_f {
                best_f = res.minimum;
                best_x = res.minimizer;
            }
        }
    }

    if best_f < baseline {
        for (dst, &src) in member.expr.consts.iter_mut().zip(best_x.iter()) {
            *dst = T::from_f64(src).unwrap_or_else(T::zero);
        }
        let ok = member.evaluate(&dataset, options, evaluator);
        if !ok {
            member.expr.consts = orig_consts;
            member.birth = orig_birth;
            member.loss = orig_loss;
            member.cost = orig_cost;
            return (false, n_evals as f64);
        }
        n_evals = n_evals.saturating_add(1);
        member.birth = *next_birth;
        *next_birth += 1;
        (true, n_evals as f64)
    } else {
        member.expr.consts = orig_consts;
        member.birth = orig_birth;
        member.loss = orig_loss;
        member.cost = orig_cost;
        (false, n_evals as f64)
    }
}

pub struct OptimizeConstantsCtx<'a, 'd, T: Float, const D: usize> {
    pub dataset: TaggedDataset<'d, T>,
    pub options: &'a Options<T, D>,
    pub evaluator: &'a mut Evaluator<T, D>,
    pub grad_ctx: &'a mut GradContext<T, D>,
    pub next_birth: &'a mut u64,
}
