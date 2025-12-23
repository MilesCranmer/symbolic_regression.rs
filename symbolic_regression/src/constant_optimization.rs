use std::ops::AddAssign;

use dynamic_expressions::operator_enum::scalar;
use dynamic_expressions::{
    EvalOptions, GradContext, SubtreeCache, eval_grad_plan_array_into_cached, eval_plan_array_into_cached,
};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::Rng;
use rand_distr::Distribution;

use crate::dataset::{Dataset, TaggedDataset};
use crate::optim::{BackTracking, Objective, OptimOptions, bfgs_minimize, newton_1d_minimize};
use crate::options::Options;
use crate::pop_member::{Evaluator, PopMember};

#[allow(clippy::too_many_arguments)]
fn eval_loss_and_grad<T: Float + AddAssign, Ops, const D: usize>(
    plan: &dynamic_expressions::EvalPlan<D>,
    expr: &dynamic_expressions::expression::PostfixExpr<T, Ops, D>,
    dataset: &Dataset<T>,
    options: &Options<T, D>,
    evaluator: &mut Evaluator<T, D>,
    grad_ctx: &mut GradContext<T, D>,
    subtree_cache: &mut SubtreeCache<T>,
    dataset_key: u64,
    dy_dc_buf: &mut [T],
    eval_opts: &EvalOptions,
    dloss_dyhat: &mut [T],
    grad_out: &mut [f64],
) -> Option<f64>
where
    Ops: scalar::ScalarOpSet<T>,
{
    let n_params = expr.consts.len();
    let n_rows = dataset.n_rows;
    debug_assert_eq!(grad_out.len(), n_params);
    debug_assert_eq!(dloss_dyhat.len(), n_rows);
    debug_assert_eq!(dy_dc_buf.len(), n_params * n_rows);

    // Value + gradient (w.r.t. constants) into preallocated buffers.
    let x = dataset.x.view();
    let ok = eval_grad_plan_array_into_cached(
        &mut evaluator.yhat,
        dy_dc_buf,
        plan,
        expr,
        x,
        // variable=
        false,
        grad_ctx,
        eval_opts,
        subtree_cache,
        dataset_key,
    );
    if !ok {
        return None;
    }
    if evaluator.yhat.iter().any(|v| !v.is_finite()) {
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

    options.loss.dloss_dyhat(
        &evaluator.yhat,
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
            .zip(dy_dc_buf[base..base + n_rows].iter().copied())
            .fold(T::zero(), |a, (dl, dc)| a + dl * dc);
        *gout = acc.to_f64().unwrap_or(f64::INFINITY);
    }

    Some(loss.to_f64().unwrap_or(f64::INFINITY))
}

#[allow(clippy::too_many_arguments)]
fn eval_loss_only<T: Float, Ops, const D: usize>(
    plan: &dynamic_expressions::EvalPlan<D>,
    expr: &dynamic_expressions::expression::PostfixExpr<T, Ops, D>,
    dataset: &Dataset<T>,
    options: &Options<T, D>,
    evaluator: &mut Evaluator<T, D>,
    subtree_cache: &mut SubtreeCache<T>,
    dataset_key: u64,
    eval_opts: &EvalOptions,
) -> Option<f64>
where
    Ops: scalar::ScalarOpSet<T>,
{
    let ok = eval_plan_array_into_cached(
        &mut evaluator.yhat,
        plan,
        expr,
        dataset.x.view(),
        &mut evaluator.scratch,
        eval_opts,
        subtree_cache,
        dataset_key,
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
    Ops: scalar::ScalarOpSet<T>,
{
    plan: &'a dynamic_expressions::EvalPlan<D>,
    expr: &'a mut dynamic_expressions::expression::PostfixExpr<T, Ops, D>,
    dataset: &'a Dataset<T>,
    options: &'a Options<T, D>,
    evaluator: &'a mut Evaluator<T, D>,
    subtree_cache: &'a mut SubtreeCache<T>,
    dataset_key: u64,
    grad_ctx: &'a mut GradContext<T, D>,
    dy_dc_buf: &'a mut [T],
    eval_opts: &'a EvalOptions,
    dloss_dyhat: &'a mut [T],
}

impl<'a, T: Float + FromPrimitive + ToPrimitive + AddAssign, Ops, const D: usize> Objective
    for ConstObjective<'a, T, Ops, D>
where
    Ops: scalar::ScalarOpSet<T>,
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
            self.subtree_cache,
            self.dataset_key,
            self.eval_opts,
        )
    }

    fn fg(&mut self, x: &[f64], g_out: &mut [f64], budget: &mut crate::optim::EvalBudget) -> Option<f64> {
        budget.f_calls += 1;
        for (dst, &src) in self.expr.consts.iter_mut().zip(x.iter()) {
            *dst = T::from_f64(src)?;
        }
        eval_loss_and_grad(
            self.plan,
            self.expr,
            self.dataset,
            self.options,
            self.evaluator,
            self.grad_ctx,
            self.subtree_cache,
            self.dataset_key,
            self.dy_dc_buf,
            self.eval_opts,
            self.dloss_dyhat,
            g_out,
        )
    }
}

pub fn optimize_constants<T: Float + FromPrimitive + ToPrimitive + AddAssign, Ops, const D: usize, R: Rng>(
    rng: &mut R,
    member: &mut PopMember<T, Ops, D>,
    ctx: OptimizeConstantsCtx<'_, '_, T, D>,
) -> (bool, f64)
where
    Ops: scalar::ScalarOpSet<T>,
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
    let mut dloss_dyhat = vec![T::zero(); dataset_ref.n_rows];
    let mut dy_dc_buf = vec![T::zero(); dataset_ref.n_rows * n_params];

    // A conservative cap... (subtree_cache_max_bytes could be made an option).
    let mut subtree_cache = SubtreeCache::<T>::new(dataset_ref.n_rows, 32 * 1024 * 1024);
    let dataset_key = dataset_ref.x_key;

    let baseline = match eval_loss_only(
        &member.plan,
        &member.expr,
        dataset_ref,
        options,
        evaluator,
        &mut subtree_cache,
        dataset_key,
        &eval_opts,
    ) {
        Some(v) => v,
        None => return (false, 0.0),
    };

    let x0: Vec<f64> = member.expr.consts.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();

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
            subtree_cache: &mut subtree_cache,
            dataset_key,
            grad_ctx,
            dy_dc_buf: &mut dy_dc_buf,
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
            let eps: f64 = rand_distr::StandardNormal.sample(rng);
            *v *= 1.0 + 0.5 * eps;
        }

        let mut obj = ConstObjective {
            plan: &member.plan,
            expr: &mut member.expr,
            dataset: dataset_ref,
            options,
            evaluator,
            subtree_cache: &mut subtree_cache,
            dataset_key,
            grad_ctx,
            dy_dc_buf: &mut dy_dc_buf,
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
