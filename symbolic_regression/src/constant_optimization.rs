use std::ops::AddAssign;

use dynamic_expressions::EvalOptions;
use fastrand::Rng;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::dataset::{Dataset, TaggedDataset};
use crate::expression::{ConstantOptimizable, Evaluatable, ExprExt, ScalarConstants};
use crate::optim::{BackTracking, Objective, OptimOptions, bfgs_minimize, newton_1d_minimize};
use crate::options::Options;
use crate::pop_member::{Evaluator, PopMember};
use crate::random::standard_normal;

struct EvalWorkspace<'a, T: Float + AddAssign, const D: usize> {
    dataset: &'a Dataset<T>,
    options: &'a Options<T, D>,
    evaluator: &'a mut Evaluator<T, D>,
    grad_ctx: &'a mut dynamic_expressions::GradContext<T, D>,
    eval_opts: EvalOptions,
}

impl<'a, T: Float + AddAssign, const D: usize> EvalWorkspace<'a, T, D> {
    fn new(
        dataset: &'a Dataset<T>,
        options: &'a Options<T, D>,
        evaluator: &'a mut Evaluator<T, D>,
        grad_ctx: &'a mut dynamic_expressions::GradContext<T, D>,
    ) -> Self {
        evaluator.ensure_n_rows(dataset.n_rows);
        let eval_opts = EvalOptions {
            check_finite: true,
            early_exit: true,
        };
        Self {
            dataset,
            options,
            evaluator,
            grad_ctx,
            eval_opts,
        }
    }

    fn loss_only<Ops, E>(&mut self, plans: &[dynamic_expressions::EvalPlan<D>], expr: &E) -> Option<f64>
    where
        T: FromPrimitive + ToPrimitive,
        Ops: dynamic_expressions::OperatorSet<T = T>,
        E: Evaluatable<T, Ops, D>,
    {
        let ok = expr.eval_with_plans(
            plans,
            self.dataset.x.view(),
            &mut self.evaluator.yhat,
            &mut self.evaluator.scratch,
            &self.eval_opts,
        );
        if !ok {
            return None;
        }

        let loss = self.options.loss.loss(
            &self.evaluator.yhat,
            self.dataset.y.as_slice().unwrap(),
            self.dataset.weights.as_ref().and_then(|w| w.as_slice()),
        );
        if !loss.is_finite() {
            return None;
        }
        Some(loss.to_f64().unwrap_or(f64::INFINITY))
    }

    fn optimize_from_start<Ops, E>(
        &mut self,
        start: &[f64],
        member: &mut PopMember<T, Ops, D, E>,
        optim_opts: OptimOptions,
        ls: BackTracking,
    ) -> Option<crate::optim::OptimResult>
    where
        T: FromPrimitive + ToPrimitive,
        Ops: dynamic_expressions::OperatorSet<T = T>,
        E: ScalarConstants<T, Ops, D> + ConstantOptimizable<T, Ops, D> + Send + Sync,
    {
        let n_params = member.expr.n_scalars();
        let mut obj = ConstObjective {
            plans: &member.plans,
            expr: &mut member.expr,
            workspace: self,
            tmp_t: vec![T::zero(); n_params],
            _ops: core::marker::PhantomData,
        };

        if n_params == 1 {
            newton_1d_minimize(start[0], &mut obj, optim_opts, ls)
        } else {
            bfgs_minimize(start, &mut obj, optim_opts, ls)
        }
    }
}

struct ConstObjective<'plan, 'expr, 'work, 'data, T: Float + AddAssign, Ops, const D: usize, E>
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
    E: ScalarConstants<T, Ops, D> + ConstantOptimizable<T, Ops, D>,
{
    plans: &'plan [dynamic_expressions::EvalPlan<D>],
    expr: &'expr mut E,
    workspace: &'work mut EvalWorkspace<'data, T, D>,
    tmp_t: Vec<T>,
    // Tie `Ops` to the objective so callers don't need turbofish.
    _ops: core::marker::PhantomData<Ops>,
}

impl<'plan, 'expr, 'work, 'data, T, Ops, const D: usize, E> ConstObjective<'plan, 'expr, 'work, 'data, T, Ops, D, E>
where
    T: Float + FromPrimitive + ToPrimitive + AddAssign,
    Ops: dynamic_expressions::OperatorSet<T = T>,
    E: ScalarConstants<T, Ops, D> + ConstantOptimizable<T, Ops, D>,
{
    fn set_expr_params(&mut self, x: &[f64]) -> Option<()> {
        if self.tmp_t.len() != x.len() {
            self.tmp_t.resize(x.len(), T::zero());
        }
        for (dst, &src) in self.tmp_t.iter_mut().zip(x) {
            *dst = T::from_f64(src)?;
        }
        self.expr.unpack_scalars(&self.tmp_t);
        Some(())
    }
}

impl<'plan, 'expr, 'work, 'data, T, Ops, const D: usize, E> Objective
    for ConstObjective<'plan, 'expr, 'work, 'data, T, Ops, D, E>
where
    T: Float + FromPrimitive + ToPrimitive + AddAssign,
    Ops: dynamic_expressions::OperatorSet<T = T>,
    E: ScalarConstants<T, Ops, D> + ConstantOptimizable<T, Ops, D>,
{
    fn f_only(&mut self, x: &[f64], budget: &mut crate::optim::EvalBudget) -> Option<f64> {
        budget.f_calls += 1;
        self.set_expr_params(x)?;
        self.workspace.loss_only::<Ops, E>(self.plans, self.expr)
    }

    fn fg(&mut self, x: &[f64], g_out: &mut [f64], budget: &mut crate::optim::EvalBudget) -> Option<f64> {
        budget.f_calls += 1;
        self.set_expr_params(x)?;
        self.expr.loss_and_grad(
            self.plans,
            self.workspace.dataset,
            self.workspace.options,
            self.workspace.evaluator,
            self.workspace.grad_ctx,
            &self.workspace.eval_opts,
            g_out,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finite_diff_loss_and_grad<T, Ops, const D: usize, E>(
    expr: &mut E,
    plans: &[dynamic_expressions::EvalPlan<D>],
    dataset: &Dataset<T>,
    options: &Options<T, D>,
    evaluator: &mut Evaluator<T, D>,
    _grad_ctx: &mut dynamic_expressions::GradContext<T, D>,
    eval_opts: &EvalOptions,
    grad_out: &mut [f64],
) -> Option<f64>
where
    T: Float + FromPrimitive + ToPrimitive + AddAssign,
    Ops: dynamic_expressions::OperatorSet<T = T>,
    E: ScalarConstants<T, Ops, D> + Evaluatable<T, Ops, D>,
{
    let n_params = expr.n_scalars();
    if n_params == 0 {
        return None;
    }
    if grad_out.len() != n_params {
        return None;
    }

    let mut base_t: Vec<T> = Vec::with_capacity(n_params);
    expr.pack_scalars(&mut base_t);
    let mut x: Vec<f64> = base_t.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
    let mut tmp_t = base_t.clone();

    let mut loss_at = |expr: &mut E, tmp_t: &mut [T], x: &[f64]| -> Option<f64> {
        for (dst, &src) in tmp_t.iter_mut().zip(x) {
            *dst = T::from_f64(src)?;
        }
        expr.unpack_scalars(tmp_t);
        evaluator.ensure_n_rows(dataset.n_rows);
        let ok = expr.eval_with_plans(
            plans,
            dataset.x.view(),
            &mut evaluator.yhat,
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
    };

    let base_loss = loss_at(expr, &mut tmp_t, &x)?;

    for i in 0..n_params {
        let xi = x[i];
        let h = 1e-6 * (xi.abs() + 1.0);
        if !h.is_finite() || h == 0.0 {
            grad_out[i] = 0.0;
            continue;
        }

        x[i] = xi + h;
        let f_plus = loss_at(expr, &mut tmp_t, &x)?;
        x[i] = xi - h;
        let f_minus = loss_at(expr, &mut tmp_t, &x)?;
        x[i] = xi;

        grad_out[i] = (f_plus - f_minus) / (2.0 * h);
    }

    expr.unpack_scalars(&base_t);
    Some(base_loss)
}

pub fn optimize_constants<T: Float + FromPrimitive + ToPrimitive + AddAssign, Ops, const D: usize, E>(
    rng: &mut Rng,
    member: &mut PopMember<T, Ops, D, E>,
    ctx: OptimizeConstantsCtx<'_, '_, T, D>,
) -> (bool, f64)
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
    E: ExprExt<T, Ops, D> + ConstantOptimizable<T, Ops, D>,
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
    let n_params = member.expr.n_scalars();
    if n_params == 0 {
        return (false, 0.0);
    }

    let mut orig_flat: Vec<T> = Vec::with_capacity(n_params);
    member.expr.pack_scalars(&mut orig_flat);
    let orig_birth = member.birth;
    let orig_loss = member.loss;
    let orig_cost = member.cost;

    let mut workspace = EvalWorkspace::new(dataset_ref, options, evaluator, grad_ctx);

    let baseline = match workspace.loss_only::<Ops, E>(&member.plans, &member.expr) {
        Some(v) => v,
        None => return (false, 0.0),
    };

    let x0: Vec<f64> = orig_flat.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();

    let mut best_x = x0.clone();
    let mut best_f = baseline;

    let optim_opts = OptimOptions {
        iterations: options.optimizer_iterations,
        f_calls_limit: options.optimizer_f_calls_limit,
        g_abstol: 1e-8,
    };
    let ls = BackTracking::default();

    let mut n_evals: u64 = 0;

    {
        let res = workspace.optimize_from_start::<Ops, E>(&x0, member, optim_opts, ls);
        if let Some(res) = res {
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
            let eps: f64 = standard_normal(rng);
            *v *= 1.0 + 0.5 * eps;
        }

        let res = workspace.optimize_from_start::<Ops, E>(&xt, member, optim_opts, ls);
        if let Some(res) = res {
            n_evals = n_evals.saturating_add(res.f_calls as u64);
            if res.minimum < best_f {
                best_f = res.minimum;
                best_x = res.minimizer;
            }
        }
    }

    if best_f < baseline {
        // Apply best.
        let mut best_t: Vec<T> = vec![T::zero(); n_params];
        for (dst, &src) in best_t.iter_mut().zip(best_x.iter()) {
            *dst = T::from_f64(src).unwrap_or_else(T::zero);
        }
        member.expr.unpack_scalars(&best_t);

        let ok = member.evaluate(&dataset, options, evaluator);
        if !ok {
            member.expr.unpack_scalars(&orig_flat);
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
        member.expr.unpack_scalars(&orig_flat);
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
    pub grad_ctx: &'a mut dynamic_expressions::GradContext<T, D>,
    pub next_birth: &'a mut u64,
}
