use std::ops::AddAssign;

use dynamic_expressions::utils::ZipEq;
use dynamic_expressions::{EvalOptions, GradContext, OperatorSet};
use fastrand::Rng;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::dataset::{Dataset, TaggedDataset};
#[cfg(wgpu)]
use crate::loss_functions::{LossKind, loss_to_cost};
use crate::optim::{BackTracking, Objective, OptimOptions, bfgs_minimize, newton_1d_minimize};
use crate::options::Options;
use crate::pop_member::{Evaluator, PopMember, get_birth_order};
use crate::random::standard_normal;

struct EvalWorkspace<'a, T: Float + AddAssign, const D: usize> {
    dataset: &'a Dataset<T>,
    options: &'a Options<T, D>,
    evaluator: &'a mut Evaluator<T, D>,
    grad_ctx: &'a mut GradContext<T, D>,
    eval_opts: EvalOptions,
    dloss_dyhat: Vec<T>,
}

impl<'a, T: Float + AddAssign, const D: usize> EvalWorkspace<'a, T, D> {
    fn new(
        dataset: &'a Dataset<T>,
        options: &'a Options<T, D>,
        evaluator: &'a mut Evaluator<T, D>,
        grad_ctx: &'a mut GradContext<T, D>,
    ) -> Self {
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
            dloss_dyhat: vec![T::zero(); dataset.n_rows],
        }
    }

    fn loss_only<Ops>(
        &mut self,
        plan: &dynamic_expressions::EvalPlan<D>,
        expr: &dynamic_expressions::expression::PostfixExpr<T, Ops, D>,
    ) -> Option<f64>
    where
        Ops: OperatorSet<T = T>,
    {
        let ok = dynamic_expressions::eval_plan_array_into(
            &mut self.evaluator.yhat,
            plan,
            expr,
            self.dataset.x.view(),
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

    fn loss_and_grad<Ops>(
        &mut self,
        _plan: &dynamic_expressions::EvalPlan<D>,
        expr: &dynamic_expressions::expression::PostfixExpr<T, Ops, D>,
        grad_out: &mut [f64],
    ) -> Option<f64>
    where
        Ops: OperatorSet<T = T>,
    {
        let n_params = expr.consts.len();
        let n_rows = self.dataset.n_rows;
        debug_assert_eq!(grad_out.len(), n_params);
        debug_assert_eq!(self.dloss_dyhat.len(), n_rows);

        let x = self.dataset.x.view();
        let (yhat, dy_dc, ok) =
            dynamic_expressions::eval_grad_tree_array(expr, x, false, self.grad_ctx, &self.eval_opts);
        if !ok {
            return None;
        }
        if yhat.iter().any(|v| !v.is_finite()) {
            return None;
        }

        let loss = self.options.loss.loss(
            &yhat,
            self.dataset.y.as_slice().unwrap(),
            self.dataset.weights.as_ref().and_then(|w| w.as_slice()),
        );
        if !loss.is_finite() {
            return None;
        }

        self.options.loss.dloss_dyhat(
            &yhat,
            self.dataset.y.as_slice().unwrap(),
            self.dataset.weights.as_ref().and_then(|w| w.as_slice()),
            &mut self.dloss_dyhat,
        );

        for (ci, gout) in grad_out.iter_mut().enumerate() {
            if ci >= n_params {
                break;
            }
            let base = ci * n_rows;
            let acc = self
                .dloss_dyhat
                .iter()
                .copied()
                .zip_eq(dy_dc.data[base..base + n_rows].iter().copied())
                .fold(T::zero(), |a, (dl, dc)| a + dl * dc);
            *gout = acc.to_f64().unwrap_or(f64::INFINITY);
        }

        Some(loss.to_f64().unwrap_or(f64::INFINITY))
    }

    fn optimize_from_start<Ops>(
        &mut self,
        start: &[f64],
        n_params: usize,
        member: &mut PopMember<T, Ops, D>,
        optim_opts: OptimOptions,
        ls: BackTracking,
    ) -> Option<crate::optim::OptimResult>
    where
        T: FromPrimitive,
        Ops: OperatorSet<T = T>,
    {
        let mut obj = ConstObjective {
            plan: &member.plan,
            expr: &mut member.expr,
            workspace: self,
        };

        if n_params == 1 {
            newton_1d_minimize(start[0], &mut obj, optim_opts, ls)
        } else {
            bfgs_minimize(start, &mut obj, optim_opts, ls)
        }
    }
}

struct ConstObjective<'plan, 'expr, 'work, 'data, T: Float + AddAssign, Ops, const D: usize> {
    plan: &'plan dynamic_expressions::EvalPlan<D>,
    expr: &'expr mut dynamic_expressions::expression::PostfixExpr<T, Ops, D>,
    workspace: &'work mut EvalWorkspace<'data, T, D>,
}

impl<'plan, 'expr, 'work, 'data, T: Float + FromPrimitive + AddAssign, Ops, const D: usize> Objective
    for ConstObjective<'plan, 'expr, 'work, 'data, T, Ops, D>
where
    Ops: OperatorSet<T = T>,
{
    fn f_only(&mut self, x: &[f64], budget: &mut crate::optim::EvalBudget) -> Option<f64> {
        budget.f_calls += 1;
        for (dst, &src) in self.expr.consts.iter_mut().zip_eq(x) {
            *dst = T::from_f64(src)?;
        }
        self.workspace.loss_only::<Ops>(self.plan, self.expr)
    }

    fn fg(&mut self, x: &[f64], g_out: &mut [f64], budget: &mut crate::optim::EvalBudget) -> Option<f64> {
        budget.f_calls += 1;
        for (dst, &src) in self.expr.consts.iter_mut().zip_eq(x) {
            *dst = T::from_f64(src)?;
        }
        self.workspace.loss_and_grad::<Ops>(self.plan, self.expr, g_out)
    }
}

pub fn optimize_constants<T: Float + FromPrimitive + ToPrimitive + AddAssign, Ops, const D: usize>(
    rng: &mut Rng,
    member: &mut PopMember<T, Ops, D>,
    ctx: OptimizeConstantsCtx<'_, '_, T, D>,
) -> (bool, f64)
where
    Ops: OperatorSet<T = T>,
{
    let OptimizeConstantsCtx {
        dataset,
        options,
        evaluator,
        grad_ctx,
        #[cfg(wgpu)]
        gpu,
    } = ctx;
    let dataset_ref: &Dataset<T> = dataset.data;
    evaluator.ensure_n_rows(dataset.n_rows);
    grad_ctx.n_rows = dataset.n_rows;
    let n_params = member.expr.consts.len();
    if n_params == 0 {
        return (false, 0.0);
    }
    // Fast-path: if a GPU client is available, do a fused Adam optimization entirely on-GPU.
    // This avoids the disastrous CPU↔GPU round-trip that happens if we use a CPU optimizer that
    // calls into the GPU objective/gradient one evaluation at a time.
    #[cfg(wgpu)]
    {
        if let Some(g) = gpu.filter(|g| {
            options.loss_kind == LossKind::Mse && dataset.n_rows == g.n_rows && dataset.n_features == g.n_features
        }) {
            if let Some(base) = crate::gpu::pack_expr(&member.expr) {
                let n_consts = n_params.min(crate::gpu::MAX_CONSTS);
                let n_restarts = 1 + options.optimizer_nrestarts;
                let iters = (options.optimizer_iterations as u32).saturating_mul(4).clamp(16, 1024);
                let params = crate::gpu::AdamParams {
                    iters,
                    ..Default::default()
                };

                let mut programs: Vec<crate::gpu::PackedProgram> = Vec::with_capacity(n_restarts);
                for r in 0..n_restarts {
                    let mut p = base;
                    if r > 0 {
                        for j in 0..n_consts {
                            let scale = 1.0 + 0.5 * (standard_normal(rng) as f32);
                            p.consts[j] = base.consts[j] * scale;
                        }
                    }
                    programs.push(p);
                }

                let mut losses = vec![0.0f32; programs.len()];
                g.optimize_adam_many(&mut programs, params, &mut losses);

                let mut best_loss = f32::INFINITY;
                let mut best_consts = base.consts;
                for (p, &l) in programs.iter().zip(losses.iter()) {
                    if l.is_finite() && l < best_loss {
                        best_loss = l;
                        best_consts = p.consts;
                    }
                }

                let baseline = member.loss.to_f32().unwrap_or(f32::INFINITY);
                let evals = (params.iters as f64) * (programs.len() as f64);

                if best_loss.is_finite() && best_loss < baseline {
                    for (dst, &src) in member
                        .expr
                        .consts
                        .iter_mut()
                        .take(n_consts)
                        .zip(best_consts.iter().take(n_consts))
                    {
                        *dst = T::from_f32(src).unwrap_or_else(T::nan);
                    }
                    member.loss = T::from_f32(best_loss).unwrap_or_else(T::nan);
                    member.cost = loss_to_cost(
                        member.loss,
                        member.complexity,
                        options.parsimony,
                        options.use_baseline,
                        dataset.baseline_loss,
                    );
                    member.birth = get_birth_order(options.deterministic);
                    return (true, evals);
                }
                return (false, evals);
            }
        }
    }

    let orig_consts = member.expr.consts.clone();
    let orig_birth = member.birth;
    let orig_loss = member.loss;
    let orig_cost = member.cost;

    let mut workspace = EvalWorkspace::new(dataset_ref, options, evaluator, grad_ctx);

    let baseline = match workspace.loss_only::<Ops>(&member.plan, &member.expr) {
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

    {
        let res = workspace.optimize_from_start(&x0, n_params, member, optim_opts, ls);
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

        let res = { workspace.optimize_from_start(&xt, n_params, member, optim_opts, ls) };
        if let Some(res) = res {
            n_evals = n_evals.saturating_add(res.f_calls as u64);
            if res.minimum < best_f {
                best_f = res.minimum;
                best_x = res.minimizer;
            }
        }
    }

    if best_f < baseline {
        for (dst, &src) in member.expr.consts.iter_mut().zip_eq(&best_x) {
            *dst = T::from_f64(src).unwrap_or_else(T::zero);
        }
        let ok = member.evaluate_with_gpu(
            &dataset,
            options,
            evaluator,
            #[cfg(wgpu)]
            gpu,
        );
        if !ok {
            member.expr.consts = orig_consts;
            member.birth = orig_birth;
            member.loss = orig_loss;
            member.cost = orig_cost;
            return (false, n_evals as f64);
        }
        n_evals = n_evals.saturating_add(1);
        member.birth = get_birth_order(options.deterministic);
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

    #[cfg(wgpu)]
    pub gpu: Option<&'a crate::gpu::GpuClient>,
}
