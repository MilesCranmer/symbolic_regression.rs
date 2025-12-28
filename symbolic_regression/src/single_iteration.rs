use std::ops::AddAssign;

use fastrand::Rng;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::constant_optimization::{OptimizeConstantsCtx, optimize_constants};
use crate::dataset::TaggedDataset;
use crate::hall_of_fame::HallOfFame;
use crate::options::Options;
use crate::pop_member::Evaluator;
use crate::population::Population;
use crate::regularized_evolution::{RegEvolCtx, reg_evol_cycle};
use crate::stop_controller::StopController;

pub struct IterationCtx<'a, T: Float + AddAssign, Ops, const D: usize> {
    pub rng: &'a mut Rng,
    pub full_dataset: TaggedDataset<'a, T>,
    pub curmaxsize: usize,
    pub stats: &'a RunningSearchStatistics,
    pub options: &'a Options<T, D>,
    pub evaluator: &'a mut Evaluator<T, D>,
    pub grad_ctx: &'a mut dynamic_expressions::GradContext<T, D>,
    pub next_id: &'a mut u64,
    pub controller: &'a StopController,
    pub _ops: core::marker::PhantomData<Ops>,
}

pub fn s_r_cycle<T, Ops, const D: usize>(
    pop: &mut Population<T, Ops, D>,
    ctx: &mut IterationCtx<'_, T, Ops, D>,
    eval_dataset: TaggedDataset<'_, T>,
) -> (f64, HallOfFame<T, Ops, D>)
where
    T: Float + FromPrimitive + ToPrimitive + AddAssign,
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    let max_temp = 1.0;
    let min_temp = if ctx.options.annealing { 0.0 } else { 1.0 };
    let ncycles = ctx.options.ncycles_per_iteration.max(1);
    let mut num_evals = 0.0;
    let mut best_seen = HallOfFame::new(ctx.options.maxsize);
    best_seen.update_from_members(&pop.members, ctx.options, ctx.curmaxsize);

    for i in 0..ncycles {
        if ctx.controller.is_cancelled() {
            break;
        }
        let temperature = if ncycles <= 1 {
            max_temp
        } else {
            let t = (i as f64) / ((ncycles - 1) as f64);
            max_temp + t * (min_temp - max_temp)
        };
        num_evals += reg_evol_cycle(
            pop,
            RegEvolCtx {
                rng: ctx.rng,
                dataset: eval_dataset,
                temperature,
                curmaxsize: ctx.curmaxsize,
                stats: ctx.stats,
                options: ctx.options,
                evaluator: ctx.evaluator,
                next_id: ctx.next_id,
                controller: ctx.controller,
                _ops: core::marker::PhantomData,
            },
        );
        best_seen.update_from_members(&pop.members, ctx.options, ctx.curmaxsize);
    }
    (num_evals, best_seen)
}

pub fn optimize_and_simplify_population<T, Ops, const D: usize>(
    pop: &mut Population<T, Ops, D>,
    ctx: &mut IterationCtx<'_, T, Ops, D>,
    opt_dataset: TaggedDataset<'_, T>,
) -> f64
where
    T: Float + FromPrimitive + ToPrimitive + AddAssign,
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    let mut num_evals = 0.0;

    if ctx.options.should_simplify {
        for m in &mut pop.members {
            if ctx.controller.is_cancelled() {
                return num_evals;
            }
            let changed = dynamic_expressions::simplify_in_place(&mut m.expr, &ctx.evaluator.eval_opts);
            if changed {
                m.rebuild_plan(ctx.full_dataset.n_features);
            }
        }
    }

    if ctx.options.should_optimize_constants && ctx.options.optimizer_probability > 0.0 {
        ctx.evaluator.ensure_n_rows(opt_dataset.n_rows);
        ctx.grad_ctx.n_rows = opt_dataset.n_rows;
        for m in &mut pop.members {
            if ctx.controller.is_cancelled() {
                return num_evals;
            }
            if ctx.rng.f64() < ctx.options.optimizer_probability {
                let (improved, evals) = optimize_constants(
                    ctx.rng,
                    m,
                    OptimizeConstantsCtx {
                        dataset: opt_dataset,
                        options: ctx.options,
                        evaluator: ctx.evaluator,
                        grad_ctx: ctx.grad_ctx,
                    },
                );
                let _ = improved;
                num_evals += evals;
            }
        }
    }

    // Match SymbolicRegression.jl `finalize_costs`: only re-evaluate on the full dataset when
    // batching is enabled (i.e., members were evolved on a batch and need final losses/costs).
    if ctx.options.batching {
        ctx.evaluator.ensure_n_rows(ctx.full_dataset.n_rows);
        for m in &mut pop.members {
            if ctx.controller.is_cancelled() {
                return num_evals;
            }
            let _ = m.evaluate(&ctx.full_dataset, ctx.options, ctx.evaluator);
            num_evals += 1.0;
        }
    }

    num_evals
}
