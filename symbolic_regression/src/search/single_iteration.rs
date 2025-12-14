use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::constant_optimization::optimize_constants;
use crate::hall_of_fame::HallOfFame;
use crate::member::Evaluator;
use crate::options::Options;
use crate::population::Population;
use crate::search::regularized_evolution::{reg_evol_cycle, RegEvolCtx};
use dynamic_expressions::operators::scalar::ScalarOpSet;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::Rng;

pub struct IterationCtx<'a, T: Float, Ops, const D: usize, R: Rng> {
    pub rng: &'a mut R,
    pub dataset: &'a crate::dataset::Dataset<T>,
    pub curmaxsize: usize,
    pub stats: &'a RunningSearchStatistics,
    pub options: &'a Options<T, D>,
    pub evaluator: &'a mut Evaluator<T, D>,
    pub grad_ctx: &'a mut dynamic_expressions::GradContext<T, D>,
    pub next_id: &'a mut u64,
    pub next_birth: &'a mut u64,
    pub _ops: core::marker::PhantomData<Ops>,
}

pub fn s_r_cycle<T, Ops, const D: usize, R: Rng>(
    pop: &mut Population<T, Ops, D>,
    ctx: &mut IterationCtx<'_, T, Ops, D, R>,
) -> (f64, HallOfFame<T, Ops, D>)
where
    T: Float + FromPrimitive + ToPrimitive,
    Ops: ScalarOpSet<T>,
{
    let max_temp = 1.0;
    let min_temp = if ctx.options.annealing { 0.0 } else { 1.0 };
    let ncycles = ctx.options.ncycles_per_iteration.max(1);
    let mut num_evals = 0.0;
    let mut best_seen = HallOfFame::new(ctx.options.maxsize);
    best_seen.update_from_members(&pop.members, ctx.options, ctx.curmaxsize);

    for i in 0..ncycles {
        let temperature = if ncycles <= 1 {
            max_temp
        } else {
            let t = (i as f64) / ((ncycles - 1) as f64);
            max_temp + t * (min_temp - max_temp)
        };
        num_evals += reg_evol_cycle::<T, Ops, D, _>(
            pop,
            RegEvolCtx {
                rng: ctx.rng,
                dataset: ctx.dataset,
                temperature,
                curmaxsize: ctx.curmaxsize,
                stats: ctx.stats,
                options: ctx.options,
                evaluator: ctx.evaluator,
                next_id: ctx.next_id,
                next_birth: ctx.next_birth,
                _ops: core::marker::PhantomData,
            },
        );
        best_seen.update_from_members(&pop.members, ctx.options, ctx.curmaxsize);
    }
    (num_evals, best_seen)
}

pub fn optimize_and_simplify_population<T, Ops, const D: usize, R: Rng>(
    pop: &mut Population<T, Ops, D>,
    ctx: &mut IterationCtx<'_, T, Ops, D, R>,
) -> f64
where
    T: Float + FromPrimitive + ToPrimitive,
    Ops: ScalarOpSet<T>,
{
    let mut num_evals = 0.0;

    if ctx.options.should_simplify {
        // Stub: real simplification would rewrite `expr.nodes` using algebraic rules.
        let _ = ctx.curmaxsize;
    }

    if ctx.options.should_optimize_constants && ctx.options.optimizer_probability > 0.0 {
        for m in &mut pop.members {
            if ctx.rng.gen::<f64>() < ctx.options.optimizer_probability {
                let (improved, evals) = optimize_constants::<T, Ops, D, _>(
                    ctx.rng,
                    m,
                    ctx.dataset,
                    ctx.options,
                    ctx.evaluator,
                    ctx.grad_ctx,
                    ctx.next_birth,
                );
                let _ = improved;
                num_evals += evals;
            }
        }
    }

    for m in &mut pop.members {
        let _ = m.evaluate(ctx.dataset, ctx.options, ctx.evaluator);
        num_evals += 1.0;
    }

    num_evals
}
