use std::ops::AddAssign;

use fastrand::Rng;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::dataset::TaggedDataset;
use crate::mutate::{self, CrossoverCtx, NextGenerationCtx};
use crate::options::Options;
use crate::pop_member::Evaluator;
use crate::population::Population;
use crate::selection::best_of_sample;
use crate::stop_controller::StopController;

pub struct RegEvolCtx<'a, T: Float + AddAssign, Ops, const D: usize> {
    pub rng: &'a mut Rng,
    pub dataset: TaggedDataset<'a, T>,
    pub temperature: f64,
    pub curmaxsize: usize,
    pub stats: &'a RunningSearchStatistics,
    pub options: &'a Options<T, D>,
    pub evaluator: &'a mut Evaluator<T, D>,
    pub next_id: &'a mut u64,
    pub controller: &'a StopController,
    pub _ops: core::marker::PhantomData<Ops>,
}

pub fn reg_evol_cycle<T, Ops, const D: usize>(pop: &mut Population<T, Ops, D>, ctx: RegEvolCtx<'_, T, Ops, D>) -> f64
where
    T: Float + FromPrimitive + ToPrimitive + AddAssign,
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    let mut num_evals = 0.0;
    let n_evol_cycles = ((pop.len() as f64) / (ctx.options.tournament_selection_n as f64)).ceil() as usize;

    for _ in 0..n_evol_cycles {
        if ctx.controller.is_cancelled() {
            break;
        }
        if ctx.rng.f64() > ctx.options.crossover_probability {
            let allstar = best_of_sample(ctx.rng, pop, ctx.stats, ctx.options);
            let (baby, accepted, tmp) = mutate::next_generation(
                &allstar,
                NextGenerationCtx {
                    rng: ctx.rng,
                    dataset: ctx.dataset,
                    temperature: ctx.temperature,
                    curmaxsize: ctx.curmaxsize,
                    stats: ctx.stats,
                    options: ctx.options,
                    evaluator: ctx.evaluator,
                    next_id: ctx.next_id,
                    _ops: core::marker::PhantomData,
                },
            );
            num_evals += tmp;
            if !accepted && ctx.options.skip_mutation_failures {
                continue;
            }
            pop.replace_oldest(baby);
        } else {
            let allstar1 = best_of_sample(ctx.rng, pop, ctx.stats, ctx.options);
            let allstar2 = best_of_sample(ctx.rng, pop, ctx.stats, ctx.options);
            let (baby1, baby2, accepted, tmp) = mutate::crossover_generation(
                &allstar1,
                &allstar2,
                CrossoverCtx {
                    rng: ctx.rng,
                    dataset: ctx.dataset,
                    curmaxsize: ctx.curmaxsize,
                    options: ctx.options,
                    evaluator: ctx.evaluator,
                    next_id: ctx.next_id,
                    _ops: core::marker::PhantomData,
                },
            );
            num_evals += tmp;
            if !accepted && ctx.options.skip_mutation_failures {
                continue;
            }
            pop.replace_two_oldest(baby1, baby2);
        }
    }

    num_evals
}
