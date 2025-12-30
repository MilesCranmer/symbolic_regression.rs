use std::ops::AddAssign;

use fastrand::Rng;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::dataset::TaggedDataset;
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
use crate::gpu::pack_expr;
use crate::mutate::{self, CrossoverCtx, NextGenerationCtx};
use crate::options::{OperatorSet, Options};
use crate::pop_member::Evaluator;
use crate::population::Population;
use crate::selection::best_of_sample;
use crate::stop_controller::StopController;
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
use crate::{
    check_constraints::check_constraints,
    complexity::compute_complexity,
    loss_functions::{LossKind, loss_to_cost},
    mutate::MutationResult,
    pop_member::{PopMember, get_birth_order},
};

pub struct RegEvolCtx<'a, T: Float + AddAssign, Ops, const D: usize> {
    pub rng: &'a mut Rng,
    pub dataset: TaggedDataset<'a, T>,
    pub temperature: f64,
    pub curmaxsize: usize,
    pub stats: &'a RunningSearchStatistics,
    pub options: &'a Options<T, D>,
    pub evaluator: &'a mut Evaluator<T, D>,
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub gpu: Option<&'a crate::gpu::GpuClient>,
    pub controller: &'a StopController,
    pub _ops: core::marker::PhantomData<Ops>,
}

pub fn reg_evol_cycle<T, Ops, const D: usize>(pop: &mut Population<T, Ops, D>, ctx: RegEvolCtx<'_, T, Ops, D>) -> f64
where
    T: Float + AddAssign + FromPrimitive + ToPrimitive,
    Ops: OperatorSet<T = T>,
{
    let n_evol_cycles = ((pop.len() as f64) / (ctx.options.tournament_selection_n as f64)).ceil() as usize;

    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    {
        if let Some(gpu) = ctx.gpu.filter(|g| {
            ctx.options.loss_kind == LossKind::Mse
                && ctx.dataset.data.n_rows == g.n_rows
                && ctx.dataset.data.n_features == g.n_features
        }) {
            return reg_evol_cycle_batched_gpu(pop, ctx, gpu, n_evol_cycles);
        }
    }

    // Fallback path (CPU or legacy GPU-per-eval path).
    let mut num_evals = 0.0;
    for _ in 0..n_evol_cycles {
        if ctx.controller.is_cancelled() {
            break;
        }
        let allstar = best_of_sample(ctx.rng, pop, ctx.stats, ctx.options);

        if ctx.rng.f64() > ctx.options.crossover_probability {
            let (baby, accepted, evals) = mutate::next_generation(
                &allstar,
                NextGenerationCtx {
                    rng: ctx.rng,
                    dataset: ctx.dataset,
                    temperature: ctx.temperature,
                    curmaxsize: ctx.curmaxsize,
                    stats: ctx.stats,
                    options: ctx.options,
                    evaluator: ctx.evaluator,
                    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
                    gpu: ctx.gpu,
                    _ops: core::marker::PhantomData,
                },
            );
            num_evals += evals;
            if !accepted && ctx.options.skip_mutation_failures {
                continue;
            }
            pop.replace_oldest(baby);
        } else {
            let allstar2 = best_of_sample(ctx.rng, pop, ctx.stats, ctx.options);
            let (baby1, baby2, accepted, evals) = mutate::crossover_generation(
                &allstar,
                &allstar2,
                CrossoverCtx {
                    rng: ctx.rng,
                    dataset: ctx.dataset,
                    curmaxsize: ctx.curmaxsize,
                    options: ctx.options,
                    evaluator: ctx.evaluator,
                    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
                    gpu: ctx.gpu,
                    _ops: core::marker::PhantomData,
                },
            );
            num_evals += evals;
            if !accepted && ctx.options.skip_mutation_failures {
                continue;
            }
            pop.replace_two_oldest(baby1, baby2);
        }
    }
    num_evals
}

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
fn reg_evol_cycle_batched_gpu<T, Ops, const D: usize>(
    pop: &mut Population<T, Ops, D>,
    ctx: RegEvolCtx<'_, T, Ops, D>,
    gpu: &crate::gpu::GpuClient,
    n_evol_cycles: usize,
) -> f64
where
    T: Float + AddAssign + FromPrimitive + ToPrimitive,
    Ops: OperatorSet<T = T>,
{
    /// Work item pending a (batched) GPU evaluation.
    enum PendingOp<T: Float + AddAssign, Ops: OperatorSet<T = T>, const D: usize> {
        Mut {
            parent: PopMember<T, Ops, D>,
            expr: dynamic_expressions::expression::PostfixExpr<T, Ops, D>,
            old_size: usize,
            new_size: usize,
            evals_before_eval: f64,
            before_cost: f64,
        },
        Cross {
            parent1: PopMember<T, Ops, D>,
            parent2: PopMember<T, Ops, D>,
            expr1: dynamic_expressions::expression::PostfixExpr<T, Ops, D>,
            expr2: dynamic_expressions::expression::PostfixExpr<T, Ops, D>,
            size1: usize,
            size2: usize,
        },
    }

    #[allow(clippy::too_many_arguments)]
    fn accept_mutation_td<T: Float, const D: usize>(
        rng: &mut Rng,
        options: &Options<T, D>,
        stats: &RunningSearchStatistics,
        before_cost: f64,
        after_cost: f64,
        old_size: usize,
        new_size: usize,
        temperature: f64,
    ) -> bool {
        if after_cost.is_nan() {
            return false;
        }

        let mut prob = 1.0;
        if options.annealing {
            let temp = if temperature > 1e-6 { temperature } else { 1e-6 };
            let delta = after_cost - before_cost;
            prob = (-delta / temp).exp();
        }

        if options.use_frequency {
            let old_f = if old_size > 0 && old_size <= options.maxsize {
                stats.freq(old_size)
            } else {
                1e-6
            };
            let new_f = if new_size > 0 && new_size <= options.maxsize {
                stats.freq(new_size)
            } else {
                1e-6
            };
            prob *= old_f / new_f;
        }

        prob >= rng.f64()
    }

    let max_possible = n_evol_cycles.saturating_mul(2);
    let flush_at = gpu.max_batch().min(max_possible).max(1);

    let mut pending: Vec<PendingOp<T, Ops, D>> = Vec::with_capacity(flush_at);
    let mut pending_programs: usize = 0;

    let mut num_evals: f64 = 0.0;

    // Flush helper: evaluate all queued programs in one go, then insert accepted children.
    #[allow(clippy::too_many_arguments)]
    fn flush_pending<T, Ops, const D: usize>(
        pop: &mut Population<T, Ops, D>,
        rng: &mut Rng,
        dataset: TaggedDataset<'_, T>,
        temperature: f64,
        _curmaxsize: usize,
        stats: &RunningSearchStatistics,
        options: &Options<T, D>,
        evaluator: &mut Evaluator<T, D>,
        gpu: &crate::gpu::GpuClient,
        pending: &mut Vec<PendingOp<T, Ops, D>>,
        pending_programs: &mut usize,
        num_evals: &mut f64,
    ) where
        T: Float + AddAssign + FromPrimitive + ToPrimitive,
        Ops: OperatorSet<T = T>,
    {
        if pending.is_empty() {
            *pending_programs = 0;
            return;
        }

        // Pack and evaluate everything we can on GPU.
        let mut packed: Vec<crate::gpu::PackedProgram> = Vec::with_capacity(*pending_programs);
        let mut map: Vec<(usize, u8)> = Vec::with_capacity(*pending_programs);

        let mut loss0: Vec<Option<f32>> = vec![None; pending.len()];
        let mut loss1: Vec<Option<f32>> = vec![None; pending.len()];

        for (op_i, op) in pending.iter().enumerate() {
            match op {
                PendingOp::Mut { expr, .. } => {
                    if let Some(p) = pack_expr(expr) {
                        packed.push(p);
                        map.push((op_i, 0));
                    }
                }
                PendingOp::Cross { expr1, expr2, .. } => {
                    if let Some(p) = pack_expr(expr1) {
                        packed.push(p);
                        map.push((op_i, 0));
                    }
                    if let Some(p) = pack_expr(expr2) {
                        packed.push(p);
                        map.push((op_i, 1));
                    }
                }
            }
        }

        let mut packed_losses = vec![0.0f32; packed.len()];
        gpu.eval_mse_many(&packed, &mut packed_losses);

        for (packed_i, (op_i, child_i)) in map.iter().enumerate() {
            let loss = packed_losses[packed_i];
            if *child_i == 0 {
                loss0[*op_i] = Some(loss);
            } else {
                loss1[*op_i] = Some(loss);
            }
        }

        // CPU fallback for un-packable children.
        for (op_i, op) in pending.iter().enumerate() {
            match op {
                PendingOp::Mut { expr, .. } => {
                    if loss0[op_i].is_none() {
                        let mut m = PopMember::from_expr(expr.clone(), dataset.data.n_features, options);
                        let ok = m.evaluate(&dataset, options, evaluator);
                        loss0[op_i] = Some(if ok {
                            m.loss.to_f32().unwrap_or(f32::NAN)
                        } else {
                            f32::NAN
                        });
                    }
                }
                PendingOp::Cross { expr1, expr2, .. } => {
                    if loss0[op_i].is_none() {
                        let mut m = PopMember::from_expr(expr1.clone(), dataset.data.n_features, options);
                        let ok = m.evaluate(&dataset, options, evaluator);
                        loss0[op_i] = Some(if ok {
                            m.loss.to_f32().unwrap_or(f32::NAN)
                        } else {
                            f32::NAN
                        });
                    }
                    if loss1[op_i].is_none() {
                        let mut m = PopMember::from_expr(expr2.clone(), dataset.data.n_features, options);
                        let ok = m.evaluate(&dataset, options, evaluator);
                        loss1[op_i] = Some(if ok {
                            m.loss.to_f32().unwrap_or(f32::NAN)
                        } else {
                            f32::NAN
                        });
                    }
                }
            }
        }

        // Consume pending ops and insert results.
        for (op_i, op) in pending.drain(..).enumerate() {
            match op {
                PendingOp::Mut {
                    parent,
                    expr,
                    old_size,
                    new_size,
                    evals_before_eval,
                    before_cost,
                } => {
                    *num_evals += evals_before_eval + 1.0;

                    let loss_f32 = loss0[op_i].unwrap_or(f32::NAN);
                    if !loss_f32.is_finite() {
                        // Treat non-finite as a failed mutation.
                        if !options.skip_mutation_failures {
                            let mut reject = parent;
                            reject.birth = get_birth_order(options.deterministic);
                            pop.replace_oldest(reject);
                        }
                        continue;
                    }

                    let loss_t = T::from_f32(loss_f32).unwrap_or_else(T::nan);
                    let cost_t = loss_to_cost(
                        loss_t,
                        new_size,
                        options.parsimony,
                        options.use_baseline,
                        dataset.baseline_loss,
                    );
                    let after_cost = cost_t.to_f64().unwrap_or(f64::INFINITY);

                    let accepted = accept_mutation_td(
                        rng,
                        options,
                        stats,
                        before_cost,
                        after_cost,
                        old_size,
                        new_size,
                        temperature,
                    );

                    if accepted {
                        let mut baby = PopMember::from_expr(expr, dataset.data.n_features, options);
                        baby.complexity = new_size;
                        baby.loss = loss_t;
                        baby.cost = cost_t;
                        pop.replace_oldest(baby);
                    } else if !options.skip_mutation_failures {
                        let mut reject = parent;
                        reject.birth = get_birth_order(options.deterministic);
                        pop.replace_oldest(reject);
                    }
                }
                PendingOp::Cross {
                    parent1,
                    parent2,
                    expr1,
                    expr2,
                    size1,
                    size2,
                } => {
                    *num_evals += 2.0;

                    let l1 = loss0[op_i].unwrap_or(f32::NAN);
                    let l2 = loss1[op_i].unwrap_or(f32::NAN);
                    if !l1.is_finite() || !l2.is_finite() {
                        if !options.skip_mutation_failures {
                            pop.replace_two_oldest(parent1, parent2);
                        }
                        continue;
                    }

                    let loss1_t = T::from_f32(l1).unwrap_or_else(T::nan);
                    let loss2_t = T::from_f32(l2).unwrap_or_else(T::nan);

                    let cost1_t = loss_to_cost(
                        loss1_t,
                        size1,
                        options.parsimony,
                        options.use_baseline,
                        dataset.baseline_loss,
                    );
                    let cost2_t = loss_to_cost(
                        loss2_t,
                        size2,
                        options.parsimony,
                        options.use_baseline,
                        dataset.baseline_loss,
                    );

                    let mut baby1 = PopMember::from_expr(expr1, dataset.data.n_features, options);
                    baby1.complexity = size1;
                    baby1.loss = loss1_t;
                    baby1.cost = cost1_t;

                    let mut baby2 = PopMember::from_expr(expr2, dataset.data.n_features, options);
                    baby2.complexity = size2;
                    baby2.loss = loss2_t;
                    baby2.cost = cost2_t;

                    pop.replace_two_oldest(baby1, baby2);
                }
            }
        }

        *pending_programs = 0;
    }

    for _ in 0..n_evol_cycles {
        if ctx.controller.is_cancelled() {
            break;
        }

        let allstar = best_of_sample(ctx.rng, pop, ctx.stats, ctx.options);

        if ctx.rng.f64() > ctx.options.crossover_probability {
            // Mutation.
            let before_cost = allstar.cost.to_f64().unwrap_or(f64::INFINITY);
            let old_size = allstar.complexity;

            let mut successful = false;
            let mut evals_before_eval = 0.0;
            let mut chosen_expr = None;

            for _ in 0..10 {
                let mut weights = ctx.options.mutation_weights.clone();
                mutate::condition_mutation_weights(
                    &mut weights,
                    &allstar,
                    ctx.options,
                    ctx.curmaxsize,
                    ctx.dataset.data.n_features,
                );
                let choice = mutate::sample_mutation(ctx.rng, &weights);
                let res = mutate::apply_mutation_choice(
                    choice,
                    &allstar,
                    NextGenerationCtx {
                        rng: ctx.rng,
                        dataset: ctx.dataset,
                        temperature: ctx.temperature,
                        curmaxsize: ctx.curmaxsize,
                        stats: ctx.stats,
                        options: ctx.options,
                        evaluator: ctx.evaluator,
                        gpu: Some(gpu),
                        _ops: core::marker::PhantomData,
                    },
                );

                match res {
                    MutationResult::ProposedExpr { expr, evals } => {
                        evals_before_eval += evals;
                        if check_constraints(&expr, ctx.options, ctx.curmaxsize) {
                            chosen_expr = Some(expr);
                            successful = true;
                            break;
                        }
                    }
                    MutationResult::ProposedMember { member, evals } => {
                        num_evals += evals;
                        pop.replace_oldest(member);
                        successful = true;
                        break;
                    }
                }
            }

            if !successful {
                if !ctx.options.skip_mutation_failures {
                    let mut reject = allstar;
                    reject.birth = get_birth_order(ctx.options.deterministic);
                    pop.replace_oldest(reject);
                }
                continue;
            }

            if let Some(tree) = chosen_expr {
                let new_size = compute_complexity(&tree.nodes, ctx.options);
                pending.push(PendingOp::Mut {
                    parent: allstar,
                    expr: tree,
                    old_size,
                    new_size,
                    evals_before_eval,
                    before_cost,
                });
                pending_programs += 1;
            }
        } else {
            // Crossover.
            let allstar2 = best_of_sample(ctx.rng, pop, ctx.stats, ctx.options);

            let mut successful = false;
            let mut c1 = allstar.expr.clone();
            let mut c2 = allstar2.expr.clone();

            for _ in 0..10 {
                let (c1_try, c2_try) =
                    crate::mutation_functions::crossover_trees(ctx.rng, &allstar.expr, &allstar2.expr);
                if check_constraints(&c1_try, ctx.options, ctx.curmaxsize)
                    && check_constraints(&c2_try, ctx.options, ctx.curmaxsize)
                {
                    c1 = c1_try;
                    c2 = c2_try;
                    successful = true;
                    break;
                }
            }

            if !successful {
                if !ctx.options.skip_mutation_failures {
                    pop.replace_two_oldest(allstar, allstar2);
                }
                continue;
            }

            let size1 = compute_complexity(&c1.nodes, ctx.options);
            let size2 = compute_complexity(&c2.nodes, ctx.options);

            pending.push(PendingOp::Cross {
                parent1: allstar,
                parent2: allstar2,
                expr1: c1,
                expr2: c2,
                size1,
                size2,
            });
            pending_programs += 2;
        }

        if pending_programs >= flush_at {
            flush_pending(
                pop,
                ctx.rng,
                ctx.dataset,
                ctx.temperature,
                ctx.curmaxsize,
                ctx.stats,
                ctx.options,
                ctx.evaluator,
                gpu,
                &mut pending,
                &mut pending_programs,
                &mut num_evals,
            );
        }
    }

    // Final flush.
    flush_pending(
        pop,
        ctx.rng,
        ctx.dataset,
        ctx.temperature,
        ctx.curmaxsize,
        ctx.stats,
        ctx.options,
        ctx.evaluator,
        gpu,
        &mut pending,
        &mut pending_programs,
        &mut num_evals,
    );

    num_evals
}
