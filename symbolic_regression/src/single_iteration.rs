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
#[cfg(wgpu)]
use crate::{
    gpu::{LM_EVAL_PASSES_PER_STEP, LmGridParams, MAX_CONSTS, pack_expr},
    loss_functions::{LossKind, loss_to_cost},
    pop_member::get_birth_order,
    random::standard_normal,
};

pub struct IterationCtx<'a, T: Float + AddAssign, Ops, const D: usize> {
    pub rng: &'a mut Rng,
    pub full_dataset: TaggedDataset<'a, T>,
    pub curmaxsize: usize,
    pub stats: &'a RunningSearchStatistics,
    pub options: &'a Options<T, D>,
    pub evaluator: &'a mut Evaluator<T, D>,
    pub grad_ctx: &'a mut dynamic_expressions::GradContext<T, D>,
    #[cfg(wgpu)]
    pub gpu: Option<&'a crate::gpu::GpuClient>,
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
                #[cfg(wgpu)]
                gpu: ctx.gpu,
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
        ctx.evaluator.ensure_n_rows(opt_dataset.data.n_rows);
        ctx.grad_ctx.n_rows = opt_dataset.data.n_rows;

        #[cfg(wgpu)]
        if let Some(gpu) = ctx.gpu.filter(|g| {
            ctx.options.loss_kind == LossKind::Mse
                && opt_dataset.data.n_rows == g.n_rows
                && opt_dataset.data.n_features == g.n_features
        }) {
            // Batched GPU constant optimization:
            // Optimize many members (and their random restarts) in a single fused LM-grid kernel call.
            let n_restarts = 1 + ctx.options.optimizer_nrestarts;
            let params = LmGridParams {
                steps: (ctx.options.optimizer_iterations as u32).clamp(1, 64),
                ..Default::default()
            };

            let mut packed_programs: Vec<crate::gpu::PackedProgram> = Vec::new();
            let mut map: Vec<(usize, usize)> = Vec::new(); // packed_idx -> (member_idx, restart_idx)
            let mut selected_members: Vec<usize> = Vec::new();
            let mut cpu_fallback: Vec<usize> = Vec::new();

            for (mi, m) in pop.members.iter().enumerate() {
                if ctx.controller.is_cancelled() {
                    break;
                }
                if ctx.rng.f64() >= ctx.options.optimizer_probability {
                    continue;
                }
                if m.expr.consts.is_empty() {
                    continue;
                }

                if let Some(base) = pack_expr(&m.expr) {
                    selected_members.push(mi);

                    let n_consts = m.expr.consts.len().min(MAX_CONSTS);
                    for r in 0..n_restarts {
                        let mut p = base;
                        if r > 0 {
                            for (dst, &src) in p
                                .consts
                                .iter_mut()
                                .take(n_consts)
                                .zip(base.consts.iter().take(n_consts))
                            {
                                let scale = 1.0 + 0.5 * (standard_normal(ctx.rng) as f32);
                                *dst = src * scale;
                            }
                        }
                        packed_programs.push(p);
                        map.push((mi, r));
                    }
                } else {
                    cpu_fallback.push(mi);
                }
            }

            if !packed_programs.is_empty() {
                let mut losses = vec![0.0f32; packed_programs.len()];
                gpu.optimize_lm_grid_many(&mut packed_programs, params, &mut losses);

                let mut best_loss: Vec<f32> = vec![f32::INFINITY; pop.members.len()];
                let mut best_consts: Vec<[f32; MAX_CONSTS]> = vec![[0.0f32; MAX_CONSTS]; pop.members.len()];
                let mut has_best: Vec<bool> = vec![false; pop.members.len()];

                for (pi, (mi, _r)) in map.iter().enumerate() {
                    let l = losses[pi];
                    if l.is_finite() && l < best_loss[*mi] {
                        best_loss[*mi] = l;
                        best_consts[*mi] = packed_programs[pi].consts;
                        has_best[*mi] = true;
                    }
                }

                for &mi in &selected_members {
                    if !has_best[mi] {
                        continue;
                    }

                    // Only accept if it actually improved the member (same semantics as CPU path).
                    let baseline = pop.members[mi].loss.to_f32().unwrap_or(f32::INFINITY);
                    if best_loss[mi] < baseline {
                        let n_consts = pop.members[mi].expr.consts.len().min(MAX_CONSTS);
                        for (dst, &src) in pop.members[mi]
                            .expr
                            .consts
                            .iter_mut()
                            .take(n_consts)
                            .zip(best_consts[mi].iter().take(n_consts))
                        {
                            *dst = T::from_f32(src).unwrap_or_else(T::nan);
                        }

                        pop.members[mi].loss = T::from_f32(best_loss[mi]).unwrap_or_else(T::nan);
                        pop.members[mi].cost = loss_to_cost(
                            pop.members[mi].loss,
                            pop.members[mi].complexity,
                            ctx.options.parsimony,
                            ctx.options.use_baseline,
                            opt_dataset.baseline_loss,
                        );
                        pop.members[mi].birth = get_birth_order(ctx.options.deterministic);
                    }
                }

                num_evals += (params.steps as f64)
                    * (LM_EVAL_PASSES_PER_STEP as f64)
                    * (n_restarts as f64)
                    * (selected_members.len() as f64);
            }

            // CPU fallback for programs that can't be represented on the GPU.
            for mi in cpu_fallback {
                if ctx.controller.is_cancelled() {
                    break;
                }
                let (_, evals) = optimize_constants(
                    ctx.rng,
                    &mut pop.members[mi],
                    OptimizeConstantsCtx {
                        dataset: opt_dataset,
                        options: ctx.options,
                        evaluator: ctx.evaluator,
                        grad_ctx: ctx.grad_ctx,
                        gpu: ctx.gpu,
                    },
                );
                num_evals += evals;
            }
        } else {
            for m in &mut pop.members {
                if ctx.rng.f64() < ctx.options.optimizer_probability {
                    let (_, evals) = optimize_constants(
                        ctx.rng,
                        m,
                        OptimizeConstantsCtx {
                            dataset: opt_dataset,
                            options: ctx.options,
                            evaluator: ctx.evaluator,
                            grad_ctx: ctx.grad_ctx,
                            gpu: ctx.gpu,
                        },
                    );
                    num_evals += evals;
                }
            }
        }

        #[cfg(not(wgpu))]
        for m in &mut pop.members {
            if ctx.rng.f64() < ctx.options.optimizer_probability {
                let (_, evals) = optimize_constants(
                    ctx.rng,
                    m,
                    OptimizeConstantsCtx {
                        dataset: opt_dataset,
                        options: ctx.options,
                        evaluator: ctx.evaluator,
                        grad_ctx: ctx.grad_ctx,
                    },
                );
                num_evals += evals;
            }
        }
    }

    // Match SymbolicRegression.jl `finalize_costs`: only re-evaluate on the full dataset when
    // batching is enabled (i.e., members were evolved on a batch and need final losses/costs).
    if ctx.options.batching {
        ctx.evaluator.ensure_n_rows(ctx.full_dataset.n_rows);
        #[cfg(wgpu)]
        if let Some(gpu) = ctx.gpu.filter(|g| {
            ctx.options.loss_kind == crate::loss_functions::LossKind::Mse
                && ctx.full_dataset.n_rows == g.n_rows
                && ctx.full_dataset.n_features == g.n_features
        }) {
            let mut packed_programs: Vec<crate::gpu::PackedProgram> = Vec::new();
            let mut packed_indices: Vec<usize> = Vec::new();
            for (i, m) in pop.members.iter().enumerate() {
                if let Some(packed) = crate::gpu::pack_expr(&m.expr) {
                    packed_programs.push(packed);
                    packed_indices.push(i);
                }
            }

            let mut losses: Vec<f32> = vec![0.0; packed_programs.len()];
            gpu.eval_mse_many(&packed_programs, &mut losses);
            {
                let mut packed_cursor: usize = 0;
                for (i, m) in pop.members.iter_mut().enumerate() {
                    if ctx.controller.is_cancelled() {
                        return num_evals;
                    }

                    let mut used_gpu = false;
                    if packed_cursor < packed_indices.len() && packed_indices[packed_cursor] == i {
                        used_gpu = true;
                        m.complexity = crate::complexity::compute_complexity(&m.expr.nodes, ctx.options);
                        let loss = T::from(losses[packed_cursor]).unwrap_or_else(T::nan);
                        packed_cursor += 1;

                        if !loss.is_finite() {
                            m.loss = T::infinity();
                            m.cost = T::infinity();
                        } else {
                            m.loss = loss;
                            m.cost = crate::loss_functions::loss_to_cost(
                                loss,
                                m.complexity,
                                ctx.options.parsimony,
                                ctx.options.use_baseline,
                                ctx.full_dataset.baseline_loss,
                            );
                        }
                    }

                    if !used_gpu {
                        let _ = m.evaluate(&ctx.full_dataset, ctx.options, ctx.evaluator);
                    }

                    num_evals += 1.0;
                }

                return num_evals;
            }
        }

        for m in &mut pop.members {
            if ctx.controller.is_cancelled() {
                return num_evals;
            }
            let _ = m.evaluate_with_gpu(
                &ctx.full_dataset,
                ctx.options,
                ctx.evaluator,
                #[cfg(wgpu)]
                ctx.gpu,
            );
            num_evals += 1.0;
        }
    }

    num_evals
}
