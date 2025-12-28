use std::ops::AddAssign;

use fastrand::Rng;
use num_traits::Float;

use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::constant_optimization::{OptimizeConstantsCtx, optimize_constants};
use crate::dataset::TaggedDataset;
use crate::expression::{ConstantOptimizable, ExprExt};
use crate::loss_functions::loss_to_cost;
use crate::mutation_functions;
use crate::options::{MutationWeights, Options};
use crate::pop_member::{Evaluator, MemberId, PopMember};
use crate::random::usize_range_inclusive;
use crate::selection::weighted_index;

pub type CrossoverGenerationResult<T, Ops, const D: usize, E> =
    (PopMember<T, Ops, D, E>, PopMember<T, Ops, D, E>, bool, f64);

#[cfg(test)]
pub fn compress_constants<T: Clone, Ops, const D: usize>(expr: &mut dynamic_expressions::PostfixExpr<T, Ops, D>) {
    dynamic_expressions::compress_constants(expr);
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum MutationChoice {
    MutateConstant,
    MutateOperator,
    MutateFeature,
    SwapOperands,
    RotateTree,
    AddNode,
    InsertNode,
    DeleteNode,
    Simplify,
    Randomize,
    DoNothing,
    Optimize,
}

pub struct NextGenerationCtx<'a, T: Float + AddAssign, Ops, const D: usize> {
    pub rng: &'a mut Rng,
    pub dataset: TaggedDataset<'a, T>,
    pub temperature: f64,
    pub curmaxsize: usize,
    pub stats: &'a RunningSearchStatistics,
    pub options: &'a Options<T, D>,
    pub evaluator: &'a mut Evaluator<T, D>,
    pub next_id: &'a mut u64,
    pub next_birth: &'a mut u64,
    pub _ops: core::marker::PhantomData<Ops>,
}

pub struct CrossoverCtx<'a, T: Float, Ops, const D: usize> {
    pub rng: &'a mut Rng,
    pub dataset: TaggedDataset<'a, T>,
    pub curmaxsize: usize,
    pub options: &'a Options<T, D>,
    pub evaluator: &'a mut Evaluator<T, D>,
    pub next_id: &'a mut u64,
    pub next_birth: &'a mut u64,
    pub _ops: core::marker::PhantomData<Ops>,
}

pub fn condition_mutation_weights<T: Float + AddAssign, Ops, const D: usize, E>(
    weights: &mut MutationWeights,
    member: &PopMember<T, Ops, D, E>,
    options: &Options<T, D>,
    curmaxsize: usize,
    nfeatures: usize,
) where
    Ops: dynamic_expressions::OperatorSet<T = T>,
    E: ExprExt<T, Ops, D>,
{
    if member.expr.is_leaf() {
        weights.mutate_operator = 0.0;
        weights.swap_operands = 0.0;
        weights.delete_node = 0.0;
        weights.simplify = 0.0;
        if member.expr.n_scalars() == 0 {
            weights.optimize = 0.0;
            weights.mutate_constant = 0.0;
        } else {
            weights.mutate_feature = 0.0;
        }
        return;
    }

    if !member.expr.has_binary_op() {
        weights.swap_operands = 0.0;
    }

    let nconst = member.expr.count_constant_nodes();
    weights.mutate_constant *= (nconst.min(8) as f64) / 8.0;

    if !member.expr.feature_mutation_possible(nfeatures) {
        weights.mutate_feature = 0.0;
    }

    let complexity = member.complexity;
    if complexity >= curmaxsize {
        weights.add_node = 0.0;
        weights.insert_node = 0.0;
    }

    if !options.should_simplify {
        weights.simplify = 0.0;
    }

    if !options.should_optimize_constants || options.optimizer_probability == 0.0 || member.expr.n_scalars() == 0 {
        weights.optimize = 0.0;
    }
}

pub fn sample_mutation(rng: &mut Rng, weights: &MutationWeights) -> MutationChoice {
    let choices = [
        (MutationChoice::MutateConstant, weights.mutate_constant),
        (MutationChoice::MutateOperator, weights.mutate_operator),
        (MutationChoice::MutateFeature, weights.mutate_feature),
        (MutationChoice::SwapOperands, weights.swap_operands),
        (MutationChoice::RotateTree, weights.rotate_tree),
        (MutationChoice::AddNode, weights.add_node),
        (MutationChoice::InsertNode, weights.insert_node),
        (MutationChoice::DeleteNode, weights.delete_node),
        (MutationChoice::Simplify, weights.simplify),
        (MutationChoice::Randomize, weights.randomize),
        (MutationChoice::DoNothing, weights.do_nothing),
        (MutationChoice::Optimize, weights.optimize),
    ];
    let w: Vec<f64> = choices.iter().map(|(_, v)| *v).collect();
    let idx = weighted_index(rng, &w);
    choices[idx].0
}

struct MutationOutcome<E> {
    expr: E,
    mutated: bool,
    evals: f64,
    return_immediately: bool,
}

struct MutationApplyCtx<'a, 'd, T: Float + AddAssign, Ops, const D: usize, E>
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
    E: ExprExt<T, Ops, D> + ConstantOptimizable<T, Ops, D>,
{
    rng: &'a mut Rng,
    member: &'a PopMember<T, Ops, D, E>,
    expr: E,
    dataset: TaggedDataset<'d, T>,
    temperature: f64,
    curmaxsize: usize,
    options: &'a Options<T, D>,
    evaluator: &'a mut Evaluator<T, D>,
}

impl MutationChoice {
    #[allow(clippy::too_many_arguments)]
    fn apply<T, Ops, const D: usize, E>(self, ctx: MutationApplyCtx<'_, '_, T, Ops, D, E>) -> MutationOutcome<E>
    where
        T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive + AddAssign,
        Ops: dynamic_expressions::OperatorSet<T = T>,
        E: ExprExt<T, Ops, D> + ConstantOptimizable<T, Ops, D>,
    {
        let MutationApplyCtx {
            rng,
            member,
            mut expr,
            dataset,
            temperature,
            curmaxsize,
            options,
            evaluator,
        } = ctx;
        let n_features = dataset.n_features;
        match self {
            MutationChoice::MutateConstant => MutationOutcome {
                mutated: expr.mutate_constant(rng, temperature, options),
                expr,
                evals: 0.0,
                return_immediately: false,
            },
            MutationChoice::MutateOperator => MutationOutcome {
                mutated: {
                    let i = rng.usize(0..expr.n_trees());
                    mutation_functions::mutate_operator_in_place(rng, expr.tree_mut(i), &options.operators)
                },
                expr,
                evals: 0.0,
                return_immediately: false,
            },
            MutationChoice::MutateFeature => MutationOutcome {
                mutated: {
                    let i = rng.usize(0..expr.n_trees());
                    let nf = expr.tree_nfeatures(i, n_features);
                    mutation_functions::mutate_feature_in_place(rng, expr.tree_mut(i), nf)
                },
                expr,
                evals: 0.0,
                return_immediately: false,
            },
            MutationChoice::SwapOperands => MutationOutcome {
                mutated: {
                    let i = rng.usize(0..expr.n_trees());
                    mutation_functions::swap_operands_in_place(rng, expr.tree_mut(i))
                },
                expr,
                evals: 0.0,
                return_immediately: false,
            },
            MutationChoice::RotateTree => MutationOutcome {
                mutated: {
                    let i = rng.usize(0..expr.n_trees());
                    mutation_functions::rotate_tree_in_place(rng, expr.tree_mut(i))
                },
                expr,
                evals: 0.0,
                return_immediately: false,
            },
            MutationChoice::AddNode => MutationOutcome {
                mutated: {
                    let i = rng.usize(0..expr.n_trees());
                    let nf = expr.tree_nfeatures(i, n_features);
                    mutation_functions::add_node_in_place(rng, expr.tree_mut(i), &options.operators, nf)
                },
                expr,
                evals: 0.0,
                return_immediately: false,
            },
            MutationChoice::InsertNode => MutationOutcome {
                mutated: {
                    let i = rng.usize(0..expr.n_trees());
                    let nf = expr.tree_nfeatures(i, n_features);
                    mutation_functions::insert_random_op_in_place(rng, expr.tree_mut(i), &options.operators, nf)
                },
                expr,
                evals: 0.0,
                return_immediately: false,
            },
            MutationChoice::DeleteNode => MutationOutcome {
                mutated: {
                    let i = rng.usize(0..expr.n_trees());
                    mutation_functions::delete_random_op_in_place(rng, expr.tree_mut(i))
                },
                expr,
                evals: 0.0,
                return_immediately: false,
            },
            MutationChoice::Simplify => {
                let _ = expr.simplify_in_place(&evaluator.eval_opts);
                MutationOutcome {
                    mutated: true,
                    expr,
                    evals: 0.0,
                    return_immediately: true,
                }
            }
            MutationChoice::Randomize => {
                // Match SymbolicRegression.jl: sample a *uniform* random size in 1:curmaxsize.
                let max_size = curmaxsize.max(1).min(options.maxsize.max(1));
                let target_size = usize_range_inclusive(rng, 1..=max_size);
                MutationOutcome {
                    mutated: true,
                    expr: expr.randomize(rng, &options.operators, n_features, target_size),
                    evals: 0.0,
                    return_immediately: false,
                }
            }
            MutationChoice::DoNothing => MutationOutcome {
                mutated: true,
                expr,
                evals: 0.0,
                return_immediately: false,
            },
            MutationChoice::Optimize => {
                // Match SymbolicRegression.jl: `:optimize` is a mutation that runs constant
                // optimization without structural changes.
                let mut tmp = PopMember::from_expr(MemberId(0), None, 0, expr, n_features);
                // Avoid consuming global birth counters: the caller already assigns birth/id.
                let orig_birth = tmp.birth;
                let mut dummy_next_birth = orig_birth;

                // Preserve cached plan/loss/cost as the starting point.
                tmp.plans = member.plans.clone();
                tmp.complexity = member.complexity;
                tmp.loss = member.loss;
                tmp.cost = member.cost;

                let mut grad_ctx = dynamic_expressions::GradContext::new(dataset.n_rows);
                let (_improved, evals) = optimize_constants(
                    rng,
                    &mut tmp,
                    OptimizeConstantsCtx {
                        dataset,
                        options,
                        evaluator,
                        grad_ctx: &mut grad_ctx,
                        next_birth: &mut dummy_next_birth,
                    },
                );
                tmp.birth = orig_birth;

                MutationOutcome {
                    mutated: true,
                    expr: tmp.expr,
                    evals,
                    return_immediately: false,
                }
            }
        }
    }
}

pub fn next_generation<
    T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive + AddAssign,
    Ops,
    const D: usize,
    E,
>(
    member: &PopMember<T, Ops, D, E>,
    ctx: NextGenerationCtx<'_, T, Ops, D>,
) -> (PopMember<T, Ops, D, E>, bool, f64)
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
    E: ExprExt<T, Ops, D> + ConstantOptimizable<T, Ops, D>,
{
    let NextGenerationCtx {
        rng,
        dataset,
        temperature,
        curmaxsize,
        stats,
        options,
        evaluator,
        next_id,
        next_birth,
        ..
    } = ctx;

    let before_cost = member.cost.to_f64().unwrap_or(f64::INFINITY);
    let _before_loss = member.loss.to_f64().unwrap_or(f64::INFINITY);
    let n_features = dataset.n_features;

    let mut weights = options.mutation_weights.clone();
    condition_mutation_weights(&mut weights, member, options, curmaxsize, n_features);
    let choice = sample_mutation(rng, &weights);

    let max_attempts = 10;
    let mut successful = false;
    let mut return_immediately = false;
    let mut expr = member.expr.clone();
    let mut evals = 0.0f64;

    for _ in 0..max_attempts {
        let outcome = choice.apply(MutationApplyCtx {
            rng,
            member,
            expr: member.expr.clone(),
            dataset,
            temperature,
            curmaxsize,
            options,
            evaluator,
        });
        evals += outcome.evals;
        if !outcome.mutated {
            continue;
        }
        expr = outcome.expr;
        expr.compress_constants();
        if expr.check_constraints(options, curmaxsize) {
            successful = true;
            return_immediately = outcome.return_immediately;
            break;
        }
    }

    let id = MemberId(*next_id);
    *next_id += 1;
    let birth = *next_birth;
    *next_birth += 1;

    if !successful {
        let mut baby = PopMember::from_expr(id, Some(member.id), birth, member.expr.clone(), n_features);
        baby.complexity = member.complexity;
        baby.loss = member.loss;
        baby.cost = member.cost;
        return (baby, false, 0.0);
    }

    if return_immediately {
        let mut baby = PopMember::from_expr(id, Some(member.id), birth, expr, n_features);
        baby.rebuild_plan(n_features);
        baby.loss = member.loss;
        baby.complexity = baby.expr.complexity(options);
        baby.cost = loss_to_cost(
            baby.loss,
            baby.complexity,
            options.parsimony,
            options.use_baseline,
            dataset.baseline_loss,
        );
        return (baby, true, 0.0);
    }

    let mut baby = PopMember::from_expr(id, Some(member.id), birth, expr, n_features);
    let ok = baby.evaluate(&dataset, options, evaluator);
    evals += 1.0;
    let after_cost = baby.cost.to_f64().unwrap_or(f64::INFINITY);
    let after_loss = baby.loss.to_f64().unwrap_or(f64::INFINITY);
    let _ = after_loss;
    if !ok || !after_cost.is_finite() {
        let mut reject = PopMember::from_expr(id, Some(member.id), birth, member.expr.clone(), n_features);
        reject.complexity = member.complexity;
        reject.loss = member.loss;
        reject.cost = member.cost;
        return (reject, false, 0.0);
    }

    let mut prob = 1.0f64;
    if options.annealing {
        let delta = after_cost - before_cost;
        prob *= (-delta / (temperature * options.alpha)).exp();
    }
    if options.use_frequency {
        let old_size = member.complexity;
        let new_size = baby.complexity;
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

    if prob < rng.f64() {
        let mut reject = PopMember::from_expr(id, Some(member.id), birth, member.expr.clone(), n_features);
        reject.complexity = member.complexity;
        reject.loss = member.loss;
        reject.cost = member.cost;
        return (reject, false, evals);
    }

    (baby, true, evals)
}

pub fn crossover_generation<T: Float + AddAssign, Ops, const D: usize, E>(
    member1: &PopMember<T, Ops, D, E>,
    member2: &PopMember<T, Ops, D, E>,
    ctx: CrossoverCtx<'_, T, Ops, D>,
) -> CrossoverGenerationResult<T, Ops, D, E>
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
    E: ExprExt<T, Ops, D>,
{
    let CrossoverCtx {
        rng,
        dataset,
        curmaxsize,
        options,
        evaluator,
        next_id,
        next_birth,
        ..
    } = ctx;

    let max_tries = 10;
    let mut tries = 0;
    loop {
        let i1 = rng.usize(0..member1.expr.n_trees());
        let i2 = rng.usize(0..member2.expr.n_trees());
        let nf1 = member1.expr.tree_nfeatures(i1, dataset.n_features);
        let nf2 = member2.expr.tree_nfeatures(i2, dataset.n_features);
        if nf1 != nf2 {
            tries += 1;
            if tries >= max_tries {
                let baby1 = member1.clone();
                let baby2 = member2.clone();
                return (baby1, baby2, false, 0.0);
            }
            continue;
        }

        let (c1_tree, c2_tree) = mutation_functions::crossover_trees(rng, member1.expr.tree(i1), member2.expr.tree(i2));
        let mut c1_expr = member1.expr.clone();
        let mut c2_expr = member2.expr.clone();
        *c1_expr.tree_mut(i1) = c1_tree;
        *c2_expr.tree_mut(i2) = c2_tree;
        c1_expr.compress_constants();
        c2_expr.compress_constants();
        tries += 1;
        if c1_expr.check_constraints(options, curmaxsize) && c2_expr.check_constraints(options, curmaxsize) {
            let id1 = MemberId(*next_id);
            *next_id += 1;
            let b1 = *next_birth;
            *next_birth += 1;
            let id2 = MemberId(*next_id);
            *next_id += 1;
            let b2 = *next_birth;
            *next_birth += 1;

            let mut baby1 = PopMember::from_expr(id1, Some(member1.id), b1, c1_expr, dataset.n_features);
            let mut baby2 = PopMember::from_expr(id2, Some(member2.id), b2, c2_expr, dataset.n_features);
            let _ = baby1.evaluate(&dataset, options, evaluator);
            let _ = baby2.evaluate(&dataset, options, evaluator);
            return (baby1, baby2, true, 2.0);
        }
        if tries >= max_tries {
            let baby1 = member1.clone();
            let baby2 = member2.clone();
            return (baby1, baby2, false, 0.0);
        }
    }
}
