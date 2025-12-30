use std::ops::AddAssign;

use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::node::PNode;
use fastrand::Rng;
use num_traits::Float;

use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::check_constraints::check_constraints;
use crate::complexity::compute_complexity;
use crate::constant_optimization::{OptimizeConstantsCtx, optimize_constants};
use crate::dataset::TaggedDataset;
use crate::loss_functions::loss_to_cost;
use crate::mutation_functions;
use crate::options::{MutationWeights, Options};
use crate::pop_member::{Evaluator, PopMember};
use crate::random::usize_range_inclusive;
use crate::selection::weighted_index;

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
    pub _ops: core::marker::PhantomData<Ops>,
}

pub struct CrossoverCtx<'a, T: Float, Ops, const D: usize> {
    pub rng: &'a mut Rng,
    pub dataset: TaggedDataset<'a, T>,
    pub curmaxsize: usize,
    pub options: &'a Options<T, D>,
    pub evaluator: &'a mut Evaluator<T, D>,
    pub _ops: core::marker::PhantomData<Ops>,
}

fn count_constants(nodes: &[PNode]) -> usize {
    nodes.iter().filter(|n| matches!(n, PNode::Const { .. })).count()
}

pub fn condition_mutation_weights<T: Float + AddAssign, Ops, const D: usize>(
    weights: &mut MutationWeights,
    member: &PopMember<T, Ops, D>,
    options: &Options<T, D>,
    curmaxsize: usize,
    nfeatures: usize,
) {
    let tree_is_leaf = member
        .expr
        .nodes
        .iter()
        .all(|n| matches!(n, PNode::Var { .. } | PNode::Const { .. }));
    if tree_is_leaf {
        weights.mutate_operator = 0.0;
        weights.swap_operands = 0.0;
        weights.delete_node = 0.0;
        weights.simplify = 0.0;
        if member.expr.consts.is_empty() {
            weights.optimize = 0.0;
            weights.mutate_constant = 0.0;
        } else {
            weights.mutate_feature = 0.0;
        }
        return;
    }

    if !member.expr.nodes.iter().any(mutation_functions::is_swappable_op) {
        weights.swap_operands = 0.0;
    }

    let nconst = count_constants(&member.expr.nodes);
    weights.mutate_constant *= (nconst.min(8) as f64) / 8.0;

    if nfeatures <= 1 {
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

enum MutationResult<T: Float + AddAssign, Ops, const D: usize> {
    ProposedExpr { expr: PostfixExpr<T, Ops, D>, evals: f64 },
    ProposedMember { member: PopMember<T, Ops, D>, evals: f64 },
}

struct MutationApplyCtx<'a, 'd, T: Float + AddAssign, Ops, const D: usize> {
    rng: &'a mut Rng,
    member: &'a PopMember<T, Ops, D>,
    expr: PostfixExpr<T, Ops, D>,
    dataset: TaggedDataset<'d, T>,
    temperature: f64,
    curmaxsize: usize,
    options: &'a Options<T, D>,
    evaluator: &'a mut Evaluator<T, D>,
}

impl MutationChoice {
    #[allow(clippy::too_many_arguments)]
    fn apply<T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive + AddAssign, Ops, const D: usize>(
        self,
        ctx: MutationApplyCtx<'_, '_, T, Ops, D>,
    ) -> MutationResult<T, Ops, D>
    where
        Ops: dynamic_expressions::OperatorSet<T = T>,
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
            MutationChoice::MutateConstant => {
                let _ = mutation_functions::mutate_constant_in_place(rng, &mut expr, temperature, options);
                MutationResult::ProposedExpr { expr, evals: 0.0 }
            }
            MutationChoice::MutateOperator => {
                let _ = mutation_functions::mutate_operator_in_place(rng, &mut expr, &options.operators);
                MutationResult::ProposedExpr { expr, evals: 0.0 }
            }
            MutationChoice::MutateFeature => {
                mutation_functions::mutate_feature_in_place(rng, &mut expr, n_features);
                MutationResult::ProposedExpr { expr, evals: 0.0 }
            }
            MutationChoice::SwapOperands => {
                let _ = mutation_functions::swap_operands_in_place(rng, &mut expr);
                MutationResult::ProposedExpr { expr, evals: 0.0 }
            }
            MutationChoice::RotateTree => {
                let _ = mutation_functions::rotate_tree_in_place(rng, &mut expr);
                MutationResult::ProposedExpr { expr, evals: 0.0 }
            }
            MutationChoice::AddNode => {
                let _ = mutation_functions::add_node_in_place(rng, &mut expr, &options.operators, n_features);
                MutationResult::ProposedExpr { expr, evals: 0.0 }
            }
            MutationChoice::InsertNode => {
                let _ = mutation_functions::insert_random_op_in_place(rng, &mut expr, &options.operators, n_features);
                MutationResult::ProposedExpr { expr, evals: 0.0 }
            }
            MutationChoice::DeleteNode => {
                let _ = mutation_functions::delete_random_op_in_place(rng, &mut expr);
                MutationResult::ProposedExpr { expr, evals: 0.0 }
            }
            MutationChoice::Simplify => {
                let _ = dynamic_expressions::simplify_in_place(&mut expr, &evaluator.eval_opts);
                let mut out = PopMember::from_expr(expr, n_features, options);

                // Match the intended behavior (and current SymbolicRegression.jl main):
                // simplify returns immediately and keeps the old loss, but refreshes complexity/cost.
                out.loss = member.loss;
                out.complexity = compute_complexity(&out.expr.nodes, options);
                out.cost = loss_to_cost(
                    out.loss,
                    out.complexity,
                    options.parsimony,
                    options.use_baseline,
                    dataset.baseline_loss,
                );

                MutationResult::ProposedMember {
                    member: out,
                    evals: 0.0,
                }
            }
            MutationChoice::Randomize => {
                // Match SymbolicRegression.jl: sample a *uniform* random size in 1:curmaxsize.
                let max_size = curmaxsize.max(1).min(options.maxsize.max(1));
                let target_size = usize_range_inclusive(rng, 1..=max_size);
                let expr = mutation_functions::random_expr(rng, &options.operators, n_features, target_size);
                MutationResult::ProposedExpr { expr, evals: 0.0 }
            }
            MutationChoice::DoNothing => {
                // Match SymbolicRegression.jl: identity mutation is accepted immediately and keeps
                // the old loss/cost.
                let mut out = PopMember::from_expr(expr, n_features, options);
                out.plan = member.plan.clone();
                out.complexity = member.complexity;
                out.loss = member.loss;
                out.cost = member.cost;
                MutationResult::ProposedMember {
                    member: out,
                    evals: 0.0,
                }
            }
            MutationChoice::Optimize => {
                let mut out = PopMember::from_expr(expr, n_features, options);

                // Match SymbolicRegression.jl: optimize returns immediately with loss/cost already
                // computed by constant optimization.
                out.plan = member.plan.clone();
                out.complexity = member.complexity;
                out.loss = member.loss;
                out.cost = member.cost;
                out.birth = member.birth;

                let mut grad_ctx = dynamic_expressions::GradContext::new(dataset.n_rows);
                let (_improved, evals) = optimize_constants(
                    rng,
                    &mut out,
                    OptimizeConstantsCtx {
                        dataset,
                        options,
                        evaluator,
                        grad_ctx: &mut grad_ctx,
                    },
                );

                MutationResult::ProposedMember { member: out, evals }
            }
        }
    }
}

pub fn next_generation<
    T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive + AddAssign,
    Ops,
    const D: usize,
>(
    member: &PopMember<T, Ops, D>,
    ctx: NextGenerationCtx<'_, T, Ops, D>,
) -> (PopMember<T, Ops, D>, bool, f64)
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    let NextGenerationCtx {
        rng,
        dataset,
        temperature,
        curmaxsize,
        stats,
        options,
        evaluator,
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
    let mut tree = member.expr.clone();
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
        match outcome {
            MutationResult::ProposedExpr { expr, evals: e } => {
                evals += e;
                if check_constraints(&expr, options, curmaxsize) {
                    successful = true;
                    tree = expr;
                    break;
                }
            }
            MutationResult::ProposedMember {
                member: out,
                evals: delta_evals,
            } => {
                evals += delta_evals;
                return (out, true, evals);
            }
        }
    }

    if !successful {
        let mut baby = PopMember::from_expr(member.expr.clone(), n_features, options);
        baby.complexity = member.complexity;
        baby.loss = member.loss;
        baby.cost = member.cost;
        return (baby, false, 0.0);
    }

    let mut baby = PopMember::from_expr(tree, n_features, options);
    let _ok = baby.evaluate(&dataset, options, evaluator);
    evals += 1.0;
    let after_cost = baby.cost.to_f64().unwrap_or(f64::INFINITY);
    if after_cost.is_nan() {
        let mut reject = PopMember::from_expr(member.expr.clone(), n_features, options);
        reject.complexity = member.complexity;
        reject.loss = member.loss;
        reject.cost = member.cost;
        return (reject, false, evals);
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
        let mut reject = PopMember::from_expr(member.expr.clone(), n_features, options);
        reject.complexity = member.complexity;
        reject.loss = member.loss;
        reject.cost = member.cost;
        return (reject, false, evals);
    }

    (baby, true, evals)
}

pub fn crossover_generation<T: Float + AddAssign, Ops, const D: usize>(
    member1: &PopMember<T, Ops, D>,
    member2: &PopMember<T, Ops, D>,
    ctx: CrossoverCtx<'_, T, Ops, D>,
) -> (PopMember<T, Ops, D>, PopMember<T, Ops, D>, bool, f64)
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    let CrossoverCtx {
        rng,
        dataset,
        curmaxsize,
        options,
        evaluator,
        ..
    } = ctx;

    let max_tries = 10;
    let mut tries = 0;
    loop {
        let (c1_expr, c2_expr) = mutation_functions::crossover_trees(rng, &member1.expr, &member2.expr);
        tries += 1;
        if check_constraints(&c1_expr, options, curmaxsize) && check_constraints(&c2_expr, options, curmaxsize) {
            let mut baby1 = PopMember::from_expr(c1_expr, dataset.n_features, options);
            let mut baby2 = PopMember::from_expr(c2_expr, dataset.n_features, options);
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
