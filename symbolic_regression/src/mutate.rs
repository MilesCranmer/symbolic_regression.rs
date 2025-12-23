use std::ops::AddAssign;

pub use dynamic_expressions::compress_constants;
use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::node::PNode;
use num_traits::Float;
use rand::Rng;
use rand::distr::{self, Distribution};

use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::check_constraints::check_constraints;
use crate::complexity::compute_complexity;
use crate::constant_optimization::{OptimizeConstantsCtx, optimize_constants};
use crate::dataset::TaggedDataset;
use crate::loss_functions::loss_to_cost;
use crate::mutation_functions;
use crate::options::{MutationWeights, Options};
use crate::pop_member::{Evaluator, MemberId, PopMember};

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

pub struct NextGenerationCtx<'a, T: Float + AddAssign, Ops, const D: usize, R: Rng> {
    pub rng: &'a mut R,
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

pub struct CrossoverCtx<'a, T: Float, Ops, const D: usize, R: Rng> {
    pub rng: &'a mut R,
    pub dataset: TaggedDataset<'a, T>,
    pub curmaxsize: usize,
    pub options: &'a Options<T, D>,
    pub evaluator: &'a mut Evaluator<T, D>,
    pub next_id: &'a mut u64,
    pub next_birth: &'a mut u64,
    pub _ops: core::marker::PhantomData<Ops>,
}

fn count_constants(nodes: &[PNode]) -> usize {
    nodes.iter().filter(|n| matches!(n, PNode::Const { .. })).count()
}

fn has_binary_op(nodes: &[PNode]) -> bool {
    nodes.iter().any(|n| matches!(n, PNode::Op { arity: 2, .. }))
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

    if !has_binary_op(&member.expr.nodes) {
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

    if !options.should_optimize_constants || options.optimizer_probability == 0.0 || member.expr.consts.is_empty() {
        weights.optimize = 0.0;
    }
}

pub fn sample_mutation<R: Rng>(rng: &mut R, weights: &MutationWeights) -> MutationChoice {
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
    let dist = distr::weighted::WeightedIndex::new(w).expect("at least one mutation weight must be > 0");
    choices[dist.sample(rng)].0
}

struct MutationOutcome<T: Float + AddAssign, Ops, const D: usize> {
    expr: PostfixExpr<T, Ops, D>,
    mutated: bool,
    evals: f64,
    return_immediately: bool,
}

struct MutationApplyCtx<'a, 'd, T: Float + AddAssign, Ops, const D: usize, R: Rng> {
    rng: &'a mut R,
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
    fn apply<T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive + AddAssign, Ops, const D: usize, R: Rng>(
        self,
        ctx: MutationApplyCtx<'_, '_, T, Ops, D, R>,
    ) -> MutationOutcome<T, Ops, D>
    where
        Ops: dynamic_expressions::operator_enum::scalar::ScalarOpSet<T>
            + dynamic_expressions::operator_registry::OpRegistry,
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
                mutated: mutation_functions::mutate_constant_in_place(rng, &mut expr, temperature, options),
                expr,
                evals: 0.0,
                return_immediately: false,
            },
            MutationChoice::MutateOperator => MutationOutcome {
                mutated: mutation_functions::mutate_operator_in_place(rng, &mut expr, &options.operators),
                expr,
                evals: 0.0,
                return_immediately: false,
            },
            MutationChoice::MutateFeature => MutationOutcome {
                mutated: mutation_functions::mutate_feature_in_place(rng, &mut expr, n_features),
                expr,
                evals: 0.0,
                return_immediately: false,
            },
            MutationChoice::SwapOperands => MutationOutcome {
                mutated: mutation_functions::swap_operands_in_place(rng, &mut expr),
                expr,
                evals: 0.0,
                return_immediately: false,
            },
            MutationChoice::RotateTree => MutationOutcome {
                mutated: mutation_functions::rotate_tree_in_place(rng, &mut expr),
                expr,
                evals: 0.0,
                return_immediately: false,
            },
            MutationChoice::AddNode => MutationOutcome {
                mutated: mutation_functions::add_node_in_place(rng, &mut expr, &options.operators, n_features),
                expr,
                evals: 0.0,
                return_immediately: false,
            },
            MutationChoice::InsertNode => MutationOutcome {
                mutated: mutation_functions::insert_random_op_in_place(rng, &mut expr, &options.operators, n_features),
                expr,
                evals: 0.0,
                return_immediately: false,
            },
            MutationChoice::DeleteNode => MutationOutcome {
                mutated: mutation_functions::delete_random_op_in_place(rng, &mut expr),
                expr,
                evals: 0.0,
                return_immediately: false,
            },
            MutationChoice::Simplify => {
                let _ = dynamic_expressions::simplify_in_place(&mut expr, &evaluator.eval_opts);
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
                let target_size = rng.random_range(1..=max_size);
                MutationOutcome {
                    mutated: true,
                    expr: mutation_functions::random_expr(rng, &options.operators, n_features, target_size),
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
                tmp.plan = member.plan.clone();
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
    R: Rng,
>(
    member: &PopMember<T, Ops, D>,
    ctx: NextGenerationCtx<'_, T, Ops, D, R>,
) -> (PopMember<T, Ops, D>, bool, f64)
where
    Ops:
        dynamic_expressions::operator_enum::scalar::ScalarOpSet<T> + dynamic_expressions::operator_registry::OpRegistry,
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
        evals += outcome.evals;
        if !outcome.mutated {
            continue;
        }
        tree = outcome.expr;
        compress_constants(&mut tree);
        if check_constraints(&tree, options, curmaxsize) {
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
        let mut baby = PopMember::from_expr(id, Some(member.id), birth, tree, n_features);
        baby.rebuild_plan(n_features);
        baby.loss = member.loss;
        baby.complexity = compute_complexity(&baby.expr.nodes, options);
        baby.cost = loss_to_cost(
            baby.loss,
            baby.complexity,
            options.parsimony,
            options.use_baseline,
            dataset.baseline_loss,
        );
        return (baby, true, 0.0);
    }

    let mut baby = PopMember::from_expr(id, Some(member.id), birth, tree, n_features);
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

    if prob < rng.random::<f64>() {
        let mut reject = PopMember::from_expr(id, Some(member.id), birth, member.expr.clone(), n_features);
        reject.complexity = member.complexity;
        reject.loss = member.loss;
        reject.cost = member.cost;
        return (reject, false, evals);
    }

    (baby, true, evals)
}

pub fn crossover_generation<T: Float + AddAssign, Ops, const D: usize, R: Rng>(
    member1: &PopMember<T, Ops, D>,
    member2: &PopMember<T, Ops, D>,
    ctx: CrossoverCtx<'_, T, Ops, D, R>,
) -> (PopMember<T, Ops, D>, PopMember<T, Ops, D>, bool, f64)
where
    Ops: dynamic_expressions::operator_enum::scalar::ScalarOpSet<T>,
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
        let (c1_expr, c2_expr) = mutation_functions::crossover_trees(rng, &member1.expr, &member2.expr);
        tries += 1;
        if check_constraints(&c1_expr, options, curmaxsize) && check_constraints(&c2_expr, options, curmaxsize) {
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
