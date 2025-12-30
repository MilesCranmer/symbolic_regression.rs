use core::marker::PhantomData;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::builtin;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF32;
use dynamic_expressions::utils::ZipEq;
use dynamic_expressions::{HasOp, OpId, Operators};
use fastrand::Rng as FastRand;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use symbolic_regression::operator_selection::OperatorsSampling;
use symbolic_regression::{
    Dataset, Evaluator, MemberId, NextGenerationCtx, OptimizeConstantsCtx, Options, PopMember, Population,
    RunningSearchStatistics, TaggedDataset, best_of_sample, check_constraints, equation_search,
    insert_random_op_in_place, next_generation, optimize_constants, rotate_tree_in_place,
};

type T = f32;
const D: usize = 3;
type Ops = BuiltinOpsF32;
const POP_SIZE: usize = 100;

fn random_leaf<R: Rng>(rng: &mut R, n_features: usize, consts: &mut Vec<T>) -> PNode {
    if rng.random_bool(0.5) {
        let val: T = rng.sample(StandardNormal);
        let idx: u16 = consts.len().try_into().expect("too many constants");
        consts.push(val);
        PNode::Const { idx }
    } else {
        let feature: u16 = rng
            .random_range(0..n_features)
            .try_into()
            .unwrap_or_else(|_| panic!("too many features to index in u16"));
        PNode::Var { feature }
    }
}

fn random_expr<Ops2, const D2: usize, R: Rng>(
    rng: &mut R,
    operators: &Operators<D2>,
    n_features: usize,
    target_size: usize,
) -> PostfixExpr<T, Ops2, D2> {
    assert!(target_size >= 1);
    let mut nodes: Vec<PNode> = Vec::with_capacity(target_size);
    let mut consts: Vec<T> = Vec::new();
    nodes.push(random_leaf(rng, n_features, &mut consts));

    while nodes.len() < target_size && operators.total_ops_up_to(D2.min(target_size - nodes.len())) > 0 {
        let rem = target_size - nodes.len();
        let max_arity = rem.min(D2);
        let total: usize = (1..=max_arity).map(|a| operators.nops(a)).sum();
        let mut r = rng.random_range(0..total);
        let mut arity = 1usize;
        for a in 1..=max_arity {
            let n = operators.nops(a);
            if r < n {
                arity = a;
                break;
            }
            r -= n;
        }

        let choices = &operators.ops_by_arity[arity - 1];
        let op = choices[rng.random_range(0..choices.len())];

        let leaf_positions: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| matches!(n, PNode::Var { .. } | PNode::Const { .. }).then_some(i))
            .collect();
        let leaf_idx = leaf_positions[rng.random_range(0..leaf_positions.len())];

        let mut repl: Vec<PNode> = Vec::with_capacity(arity + 1);
        for _ in 0..arity {
            repl.push(random_leaf(rng, n_features, &mut consts));
        }
        repl.push(PNode::Op {
            arity: arity as u8,
            op: op.id,
        });
        nodes.splice(leaf_idx..=leaf_idx, repl);
    }

    PostfixExpr::new(nodes, consts, Default::default())
}

fn make_ops_search() -> Operators<D> {
    Ops::from_names(["exp", "abs", "+", "sub", "*", "/"]).expect("search operators")
}

fn make_ops_utils() -> Operators<D> {
    Ops::from_names(["sin", "cos", "+", "sub", "*", "/"]).expect("utils operators")
}

fn make_search_options(seed: u64) -> Options<T, D> {
    let mut options = Options::<T, D> {
        seed,
        niterations: 30,
        populations: 1,
        population_size: 64,
        operators: make_ops_search(),
        progress: false,
        ..Default::default()
    };
    options.mutation_weights.swap_operands = 0.0;
    options.mutation_weights.form_connection = 0.0;
    options.mutation_weights.break_connection = 0.0;
    options
}

fn make_utils_options() -> Options<T, D> {
    Options::<T, D> {
        operators: make_ops_utils(),
        progress: false,
        ..Default::default()
    }
}

fn make_dataset(seed: u64, n_rows: usize, n_features: usize) -> Dataset<T> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x: Vec<T> = Vec::with_capacity(n_rows * n_features);
    for _ in 0..n_rows * n_features {
        x.push(rng.random_range(-5.0f32..5.0f32));
    }
    let x_arr = Array2::from_shape_vec((n_features, n_rows), x).unwrap();

    let mut y = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let noise: f32 = rng.sample(StandardNormal);
        let x0 = x_arr[(0, r)];
        let x1 = x_arr[(1, r)];
        let x2 = x_arr[(2, r)];
        let x3 = x_arr[(3, r)];
        let a = (2.13f32 * x0).cos();
        let b = x1 * x2.abs().powf(0.9f32) * 0.5f32;
        let c = x3.abs().powf(1.5f32) * 0.3f32;
        y.push(a + b - c + 0.1f32 * noise);
    }

    Dataset::new(x_arr, Array1::from_vec(y))
}

fn make_random_dataset(seed: u64, n_rows: usize, n_features: usize) -> Dataset<T> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x: Vec<T> = Vec::with_capacity(n_rows * n_features);
    for _ in 0..n_rows * n_features {
        let v: f32 = rng.sample(StandardNormal);
        x.push(v);
    }
    let mut y = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let v: f32 = rng.sample(StandardNormal);
        y.push(v);
    }
    Dataset::new(
        Array2::from_shape_vec((n_features, n_rows), x).unwrap(),
        Array1::from_vec(y),
    )
}

fn make_population(
    seed: u64,
    dataset: &Dataset<T>,
    options: &Options<T, D>,
    pop_size: usize,
    tree_size: usize,
) -> (Population<T, Ops, D>, RunningSearchStatistics) {
    let mut rng = StdRng::seed_from_u64(seed);
    let tagged = TaggedDataset::new(dataset, None);
    let mut evaluator = Evaluator::new(dataset.n_rows);

    let mut members = Vec::with_capacity(pop_size);
    for i in 0..pop_size {
        let expr = random_expr::<Ops, D, _>(&mut rng, &options.operators, dataset.n_features, tree_size);
        let mut member = PopMember::from_expr(MemberId(i as u64), None, expr, dataset.n_features, options);
        let _ = member.evaluate(&tagged, options, &mut evaluator);
        members.push(member);
    }

    let mut stats = RunningSearchStatistics::new(options.maxsize, 100_000);
    stats.update_from_population(members.iter().map(|m| m.complexity));
    stats.normalize();

    (Population::new(members), stats)
}

fn bench_search(c: &mut Criterion) {
    let seeds = [1u64, 2, 3];
    let datasets: Vec<_> = seeds.iter().map(|&seed| make_dataset(seed, 1_000, 5)).collect();
    let options: Vec<_> = seeds.iter().map(|&seed| make_search_options(seed)).collect();

    let mut group = c.benchmark_group("search");
    group.sample_size(10);
    group.bench_function("equation_search", |b| {
        b.iter(|| {
            for (dataset, options) in datasets.iter().zip_eq(&options) {
                let _ = equation_search::<T, Ops, D>(dataset, options);
            }
        })
    });
    group.finish();
}

fn bench_utils(c: &mut Criterion) {
    let mut group = c.benchmark_group("utils");

    // best_of_sample
    {
        let options = make_utils_options();
        let dataset = make_random_dataset(0, 32, 1);
        let (population, stats) = make_population(5, &dataset, &options, POP_SIZE, 20);
        let mut rng = FastRand::with_seed(99);

        group.bench_function("best_of_sample", |b| {
            b.iter(|| {
                let _ = best_of_sample(&mut rng, &population, &stats, &options);
            })
        });
    }

    // next_generation_x100
    {
        let dataset = make_random_dataset(1, 32, 1);
        let mut options = make_utils_options();
        let mut mutation_weights = options.mutation_weights.clone();
        mutation_weights.mutate_constant = 1.0;
        mutation_weights.mutate_operator = 1.0;
        mutation_weights.swap_operands = 1.0;
        mutation_weights.rotate_tree = 1.0;
        mutation_weights.add_node = 1.0;
        mutation_weights.insert_node = 1.0;
        mutation_weights.simplify = 0.0;
        mutation_weights.randomize = 0.0;
        mutation_weights.do_nothing = 0.0;
        mutation_weights.form_connection = 0.0;
        mutation_weights.break_connection = 0.0;
        options.mutation_weights = mutation_weights;

        let (population, stats) = make_population(6, &dataset, &options, POP_SIZE, 15);

        group.bench_function("next_generation_x100", |b| {
            b.iter_batched(
                || {
                    let tagged = TaggedDataset::new(&dataset, None);
                    let evaluator = Evaluator::new(dataset.n_rows);
                    let rng = FastRand::with_seed(6);
                    let next_id = population.len() as u64;
                    (tagged, evaluator, rng, next_id)
                },
                |(tagged, mut evaluator, mut rng, mut next_id)| {
                    for member in population.members.iter() {
                        let ctx = NextGenerationCtx {
                            rng: &mut rng,
                            dataset: tagged,
                            temperature: 1.0,
                            curmaxsize: 20,
                            stats: &stats,
                            options: &options,
                            evaluator: &mut evaluator,
                            next_id: &mut next_id,
                            _ops: PhantomData::<Ops>,
                        };
                        let _ = next_generation(member, ctx);
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }

    // optimize_constants_x10
    {
        let dataset = make_random_dataset(9, 512, 1);
        let options = make_utils_options();
        let mut rng = FastRand::with_seed(42);
        let mut expr_rng = StdRng::seed_from_u64(42);
        let mut members = Vec::with_capacity(10);
        for i in 0..10 {
            let expr = random_expr::<Ops, D, _>(&mut expr_rng, &options.operators, dataset.n_features, 20);
            let member = PopMember::from_expr(MemberId(i as u64), None, expr, dataset.n_features, &options);
            members.push(member);
        }

        group.bench_function("optimize_constants_x10", |b| {
            b.iter(|| {
                let mut evaluator = Evaluator::new(dataset.n_rows);
                let mut grad_ctx = dynamic_expressions::GradContext::<T, D>::new(dataset.n_rows);

                for member in &members {
                    let mut m = member.clone();
                    let ctx = OptimizeConstantsCtx {
                        dataset: TaggedDataset::new(&dataset, None),
                        options: &options,
                        evaluator: &mut evaluator,
                        grad_ctx: &mut grad_ctx,
                    };
                    let _ = optimize_constants(&mut rng, &mut m, ctx);
                }
            })
        });
    }

    // compute_complexity_x10 (Rust uses fixed u16 complexity types)
    {
        let options = make_utils_options();
        let mut rng = StdRng::seed_from_u64(7);
        let trees: Vec<_> = (0..10)
            .map(|_| random_expr::<Ops, D, _>(&mut rng, &options.operators, 3, 20))
            .collect();

        group.bench_function(BenchmarkId::new("compute_complexity_x10", "u16"), |b| {
            b.iter(|| {
                for tree in &trees {
                    let _ = symbolic_regression::compute_complexity(&tree.nodes, &options);
                }
            })
        });
    }

    // randomly_rotate_tree_x10
    {
        let options = make_utils_options();
        group.bench_function("randomly_rotate_tree_x10", |b| {
            b.iter_batched(
                || {
                    let mut expr_rng = StdRng::seed_from_u64(11);
                    let rng = FastRand::with_seed(11);
                    let trees: Vec<_> = (0..10)
                        .map(|_| random_expr::<Ops, D, _>(&mut expr_rng, &options.operators, 3, 20))
                        .collect();
                    (rng, trees)
                },
                |(mut rng, mut trees)| {
                    for tree in trees.iter_mut() {
                        rotate_tree_in_place(&mut rng, tree);
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }

    // insert_random_op_x10
    {
        let options = make_utils_options();
        group.bench_function("insert_random_op_x10", |b| {
            b.iter_batched(
                || {
                    let mut expr_rng = StdRng::seed_from_u64(12);
                    let rng = FastRand::with_seed(12);
                    let trees: Vec<_> = (0..10)
                        .map(|_| random_expr::<Ops, D, _>(&mut expr_rng, &options.operators, 3, 20))
                        .collect();
                    (rng, trees)
                },
                |(mut rng, mut trees)| {
                    for tree in trees.iter_mut() {
                        insert_random_op_in_place(&mut rng, tree, &options.operators, 3);
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }

    // check_constraints_x10
    {
        let mut options = make_utils_options();
        options.maxsize = 30;
        options.maxdepth = 20;

        let add: OpId = <Ops as HasOp<builtin::Add>>::op_id();
        let sub: OpId = <Ops as HasOp<builtin::Sub>>::op_id();
        let div: OpId = <Ops as HasOp<builtin::Div>>::op_id();
        let sin: OpId = <Ops as HasOp<builtin::Sin>>::op_id();
        let cos: OpId = <Ops as HasOp<builtin::Cos>>::op_id();

        options.op_constraints.set_op_arg_constraint(add, 1, 10);
        options.op_constraints.set_op_arg_constraint(div, 0, 10);
        options.op_constraints.set_op_arg_constraint(div, 1, 10);
        options.op_constraints.set_op_arg_constraint(sin, 0, 12);
        options.op_constraints.set_op_arg_constraint(cos, 0, 5);

        options.nested_constraints.add_nested_constraint(add, div, 1);
        options.nested_constraints.add_nested_constraint(add, add, 2);
        options.nested_constraints.add_nested_constraint(sin, sin, 0);
        options.nested_constraints.add_nested_constraint(sin, cos, 2);
        options.nested_constraints.add_nested_constraint(cos, sin, 0);
        options.nested_constraints.add_nested_constraint(cos, cos, 0);
        options.nested_constraints.add_nested_constraint(cos, add, 1);
        options.nested_constraints.add_nested_constraint(cos, sub, 1);

        let mut rng = StdRng::seed_from_u64(13);
        let trees: Vec<_> = (0..10)
            .map(|_| random_expr::<Ops, D, _>(&mut rng, &options.operators, 3, 20))
            .collect();

        group.bench_function("check_constraints_x10", |b| {
            b.iter(|| {
                for tree in &trees {
                    let _ = check_constraints(tree, &options, options.maxsize);
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_search, bench_utils);
criterion_main!(benches);
