use core::marker::PhantomData;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use dynamic_expressions::operator_enum::presets::BuiltinOpsF32;
use dynamic_expressions::operator_enum::{builtin, scalar};
use dynamic_expressions::utils::ZipEq;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use symbolic_regression::{
    Dataset, Evaluator, MemberId, NextGenerationCtx, Operators, OptimizeConstantsCtx, Options, PopMember, Population,
    RunningSearchStatistics, TaggedDataset, best_of_sample, check_constraints, equation_search,
    insert_random_op_in_place, next_generation, optimize_constants, random_expr, rotate_tree_in_place,
};

type T = f32;
const D: usize = 3;
type Ops = BuiltinOpsF32;
const POP_SIZE: usize = 100;

fn make_ops_search() -> Operators<D> {
    Operators::<D>::from_names_by_arity::<Ops>(&["exp", "abs"], &["+", "-", "*", "/"], &[]).expect("search operators")
}

fn make_ops_utils() -> Operators<D> {
    Operators::<D>::from_names_by_arity::<Ops>(&["sin", "cos"], &["+", "-", "*", "/"], &[]).expect("utils operators")
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
        let expr = random_expr::<T, Ops, D, _>(&mut rng, &options.operators, dataset.n_features, tree_size);
        let mut member = PopMember::from_expr(MemberId(i as u64), None, i as u64, expr, dataset.n_features);
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
        let mut rng = StdRng::seed_from_u64(99);

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
                    let rng = StdRng::seed_from_u64(6);
                    let next_id = population.len() as u64;
                    let next_birth = population.len() as u64;
                    (tagged, evaluator, rng, next_id, next_birth)
                },
                |(tagged, mut evaluator, mut rng, mut next_id, mut next_birth)| {
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
                            next_birth: &mut next_birth,
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
        let mut rng = StdRng::seed_from_u64(42);
        let mut members = Vec::with_capacity(10);
        for i in 0..10 {
            let expr = random_expr::<T, Ops, D, _>(&mut rng, &options.operators, dataset.n_features, 20);
            let member = PopMember::from_expr(MemberId(i as u64), None, i as u64, expr, dataset.n_features);
            members.push(member);
        }

        group.bench_function("optimize_constants_x10", |b| {
            b.iter(|| {
                let mut evaluator = Evaluator::new(dataset.n_rows);
                let mut grad_ctx = dynamic_expressions::GradContext::<T, D>::new(dataset.n_rows);
                let mut next_birth = 0u64;

                for member in &members {
                    let mut m = member.clone();
                    let ctx = OptimizeConstantsCtx {
                        dataset: TaggedDataset::new(&dataset, None),
                        options: &options,
                        evaluator: &mut evaluator,
                        grad_ctx: &mut grad_ctx,
                        next_birth: &mut next_birth,
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
            .map(|_| random_expr::<T, Ops, D, _>(&mut rng, &options.operators, 3, 20))
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
                    let mut rng = StdRng::seed_from_u64(11);
                    let trees: Vec<_> = (0..10)
                        .map(|_| random_expr::<T, Ops, D, _>(&mut rng, &options.operators, 3, 20))
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
                    let mut rng = StdRng::seed_from_u64(12);
                    let trees: Vec<_> = (0..10)
                        .map(|_| random_expr::<T, Ops, D, _>(&mut rng, &options.operators, 3, 20))
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

        let add = scalar::OpId {
            arity: 2,
            id: <Ops as scalar::HasOp<builtin::Add, 2>>::ID,
        };
        let sub = scalar::OpId {
            arity: 2,
            id: <Ops as scalar::HasOp<builtin::Sub, 2>>::ID,
        };
        let div = scalar::OpId {
            arity: 2,
            id: <Ops as scalar::HasOp<builtin::Div, 2>>::ID,
        };
        let sin = scalar::OpId {
            arity: 1,
            id: <Ops as scalar::HasOp<builtin::Sin, 1>>::ID,
        };
        let cos = scalar::OpId {
            arity: 1,
            id: <Ops as scalar::HasOp<builtin::Cos, 1>>::ID,
        };

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
            .map(|_| random_expr::<T, Ops, D, _>(&mut rng, &options.operators, 3, 20))
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
