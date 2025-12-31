use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn bench_search(c: &mut Criterion) {
    let env = symbolic_regression::bench::search_env();

    let mut group = c.benchmark_group("search");
    group.sample_size(10);
    group.bench_function("equation_search", |b| {
        b.iter(|| {
            symbolic_regression::bench::run_equation_search(&env);
        })
    });
    group.finish();
}

fn bench_utils(c: &mut Criterion) {
    let mut group = c.benchmark_group("utils");

    {
        let mut env = symbolic_regression::bench::best_of_sample_env();
        group.bench_function("best_of_sample", |b| {
            b.iter(|| {
                symbolic_regression::bench::run_best_of_sample(&mut env);
            })
        });
    }

    {
        let env = symbolic_regression::bench::next_generation_env();
        group.bench_function("next_generation_x100", |b| {
            b.iter(|| {
                symbolic_regression::bench::run_next_generation_x100(&env);
            })
        });
    }

    {
        let env = symbolic_regression::bench::optimize_constants_env();
        group.bench_function("optimize_constants_x10", |b| {
            b.iter(|| {
                symbolic_regression::bench::run_optimize_constants_x10(&env);
            })
        });
    }

    {
        let env = symbolic_regression::bench::complexity_env();
        group.bench_function(BenchmarkId::new("compute_complexity_x10", "u16"), |b| {
            b.iter(|| {
                symbolic_regression::bench::run_compute_complexity_x10(&env);
            })
        });
    }

    {
        let env = symbolic_regression::bench::rotate_tree_env();
        group.bench_function("randomly_rotate_tree_x10", |b| {
            b.iter(|| {
                symbolic_regression::bench::run_rotate_tree_x10(&env);
            })
        });
    }

    {
        let env = symbolic_regression::bench::insert_random_op_env();
        group.bench_function("insert_random_op_x10", |b| {
            b.iter(|| {
                symbolic_regression::bench::run_insert_random_op_x10(&env);
            })
        });
    }

    {
        let env = symbolic_regression::bench::constraints_env();
        group.bench_function("check_constraints_x10", |b| {
            b.iter(|| {
                symbolic_regression::bench::run_check_constraints_x10(&env);
            })
        });
    }

    group.finish();
}

fn bench_constant_optimization(c: &mut Criterion) {
    let env = symbolic_regression::bench::constant_opt_linear_env();

    let mut group = c.benchmark_group("constant_optimization");
    group.bench_with_input(BenchmarkId::new("linear", 0), &env, |b, env| {
        b.iter(|| {
            let _ = symbolic_regression::bench::run_constant_opt_linear(env);
        })
    });
    group.finish();
}

fn bench_bfgs(c: &mut Criterion) {
    let mut group = c.benchmark_group("optim");
    group.bench_function("bfgs_quadratic_n16", |b| {
        b.iter(|| {
            let _ = symbolic_regression::bench::bfgs_quadratic_n16();
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_search,
    bench_utils,
    bench_bfgs,
    bench_constant_optimization
);
criterion_main!(benches);
