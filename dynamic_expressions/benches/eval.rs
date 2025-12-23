use criterion::{criterion_group, criterion_main};
use dynamic_expressions::evaluate::EvalOptions;
use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::operator_enum::scalar::ScalarOpSet;
use dynamic_expressions::operator_registry::OpRegistry;
use dynamic_expressions::opset;
use ndarray::Array2;
use num_traits::Float;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N_FEATURES: usize = 5;
const TREE_SIZE: usize = 20;
const N_TREES: usize = 100;
const N_ROWS: usize = 1_000;

opset! {
    pub struct BenchOpsF32<f32>;
    ops {
        (1, UnaryF32) { Cos, Exp, }
        (2, BinaryF32) { Add, Sub, Mul, Div, }
    }
}

opset! {
    pub struct BenchOpsF64<f64>;
    ops {
        (1, UnaryF64) { Cos, Exp, }
        (2, BinaryF64) { Add, Sub, Mul, Div, }
    }
}

fn random_leaf<T: Float, R: Rng>(rng: &mut R, n_features: usize, consts: &mut Vec<T>) -> PNode {
    if rng.random_bool(0.5) {
        let val: T = T::from(rng.random_range(-2.0..2.0)).unwrap();
        let idx: u16 = consts.len().try_into().expect("too many constants");
        consts.push(val);
        PNode::Const { idx }
    } else {
        let f: u16 = rng
            .random_range(0..n_features)
            .try_into()
            .expect("feature index overflow");
        PNode::Var { feature: f }
    }
}

fn gen_random_tree_fixed_size<T: Float, Ops: OpRegistry, const D: usize, R: Rng>(
    rng: &mut R,
    target_size: usize,
    n_features: usize,
) -> PostfixExpr<T, Ops, D> {
    assert!(target_size >= 1);
    let mut nodes = Vec::with_capacity(target_size);
    let mut consts: Vec<T> = Vec::new();
    nodes.push(random_leaf(rng, n_features, &mut consts));

    let ops_by_arity: [Vec<_>; D] = core::array::from_fn(|arity_minus_one| {
        let arity = (arity_minus_one + 1) as u8;
        Ops::registry()
            .iter()
            .filter(|info| info.op.arity == arity)
            .map(|info| info.op)
            .collect()
    });

    while nodes.len() < target_size {
        let rem = target_size - nodes.len();
        let max_arity = rem.min(D);
        let arity = match 1..=max_arity {
            range if range.is_empty() => break,
            range => rng.random_range(range),
        } as u8;

        let Some(choices) = ops_by_arity.get(usize::from(arity) - 1) else {
            break;
        };
        if choices.is_empty() {
            break;
        }
        let op = choices[rng.random_range(0..choices.len())];

        let leaves: Vec<_> = nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| matches!(n, PNode::Var { .. } | PNode::Const { .. }).then_some(i))
            .collect();
        if leaves.is_empty() {
            break;
        }
        let pos = leaves[rng.random_range(0..leaves.len())];

        let mut repl = Vec::with_capacity(usize::from(arity) + 1);
        for _ in 0..arity {
            repl.push(random_leaf(rng, n_features, &mut consts));
        }
        repl.push(PNode::Op { arity, op: op.id });
        nodes.splice(pos..=pos, repl);
    }

    PostfixExpr::new(nodes, consts, Default::default())
}

fn make_data<T: Float>() -> Array2<T> {
    // Column-major layout for evaluation: shape = (n_features, n_rows).
    let mut data: Vec<T> = Vec::with_capacity(N_FEATURES * N_ROWS);
    for feature in 0..N_FEATURES {
        for row in 0..N_ROWS {
            let v = (row as f64 * 0.01) + (feature as f64 * 0.1);
            data.push(T::from(v).unwrap());
        }
    }
    Array2::from_shape_vec((N_FEATURES, N_ROWS), data).unwrap()
}

fn bench_eval_group<T, Ops, const D: usize>(c: &mut criterion::Criterion, type_name: &str)
where
    T: Float + core::ops::AddAssign + Send + Sync,
    Ops: OpRegistry + ScalarOpSet<T> + Send + Sync,
{
    let mut rng = StdRng::seed_from_u64(0);
    let trees: Vec<PostfixExpr<T, Ops, D>> = (0..N_TREES)
        .map(|_| gen_random_tree_fixed_size(&mut rng, TREE_SIZE, N_FEATURES))
        .collect();
    let x = make_data::<T>();
    let x_view = x.view();
    let opts = EvalOptions {
        check_finite: false,
        early_exit: false,
    };

    let mut group = c.benchmark_group(format!("evaluation/{type_name}"));
    group.bench_function(criterion::BenchmarkId::from_parameter("eval"), |b| {
        let mut out = vec![T::zero(); N_ROWS];
        let mut ctx = dynamic_expressions::EvalContext::<T, D>::new(N_ROWS);
        b.iter(|| {
            for tree in &trees {
                let _ = dynamic_expressions::eval_tree_array_into(&mut out, tree, x_view, &mut ctx, &opts);
            }
        })
    });

    if T::from(0.0f32).unwrap().is_finite() {
        group.bench_function(criterion::BenchmarkId::from_parameter("derivative"), |b| {
            let mut gctx = dynamic_expressions::GradContext::<T, D>::new(N_ROWS);
            b.iter(|| {
                for tree in &trees {
                    let _ = dynamic_expressions::eval_grad_tree_array(tree, x_view, true, &mut gctx, &opts);
                }
            })
        });
    }
    group.finish();
}

fn bench_utilities<T, Ops, const D: usize>(c: &mut criterion::Criterion, type_name: &str)
where
    T: Float + Send + Sync,
    Ops: OpRegistry + ScalarOpSet<T> + Send + Sync,
{
    let mut rng = StdRng::seed_from_u64(1);
    let mut trees: Vec<PostfixExpr<T, Ops, D>> = (0..N_TREES)
        .map(|_| gen_random_tree_fixed_size(&mut rng, TREE_SIZE, N_FEATURES))
        .collect();
    let eval_opts = EvalOptions {
        check_finite: false,
        early_exit: false,
    };

    let mut group = c.benchmark_group(format!("utilities/{type_name}"));
    group.bench_function(criterion::BenchmarkId::from_parameter("clone"), |b| {
        b.iter(|| trees.to_vec())
    });

    group.bench_function(criterion::BenchmarkId::from_parameter("simplify"), |b| {
        b.iter(|| {
            for tree in &mut trees {
                let _ = dynamic_expressions::simplify_in_place(tree, &eval_opts);
            }
        })
    });

    group.bench_function(criterion::BenchmarkId::from_parameter("combine_operators"), |b| {
        b.iter(|| {
            for tree in &mut trees {
                let _ = dynamic_expressions::combine_operators_in_place(tree);
            }
        })
    });

    group.bench_function(criterion::BenchmarkId::from_parameter("counting"), |b| {
        b.iter(|| {
            for tree in &trees {
                let _ = dynamic_expressions::node_utils::count_nodes(&tree.nodes);
                let _ = dynamic_expressions::node_utils::count_depth(&tree.nodes);
                let _ = dynamic_expressions::node_utils::count_constant_nodes(&tree.nodes);
            }
        })
    });

    group.finish();
}

fn update_consts(consts: &mut [f64], tick: u64) {
    let t = tick as f64 * 0.001;
    for (i, c) in consts.iter_mut().enumerate() {
        *c = (t + i as f64).sin() * 0.5;
    }
}

fn build_cache_expr(n_features: usize) -> PostfixExpr<f64, BuiltinOpsF64, 3> {
    let add = <BuiltinOpsF64 as OpRegistry>::lookup_with_arity("+", 2)
        .expect("missing add")
        .op
        .id;
    let sub = <BuiltinOpsF64 as OpRegistry>::lookup_with_arity("-", 2)
        .expect("missing sub")
        .op
        .id;
    let mul = <BuiltinOpsF64 as OpRegistry>::lookup_with_arity("*", 2)
        .expect("missing mul")
        .op
        .id;
    let sin = <BuiltinOpsF64 as OpRegistry>::lookup_with_arity("sin", 1)
        .expect("missing sin")
        .op
        .id;
    let cos = <BuiltinOpsF64 as OpRegistry>::lookup_with_arity("cos", 1)
        .expect("missing cos")
        .op
        .id;

    let mut nodes: Vec<PNode> = Vec::new();

    // Constant-free subtree: a deep fold over variables with alternating unary ops.
    nodes.push(PNode::Var { feature: 0 });
    for i in 1..n_features {
        nodes.push(PNode::Var {
            feature: i.try_into().expect("feature index overflow"),
        });
        let op = if i % 2 == 0 { add } else { mul };
        nodes.push(PNode::Op { arity: 2, op });
        let uop = if i % 2 == 0 { sin } else { cos };
        nodes.push(PNode::Op { arity: 1, op: uop });
    }

    // Constant-dependent subtree.
    nodes.push(PNode::Const { idx: 0 });
    nodes.push(PNode::Var { feature: 0 });
    nodes.push(PNode::Op { arity: 2, op: mul });
    nodes.push(PNode::Const { idx: 1 });
    nodes.push(PNode::Op { arity: 2, op: add });
    nodes.push(PNode::Const { idx: 2 });
    nodes.push(PNode::Op { arity: 2, op: sub });
    nodes.push(PNode::Op { arity: 1, op: sin });

    // Combine subtrees.
    nodes.push(PNode::Op { arity: 2, op: add });

    PostfixExpr::new(nodes, vec![0.1, -0.2, 0.3], Default::default())
}

fn make_cache_data(n_features: usize, n_rows: usize) -> Array2<f64> {
    // Column-major layout for evaluation: shape = (n_features, n_rows).
    let mut data: Vec<f64> = Vec::with_capacity(n_features * n_rows);
    for feature in 0..n_features {
        for row in 0..n_rows {
            let v = ((row as f64 + 1.0) * (feature as f64 + 1.0)).sin() * 0.1;
            data.push(v);
        }
    }
    Array2::from_shape_vec((n_features, n_rows), data).unwrap()
}

fn bench_subtree_cache(c: &mut criterion::Criterion) {
    const CACHE_N_FEATURES: usize = 8;
    const CACHE_N_ROWS: usize = 4_096;

    let x = make_cache_data(CACHE_N_FEATURES, CACHE_N_ROWS);
    let x_view = x.view();
    let expr = build_cache_expr(CACHE_N_FEATURES);
    let plan = dynamic_expressions::compile_plan(&expr.nodes, CACHE_N_FEATURES, expr.consts.len());
    let opts = EvalOptions {
        check_finite: false,
        early_exit: false,
    };

    let mut group = c.benchmark_group("subtree_cache");
    group.bench_function("uncached_eval", |b| {
        let mut expr = expr.clone();
        let mut scratch = Array2::<f64>::zeros((0, 0));
        let mut out = vec![0.0f64; CACHE_N_ROWS];
        let mut tick = 0u64;
        b.iter(|| {
            tick = tick.wrapping_add(1);
            update_consts(&mut expr.consts, tick);
            let _ = dynamic_expressions::eval_plan_array_into(&mut out, &plan, &expr, x_view, &mut scratch, &opts);
        })
    });

    group.bench_function("cached_eval", |b| {
        let mut expr = expr.clone();
        let mut scratch = Array2::<f64>::zeros((0, 0));
        let mut out = vec![0.0f64; CACHE_N_ROWS];
        let mut cache = dynamic_expressions::SubtreeCache::new(CACHE_N_ROWS, 64 * 1024 * 1024);
        let dataset_key = 0xC0FFEEu64;

        let _ = dynamic_expressions::eval_plan_array_into_cached(
            &mut out,
            &plan,
            &expr,
            x_view,
            &mut scratch,
            &opts,
            &mut cache,
            dataset_key,
        );

        let mut tick = 0u64;
        b.iter(|| {
            tick = tick.wrapping_add(1);
            update_consts(&mut expr.consts, tick);
            let _ = dynamic_expressions::eval_plan_array_into_cached(
                &mut out,
                &plan,
                &expr,
                x_view,
                &mut scratch,
                &opts,
                &mut cache,
                dataset_key,
            );
        })
    });

    group.finish();
}

fn criterion_benchmark(c: &mut criterion::Criterion) {
    bench_eval_group::<f32, BenchOpsF32, 2>(c, "Float32");
    bench_eval_group::<f64, BenchOpsF64, 2>(c, "Float64");

    bench_utilities::<f32, BenchOpsF32, 2>(c, "Float32");
    bench_utilities::<f64, BenchOpsF64, 2>(c, "Float64");

    // Builtin opsets include ternary operators; benchmark them as well.
    bench_eval_group::<f32, dynamic_expressions::operator_enum::presets::BuiltinOpsF32, 3>(c, "BuiltinF32");
    bench_eval_group::<f64, dynamic_expressions::operator_enum::presets::BuiltinOpsF64, 3>(c, "BuiltinF64");

    bench_subtree_cache(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
