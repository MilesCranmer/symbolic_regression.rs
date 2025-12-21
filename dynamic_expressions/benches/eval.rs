use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dynamic_expressions::evaluate::EvalOptions;
use dynamic_expressions::evaluate_derivative::GradContext;
use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::node::PNode;
use dynamic_expressions::node_utils::{count_constant_nodes, count_depth, count_nodes};
use dynamic_expressions::operator_enum::presets::{BuiltinOpsF32, BuiltinOpsF64};
use dynamic_expressions::operator_enum::scalar::ScalarOpSet;
use dynamic_expressions::operator_registry::OpRegistry;
use dynamic_expressions::opset;
use dynamic_expressions::{combine_operators_in_place, eval_grad_tree_array, eval_tree_array_into};
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
    let mut data: Vec<T> = Vec::with_capacity(N_FEATURES * N_ROWS);
    for row in 0..N_ROWS {
        for feature in 0..N_FEATURES {
            let v = (row as f64 * 0.01) + (feature as f64 * 0.1);
            data.push(T::from(v).unwrap());
        }
    }
    Array2::from_shape_vec((N_ROWS, N_FEATURES), data).unwrap()
}

fn bench_eval_group<T, Ops, const D: usize>(c: &mut Criterion, type_name: &str)
where
    T: Float + Send + Sync,
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
    group.bench_function(BenchmarkId::from_parameter("eval"), |b| {
        let mut out = vec![T::zero(); N_ROWS];
        let mut ctx = dynamic_expressions::EvalContext::<T, D>::new(N_ROWS);
        b.iter(|| {
            for tree in &trees {
                let _ = eval_tree_array_into(&mut out, tree, x_view, &mut ctx, &opts);
            }
        })
    });

    if T::from(0.0f32).unwrap().is_finite() {
        group.bench_function(BenchmarkId::from_parameter("derivative"), |b| {
            let mut gctx = GradContext::<T, D>::new(N_ROWS);
            b.iter(|| {
                for tree in &trees {
                    let _ = eval_grad_tree_array(tree, x_view, true, &mut gctx, &opts);
                }
            })
        });
    }
    group.finish();
}

fn bench_utilities<T, Ops, const D: usize>(c: &mut Criterion, type_name: &str)
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
    group.bench_function(BenchmarkId::from_parameter("clone"), |b| {
        b.iter(|| trees.to_vec())
    });

    group.bench_function(BenchmarkId::from_parameter("simplify"), |b| {
        b.iter(|| {
            for tree in &mut trees {
                let _ = dynamic_expressions::simplify_in_place(tree, &eval_opts);
            }
        })
    });

    group.bench_function(BenchmarkId::from_parameter("combine_operators"), |b| {
        b.iter(|| {
            for tree in &mut trees {
                let _ = combine_operators_in_place(tree);
            }
        })
    });

    group.bench_function(BenchmarkId::from_parameter("counting"), |b| {
        b.iter(|| {
            for tree in &trees {
                let _ = count_nodes(&tree.nodes);
                let _ = count_depth(&tree.nodes);
                let _ = count_constant_nodes(&tree.nodes);
            }
        })
    });

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_eval_group::<f32, BenchOpsF32, 2>(c, "Float32");
    bench_eval_group::<f64, BenchOpsF64, 2>(c, "Float64");

    bench_utilities::<f32, BenchOpsF32, 2>(c, "Float32");
    bench_utilities::<f64, BenchOpsF64, 2>(c, "Float64");

    // Builtin opsets include ternary operators; benchmark them as well.
    bench_eval_group::<f32, BuiltinOpsF32, 3>(c, "BuiltinF32");
    bench_eval_group::<f64, BuiltinOpsF64, 3>(c, "BuiltinF64");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
