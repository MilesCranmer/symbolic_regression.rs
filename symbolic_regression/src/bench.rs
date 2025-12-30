use dynamic_expressions::OperatorSet;
use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::utils::ZipEq;
use fastrand::Rng;
use ndarray::{Array1, Array2};

use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng};
use rand_distr::StandardNormal;

use crate::constant_optimization::{OptimizeConstantsCtx, optimize_constants};
use crate::dataset::TaggedDataset;
use crate::loss_functions::baseline_loss_from_zero_expression;
use crate::operator_selection::OperatorsSampling;
use crate::optim::{BackTracking, EvalBudget, Objective, OptimOptions, bfgs_minimize};
use crate::pop_member::Evaluator;
use crate::{Dataset, OperatorConstraints, OperatorLibrary, Options, PopMember};

const D: usize = 3;
type T = f64;
type Ops = BuiltinOpsF64;

struct QuadND<'a> {
    target: &'a [f64],
    weight: &'a [f64],
}

impl Objective for QuadND<'_> {
    fn f_only(&mut self, x: &[f64], budget: &mut EvalBudget) -> Option<f64> {
        budget.f_calls += 1;
        let mut acc = 0.0;
        for ((&xi, &ti), &wi) in x.iter().zip_eq(self.target).zip_eq(self.weight) {
            let d = xi - ti;
            acc += wi * d * d;
        }
        Some(acc)
    }

    fn fg(&mut self, x: &[f64], g_out: &mut [f64], budget: &mut EvalBudget) -> Option<f64> {
        budget.f_calls += 1;
        let mut acc = 0.0;
        for (((&xi, &ti), &wi), go) in x
            .iter()
            .zip_eq(self.target)
            .zip_eq(self.weight)
            .zip_eq(g_out.iter_mut())
        {
            let d = xi - ti;
            acc += wi * d * d;
            *go = 2.0 * wi * d;
        }
        Some(acc)
    }
}

pub fn bfgs_quadratic_n16() -> Option<(Vec<f64>, f64)> {
    let n = 16;
    let x0 = vec![0.0f64; n];
    let target: [f64; 16] = core::array::from_fn(|i| (i as f64) / 7.0 - 1.0);
    let weight: [f64; 16] = core::array::from_fn(|i| 1.0 + (i as f64) * 0.01);
    let mut obj = QuadND {
        target: &target,
        weight: &weight,
    };
    let opts = OptimOptions {
        iterations: 40,
        f_calls_limit: 0,
        g_abstol: 1e-10,
    };
    let ls = BackTracking::default();
    let res = bfgs_minimize(&x0, &mut obj, opts, ls)?;
    Some((res.minimizer, res.minimum))
}

fn build_linear_expr_for_constant_optimization() -> PostfixExpr<T, Ops, D> {
    let mul = Ops::lookup("*").unwrap();
    let add = Ops::lookup("+").unwrap();

    // expr: c0 * x0 + c1
    PostfixExpr::new(
        vec![
            PNode::Const { idx: 0 },
            PNode::Var { feature: 0 },
            PNode::Op {
                arity: mul.arity,
                op: mul.id,
            },
            PNode::Const { idx: 1 },
            PNode::Op {
                arity: add.arity,
                op: add.id,
            },
        ],
        vec![0.0, 0.0],
        Metadata::default(),
    )
}

pub struct ConstantOptLinearEnv {
    dataset: Dataset<T>,
    options: Options<T, D>,
}

pub fn constant_opt_linear_env() -> ConstantOptLinearEnv {
    let n_rows = 512;
    let n_features = 1;
    let x: Vec<f64> = (0..n_rows).map(|i| (i as f64) / (n_rows as f64)).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 3.0).collect();
    let dataset = Dataset::new(
        Array2::from_shape_vec((n_features, n_rows), x).unwrap(),
        Array1::from_vec(y),
    );

    let options = Options::<T, D> {
        operators: OperatorLibrary::sr_default::<Ops, D>(),
        should_optimize_constants: true,
        optimizer_iterations: 40,
        optimizer_nrestarts: 0,
        ..Default::default()
    };

    ConstantOptLinearEnv { dataset, options }
}

pub fn run_constant_opt_linear(env: &ConstantOptLinearEnv) -> (bool, f64, Vec<f64>) {
    let expr = build_linear_expr_for_constant_optimization();
    let mut member = PopMember::from_expr(expr, env.dataset.n_features, &env.options);
    let mut evaluator = Evaluator::new(env.dataset.n_rows);
    let mut grad_ctx = dynamic_expressions::GradContext::new(env.dataset.n_rows);
    let baseline_loss = if env.options.use_baseline {
        baseline_loss_from_zero_expression::<T, Ops, D>(&env.dataset, env.options.loss.as_ref())
    } else {
        None
    };
    let full_dataset = TaggedDataset::new(&env.dataset, baseline_loss);
    let _ = member.evaluate(&full_dataset, &env.options, &mut evaluator);

    let mut rng = Rng::with_seed(0);

    let (improved, evals) = optimize_constants(
        &mut rng,
        &mut member,
        OptimizeConstantsCtx {
            dataset: full_dataset,
            options: &env.options,
            evaluator: &mut evaluator,
            grad_ctx: &mut grad_ctx,
        },
    );

    (improved, evals, member.expr.consts.clone())
}

// --- Bench helpers mirrored from the historical `benches/optim.rs` ---

type BenchT = f32;
const BENCH_D: usize = 3;
type BenchOps = dynamic_expressions::operator_enum::presets::BuiltinOpsF32;
const POP_SIZE: usize = 100;

fn random_leaf<R: rand::Rng>(rng: &mut R, n_features: usize, consts: &mut Vec<BenchT>) -> PNode {
    if rng.random_bool(0.5) {
        let val: BenchT = rng.sample(StandardNormal);
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

fn random_expr<Ops2, const D2: usize, R: rand::Rng>(
    rng: &mut R,
    operators: &dynamic_expressions::Operators<D2>,
    n_features: usize,
    target_size: usize,
) -> PostfixExpr<BenchT, Ops2, D2> {
    assert!(target_size >= 1);
    let mut nodes: Vec<PNode> = Vec::with_capacity(target_size);
    let mut consts: Vec<BenchT> = Vec::new();
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

fn make_ops_search() -> dynamic_expressions::Operators<BENCH_D> {
    BenchOps::from_names::<BENCH_D, _>(["exp", "abs", "+", "sub", "*", "/"]).expect("search operators")
}

fn make_ops_utils() -> dynamic_expressions::Operators<BENCH_D> {
    BenchOps::from_names::<BENCH_D, _>(["sin", "cos", "+", "sub", "*", "/"]).expect("utils operators")
}

fn make_search_options(seed: u64) -> Options<BenchT, BENCH_D> {
    let mut options = Options::<BenchT, BENCH_D> {
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

fn make_utils_options() -> Options<BenchT, BENCH_D> {
    Options::<BenchT, BENCH_D> {
        operators: make_ops_utils(),
        progress: false,
        ..Default::default()
    }
}

fn make_dataset(seed: u64, n_rows: usize, n_features: usize) -> Dataset<BenchT> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x: Vec<BenchT> = Vec::with_capacity(n_rows * n_features);
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

fn make_random_dataset(seed: u64, n_rows: usize, n_features: usize) -> Dataset<BenchT> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x: Vec<BenchT> = Vec::with_capacity(n_rows * n_features);
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
    dataset: &Dataset<BenchT>,
    options: &Options<BenchT, BENCH_D>,
    pop_size: usize,
    tree_size: usize,
) -> (crate::population::Population<BenchT, BenchOps, BENCH_D>, crate::adaptive_parsimony::RunningSearchStatistics) {
    let mut rng = StdRng::seed_from_u64(seed);
    let tagged = TaggedDataset::new(dataset, None);
    let mut evaluator = Evaluator::new(dataset.n_rows);

    let mut members = Vec::with_capacity(pop_size);
    for _ in 0..pop_size {
        let expr = random_expr::<BenchOps, BENCH_D, _>(&mut rng, &options.operators, dataset.n_features, tree_size);
        let mut member = PopMember::from_expr(expr, dataset.n_features, options);
        let _ = member.evaluate(&tagged, options, &mut evaluator);
        members.push(member);
    }

    let mut stats = crate::adaptive_parsimony::RunningSearchStatistics::new(options.maxsize, 100_000);
    stats.update_from_population(members.iter().map(|m| m.complexity));
    stats.normalize();

    (crate::population::Population::new(members), stats)
}

pub struct SearchBenchEnv {
    datasets: Vec<Dataset<BenchT>>,
    options: Vec<Options<BenchT, BENCH_D>>,
}

pub fn search_env() -> SearchBenchEnv {
    let seeds = [1u64, 2, 3];
    let datasets: Vec<_> = seeds.iter().map(|&seed| make_dataset(seed, 1_000, 5)).collect();
    let options: Vec<_> = seeds.iter().map(|&seed| make_search_options(seed)).collect();
    SearchBenchEnv { datasets, options }
}

pub fn run_equation_search(env: &SearchBenchEnv) {
    for (dataset, options) in env.datasets.iter().zip_eq(&env.options) {
        let _ = crate::equation_search::<BenchT, BenchOps, BENCH_D>(dataset, options);
    }
}

pub struct BestOfSampleBenchEnv {
    rng: Rng,
    pop: crate::population::Population<BenchT, BenchOps, BENCH_D>,
    stats: crate::adaptive_parsimony::RunningSearchStatistics,
    options: Options<BenchT, BENCH_D>,
}

pub fn best_of_sample_env() -> BestOfSampleBenchEnv {
    let options = make_utils_options();
    let dataset = make_random_dataset(0, 32, 1);
    let (population, stats) = make_population(5, &dataset, &options, POP_SIZE, 20);
    let rng = Rng::with_seed(99);

    BestOfSampleBenchEnv {
        rng,
        pop: population,
        stats,
        options,
    }
}

pub fn run_best_of_sample(env: &mut BestOfSampleBenchEnv) {
    let _ = crate::selection::best_of_sample(&mut env.rng, &env.pop, &env.stats, &env.options);
}

pub struct NextGenerationBenchEnv {
    dataset: Dataset<BenchT>,
    pop: crate::population::Population<BenchT, BenchOps, BENCH_D>,
    stats: crate::adaptive_parsimony::RunningSearchStatistics,
    options: Options<BenchT, BENCH_D>,
}

pub fn next_generation_env() -> NextGenerationBenchEnv {
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

    NextGenerationBenchEnv {
        dataset,
        pop: population,
        stats,
        options,
    }
}

pub fn run_next_generation_x100(env: &NextGenerationBenchEnv) {
    let tagged = TaggedDataset::new(&env.dataset, None);
    let mut evaluator = Evaluator::new(env.dataset.n_rows);
    let mut rng = Rng::with_seed(6);

    for member in env.pop.members.iter() {
        let ctx = crate::mutate::NextGenerationCtx {
            rng: &mut rng,
            dataset: tagged,
            temperature: 1.0,
            curmaxsize: 20,
            stats: &env.stats,
            options: &env.options,
            evaluator: &mut evaluator,
            _ops: core::marker::PhantomData::<BenchOps>,
        };
        let _ = crate::mutate::next_generation(member, ctx);
    }
}

pub struct OptimizeConstantsBenchEnv {
    dataset: Dataset<BenchT>,
    options: Options<BenchT, BENCH_D>,
    members: Vec<PopMember<BenchT, BenchOps, BENCH_D>>,
}

pub fn optimize_constants_env() -> OptimizeConstantsBenchEnv {
    let dataset = make_random_dataset(9, 512, 1);
    let options = make_utils_options();
    let mut expr_rng = StdRng::seed_from_u64(42);
    let mut members = Vec::with_capacity(10);
    for _ in 0..10 {
        let expr = random_expr::<BenchOps, BENCH_D, _>(&mut expr_rng, &options.operators, dataset.n_features, 20);
        let member = PopMember::from_expr(expr, dataset.n_features, &options);
        members.push(member);
    }

    OptimizeConstantsBenchEnv {
        dataset,
        options,
        members,
    }
}

pub fn run_optimize_constants_x10(env: &OptimizeConstantsBenchEnv) {
    let mut rng = Rng::with_seed(42);
    let mut evaluator = Evaluator::new(env.dataset.n_rows);
    let mut grad_ctx = dynamic_expressions::GradContext::<BenchT, BENCH_D>::new(env.dataset.n_rows);

    for member in &env.members {
        let mut m = member.clone();
        let ctx = OptimizeConstantsCtx {
            dataset: TaggedDataset::new(&env.dataset, None),
            options: &env.options,
            evaluator: &mut evaluator,
            grad_ctx: &mut grad_ctx,
        };
        let _ = optimize_constants::<BenchT, BenchOps, BENCH_D>(&mut rng, &mut m, ctx);
    }
}

pub struct ComplexityBenchEnv {
    options: Options<BenchT, BENCH_D>,
    trees: Vec<PostfixExpr<BenchT, BenchOps, BENCH_D>>,
}

pub fn complexity_env() -> ComplexityBenchEnv {
    let options = make_utils_options();
    let mut rng = StdRng::seed_from_u64(7);
    let trees: Vec<_> = (0..10)
        .map(|_| random_expr::<BenchOps, BENCH_D, _>(&mut rng, &options.operators, 3, 20))
        .collect();
    ComplexityBenchEnv { options, trees }
}

pub fn run_compute_complexity_x10(env: &ComplexityBenchEnv) {
    for tree in &env.trees {
        let _ = crate::compute_complexity::<BenchT, BENCH_D>(&tree.nodes, &env.options);
    }
}

pub struct RotateTreeBenchEnv {
    options: Options<BenchT, BENCH_D>,
}

pub fn rotate_tree_env() -> RotateTreeBenchEnv {
    RotateTreeBenchEnv {
        options: make_utils_options(),
    }
}

pub fn run_rotate_tree_x10(env: &RotateTreeBenchEnv) {
    let mut rng = Rng::with_seed(11);
    let mut expr_rng = StdRng::seed_from_u64(11);
    let mut trees: Vec<_> = (0..10)
        .map(|_| random_expr::<BenchOps, BENCH_D, _>(&mut expr_rng, &env.options.operators, 3, 20))
        .collect();
    for tree in trees.iter_mut() {
        crate::mutation_functions::rotate_tree_in_place(&mut rng, tree);
    }
}

pub struct InsertRandomOpBenchEnv {
    options: Options<BenchT, BENCH_D>,
}

pub fn insert_random_op_env() -> InsertRandomOpBenchEnv {
    InsertRandomOpBenchEnv {
        options: make_utils_options(),
    }
}

pub fn run_insert_random_op_x10(env: &InsertRandomOpBenchEnv) {
    let mut rng = Rng::with_seed(12);
    let mut expr_rng = StdRng::seed_from_u64(12);
    let mut trees: Vec<_> = (0..10)
        .map(|_| random_expr::<BenchOps, BENCH_D, _>(&mut expr_rng, &env.options.operators, 3, 20))
        .collect();
    for tree in trees.iter_mut() {
        let _ = crate::mutation_functions::insert_random_op_in_place(&mut rng, tree, &env.options.operators, 3);
    }
}

pub struct ConstraintsBenchEnv {
    options: Options<BenchT, BENCH_D>,
    trees: Vec<PostfixExpr<BenchT, BenchOps, BENCH_D>>,
}

pub fn constraints_env() -> ConstraintsBenchEnv {
    let mut options = make_utils_options();
    options.maxsize = 30;
    options.maxdepth = 20;

    let add: dynamic_expressions::OpId = <BenchOps as dynamic_expressions::HasOp<dynamic_expressions::operator_enum::builtin::Add>>::op_id();
    let sub: dynamic_expressions::OpId = <BenchOps as dynamic_expressions::HasOp<dynamic_expressions::operator_enum::builtin::Sub>>::op_id();
    let div: dynamic_expressions::OpId = <BenchOps as dynamic_expressions::HasOp<dynamic_expressions::operator_enum::builtin::Div>>::op_id();
    let sin: dynamic_expressions::OpId = <BenchOps as dynamic_expressions::HasOp<dynamic_expressions::operator_enum::builtin::Sin>>::op_id();
    let cos: dynamic_expressions::OpId = <BenchOps as dynamic_expressions::HasOp<dynamic_expressions::operator_enum::builtin::Cos>>::op_id();

    let mut constraints: OperatorConstraints<BENCH_D> = Default::default();
    constraints.set_op_arg_max_complexity(add, 1, 10);
    constraints.set_op_arg_max_complexity(div, 0, 10);
    constraints.set_op_arg_max_complexity(div, 1, 10);
    constraints.set_op_arg_max_complexity(sin, 0, 12);
    constraints.set_op_arg_max_complexity(cos, 0, 5);

    constraints.add_nested_limit(add, div, 1);
    constraints.add_nested_limit(add, add, 2);
    constraints.add_nested_limit(sin, sin, 0);
    constraints.add_nested_limit(sin, cos, 2);
    constraints.add_nested_limit(cos, sin, 0);
    constraints.add_nested_limit(cos, cos, 0);
    constraints.add_nested_limit(cos, add, 1);
    constraints.add_nested_limit(cos, sub, 1);
    options.operator_constraints = constraints;

    let mut rng = StdRng::seed_from_u64(13);
    let trees: Vec<_> = (0..10)
        .map(|_| random_expr::<BenchOps, BENCH_D, _>(&mut rng, &options.operators, 3, 20))
        .collect();

    ConstraintsBenchEnv { options, trees }
}

pub fn run_check_constraints_x10(env: &ConstraintsBenchEnv) {
    for tree in &env.trees {
        let _ = crate::check_constraints::check_constraints(tree, &env.options, env.options.maxsize);
    }
}
