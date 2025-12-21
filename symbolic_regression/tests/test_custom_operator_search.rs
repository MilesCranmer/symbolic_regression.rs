use dynamic_expressions::operator_enum::scalar::{
    diff_nary, eval_nary, grad_nary, DiffKernelCtx, EvalKernelCtx, GradKernelCtx, OpId, ScalarOpSet,
};
use dynamic_expressions::operator_registry::{OpInfo, OpRegistry};
use dynamic_expressions::strings::OpNames;
use ndarray::{Array1, Array2};
use symbolic_regression::{equation_search, Dataset, MutationWeights, Operators, Options};

#[derive(Copy, Clone, Debug, Default)]
struct CustomOps;

const SQUARE: OpId = OpId { arity: 1, id: 0 };

const CUSTOM_REGISTRY: [OpInfo; 1] = [OpInfo {
    op: SQUARE,
    name: "square",
    display: "square",
    infix: None,
    commutative: false,
    associative: false,
    complexity: 1,
}];

fn square_eval(args: &[f64; 1]) -> f64 {
    args[0] * args[0]
}

fn square_partial(args: &[f64; 1], idx: usize) -> f64 {
    match idx {
        0 => 2.0 * args[0],
        _ => unreachable!(),
    }
}

impl ScalarOpSet<f64> for CustomOps {
    fn eval(op: OpId, ctx: EvalKernelCtx<'_, '_, f64>) -> bool {
        match (op.arity, op.id) {
            (1, 0) => eval_nary::<1, f64>(square_eval, ctx.out, ctx.args, ctx.opts),
            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
        }
    }

    fn diff(op: OpId, ctx: DiffKernelCtx<'_, '_, f64>) -> bool {
        match (op.arity, op.id) {
            (1, 0) => diff_nary::<1, f64>(
                square_eval,
                square_partial,
                ctx.out_val,
                ctx.out_der,
                ctx.args,
                ctx.dargs,
                ctx.opts,
            ),
            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
        }
    }

    fn grad(op: OpId, ctx: GradKernelCtx<'_, '_, f64>) -> bool {
        match (op.arity, op.id) {
            (1, 0) => grad_nary::<1, f64>(square_eval, square_partial, ctx),
            _ => panic!("unknown op id {} for arity {}", op.id, op.arity),
        }
    }
}

impl OpRegistry for CustomOps {
    fn registry() -> &'static [OpInfo] {
        &CUSTOM_REGISTRY
    }
}

impl OpNames for CustomOps {
    fn op_name(op: OpId) -> &'static str {
        match (op.arity, op.id) {
            (1, 0) => "square",
            _ => "unknown_op",
        }
    }
}

#[test]
fn custom_operator_is_used_in_end_to_end_search() {
    let n_rows = 64usize;
    let x: Vec<f64> = (0..n_rows).map(|i| (i as f64) * 0.1 - 3.0).collect();
    let y: Vec<f64> = x.iter().map(|&v| v * v).collect();

    let x = Array2::from_shape_vec((n_rows, 1), x).unwrap();
    let y = Array1::from_vec(y);
    let dataset = Dataset::new(x, y);

    let operators = Operators::<1>::from_names::<CustomOps>(&["square"]).unwrap();
    let mutation_weights = MutationWeights {
        mutate_constant: 0.0,
        mutate_operator: 0.0,
        mutate_feature: 0.0,
        swap_operands: 0.0,
        rotate_tree: 0.0,
        add_node: 0.0,
        insert_node: 0.0,
        delete_node: 0.0,
        simplify: 0.0,
        randomize: 1.0,
        do_nothing: 0.0,
        optimize: 0.0,
        form_connection: 0.0,
        break_connection: 0.0,
    };
    let options = Options::<f64, 1> {
        seed: 0,
        niterations: 1,
        populations: 1,
        population_size: 128,
        ncycles_per_iteration: 20,
        maxsize: 2,
        maxdepth: 2,
        progress: false,
        should_optimize_constants: false,
        annealing: false,
        operators,
        mutation_weights,
        ..Default::default()
    };

    let result = equation_search::<f64, CustomOps, 1>(&dataset, &options);

    let eqn = dynamic_expressions::string_tree(&result.best.expr, Default::default());
    assert_eq!(eqn, "square(x0)");
    assert!(
        result.best.loss <= 1e-12,
        "best loss was {}",
        result.best.loss
    );
}
