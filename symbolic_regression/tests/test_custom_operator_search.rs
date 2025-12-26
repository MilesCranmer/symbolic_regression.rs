use ndarray::{Array1, Array2};
use symbolic_regression::{Dataset, MutationWeights, Operators, Options, equation_search};

symbolic_regression::custom_opset! {
    struct CustomOps<T = f64>;

    1 => {
        square { eval: |[x]| x * x, partial: |[x]| 2.0 * x },
    },
}

#[test]
fn custom_operator_is_used_in_end_to_end_search() {
    let n_rows = 64usize;
    let x: Vec<f64> = (0..n_rows).map(|i| (i as f64) * 0.1 - 3.0).collect();
    let y: Vec<f64> = x.iter().map(|&v| v * v).collect();

    let x = Array2::from_shape_vec((1, n_rows), x).unwrap();
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
    assert!(result.best.loss <= 1e-12, "best loss was {}", result.best.loss);
}
