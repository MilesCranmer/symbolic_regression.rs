use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;
use symbolic_regression::prelude::*;

// Mirrors `SymbolicRegression.jl/example.jl`.

fn main() {
    let n_features = 5;
    let n_rows = 100;

    let x: Array2<f32> = Array2::random((n_rows, n_features), StandardNormal);
    let y: Array1<f32> = x.map_axis(Axis(1), |row| {
        let x1 = row[0];
        let x4 = row[3];
        2.0 * x4.cos() + x1 * x1 - 2.0
    });

    let dataset = Dataset::new(x, y);

    let operators = Operators::<2>::from_names_by_arity::<BuiltinOpsF32>(
        &["cos", "exp", "sin"],
        &["+", "-", "*", "/"],
        &[],
    )
    .expect("failed to build operators");

    let options = Options::<f32, 2> {
        operators,
        niterations: 200,
        ..Default::default()
    };

    let result = equation_search::<f32, BuiltinOpsF32, 2>(&dataset, &options);
    let dominating = result.hall_of_fame.pareto_front();

    let tree = dominating
        .last()
        .expect("no members on the pareto front")
        .expr
        .clone();
    let _ =
        eval_tree_array::<f32, BuiltinOpsF32, 2>(&tree, dataset.x.view(), &EvalOptions::default());

    println!("Complexity\tMSE\tEquation");
    for member in dominating {
        println!("{}\t{}\t{}", member.complexity, member.loss, member.expr);
    }
}
