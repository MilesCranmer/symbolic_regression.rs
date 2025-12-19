use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use symbolic_regression::prelude::*;

// Mirrors `SymbolicRegression.jl/example.jl`.

fn main() {
    let n_features = 5;
    let n_rows = 100;

    let mut rng = StdRng::seed_from_u64(0);

    let mut x = Array2::zeros((n_rows, n_features));
    let mut y = Array1::zeros(n_rows);

    for i in 0..n_rows {
        for j in 0..n_features {
            x[(i, j)] = rng.random_range(-3.0f32..3.0f32);
        }
        let x1 = x[(i, 1)];
        let x4 = x[(i, 4)];
        y[i] = 2.0 * x4.cos() + x1 * x1 - 2.0;
    }

    let dataset = Dataset::new(x, y);

    let operators =
        BuiltinOpsF32::from_names_by_arity(&["cos", "exp", "sin"], &["+", "-", "*", "/"], &[])
            .unwrap();

    let options = Options::<f32, _> {
        operators,
        niterations: 200,
        ..Default::default()
    };

    let result = equation_search::<_, BuiltinOpsF32, _>(&dataset, &options);
    let dominating = result.hall_of_fame.pareto_front();

    println!("Final Pareto front:");
    println!("Complexity\tMSE\tEquation");
    for member in dominating {
        println!("{}\t{}\t{}", member.complexity, member.loss, member.expr);
    }
    // To evaluate the expression, use:
    /*
        let tree = dominating
            .last()
            .unwrap()
            .expr
            .clone();
        let _ = eval_tree_array::<f32, BuiltinOpsF32, 2>(
            &tree,
            dataset.x.view(),
            &EvalOptions::default(),
        );
    */
}
