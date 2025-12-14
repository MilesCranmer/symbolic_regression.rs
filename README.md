<p align="center">
  <img
    src="https://raw.githubusercontent.com/MilesCranmer/SymbolicRegression.jl/refs/heads/master/docs/src/assets/logo.png"
    height="200"
    alt="SymbolicRegression.jl logo"
  />
</p>

<h1 align="center">symbolic_regression.rs</h1>

Experimental Rust port of [`SymbolicRegression.jl`](https://github.com/MilesCranmer/SymbolicRegression.jl).

> [!WARNING]
> This package is an **experiment**. The API is not stabilized, and you should expect large breaking changes in the syntax.
> This library is not ready for use.

This workspace contains two crates:

- `symbolic_regression`: symbolic regression search loop (`equation_search`, populations, mutations, HoF, constant optimization).
- `dynamic_expressions`: fast batched evaluation + forward-mode derivatives for symbolic expressions.

## Quickstart

Execute `examples/example.rs`, which is the standard example from the [`SymbolicRegression.jl` README](https://github.com/MilesCranmer/SymbolicRegression.jl).

```sh
cargo run -p symbolic_regression --example example --release
```

The code executed is:

```rust
use symbolic_regression::prelude::*;

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;

// Mirrors `example.jl` in the upstream `SymbolicRegression.jl` repository.

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

    let operators = Operators::<2>::builder::<BuiltinOpsF32>()
        .sr_default_binary()
        .unary::<Cos>()
        .unary::<Exp>()
        .build();

    let options = Options::<f32, 2> {
        operators,
        niterations: 100,
        ..Default::default()
    };

    let result = equation_search::<f32, BuiltinOpsF32, 2>(&dataset, &options);
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
            .expect("no members on the pareto front")
            .expr
            .clone();
        let _ = eval_tree_array::<f32, BuiltinOpsF32, 2>(
            &tree,
            dataset.x.view(),
            &EvalOptions::default(),
        );
    */
}
```
