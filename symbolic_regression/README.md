# symbolic_regression

[![crates.io](https://img.shields.io/crates/v/symbolic_regression)](https://crates.io/crates/symbolic_regression)

Rust port of the core engine from `SymbolicRegression.jl` (regularized evolution + Pareto hall-of-fame),
built on top of the `dynamic_expressions` crate in this workspace.

For a repo-level overview and examples, see `README.md` at the repo root.

> [!WARNING]
> This crate is an **experiment**. The API is not stabilized and may change substantially.

## Minimal example

```rust,no_run
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use symbolic_regression::prelude::*;

fn main() {
    const D: usize = 3;
    let n_features = 5;
    let n_rows = 100;

    let mut rng = StdRng::seed_from_u64(0);

    let mut x = Array2::zeros((n_features, n_rows));
    let mut y = Array1::zeros(n_rows);

    for i in 0..n_rows {
        for j in 0..n_features {
            x[(j, i)] = rng.random_range(-3.0f32..3.0f32);
        }
        let x1 = x[(1, i)];
        let x4 = x[(4, i)];
        y[i] = 2.0 * x4.cos() + x1 * x1 - 2.0;
    }

    let dataset = Dataset::new(x, y);

    let operators = BuiltinOpsF32::from_names(["cos", "exp", "sin", "+", "sub", "*", "/"]).unwrap();

    let options = Options::<f32, D> {
        operators,
        niterations: 200,
        ..Default::default()
    };

    let result = equation_search::<f32, BuiltinOpsF32, D>(&dataset, &options);
    let dominating = result.hall_of_fame.pareto_front();

    println!("Final Pareto front:");
    println!("Complexity\tMSE\tEquation");
    for member in dominating {
        println!("{}\t{}\t{}", member.complexity, member.loss, member.expr);
    }
}
```

Custom operators can be defined with [`op!`](https://docs.rs/symbolic_regression/latest/symbolic_regression/macro.op.html)
and grouped with [`opset!`](https://docs.rs/symbolic_regression/latest/symbolic_regression/macro.opset.html).

## Operators: universe vs selection

- `BuiltinOpsF32` (or your own `opset!` type) defines the *universe* of operators available to the engine.
- `Options::operators` is an `Operators<D>` value that selects which ops (by `OpId`) are allowed for this run.
  You can build one by pushing `OpId`s into `Operators::new()` or by using `BuiltinOpsF32::from_names(["+", "*"])`.

## Run the example binary

```bash
cargo run -p symbolic_regression --example example --release
```
