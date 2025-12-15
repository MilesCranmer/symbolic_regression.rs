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

| Crate | crates.io | CI |
|---|---|---|
| [`symbolic_regression`](./symbolic_regression) | [![crates.io](https://img.shields.io/crates/v/symbolic_regression)](https://crates.io/crates/symbolic_regression) | [![CI (symbolic_regression)](https://github.com/MilesCranmer/symbolic_regression.rs/actions/workflows/ci-symbolic-regression.yml/badge.svg?branch=main)](https://github.com/MilesCranmer/symbolic_regression.rs/actions/workflows/ci-symbolic-regression.yml) |
| [`dynamic_expressions`](./dynamic_expressions) | [![crates.io](https://img.shields.io/crates/v/dynamic_expressions)](https://crates.io/crates/dynamic_expressions) | [![CI (dynamic_expressions)](https://github.com/MilesCranmer/symbolic_regression.rs/actions/workflows/ci-dynamic-expressions.yml/badge.svg?branch=main)](https://github.com/MilesCranmer/symbolic_regression.rs/actions/workflows/ci-dynamic-expressions.yml) |

## CLI (`symreg`)

This repo optionally ships an experimental CLI binary named `symreg`, behind the `cli` feature.

Install from the git repo:

```bash
cargo install \
  --git https://github.com/MilesCranmer/symbolic_regression.rs \
  --package symbolic_regression \
  --features cli \
  --bin symreg
```

Run on a CSV dataset:

```bash
symreg data.csv --y target --x x1,x2,x3 --niterations=100 --populations=4 --population-size=64
```

Notes:

- Use `symreg --list-operators` to see available builtin operators.
- Quote operator lists in shells to avoid globbing (e.g. `--binary-operators='+,*'`).

## Low-level API

Execute `examples/example.rs`, which is the standard example from the [`SymbolicRegression.jl` README](https://github.com/MilesCranmer/SymbolicRegression.jl).

```bash
cargo run -p symbolic_regression --example example --release
```


The code executed is:

```rust
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

    const MAX_ARITY: usize = 2;
    let operators = Operators::<MAX_ARITY>::from_names_by_arity::<BuiltinOpsF32>(
        &["cos", "exp", "sin"],
        &["+", "-", "*", "/"],
        &[],
    )
    .expect("failed to build operators");

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

## WASM

This workspace includes a thin `wasm-bindgen` wrapper crate at `web/wasm/` and a minimal browser UI at `web/ui/` (Vite + WebWorker).

```bash
rustup target add wasm32-unknown-unknown

# one-time
cargo install wasm-pack

# build the wasm package into the Vite app
cd web/wasm
wasm-pack build --target web --out-dir ../ui/src/pkg

# run the dev server
cd ../ui
npm install
npm run dev
```

See `web/ui/README.md` for details.
