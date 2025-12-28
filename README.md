<p align="center">
  <img
    src="https://raw.githubusercontent.com/MilesCranmer/SymbolicRegression.jl/refs/heads/master/docs/src/assets/logo.png"
    height="200"
    alt="SymbolicRegression.jl logo"
  />
</p>

<h1 align="center">symbolic_regression.rs</h1>

Rust port of [`SymbolicRegression.jl`](https://github.com/MilesCranmer/SymbolicRegression.jl) with support for WebAssembly.

Try out a fully browser-based demo of WebAssembly-compiled symbolic regression [here](https://astroautomata.com/symbolic_regression.rs/).

> [!WARNING]
> This package is an **experiment**. The API is not stabilized, and you should expect large breaking changes in the syntax.
> This library is not ready for use.

This workspace contains three crates:

| Crate | crates.io | CI |
|---|---|---|
| [`symbolic_regression`](./symbolic_regression) | [![crates.io](https://img.shields.io/crates/v/symbolic_regression)](https://crates.io/crates/symbolic_regression) | [![CI (symbolic_regression)](https://github.com/astro-automata/symbolic_regression.rs/actions/workflows/ci-symbolic-regression.yml/badge.svg?branch=main)](https://github.com/astro-automata/symbolic_regression.rs/actions/workflows/ci-symbolic-regression.yml) |
| [`dynamic_expressions`](./dynamic_expressions) | [![crates.io](https://img.shields.io/crates/v/dynamic_expressions)](https://crates.io/crates/dynamic_expressions) | [![CI (dynamic_expressions)](https://github.com/astro-automata/symbolic_regression.rs/actions/workflows/ci-dynamic-expressions.yml/badge.svg?branch=main)](https://github.com/astro-automata/symbolic_regression.rs/actions/workflows/ci-dynamic-expressions.yml) |
| [`symbolic_regression_wasm`](./web/wasm) | [![crates.io](https://img.shields.io/crates/v/symbolic_regression_wasm)](https://crates.io/crates/symbolic_regression_wasm) | [![CI (Web UI)](https://github.com/astro-automata/symbolic_regression.rs/actions/workflows/ci-web.yml/badge.svg?branch=main)](https://github.com/astro-automata/symbolic_regression.rs/actions/workflows/ci-web.yml) |

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

    let operators = Operators::<D>::from_names_by_arity::<BuiltinOpsF32>(&["cos", "exp", "sin"], &["+", "-", "*", "/"], &[])
        .unwrap();

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
    // To evaluate the expression, use:
    /*
        let tree = dominating
            .last()
            .unwrap()
            .expr
            .clone();
        let _ = eval_tree_array::<f32, BuiltinOpsF32, D>(
            &tree,
            dataset.x.view(),
            &EvalOptions::default(),
        );
    */
}
```

## Custom operators

Define light-weight operator sets with inline evaluation and derivatives using the `custom_opset!` macro (re-exported from `symbolic_regression`):

```rust
use symbolic_regression::custom_opset;
use symbolic_regression::prelude::*;

custom_opset! {
    pub struct CustomOps<f64> {
        1 {
            square {
                eval(args) { args[0] * args[0] },
                partial(args, _idx) { 2.0 * args[0] },
            }
            exp {
                eval(args) { args[0].exp() },
                partial(args, _idx) { args[0].exp() },
            }
        }
        2 {
            add {
                eval(args) { args[0] + args[1] },
                partial(_args, _idx) { 1.0 },
            }
            sub {
                infix: "-",    // optional
                complexity: 2, // optional
                eval(args) { args[0] - args[1] },
                partial(_args, idx) {
                    if idx == 0 { 1.0 } else { -1.0 }
                },
            }
        }
    }
}

let operators = CustomOps::from_names_by_arity(&["square", "exp"], &["add", "sub"], &[]).unwrap();
let options = Options::<f64, _> { operators, ..Default::default() };
```

## WASM

This workspace includes a thin `wasm-bindgen` wrapper crate (`symbolic_regression_wasm`) at `web/wasm/` and a minimal browser UI at `web/ui/` (Vite + WebWorker).

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
