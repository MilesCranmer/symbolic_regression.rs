# dynamic_expressions

[![crates.io](https://img.shields.io/crates/v/dynamic_expressions)](https://crates.io/crates/dynamic_expressions)

Fast batched evaluation + forward-mode derivatives for symbolic expressions (Rust port of `DynamicExpressions.jl`).

This crate is the evaluation backend used by `symbolic_regression`.

## Data layout

Evaluation APIs take `ndarray::ArrayView2<'_, T>` for `X` with shape `(n_features, n_rows)`.
