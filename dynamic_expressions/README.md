# dynamic_expressions

Fast batched evaluation + forward-mode derivatives for symbolic expressions (Rust port of `DynamicExpressions.jl`).

This crate is the evaluation backend used by `symbolic_regression`.

## Data layout

Evaluation APIs take `ndarray::ArrayView2<'_, T>` for `X` with shape `(n_rows, n_features)` (row-major / C order).

