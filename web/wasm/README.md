# symbolic_regression_wasm

WASM bindings for `symbolic_regression` intended for browser use.

## Build

```sh
# one-time
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# build a web-compatible wasm package (for Vite app in `../ui`)
# (uses nightly + build-std/atomics flags configured in `.cargo/config.toml`)
wasm-pack build --target web --out-dir ../ui/src/pkg
```
