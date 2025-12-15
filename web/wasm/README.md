# sr_wasm

WASM bindings for `symbolic_regression` intended for browser use.

## Build

```sh
# one-time
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# build a web-compatible wasm package (for Vite app in `../ui`)
wasm-pack build --target web --out-dir ../ui/src/pkg
```
