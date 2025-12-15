# sr_web

Minimal browser UI for `symbolic_regression_wasm` (runs symbolic regression in a WebWorker).

## Prereqs

- Rust target: `wasm32-unknown-unknown`
- `wasm-pack`
- Node.js + npm

## Dev

Build the wasm package into the web app:

```sh
cd web/wasm
wasm-pack build --target web --out-dir ../ui/src/pkg
```

Start the dev server:

```sh
cd web/ui
npm install
npm run dev
```
