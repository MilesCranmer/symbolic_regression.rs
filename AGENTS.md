# AGENTS.md

## Cursor Cloud specific instructions

This is a Rust workspace (`nightly-2025-12-19`, pinned in `rust-toolchain.toml`) with three crates (`dynamic_expressions`, `symbolic_regression`, `symbolic_regression_wasm`) and a React/Vite web UI at `web/ui/`.

### Services overview

| Component | Purpose | Run command |
|---|---|---|
| Rust workspace | Core symbolic regression library | `cargo check --workspace` / `cargo test --workspace` |
| WASM build | Browser-compiled version of the library | `cd web/ui && bash scripts/build_wasm.sh` |
| Web UI (Vite) | React frontend for browser-based symbolic regression | `cd web/ui && npm run dev` |

### Lint / Test / Build commands

See `.pre-commit-config.yaml` for the full set of hooks. Key commands:

- **Format:** `cargo fmt --all -- --check`
- **Clippy:** `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- **Rust tests:** `cargo test --workspace`
- **Web UI unit tests:** `cd web/ui && npm test`
- **Web UI e2e tests:** `cd web/ui && npm run test:e2e` (requires Playwright + Chromium)
- **Run example:** `cargo run -p symbolic_regression --example example --release`

### Non-obvious caveats

- The WASM build uses `build-std` and requires `rust-src` component and `wasm32-unknown-unknown` target for the nightly toolchain. The build script at `web/ui/scripts/build_wasm.sh` validates all prerequisites.
- The WASM build outputs to `web/ui/src/pkg/`. The Vite dev server (`npm run dev`) expects this directory to exist if you want the WASM features to work in the browser. Run `bash scripts/build_wasm.sh` from `web/ui/` first.
- The Vite dev server sets COOP/COEP headers for `SharedArrayBuffer` support (needed for threaded WASM with `wasm-bindgen-rayon`).
- `npm run gen:csv` (in `web/ui`) generates default CSV data and is automatically run by both `npm run dev` and `npm test`.
- Code style conventions for Rust imports are documented in `CONTRIBUTORS.md`.
