#!/usr/bin/env bash
set -euo pipefail

TOOLCHAIN="${TOOLCHAIN:-nightly-2025-12-19}"

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "wasm-pack missing. Install with: cargo install wasm-pack" >&2
  exit 1
fi

if ! command -v rustup >/dev/null 2>&1; then
  echo "rustup missing. Install from https://rustup.rs" >&2
  exit 1
fi

if ! rustup toolchain list | grep -q "^${TOOLCHAIN}"; then
  echo "Missing toolchain ${TOOLCHAIN}. Install with: rustup toolchain install ${TOOLCHAIN} --component rust-src --target wasm32-unknown-unknown" >&2
  exit 1
fi

if ! rustup component list --installed --toolchain "${TOOLCHAIN}" | grep -q "^rust-src"; then
  echo "Missing rust-src for ${TOOLCHAIN}. Install with: rustup component add rust-src --toolchain ${TOOLCHAIN}" >&2
  exit 1
fi

if ! rustup target list --installed --toolchain "${TOOLCHAIN}" | grep -q "^wasm32-unknown-unknown$"; then
  echo "Missing target wasm32-unknown-unknown for ${TOOLCHAIN}. Install with: rustup target add wasm32-unknown-unknown --toolchain ${TOOLCHAIN}" >&2
  exit 1
fi

cd ../wasm
export RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+atomics,+bulk-memory,+mutable-globals"
export WASM_BINDGEN_FLAGS=--enable-threads
wasm-pack build . --release --target web --out-dir ../ui/src/pkg
