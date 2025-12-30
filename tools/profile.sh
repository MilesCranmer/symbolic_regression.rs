#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: ./tools/profile.sh

Profiles `symbolic_regression/examples/example.rs` (built with the `profiling` Cargo profile) using Samply.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "No args supported." >&2
  usage
  exit 2
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

export RUSTFLAGS="${RUSTFLAGS:-} -C force-frame-pointers=yes"
cargo --config 'build.rustc-wrapper=""' build -p symbolic_regression --profile profiling --example example

bin="target/profiling/examples/example"

mkdir -p profiles
timestamp="$(date +%Y%m%d-%H%M%S)"

if [[ "$(uname -s)" == "Darwin" ]]; then
  samply setup --yes >/dev/null 2>&1 || {
    echo "Warning: \`samply setup\` failed; profiling may fail on macOS." >&2
  }
fi

out="profiles/example-${timestamp}.json.gz"
echo "Writing profile to: $out" >&2
exec samply record --rate 1000 --symbol-dir "target/profiling/examples" -o "$out" -- "$bin"
