#!/bin/sh
# wasm.sh OUT [extra flags]: build wasm.c into a WebAssembly module, then shrink it with wasm-opt.
set -e
D=$(dirname "$0"); OUT=$1; shift
clang --target=wasm32 -Oz -nostdlib -ffreestanding -fno-builtin -fno-math-errno "$@" \
  -Wl,--no-entry -Wl,--export-memory=m -Wl,--strip-all -Wl,--gc-sections -o "$OUT.raw" "$D/wasm.c"
wasm-opt -Oz --all-features --strip-target-features "$OUT.raw" -o "$OUT"
rm "$OUT.raw"
