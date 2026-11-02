#!/bin/sh
# build.sh SRC OUT [extra flags]: the no-libc build used for every size row.
SRC=$1; OUT=$2; shift 2
clang --target=aarch64-linux-gnu ${OPT:--Oz} -ffreestanding -fno-builtin -nostdlib -static \
  -fno-asynchronous-unwind-tables -fno-unwind-tables -fno-stack-protector -fno-math-errno \
  -ffunction-sections -fdata-sections -fno-pic -fno-pie -fno-ident \
  -fuse-ld=lld -Wl,--gc-sections -Wl,--build-id=none -Wl,-z,norelro -Wl,-N -Wl,-T,"$(dirname "$0")/tiny.ld" \
  "$@" -o "$OUT" "$SRC" && { [ -n "$NOSTRIP" ] || { strip -s "$OUT" && python3 "$(dirname "$0")/sstrip.py" "$OUT"; }; }
