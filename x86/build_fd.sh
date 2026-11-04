#!/bin/sh
# Reproducible descriptor-input builds. All commands run on Linux x86-64.
set -eu
cd "$(dirname "$0")"
python3 pack_lz_fd.py smol_fd.asm smol_fd
python3 pack_lz_fd.py smol_greedy_fd.asm smol_greedy_fd --base 0x97 --length-codes 3 --zero-max 3
wc -c smol_fd smol_greedy_fd
