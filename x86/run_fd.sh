#!/bin/sh
# Run from the directory containing tokenizer.json and model.safetensors.
set -eu
engine=$1
shift
exec "$engine" "$@" 0<tokenizer.json 3<model.safetensors
