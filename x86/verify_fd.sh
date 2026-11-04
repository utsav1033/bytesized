#!/bin/sh
# Verify descriptor-input builds on Linux x86-64 against the supplied model.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
model=$HOME/smol
if [ "$#" -gt 0 ]; then model=$1; fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
python3 "$here/pack_lz_fd.py" "$here/smol_fd.asm" "$tmp/exact"
python3 "$here/pack_lz_fd.py" "$here/smol_fd.asm" "$tmp/exactc" CHECK
python3 "$here/pack_lz_fd.py" "$here/smol_fd.asm" "$tmp/exactt" TRACE
python3 "$here/pack_lz_fd.py" "$here/smol_greedy_fd.asm" "$tmp/greedy" --base 0x97 --length-codes 3 --zero-max 3
python3 "$here/pack_lz_fd.py" "$here/smol_greedy_fd.asm" "$tmp/greedyc" CHECK --base 0x97 --length-codes 3 --zero-max 3
python3 "$here/pack_lz_fd.py" "$here/smol_greedy_fd.asm" "$tmp/greedyt" TRACE --base 0x97 --length-codes 3 --zero-max 3
for name in exact exactc exactt greedy greedyc greedyt; do
    printf '#!/bin/sh\nexec "%s" "$@" 0<tokenizer.json 3<model.safetensors\n' "$tmp/$name" > "$tmp/$name-run"
    chmod +x "$tmp/$name-run"
done
wc -c "$tmp/exact" "$tmp/greedy"
(cd "$model" && python3 "$here/tokcheck2.py" "$tmp/exactc-run")
(cd "$model" && python3 "$here/../check.py" "$tmp/exactt-run")
python3 - "$here/../answer_key.json" "$tmp/exact-run" "$model" <<'TEXTCHECK'
import json, subprocess, sys
key_path, binary, model = sys.argv[1:]
for case in json.load(open(key_path)):
    result = subprocess.run([binary, case["prompt"]], cwd=model, capture_output=True, text=True)
    assert result.returncode == 0, (case["prompt"], result.returncode)
    assert result.stdout == case["prompt"] + case["text"] + "\n", case["prompt"]
    print("PASS (text)", repr(case["prompt"]))
TEXTCHECK
status=0
(cd "$model" && python3 "$here/tokcheck2.py" "$tmp/greedyc-run") > "$tmp/greedy-tok.log" || status=$?
cat "$tmp/greedy-tok.log"
bad=$(grep -c '^BAD' "$tmp/greedy-tok.log" || true)
test "$status" -eq 7 && test "$bad" -eq 7
(cd "$model" && python3 "$here/../check.py" "$tmp/greedyt-run")
(cd "$model" && "$tmp/exact-run" "The cat sat" >/dev/null)
(cd "$model" && "$tmp/greedy-run" "The cat sat" >/dev/null)
