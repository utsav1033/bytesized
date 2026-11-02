# tokcheck.py BINARY: compare the engine's prompt tokenization with HF on tricky strings.
import subprocess, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(".")
tests = ["Hello world", "I'm here, you're there; we'll see.", "  two  spaces", "line\n\nbreak",
         "price: $1234.56!", "tabs\tand\ttabs", "end with space ", "CamelCase and snake_case",
         "don't 'quote' it", "a\n b\n  c", "100% sure... right?!", "x = [1, 2, 3]"]
bad = 0
for t in tests:
    want = tok(t)["input_ids"]
    r = subprocess.run([sys.argv[1], t, "0"], capture_output=True, text=True)
    got = [int(x) for x in r.stderr.split("|")[0].split()]
    if got != want:
        bad += 1
    print("ok  " if got == want else "BAD ", repr(t), "" if got == want else f"got {got} want {want}")
sys.exit(bad)
