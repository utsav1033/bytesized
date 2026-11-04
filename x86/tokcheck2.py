# tokcheck2.py BINARY: tokcheck.py's 12 strings + unicode/edge cases, vs HF.
import subprocess, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(".")
tests = ["Hello world", "I'm here, you're there; we'll see.", "  two  spaces", "line\n\nbreak",
         "price: $1234.56!", "tabs\tand\ttabs", "end with space ", "CamelCase and snake_case",
         "don't 'quote' it", "a\n b\n  c", "100% sure... right?!", "x = [1, 2, 3]",
         "café naïve résumé", "日本語のテキスト", "emoji 🎉🚀 ok", "Ünïcödé — “quotes”",
         "The cat sat", "Once upon a time", "The capital of France is", "def fibonacci(n):"]
bad = 0
for t in tests:
    want = tok(t)["input_ids"]
    r = subprocess.run([sys.argv[1], t, "0"], capture_output=True)
    got = [int(x) for x in r.stderr.decode().split("|")[0].split()]
    bad += got != want
    print("ok  " if got == want else "BAD ", repr(t), "" if got == want else f"got {got} want {want}")
sys.exit(bad)
