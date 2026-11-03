# check.py BINARY: run the engine on every answer-key prompt (inside the model folder), compare ids.
import json, subprocess, sys, os, time
key = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "answer_key.json")))
binary, ok = sys.argv[1], True
for k in key:
    t = time.time()
    r = subprocess.run([binary, k["prompt"], "100"], capture_output=True, text=True)
    dt = time.time() - t
    nums = r.stderr.replace("|", " ").split()
    got = [int(x) for x in nums if x.isdigit()]
    want = (k["prompt_ids"] + k["new_ids"]) if "|" in r.stderr else k["new_ids"]
    if not got:   # binary built without ids: compare the text instead
        match = r.stdout == k["prompt"] + k["text"]
        ok &= match
        print(f"{'PASS' if match else 'FAIL'} (text) {k['prompt']!r}: {dt:.1f}s")
        continue
    first_bad = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
    match = got == want
    ok &= match
    print(f"{'PASS' if match else 'FAIL'} {k['prompt']!r}: {len(got)} ids, {dt:.1f}s"
          + ("" if match else f", first diff at {first_bad}: got {got[first_bad:first_bad+3]} want {want[first_bad:first_bad+3]}"))
sys.exit(0 if ok else 1)
