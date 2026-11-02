# bench.py BIN...: same prompt, 3 runs each, inside the model folder. Prompt "The cat sat" = 3 tokens, then 100 new.
import subprocess, sys, time, statistics
for b in sys.argv[1:]:
    ts = []
    for _ in range(3):
        t = time.perf_counter(); subprocess.run([b, "The cat sat", "100"], capture_output=True); ts.append(time.perf_counter() - t)
    m = statistics.median(ts)
    print(f"{b}: runs {', '.join(f'{x:.2f}s' for x in ts)}  median {m:.2f}s  = {100 / m:.1f} tok/s (3-token prompt, 100 new, context <= 103)")
