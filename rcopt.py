# rcopt.py IMG: re-parse with costs learned from the previous pass (approximates the adaptive coder).
import sys, math
from collections import Counter
import lz, rc

def tokens_cost_tables(data, choice):
    lit, oh, ln, flag = Counter(), Counter(), Counter(), Counter()
    i = 0
    while i < len(data):
        if choice[i] is None: lit[(i & 3, data[i])] += 1; flag[0] += 1; i += 1
        else:
            off, L = choice[i]; oh[((off - 1) >> rc.K) + 1] += 1; ln[L - rc.MIN_LEN + 1] += 1; flag[1] += 1; i += L
    def mk(c, keys, prior):
        tot = sum(c.values()) + prior * len(keys)
        return {k: -math.log2((c[k] + prior) / tot) for k in keys}
    litc = mk(lit, [(l, b) for l in range(4) for b in range(256)], 0.3)
    ohc = mk(oh, range(1, 1 << 11), 0.3); lnc = mk(ln, range(1, 300), 0.3); fc = mk(flag, [0, 1], 1)
    return litc, ohc, lnc, fc

def parse(data, tables, max_off=1 << 13):
    litc, ohc, lnc, fc = tables
    n = len(data); INF = float("inf")
    cost = [INF] * (n + 1); cost[n] = 0; choice = [None] * (n + 1)
    for i in range(n - 1, -1, -1):
        best, ch = fc[0] + litc[(i & 3, data[i])] + cost[i + 1], None
        for off in range(1, min(i, max_off) + 1):
            l = 0
            while i + l < n and data[i + l] == data[i + l - off]: l += 1
            for L in range(rc.MIN_LEN, min(l, 298) + 1):
                c = fc[1] + ohc[((off - 1) >> rc.K) + 1] + rc.K * 0.9 + lnc[L - rc.MIN_LEN + 1] + cost[i + L]
                if c < best: best, ch = c, (off, L)
        cost[i], choice[i] = best, ch
    return choice

def encode(data, choice):
    real = lz.parse
    lz.parse = lambda d, **kw: (0, choice)
    try: return rc.pack(data)
    finally: lz.parse = real

data = open(sys.argv[1], "rb").read()
_, ch = lz.parse(data, k=rc.K, min_len=rc.MIN_LEN)
best = encode(data, ch); print("pass 0", len(best))
for it in range(4):
    ch = parse(data, tokens_cost_tables(data, ch))
    p = encode(data, ch); print("pass", it + 1, len(p))
    if len(p) < len(best): best = p
assert rc.unpack(best, len(data)) == data
if len(sys.argv) > 2: open(sys.argv[2], "wb").write(best)
