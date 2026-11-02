# bpe_sim.py: does "merge (a,b) iff a+b is a vocab token, lowest merged id first" equal real BPE?
import json, re, glob, sys
t = json.load(open(sys.argv[1]))
v = t["model"]["vocab"]
ranks = {tuple(m.split(" ")): r for r, m in enumerate(t["model"]["merges"])}

def real(word):
    p = list(word)
    while True:
        best = min(((ranks.get((p[i], p[i+1]), 1 << 30), i) for i in range(len(p) - 1)), default=(1 << 30, -1))
        if best[0] == 1 << 30: return p
        i = best[1]; p[i:i+2] = [p[i] + p[i+1]]

def alt(word):
    p = list(word)
    while True:
        best = min(((v.get(p[i] + p[i+1], 1 << 30), i) for i in range(len(p) - 1)), default=(1 << 30, -1))
        if best[0] == 1 << 30: return p
        i = best[1]; p[i:i+2] = [p[i] + p[i+1]]

# GPT-2 alphabet so words look like vocab strings
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n); n += 1
enc = dict(zip(bs, map(chr, cs)))
pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?[^\s\w]+|\s+(?!\S)|\s+|\d""")
words = set()
for f in glob.glob("/usr/lib/python3*/**/*.py", recursive=True)[:3000]:
    for w in pat.findall(open(f, errors="ignore").read()):
        words.add("".join(enc[b] for b in w.encode()))
bad = [w for w in words if real(w) != alt(w)]
print(len(words), "distinct pieces,", len(bad), "differ")
for w in bad[:10]: print(" ", repr(w), real(w), alt(w))
