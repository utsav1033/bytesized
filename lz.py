# lz.py: tiny LZ77 with Elias-gamma codes + optimal parse. Measures (and later writes) the packed payload.
import sys

def gamma_bits(v):            # v >= 1
    return 2 * v.bit_length() - 1

def parse(data, lit_bits=9, min_len=2, max_off=1 << 13, k=0):
    n = len(data)
    INF = 1 << 60
    cost = [INF] * (n + 1); cost[n] = 0
    choice = [None] * (n + 1)
    for i in range(n - 1, -1, -1):
        best, ch = lit_bits + cost[i + 1], None
        for off in range(1, min(i, max_off) + 1):
            l = 0
            while i + l < n and data[i + l] == data[i + l - off]: l += 1
            for L in range(min_len, l + 1):
                c = 1 + gamma_bits(((off - 1) >> k) + 1) + k + gamma_bits(L - min_len + 1) + cost[i + L]
                if c < best: best, ch = c, (off, L)
        cost[i], choice[i] = best, ch
    return cost[0], choice

K, MIN_LEN = 7, 2   # offset: gamma(high bits) + 7 raw low bits; shortest match: 2 bytes

def pack(data):
    _, choice = parse(data, k=K, min_len=MIN_LEN)
    out = []
    def put(v, n):
        for b in range(n - 1, -1, -1): out.append(v >> b & 1)
    def gamma(v):
        n = v.bit_length() - 1
        put(0, n); put(v, n + 1)
    i = 0
    while i < len(data):
        if choice[i] is None:
            put(0, 1); put(data[i], 8); i += 1
        else:
            off, L = choice[i]
            put(1, 1); gamma(((off - 1) >> K) + 1); put((off - 1) & ((1 << K) - 1), K); gamma(L - MIN_LEN + 1)
            i += L
    out += [0] * (-len(out) % 8)
    return bytes(int("".join(map(str, out[j:j + 8])), 2) for j in range(0, len(out), 8))

def unpack(p, size):   # reference decoder, same logic as stub.c
    pos = [0]
    def bit():
        b = p[pos[0] >> 3] >> (7 - (pos[0] & 7)) & 1; pos[0] += 1; return b
    def bits(n):
        v = 0
        for _ in range(n): v = v << 1 | bit()
        return v
    def gamma():
        n = 0
        while not bit(): n += 1
        return 1 << n | bits(n)
    d = bytearray()
    while len(d) < size:
        if not bit(): d.append(bits(8))
        else:
            off = ((gamma() - 1) << K | bits(K)) + 1
            for _ in range(gamma() + MIN_LEN - 1): d.append(d[-off])
    return bytes(d)

if __name__ == "__main__":   # lz.py IN OUT: write packed payload, verify round trip
    d = open(sys.argv[1], "rb").read()
    p = pack(d)
    assert unpack(p, len(d)) == d
    open(sys.argv[2], "wb").write(p)
    print(len(d), "->", len(p), "bytes")
