# rc.py IN OUT: LZ77 tokens coded with an adaptive binary range coder (LZMA-style).
# The probability layout and bit order here must match stubrc.S exactly.
import sys
import lz

PROB_BITS, SHIFT = 12, 4
F, L, OH, OL, LEN = 0, 8, 1032, 1320, 1327   # bases into one probability array
N_PROBS = 1615
K, MIN_LEN = lz.K, lz.MIN_LEN


class Enc:
    def __init__(s):
        s.low, s.range, s.cache, s.csize, s.out = 0, 0xFFFFFFFF, 0, 1, bytearray()
        s.p = [1 << (PROB_BITS - 1)] * N_PROBS

    def shift_low(s):
        if s.low < 0xFF000000 or s.low >= 1 << 32:
            carry, t = s.low >> 32, s.cache
            while True:
                s.out.append((t + carry) & 0xFF)
                t = 0xFF
                s.csize -= 1
                if not s.csize: break
            s.cache = (s.low >> 24) & 0xFF
        s.csize += 1
        s.low = (s.low & 0x00FFFFFF) << 8

    def bit(s, i, b):
        p = s.p[i]
        bound = (s.range >> PROB_BITS) * p
        if b == 0:
            s.range = bound
            s.p[i] = p + (((1 << PROB_BITS) - p) >> SHIFT)
        else:
            s.low += bound
            s.range -= bound
            s.p[i] = p - (p >> SHIFT)
        while s.range < 1 << 24:
            s.range = (s.range << 8) & 0xFFFFFFFF
            s.shift_low()

    def flush(s):
        for _ in range(5): s.shift_low()
        return bytes(s.out)


class Dec:
    def __init__(s, data):
        s.d, s.pos, s.range, s.code = data, 5, 0xFFFFFFFF, int.from_bytes(data[:5], "big")
        s.p = [1 << (PROB_BITS - 1)] * N_PROBS

    def bit(s, i):
        p = s.p[i]
        bound = (s.range >> PROB_BITS) * p
        if s.code < bound:
            s.range, s.p[i], b = bound, p + (((1 << PROB_BITS) - p) >> SHIFT), 0
        else:
            s.range, s.code, s.p[i], b = s.range - bound, s.code - bound, p - (p >> SHIFT), 1
        while s.range < 1 << 24:   # same loop as the encoder: a very unlikely bit can need 2 shifts
            s.range = (s.range << 8) & 0xFFFFFFFF
            s.code = ((s.code << 8) | (s.d[s.pos] if s.pos < len(s.d) else 0)) & 0xFFFFFFFF
            s.pos += 1
        return b


def gamma_enc(e, base, v):
    n = v.bit_length() - 1
    for j in range(n): e.bit(base + j, 0)
    e.bit(base + n, 1)
    for j in range(n - 1, -1, -1): e.bit(base + 32 + n * 16 + j, v >> j & 1)


def gamma_dec(d, base):
    n = 0
    while not d.bit(base + n): n += 1
    v = 1
    for j in range(n - 1, -1, -1): v = v << 1 | d.bit(base + 32 + n * 16 + j)
    return v


def pack(data):
    _, choice = lz.parse(data, k=K, min_len=MIN_LEN)
    e, i, prev = Enc(), 0, 0
    while i < len(data):
        lane = i & 3
        if choice[i] is None:
            e.bit(F + lane + 4 * prev, 0)
            node = 1
            for j in range(7, -1, -1):
                b = data[i] >> j & 1
                e.bit(L + lane * 256 + node, b)
                node = node * 2 + b
            i, prev = i + 1, 0
        else:
            off, ln = choice[i]
            e.bit(F + lane + 4 * prev, 1)
            gamma_enc(e, OH, ((off - 1) >> K) + 1)
            for j in range(K - 1, -1, -1): e.bit(OL + j, (off - 1) >> j & 1)
            gamma_enc(e, LEN, ln - MIN_LEN + 1)
            i, prev = i + ln, 1
    return e.flush()


def unpack(p, size):   # reference decoder, same steps as stubrc.S
    d, out, prev = Dec(p), bytearray(), 0
    while len(out) < size:
        lane = len(out) & 3
        if not d.bit(F + lane + 4 * prev):
            node = 1
            while node < 256: node = node * 2 + d.bit(L + lane * 256 + node)
            out.append(node & 0xFF)
            prev = 0
        else:
            off = gamma_dec(d, OH) - 1
            for j in range(K - 1, -1, -1): off = off << 1 | d.bit(OL + j)
            off += 1
            for _ in range(gamma_dec(d, LEN) + MIN_LEN - 1): out.append(out[-off])
            prev = 1
    return bytes(out)


if __name__ == "__main__":
    data = open(sys.argv[1], "rb").read()
    p = pack(data)
    assert unpack(p, len(data)) == data
    open(sys.argv[2], "wb").write(p)
    print(len(data), "->", len(p), "bytes")
