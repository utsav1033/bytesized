// stub.c: unpacks the LZ77 payload (see lz.py) to OUT, flushes the caches, jumps to ENTRY.
// OUT, SIZE, ENTRY come from pack.sh; the payload is appended by the linker script.
typedef unsigned char u8;
extern const u8 payload[];
static const u8 *src = payload;
static unsigned buf, cnt;

static unsigned bit(void) {
    if (!cnt) buf = *src++, cnt = 8;
    return buf >> --cnt & 1;
}
static unsigned bits(int n) { unsigned v = 0; while (n--) v = v << 1 | bit(); return v; }
static unsigned gamma(void) { int n = 0; while (!bit()) n++; return 1u << n | bits(n); }

__attribute__((used)) void unpack(void) {
    u8 *d = (u8 *)OUT;
    while (d < (u8 *)OUT + SIZE) {
        if (!bit()) *d++ = bits(8);
        else {
            int off = ((gamma() - 1) << 7 | bits(7)) + 1, len = gamma() + 1;   // signed: d[-off] must go back
            while (len--) *d = d[-off], d++;
        }
    }
    for (u8 *p = (u8 *)OUT; p < d; p += 64) asm volatile("dc cvau, %0\n ic ivau, %0" ::"r"(p));
    asm volatile("dsb ish\n isb");
}

asm(".global _start\n_start:\n bl unpack\n ldr x16, =" ENTRY_S "\n br x16");
