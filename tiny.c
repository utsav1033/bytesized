// tiny.c: SmolLM2-135M with no libc. Linux aarch64, raw syscalls, own math.
// Same math as v0.c. Run inside the model folder: ./tiny "prompt" [n]
// Build with -DCHECK to also print token ids to stderr (for check.py).
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned u32;
typedef unsigned long u64;

#define DIM 576
#define HID 1536
#define LAYERS 30
#define HEADS 9
#define KV_DIM 192
#define HD 64
#define VOCAB 49152
#define MAX_SEQ 1024

// ---------------- syscalls ----------------
static long sys(long n, long a, long b, long c, long d, long e) {
    register long x8 asm("x8") = n, x0 asm("x0") = a, x1 asm("x1") = b, x2 asm("x2") = c,
                  x3 asm("x3") = d, x4 asm("x4") = e, x5 asm("x5") = 0;   // x5: mmap offset
    asm volatile("svc 0" : "+r"(x0) : "r"(x8), "r"(x1), "r"(x2), "r"(x3), "r"(x4), "r"(x5) : "memory");
    return x0;
}
static const char *map(const char *path) {
    long fd = sys(56, -100, (long)path, 0, 0, 0);          // openat(AT_FDCWD, path, O_RDONLY)
    long size = sys(62, fd, 0, 2, 0, 0);                   // lseek(fd, 0, SEEK_END)
    return (const char *)sys(222, 0, size, 1, 2, fd);      // mmap(PROT_READ, MAP_PRIVATE), offset 0
}
static void out(int fd, const void *p, long n) { sys(64, fd, (long)p, n, 0, 0); }

// compilers may emit calls to this
void *memset(void *d, int c, u64 n) { u8 *p = d; while (n--) *p++ = c; return d; }

// ---------------- math (double inside, so results round like libm's float versions) ----------------
static double exp_d(double x) {   // no range clamp: every argument here stays well inside +-700
    double k = __builtin_round(x * 1.4426950408889634);
    double r = x - k * 0.6931471805599453, t = 1;   // error from the short ln2 is ~1e-15: invisible in float
    for (int i = 17; i > 0; i--) t = 1 + t * r / i;           // Taylor, Horner form
    union { double d; u64 u; } s = { .u = (u64)((long)k + 1023) << 52 };
    return t * s.d;
}
// sin and cos of a float angle: reduce to |r| <= pi/4, then Taylor
static void sincos_f(float a, float *s, float *c) {
    double k = __builtin_round(a * 0.6366197723675814), r = a - k * 1.5707963267948966;
    double sr = 1, cr = 1;
    for (int i = 21; i > 1; i -= 2) sr = 1 - sr * r * r / (i * (i - 1)), cr = 1 - cr * r * r / ((i - 1) * (i - 2));
    sr *= r;
    long q = (long)k & 3;
    if (q & 1) { double t = sr; sr = cr, cr = -t; }
    if (q & 2) sr = -sr, cr = -cr;
    *s = sr, *c = cr;
}

// ---------------- weights ----------------
static float bf(u16 b) { union { u32 u; float f; } v = { (u32)b << 16 }; return v.f; }

static const char *find(const char *h, const char *s) {
    for (;; h++) {
        int i = 0;
        while (s[i] && h[i] == s[i]) i++;
        if (!s[i]) return h + i;
    }
}
static u64 num(const char **pp) {
    const char *p = *pp;
    u64 n = 0;
    while (*p < '0' || *p > '9') p++;
    while (*p >= '0' && *p <= '9') n = n * 10 + *p++ - '0';
    *pp = p;
    return n;
}

// The header lists tensors sorted by name: embed_tokens, then per layer (in string order 0,1,10,..)
// input_layernorm, mlp.down/gate/up, post_attention_layernorm, self_attn.k/o/q/v, then norm.
enum { IN_NORM, DOWN, GATE, UP, POST_NORM, WK, WO, WQ, WV };
static const u16 *embed, *fnorm, *W[LAYERS][9];

static void load_model(void) {
    const char *f = map("model.safetensors"), *p = f + 8;
    const u8 *data = (const u8 *)p + *(const u64 *)f;
    for (int i = 0; i < 272; i++) {
        p = find(p, "\"model.");
        const u16 **slot = *p == 'e' ? &embed : *p == 'n' ? &fnorm : &W[num(&p)][(i - 1) % 9];
        p = find(p, "data_offsets");
        *slot = (const u16 *)(data + num(&p));
    }
}

// ---------------- kernels ----------------
static void matmul(float *o, const u16 *w, const float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++, w += cols) {
        float s = 0;
        for (int c = 0; c < cols; c++) s += bf(w[c]) * x[c];
        o[r] = s;
    }
}
static void rmsnorm(float *o, const float *x, const u16 *w) {
    float ss = 0;
    for (int i = 0; i < DIM; i++) ss += x[i] * x[i];
    float k = 1 / __builtin_sqrtf(ss / DIM + 1e-5f);
    for (int i = 0; i < DIM; i++) o[i] = bf(w[i]) * (x[i] * k);
}
static void rope(float *v, int heads, int pos) {
    for (int i = 0; i < HD / 2; i++) {
        float freq = 1 / (float)exp_d(i * 0.35977892077594466);   // 1 / 100000^(2i/64), ln(100000)/32 = 0.3598
        float s, c;
        sincos_f(pos * freq, &s, &c);
        for (int h = 0; h < heads; h++) {
            float *x = v + h * HD, a = x[i], b = x[i + HD / 2];
            x[i] = a * c - b * s;
            x[i + HD / 2] = b * c + a * s;
        }
    }
}

// ---------------- forward ----------------
static float x[DIM], xb[DIM], q[DIM], att[DIM], tmp[DIM], hb[HID], hb2[HID], sc[MAX_SEQ], logits[VOCAB];
static float kc[LAYERS][MAX_SEQ][KV_DIM], vc[LAYERS][MAX_SEQ][KV_DIM];

static void forward(int tok, int pos) {
    for (int i = 0; i < DIM; i++) x[i] = bf(embed[tok * DIM + i]);
    for (int l = 0; l < LAYERS; l++) {
        const u16 **w = W[l];
        float *k = kc[l][pos], *v = vc[l][pos];
        rmsnorm(xb, x, w[IN_NORM]);
        matmul(q, w[WQ], xb, DIM, DIM);
        matmul(k, w[WK], xb, KV_DIM, DIM);
        matmul(v, w[WV], xb, KV_DIM, DIM);
        rope(q, HEADS, pos);
        rope(k, 3, pos);
        for (int h = 0; h < HEADS; h++) {
            int kv = h / 3 * HD;
            float m = -1e30f, sum = 0, *o = att + h * HD;
            for (int t = 0; t <= pos; t++) {
                float s = 0;
                for (int i = 0; i < HD; i++) s += q[h * HD + i] * kc[l][t][kv + i];
                sc[t] = s / 8;                                   // / sqrt(64)
                if (sc[t] > m) m = sc[t];
            }
            for (int t = 0; t <= pos; t++) sum += sc[t] = exp_d(sc[t] - m);
            for (int i = 0; i < HD; i++) o[i] = 0;
            for (int t = 0; t <= pos; t++)
                for (int i = 0; i < HD; i++) o[i] += sc[t] / sum * vc[l][t][kv + i];
        }
        matmul(tmp, w[WO], att, DIM, DIM);
        for (int i = 0; i < DIM; i++) x[i] += tmp[i];
        rmsnorm(xb, x, w[POST_NORM]);
        matmul(hb, w[GATE], xb, HID, DIM);
        matmul(hb2, w[UP], xb, HID, DIM);
        for (int i = 0; i < HID; i++) hb[i] = hb[i] / (float)(1 + exp_d(-hb[i])) * hb2[i];
        matmul(tmp, w[DOWN], hb, DIM, HID);
        for (int i = 0; i < DIM; i++) x[i] += tmp[i];
    }
    rmsnorm(xb, x, fnorm);
    matmul(logits, embed, xb, VOCAB, DIM);
}

// ---------------- tokenizer ----------------
// Only the vocab is read. In this tokenizer every merge "a b" produces token id rank + 252,
// so "merge the adjacent pair whose concatenation has the lowest id" is exactly BPE.
static u8 pool[1 << 20], *tb[VOCAB];
static int tl[VOCAB];

static int lookup(const u8 *s, int n) {
    for (int i = 0; i < VOCAB; i++) {
        int j = 0;
        while (j < n && j < tl[i] && tb[i][j] == s[j]) j++;
        if (j == n && tl[i] == n) return i;
    }
    return 1 << 30;
}
static void load_vocab(void) {
    const char *p = find(map("tokenizer.json"), "\"vocab\": {");
    u8 *o = pool;
    while (*p != '}') {
        u8 *s = o;
        while (*p++ != '"');
        while (*p != '"') {   // only \" and \\ escapes, and only 1-2 byte UTF-8, appear in the vocab
            u32 c = (u8)*p++;
            if (c == '\\') c = *p++;
            else if (c >= 0x80) c = (c & 31) << 6 | (*p++ & 63);
            // undo GPT-2's alphabet: printable bytes are themselves, the n-th unprintable byte is 256 + n
            *o++ = c < 256 ? c : c < 289 ? c - 256 : c < 323 ? c - 162 : 173;
        }
        p++;
        int id = num(&p);
        tb[id] = s, tl[id] = o - s;
        while (*p == ',' || *p == ' ' || *p == '\n') p++;
    }
}
// BPE one pre-token, tracking token boundaries inside s
static int bpe(const u8 *s, int n, int *ids) {
    int st[256], ln[256];
    for (int i = 0; i < n; i++) st[i] = i, ln[i] = 1;
    for (;;) {
        int best = 1 << 30, at = 0;
        for (int i = 0; i + 1 < n; i++) {
            int id = lookup(s + st[i], ln[i] + ln[i + 1]);
            if (id < best) best = id, at = i;
        }
        if (best == 1 << 30) break;
        ln[at] += ln[at + 1];
        for (int i = at + 1; i < n - 1; i++) st[i] = st[i + 1], ln[i] = ln[i + 1];
        n--;
    }
    for (int i = 0; i < n; i++) ids[i] = lookup(s + st[i], ln[i]);
    return n;
}
// character class: 0 whitespace, 1 letter (any byte >= 0x80 counts), 2 digit, 3 other
static int cls(int c) {
    return c == ' ' || c == '\n' || c == '\t' || c == '\r' ? 0 : c >= '0' && c <= '9' ? 2
         : ((c | 32) >= 'a' && (c | 32) <= 'z') || c >= 0x80 ? 1 : 3;
}
// GPT-2 pre-tokenizer split (+ every digit alone), then BPE each piece
static int encode(const u8 *s, int len, int *ids) {
    int n = 0;
    for (int i = 0; i < len;) {
        int c = s[i + 1], j = i, k = i + (s[i] == ' ' && c && cls(c));   // k: skip one leading space
        if (s[i] == '\'' && (c == 's' || c == 't' || c == 'm' || c == 'd')) j = i + 2;
        else if (s[i] == '\'' && (((c == 'r' || c == 'v') && s[i + 2] == 'e') || (c == 'l' && s[i + 2] == 'l'))) j = i + 3;
        else if (cls(s[i]) == 2) j = i + 1;
        else if (cls(s[k]) & 1) for (j = k; j < len && cls(s[j]) == cls(s[k]); j++);   // run of letters, or of "other"
        else {   // whitespace run; leave the last one for the next word
            while (j < len && !cls(s[j])) j++;
            if (j < len && j - i > 1) j--;
        }
        n += bpe(s + i, j - i, ids + n);
        i = j;
    }
    return n;
}

#ifdef CHECK
static void put_num(int v) {
    char b[12], *p = b + 11;
    *p = ' ';
    do *--p = '0' + v % 10; while (v /= 10);
    out(2, p, b + 12 - p);
}
#endif

// ---------------- main ----------------
static int ids[MAX_SEQ];
__attribute__((used)) void start(long *sp) {
    char **argv = (char **)(sp + 1);
    const char *a = argv[2];
    int n_new = sp[0] > 2 ? num(&a) : 100, len = 0;
    load_model();
    load_vocab();
    while (argv[1][len]) len++;
    int n = encode((const u8 *)argv[1], len, ids);
#ifdef CHECK
    for (int i = 0; i < n; i++) put_num(ids[i]);
    out(2, "| ", 2);
#endif
    out(1, argv[1], len);
    for (int pos = 0; pos < n + n_new - 1 && pos < MAX_SEQ - 1; pos++) {
        forward(ids[pos], pos);
        if (pos < n - 1) continue;
        int best = 0;
        for (int i = 1; i < VOCAB; i++) if (logits[i] > logits[best]) best = i;
        ids[pos + 1] = best;
        out(1, tb[best], tl[best]);
#ifdef CHECK
        put_num(best);
#endif
    }
    out(1, "\n", 1);
    sys(93, 0, 0, 0, 0, 0);   // exit(0)
}

asm(".global _start\n_start:\n mov x0, sp\n b start");
