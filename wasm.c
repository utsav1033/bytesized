// wasm.c: tiny.c as a WebAssembly module. No libc, no syscalls: the host (run.js or a web page)
// copies tokenizer.json, model.safetensors and the prompt into memory at fixed addresses,
// then calls r(prompt_len, n_new). Text goes out through the one import, a.p(ptr, len).
// Build with -DCHECK to also report token ids through a.q(id), -1 = end of prompt.
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned u32;
typedef unsigned long long u64;

#define DIM 576
#define HID 1536
#define LAYERS 30
#define HEADS 9
#define KV_DIM 192
#define HD 64
#define VOCAB 49152
#define MAX_SEQ 1024

// ---------------- host ----------------
#define PROMPT_AT 0x3300000   // above the ~48 MB of static buffers
#define TOK_AT 0x3400000
#define MODEL_AT 0x3800000
__attribute__((import_module("a"), import_name("p"))) void out(const void *p, int n);
#ifdef CHECK
__attribute__((import_module("a"), import_name("q"))) void put_num(int id);
#endif

// compilers may emit calls to this
void *memset(void *d, int c, u64 n) { u8 *p = d; while (n--) *p++ = c; return d; }

// ---------------- math (double inside, so results round like libm's float versions) ----------------
static double exp_d(double x) {   // no range clamp: every argument here stays well inside +-700
    double k = __builtin_rint(x * 1.4426950408889634);
    double r = x - k * 0.6931471805599453, t = 1;   // error from the short ln2 is ~1e-15: invisible in float
    for (int i = 17; i > 0; i--) t = 1 + t * r / i;           // Taylor, Horner form
    union { double d; u64 u; } s = { .u = (u64)((long long)k + 1023) << 52 };
    return t * s.d;
}
// cos: reduce to |r| <= pi, then Taylor. sin(a) = cos(a - pi/2)
static float cos_f(double a) {
    double r = a - __builtin_rint(a * 0.15915494309189535) * 6.283185307179586, t = 1;
    for (int i = 40; i > 0; i -= 2) t = 1 - t * r * r / (i * (i - 1));
    return t;
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
    const char *f = (const char *)MODEL_AT, *p = f + 8;
    const u8 *data = (const u8 *)p + *(const u64 *)f;
    for (int i = 0; i < 272; i++) {
        p = find(p, ".");
        const u16 **slot = *p == 'e' ? &embed : *p == 'n' ? &fnorm : &W[num(&p)][(i - 1) % 9];
        p = find(p, "ts");
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
        float a = pos * freq, s = cos_f(a - 1.5707963267948966), c = cos_f(a);
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
    const char *p = find((const char *)TOK_AT, "b\":");
    u8 *o = pool;
    for (int k = 0; k < VOCAB; k++) {   // the vocab has exactly VOCAB entries
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
    }
}
// BPE one pre-token, tracking token boundaries inside s
static int bpe(const u8 *s, int n, int *ids) {
    static int st[256], ln[256];
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


// ---------------- main ----------------
static int ids[MAX_SEQ];
__attribute__((export_name("r"))) void r(int len, int n_new) {
    const u8 *prompt = (const u8 *)PROMPT_AT;
    load_model();
    load_vocab();
    int n = encode(prompt, len, ids);
#ifdef CHECK
    for (int i = 0; i < n; i++) put_num(ids[i]);
    put_num(-1);
#endif
    out(prompt, len);
    for (int pos = 0; pos < n + n_new - 1 && pos < MAX_SEQ - 1; pos++) {
        forward(ids[pos], pos);
        if (pos < n - 1) continue;
        int best = 0;
        for (int i = 1; i < VOCAB; i++) if (logits[i] > logits[best]) best = i;
        ids[pos + 1] = best;
        out(tb[best], tl[best]);
#ifdef CHECK
        put_num(best);
#endif
    }
}
