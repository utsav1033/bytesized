// v0.c: plain, readable SmolLM2-135M inference. Reads the original HF files.
// usage (run inside the model folder): ./v0 "prompt" [n_tokens]
// stdout: generated text. stderr: generated token ids (for the exact-match check).
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>

// ---- model shape (SmolLM2-135M config.json) ----
#define DIM 576
#define HID 1536
#define LAYERS 30
#define HEADS 9
#define KV_HEADS 3
#define HEAD_DIM 64
#define KV_DIM (KV_HEADS * HEAD_DIM)
#define VOCAB 49152
#define MAX_SEQ 1024
#define ROPE_THETA 100000.0f
#define EPS 1e-5f

static void *map_file(const char *path, size_t *size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    struct stat st;
    fstat(fd, &st);
    *size = st.st_size;
    void *p = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    return p;
}

// ================= weights (safetensors) =================
// File = 8-byte header length, JSON header, raw tensor bytes.
// Every tensor here is bf16: the top 16 bits of an fp32.
typedef uint16_t bf16;
static const char *st_header;
static const uint8_t *st_data;

static const bf16 *tensor(const char *name) {
    char key[128];
    snprintf(key, sizeof key, "\"%s\"", name);
    const char *p = strstr(st_header, key);
    if (!p) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    p = strstr(p, "data_offsets") + strlen("data_offsets\":[");
    return (const bf16 *)(st_data + strtoull(p, 0, 10));
}

static float bf2f(bf16 b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

typedef struct {
    const bf16 *attn_norm, *wq, *wk, *wv, *wo, *mlp_norm, *gate, *up, *down;
} Layer;
static const bf16 *embed, *final_norm;
static Layer layers[LAYERS];

static void load_model(void) {
    size_t size;
    const uint8_t *f = map_file("model.safetensors", &size);
    uint64_t hlen;
    memcpy(&hlen, f, 8);
    char *h = malloc(hlen + 1);
    memcpy(h, f + 8, hlen);
    h[hlen] = 0;
    st_header = h;
    st_data = f + 8 + hlen;

    embed = tensor("model.embed_tokens.weight");
    final_norm = tensor("model.norm.weight");
    char n[128];
    for (int l = 0; l < LAYERS; l++) {
        Layer *L = &layers[l];
#define T(field, suffix) snprintf(n, sizeof n, "model.layers.%d.%s", l, suffix), L->field = tensor(n)
        T(attn_norm, "input_layernorm.weight");
        T(wq, "self_attn.q_proj.weight");
        T(wk, "self_attn.k_proj.weight");
        T(wv, "self_attn.v_proj.weight");
        T(wo, "self_attn.o_proj.weight");
        T(mlp_norm, "post_attention_layernorm.weight");
        T(gate, "mlp.gate_proj.weight");
        T(up, "mlp.up_proj.weight");
        T(down, "mlp.down_proj.weight");
#undef T
    }
}

// ================= the 5 kernels =================
// out[r] = sum_c W[r][c] * x[c]   (W is rows x cols, row-major, bf16)
static void matmul(float *out, const bf16 *W, const float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const bf16 *w = W + (size_t)r * cols;
        float s = 0;
        for (int c = 0; c < cols; c++) s += bf2f(w[c]) * x[c];
        out[r] = s;
    }
}

static void rmsnorm(float *out, const float *x, const bf16 *w) {
    float ss = 0;
    for (int i = 0; i < DIM; i++) ss += x[i] * x[i];
    float k = 1.0f / sqrtf(ss / DIM + EPS);
    for (int i = 0; i < DIM; i++) out[i] = bf2f(w[i]) * (x[i] * k);
}

// rotate-half RoPE: pairs (i, i+32) inside each 64-wide head
static void rope(float *v, int n_heads, int pos) {
    for (int h = 0; h < n_heads; h++) {
        float *x = v + h * HEAD_DIM;
        for (int i = 0; i < HEAD_DIM / 2; i++) {
            float freq = 1.0f / powf(ROPE_THETA, (float)(2 * i) / HEAD_DIM);
            float a = pos * freq, c = cosf(a), s = sinf(a);
            float x1 = x[i], x2 = x[i + HEAD_DIM / 2];
            x[i] = x1 * c - x2 * s;
            x[i + HEAD_DIM / 2] = x2 * c + x1 * s;
        }
    }
}

static void softmax(float *x, int n) {
    float m = x[0], sum = 0;
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    for (int i = 0; i < n; i++) sum += x[i] = expf(x[i] - m);
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

// ================= forward: one token in, 49152 scores out =================
static float x[DIM], xb[DIM], q[DIM], att_out[DIM], tmp[DIM];
static float hb[HID], hb2[HID], scores[MAX_SEQ], logits[VOCAB];
static float key_cache[LAYERS][MAX_SEQ][KV_DIM], val_cache[LAYERS][MAX_SEQ][KV_DIM];

static float *forward(int token, int pos) {
    const bf16 *e = embed + (size_t)token * DIM;
    for (int i = 0; i < DIM; i++) x[i] = bf2f(e[i]);

    for (int l = 0; l < LAYERS; l++) {
        Layer *L = &layers[l];
        float *k = key_cache[l][pos], *v = val_cache[l][pos];

        // attention
        rmsnorm(xb, x, L->attn_norm);
        matmul(q, L->wq, xb, DIM, DIM);
        matmul(k, L->wk, xb, KV_DIM, DIM);
        matmul(v, L->wv, xb, KV_DIM, DIM);
        rope(q, HEADS, pos);
        rope(k, KV_HEADS, pos);
        for (int h = 0; h < HEADS; h++) {
            int kvh = h / (HEADS / KV_HEADS);   // 3 query heads share 1 kv head
            float *qh = q + h * HEAD_DIM;
            for (int t = 0; t <= pos; t++) {
                float *kt = key_cache[l][t] + kvh * HEAD_DIM, s = 0;
                for (int i = 0; i < HEAD_DIM; i++) s += qh[i] * kt[i];
                scores[t] = s / sqrtf(HEAD_DIM);
            }
            softmax(scores, pos + 1);
            float *o = att_out + h * HEAD_DIM;
            memset(o, 0, HEAD_DIM * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *vt = val_cache[l][t] + kvh * HEAD_DIM;
                for (int i = 0; i < HEAD_DIM; i++) o[i] += scores[t] * vt[i];
            }
        }
        matmul(tmp, L->wo, att_out, DIM, DIM);
        for (int i = 0; i < DIM; i++) x[i] += tmp[i];

        // MLP
        rmsnorm(xb, x, L->mlp_norm);
        matmul(hb, L->gate, xb, HID, DIM);
        matmul(hb2, L->up, xb, HID, DIM);
        for (int i = 0; i < HID; i++) hb[i] = silu(hb[i]) * hb2[i];
        matmul(tmp, L->down, hb, DIM, HID);
        for (int i = 0; i < DIM; i++) x[i] += tmp[i];
    }
    rmsnorm(xb, x, final_norm);
    matmul(logits, embed, xb, VOCAB, DIM);   // tied: the embedding table scores every token
    return logits;
}

// ================= tokenizer (byte-level BPE from tokenizer.json) =================
// Vocab strings use GPT-2's byte->unicode trick (space = 'Ġ', U+0120).
// We turn every token back into raw bytes, and do BPE on raw bytes.
static char *tok_bytes[VOCAB];   // raw bytes of each token
static int tok_len[VOCAB];
static int cp_to_byte[324];      // unicode codepoint (GPT-2 alphabet) -> byte

static void build_byte_table(void) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174);
        cp_to_byte[printable ? b : 256 + n++] = b;
    }
    cp_to_byte[' '] = ' ';   // not in the alphabet: only the separator inside "a b" merges
}

// read one JSON string starting after the opening quote; returns raw bytes (GPT-2 alphabet undone)
static int sep_at;   // where a literal ' ' was seen (the separator in "a b" merges)
static const char *read_token_string(const char *p, char *out, int *len) {
    *len = 0;
    while (*p != '"') {
        unsigned cp;
        if (*p == '\\') {
            p++;
            if (*p == 'u') { cp = strtoul((char[]){p[1], p[2], p[3], p[4], 0}, 0, 16); p += 5; }
            else { cp = *p == 'n' ? '\n' : *p == 't' ? '\t' : *p; p++; }
        } else if ((uint8_t)*p < 0x80) cp = *p++;
        else if ((*p & 0xE0) == 0xC0) { cp = (p[0] & 0x1F) << 6 | (p[1] & 0x3F); p += 2; }
        else { cp = (p[0] & 0x0F) << 12 | (p[1] & 0x3F) << 6 | (p[2] & 0x3F); p += 3; }
        if (cp == ' ') sep_at = *len;
        out[(*len)++] = cp < 324 ? cp_to_byte[cp] : '?';
    }
    return p + 1;
}

// merge table: hash (left id, right id) -> rank and merged id
#define MH (1 << 17)
static int mh_left[MH], mh_right[MH], mh_rank[MH], mh_out[MH];

// vocab hash: raw bytes -> id
static int vh[MH];   // id + 1, 0 = empty
static unsigned str_hash(const char *s, int len) {
    unsigned h = 2166136261u;
    while (len--) h = (h ^ (uint8_t)*s++) * 16777619u;
    return h & (MH - 1);
}
static void add_token(int id) {
    unsigned h = str_hash(tok_bytes[id], tok_len[id]);
    while (vh[h]) h = (h + 1) & (MH - 1);
    vh[h] = id + 1;
}
static int find_token(const char *s, int len) {
    for (unsigned h = str_hash(s, len); vh[h]; h = (h + 1) & (MH - 1)) {
        int i = vh[h] - 1;
        if (tok_len[i] == len && !memcmp(tok_bytes[i], s, len)) return i;
    }
    return -1;
}

static int byte_token[256];

static void load_tokenizer(void) {
    build_byte_table();
    size_t size;
    const char *j = map_file("tokenizer.json", &size);
    char buf[1024];
    int len;

    // "vocab": {"tok": id, ...}
    const char *p = strstr(j, "\"vocab\"");
    p = strchr(p, '{') + 1;
    while (*p != '}') {
        p = strchr(p, '"') + 1;
        p = read_token_string(p, buf, &len);
        int id = strtol(p + 1, (char **)&p, 10);
        tok_bytes[id] = malloc(len);
        memcpy(tok_bytes[id], buf, len);
        tok_len[id] = len;
        add_token(id);
        while (*p == ',' || *p == ' ' || *p == '\n') p++;
    }
    for (int b = 0; b < 256; b++) byte_token[b] = find_token((char[]){b}, 1);

    // "merges": ["a b", ...]  (rank = position)
    for (int i = 0; i < MH; i++) mh_left[i] = -1;
    p = strstr(p, "\"merges\"");
    p = strchr(p, '[') + 1;
    for (int rank = 0;; rank++) {
        while (*p != '"' && *p != ']') p++;
        if (*p == ']') break;
        p = read_token_string(p + 1, buf, &len);
        char *sp = buf + sep_at;
        int a = find_token(buf, sp - buf), b = find_token(sp + 1, buf + len - sp - 1);
        memmove(sp, sp + 1, buf + len - sp - 1);
        int out = find_token(buf, len - 1);
        unsigned h = ((unsigned)a * 49157u + b) & (MH - 1);
        while (mh_left[h] >= 0) h = (h + 1) & (MH - 1);
        mh_left[h] = a, mh_right[h] = b, mh_rank[h] = rank, mh_out[h] = out;
    }
}

static int merge_lookup(int a, int b, int *out) {
    for (unsigned h = ((unsigned)a * 49157u + b) & (MH - 1); mh_left[h] >= 0; h = (h + 1) & (MH - 1))
        if (mh_left[h] == a && mh_right[h] == b) { *out = mh_out[h]; return mh_rank[h]; }
    return 1 << 30;
}

// BPE one pre-token (raw bytes) into ids
static int bpe(const char *s, int n, int *ids) {
    for (int i = 0; i < n; i++) ids[i] = byte_token[(uint8_t)s[i]];
    for (;;) {
        int best = 1 << 30, at = -1, merged = 0, m;
        for (int i = 0; i + 1 < n; i++) {
            int r = merge_lookup(ids[i], ids[i + 1], &m);
            if (r < best) best = r, at = i, merged = m;
        }
        if (at < 0) return n;
        ids[at] = merged;
        memmove(ids + at + 1, ids + at + 2, (n - at - 2) * sizeof(int));
        n--;
    }
}

// GPT-2 pre-tokenizer (ASCII view; bytes >= 0x80 count as letters) + single digits.
static int is_letter(int c) { return ((c | 32) >= 'a' && (c | 32) <= 'z') || c >= 0x80; }
static int is_digit(int c) { return c >= '0' && c <= '9'; }
static int is_space(int c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }

static int encode(const char *s, int *ids) {
    int n = 0, len = strlen(s);
    for (int i = 0; i < len;) {
        int j = i;
        if (s[i] == '\'' && strchr("stmd", s[i + 1]) && s[i + 1]) j = i + 2;
        else if (s[i] == '\'' && (!strncmp(s + i, "'re", 3) || !strncmp(s + i, "'ve", 3) || !strncmp(s + i, "'ll", 3))) j = i + 3;
        else if (is_digit(s[i])) j = i + 1;
        else {
            int k = i + (s[i] == ' ' && i + 1 < len && !is_space(s[i + 1]));
            if (is_letter((uint8_t)s[k])) { j = k; while (j < len && is_letter((uint8_t)s[j])) j++; }
            else if (!is_space(s[k]) && !is_digit(s[k])) { j = k; while (j < len && !is_space(s[j]) && !is_letter((uint8_t)s[j]) && !is_digit(s[j])) j++; }
            else {   // whitespace run; leave the last space for the next word
                while (j < len && is_space(s[j])) j++;
                if (j < len && j - i > 1) j--;
            }
        }
        n += bpe(s + i, j - i, ids + n);
        i = j;
    }
    return n;
}

// ================= main: encode, run, greedy pick, decode =================
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s \"prompt\" [n]\n", argv[0]); return 1; }
    int n_new = argc > 2 ? atoi(argv[2]) : 100;
    load_model();
    load_tokenizer();

    int ids[MAX_SEQ];
    int n = encode(argv[1], ids);
    if (n + n_new > MAX_SEQ) n_new = MAX_SEQ - n;
    for (int i = 0; i < n; i++) fprintf(stderr, "%d ", ids[i]);
    fputs("| ", stderr);
    fputs(argv[1], stdout);

    for (int pos = 0; pos < n + n_new - 1; pos++) {
        float *lg = forward(ids[pos], pos);
        if (pos < n - 1) continue;          // still reading the prompt
        int best = 0;
        for (int i = 1; i < VOCAB; i++) if (lg[i] > lg[best]) best = i;
        ids[pos + 1] = best;
        fwrite(tok_bytes[best], 1, tok_len[best], stdout);
        fflush(stdout);
        fprintf(stderr, "%d ", best);
    }
    fputc('\n', stdout);
    fputc('\n', stderr);
    return 0;
}
