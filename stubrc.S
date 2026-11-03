// stubrc2.S: range-decode the LZ77 payload (format in rc.py) to OUT, flush caches, jump to OUT.
// OUT, SIZE, PAYLOAD come from pack.sh; the engine's entry point must be OUT.
//   x0 payload read pointer   w1 range   w2 code   x3 probabilities (u32, at PROBS)
//   x4 write pointer          x6 end of output      x7 OUT
// Probabilities are stored as p - 2048, so the zeroed memory the kernel gives us already means 1/2.
#define PROBS 0x480000
#define OH 1032
#define OL 1320
#define LEN 1327
.global _start
_start:
    adr  x0, payload
    mov  x3, #PROBS
    mov  w1, #-1                      // range = 0xffffffff
    ldr  w2, [x0], #4                 // code = first 4 payload bytes, big-endian
    rev  w2, w2
    mov  x7, #OUT
    mov  x4, #OUT
    add  x6, x4, #(SIZE & 0xfff)
.if SIZE > 0xfff
    add  x6, x6, #(SIZE >> 12), lsl #12
.endif
loop:
    cmp  x4, x6
    b.hs done
    ubfiz w20, w4, #8, #2             // lane * 256: slot 0 is this lane's match flag, 1..255 its literal tree
    mov  w9, w20
    bl   dbit
    cbnz w10, match
    mov  w21, #1
1:  add  w9, w20, w21
    bl   dbit
    add  w21, w10, w21, lsl #1
    tbz  w21, #8, 1b
    strb w21, [x4], #1
    b    loop
match:
    mov  w16, #OH                     // offset = ((gamma - 1) << 3 | 3 bits) + 1
    bl   gamma
    sub  w22, w17, #1
    mov  w23, #3
1:  sub  w23, w23, #1
    add  w9, w23, #OL
    bl   dbit
    add  w22, w10, w22, lsl #1
    cbnz w23, 1b
    add  w22, w22, #1
    mov  w16, #LEN                    // length = gamma + 1
    bl   gamma
    add  w17, w17, #1
1:  sub  x9, x4, x22
    ldrb w9, [x9]
    strb w9, [x4], #1
    subs w17, w17, #1
    b.ne 1b
    b    loop

// gamma: unary n (contexts base+j), then n bits (contexts base+16n+j) -> w17 = 1 << n | bits
gamma:
    mov  x15, x30
    mov  w18, #-1
1:  add  w18, w18, #1
    add  w9, w16, w18
    bl   dbit
    cbz  w10, 1b
    mov  w17, #1
    add  w24, w16, w18, lsl #4
2:  cbz  w18, 3f
    sub  w18, w18, #1
    add  w9, w24, w18
    bl   dbit
    add  w17, w10, w17, lsl #1
    b    2b
3:  br   x15

// dbit: decode one bit with probability index w9 -> w10, adapt, renormalize
dbit:
    add  x11, x3, x9, lsl #2
    ldr  w12, [x11]                   // q = p - 2048
    add  w8, w12, #2048               // p
    lsr  w13, w1, #12
    mul  w13, w13, w8                 // bound
    cmp  w2, w13
    cset w10, hs
    b.hs 1f
    mov  w1, w13                      // 0: range = bound, p += (4096 - p) >> 4
    mov  w14, #2048
    sub  w14, w14, w12
    add  w12, w12, w14, lsr #4
    b    2f
1:  sub  w1, w1, w13                  // 1: range -= bound, code -= bound, p -= p >> 4
    sub  w2, w2, w13
    sub  w12, w12, w8, lsr #4
2:  str  w12, [x11]
3:  lsr  w14, w1, #24                 // while range < 2^24: shift in a byte
    cbnz w14, 4f
    lsl  w1, w1, #8
    ldrb w14, [x0], #1
    orr  w2, w14, w2, lsl #8
    b    3b
4:  ret

done:                                 // make the new code visible to instruction fetch, last line first
    dc   cvau, x4
    ic   ivau, x4
    sub  x4, x4, #64
    cmp  x4, x7
    b.hs done
    dsb  ish
    isb
    br   x7
payload:
    .incbin PAYLOAD
