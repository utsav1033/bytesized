// stubrc.S: range-decode the LZ77 payload (format in rc.py) to OUT, flush caches, jump to ENTRY.
// OUT, SIZE, ENTRY, PAYLOAD come from pack.sh.
//   x0 payload read pointer   w1 range   w2 code   x3 probabilities (u32, at PROBS)
//   x4 write pointer          w5 previous token was a match   x6 end of output
// Linux starts a process with every general register zero, so w2 and w5 need no init.
#define PROBS 0x480000
#define L 8
#define OH 1032
#define OL 1320
#define LEN 1327
#define N_PROBS 1615
.global _start
_start:
    adr  x0, payload
    mov  x3, #PROBS
    mov  w9, #2048                    // every probability starts at 1/2
    mov  w12, #N_PROBS
1:  str  w9, [x3, x12, lsl #2]
    subs w12, w12, #1
    b.pl 1b
    mov  w1, #-1                      // range = 0xffffffff
    mov  w12, #5                      // code = first 5 bytes, big-endian
1:  ldrb w13, [x0], #1
    orr  w2, w13, w2, lsl #8
    subs w12, w12, #1
    b.ne 1b
    mov  x4, #OUT
    add  x6, x4, #(SIZE & 0xfff)
.if SIZE > 0xfff
    add  x6, x6, #(SIZE >> 12), lsl #12
.endif
loop:
    cmp  x4, x6
    b.hs done
    and  w20, w4, #3                  // lane = position in the 4-byte instruction
    add  w9, w20, w5, lsl #2          // flag context: lane + 4 * previous
    bl   dbit
    cbnz w10, match
    lsl  w20, w20, #8                 // literal: 8-bit tree, one tree per lane
    add  w20, w20, #L
    mov  w21, #1
1:  add  w9, w20, w21
    bl   dbit
    add  w21, w10, w21, lsl #1
    cmp  w21, #256
    b.lo 1b
    strb w21, [x4], #1
    mov  w5, #0
    b    loop
match:
    mov  w16, #OH                     // offset = ((gamma - 1) << 7 | 7 bits) + 1
    bl   gamma
    sub  w22, w17, #1
    mov  w23, #7
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
    mov  w5, #1
    b    loop

// gamma: unary n (contexts base+j), then n bits (contexts base+32+16n+j) -> w17 = 1 << n | bits
gamma:
    mov  x15, x30
    mov  w18, #0
1:  add  w9, w16, w18
    bl   dbit
    cbnz w10, 2f
    add  w18, w18, #1
    b    1b
2:  mov  w17, #1
    mov  w19, w18
3:  cbz  w19, 4f
    sub  w19, w19, #1
    add  w9, w16, w18, lsl #4
    add  w9, w9, w19
    add  w9, w9, #32
    bl   dbit
    add  w17, w10, w17, lsl #1
    b    3b
4:  br   x15

// dbit: decode one bit with probability index w9 -> w10, adapt, renormalize
dbit:
    add  x11, x3, x9, lsl #2
    ldr  w12, [x11]
    lsr  w13, w1, #12
    mul  w13, w13, w12                // bound
    cmp  w2, w13
    b.hs 1f
    mov  w1, w13                      // 0: range = bound, p += (4096 - p) >> 4
    mov  w14, #4096
    sub  w14, w14, w12
    add  w12, w12, w14, lsr #4
    mov  w10, #0
    b    2f
1:  sub  w1, w1, w13                  // 1: range -= bound, code -= bound, p -= p >> 4
    sub  w2, w2, w13
    sub  w12, w12, w12, lsr #4
    mov  w10, #1
2:  str  w12, [x11]
3:  lsr  w14, w1, #24                 // while range < 2^24: shift in a byte
    cbnz w14, 4f
    lsl  w1, w1, #8
    ldrb w14, [x0], #1
    orr  w2, w14, w2, lsl #8
    b    3b
4:  ret

done:
    mov  x5, #OUT                     // make the new code visible to instruction fetch
1:  dc   cvau, x5
    ic   ivau, x5
    add  x5, x5, #64
    cmp  x5, x4
    b.lo 1b
    dsb  ish
    isb
    movz x16, #(ENTRY & 0xffff)
    movk x16, #(ENTRY >> 16), lsl #16
    br   x16
payload:
    .incbin PAYLOAD
