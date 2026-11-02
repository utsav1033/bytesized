// stub.S: unpack the LZ77 payload (format in lz.py) to OUT, flush caches, jump to ENTRY.
// OUT, SIZE, ENTRY come from pack.sh.
//   x0 payload read pointer      w1 bit buffer: unread bits left-aligned, then a 1 sentinel
//   x4 write pointer             x6 end of output     w2/w3 count/value for rd     x7 return for rd/gamma
.global _start
_start:
    adr  x0, payload
    mov  w1, #0x80000000              // sentinel only: the first getbit refills
    mov  x4, #OUT
    add  x6, x4, #(SIZE & 0xfff)
.if SIZE > 0xfff
    add  x6, x6, #(SIZE >> 12), lsl #12
.endif
loop:
    cmp  x4, x6
    b.hs done
    bl   getbit
    b.cs match
    mov  w2, #8                       // literal: 8 raw bits
    mov  w3, #0
    adr  x7, 1f
    b    rd
1:  strb w3, [x4], #1
    b    loop
match:
    adr  x7, 1f                       // offset = ((gamma - 1) << 7 | 7 raw bits) + 1
    b    gamma
1:  sub  w3, w3, #1
    mov  w2, #7
    adr  x7, 1f
    b    rd
1:  add  w8, w3, #1
    adr  x7, 1f                       // length = gamma + 1
    b    gamma
1:  add  w3, w3, #1
2:  sub  x9, x4, x8
    ldrb w9, [x9]
    strb w9, [x4], #1
    subs w3, w3, #1
    b.ne 2b
    b    loop

// gamma: n zero bits, a 1, then n bits  ->  w3 = 1 << n | bits
gamma:
    mov  w2, #0
1:  bl   getbit
    b.cs 2f
    add  w2, w2, #1
    b    1b
2:  mov  w3, #1
// rd: shift w2 more bits into w3, return to x7
rd: cbz  w2, 1f
    bl   getbit
    adc  w3, w3, w3
    sub  w2, w2, #1
    b    rd
1:  br   x7

// getbit: next bit -> carry flag
getbit:
    adds w1, w1, w1                   // shift left; the bit falls into carry
    b.ne 1f
    ldrb w1, [x0], #1                 // refill: byte in the top 8 bits, sentinel below it
    lsl  w1, w1, #24
    orr  w1, w1, #0x800000
    adds w1, w1, w1                   // shift left; the bit falls into carry
1:  ret

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
