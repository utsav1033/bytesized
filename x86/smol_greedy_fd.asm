; smol.asm: SmolLM2-135M in hand-written x86-64, Linux, no libc. nasm -f bin.
; Experimental loose variant: greedy longest-match tokenizer, no pre-tokenizer, no final newline.
; Run in the model folder with 0<tokenizer.json 3<model.safetensors; 100 new tokens.
BITS 64
DEFAULT REL
ORG 0x10000

V      equ 49152
X      equ 0x80000000        ; residual stream; negative 32-bit address marks accumulation
XB     equ 0x80000           ; normed x
SC     equ 0x90000           ; attention scores, reused after each head
HB     equ 0x90000           ; MLP gate/up, after attention scores are dead
LG     equ 0x90000           ; logits, after all layers
IDS    equ 0xC0000           ; token IDs; starts immediately after 49,152 logits
M      equ 0xCB000           ; per prompt byte: 0 token start, 3 pre-token start, 1 merged
TMP    equ M-96              ; 96-byte scratch before markers (max decoded vocab: 81 bytes)
F      equ 0x8000000         ; every weight as fp32, layers in file (name) order
FL     equ F + 113246208     ; layer blocks, 14160384 bytes each
C      equ 0x40000000        ; cache: [layer][pos+1] rows of k | o (unused) | q | v, 6144 bytes
CSTEP  equ 1024*6144         ; cache bytes per layer
MEND   equ X + 0x1000

%ifdef DBG
ehdr:
    db 0x7F, "ELF", 2, 1, 1, 0
write:                     ; e_ident[8:16] is ignored by Linux
    push 1
    pop rax
    mov edi, eax
    syscall
    ret
    dw 2, 62
    dd 1
    dq _start
    dq phdr - $$
r532: dq 0.6978305848598664 ; ELF section-header offset, unused by Linux
l2e8: dd 0.18033688011112042 ; ELF flags, unused by Linux
    dw 64, 56
%ifdef DBG
    dw 1, 0, 0, 0            ; separate phdr, so gdb accepts the file
phdr:
    dd 1, 7
%else
phdr:
    dd 1, 7                  ; also e_phnum=1, e_shentsize=0, e_shnum=7, e_shstrndx=0
%endif
    dq 0, $$
eps: dd 0.00576              ; ELF physical address, unused by Linux
c24: dd 24
    dq filesz, MEND - $$


%else
ehdr:
    db 0x7F, "ELF", 2, 1, 1, 0
write:
    push 1
    pop rax
    mov edi, eax
    syscall
    ret
    dw 2, 62
    dd 1
    dq _start
    dq phdr - $$
r532: dq 0.6978305848598664 ; e_shoff
phdr: dd 1                   ; p_type, also e_flags
    dd 0x00380047            ; p_flags=RWX, e_ehsize=71, e_phentsize=56
    dq 1                     ; p_offset=1, e_phnum=1
    dq $$ + 1                ; p_vaddr congruent to p_offset
 eps: dd 0.00576             ; p_paddr
 c24: dd 24
    dq filesz - 1            ; p_filesz
l2e8: dd 0.18033688011112042 ; low 32 bits of p_memsz
    dd 1                     ; reserve >2 GiB virtual space for X


%endif

_start:
    xor edi, edi         ; tokenizer.json on fd 0
    call mapf
    push 24
    pop rcx
.fv: lodsb                   ; the vocab object is the 24th '{'
    cmp al, '{'
    jne .fv
    loop .fv
    push rsi                 ; vocab: entry k has id k

    ; ---- prompt: r12 = s, r13 = len
    mov r12, [rsp+24]
    mov rdi, r12
    xor eax, eax
    lea ecx, [rax-1]
    repne scasb
    not ecx
    dec ecx
    mov r13d, ecx

    ; ---- pre-token boundaries from the classes of 3 bytes: ' ' 0, letter/other 1, digit 2, other space 3
    mov r8d, IDS
    xor r15d, r15d           ; byte offset in prompt
    xor r14d, r14d           ; token count
.pos:
    pop rsi
    push rsi                 ; vocab start
    xor ebp, ebp             ; vocabulary id
    xor r10d, r10d           ; best byte length
.entry:
    mov edi, TMP
    call dec
    lea ecx, [rdi-TMP]
    cmp ecx, r10d
    jbe .skip
    push rsi
    push rcx
    lea rsi, [r12+r15]
    mov edi, TMP
    repe cmpsb
    pop rcx
    pop rsi
    jne .skip
    mov r10d, ecx
    mov r11d, ebp
.skip:
    inc ebp
    cmp bp, V
    jne .entry
    mov [r8+r14*4], r11d
    inc r14d
    add r15d, r10d
    cmp r15d, r13d
    jb .pos
    mov edi, r14d
%ifdef CHECK
    mov ebx, edi
    mov esi, IDS
.ck: lodsd
    push rsi
    mov edi, LG + 16
    mov byte [rdi], ' '
    push 10
    pop rcx
.dg: xor edx, edx
    div ecx
    add dl, '0'
    dec edi
    mov [rdi], dl
    test eax, eax
    jnz .dg
    mov edx, LG + 17
    sub edx, edi
    mov rsi, rdi
    push 2
    pop rdi
    push 1
    pop rax
    syscall
    pop rsi
    dec ebx
    jnz .ck
    mov edi, LG
    mov word [rdi], '| '
    mov rsi, rdi
    push 2
    pop rdx
    push 2
    pop rdi
    push 1
    pop rax
    syscall
    jmp exit
%endif
    push rdi

    ; ---- weights: bf16 -> fp32
    push 3                ; model.safetensors on fd 3
    pop rdi
    call mapf
    lodsq
    add rsi, rax
    mov edi, F
    mov ecx, 134515008
.cv: lodsw
    shl eax, 16
    stosd
    loop .cv
    fld1                     ; stays under everything: mv divides by st1 (1, or a softmax sum)

    ; ---- generate
    pop rax
    push rax                 ; [rsp+8] = prompt tokens, [rsp+16] = vocab
    add eax, 99
    push rax                 ; [rsp] = last n: 100 new tokens
    xor r12d, r12d
.gen: mov eax, [IDS + r12*4]
.g2: push rax
    call puttok              ; the prompt tokens spell the prompt
    pop rax
    inc r12d                 ; r12 = n = pos + 1
    cmp r12d, [rsp]
    ja .gd
    call forward
    cmp r12d, [rsp+8]
    jb .gen
%ifdef DUMP
    push 1
    pop rax
    push 2
    pop rdi
    mov esi, LG
    mov edx, V*4
    syscall
%endif
    mov esi, LG
    mov ebx, 0x80000000
    mov ch, 192             ; forward() leaves RCX zero
.am: lodsd                   ; float -> int with the same order
    cdq
    shr edx, 1
    xor eax, edx
    cmp eax, ebx
    jle .an
    xchg eax, ebx
    mov edi, ecx
.an: loop .am
    mov eax, V
    sub eax, edi
%ifdef TRACE
    ; Decimal generated token ID on stderr. Excluded from the measured build.
    push rax
    mov edi, LG + 16
    mov byte [rdi], 10
    mov ebx, 10
.td: xor edx, edx
    div ebx
    add dl, '0'
    dec edi
    mov [rdi], dl
    test eax, eax
    jnz .td
    mov edx, LG + 17
    sub edx, edi
    mov rsi, rdi
    mov edi, 2
    mov eax, 1
    syscall
    pop rax
%endif
    jmp .g2
.gd:
exit: push 60
    pop rax
    xor edi, edi
    syscall

; ---- forward(eax = token, r12 = n) -> LG
forward:
    imul esi, eax, 2304
    bts esi, 27              ; + F
    mov edi, X
    mov cx, 576
    rep movsd
    push 4
    pop r8
    xor r13d, r13d
    mov r15d, C
    imul r10d, r12d, 6144
    add r10, r15             ; r10 = this position's cache row
.layer:
    mov eax, r13d            ; rank of the layer in name order: 0 1 10..19 2 20..29 3..9
    cmp al, 10
    jb .one
    sub al, 8
    cmp al, 12
    sbb al, -1
    jmp .rk
.one: cmp al, 2
    jb .rk
    add al, 10
    cmp al, 13
    jb .rk
    add al, 10
.rk: imul eax, eax, 14160384
    lea r14, [rax + FL]
    mov rdx, r14                ; input norm
    mov rdi, r10
    lea rsi, [r14 + 10621440]   ; k, o, q, v
    mov ch, 6
    call mvx
    mov rdi, r10
    call rope
    mov r9w, 6144
    lea r11, [r10 + 3072]
    push r11
    xor ebp, ebp
.head:
    imul eax, ebp, 86        ; kv head * 256
    xor al, al
    lea rsi, [r15 + rax + 6144]
    push rsi
    mov edi, SC
    mov rbx, r11
    mov ecx, r12d
    push 64
    pop rdx
    call mv                  ; scores over the cache rows
    mov ecx, r12d
    fldz
.sm: sub edi, 4
    fld dword [rdi]
    fmul dword [l2e8]
    call exp2
    fst dword [rdi]
    faddp st1
    loop .sm
    mov ebx, edi
    xchg r8, r9
    pop rsi
    add esi, 5376
    mov rdi, r11
    mov cl, 64
    mov edx, r12d
    call mv                  ; weighted v / sum, down the cache rows; overwrites q
    fstp st0
    xchg r8, r9
    mov r11, rdi
    inc ebp
    cmp ebp, 9
    jb .head
    lea rsi, [r14 + 11063808]   ; o
    pop rbx
    mov dx, 576
    call mvX
    lea rdx, [r14 + 10619136]   ; post norm
    mov edi, HB
    lea rsi, [r14 + 3541248]    ; gate, up
    mov ch, 12
    call mvx
    mov esi, HB
    mov ch, 6
.sl: fld dword [rsi]
    fld st0
    fchs
    fldl2e
    fmulp st1
    call exp2
    fld1
    faddp st1
    fdivp st1
    fmul dword [rsi + 6144]
    fstp dword [rsi]
    lodsd
    loop .sl
    lea rsi, [r14 + 2304]       ; down
    mov ebx, HB
    mov dx, 1536
    call mvX
    add r15, CSTEP
    add r10, CSTEP
    inc r13d
    cmp r13d, 30
    jb .layer
    mov edx, FL + 30*14160384   ; final norm
    mov edi, LG
    mov esi, F
    mov ch, 192
mvx: push rdi
    push rcx
    call rms
    pop rcx
    pop rdi
    mov ebx, XB
    mov edx, 576
    jmp mvn
mvX: mov edi, X
    mov cx, 576
mvn: imul r9d, edx, 4
; out[r] (+)= W[r] . x   rdi out, rsi W, rbx x, ecx rows, edx cols, r8 column step, r9 row step
mv:
.r: push rsi
    fldz
    test edi, edi
    jns .z
    fadd dword [rdi]
.z: xor eax, eax
.c: fld dword [rsi]
    add rsi, r8
    fmul dword [rbx+rax*4]
    faddp st1
    inc eax
    cmp eax, edx
    jb .c
    pop rsi
    add rsi, r9
    fdiv st0, st1
    fstp dword [rdi]
    scasd
    loop .r
    ret

; xb = w * x / rms(x)   rdx = w
rms:
    mov ebx, X
    fldz
    mov cx, 576
.s: fld dword [rbx+rcx*4-4]
    fmul st0, st0
    faddp st1
    loop .s
    fadd dword [eps]         ; 1/sqrt(ss/576 + 1e-5) = 24/sqrt(ss + 0.00576)
    fsqrt
    fidivr dword [c24]
    mov cx, 576
    mov edi, XB
.m: fld dword [rbx+rcx*4-4]
    fmul st0, st1
    fmul dword [rdx+rcx*4-4]
    fstp dword [rdi+rcx*4-4]
    loop .m
    fstp st0
    ret

; rotate the 21 heads at rdi (k, unused o, q) by position r12-1
rope:
    lea eax, [r12-1]
    push rax
    push 21
    pop rdx
    sub edi, -64
.h: fld1                     ; f
    mov cl, 32
.i: fld st0
    fimul dword [rsp]
    fsincos                  ; c s f r
    fld dword [rdi-64]       ; A c s
    fld dword [rdi+64]       ; B A c s
    fld st1
    fmul st0, st3            ; Ac B A c s
    fld st1
    fmul st0, st5            ; Bs Ac B A c s
    fsubp st1                ; Ac-Bs B A c s
    fstp dword [rdi-64]
    fmul st0, st2            ; Bc A c s
    fxch
    fmul st0, st3            ; As Bc c s
    faddp st1
    fstp dword [rdi+64]
    fcompp
    fmul qword [r532]        ; next frequency
    scasd
    loop .i
    fstp st0
    sub edi, -128
    dec edx
    jnz .h
    pop rax
    ret

; st0 = 2^st0
exp2:
    fld st0
    frndint
    fsub st1, st0
    fxch
    f2xm1
    fld1
    faddp st1
    fscale
    fstp st1
    ret

; decode the next vocab entry: rsi in tokenizer.json -> bytes at rdi..
dec:
.q: lodsb
    cmp al, '"'
    jne .q
.ch: lodsb
    cmp al, '"'
    je .r
    cmp al, '\'
    jne .nb
    lodsb
.nb: test al, al
    jns .st
    and eax, 31              ; 2-byte UTF-8 -> code point
    shl eax, 6
    xchg edx, eax
    lodsb
    and eax, 63
    or eax, edx
    cmp eax, 289             ; undo GPT-2's byte alphabet (low byte of c is c-256)
    jb .st
    add al, 94
    cmp al, 0xA1
    jne .st
    mov al, 173
.st: stosb
    jmp .ch
.r: ret

; print vocab entry eax
puttok:
    lea ecx, [rax+1]
    mov rsi, [rsp+32]
.k: mov edi, TMP
    call dec
    loop .k
    mov edx, edi
    mov esi, TMP
    sub edx, esi
    jmp write

; rsi = mmap(file at edi)
mapf:
    push rdi
    pop r8
    xor edi, edi
    mov esi, 0x40000000
    push 1
    pop rdx
    mov r10d, edx            ; MAP_SHARED
    push 9
    pop rax
    syscall
    xchg rsi, rax
    ret

; break before byte j, from classes (j-1, j, j+1) packed as 2-bit fields
%assign BM 0
%assign w 0
%rep 64
  %assign p w >> 4
  %assign c (w >> 2) & 3
  %assign n w & 3
  %assign b 0
  %if p == 2 || c == 2
    %assign b 1
  %elif p == 0 || p == 3
    %if c == 0 || c == 3
      %if n == 1
        %assign b 1
      %endif
    %elif p == 3
      %assign b 1
    %endif
  %endif
  %assign BM BM | (b << w)
  %assign w w + 1
%endrep
BMASK equ BM


filesz equ $ - $$
