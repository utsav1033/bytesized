#!/usr/bin/env python3
"""Build a byte-LZ self-unpacking ELF from a descriptor-input x86 assembly source.

Usage: python3 pack_lz_fd.py SOURCE.asm OUTPUT [CHECK|TRACE] [--base 0x93] [--length-codes 4] [--zero-max 3]
Requires Python 3 and NASM on Linux x86-64.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source',type=Path)
p.add_argument('output',type=Path)
p.add_argument('mode',nargs='?',default='')
p.add_argument('--base',type=lambda v:int(v,0),default=0x93)
p.add_argument('--length-codes',type=int,default=4)
p.add_argument('--zero-max',type=int,default=3)
a=p.parse_args()
assert a.mode in ('','CHECK','TRACE')
assert 1<=a.length_codes<=32 and 2<=a.zero_max<=16 and a.base+2*a.length_codes+a.zero_max-1<256
source=a.source.resolve();output=a.output.resolve()
src=source.read_text()
header_start=src.find('\n%ifdef DBG\nehdr:\n')
if header_start<0:header_start=src.index('\nehdr:\n')
pre=src[:header_start]
pre,n=re.subn(r'ORG 0x[0-9A-Fa-f]+','''ORG 0x200D8
write equ $$+(0x10008-0x200D8)
r532 equ $$+(0x10028-0x200D8)
l2e8 equ $$+(0x10058-0x200D8)
eps equ $$+(0x10048-0x200D8)
c24 equ $$+(0x1004c-0x200D8)''',pre,count=1)
assert n==1
body=src[src.index('\n_start:\n')+1:]

image_header='''ehdr:
 db 0x7f,"ELF",2,1,1,0
 dq 0
 dw 2,62
 dd 1
 dq _start
 dq phdr - $$
 dq 0
 dd 0
 dw 64,56
phdr:
 dd 1,7
 dq 0, $$, $$
 dq filesz, MEND - $$
 dq 0x1000
'''
x_match=re.search(r'^X\s+equ\s+(0x[0-9a-fA-F]+|[0-9]+)\b',src,re.M)
assert x_match, 'X equ address not found'
high_residual=int(int(x_match.group(1),0)>=0x80000000)
base=a.base;k=a.length_codes;esc=base+2*k;zmax=esc+a.zero_max-1

# Match token: one byte for (length-3, offset high bit), one byte for
# offset low bits. Reserved literal bytes use an escape and a shifted byte.
def encode(data):
    n=len(data)
    matches=[[] for _ in range(n)]
    for i in range(n):
        for off in range(1,min(i,511)+1):
            length=0
            while i+length<n and length<k+2 and data[i+length]==data[i+length-off]:
                length+=1
            if length>=3:matches[i].append((off,length))
    cost=[10**9]*(n+1);cost[n]=0;choice=[None]*n
    for i in range(n-1,-1,-1):
        best=(2 if base<=data[i]<=zmax else 1)+cost[i+1];ch=None
        for run in range(2,a.zero_max+1):
            if data[i:i+run]==bytes(run) and 1+cost[i+run]<best:
                best=1+cost[i+run];ch=('zero',run)
        for off,L in matches[i]:
            for length in range(3,L+1):
                c=2+cost[i+length]
                if c<best:best=c;ch=(off,length)
        cost[i]=best;choice[i]=ch
    packed=bytearray();i=0
    while i<n:
        ch=choice[i]
        if ch:
            if ch[0]=='zero':
                packed.append(esc+ch[1]-1);i+=ch[1];continue
            off,length=ch
            packed.extend((base+2*(length-3)+(off>>8),off&255));i+=length
        else:
            b=data[i]
            if base<=b<=zmax:packed.extend((esc,(b-base)&255))
            else:packed.append(b)
            i+=1
    assert len(packed)==cost[0]
    return bytes(packed)

def decode(packed):
    out=bytearray();j=0
    while j<len(packed):
        b=packed[j];j+=1
        if b==esc:out.append((packed[j]+base)&255);j+=1
        elif esc<b<=zmax:out.extend(bytes(b-esc+1))
        elif base<=b<esc:
            off=packed[j] + ((b-base)&1)*256;j+=1
            for _ in range((b-base)//2+3):out.append(out[-off])
        else:out.append(b)
    return bytes(out)

with tempfile.TemporaryDirectory(prefix='bytesized-lz-') as temp:
    d=Path(temp)
    subprocess.run(['nasm','-f','bin','-o',str(d/'main.bin'),str(source)],check=True)
    main=(d/'main.bin').read_bytes()
    assert main[8:16]==bytes.fromhex('6a015889c70f05c3')
    (d/'image.asm').write_text(pre+'\n'+image_header+body)
    cmd=['nasm','-f','bin']
    if a.mode:cmd.append('-D'+a.mode)
    subprocess.run(cmd+['-o',str(d/'image.bin'),str(d/'image.asm')],check=True)
    image=(d/'image.bin').read_bytes()
    assert image[:4]==b'\x7fELF' and len(image)>112
    data=image[112:]
    packed=encode(data)
    assert decode(packed)==data
    (d/'payload.bin').write_bytes(packed)
    stub='''BITS 64
ORG 0x10000
ehdr:
 db 0x7f,"ELF",2,1,1,0
write:
 push 1
 pop rax
 mov edi,eax
 syscall
 ret
 dw 2,62
 dd 1
 dq start
 dq phdr-$$
r532: dq 0.6978305848598664
phdr: dd 1
 dd 0x00380047
 dq 1
 dq $$+1
eps: dd 0.00576
c24: dd 24
 dq filesz-1
l2e8: dd 0.18033688011112042
 dd HIGH_BSS
start:
 mov esi,payload
 lea edi,[rsi+rsi]
 xor eax,eax
.loop:
 lodsb
 sub al,BASE
 cmp al,2*LENGTH_CODES
 jb .match
 je .escape
 cmp al,2*LENGTH_CODES+ZERO_MAX
 jae .literal
.zeros:
 sub al,2*LENGTH_CODES-1
 xchg eax,ecx
 xor eax,eax
 rep stosb
 jmp .next
.escape:
 lodsb
.literal:
 add al,BASE
 stosb
 jmp .next
.match:
 shr al,1
 lea ecx,[rax+3]
 lodsb
 rcl ah,1
 push rsi
 mov esi,edi
 sub esi,eax
 xor ah,ah              ; preserve zero high byte for next token length
 rep movsb
 pop rsi
.next:
 cmp di,(0x20148+OUTLEN)&0xffff
 jb .loop
 jmp 0x20148
payload: incbin "payload.bin"
filesz equ $-$$
'''
    stub=stub.replace('HIGH_BSS',str(int(high_residual))).replace('BASE',str(base)).replace('LENGTH_CODES',str(k)).replace('OUTLEN',str(len(data))).replace('ZERO_MAX',str(a.zero_max))
    (d/'stub.asm').write_text(stub)
    subprocess.run(['nasm','-f','bin','-o',str(output),'stub.asm'],check=True,cwd=d)
    output.chmod(0o755)
    payload_offset = output.stat().st_size - len(packed)
    assert 0x200D8 + 112 == 0x20148 == 2 * (0x10000 + payload_offset), (
        'relocated body and twice-payload destination diverged', payload_offset)
    print(f'source={len(main)} body={len(data)} payload={len(packed)} packed={output.stat().st_size} base={base:#x} lengths={k} zero_max={a.zero_max}')
