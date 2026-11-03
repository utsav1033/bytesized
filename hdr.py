# hdr.py FILE: overlap the program header with the last 8 bytes of the ELF header.
# Those 8 bytes are e_phnum=1, e_shentsize=0 (= p_type 1) and e_shnum=7, e_shstrndx=0 (= p_flags 7);
# the kernel never reads the section header fields. The code after the headers is position-independent.
import struct, sys
b = bytearray(open(sys.argv[1], "rb").read())
assert struct.unpack_from("<Q", b, 0x20)[0] == 0x40 and struct.unpack_from("<II", b, 0x40) == (1, 7)
b = b[:0x38] + b[0x40:]
struct.pack_into("<Q", b, 0x20, 0x38)                                   # e_phoff
struct.pack_into("<Q", b, 0x18, struct.unpack_from("<Q", b, 0x18)[0] - 8)   # e_entry
struct.pack_into("<Q", b, 0x38 + 32, struct.unpack_from("<Q", b, 0x38 + 32)[0] - 8)   # p_filesz
open(sys.argv[1], "wb").write(b)
