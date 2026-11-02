# sstrip.py FILE: drop everything the kernel doesn't read (section headers + anything past the last segment).
import struct, sys
b = bytearray(open(sys.argv[1], "rb").read())
phoff, = struct.unpack_from("<Q", b, 0x20)
phnum, = struct.unpack_from("<H", b, 0x38)
end = 0x40 + phnum * 56
for i in range(phnum):
    off, filesz = struct.unpack_from("<Q", b, phoff + i * 56 + 8)[0], struct.unpack_from("<Q", b, phoff + i * 56 + 32)[0]
    end = max(end, off + filesz)
struct.pack_into("<Q", b, 0x28, 0)          # e_shoff = 0
struct.pack_into("<HHH", b, 0x3a, 0, 0, 0)  # e_shentsize, e_shnum, e_shstrndx = 0
open(sys.argv[1], "wb").write(b[:end])
