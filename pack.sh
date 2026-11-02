#!/bin/sh
# pack.sh SRC OUT [extra flags]: build SRC as a flat image at 0x500000, LZ-pack it, wrap it in stub.c.
set -e
D=$(dirname "$0"); SRC=$1; OUT=$2; shift 2; T=$(mktemp -d)
CF="--target=aarch64-linux-gnu -Oz -mno-outline -ffreestanding -fno-builtin -nostdlib -static -fno-asynchronous-unwind-tables
 -fno-unwind-tables -fno-stack-protector -fno-math-errno -ffunction-sections -fdata-sections -fno-pic -fno-pie -fno-ident
 -fuse-ld=lld -Wl,--gc-sections -Wl,--build-id=none -Wl,-z,norelro -Wl,-N"
cat > $T/img.ld <<LD
SECTIONS {
  . = 0x500000;
  .text : { *(.text*) *(.rodata*) *(.data*) }
  .bss : { *(.bss*) *(COMMON) }
  _end = .;
  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) *(.ARM*) }
}
LD
clang $CF -Wl,-T,$T/img.ld "$@" -o $T/img.elf "$SRC"
objcopy -O binary -j .text $T/img.elf $T/img.bin
ENTRY=0x$(nm $T/img.elf | awk '$3=="_start"{print $1}')
END=0x$(nm $T/img.elf | awk '$3=="_end"{print $1}')
SIZE=$(stat -c %s $T/img.bin)
python3 "$D/${CODER:-rcopt}.py" $T/img.bin $T/payload.bin
cat > $T/stub.ld <<LD
PHDRS { all PT_LOAD FILEHDR PHDRS FLAGS(7); }
SECTIONS {
  . = 0x400000 + SIZEOF_HEADERS;
  .text : { *(.text*) } :all
  .bss (NOLOAD) : { . += $END - ABSOLUTE(.); } :all
}
LD
clang --target=aarch64-linux-gnu -nostdlib -static -fuse-ld=lld -Wl,--build-id=none -Wl,-N -Wl,-T,$T/stub.ld \
  -DOUT=0x500000 -DSIZE=$SIZE -DENTRY=$ENTRY -DPAYLOAD="\"$T/payload.bin\"" -o "$OUT" "$D/${STUB:-stubrc}.S"
strip -s "$OUT"; python3 "$D/sstrip.py" "$OUT"
echo "image $SIZE bytes, entry $ENTRY, end $END -> $(stat -c %s "$OUT") bytes"
[ -n "$KEEP" ] && echo $T || rm -rf $T
