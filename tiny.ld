/* one RWX segment that also maps the ELF + program headers; nothing else in the file */
PHDRS { all PT_LOAD FILEHDR PHDRS FLAGS(7); }
SECTIONS {
  . = 0x400000 + SIZEOF_HEADERS;
  .text : { *(.text*) *(.rodata*) *(.data*) } :all
  .bss : { *(.bss*) *(COMMON) } :all
  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) *(.ARM*) }
}
