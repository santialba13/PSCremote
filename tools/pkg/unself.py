"""Convierte un fake SELF (eboot.bin / .prx de homebrew) a ELF. Uso: python3 unself.py in out.elf"""
import sys,struct,zlib
d=open(sys.argv[1],'rb').read()
n,=struct.unpack_from('<H',d,0x18)
ents=[struct.unpack_from('<QQQQ',d,0x20+i*32) for i in range(n)]
eo=0x20+n*32
e_phoff,=struct.unpack_from('<Q',d,eo+0x20); e_phnum,=struct.unpack_from('<H',d,eo+0x38); e_phentsize,=struct.unpack_from('<H',d,eo+0x36)
phs=[struct.unpack_from('<IIQQQQQQ',d,eo+e_phoff+i*e_phentsize) for i in range(e_phnum)]
size=max(p[2]+p[5] for p in phs)
out=bytearray(size+0x1000)
hdrlen=e_phoff+e_phnum*e_phentsize
out[:hdrlen]=d[eo:eo+hdrlen]
for props,off,fsz,msz in ents:
    if not props&0x800: continue
    idx=(props>>20)&0xfff
    if props&0x2: print('encrypted seg',idx); continue
    data=d[off:off+fsz]
    if props&0x8: data=zlib.decompress(data)
    p=phs[idx]; out[p[2]:p[2]+len(data)]=data
for p in phs: print('phdr type=%x off=%x va=%x fsz=%x'%(p[0],p[2],p[3],p[5]))
open(sys.argv[2],'wb').write(bytes(out[:size]))
