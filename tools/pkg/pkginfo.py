"""Muestra la tabla de entradas de un PKG de PS4. Uso: python3 pkginfo.py archivo.pkg"""
import struct,sys
d=open(sys.argv[1],'rb').read()
be=lambda o,n:int.from_bytes(d[o:o+n],'big')
cnt=be(0x10,4); tbl=be(0x18,4)
print('entries',cnt,'table',hex(tbl))
print('body off',hex(be(0x20,8)),'size',hex(be(0x28,8)))
print('pfs off',hex(be(0x410,8)),'size',hex(be(0x418,8)))
names_off=None
ents=[]
for i in range(cnt):
    o=tbl+i*32
    eid,no,fl1,fl2,off,sz=struct.unpack('>IIIIII',d[o:o+24])
    ents.append((eid,no,fl1,fl2,off,sz))
    if eid==0x200: names_off=off
for eid,no,fl1,fl2,off,sz in ents:
    nm=''
    if names_off and no: nm=d[names_off+no:d.index(b'\0',names_off+no)].decode()
    print(hex(eid),nm,hex(fl1),hex(fl2),hex(off),sz)
