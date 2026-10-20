"""Descifra el PFS de un fPKG de PS4 (passcode por defecto) y extrae sus archivos.

Uso: python3 pfs_extract.py juego.pkg carpeta_salida
Basado en el formato documentado por LibOrbisPkg (maxton/LibOrbisPkg, LGPL).
"""
import sys,os,struct,hashlib,hmac,zlib
from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
pkgp,out=sys.argv[1],sys.argv[2]
d=open(pkgp,'rb').read()
cid=d[0x40:0x64].decode()
S=lambda b:hashlib.sha256(b).digest()
ekpfs=S(S((1).to_bytes(4,'big'))+S(cid.encode().ljust(48,b'\0'))+b'0'*32)
po=int.from_bytes(d[0x410:0x418],'big'); ps=int.from_bytes(d[0x418:0x420],'big')
pfs_flags=int.from_bytes(d[0x420:0x428],'big')
outer=bytearray(d[po:po+ps])
def hdr(b): 
    v,m=struct.unpack_from('<qq',b,0); mode,=struct.unpack_from('<H',b,0x1c); bs,=struct.unpack_from('<I',b,0x20)
    dc,nd,dbc=struct.unpack_from('<qqq',b,0x30); return dict(v=v,m=m,mode=mode,bs=bs,dc=dc,dbc=dbc,seed=bytes(b[0x370:0x380]))
def decrypt(b,h,newc):
    k=hmac.new(hmac.new(ekpfs,h['seed'],'sha256').digest() if newc else ekpfs,(1).to_bytes(4,'little')+h['seed'],'sha256').digest()
    tweak,data=k[:16],k[16:]
    r=bytearray(b); start=h['bs']//0x1000
    for s in range(start,len(b)//0x1000):
        dec=Cipher(algorithms.AES(data+tweak),modes.XTS(s.to_bytes(16,'little'))).decryptor()
        r[s*0x1000:(s+1)*0x1000]=dec.update(bytes(b[s*0x1000:(s+1)*0x1000]))+dec.finalize()
    return r
def parse(img):
    h=hdr(img); assert h['v']==1 and h['m']==20130315,h
    signed=h['mode']&1; isz=0x2C8 if signed else 0xA8
    inodes=[];per=h['bs']//isz
    for i in range(h['dbc']):
        base=h['bs']*(1+i)
        for j in range(per):
            if len(inodes)>=h['dc']: break
            o=base+j*isz
            mode,nl,fl,size,csize=struct.unpack_from('<HHIqq',img,o)
            blocks,=struct.unpack_from('<I',img,o+0x60)
            if signed: sb,=struct.unpack_from('<i',img,o+0x64+32)
            else: sb,=struct.unpack_from('<i',img,o+0x64)
            inodes.append(dict(size=size,csize=csize,blocks=blocks,sb=sb,fl=fl))
    files={}
    def ld(ino,path):
        n=inodes[ino]
        for x in range(n['sb'],n['sb']+n['blocks']):
            p=x*h['bs'];end=p+h['bs']
            while p<end:
                inum,typ,nlen,es=struct.unpack_from('<IiiI',img,p)
                if es==0: break
                name=img[p+16:p+16+nlen].decode()
                if typ==2: files[path+'/'+name]=inodes[inum]
                elif typ==3: ld(inum,path+'/'+name)
                p+=es
    ld(0,''); return h,files
h=hdr(outer); print('outer',{k:v for k,v in h.items() if k!='seed'},hex(pfs_flags))
if h['mode']&4:
    for newc in (False,True):
        try: img=decrypt(outer,h,newc); hh,files=parse(img); print('newcrypt',newc); break
        except Exception as e: print('fail',newc,e)
else: hh,files=parse(outer); img=outer
for k,v in files.items(): print(k,v)
f=files['/uroot/pfs_image.dat']; pf=img[f['sb']*hh['bs']:f['sb']*hh['bs']+f['size']]
mg,u4,u8,bsz,bsz2,boff,dst,dlen=struct.unpack_from('<iiiiqqQq',pf,0)
nb=dlen//bsz2; sm=struct.unpack_from('<%dq'%(nb+1),pf,boff)
inner=bytearray()
for i in range(nb):
    a,b=sm[i],sm[i+1]
    if b-a==bsz2: inner+=pf[a:a+bsz]
    elif b-a>bsz2: inner+=bytes(bsz)
    else: inner+=zlib.decompress(bytes(pf[a+2:b]),-15).ljust(bsz,b'\0')
ih,ifiles=parse(inner)
for k,v in ifiles.items():
    if not k.startswith('/uroot'): continue
    p=os.path.join(out,k[len('/uroot/'):]); os.makedirs(os.path.dirname(p),exist_ok=True)
    if v['size']!=v['csize']: print('COMPRESSED',k)
    open(p,'wb').write(inner[v['sb']*ih['bs']:v['sb']*ih['bs']+v['size']]); print(k,v['size'])
