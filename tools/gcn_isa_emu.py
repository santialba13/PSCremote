#!/usr/bin/env python3
"""Emulador de la GPU de la PS4 a nivel de instrucciones (ISA GCN gfx701, "Liverpool").

gpu_emu (plugin/test) ejecuta el DCB del plugin con los kernels .cl recompilados
para x86. Esto va un paso mas cerca de la consola: ejecuta el CODIGO MAQUINA GCN
que el plugin copia a la memoria de la GPU (nx_shaders.h), instruccion por
instruccion, con wavefronts de 64 lanes, EXEC/VCC/SCC, SGPR/VGPR, LDS, descriptores
T#/S#/V# leidos de la memoria, y el mismo DCB PM4 (SET_SH_REG, DISPATCH_DIRECT,
EVENT_WRITE_EOP) que arma nx_pipeline_build().

Como LLVM 18 no desensambla gfx7, el codigo se obtiene al reves: los .s que genera
shaders/build_gcn.sh se ensamblan con llvm-mc (-show-encoding) y se verifica que los
bytes sean IDENTICOS a los que hay en la memoria de la GPU en la direccion de cada
COMPUTE_PGM; recien ahi se ejecutan esas instrucciones.

  NX_DUMP=job.bin plugin/test/gpu_emu W H f0.rgba f1.rgba mid.rgba [modo] [real.rgba]
  python3 tools/gcn_isa_emu.py job.bin --out-mid mid_isa.rgba [--out-real real_isa.rgba]

Lo que no modela (y por que no importa aca): tiempos/caches (se ejecuta en orden,
cada dispatch completo antes del siguiente, como garantizan las barreras del DCB),
el tiling de los buffers del juego (los de prueba son lineales, como en gpu_emu) y
la precision de 8 bits de los pesos del filtro bilineal del hardware (se usan floats).
"""
import argparse
import os
import re
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "shaders", "build")
SOURCES = ["nxflow", "fsr1", "smaa", "nxsr", "nxsr_hq", "osd"]
NL = 64  # lanes por wavefront

f32 = np.float32
u32 = np.uint32
i32 = np.int32


# --------------------------------------------------------------------- ensamblado
class Inst:
    __slots__ = ("op", "ops", "mods", "size", "addr", "text", "target", "enc")

    def __init__(self, op, ops, mods, size, text):
        self.op, self.ops, self.mods, self.size, self.text = op, ops, mods, size, text
        self.addr = 0
        self.target = None


def _split_ops(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


MOD_RE = re.compile(r"\b(offen|idxen|addr64|glc|slc|tfe|unorm|da|r128|lwe|clamp|gds|"
                    r"offset:\S+|offset0:\S+|offset1:\S+|dmask:\S+|mul:\S+|div:\S+)\b")


def assemble(name_src):
    """-> {kernel: (bytes, [Inst])} para cada funcion del .s"""
    path = os.path.join(BUILD, name_src + ".s")
    txt = subprocess.run(["llvm-mc", "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx701", "-show-encoding", path],
                         capture_output=True, text=True, check=True).stdout
    funcs, cur, labels = {}, None, {}
    for line in txt.splitlines():
        m = re.match(r"^([.A-Za-z_][\w.]*):", line)
        if m:
            lab = m.group(1)
            if lab.startswith(".L"):
                if cur is not None:
                    labels[lab] = (cur, len(funcs[cur]))
            elif not lab.startswith("."):
                cur = lab
                funcs[cur] = []
            continue
        if cur is None or "; encoding:" not in line:
            continue
        code, enc = line.split("; encoding:")
        enc = enc.strip()[1:-1].split(",")
        code = code.strip()
        parts = code.split(None, 1)
        op = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        mods = {}
        for mm in MOD_RE.findall(rest):
            if ":" in mm:
                k, v = mm.split(":", 1)
                mods[k] = v
            else:
                mods[mm] = True
        rest = MOD_RE.sub("", rest).strip()
        ops = _split_ops(rest) if rest else []
        ins = Inst(op, ops, mods, len(enc), code)
        ins.enc = enc
        funcs[cur].append(ins)
    progs = {}
    for fn, ins in funcs.items():
        a = 0
        for i in ins:
            i.addr = a
            a += i.size
        for i in ins:
            if i.op.startswith("s_cbranch") or i.op == "s_branch":
                lab = i.ops[0]
                _, idx = labels[lab]
                i.target = idx
        # bytes con los saltos resueltos (simm16 en dwords relativo a la siguiente)
        blob = bytearray()
        for i in ins:
            enc = list(i.enc)
            if i.target is not None:
                t_addr = ins[i.target].addr if i.target < len(ins) else a
                off = (t_addr - (i.addr + 4)) // 4
                lo, hi = off & 0xFF, (off >> 8) & 0xFF
                enc[0], enc[1] = hex(lo), hex(hi)
            blob += bytes(int(x, 16) for x in enc)
        progs[fn] = (bytes(blob), ins)
    return progs


def load_programs():
    progs = {}
    for s in SOURCES:
        progs.update(assemble(s))
    return progs


# ------------------------------------------------------------------- memoria
class Memory:
    def __init__(self):
        self.regions = []  # (base, ndarray uint8)

    def add(self, base, arr):
        self.regions.append((base, arr))

    def find(self, addr, n=1):
        for b, a in self.regions:
            if b <= addr and addr + n <= b + a.size:
                return a, addr - b
        raise RuntimeError("direccion GPU sin mapear 0x%x (+%d)" % (addr, n))

    def read(self, addr, n):
        a, o = self.find(addr, n)
        return a[o:o + n]

    def rd32(self, addr, n=1):
        return self.read(addr, 4 * n).view(np.uint32).copy()

    def gather(self, addrs, nbytes):
        """addrs: array uint64 por lane -> uint8 (lanes, nbytes)."""
        out = np.zeros((len(addrs), nbytes), np.uint8)
        for k, ad in enumerate(addrs):
            out[k] = self.read(int(ad), nbytes)
        return out

    def scatter(self, addrs, data):
        for k, ad in enumerate(addrs):
            a, o = self.find(int(ad), data.shape[1])
            a[o:o + data.shape[1]] = data[k]


# --------------------------------------------------------- descriptores de imagen
class Image:
    def __init__(self, mem, t):
        t = [int(x) for x in t]
        self.base = ((t[0] | (t[1] & 0xFF) << 32) << 8)
        self.dfmt = (t[1] >> 20) & 0x3F
        self.nfmt = (t[1] >> 26) & 0xF
        self.w = (t[2] & 0x3FFF) + 1
        self.h = ((t[2] >> 14) & 0x3FFF) + 1
        self.sel = [(t[3] >> s) & 7 for s in (0, 3, 6, 9)]
        self.tile = (t[3] >> 20) & 0x1F
        self.pitch = ((t[4] >> 13) & 0x3FFF) + 1
        if self.dfmt != 10 or self.nfmt != 0:
            raise RuntimeError("formato de imagen no soportado (dfmt %d nfmt %d)" % (self.dfmt, self.nfmt))
        if self.tile != 8:
            raise RuntimeError("tiling %d no soportado (los buffers de prueba son lineales)" % self.tile)
        arr, off = mem.find(self.base, self.pitch * self.h * 4)
        self.pix = arr[off:off + self.pitch * self.h * 4].reshape(self.h, self.pitch, 4)

    def fetch(self, x, y):
        """x, y int arrays -> (n,4) float32 con swizzle (clamp al borde)."""
        x = np.clip(x, 0, self.w - 1)
        y = np.clip(y, 0, self.h - 1)
        raw = self.pix[y, x].astype(f32) * f32(1.0 / 255.0)
        out = np.zeros((len(x), 4), f32)
        for c, s in enumerate(self.sel):
            out[:, c] = 0.0 if s == 0 else 1.0 if s == 1 else raw[:, s - 4]
        return out

    def store(self, x, y, v):
        # inversa del swizzle (los nuestros son identidad)
        inv = np.zeros_like(v)
        for c, s in enumerate(self.sel):
            if s >= 4:
                inv[:, s - 4] = v[:, c]
        q = np.clip(np.floor(np.clip(inv, 0.0, 1.0) * f32(255.0) + f32(0.5)), 0, 255).astype(np.uint8)
        ok = (x >= 0) & (x < self.w) & (y >= 0) & (y < self.h)
        self.pix[y[ok], x[ok]] = q[ok]


def sampler_bilinear(s):
    s = [int(x) for x in s]
    unnorm = (s[0] >> 15) & 1
    mag = (s[2] >> 20) & 3
    return unnorm, mag


# --------------------------------------------------------------------- wavefront
INLINE_F = {"0.5": 0x3F000000, "-0.5": 0xBF000000, "1.0": 0x3F800000, "-1.0": 0xBF800000,
            "2.0": 0x40000000, "-2.0": 0xC0000000, "4.0": 0x40800000, "-4.0": 0xC0800000}


def fbits(a):
    return np.asarray(a, f32).view(u32)


def bitsf(a):
    return np.asarray(a, u32).view(f32)


class Wave:
    def __init__(self, emu, prog, sgpr_init, tid, lds):
        self.e = emu
        self.mem = emu.mem
        self.prog = prog
        self.s = np.zeros(112, np.uint64)  # SGPRs como enteros (se enmascaran a 32 bits)
        for k, v in enumerate(sgpr_init):
            self.s[k] = v & 0xFFFFFFFF
        self.v = np.zeros((256, NL), u32)
        self.v[0] = tid[0]
        self.v[1] = tid[1]
        self.exec = (1 << NL) - 1
        self.vcc = 0
        self.scc = 0
        self.m0 = 0xFFFFFFFF
        self.lds = lds

    # -- utilidades de mascaras
    def emask(self):
        return np.array([(self.exec >> k) & 1 for k in range(NL)], bool)

    @staticmethod
    def tomask(b):
        m = 0
        for k in np.nonzero(b)[0]:
            m |= 1 << int(k)
        return m

    @staticmethod
    def frommask(m):
        return np.array([(m >> k) & 1 for k in range(NL)], bool)

    # -- lectura de operandos
    def sreg64(self, o):
        if o in ("vcc",):
            return self.vcc
        if o == "exec":
            return self.exec
        m = re.match(r"s\[(\d+):(\d+)\]", o)
        if m:
            a = int(m.group(1))
            return int(self.s[a]) | (int(self.s[a + 1]) << 32)
        return self.const(o, 64)

    def set_sreg64(self, o, val):
        val &= (1 << 64) - 1
        if o == "vcc":
            self.vcc = val
        elif o == "exec":
            self.exec = val
        else:
            m = re.match(r"s\[(\d+):(\d+)\]", o)
            a = int(m.group(1))
            self.s[a] = val & 0xFFFFFFFF
            self.s[a + 1] = val >> 32

    def const(self, o, bits=32):
        if o in INLINE_F:
            return INLINE_F[o]
        if o.startswith("0x") or o.startswith("-0x"):
            v = int(o, 16)
        else:
            v = int(o)
        return v & ((1 << bits) - 1)

    def sread(self, o):
        """escalar de 32 bits (SALU o fuente uniforme)"""
        if re.fullmatch(r"s\d+", o):
            return int(self.s[int(o[1:])])
        if o == "vcc_lo":
            return self.vcc & 0xFFFFFFFF
        if o == "vcc_hi":
            return self.vcc >> 32
        if o == "exec_lo":
            return self.exec & 0xFFFFFFFF
        if o == "exec_hi":
            return self.exec >> 32
        if o == "m0":
            return self.m0
        if o == "vcc":
            return self.vcc & 0xFFFFFFFF
        return self.const(o)

    def swrite(self, o, val):
        val = int(val) & 0xFFFFFFFF
        if re.fullmatch(r"s\d+", o):
            self.s[int(o[1:])] = val
        elif o == "vcc_lo":
            self.vcc = (self.vcc & ~0xFFFFFFFF) | val
        elif o == "vcc_hi":
            self.vcc = (self.vcc & 0xFFFFFFFF) | (val << 32)
        elif o == "exec_lo":
            self.exec = (self.exec & ~0xFFFFFFFF) | val
        elif o == "exec_hi":
            self.exec = (self.exec & 0xFFFFFFFF) | (val << 32)
        elif o == "m0":
            self.m0 = val
        else:
            raise RuntimeError("destino escalar " + o)

    def vread(self, o):
        """fuente VALU de 32 bits por lane (uint32), con modificadores neg/abs aparte"""
        if re.fullmatch(r"v\d+", o):
            return self.v[int(o[1:])].copy()
        return np.full(NL, self.sread(o), u32)

    def vsrc_f(self, o):
        neg = absv = False
        oo = o
        if oo.startswith("-"):
            neg, oo = True, oo[1:]
        if oo.startswith("|") and oo.endswith("|"):
            absv, oo = True, oo[1:-1]
        if oo in INLINE_F or re.fullmatch(r"-?\d+\.\d+(e-?\d+)?", oo):
            val = np.full(NL, INLINE_F.get(oo, None) if oo in INLINE_F else fbits(f32(float(oo))), u32)
            x = bitsf(val).copy()
        elif neg and not absv and not re.match(r"[sv]\d|vcc|m0|exec", oo):
            # entero negativo literal (-1, -16...)
            return bitsf(self.vread(o))
        else:
            x = bitsf(self.vread(oo)).copy()
        if absv:
            x = np.abs(x)
        if neg:
            x = -x
        return x

    def vwrite(self, o, val, mask):
        r = int(o[1:])
        self.v[r][mask] = np.asarray(val).astype(u32)[mask] if np.asarray(val).dtype != u32 else np.asarray(val)[mask]

    def vwrite_f(self, o, val, mask, clamp=False):
        val = np.asarray(val, f32)
        if clamp:
            val = np.clip(val, f32(0.0), f32(1.0))
        self.vwrite(o, val.view(u32), mask)

    def vreg_range(self, o):
        m = re.match(r"v\[(\d+):(\d+)\]", o)
        if m:
            return list(range(int(m.group(1)), int(m.group(2)) + 1))
        return [int(o[1:])]

    def sreg_range(self, o):
        m = re.match(r"s\[(\d+):(\d+)\]", o)
        return list(range(int(m.group(1)), int(m.group(2)) + 1))

    # ----------------------------------------------------------------- ejecucion
    def run(self, max_steps=50_000_000):
        ins = self.prog
        pc = 0
        steps = 0
        while pc < len(ins):
            i = ins[pc]
            steps += 1
            if steps > max_steps:
                raise RuntimeError("demasiadas instrucciones (bucle infinito?)")
            nxt = self.step(i)
            if nxt == "end":
                return steps
            pc = nxt if nxt is not None else pc + 1
        return steps

    def step(self, i):
        op = i.op
        if op.startswith("s_"):
            return self.salu(i)
        if op.startswith("v_"):
            self.valu(i)
            return None
        if op.startswith("buffer_"):
            self.mubuf(i)
            return None
        if op.startswith("image_"):
            self.mimg(i)
            return None
        if op.startswith("ds_"):
            self.ds(i)
            return None
        raise RuntimeError("instruccion no soportada: " + i.text)

    # ---------------------------------------------------------------- SALU/SOPP
    def salu(self, i):
        op, o = i.op, i.ops
        if op == "s_endpgm":
            return "end"
        if op in ("s_waitcnt", "s_barrier", "s_nop", "s_setreg_imm32_b32", "s_setprio", "s_sleep"):
            return None
        if op == "s_branch":
            return i.target
        if op.startswith("s_cbranch"):
            cond = {"s_cbranch_scc0": self.scc == 0, "s_cbranch_scc1": self.scc == 1,
                    "s_cbranch_execz": self.exec == 0, "s_cbranch_execnz": self.exec != 0,
                    "s_cbranch_vccz": self.vcc == 0, "s_cbranch_vccnz": self.vcc != 0}[op]
            return i.target if cond else None
        if op.startswith("s_load_dword"):
            n = {"s_load_dword": 1, "s_load_dwordx2": 2, "s_load_dwordx4": 4, "s_load_dwordx8": 8,
                 "s_load_dwordx16": 16}[op]
            base = self.sreg64(o[1])
            off = self.sread(o[2]) * 4 if not o[2].startswith("s") else self.sread(o[2])
            if o[2].startswith("0x") or o[2].isdigit():
                off = self.const(o[2]) * 4
            vals = self.mem.rd32((base + off) & ((1 << 48) - 1), n)
            dst = self.sreg_range(o[0]) if o[0].startswith("s[") else [int(o[0][1:])]
            for k, r in enumerate(dst):
                self.s[r] = int(vals[k])
            return None
        b64 = op.endswith("_b64") or op.endswith("_u64")
        if op in ("s_and_saveexec_b64", "s_or_saveexec_b64", "s_andn2_saveexec_b64"):
            src = self.sreg64(o[1])
            old = self.exec
            self.set_sreg64(o[0], old)
            if op == "s_and_saveexec_b64":
                self.exec = src & old
            elif op == "s_or_saveexec_b64":
                self.exec = (src | old) & ((1 << 64) - 1)
            else:
                self.exec = src & ~old & ((1 << 64) - 1)
            self.scc = int(self.exec != 0)
            return None
        if b64:
            M = (1 << 64) - 1
            if op == "s_mov_b64":
                self.set_sreg64(o[0], self.sreg64(o[1]))
                return None
            if op in ("s_and_b64", "s_or_b64", "s_xor_b64", "s_andn2_b64", "s_orn2_b64"):
                a, b = self.sreg64(o[1]), self.sreg64(o[2])
                r = {"s_and_b64": a & b, "s_or_b64": a | b, "s_xor_b64": a ^ b,
                     "s_andn2_b64": a & ~b & M, "s_orn2_b64": (a | (~b & M))}[op]
                self.set_sreg64(o[0], r)
                self.scc = int(r != 0)
                return None
            if op == "s_not_b64":
                r = ~self.sreg64(o[1]) & M
                self.set_sreg64(o[0], r)
                self.scc = int(r != 0)
                return None
            if op == "s_lshl_b64":
                r = (self.sreg64(o[1]) << (self.sread(o[2]) & 63)) & M
                self.set_sreg64(o[0], r)
                self.scc = int(r != 0)
                return None
            if op == "s_cselect_b64":
                self.set_sreg64(o[0], self.sreg64(o[1]) if self.scc else self.sreg64(o[2]))
                return None
            raise RuntimeError("SALU64 no soportada: " + i.text)
        # 32 bits
        if op in ("s_mov_b32", "s_movk_i32"):
            self.swrite(o[0], self.sread(o[1]) if op == "s_mov_b32" else self.const(o[1], 16) | (
                0xFFFF0000 if self.const(o[1], 16) & 0x8000 else 0))
            return None
        if op == "s_not_b32":
            r = ~self.sread(o[1]) & 0xFFFFFFFF
            self.swrite(o[0], r)
            self.scc = int(r != 0)
            return None
        if op == "s_brev_b32":
            v = self.sread(o[1])
            self.swrite(o[0], int("{:032b}".format(v)[::-1], 2))
            return None
        if op == "s_abs_i32":
            v = self.sread(o[1])
            v = v - (1 << 32) if v & 0x80000000 else v
            r = abs(v) & 0xFFFFFFFF
            self.swrite(o[0], r)
            self.scc = int(r != 0)
            return None

        def sx(v):
            return v - (1 << 32) if v & 0x80000000 else v
        if op in ("s_cmpk_lg_i32", "s_cmpk_eq_i32", "s_cmpk_lt_i32", "s_cmpk_gt_i32"):
            a = sx(self.sread(o[0]))
            k = self.const(o[1], 16)
            k = k - 0x10000 if k & 0x8000 else k
            self.scc = int({"s_cmpk_lg_i32": a != k, "s_cmpk_eq_i32": a == k,
                            "s_cmpk_lt_i32": a < k, "s_cmpk_gt_i32": a > k}[op])
            return None
        if op.startswith("s_cmp_"):
            a, b = self.sread(o[0]), self.sread(o[1])
            if op.endswith("_i32"):
                a, b = sx(a), sx(b)
            c = op[6:8]
            self.scc = int({"eq": a == b, "lg": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[c])
            return None
        if op == "s_bitcmp1_b32":
            self.scc = int((self.sread(o[0]) >> (self.sread(o[1]) & 31)) & 1)
            return None
        if op == "s_addk_i32":
            k = self.const(o[1], 16)
            k = k - 0x10000 if k & 0x8000 else k
            a = sx(self.sread(o[0]))
            r = a + k
            self.scc = int(r > 0x7FFFFFFF or r < -0x80000000)
            self.swrite(o[0], r)
            return None
        a = self.sread(o[1])
        b = self.sread(o[2]) if len(o) > 2 else 0
        if op == "s_add_u32":
            r = a + b
            self.scc = int(r >> 32)
        elif op == "s_addc_u32":
            r = a + b + self.scc
            self.scc = int(r >> 32)
        elif op == "s_sub_u32":
            r = a - b
            self.scc = int(b > a)
        elif op == "s_add_i32":
            r = sx(a) + sx(b)
            self.scc = int(r > 0x7FFFFFFF or r < -0x80000000)
        elif op == "s_sub_i32":
            r = sx(a) - sx(b)
            self.scc = int(r > 0x7FFFFFFF or r < -0x80000000)
        elif op == "s_mul_i32":
            r = sx(a) * sx(b)
        elif op == "s_lshl_b32":
            r = (a << (b & 31)) & 0xFFFFFFFF
            self.scc = int(r != 0)
        elif op == "s_lshr_b32":
            r = a >> (b & 31)
            self.scc = int(r != 0)
        elif op == "s_ashr_i32":
            r = sx(a) >> (b & 31)
            self.scc = int(r != 0)
        elif op in ("s_and_b32", "s_or_b32", "s_xor_b32", "s_andn2_b32"):
            r = {"s_and_b32": a & b, "s_or_b32": a | b, "s_xor_b32": a ^ b, "s_andn2_b32": a & ~b}[op] & 0xFFFFFFFF
            self.scc = int(r != 0)
        elif op in ("s_min_i32", "s_max_i32"):
            r = min(sx(a), sx(b)) if op == "s_min_i32" else max(sx(a), sx(b))
            self.scc = int((sx(a) < sx(b)) if op == "s_min_i32" else (sx(a) > sx(b)))
        elif op in ("s_min_u32", "s_max_u32"):
            r = min(a, b) if op == "s_min_u32" else max(a, b)
        elif op == "s_cselect_b32":
            r = a if self.scc else b
        elif op == "s_bfe_u32":
            off, w = b & 31, (b >> 16) & 0x7F
            r = (a >> off) & ((1 << w) - 1)
            self.scc = int(r != 0)
        else:
            raise RuntimeError("SALU no soportada: " + i.text)
        self.swrite(o[0], r)
        return None

    # --------------------------------------------------------------------- VALU
    def valu(self, i):
        op, o = i.op, i.ops
        base = re.sub(r"_e(32|64)$", "", op)
        em = self.emask()
        clamp = "clamp" in i.mods
        F = self.vsrc_f
        if base.startswith("v_cmp_"):
            m = re.match(r"v_cmp_(\w+?)_(f32|i32|u32|u64|i64)$", base)
            cond, ty = m.group(1), m.group(2)
            dst, a, b = (("vcc", o[1], o[2]) if op.endswith("_e32") else (o[0], o[1], o[2]))
            if op.endswith("_e32") and o[0] != "vcc":
                dst, a, b = "vcc", o[0], o[1]
            if ty == "f32":
                x, y = F(a), F(b)
                with np.errstate(invalid="ignore"):
                    r = {"lt": x < y, "gt": x > y, "le": x <= y, "ge": x >= y, "eq": x == y, "lg": (x < y) | (x > y),
                         "neq": ~(x == y), "nlt": ~(x < y), "ngt": ~(x > y), "nle": ~(x <= y), "nge": ~(x >= y),
                         "o": ~(np.isnan(x) | np.isnan(y)), "u": np.isnan(x) | np.isnan(y)}[cond]
            elif ty in ("u64", "i64"):
                x, y = self.vread64(a), self.vread64(b)
                if ty == "i64":
                    x, y = x.view(np.int64), y.view(np.int64)
                r = {"lt": x < y, "gt": x > y, "le": x <= y, "ge": x >= y, "eq": x == y, "ne": x != y}[cond]
            else:
                x, y = self.vread(a), self.vread(b)
                if ty == "i32":
                    x, y = x.view(i32), y.view(i32)
                r = {"lt": x < y, "gt": x > y, "le": x <= y, "ge": x >= y, "eq": x == y, "ne": x != y}[cond]
            self.set_sreg64(dst, self.tomask(r & em))
            return
        if base == "v_cndmask_b32":
            if op.endswith("_e32"):
                d, a, b, msk = o[0], o[1], o[2], self.vcc
            else:
                d, a, b, msk = o[0], o[1], o[2], self.sreg64(o[3])
            sel = self.frommask(msk)
            av = self.vread(a) if not a.startswith("-") and not a.startswith("|") else fbits(F(a))
            bv = self.vread(b) if not b.startswith("-") and not b.startswith("|") else fbits(F(b))
            self.vwrite(d, np.where(sel, bv, av), em)
            return
        if base == "v_readfirstlane_b32":
            src = self.vread(o[1])
            lanes = np.nonzero(em)[0]
            self.swrite(o[0], int(src[lanes[0] if len(lanes) else 0]))
            return
        if base == "v_mov_b32":
            self.vwrite(o[0], self.vread(o[1]), em)
            return
        if base == "v_movrels_b32":
            r = int(o[1][1:]) + (self.m0 & 0xFFFF)
            self.vwrite(o[0], self.v[r].copy(), em)
            return
        if base == "v_movreld_b32":
            r = int(o[0][1:]) + (self.m0 & 0xFFFF)
            self.v[r][em] = self.vread(o[1])[em]
            return
        # --- carry
        if base in ("v_add_i32", "v_sub_i32", "v_subrev_i32", "v_addc_u32", "v_subb_u32", "v_subbrev_u32"):
            if op.endswith("_e64"):
                d, cd, srcs = o[0], o[1], o[2:]
            else:
                d, cd, srcs = o[0], "vcc", o[1:]
                if srcs and srcs[0] == "vcc":
                    srcs = srcs[1:]
            a = self.vread(srcs[0]).astype(np.uint64)
            b = self.vread(srcs[1]).astype(np.uint64)
            cin = np.zeros(NL, np.uint64)
            if base in ("v_addc_u32", "v_subb_u32", "v_subbrev_u32"):
                cmask = self.sreg64(srcs[2]) if len(srcs) > 2 else self.vcc
                cin = self.frommask(cmask).astype(np.uint64)
            if base in ("v_add_i32", "v_addc_u32"):
                r = a + b + cin
                c = r >> np.uint64(32)
            elif base in ("v_sub_i32", "v_subb_u32"):
                r = a - b - cin
                c = (b + cin > a).astype(np.uint64)
            else:
                r = b - a - cin
                c = (a + cin > b).astype(np.uint64)
            self.vwrite(d, (r & np.uint64(0xFFFFFFFF)).astype(u32), em)
            self.set_sreg64(cd, self.tomask((c != 0) & em))
            return
        if base == "v_lshl_b64":
            x = self.vread64(o[1])
            sh = self.vread(o[2]) & 63
            r = (x << sh.astype(np.uint64))
            self.vwrite64(o[0], r, em)
            return
        if base == "v_mad_u64_u32":
            a = self.vread(o[2]).astype(np.uint64)
            b = self.vread(o[3]).astype(np.uint64)
            c = self.vread64(o[4])
            r = a * b + c
            self.vwrite64(o[0], r, em)
            self.set_sreg64(o[1], 0)
            return
        if base in ("v_div_scale_f32",):
            # sin escalado (los valores de prueba no estan cerca de los limites): d = src0
            d = o[0]
            srcs = o[2:]
            self.vwrite_f(d, F(srcs[0]), em)
            self.set_sreg64(o[1], 0)
            return
        if base == "v_div_fmas_f32":
            a, b, c = F(o[1]), F(o[2]), F(o[3])
            self.vwrite_f(o[0], (a.astype(np.float64) * b + c).astype(f32), em)
            return
        if base == "v_div_fixup_f32":
            q, den, num = F(o[1]), F(o[2]), F(o[3])
            with np.errstate(all="ignore"):
                r = np.where(den == 0, np.where(num == 0, np.nan, np.copysign(np.inf, num) * np.sign(den + 0.0)), q)
            self.vwrite_f(o[0], r.astype(f32), em)
            return
        res = self.valu_arith(base, o, F, em)
        if res is None:
            raise RuntimeError("VALU no soportada: " + i.text)
        kind, val = res
        if kind == "f":
            if "mul" in i.mods:
                val = val * f32(float(i.mods["mul"]))
            if "div" in i.mods:
                val = val / f32(float(i.mods["div"]))
            self.vwrite_f(o[0], val, em, clamp)
        else:
            self.vwrite(o[0], val, em)

    def vread64(self, o):
        if o.startswith("v["):
            r = self.vreg_range(o)
            return self.v[r[0]].astype(np.uint64) | (self.v[r[1]].astype(np.uint64) << np.uint64(32))
        return np.full(NL, self.sreg64(o), np.uint64)

    def vwrite64(self, o, val, em):
        r = self.vreg_range(o)
        val = np.asarray(val, np.uint64)
        self.v[r[0]][em] = (val & np.uint64(0xFFFFFFFF)).astype(u32)[em]
        self.v[r[1]][em] = (val >> np.uint64(32)).astype(u32)[em]

    def valu_arith(self, b, o, F, em):
        with np.errstate(all="ignore"):
            if b == "v_add_f32":
                return "f", F(o[1]) + F(o[2])
            if b == "v_sub_f32":
                return "f", F(o[1]) - F(o[2])
            if b == "v_subrev_f32":
                return "f", F(o[2]) - F(o[1])
            if b == "v_mul_f32":
                return "f", F(o[1]) * F(o[2])
            if b == "v_fma_f32":
                return "f", (F(o[1]).astype(np.float64) * F(o[2]) + F(o[3])).astype(f32)
            if b == "v_mad_f32":
                return "f", (F(o[1]) * F(o[2])) + F(o[3])
            if b == "v_mac_f32":
                return "f", (F(o[1]) * F(o[2])) + bitsf(self.vread(o[0]))
            if b == "v_madmk_f32":   # d = s0 * K + s1
                return "f", (F(o[1]) * bitsf(np.full(NL, self.const(o[2]), u32))) + F(o[3])
            if b == "v_madak_f32":   # d = s0 * s1 + K
                return "f", (F(o[1]) * F(o[2])) + bitsf(np.full(NL, self.const(o[3]), u32))
            if b in ("v_max_f32", "v_min_f32"):
                x, y = F(o[1]), F(o[2])
                return "f", np.fmax(x, y) if b == "v_max_f32" else np.fmin(x, y)
            if b in ("v_max3_f32", "v_min3_f32"):
                x, y, z = F(o[1]), F(o[2]), F(o[3])
                return "f", (np.fmax(np.fmax(x, y), z) if b == "v_max3_f32" else np.fmin(np.fmin(x, y), z))
            if b == "v_med3_f32":
                x, y, z = F(o[1]), F(o[2]), F(o[3])
                return "f", np.median(np.stack([x, y, z]), 0).astype(f32)
            if b == "v_rcp_f32" or b == "v_rcp_iflag_f32":
                return "f", (f32(1.0) / F(o[1])).astype(f32)
            if b == "v_sqrt_f32":
                return "f", np.sqrt(F(o[1])).astype(f32)
            if b == "v_rsq_f32":
                return "f", (f32(1.0) / np.sqrt(F(o[1]))).astype(f32)
            if b == "v_exp_f32":
                return "f", np.exp2(F(o[1])).astype(f32)
            if b == "v_log_f32":
                return "f", np.log2(F(o[1])).astype(f32)
            if b == "v_floor_f32":
                return "f", np.floor(F(o[1]))
            if b == "v_ceil_f32":
                return "f", np.ceil(F(o[1]))
            if b == "v_trunc_f32":
                return "f", np.trunc(F(o[1]))
            if b == "v_rndne_f32":
                return "f", np.rint(F(o[1]))
            if b == "v_fract_f32":
                x = F(o[1])
                return "f", (x - np.floor(x)).astype(f32)
            if b == "v_cvt_f32_i32":
                return "f", self.vread(o[1]).view(i32).astype(f32)
            if b == "v_cvt_f32_u32":
                return "f", self.vread(o[1]).astype(f32)
            if b.startswith("v_cvt_f32_ubyte"):
                k = int(b[-1])
                return "f", ((self.vread(o[1]) >> u32(8 * k)) & u32(0xFF)).astype(f32)
            if b == "v_cvt_i32_f32":
                x = np.nan_to_num(F(o[1]), nan=0.0)
                return "i", np.clip(np.trunc(x), -2147483648.0, 2147483647.0).astype(np.int64).astype(i32).view(u32)
            if b == "v_cvt_u32_f32":
                x = np.nan_to_num(F(o[1]), nan=0.0)
                return "i", np.clip(np.trunc(x), 0.0, 4294967295.0).astype(np.uint64).astype(u32)
            V = self.vread
            if b == "v_mul_lo_u32" or b == "v_mul_lo_i32":
                return "i", (V(o[1]).astype(np.uint64) * V(o[2]).astype(np.uint64)).astype(u32)
            if b == "v_mul_hi_u32":
                return "i", ((V(o[1]).astype(np.uint64) * V(o[2]).astype(np.uint64)) >> np.uint64(32)).astype(u32)
            if b == "v_mul_hi_i32":
                return "i", ((V(o[1]).view(i32).astype(np.int64) * V(o[2]).view(i32).astype(np.int64)) >> 32).astype(
                    np.int64).astype(u32)
            if b == "v_mul_u32_u24":
                return "i", ((V(o[1]) & u32(0xFFFFFF)).astype(np.uint64) * (V(o[2]) & u32(0xFFFFFF))).astype(u32)
            if b == "v_mul_i32_i24":
                s24 = lambda x: ((x & u32(0xFFFFFF)).astype(np.int64) ^ 0x800000) - 0x800000
                return "i", (s24(V(o[1])) * s24(V(o[2]))).astype(np.int64).astype(u32)
            if b == "v_mad_u32_u24":
                return "i", (((V(o[1]) & u32(0xFFFFFF)).astype(np.uint64) * (V(o[2]) & u32(0xFFFFFF))) + V(o[3])).astype(u32)
            if b == "v_mad_i32_i24":
                s24 = lambda x: ((x & u32(0xFFFFFF)).astype(np.int64) ^ 0x800000) - 0x800000
                return "i", (s24(V(o[1])) * s24(V(o[2])) + V(o[3]).view(i32).astype(np.int64)).astype(np.int64).astype(u32)
            if b in ("v_add_u32",):
                return "i", (V(o[1]).astype(np.uint64) + V(o[2])).astype(u32)
            if b == "v_lshlrev_b32":
                return "i", (V(o[2]).astype(np.uint64) << (V(o[1]) & u32(31)).astype(np.uint64)).astype(u32)
            if b == "v_lshrrev_b32":
                return "i", V(o[2]) >> (V(o[1]) & u32(31))
            if b == "v_ashrrev_i32":
                return "i", (V(o[2]).view(i32) >> (V(o[1]) & u32(31)).astype(i32)).view(u32)
            if b == "v_lshl_b32":
                return "i", (V(o[1]).astype(np.uint64) << (V(o[2]) & u32(31)).astype(np.uint64)).astype(u32)
            if b == "v_lshr_b32":
                return "i", V(o[1]) >> (V(o[2]) & u32(31))
            if b == "v_ashr_i32":
                return "i", (V(o[1]).view(i32) >> (V(o[2]) & u32(31)).astype(i32)).view(u32)
            if b == "v_and_b32":
                return "i", V(o[1]) & V(o[2])
            if b == "v_or_b32":
                return "i", V(o[1]) | V(o[2])
            if b == "v_xor_b32":
                return "i", V(o[1]) ^ V(o[2])
            if b == "v_not_b32":
                return "i", ~V(o[1])
            if b == "v_bfe_u32":
                off, w = V(o[2]) & u32(31), V(o[3]) & u32(31)
                return "i", (V(o[1]) >> off) & ((u32(1) << w) - u32(1))
            if b in ("v_min_i32", "v_max_i32"):
                x, y = V(o[1]).view(i32), V(o[2]).view(i32)
                return "i", (np.minimum(x, y) if b == "v_min_i32" else np.maximum(x, y)).view(u32)
            if b in ("v_min_u32", "v_max_u32"):
                x, y = V(o[1]), V(o[2])
                return "i", np.minimum(x, y) if b == "v_min_u32" else np.maximum(x, y)
            if b == "v_med3_i32":
                x = np.stack([V(o[1]).view(i32), V(o[2]).view(i32), V(o[3]).view(i32)])
                return "i", np.sort(x, 0)[1].view(u32)
            if b == "v_med3_u32":
                x = np.stack([V(o[1]), V(o[2]), V(o[3])])
                return "i", np.sort(x, 0)[1]
            if b in ("v_max3_i32", "v_min3_i32"):
                x = np.stack([V(o[1]).view(i32), V(o[2]).view(i32), V(o[3]).view(i32)])
                return "i", (x.max(0) if b == "v_max3_i32" else x.min(0)).view(u32)
        return None

    # -------------------------------------------------------------------- MUBUF
    def mubuf(self, i):
        op, o, md = i.op, i.ops, i.mods
        em = self.emask()
        if not em.any():
            return
        rsrc = [int(self.s[r]) for r in self.sreg_range(o[2])]
        base = rsrc[0] | ((rsrc[1] & 0xFFFF) << 32)
        stride = (rsrc[1] >> 16) & 0x3FFF
        nrec = rsrc[2]
        soff = self.sread(o[3]) if len(o) > 3 else 0
        ioff = int(md.get("offset", 0)) if "offset" in md else 0
        lanes = np.nonzero(em)[0]
        if md.get("addr64"):
            va = self.vread64(o[1])
            addr = (va + np.uint64(base + soff + ioff)) & np.uint64((1 << 48) - 1)
        else:
            # buffer "crudo": base del V# + soffset + offset [+ vaddr con offen] [+ indice*stride con idxen],
            # con el control de rango del hardware (fuera de num_records: lee 0, no escribe)
            vregs = [] if o[1] == "off" else self.vreg_range(o[1])
            off = np.full(NL, ioff, np.int64)
            k = 0
            idx = np.zeros(NL, np.int64)
            if md.get("idxen"):
                idx = self.v[vregs[k]].astype(np.int64)
                k += 1
            if md.get("offen"):
                off = off + self.v[vregs[k]].astype(np.int64)
            if stride:
                inrange = idx < nrec
            else:
                inrange = off < nrec
            lanes = lanes[inrange[lanes]]
            addr = (np.uint64(base + soff) + (idx * stride + off).astype(np.uint64)) & np.uint64((1 << 48) - 1)
            if not len(lanes):
                return
        nb = {"buffer_load_dword": 4, "buffer_load_dwordx2": 8, "buffer_load_dwordx3": 12, "buffer_load_dwordx4": 16,
              "buffer_load_ubyte": 1, "buffer_load_ushort": 2, "buffer_load_sbyte": 1, "buffer_load_sshort": 2,
              "buffer_store_dword": 4, "buffer_store_dwordx2": 8, "buffer_store_dwordx3": 12,
              "buffer_store_dwordx4": 16, "buffer_store_byte": 1, "buffer_store_short": 2}[op]
        regs = self.vreg_range(o[0])
        if "load" in op:
            data = self.mem.gather(addr[lanes], nb)
            if nb >= 4:
                d32 = data.view(u32).reshape(len(lanes), nb // 4)
                for k, r in enumerate(regs[:nb // 4]):
                    self.v[r][lanes] = d32[:, k]
            else:
                val = data[:, 0].astype(u32) if nb == 1 else data.view(np.uint16)[:, 0].astype(u32)
                if op == "buffer_load_sbyte":
                    val = val.astype(np.int8).astype(i32).view(u32)
                if op == "buffer_load_sshort":
                    val = val.astype(np.int16).astype(i32).view(u32)
                self.v[regs[0]][lanes] = val
        else:
            if nb >= 4:
                d = np.stack([self.v[r][lanes] for r in regs[:nb // 4]], 1).astype(u32)
                self.mem.scatter(addr[lanes], d.view(np.uint8).reshape(len(lanes), nb))
            else:
                v = self.v[regs[0]][lanes]
                d = (v.astype(np.uint8).reshape(-1, 1) if nb == 1 else v.astype(np.uint16).view(np.uint8).reshape(-1, 2))
                self.mem.scatter(addr[lanes], d)

    # --------------------------------------------------------------------- MIMG
    def mimg(self, i):
        op, o, md = i.op, i.ops, i.mods
        em = self.emask()
        lanes = np.nonzero(em)[0]
        if not len(lanes):
            return
        t = [int(self.s[r]) for r in self.sreg_range(o[2])]
        img = Image(self.mem, t)
        vd = self.vreg_range(o[0])
        va = self.vreg_range(o[1])
        dmask = int(md.get("dmask", "0xf"), 16)
        if op == "image_load":
            x = self.v[va[0]][lanes].view(i32)
            y = self.v[va[1]][lanes].view(i32)
            px = img.fetch(x, y)
        elif op == "image_sample_lz":
            smp = [int(self.s[r]) for r in self.sreg_range(o[3])]
            unnorm, filt = sampler_bilinear(smp)
            u = bitsf(self.v[va[0]][lanes]).copy()
            vv = bitsf(self.v[va[1]][lanes]).copy()
            if not unnorm:
                u *= img.w
                vv *= img.h
            if filt == 0:   # punto
                px = img.fetch(np.floor(u).astype(np.int64), np.floor(vv).astype(np.int64))
            else:
                xf, yf = u - f32(0.5), vv - f32(0.5)
                x0, y0 = np.floor(xf).astype(np.int64), np.floor(yf).astype(np.int64)
                fx, fy = (xf - x0)[:, None], (yf - y0)[:, None]
                px = (img.fetch(x0, y0) * (1 - fx) + img.fetch(x0 + 1, y0) * fx) * (1 - fy) + \
                     (img.fetch(x0, y0 + 1) * (1 - fx) + img.fetch(x0 + 1, y0 + 1) * fx) * fy
        elif op == "image_store":
            x = self.v[va[0]][lanes].view(i32)
            y = self.v[va[1]][lanes].view(i32)
            vals = np.zeros((len(lanes), 4), f32)
            k = 0
            for c in range(4):
                if dmask >> c & 1:
                    vals[:, c] = bitsf(self.v[vd[k]][lanes])
                    k += 1
            img.store(x, y, vals)
            return
        else:
            raise RuntimeError("MIMG no soportada: " + i.text)
        k = 0
        for c in range(4):
            if dmask >> c & 1:
                self.v[vd[k]][lanes] = fbits(px[:, c].astype(f32))
                k += 1

    # ----------------------------------------------------------------------- DS
    def ds(self, i):
        op, o, md = i.op, i.ops, i.mods
        em = self.emask()
        lanes = np.nonzero(em)[0]
        if not len(lanes):
            return
        L = self.lds
        if op in ("ds_read_b32", "ds_read2_b32", "ds_read_b64", "ds_read_b128", "ds_read2st64_b32"):
            addr = self.v[int(o[1][1:])][lanes].astype(np.int64)
            regs = self.vreg_range(o[0])
            if op == "ds_read_b32":
                a = addr + int(md.get("offset", 0))
                self.v[regs[0]][lanes] = L[a // 4]
            elif op in ("ds_read_b64", "ds_read_b128"):
                a = addr + int(md.get("offset", 0))
                for k, r in enumerate(regs):
                    self.v[r][lanes] = L[a // 4 + k]
            else:
                mul = 64 * 4 if "st64" in op else 4
                a0 = addr + int(md.get("offset0", 0)) * mul
                a1 = addr + int(md.get("offset1", 0)) * mul
                self.v[regs[0]][lanes] = L[a0 // 4]
                self.v[regs[1]][lanes] = L[a1 // 4]
        elif op in ("ds_write_b32", "ds_write2_b32", "ds_write2st64_b32", "ds_write_b64", "ds_write_b128"):
            addr = self.v[int(o[0][1:])][lanes].astype(np.int64)
            if op == "ds_write_b32":
                a = addr + int(md.get("offset", 0))
                L[a // 4] = self.v[int(o[1][1:])][lanes]
            elif op in ("ds_write_b64", "ds_write_b128"):
                a = addr + int(md.get("offset", 0))
                for k, r in enumerate(self.vreg_range(o[1])):
                    L[a // 4 + k] = self.v[r][lanes]
            else:
                mul = 64 * 4 if "st64" in op else 4
                a0 = addr + int(md.get("offset0", 0)) * mul
                a1 = addr + int(md.get("offset1", 0)) * mul
                L[a0 // 4] = self.v[int(o[1][1:])][lanes]
                L[a1 // 4] = self.v[int(o[2][1:])][lanes]
        else:
            raise RuntimeError("DS no soportada: " + i.text)


# ------------------------------------------------------------------------- PM4
SH_BASE = 0xB000
R_PGM_LO, R_RSRC1, R_RSRC2, R_NTX, R_UD0 = 0xB830, 0xB848, 0xB84C, 0xB81C, 0xB900


class Emu:
    def __init__(self, mem, progs, verbose=False):
        self.mem = mem
        self.progs = progs
        self.by_bytes = {}
        for name, (blob, ins) in progs.items():
            self.by_bytes.setdefault(blob[:64], []).append((name, blob, ins))
        self.verbose = verbose
        self.stats = {}

    def program_at(self, pgm):
        head = bytes(self.mem.read(pgm, 64))
        for name, blob, ins in self.by_bytes.get(head, []):
            if bytes(self.mem.read(pgm, len(blob))) == blob:
                return name, ins
        raise RuntimeError("el codigo en 0x%x no coincide con ningun kernel ensamblado" % pgm)

    def run_dcb(self, dcb):
        sh = {}
        n = 0
        k = 0
        while k < len(dcb):
            h = int(dcb[k])
            if h == 0x80000000:
                k += 1
                continue
            typ = h >> 30
            if typ != 3:
                k += 1
                continue
            opc = (h >> 8) & 0xFF
            cnt = ((h >> 16) & 0x3FFF) + 1
            body = dcb[k + 1:k + 1 + cnt]
            if opc == 0x76:   # SET_SH_REG
                reg = SH_BASE + int(body[0]) * 4
                for j in range(1, cnt):
                    sh[reg + 4 * (j - 1)] = int(body[j])
            elif opc == 0x15:   # DISPATCH_DIRECT
                self.dispatch(sh, int(body[0]), int(body[1]), int(body[2]))
                n += 1
            elif opc == 0x47:   # EVENT_WRITE_EOP: label
                addr = int(body[1]) | ((int(body[2]) & 0xFFFF) << 32)
                a, off = self.mem.find(addr, 4)
                a[off:off + 4] = np.frombuffer(struct.pack("<I", int(body[3])), np.uint8)
            k += 1 + cnt
        return n

    def dispatch(self, sh, gx, gy, gz):
        pgm = (sh[R_PGM_LO] << 8) | (sh.get(R_PGM_LO + 4, 0) << 40)
        name, ins = self.program_at(pgm)
        rsrc2 = sh[R_RSRC2]
        rsrc1 = sh[R_RSRC1]
        nusr = (rsrc2 >> 1) & 0x1F
        tgx, tgy, tgz, tgsz = (rsrc2 >> 7) & 1, (rsrc2 >> 8) & 1, (rsrc2 >> 9) & 1, (rsrc2 >> 10) & 1
        tidc = (rsrc2 >> 11) & 3
        lds_bytes = ((rsrc2 >> 15) & 0x1FF) * 256
        ntx, nty = sh[R_NTX], sh[R_NTX + 4]
        if ntx * nty != NL:
            raise RuntimeError("grupo de %dx%d hilos (se esperaba 64)" % (ntx, nty))
        ud = [sh.get(R_UD0 + 4 * j, 0) for j in range(nusr)]
        lx = np.tile(np.arange(ntx, dtype=u32), nty)
        ly = np.repeat(np.arange(nty, dtype=u32), ntx)
        t0 = time.time()
        steps = 0
        for y in range(gy):
            for x in range(gx):
                sg = list(ud)
                if tgx:
                    sg.append(x)
                if tgy:
                    sg.append(y)
                if tgz:
                    sg.append(0)
                if tgsz:
                    sg.append(NL)
                lds = np.zeros(max(lds_bytes, 64 * 1024) // 4, u32)
                w = Wave(self, ins, sg, (lx, ly if tidc >= 1 else np.zeros(NL, u32)), lds)
                steps += w.run()
        st = self.stats.setdefault(name, [0, 0, 0.0])
        st[0] += 1
        st[1] += steps
        st[2] += time.time() - t0
        if self.verbose:
            print("  %-20s %4dx%-4d grupos, %9d instrucciones de wave, %.1fs" % (name, gx, gy, steps, time.time() - t0),
                  flush=True)
        _ = rsrc1


def load_dump(path):
    b = open(path, "rb").read()
    magic, nr = struct.unpack_from("<II", b, 0)
    assert magic == 0x4E58444D, "no es un volcado de gpu_emu"
    off = 8
    mem = Memory()
    for _ in range(nr):
        g, z = struct.unpack_from("<QQ", b, off)
        off += 16
        mem.add(g, np.frombuffer(b, np.uint8, z, off).copy())
        off += z
    (nd,) = struct.unpack_from("<I", b, off)
    off += 4
    dcb = np.frombuffer(b, np.uint32, nd, off).copy()
    off += 4 * nd
    fb, ib = struct.unpack_from("<QQ", b, off)
    off += 16
    W, H, pitch = struct.unpack_from("<iii", b, off)
    return mem, dcb, fb, ib, W, H, pitch


def verify_against_header(progs):
    """Los bytes ensamblados tienen que ser los mismos que empaqueto gcn_pack.py en nx_shaders.h."""
    hdr = open(os.path.join(ROOT, "plugin", "include", "nx_shaders.h")).read()
    bad = []
    n = 0
    for m in re.finditer(r"static const uint32_t nx_code_(\w+)\[(\d+)\][^{]*\{([^}]*)\}", hdr):
        name, words = m.group(1), [int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]+", m.group(3))]
        blob = struct.pack("<%dI" % len(words), *words)
        if name not in progs:
            bad.append(name + " (no esta en los .s)")
            continue
        if progs[name][0] != blob:
            bad.append(name)
        n += 1
    return n, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out-mid")
    ap.add_argument("--out-real")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    progs = load_programs()
    n, bad = verify_against_header(progs)
    print("kernels: %d ensamblados con llvm-mc, %d comparados con nx_shaders.h: %s" %
          (len(progs), n, "bytes identicos" if not bad else "DISTINTOS: " + ", ".join(bad)))
    if bad:
        return 1
    mem, dcb, fb, ib, W, H, pitch = load_dump(a.dump)
    emu = Emu(mem, progs, a.v)
    t0 = time.time()
    nd = emu.run_dcb(dcb)
    tot = sum(s[1] for s in emu.stats.values())
    print("dispatches: %d, instrucciones de wave ejecutadas: %d, %.0fs" % (nd, tot, time.time() - t0))

    def save(path, k):
        img = mem.read(fb + k * ib, ib).reshape(H, pitch, 4)[:, :W]
        img.tofile(path)
    if a.out_mid:
        save(a.out_mid, 2)
    if a.out_real:
        save(a.out_real, 3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
