"""Empaqueta code objects AMDGPU (gfx701) en una cabecera C para el plugin.

Por cada kernel extrae el codigo maquina y su kernel descriptor (.kd):
  * COMPUTE_PGM_RSRC1/RSRC2 -> se escriben tal cual con SET_SH_REG;
  * kernel_code_properties  -> que user SGPRs espera el kernel (ABI HSA:
    private segment buffer, dispatch ptr, kernarg ptr...). El plugin los
    emula cargando COMPUTE_USER_DATA_n con punteros a un "dispatch packet"
    falso y al bloque de kernargs (ver plugin/source/gnm_dispatch.c).

  python tools/gcn_pack.py build/nxflow.co build/fsr1.co -o plugin/include/nx_shaders.h
"""
import argparse
import re
import struct

import yaml
from elftools.elf.elffile import ELFFile

PROPS = [
    (0, "PRIVATE_SEGMENT_BUFFER", 4),
    (1, "DISPATCH_PTR", 2),
    (2, "QUEUE_PTR", 2),
    (3, "KERNARG_SEGMENT_PTR", 2),
    (4, "DISPATCH_ID", 2),
    (5, "FLAT_SCRATCH_INIT", 2),
    (6, "PRIVATE_SEGMENT_SIZE", 1),
]


def kernels(path):
    with open(path, "rb") as f:
        elf = ELFFile(f)
        symtab = elf.get_section_by_name(".symtab")
        syms = {s.name: s for s in symtab.iter_symbols()}

        def read(sym):
            sec = elf.get_section(sym["st_shndx"])
            off = sym["st_value"] - sec["sh_addr"]
            return sec.data()[off:off + sym["st_size"]], sym["st_value"]

        for name, s in sorted(syms.items()):
            if not name.endswith(".kd"):
                continue
            kname = name[:-3]
            kd, kd_addr = read(s)
            (group, private, kernarg_size, _r0, entry_off) = struct.unpack_from("<IIIIq", kd, 0)
            rsrc3, rsrc1, rsrc2, props = struct.unpack_from("<IIIH", kd, 44)
            code, code_addr = read(syms[kname])
            assert kd_addr + entry_off == code_addr, kname
            assert private == 0, f"{kname} usa scratch ({private} B): no soportado por el plugin"
            yield dict(name=kname, code=code, rsrc1=rsrc1, rsrc2=rsrc2, props=props,
                       kernarg_size=kernarg_size, lds=group)


def asm_metadata(paths):
    """Lee el bloque .amdgpu_metadata de los .s: offsets de kernargs y uso de registros."""
    meta = {}
    for path in paths:
        text = open(path).read()
        m = re.search(r"\.amdgpu_metadata\n(.*?)\n\s*\.end_amdgpu_metadata", text, re.S)
        for k in yaml.safe_load(m.group(1))["amdhsa.kernels"]:
            meta[k[".name"]] = k
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("objs", nargs="+")
    ap.add_argument("--asm", nargs="*", default=[], help=".s con metadatos (offsets de argumentos)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    ks = [k for p in a.objs for k in kernels(p)]
    meta = asm_metadata(a.asm)
    L = ["// Generado por tools/gcn_pack.py desde shaders/*.cl: no editar a mano.",
         "// ISA GCN gfx701 (PS4) + registros de programa de cada kernel.",
         "#pragma once", "#include <stdint.h>", "",
         "typedef struct {",
         "    const char* name;",
         "    const uint32_t* code;   // copiar a memoria GPU alineada a 256 B",
         "    uint32_t code_dwords;",
         "    uint32_t rsrc1, rsrc2;  // COMPUTE_PGM_RSRC1 / COMPUTE_PGM_RSRC2",
         "    uint16_t props;         // kernel_code_properties (HSA)",
         "    uint16_t user_sgprs;    // = rsrc2.USER_SGPR",
         "    uint32_t kernarg_size;",
         "    const uint8_t* arg_offsets;  // offset de cada argumento (metadatos del compilador)",
         "    uint8_t num_args;",
         "    uint8_t vgprs, sgprs;",
         "} nx_shader_t;", ""]
    for bit, name, _ in PROPS:
        L.append(f"#define NX_PROP_{name} (1u << {bit})")
    L.append("")
    for k in ks:
        code = k["code"] + b"\0" * (-len(k["code"]) % 4)
        dw = struct.unpack(f"<{len(code) // 4}I", code)
        L.append(f"static const uint32_t nx_code_{k['name']}[{len(dw)}] __attribute__((aligned(256))) = {{")
        for i in range(0, len(dw), 8):
            L.append("    " + ", ".join(f"0x{v:08x}" for v in dw[i:i + 8]) + ",")
        L.append("};")
        args = meta.get(k["name"], {}).get(".args", [])
        offs = ", ".join(str(x[".offset"]) for x in args) or "0"
        L.append(f"static const uint8_t nx_args_{k['name']}[] = {{ {offs} }};")
    L.append("")
    L.append(f"#define NX_NUM_SHADERS {len(ks)}")
    L.append("static const nx_shader_t nx_shaders[NX_NUM_SHADERS] = {")
    for k in ks:
        user = (k["rsrc2"] >> 1) & 0x1F
        code_dw = (len(k["code"]) + 3) // 4
        md = meta.get(k["name"], {})
        L.append(f"    {{ \"{k['name']}\", nx_code_{k['name']}, {code_dw}, 0x{k['rsrc1']:08x}, "
                 f"0x{k['rsrc2']:08x}, 0x{k['props']:04x}, {user}, {k['kernarg_size']}, "
                 f"nx_args_{k['name']}, {len(md.get('.args', []))}, "
                 f"{md.get('.vgpr_count', 0)}, {md.get('.sgpr_count', 0)} }},")
    L.append("};")
    with open(a.out, "w") as f:
        f.write("\n".join(L) + "\n")
    for k in ks:
        props = [n for b, n, _ in PROPS if k["props"] >> b & 1]
        print(f"{k['name']:22s} {len(k['code']):6d} B  user_sgprs={(k['rsrc2'] >> 1) & 31:2d} "
              f"kernarg={k['kernarg_size']:3d}  {' '.join(props)}")


if __name__ == "__main__":
    main()
