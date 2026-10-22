#!/usr/bin/env python3
"""Mide con el emulador de instrucciones (tools/gcn_isa_emu.py) cuantas instrucciones
de wavefront ejecuta cada kernel por grupo, en todos los modos del pipeline, y
escribe plugin/test/kernel_costs.h: el simulador de consola (console_sim) usa esa
tabla como costo de GPU de cada dispatch (en vez de contar grupos, que hacia ver a
un kernel de muchos grupos livianos como caro).

  make -C plugin/test gpu_emu && python3 tools/isa_costs.py video.mp4
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "model"))
import data  # noqa: E402
import gcn_isa_emu as g  # noqa: E402

MODES = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]


def main():
    fr = data.decode_video(sys.argv[1])
    H, W = 96, 160
    progs = g.load_programs()
    tot = {}
    with tempfile.TemporaryDirectory() as td:
        p = lambda n: os.path.join(td, n)
        for k, n in ((10, "f0"), (12, "f1")):
            c = fr[k][300:300 + H, 500:500 + W]
            np.dstack([c, np.full((H, W), 255, np.uint8)]).tofile(p(n + ".rgba"))
        for m in MODES:
            env = dict(os.environ, NX_DUMP=p("job.bin"))
            subprocess.run([os.path.join(ROOT, "plugin", "test", "gpu_emu"), str(W), str(H), p("f0.rgba"),
                            p("f1.rgba"), p("mid.rgba"), str(m), p("real.rgba")], check=True, env=env,
                           stdout=subprocess.DEVNULL)
            mem, dcb, *_ = g.load_dump(p("job.bin"))
            emu = g.Emu(mem, progs)
            emu.run_dcb(dcb)
            for name, (ng, steps, _) in emu.stats.items():
                t = tot.setdefault(name, [0, 0])
                t[0] += ng
                t[1] += steps
            print("modo", m, "listo", flush=True)
    rows = sorted((n, s / gr) for n, (gr, s) in tot.items())
    out = ["// Generado por tools/isa_costs.py: instrucciones de wavefront por grupo de cada",
           "// kernel, medidas ejecutando el codigo maquina GCN real (tools/gcn_isa_emu.py).",
           "#pragma once", "typedef struct { const char* name; unsigned ipg; } kernel_cost_t;",
           "static const kernel_cost_t kernel_costs[] = {"]
    out += ['    { "%s", %d },' % (n, round(v)) for n, v in rows]
    out += ["};"]
    open(os.path.join(ROOT, "plugin", "test", "kernel_costs.h"), "w").write("\n".join(out) + "\n")
    for n, v in rows:
        print("  %-20s %8.0f" % (n, v))


if __name__ == "__main__" and not (len(sys.argv) > 1 and sys.argv[1] == "--estimate"):
    main()


def estimate(W, H, modes, names):
    """Costo estimado a W x H de cada modo: grupos de cada dispatch del DCB real x
    instrucciones por grupo medidas (kernel_costs.h)."""
    import re
    costs = dict(re.findall(r'\{ "(\w+)", (\d+) \}', open(os.path.join(ROOT, "plugin", "test", "kernel_costs.h")).read()))
    progs = g.load_programs()
    by_head = {blob[:64]: n for n, (blob, _) in progs.items()}
    with tempfile.TemporaryDirectory() as td:
        p = lambda n: os.path.join(td, n)
        img = np.random.default_rng(0).integers(0, 255, (H, W, 4), np.uint8)
        img.tofile(p("f.rgba"))
        for m in modes:
            env = dict(os.environ, NX_DUMP=p("job.bin"))
            subprocess.run([os.path.join(ROOT, "plugin", "test", "gpu_emu"), str(W), str(H), p("f.rgba"), p("f.rgba"),
                            p("mid.rgba"), str(m), p("real.rgba")], check=True, env=env, stdout=subprocess.DEVNULL)
            mem, dcb, *_ = g.load_dump(p("job.bin"))
            sh, tot, per = {}, 0, {}
            k = 0
            while k < len(dcb):
                h = int(dcb[k])
                if h == 0x80000000 or (h >> 30) != 3:
                    k += 1
                    continue
                opc, cnt = (h >> 8) & 0xFF, ((h >> 16) & 0x3FFF) + 1
                body = dcb[k + 1:k + 1 + cnt]
                if opc == 0x76:
                    for j in range(1, cnt):
                        sh[0xB000 + int(body[0]) * 4 + 4 * (j - 1)] = int(body[j])
                elif opc == 0x15:
                    pgm = (sh[0xB830] << 8) | (sh.get(0xB834, 0) << 40)
                    name = by_head.get(bytes(mem.read(pgm, 64)), "?")
                    gr = int(body[0]) * int(body[1]) * int(body[2])
                    c = gr * int(costs.get(name, 300))
                    per[name] = per.get(name, 0) + c
                    tot += c
                k += 1 + cnt
            top = sorted(per.items(), key=lambda t: -t[1])[:6]
            print("%-40s %6.1f M instr  ~%.1f-%.1f ms   [%s]" % (names.get(m, m), tot / 1e6, tot / 14.4e6, tot / 7e6,
                  ", ".join("%s %.1fM" % (n, v / 1e6) for n, v in top)), flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--estimate":
    # python3 tools/isa_costs.py --estimate [W H]
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 1920
    H = int(sys.argv[3]) if len(sys.argv) > 3 else 1080
    NAMES = {0: "generacion NXFG-3", 6: "generacion sin red", 7: "FSR+", 2: "NXFG-3 + SMAA",
             3: "NXSR Clasico", 5: "NXSR Calidad (viejo)", 8: "NXSR-L Ligero", 9: "NXFG-3 + NXSR-L + SMAA",
             10: "NXSR-L Calidad", 11: "NXRE realismo", 12: "todo (NXFG-3+NXRE+SMAA+NXSR-L+FSR+)"}
    estimate(W, H, [0, 6, 7, 3, 5, 8, 10, 11, 2, 9, 12], NAMES)
