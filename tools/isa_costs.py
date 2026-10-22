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

MODES = [0, 1, 2, 3, 5, 6, 7, 8, 9]


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


if __name__ == "__main__":
    main()
