"""Exporta NXSR a plugin/include/nxsr_weights.h con el layout de shaders/nxsr.cl:
cabeza HWIO [3][3][12][C] + b[C], cuerpo [3][3][C][C] + b[C], cola [C][12] + b[12].

  python sr/export_sr.py weights/nxsr.pkl plugin/include/nxsr_weights.h
"""
import pickle
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
p = pickle.load(open(src, "rb"))
assert set(p) == {"head", "body0", "tail"}, "shaders/nxsr.cl implementa cabeza + 1 capa de cuerpo + cola"
C = p["head"]["w"].shape[-1]
assert p["head"]["w"].shape == (3, 3, 12, C) and p["tail"]["w"].shape == (1, 1, C, 12)
parts = [p["head"]["w"], p["head"]["b"], p["body0"]["w"], p["body0"]["b"], p["tail"]["w"][0, 0], p["tail"]["b"]]
flat = np.concatenate([np.asarray(x, np.float32).ravel() for x in parts])
lines = ["// Generado por sr/export_sr.py: no editar a mano.", "#pragma once",
         f"#define NXSR_C {C}", f"#define NXSR_NUM_WEIGHTS {flat.size}",
         f"static const float nxsr_weights[{flat.size}] __attribute__((aligned(256))) = {{"]
for i in range(0, flat.size, 8):
    lines.append("    " + ", ".join(f"{v: .8e}f" for v in flat[i:i + 8]) + ",")
lines.append("};")
open(dst, "w").write("\n".join(lines) + "\n")
print(dst, flat.size, "pesos, C =", C)
