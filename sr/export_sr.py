"""Exporta un modelo NXSR a una cabecera C con el layout de shaders/nxsr.cl:
cabeza HWIO [3][3][12][C] + b[C], B x (cuerpo [3][3][C][C] + b[C]), cola [C][12] + b[12].
Las ramas de re-parametrizacion se pliegan antes (net_sr.fold).

  python sr/export_sr.py weights/nxsr.pkl plugin/include/nxsr_weights.h              # Rapido: NXSR_*
  python sr/export_sr.py weights/nxsr_hq.pkl plugin/include/nxsr_hq_weights.h --name nxsr_hq   # Calidad
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import net_sr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--name", default="nxsr")
a = ap.parse_args()
p = {k: {kk: np.asarray(vv) for kk, vv in v.items()} for k, v in net_sr.fold(pickle.load(open(a.src, "rb"))).items()}
B = sum(1 for k in p if k.startswith("body"))
assert set(p) == {"head", "tail"} | {f"body{i}" for i in range(B)} and B in (1, 2), "shaders/nxsr.cl: 1 o 2 capas de cuerpo"
C = p["head"]["w"].shape[-1]
assert p["head"]["w"].shape == (3, 3, 12, C) and p["tail"]["w"].shape == (1, 1, C, 12)
parts = [p["head"]["w"], p["head"]["b"]]
for i in range(B):
    parts += [p[f"body{i}"]["w"], p[f"body{i}"]["b"]]
parts += [p["tail"]["w"][0, 0], p["tail"]["b"]]
flat = np.concatenate([np.asarray(x, np.float32).ravel() for x in parts])
M = a.name.upper()
lines = ["// Generado por sr/export_sr.py: no editar a mano.", "#pragma once",
         f"#define {M}_C {C}", f"#define {M}_B {B}", f"#define {M}_NUM_WEIGHTS {flat.size}",
         f"static const float {a.name}_weights[{flat.size}] __attribute__((aligned(256))) = {{"]
for i in range(0, flat.size, 8):
    lines.append("    " + ", ".join(f"{v: .8e}f" for v in flat[i:i + 8]) + ",")
lines.append("};")
open(a.dst, "w").write("\n".join(lines) + "\n")
print(a.dst, flat.size, "pesos, C =", C, "B =", B)
