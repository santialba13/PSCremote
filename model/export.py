"""Exporta los pesos entrenados a una cabecera C para el plugin de PS4.

  python model/export.py weights/nxfg2.pkl plugin/include/nxflow_weights.h

Genera un array float por capa (layout HWIO, el que espera conv3x3_* en
shaders/nxflow.cl) y una tabla con el orden de ejecucion de los pases.
"""
import os
import pickle
import sys

import numpy as np

# (bloque, capa, stride, activacion, kernel) en orden de ejecucion
LAYERS = [
    ("A", "c0", 2, 1, "conv3x3_14_32"),
    ("A", "c1", 1, 1, "conv3x3_32_32"),
    ("A", "m0", 1, 1, "conv3x3_32_32"),
    ("A", "m1", 1, 1, "conv3x3_32_32"),
    ("A", "m2", 1, 1, "conv3x3_32_32"),   # + residual (salida de c1)
    ("A", "out", 1, 0, "conv3x3_32_5"),   # tras resize x2
    ("B", "c0", 2, 1, "conv3x3_17_32"),
    ("B", "c1", 1, 1, "conv3x3_32_32"),
    ("B", "m0", 1, 1, "conv3x3_32_32"),
    ("B", "m1", 1, 1, "conv3x3_32_32"),
    ("B", "m2", 1, 1, "conv3x3_32_32"),   # + residual (salida de c1)
    ("B", "out", 1, 0, "conv3x3_32_5"),   # tras resize x2
    ("C", "c0", 2, 1, "conv3x3_17_16"),   # entrada a 1/4
    ("C", "c1", 1, 1, "conv3x3_16_16"),
    ("C", "m0", 1, 1, "conv3x3_16_16"),
    ("C", "m1", 1, 1, "conv3x3_16_16"),   # + residual (salida de c1)
    ("C", "out", 1, 0, "conv3x3_16_5"),   # tras resize x2
]


# NXFG-2 (model/net2.py): la que usa el plugin desde la 2.73
LAYERS_NXFG2 = [
    ("A", "c0", 2, 1, "conv3x3_15_32"),   # entrada 1/16 -> interno 1/32
    ("A", "c1", 1, 1, "conv3x3_32_32"),
    ("A", "m0", 1, 1, "conv3x3_32_32"),
    ("A", "m1", 1, 1, "conv3x3_32_32"),   # + residual (salida de c1)
    ("A", "out", 1, 0, "conv3x3_32_5"),   # tras resize x2 (1/16)
    ("B", "c0", 2, 1, "conv3x3_17_24"),   # entrada 1/8 -> interno 1/16
    ("B", "c1", 1, 1, "conv3x3_24_24"),
    ("B", "m0", 1, 1, "conv3x3_24_24"),   # + residual (salida de c1)
    ("B", "out", 1, 0, "conv3x3_24_5"),   # tras resize x2 (1/8)
]


def fmt(arr, per_line=8):
    flat = np.asarray(arr, np.float32).ravel()
    rows = []
    for i in range(0, len(flat), per_line):
        rows.append("    " + ", ".join(f"{v: .8e}f" for v in flat[i:i + per_line]) + ",")
    return "\n".join(rows)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "rb") as f:
        p = pickle.load(f)
    layers = LAYERS_NXFG2 if "C" not in p else LAYERS
    out = ["// Generado por model/export.py: no editar a mano.",
           "// Pesos de la red de generacion de fotogramas (VCFG-NX). Layout HWIO: w[ky][kx][cin][cout].",
           "#pragma once", "", "typedef struct {", "    const float* w;", "    const float* b;",
           "    int cin, cout, stride, act;", "    const char* kernel;", "} nx_layer_t;", ""]
    table = []
    total = 0
    for blk, name, stride, act, kern in layers:
        w = np.asarray(p[blk][name]["w"]); b = np.asarray(p[blk][name]["b"])
        cname = f"nx_{blk}_{name}"
        total += w.size + b.size
        out.append(f"static const float {cname}_w[{w.size}] __attribute__((aligned(256))) = {{")
        out.append(fmt(w)); out.append("};")
        out.append(f"static const float {cname}_b[{b.size}] = {{")
        out.append(fmt(b)); out.append("};"); out.append("")
        table.append(f"    {{ {cname}_w, {cname}_b, {w.shape[2]}, {w.shape[3]}, {stride}, {act}, \"{kern}\" }},")
    blocks = []
    for blk in dict.fromkeys(b for b, *_ in layers):
        first = next(i for i, l in enumerate(layers) if l[0] == blk)
        blocks.append(f"#define NX_BLOCK_{blk} {first}, {sum(1 for l in layers if l[0] == blk)}")
    if layers is LAYERS_NXFG2:
        out.append("#define NX_NET_NXFG2 1")
    out += blocks
    out.append(f"#define NX_NUM_LAYERS {len(layers)}")
    out.append(f"#define NX_NUM_PARAMS {total}")
    out.append("static const nx_layer_t nx_layers[NX_NUM_LAYERS] = {")
    out += table
    out.append("};")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")
    print(dst, total, "parametros")


if __name__ == "__main__":
    main()
