"""Exporta NXSR-L (sr/net_srl.py) a plugin/include/nxsrl_weights.h.

  python sr/export_srl.py weights/nxsr_l.pkl plugin/include/nxsrl_weights.h
"""
import pickle
import sys

import numpy as np

LAYERS = [("c0", 1, "conv3x3_5_12"), ("c1", 1, "conv3x3_12_12"), ("c2", 1, "conv3x3_12_12"), ("out", 0, "conv3x3_12_3")]


def fmt(a):
    f = np.asarray(a, np.float32).ravel()
    return "\n".join("    " + ", ".join(f"{v: .8e}f" for v in f[i:i + 8]) + "," for i in range(0, len(f), 8))


def main():
    src, dst = sys.argv[1], sys.argv[2]
    p = pickle.load(open(src, "rb"))
    out = ["// Generado por sr/export_srl.py: no editar a mano.",
           "// Pesos de la guia de NXSR-L (sr/net_srl.py). Layout HWIO: w[ky][kx][cin][cout].",
           "#pragma once", "", "typedef struct { const float* w; const float* b; int cin, cout, act; const char* kernel; } nx_srl_layer_t;", ""]
    rows = []
    for name, act, kern in LAYERS:
        w, b = np.asarray(p[name]["w"]), np.asarray(p[name]["b"])
        out += [f"static const float nx_srl_{name}_w[{w.size}] __attribute__((aligned(256))) = {{", fmt(w), "};",
                f"static const float nx_srl_{name}_b[{b.size}] = {{", fmt(b), "};", ""]
        rows.append(f'    {{ nx_srl_{name}_w, nx_srl_{name}_b, {w.shape[2]}, {w.shape[3]}, {act}, "{kern}" }},')
    out += [f"#define NX_SRL_LAYERS {len(LAYERS)}", "static const nx_srl_layer_t nx_srl_layers[NX_SRL_LAYERS] = {"] + rows + ["};"]
    open(dst, "w").write("\n".join(out) + "\n")
    print(dst)


if __name__ == "__main__":
    main()
