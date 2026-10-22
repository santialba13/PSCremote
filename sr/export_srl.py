"""Exporta los dos modelos NXSR-L (sr/net_srl.py) a plugin/include/nxsrl_weights.h.

  python sr/export_srl.py weights/nxsr_l.pkl weights/nxsr_lc.pkl plugin/include/nxsrl_weights.h
       (Ligero)            (Calidad)
"""
import pickle
import sys

import numpy as np


def fmt(a):
    f = np.asarray(a, np.float32).ravel()
    return "\n".join("    " + ", ".join(f"{v: .8e}f" for v in f[i:i + 8]) + "," for i in range(0, len(f), 8))


def model(out, tag, p):
    names = ["c0"] + sorted((k for k in p if k.startswith("c") and k != "c0"), key=lambda k: int(k[1:])) + ["out"]
    rows = []
    for i, name in enumerate(names):
        w, b = np.asarray(p[name]["w"]), np.asarray(p[name]["b"])
        act = 0 if name == "out" else 1
        res = 1 if i == len(names) - 2 and len(names) > 2 else 0   # la ultima capa media suma c0
        kern = f"conv3x3_{w.shape[2]}_{w.shape[3]}"
        out += [f"static const float nx_srl_{tag}_{name}_w[{w.size}] __attribute__((aligned(256))) = {{", fmt(w), "};",
                f"static const float nx_srl_{tag}_{name}_b[{b.size}] = {{", fmt(b), "};", ""]
        rows.append(f'    {{ nx_srl_{tag}_{name}_w, nx_srl_{tag}_{name}_b, {w.shape[2]}, {w.shape[3]}, {act}, {res}, "{kern}" }},')
    out += [f"#define NX_SRL_{tag.upper()}_LAYERS {len(names)}",
            f"static const nx_srl_layer_t nx_srl_{tag}_layers[NX_SRL_{tag.upper()}_LAYERS] = {{"] + rows + ["};", ""]


def main():
    lite, hq, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    out = ["// Generado por sr/export_srl.py: no editar a mano.",
           "// Pesos de las guias de NXSR-L (sr/net_srl.py): Ligero y Calidad. Layout HWIO: w[ky][kx][cin][cout].",
           "#pragma once", "",
           "typedef struct { const float* w; const float* b; int cin, cout, act, residual; const char* kernel; } nx_srl_layer_t;",
           ""]
    model(out, "lite", pickle.load(open(lite, "rb")))
    model(out, "hq", pickle.load(open(hq, "rb")))
    open(dst, "w").write("\n".join(out) + "\n")
    print(dst)


if __name__ == "__main__":
    main()
