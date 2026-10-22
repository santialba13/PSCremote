"""Exporta NXRE (sr/net_re.py) a plugin/include/nxre_weights.h.

  python sr/export_re.py weights/nxre.pkl plugin/include/nxre_weights.h
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import net_re  # noqa: E402


def fmt(a):
    f = np.asarray(a, np.float32).ravel()
    return "\n".join("    " + ", ".join(f"{v: .8e}f" for v in f[i:i + 8]) + "," for i in range(0, len(f), 8))


def main():
    p = pickle.load(open(sys.argv[1], "rb"))
    out = ["// Generado por sr/export_re.py: no editar a mano.",
           "// Pesos de NXRE (sr/net_re.py). Layout HWIO: w[ky][kx][cin][cout].",
           "#pragma once", "",
           "typedef struct { const float* w; const float* b; int cin, cout, stride, act; const char* kernel; } nx_re_layer_t;",
           f"#define NX_RE_D {net_re.D}", f"#define NX_RE_LOW_W {net_re.LOW_W}", ""]
    rows = []
    for name, ci, co, s, act in net_re.LAYERS:
        w, b = np.asarray(p[name]["w"]), np.asarray(p[name]["b"])
        assert w.shape == (3, 3, ci, co)
        out += [f"static const float nx_re_{name}_w[{w.size}] __attribute__((aligned(256))) = {{", fmt(w), "};",
                f"static const float nx_re_{name}_b[{b.size}] = {{", fmt(b), "};", ""]
        rows.append(f'    {{ nx_re_{name}_w, nx_re_{name}_b, {ci}, {co}, {s}, {int(act)}, "conv3x3_{ci}_{co}" }},')
    names = [l[0] for l in net_re.LAYERS]
    out += [f"// orden: {', '.join(names)}", f"#define NX_RE_LAYERS {len(names)}",
            "static const nx_re_layer_t nx_re_layers[NX_RE_LAYERS] = {"] + rows + ["};", ""]
    m = np.asarray(p["guide"]["m"], np.float32)
    c = float(np.asarray(p["guide"]["c"]))
    out += [f"static const float nx_re_guide[4] = {{ {m[0]:.8e}f, {m[1]:.8e}f, {m[2]:.8e}f, {c:.8e}f }};", ""]
    open(sys.argv[2], "w").write("\n".join(out) + "\n")
    print(sys.argv[2])


if __name__ == "__main__":
    main()
