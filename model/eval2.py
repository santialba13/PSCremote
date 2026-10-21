"""Compara la generacion de fotogramas de NXFG-2 (2.73) con la anterior y con el block-matching tipo VCFG.

  python model/eval2.py --nxfg2 weights/nxfg2.pkl --nxflow weights/nxflow.pkl \
      --sets val_video.npy val_synth.npy --video bbb_test.mp4 --out docs/resultados_fg2.md

Metricas por metodo: PSNR y SSIM contra el fotograma real intermedio, y el error
medio (en niveles de 0-255) en los pixeles quietos (iguales en los tres fotogramas:
HUD, textos, fondo fijo), que es donde un generador "tiembla" o deja fantasmas.
"""
import argparse
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_bm  # noqa: E402
import bm2  # noqa: E402
import data  # noqa: E402
import net  # noqa: E402
import net2  # noqa: E402
from eval import psnr, ssim, pad32  # noqa: E402


def static_err(o, t):
    q = lambda x: np.round(np.clip(x, 0, 1) * 255)
    m = np.all(np.abs(q(t[0]) - q(t[2])) < 1, -1) & np.all(np.abs(q(t[0]) - q(t[1])) < 1, -1)
    if m.sum() < 16:
        return None
    return float(np.abs(q(o) - q(t[1]))[m].mean())


def make_methods(p2, p1):
    inf2 = jax.jit(lambda p, a, b, bm: net2.interpolate(p, a, b, bm)[0])
    inf1 = jax.jit(lambda p, a, b, bm: net.interpolate(p, a, b, bm)[0])
    zero = jax.tree_util.tree_map(lambda x: x * 0, p2)

    def bm_of(a, b):
        return np.concatenate(bm2.motion(bm2.pyramid(a), bm2.pyramid(b)), -1)

    def nx2(params):
        def f(i0, i1):
            a, H, W = pad32(i0); b, _, _ = pad32(i1)
            return np.asarray(inf2(params, a[None], b[None], bm_of(a, b)[None]))[0][:H, :W]
        return f

    def nx1(i0, i1):
        a, H, W = pad32(i0); b, _, _ = pad32(i1)
        return np.asarray(inf1(p1, a[None], b[None], baseline_bm.block_vectors(a, b)[None]))[0][:H, :W]

    m = {
        "mezcla 50/50": lambda i0, i1: 0.5 * (i0 + i1),
        "block-matching tipo VCFG": baseline_bm.interpolate,
    }
    if p1 is not None:
        m["NXFlow (IA de la 2.72)"] = nx1
    # sin la red (todas las salidas en 0): movimiento de 4 niveles + eleccion por pixel
    # + protecciones; es lo que hace la generacion sin red en el camino propio
    m["NXFG-2 sin red"] = nx2(zero)
    m["NXFG-2 IA (2.73)"] = nx2(p2)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nxfg2", required=True)
    ap.add_argument("--nxflow")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--video")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    load = lambda p: jax.tree_util.tree_map(jnp.asarray, pickle.load(open(p, "rb")))
    methods = make_methods(load(a.nxfg2), load(a.nxflow) if a.nxflow else None)
    sets = []
    for s in a.sets:
        d = np.load(s).astype(np.float32)
        sets.append((os.path.basename(s), d[:a.limit] if a.limit else d))
    if a.video:
        fr = data.decode_video(a.video).astype(np.float32) / 255.0
        sets.append((os.path.basename(a.video) + " (completo)", [fr[i:i + 3] for i in range(0, len(fr) - 2, 3)]))
    lines = ["| conjunto | metodo | PSNR (dB) | SSIM | error en pixeles quietos (0-255) |", "|---|---|---|---|---|"]
    for sname, tris in sets:
        for name, fn in methods.items():
            ps, ss, se = [], [], []
            for t in tris:
                o = np.clip(fn(t[0], t[2]), 0, 1)
                ps.append(psnr(o, t[1])); ss.append(ssim(o, t[1]))
                e = static_err(o, t)
                if e is not None:
                    se.append(e)
            st = f"{np.mean(se):.2f}" if se else "-"
            lines.append(f"| {sname} | {name} | {np.mean(ps):.2f} | {np.mean(ss):.4f} | {st} |")
            print(lines[-1], flush=True)
    if a.out:
        with open(a.out, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
