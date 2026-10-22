"""Compara NXFG-3 con NXFG-2 y con el block-matching tipo VCFG, en GTA V y en los
conjuntos de siempre.

  python model/eval3.py --nxfg2 weights/nxfg2.pkl --gta gta.npy --sets val_video.npy val_synth.npy \
      --out docs/resultados_fg3.md

gta.npy: fotogramas de GTA V a 30 fps (N, H, W, 3) uint8; se toman tripletes del 30 %
final (i, i+1, i+2) y se genera el del medio a partir de los otros dos (el doble de
movimiento que en el uso real, 30 -> 60). Metricas: PSNR, SSIM, fallas (% de pixeles
con error medio > 48/255: lo que se ve como bloques rotos o desgarros) y el error en
pixeles quietos (HUD, textos).
"""
import argparse
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bm2  # noqa: E402
import eval2  # noqa: E402
import net3  # noqa: E402
from eval import psnr, ssim, pad32  # noqa: E402


def fallas(o, t):
    return float((np.abs(o - t).mean(-1) > 48 / 255).mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nxfg2", required=True)
    ap.add_argument("--gta")
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args()
    p2 = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(a.nxfg2, "rb")))
    methods = eval2.make_methods(p2, None)
    del methods["mezcla 50/50"]
    inf3 = jax.jit(lambda i0, i1, bm: net3.interpolate(i0, i1, bm))

    def nx3(i0, i1):
        x, H, W = pad32(i0); y, _, _ = pad32(i1)
        bm = np.concatenate(bm2.motion(bm2.pyramid(x), bm2.pyramid(y)), -1)
        if bm[..., 2].mean() > net3.SCENE_CUT:
            return i1
        return np.asarray(inf3(x[None], y[None], bm[None]))[0][:H, :W]
    methods["NXFG-3 (2.77)"] = nx3
    sets = []
    if a.gta:
        g = np.load(a.gta, mmap_mode="r")
        v0 = int(len(g) * 0.7)
        tr = [g[i:i + 3].astype(np.float32) / 255 for i in range(v0, len(g) - 2, a.step)]
        sets.append(("GTA V", [t for t in tr if np.abs(t[0] - t[2]).mean() < 0.12]))
    for s in a.sets:
        sets.append((os.path.basename(s), np.load(s).astype(np.float32)))
    lines = ["| conjunto | metodo | PSNR (dB) | SSIM | fallas (%) | error en pixeles quietos (0-255) |",
             "|---|---|---|---|---|---|"]
    for sname, tris in sets:
        for name, fn in methods.items():
            ps, ss, fl, se = [], [], [], []
            for t in tris:
                o = np.clip(fn(t[0], t[2]), 0, 1)
                ps.append(psnr(o, t[1])); ss.append(ssim(o, t[1])); fl.append(fallas(o, t[1]))
                e = eval2.static_err(o, t)
                if e is not None:
                    se.append(e)
            st = f"{np.mean(se):.2f}" if se else "-"
            lines.append(f"| {sname} | {name} | {np.mean(ps):.2f} | {np.mean(ss):.4f} | {np.mean(fl):.2f} | {st} |")
            print(lines[-1], flush=True)
    if a.out:
        with open(a.out, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
