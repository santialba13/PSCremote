"""Compara metodos de generacion de fotogramas sobre tripletes de validacion.

  python model/eval.py --weights weights/nxflow.pkl --sets val_video.npy val_synth.npy \
      --video bbb_test.mp4 --out docs/resultados.md
"""
import argparse
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_bm  # noqa: E402
import data  # noqa: E402
import net  # noqa: E402


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(np.mean((np.clip(a, 0, 1) - b) ** 2), 1e-10)))


def ssim(a, b):
    """SSIM global en luma con ventana gaussiana 7x7 (suficiente para comparar)."""
    from numpy.lib.stride_tricks import sliding_window_view as sw
    a = baseline_bm.luma(np.clip(a, 0, 1)); b = baseline_bm.luma(b)
    g = np.exp(-((np.arange(7) - 3) ** 2) / (2 * 1.5 ** 2)); g = np.outer(g, g); g /= g.sum()
    f = lambda x: (sw(x, (7, 7)) * g).sum((-1, -2))
    mu_a, mu_b = f(a), f(b)
    va = f(a * a) - mu_a ** 2; vb = f(b * b) - mu_b ** 2; cov = f(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


def pad32(x):
    H, W = x.shape[:2]
    ph, pw = (-H) % 32, (-W) % 32
    return np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="edge"), H, W


def make_methods(params):
    infer = jax.jit(lambda p, a, b, bm: net.interpolate(p, a, b, bm)[0])

    def nx(i0, i1):
        a, H, W = pad32(i0); b, _, _ = pad32(i1)
        bm = baseline_bm.block_vectors(a, b)
        return np.asarray(infer(params, a[None], b[None], bm[None]))[0][:H, :W]

    return {
        "repetir fotograma": lambda i0, i1: i0,
        "mezcla 50/50": lambda i0, i1: 0.5 * (i0 + i1),
        "block-matching (tipo VCFG)": baseline_bm.interpolate,
        "NXFlow hibrido (nuevo)": nx,
    }


def run_set(methods, tris):
    res = {}
    for name, fn in methods.items():
        ps, ss = [], []
        t0 = time.time()
        for t in tris:
            o = fn(t[0], t[2])
            ps.append(psnr(o, t[1])); ss.append(ssim(o, t[1]))
        res[name] = (np.mean(ps), np.mean(ss), (time.time() - t0) / len(tris))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--video", help="video de prueba a resolucion completa")
    ap.add_argument("--out")
    a = ap.parse_args()
    with open(a.weights, "rb") as f:
        params = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
    methods = make_methods(params)
    lines = ["| conjunto | metodo | PSNR (dB) | SSIM |", "|---|---|---|---|"]
    sets = [(os.path.basename(s), np.load(s).astype(np.float32)) for s in a.sets]
    if a.video:
        fr = data.decode_video(a.video).astype(np.float32) / 255.0
        sets.append((os.path.basename(a.video) + " (completo)",
                     [fr[i:i + 3] for i in range(0, len(fr) - 2, 3)]))
    for sname, tris in sets:
        for m, (p, s, t) in run_set(methods, tris).items():
            lines.append(f"| {sname} | {m} | {p:.2f} | {s:.4f} |")
            print(lines[-1], flush=True)
    if a.out:
        with open(a.out, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
