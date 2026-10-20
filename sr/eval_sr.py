"""Evalua realce/reescalado sobre pares (degradado, limpio).

Ademas de PSNR/SSIM mide justamente lo que se le critica a GNMScale:
  * grano: PSNR solo en zonas planas del objetivo (desvio local < 2/255);
    si un metodo amplifica ruido, ahi cae por debajo de "sin procesar";
  * halos: % de pixeles que se salen del rango local (3x3) del objetivo
    por mas de 4/255 (sobreimpulsos alrededor de bordes).

  python sr/eval_sr.py --data val_sr.npz --nxsr weights/nxsr.pkl --gnmscale DIR_MODELOS --out tabla.md
"""
import argparse
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "upscale"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsr1  # noqa: E402
import net_sr  # noqa: E402


def local_std(x, r=1):
    p = np.pad(x, ((0, 0), (r, r), (r, r), (0, 0)), mode="edge")
    H, W = x.shape[1:3]
    st = np.stack([p[:, i:i + H, j:j + W] for i in range(2 * r + 1) for j in range(2 * r + 1)])
    return st.std(0).max(-1), st.min(0), st.max(0)


def ssim_y(a, b):
    from numpy.lib.stride_tricks import sliding_window_view as sw
    y = lambda x: x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114
    g = np.exp(-((np.arange(7) - 3) ** 2) / (2 * 1.5 ** 2)); g = np.outer(g, g); g /= g.sum()
    vals = []
    for A, B in zip(y(a), y(b)):
        f = lambda x: (sw(x, (7, 7)) * g).sum((-1, -2))
        ma, mb = f(A), f(B)
        va, vb, cov = f(A * A) - ma ** 2, f(B * B) - mb ** 2, f(A * B) - ma * mb
        s = ((2 * ma * mb + 1e-4) * (2 * cov + 9e-4)) / ((ma ** 2 + mb ** 2 + 1e-4) * (va + vb + 9e-4))
        vals.append(s.mean())
    return float(np.mean(vals))


def metrics(out, hr, flat, lo, hi):
    out = np.clip(out, 0, 1)
    mse = np.mean((out - hr) ** 2)
    mse_flat = np.sum(((out - hr) ** 2).mean(-1) * flat) / max(flat.sum(), 1)
    halo = np.mean(((out < lo - 4 / 255) | (out > hi + 4 / 255)).any(-1))
    return (10 * np.log10(1 / mse), ssim_y(out, hr), 10 * np.log10(1 / max(mse_flat, 1e-12)), 100 * halo)


def run(fn, lr, bs=32):
    return np.concatenate([np.asarray(fn(jnp.asarray(lr[i:i + bs]))) for i in range(0, len(lr), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--nxsr", nargs="*", default=[])
    ap.add_argument("--gnmscale", help="carpeta models/ de GNMScale (para compararlos)")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = np.load(a.data)
    hr, lr = d["hr"].astype(np.float32) / 255, d["lr"].astype(np.float32) / 255
    sd, lo, hi = local_std(hr)
    flat = sd < 2 / 255
    methods = {"sin procesar": lambda x: x,
               "FSR1 RCAS (0.25)": lambda x: jnp.asarray(np.stack([fsr1.rcas(np.asarray(i), 0.25) for i in x]))}
    if a.gnmscale:
        for name in sorted(os.listdir(a.gnmscale)):
            md = os.path.join(a.gnmscale, name, "fp16")
            if os.path.exists(os.path.join(md, "weights.bin")):
                p = net_sr.load_gnmscale(md)
                methods[f"GNMScale {name}"] = jax.jit(lambda x, p=p: net_sr.gnmscale_apply(p, x))
    for w in a.nxsr:
        with open(w, "rb") as f:
            p = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
        tag = os.path.basename(w).replace(".pkl", "")
        methods[f"{tag} ({net_sr.macs_per_pixel(p):.0f} MAC/px)"] = jax.jit(lambda x, p=p: net_sr.apply(p, x))
    lines = ["| metodo | PSNR | SSIM | PSNR zonas planas (grano) | halos % |", "|---|---|---|---|---|"]
    for name, fn in methods.items():
        m = metrics(run(fn, lr), hr, flat, lo, hi)
        lines.append(f"| {name} | {m[0]:.2f} | {m[1]:.4f} | {m[2]:.2f} | {m[3]:.2f} |")
        print(lines[-1], flush=True)
    if a.out:
        open(a.out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
