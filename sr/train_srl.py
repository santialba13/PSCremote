"""Entrena NXSR-L (sr/net_srl.py). Uso:

  python sr/train_srl.py --data train_sr.npz --val val_sr.npz --steps 8000 --out weights/nxsr_l.pkl
"""
import argparse
import functools
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import net_srl  # noqa: E402


def charb(x):
    return jnp.sqrt(x * x + 1e-6)


SHIFT_W = 0.0
GRAD_W = 0.5    # peso del error de gradientes (mas alto = bordes y texturas mas definidos)   # peso de la perdida de consistencia ante desplazamientos (--shift)


def loss_fn(p, lr, hr, dy, dx):
    out = net_srl.apply(p, lr)
    gx = lambda a: a[:, :, 1:] - a[:, :, :-1]
    gy = lambda a: a[:, 1:] - a[:, :-1]
    l = charb(out - hr).mean() + GRAD_W * (charb(gx(out) - gx(hr)).mean() + charb(gy(out) - gy(hr)).mean())
    return l


def psnr(a, b):
    return float(10 * np.log10(1 / np.mean((np.clip(a, 0, 1) - b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--C", type=int, default=12)
    ap.add_argument("--mid", type=int, default=2)
    ap.add_argument("--feat", default="full", choices=["full", "half", "sub"])
    ap.add_argument("--bands", type=int, default=2, choices=[2, 3])
    ap.add_argument("--gradw", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    net_srl.MARGIN = a.margin
    net_srl.FEAT = a.feat
    net_srl.BANDS = a.bands
    global GRAD_W
    GRAD_W = a.gradw
    d = np.load(a.data)
    HR, LR = d["hr"], d["lr"]
    v = np.load(a.val)
    vhr, vlr = v["hr"][:256].astype(np.float32) / 255, v["lr"][:256].astype(np.float32) / 255
    p = net_srl.init(jax.random.PRNGKey(a.seed), a.C, a.mid)
    if a.init:
        with open(a.init, "rb") as f:
            p = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
    print(f"MAC/px de la red {net_srl.macs_per_pixel(a.C, a.mid):.0f}, datos {HR.shape}", flush=True)
    sched = optax.warmup_cosine_decay_schedule(0, a.lr, min(300, a.steps // 4), a.steps, a.lr * 0.02)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    st = opt.init(p)

    @functools.partial(jax.jit, static_argnums=(4, 5))
    def step(p, st, lr, hr, dy, dx, scale):
        l, g = jax.value_and_grad(loss_fn)(p, lr, hr, dy, dx)
        u, st = opt.update(g, st, p)
        u = jax.tree_util.tree_map(lambda v: v * scale, u)
        return optax.apply_updates(p, u), st, l

    infer = jax.jit(lambda p, x: net_srl.apply(p, x))
    evaluate = lambda p: psnr(np.concatenate([np.asarray(infer(p, vlr[i:i + 32])) for i in range(0, len(vlr), 32)]), vhr)

    def shift_err(p):   # cambio medio (en /255) al mover la imagen 1 pixel en diagonal
        x = vlr[:64]
        A = np.asarray(infer(p, x[:, :-8, :-8]))[:, 1:-8, 1:-8]
        B = np.asarray(infer(p, x[:, 1:-7, 1:-7]))[:, :-9, :-9]
        return float(np.abs(A - B).mean() * 255)
    print(f"val sin procesar {psnr(vlr, vhr):.2f} dB", flush=True)
    rng = np.random.default_rng(a.seed)
    best, t0 = -1, time.time()
    best_p, scale = p, 1.0
    for it in range(1, a.steps + 1):
        idx = rng.integers(0, len(HR), a.batch)
        lr, hr = LR[idx].astype(np.float32) / 255, HR[idx].astype(np.float32) / 255
        if rng.random() < 0.5:  # permutacion de canales: no depende de RGB/BGR
            perm = rng.permutation(3)
            lr, hr = lr[..., perm], hr[..., perm]
        dy, dx = [(0, 1), (1, 0), (1, 1)][it % 3]
        p, st, l = step(p, st, lr, hr, dy, dx, scale)
        if it % 500 == 0 or it == a.steps:
            vp = evaluate(p)
            print(f"it {it} loss {float(l):.4f} val {vp:.2f} dB desplaz {shift_err(p):.3f} {time.time() - t0:.0f}s", flush=True)
            if best > 0 and (vp < best - 1.0 or not np.isfinite(vp)):
                # divergio: se vuelve al mejor punto con la mitad de paso
                scale *= 0.5
                p, st = best_p, opt.init(best_p)
                print(f"  divergencia: vuelvo al mejor ({best:.2f} dB), paso x{scale}", flush=True)
                continue
            if vp > best:
                best, best_p = vp, p
                with open(a.out, "wb") as f:
                    pickle.dump(jax.tree_util.tree_map(np.asarray, p), f)
    print(f"mejor val {best:.2f} dB -> {a.out}")


if __name__ == "__main__":
    main()
