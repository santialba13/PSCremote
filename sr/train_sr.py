"""Entrena NXSR. Uso:

  python sr/train_sr.py --data train_sr.npz --val val_sr.npz --C 16 --body 1 --steps 8000 --out weights/nxsr.pkl
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
import net_sr  # noqa: E402


def charb(x):
    return jnp.sqrt(x * x + 1e-6)


SHIFT_W = 0.0   # peso de la perdida de consistencia ante desplazamientos (--shift)


def loss_fn(p, lr, hr, dy, dx):
    out = net_sr.apply(p, lr)
    gx = lambda a: a[:, :, 1:] - a[:, :, :-1]
    gy = lambda a: a[:, 1:] - a[:, :-1]
    l = charb(out - hr).mean() + 0.5 * (charb(gx(out) - gx(hr)).mean() + charb(gy(out) - gy(hr)).mean())
    if SHIFT_W:
        a, b = net_sr.shift_pair(lambda z: net_sr.apply(p, z), lr, dy, dx)
        l = l + SHIFT_W * charb(a - b).mean()
    return l


def psnr(a, b):
    return float(10 * np.log10(1 / np.mean((np.clip(a, 0, 1) - b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--body", type=int, default=1)
    ap.add_argument("--s", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--nolimit", action="store_true")
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--rep", action="store_true", help="ramas 1x1/identidad plegables (fold)")
    ap.add_argument("--shift", type=float, default=0.0, help="peso de consistencia ante desplazamientos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    net_sr.MARGIN = a.margin
    net_sr.WINDOW = a.window
    global SHIFT_W
    SHIFT_W = a.shift
    if a.nolimit:
        net_sr.MARGIN = 1e6
    d = np.load(a.data)
    HR, LR = d["hr"], d["lr"]
    v = np.load(a.val)
    vhr, vlr = v["hr"][:256].astype(np.float32) / 255, v["lr"][:256].astype(np.float32) / 255
    p = net_sr.init(jax.random.PRNGKey(a.seed), a.C, a.body, a.s, a.rep)
    if a.init:
        with open(a.init, "rb") as f:
            p = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
    print(f"MAC/px {net_sr.macs_per_pixel(p):.0f}, datos {HR.shape}", flush=True)
    sched = optax.warmup_cosine_decay_schedule(0, a.lr, min(300, a.steps // 4), a.steps, a.lr * 0.02)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    st = opt.init(p)

    @functools.partial(jax.jit, static_argnums=(4, 5))
    def step(p, st, lr, hr, dy, dx):
        l, g = jax.value_and_grad(loss_fn)(p, lr, hr, dy, dx)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    infer = jax.jit(lambda p, x: net_sr.apply(p, x))
    evaluate = lambda p: psnr(np.concatenate([np.asarray(infer(p, vlr[i:i + 32])) for i in range(0, len(vlr), 32)]), vhr)

    def shift_err(p):   # cambio medio (en /255) al mover la imagen 1 pixel en diagonal
        a, b = net_sr.shift_pair(lambda z: infer(p, z), vlr[:64], 1, 1)
        return float(np.abs(np.asarray(a) - np.asarray(b)).mean() * 255)
    print(f"val sin procesar {psnr(vlr, vhr):.2f} dB", flush=True)
    rng = np.random.default_rng(a.seed)
    best, t0 = -1, time.time()
    for it in range(1, a.steps + 1):
        idx = rng.integers(0, len(HR), a.batch)
        lr, hr = LR[idx].astype(np.float32) / 255, HR[idx].astype(np.float32) / 255
        if rng.random() < 0.5:  # permutacion de canales: no depende de RGB/BGR
            perm = rng.permutation(3)
            lr, hr = lr[..., perm], hr[..., perm]
        dy, dx = [(0, 1), (1, 0), (1, 1)][it % 3]
        p, st, l = step(p, st, lr, hr, dy, dx)
        if it % 500 == 0 or it == a.steps:
            vp = evaluate(p)
            print(f"it {it} loss {float(l):.4f} val {vp:.2f} dB desplaz {shift_err(p):.3f} {time.time() - t0:.0f}s", flush=True)
            if vp > best:
                best = vp
                with open(a.out, "wb") as f:
                    pickle.dump(jax.tree_util.tree_map(np.asarray, p), f)
    print(f"mejor val {best:.2f} dB -> {a.out}")


if __name__ == "__main__":
    main()
