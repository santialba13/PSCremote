"""Entrena NXSR. Uso:

  python sr/train_sr.py --data train_sr.npz --val val_sr.npz --C 16 --body 1 --steps 8000 --out weights/nxsr.pkl
"""
import argparse
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


def loss_fn(p, lr, hr):
    out = net_sr.apply(p, lr)
    gx = lambda a: a[:, :, 1:] - a[:, :, :-1]
    gy = lambda a: a[:, 1:] - a[:, :-1]
    return charb(out - hr).mean() + 0.5 * (charb(gx(out) - gx(hr)).mean() + charb(gy(out) - gy(hr)).mean())


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
    ap.add_argument("--init")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    net_sr.MARGIN = a.margin
    net_sr.WINDOW = a.window
    if a.nolimit:
        net_sr.MARGIN = 1e6
    d = np.load(a.data)
    HR, LR = d["hr"], d["lr"]
    v = np.load(a.val)
    vhr, vlr = v["hr"][:256].astype(np.float32) / 255, v["lr"][:256].astype(np.float32) / 255
    p = net_sr.init(jax.random.PRNGKey(0), a.C, a.body, a.s)
    if a.init:
        with open(a.init, "rb") as f:
            p = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
    print(f"MAC/px {net_sr.macs_per_pixel(p):.0f}, datos {HR.shape}", flush=True)
    sched = optax.warmup_cosine_decay_schedule(0, a.lr, 300, a.steps, a.lr * 0.02)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    st = opt.init(p)

    @jax.jit
    def step(p, st, lr, hr):
        l, g = jax.value_and_grad(loss_fn)(p, lr, hr)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    infer = jax.jit(lambda p, x: net_sr.apply(p, x))
    evaluate = lambda p: psnr(np.concatenate([np.asarray(infer(p, vlr[i:i + 32])) for i in range(0, len(vlr), 32)]), vhr)
    print(f"val sin procesar {psnr(vlr, vhr):.2f} dB", flush=True)
    rng = np.random.default_rng(0)
    best, t0 = -1, time.time()
    for it in range(1, a.steps + 1):
        idx = rng.integers(0, len(HR), a.batch)
        lr, hr = LR[idx].astype(np.float32) / 255, HR[idx].astype(np.float32) / 255
        if rng.random() < 0.5:  # permutacion de canales: no depende de RGB/BGR
            perm = rng.permutation(3)
            lr, hr = lr[..., perm], hr[..., perm]
        p, st, l = step(p, st, lr, hr)
        if it % 500 == 0 or it == a.steps:
            vp = evaluate(p)
            print(f"it {it} loss {float(l):.4f} val {vp:.2f} dB {time.time() - t0:.0f}s", flush=True)
            if vp > best:
                best = vp
                with open(a.out, "wb") as f:
                    pickle.dump(jax.tree_util.tree_map(np.asarray, p), f)
    print(f"mejor val {best:.2f} dB -> {a.out}")


if __name__ == "__main__":
    main()
