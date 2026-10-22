"""Entrena NXRE (sr/net_re.py) con pares (fotograma del juego, version fotorrealista).

  python sr/train_re.py --data re_pairs.npz --steps 6000 --out weights/nxre.pkl

re_pairs.npz: src y dst (N x H x W x 3, uint8) alineados; los ultimos --val_frac
(en orden de tiempo) quedan para validacion. Ver sr/make_re_pairs.py.
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
import net_re  # noqa: E402

TV_W = 0.02    # suavidad de la malla (estabilidad temporal)


def charb(x):
    return jnp.sqrt(x * x + 1e-6)


def loss_fn(p, x, y):
    G = net_re.grid(p, x)
    out = net_re.slice_apply(G, net_re.guide(p, x), x)
    l = charb(out - y).mean()
    tv = jnp.abs(G[:, 1:] - G[:, :-1]).mean() + jnp.abs(G[:, :, 1:] - G[:, :, :-1]).mean()
    return l + TV_W * tv


def psnr(a, b):
    return float(10 * np.log10(1 / np.mean((np.clip(a, 0, 1) - b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = np.load(a.data)
    S, T = d["src"], d["dst"]
    nv = int(len(S) * a.val_frac)
    S, T, vS, vT = S[:-nv], T[:-nv], S[-nv:][::4], T[-nv:][::4]
    vx, vy = vS.astype(np.float32) / 255, vT.astype(np.float32) / 255
    print(f"datos {S.shape}, validacion {vS.shape}", flush=True)
    p = net_re.init(jax.random.PRNGKey(a.seed))
    sched = optax.warmup_cosine_decay_schedule(0, a.lr, min(300, a.steps // 4), a.steps, a.lr * 0.02)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    st = opt.init(p)

    @jax.jit
    def step(p, st, x, y):
        l, g = jax.value_and_grad(loss_fn)(p, x, y)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    infer = jax.jit(lambda p, x: net_re.apply(p, x))
    evaluate = lambda p: psnr(np.concatenate([np.asarray(infer(p, vx[i:i + 16])) for i in range(0, len(vx), 16)]), vy)
    print(f"val sin procesar {psnr(vx, vy):.2f} dB", flush=True)
    rng = np.random.default_rng(a.seed)
    best, t0 = -1, time.time()
    for it in range(1, a.steps + 1):
        idx = rng.integers(0, len(S), a.batch)
        x, y = S[idx].astype(np.float32) / 255, T[idx].astype(np.float32) / 255
        if rng.random() < 0.5:
            x, y = x[:, :, ::-1], y[:, :, ::-1]
        # exposicion: la misma escena mas oscura o mas clara (noche, interiores)
        k = rng.uniform(0.35, 1.15, (a.batch, 1, 1, 1)).astype(np.float32)
        x = np.clip(x ** 2.2 * k, 0, 1) ** (1 / 2.2)
        y = np.clip(y ** 2.2 * k, 0, 1) ** (1 / 2.2)
        p, st, l = step(p, st, x, y)
        if it % 250 == 0 or it == a.steps:
            vp = evaluate(p)
            print(f"it {it} loss {float(l):.4f} val {vp:.2f} dB {time.time() - t0:.0f}s", flush=True)
            if vp > best:
                best = vp
                with open(a.out, "wb") as f:
                    pickle.dump(jax.tree_util.tree_map(np.asarray, p), f)
    print(f"mejor val {best:.2f} dB -> {a.out}")


if __name__ == "__main__":
    main()
