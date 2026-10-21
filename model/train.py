"""Entrena NXFlow. Uso:

  python model/train.py --data train.npy --val val.npy --steps 6000 --out weights/nxflow.npz
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
import net  # noqa: E402
import net2  # noqa: E402

NET = net
BM_SUFFIX = "_bm.npy"


MASK_W = 0.02


def charb(x):
    return jnp.sqrt(x * x + 1e-6)


def loss_fn(params, i0, gt, i1, bm):
    out, (F0, F1, M, w0, w1) = NET.interpolate(params, i0, i1, bm)
    l_pix = charb(out - gt).mean()
    # supervision privilegiada de la mascara: donde un warp es claramente mejor
    # que el otro respecto del fotograma real, la mascara deberia elegirlo
    e0 = jax.lax.stop_gradient(jnp.abs(w0 - gt).sum(-1, keepdims=True))
    e1 = jax.lax.stop_gradient(jnp.abs(w1 - gt).sum(-1, keepdims=True))
    target = (e0 < e1).astype(jnp.float32)
    wgt = (jnp.abs(e0 - e1) > 0.1).astype(jnp.float32)
    Mc = jnp.clip(M, 1e-4, 1 - 1e-4)
    l_mask = (wgt * -(target * jnp.log(Mc) + (1 - target) * jnp.log(1 - Mc))).sum() / (wgt.sum() + 1.0)
    gx = lambda a: a[:, :, 1:] - a[:, :, :-1]
    gy = lambda a: a[:, 1:] - a[:, :-1]
    l_grad = charb(gx(out) - gx(gt)).mean() + charb(gy(out) - gy(gt)).mean()
    # suavidad leve del flujo a baja resolucion: evita "jitter" de bloques
    l_smooth = (jnp.abs(gx(F0)).mean() + jnp.abs(gy(F0)).mean()
                + jnp.abs(gx(F1)).mean() + jnp.abs(gy(F1)).mean()) / 64.0
    return l_pix + 0.5 * l_grad + 0.01 * l_smooth + MASK_W * l_mask, l_pix


def augment(rng, b, bm):
    """b: (N,3,H,W,3); bm: (N,h,w,2) vectores (dy,dx). Las transformaciones se aplican a ambos."""
    if rng.random() < 0.5:
        b = b[:, :, :, ::-1]
        bm = bm[:, :, ::-1] * np.array([1, -1, 1][:bm.shape[-1]], np.float32)
    if rng.random() < 0.3:
        b = b[:, :, ::-1]
        bm = bm[:, ::-1] * np.array([-1, 1, 1][:bm.shape[-1]], np.float32)
    if rng.random() < 0.5:
        b = b[:, ::-1]  # invertir tiempo
        bm = bm * np.array([-1, -1, 1][:bm.shape[-1]], np.float32)
    if rng.random() < 0.5:
        b = b[..., rng.permutation(3)]
    return np.ascontiguousarray(b), np.ascontiguousarray(bm)


def psnr(a, b):
    mse = np.mean((np.clip(a, 0, 1) - b) ** 2, axis=(1, 2, 3))
    return float(np.mean(10 * np.log10(1.0 / np.maximum(mse, 1e-10))))


def main():
    global MASK_W
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", help="checkpoint para arrancar (los bloques que coincidan)")
    ap.add_argument("--mask-weight", type=float, default=0.02)
    ap.add_argument("--arch", default="nxflow", choices=["nxflow", "nxfg2"])
    a = ap.parse_args()
    MASK_W = a.mask_weight
    global NET, BM_SUFFIX
    if a.arch == "nxfg2":
        NET, BM_SUFFIX = net2, "_bm2.npy"

    data = np.load(a.data)
    data_bm = np.load(a.data.replace(".npy", BM_SUFFIX))
    val = np.load(a.val).astype(np.float32)
    val_bm = np.load(a.val.replace(".npy", BM_SUFFIX))
    rng = np.random.default_rng(a.seed)
    params = NET.init(jax.random.PRNGKey(a.seed))
    if a.init:
        with open(a.init, "rb") as f:
            old = pickle.load(f)
        params.update({k: jax.tree_util.tree_map(jnp.asarray, v) for k, v in old.items() if k in params})
        print("arranque desde", a.init, "bloques:", sorted(old), flush=True)
    print("parametros:", NET.count_params(params), "datos:", data.shape, flush=True)

    sched = optax.warmup_cosine_decay_schedule(0, a.lr, min(200, a.steps // 4), a.steps, a.lr * 0.02)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
    state = opt.init(params)

    @jax.jit
    def step(params, state, b, bm):
        (l, lp), g = jax.value_and_grad(loss_fn, has_aux=True)(params, b[:, 0], b[:, 1], b[:, 2], bm)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, l, lp

    @jax.jit
    def infer(params, i0, i1, bm):
        return NET.interpolate(params, i0, i1, bm)[0]

    def evaluate(params):
        outs = [np.asarray(infer(params, val[i:i + 16, 0], val[i:i + 16, 2], val_bm[i:i + 16]))
                for i in range(0, len(val), 16)]
        return psnr(np.concatenate(outs), val[:, 1])

    print("val PSNR blend 50/50: %.2f dB" % psnr((val[:, 0] + val[:, 2]) / 2, val[:, 1]))
    t0 = time.time()
    run = 0.0
    best = -1
    for it in range(1, a.steps + 1):
        idx = rng.integers(0, len(data), a.batch)
        b, bm = augment(rng, data[idx].astype(np.float32), data_bm[idx])
        params, state, l, lp = step(params, state, b, bm)
        run = 0.98 * run + 0.02 * float(lp) if it > 1 else float(lp)
        if it % 100 == 0:
            print(f"it {it} loss {float(l):.4f} l1 {run:.4f} {time.time() - t0:.0f}s", flush=True)
        if it % 1000 == 0 or it == a.steps:
            v = evaluate(params)
            print(f"  val PSNR {v:.2f} dB", flush=True)
            if v > best:
                best = v
                with open(a.out, "wb") as f:
                    pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
    print("mejor val PSNR %.2f dB -> %s" % (best, a.out))


if __name__ == "__main__":
    main()
