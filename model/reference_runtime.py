"""Runtime de referencia de NXFlow en NumPy puro, con la misma secuencia de
pases y la misma semantica que shaders/nxflow.cl (padding, resize, warp).

Sirve para:
  * verificar que la red exportada (weights/nxflow_weights.h) reproduce a JAX;
  * depurar el port a la GPU de PS4 comparando buffers pase a pase.
"""
import numpy as np


def same_pads(n, stride):
    """Padding 'SAME' de XLA para kernel 3: devuelve (antes, despues)."""
    out = -(-n // stride)
    total = max((out - 1) * stride + 3 - n, 0)
    return total // 2, total - total // 2


def conv3x3(x, w, b, stride=1, act=False, residual=None):
    H, W, C = x.shape
    pt, pb = same_pads(H, stride)
    pl, pr = same_pads(W, stride)
    p = np.pad(x, ((pt, pb), (pl, pr), (0, 0)))
    Ho, Wo = -(-H // stride), -(-W // stride)
    out = np.broadcast_to(b, (Ho, Wo, w.shape[-1])).astype(np.float32).copy()
    for ky in range(3):
        for kx in range(3):
            patch = p[ky:ky + (Ho - 1) * stride + 1:stride, kx:kx + (Wo - 1) * stride + 1:stride]
            out += patch @ w[ky, kx]
    if act:
        out = np.where(out > 0, out, 0.1 * out)
    if residual is not None:
        out += residual
    return out


def resize_bilinear(x, Ho, Wo, scale=1.0):
    H, W, _ = x.shape
    sy = (np.arange(Ho) + 0.5) * (H / Ho) - 0.5
    sx = (np.arange(Wo) + 0.5) * (W / Wo) - 0.5
    y0 = np.floor(sy).astype(int); x0 = np.floor(sx).astype(int)
    fy = (sy - y0)[:, None, None]; fx = (sx - x0)[None, :, None]
    c = lambda a, n: np.clip(a, 0, n - 1)
    Y0, Y1, X0, X1 = c(y0, H), c(y0 + 1, H), c(x0, W), c(x0 + 1, W)
    return scale * ((x[Y0][:, X0] * (1 - fx) + x[Y0][:, X1] * fx) * (1 - fy)
                    + (x[Y1][:, X0] * (1 - fx) + x[Y1][:, X1] * fx) * fy)


def warp(img, flow):
    H, W, _ = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    x = np.clip(xx + flow[..., 0], 0, W - 1); y = np.clip(yy + flow[..., 1], 0, H - 1)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
    fx = (x - x0)[..., None]; fy = (y - y0)[..., None]
    return (img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy) + \
           (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy


def block(p, x, down):
    h = conv3x3(x, p["c0"]["w"], p["c0"]["b"], 2, True)
    h = conv3x3(h, p["c1"]["w"], p["c1"]["b"], 2 if down == 4 else 1, True)
    r = h
    i = 0
    while f"m{i}" in p:
        last = f"m{i + 1}" not in p
        h = conv3x3(h, p[f"m{i}"]["w"], p[f"m{i}"]["b"], 1, True, r if last else None)
        i += 1
    h = resize_bilinear(h, h.shape[0] * down, h.shape[1] * down)
    return conv3x3(h, p["out"]["w"], p["out"]["b"])


def avgpool(x, f):
    H, W, C = x.shape
    return x.reshape(H // f, f, W // f, f, C).mean((1, 3))


def upsample5(x, s):
    u = resize_bilinear(x, x.shape[0] * s, x.shape[1] * s)
    u[..., :4] *= s
    return u


def refine(p, a0, a1, f0, f1, m):
    x = np.concatenate([a0, a1, warp(a0, f0), warp(a1, f1), f0, f1, m], -1)
    d = block(p, x, 2)
    return f0 + d[..., 0:2], f1 + d[..., 2:4], m + d[..., 4:5]


def interpolate(params, i0, i1, bm):
    """bm: vectores (dy, dx) por bloque de 8x8 (baseline_bm.block_vectors / kernel bm_search)."""
    a0, a1 = avgpool(i0, 8), avgpool(i1, 8)
    v = bm[..., ::-1] / 8.0
    f0, f1 = -v, v
    o = block(params["A"], np.concatenate([a0, a1, warp(a0, f0), warp(a1, f1), f1], -1), 2)
    f0, f1, m = f0 + o[..., 0:2], f1 + o[..., 2:4], o[..., 4:5]
    f0, f1, m = refine(params["B"], a0, a1, f0, f1, m)
    u = upsample5(np.concatenate([f0, f1, m], -1), 2)
    f0, f1, m = refine(params["C"], avgpool(i0, 4), avgpool(i1, 4), u[..., 0:2], u[..., 2:4], u[..., 4:5])
    low = np.concatenate([f0, f1, m], -1)
    H, W, _ = i0.shape
    up = resize_bilinear(low, H, W)
    F0, F1 = up[..., 0:2] * 4, up[..., 2:4] * 4
    M = 1 / (1 + np.exp(-up[..., 4:5]))
    return M * warp(i0, F0) + (1 - M) * warp(i1, F1)


# ------------------------------------------------------------------ NXFG-2
def static_weight(i0, i1, tau=0.03):
    d = np.abs(i0 - i1).mean(-1, keepdims=True)
    return np.clip(1.0 - d / tau, 0.0, 1.0)


def block2(p, x):
    """model/net2.py: block (c0 s2, c1, medias + residual, resize x2, salida)."""
    h = conv3x3(x, p["c0"]["w"], p["c0"]["b"], 2, True)
    h = conv3x3(h, p["c1"]["w"], p["c1"]["b"], 1, True)
    r = h
    i = 0
    while f"m{i}" in p:
        last = f"m{i + 1}" not in p
        h = conv3x3(h, p[f"m{i}"]["w"], p[f"m{i}"]["b"], 1, True, r if last else None)
        i += 1
    h = resize_bilinear(h, h.shape[0] * 2, h.shape[1] * 2)
    return conv3x3(h, p["out"]["w"], p["out"]["b"])


def interpolate2(params, i0, i1, bm=None, fast=False, scene_cut=0.12):
    """NXFG-2 completo (i0, i1 con alto/ancho multiplos de 32). bm: model/bm2.motion o None."""
    import bm2
    if bm is None:
        bm = np.concatenate(bm2.motion(bm2.pyramid(i0), bm2.pyramid(i1)), -1)
    H, W, _ = i0.shape
    if bm[..., 2].mean() > scene_cut:
        return i1.copy()
    v = bm[..., 1::-1] / 8.0
    f0, f1 = -v, v
    if fast:
        low = np.concatenate([f0, f1, np.zeros_like(f0[..., :1])], -1)
        s = 16
    else:
        a0, a1 = avgpool(i0, 16), avgpool(i1, 16)
        x = np.concatenate([a0, a1, warp(a0, f0), warp(a1, f1), f1, bm[..., 2:3] * 10.0], -1)
        o = block2(params["A"], x)
        f5 = np.concatenate([f0 + o[..., 0:2], f1 + o[..., 2:4], o[..., 4:5]], -1)
        u = upsample5(f5, 2)
        q0, q1 = avgpool(i0, 8), avgpool(i1, 8)
        g0, g1, m = u[..., 0:2], u[..., 2:4], u[..., 4:5]
        d = block2(params["B"], np.concatenate([q0, q1, warp(q0, g0), warp(q1, g1), g0, g1, m], -1))
        low = np.concatenate([g0 + d[..., 0:2], g1 + d[..., 2:4], m + d[..., 4:5]], -1)
        s = 8
    mix = cand_warp(i0, i1, low, s)
    st = static_weight(i0, i1, OPTS["static_tau"])
    return st * (0.5 * (i0 + i1)) + (1 - st) * mix


MASK_MODE = "open"   # "dilate" / "erode" solo para experimentos


def _morph(s, fn):
    p = np.pad(s, 1, mode="edge")
    h, w = s.shape
    return fn([p[i:i + h, j:j + w] for i in range(3) for j in range(3)], 0)


def static_src_mask(i0, i1, tau=0.015):
    """Pixeles quietos (media resolucion): apertura 3x3 (erosion + dilatacion), asi
    las coincidencias sueltas de una textura que se mueve no cuentan y el HUD sigue entero."""
    l = lambda x: (x[..., 0] * 0.299 + x[..., 1] * 0.587) + x[..., 2] * 0.114
    s = (np.abs(l(avgpool(i0, 2)) - l(avgpool(i1, 2))) < tau).astype(np.float32)
    if MASK_MODE == "dilate":
        return _morph(s, np.max)
    if MASK_MODE == "erode":
        return _morph(s, np.min)
    return _morph(_morph(s, np.min), np.max)


# opciones del warp (experimentos con ideas de VCFG; los valores por defecto son los del plugin)
OPTS = {"sigma": 0.1, "floor": 0.0, "offscreen": False, "coherent": 0.0, "static_tau": 0.03}


def cand_warp(i0, i1, low, s, sigma=None, eps=1e-3):
    """model/net2.py: cand_warp (bloques superpuestos + eleccion por pixel + anti-fantasmas)."""
    sigma = OPTS["sigma"] if sigma is None else sigma
    H, W, _ = i0.shape
    h, w = low.shape[:2]
    S = static_src_mask(i0, i1)
    gy = (np.arange(H) + 0.5) / s - 0.5
    gx = (np.arange(W) + 0.5) / s - 0.5
    fy = (gy - np.floor(gy))[:, None]; fx = (gx - np.floor(gx))[None, :]
    y0 = np.clip(np.floor(gy).astype(int), 0, h - 1); x0 = np.clip(np.floor(gx).astype(int), 0, w - 1)
    y1 = np.clip(np.floor(gy).astype(int) + 1, 0, h - 1); x1 = np.clip(np.floor(gx).astype(int) + 1, 0, w - 1)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    def half(F):
        xs = np.clip(np.floor((xx + F[..., 0]) * 0.5).astype(int), 0, W // 2 - 1)
        ys = np.clip(np.floor((yy + F[..., 1]) * 0.5).astype(int), 0, H // 2 - 1)
        return S[ys, xs]

    def outside(F):
        px, py = xx + F[..., 0], yy + F[..., 1]
        return (px < 0) | (px > W - 1) | (py < 0) | (py > H - 1)

    cands = [(low[y0][:, x0], (1 - fy) * (1 - fx)), (low[y0][:, x1], (1 - fy) * fx),
             (low[y1][:, x0], fy * (1 - fx)), (low[y1][:, x1], fy * fx)]
    outs, errs, bws = [], [], []
    for c, bw in cands:
        F0, F1 = c[..., 0:2] * s, c[..., 2:4] * s
        a, b = warp(i0, F0), warp(i1, F1)
        m = 1 / (1 + np.exp(-c[..., 4:5]))
        e = np.abs(a - b).sum(-1)
        dis = np.clip((e - 0.06) / 0.12, 0, 1)   # solo si las dos muestras no coinciden
        r0 = (1 - half(F0) * (np.abs(F0).sum(-1) > 1.0) * dis)[..., None]
        r1 = (1 - half(F1) * (np.abs(F1).sum(-1) > 1.0) * dis)[..., None]
        if OPTS["offscreen"]:
            o0, o1 = outside(F0)[..., None], outside(F1)[..., None]
            a2 = np.where(o0 & ~o1, b, a)
            b2 = np.where(o1 & ~o0, a, b)
            a, b = a2, b2
            e = np.where((o0 | o1)[..., 0] & ~(o0 & o1)[..., 0], 0.1, e)
        wa, wb = m * r0, (1 - m) * r1
        outs.append((wa * a + wb * b + eps * 0.5 * (a + b)) / (wa + wb + eps))
        errs.append(e)
        bws.append(np.broadcast_to(bw, (H, W)))
    E = np.stack(errs)
    wt = (np.stack(bws) + OPTS["floor"]) * np.exp(-(E - E.min(0)) / sigma)
    wt /= wt.sum(0)
    out = (wt[..., None] * np.stack(outs)).sum(0)
    if OPTS["coherent"] > 0:
        # camino coherente (VCFG): si los 4 vectores casi coinciden, un solo par de muestras con el bilineal
        C = np.stack([c for c, _ in cands])[..., :4] * s
        coh = (C.max(0) - C.min(0)).max(-1) <= OPTS["coherent"]
        up = resize_bilinear(low, H, W)
        a, b = warp(i0, up[..., 0:2] * s), warp(i1, up[..., 2:4] * s)
        m = 1 / (1 + np.exp(-up[..., 4:5]))
        out = np.where(coh[..., None], m * a + (1 - m) * b, out)
    return out


# ------------------------------------------------------------------ NXFG-3
def interpolate3(i0, i1, bm=None, scene_cut=0.12, sigma=0.05, prior=0.6):
    """NXFG-3 (model/net3.py) en NumPy: referencia exacta de fg3_select + fg3_warp.
    i0, i1 con alto/ancho multiplos de 32; bm: model/bm2.motion o None."""
    import bm2
    if bm is None:
        bm = np.concatenate(bm2.motion(bm2.pyramid(i0), bm2.pyramid(i1)), -1)
    H, W, _ = i0.shape
    if bm[..., 2].mean() > scene_cut:
        return i1.copy()
    v = bm[..., 1::-1] / 8.0
    low = upsample5(np.concatenate([-v, v, np.zeros_like(v[..., :1])], -1), 2)   # malla 1/8
    hl, wl = low.shape[:2]
    q0, q1 = avgpool(i0, 4), avgpool(i1, 4)
    h, w = q0.shape[:2]
    gy = (np.arange(h) + 0.5) * 0.5 - 0.5
    gx = (np.arange(w) + 0.5) * 0.5 - 0.5
    cy = np.floor(gy + 0.5).astype(int); cx = np.floor(gx + 0.5).astype(int)
    Fs, errs, pws = [], [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            c = low[np.clip(cy + dy, 0, hl - 1)][:, np.clip(cx + dx, 0, wl - 1)]
            d2 = (gy - (cy + dy))[:, None] ** 2 + (gx - (cx + dx))[None, :] ** 2
            pws.append(np.exp(-d2 / (2 * prior * prior)))
            F0, F1 = c[..., 0:2] * 2, c[..., 2:4] * 2
            errs.append(np.abs(warp(q0, F0) - warp(q1, F1)).sum(-1))
            Fs.append(np.concatenate([F0, F1], -1) * 4)
    E = np.stack(errs)
    wt = np.stack(pws) * np.exp(-(E - E.min(0)) / sigma)
    wt /= wt.sum(0)
    F = resize_bilinear((wt[..., None] * np.stack(Fs)).sum(0), H, W)
    F0, F1 = F[..., 0:2], F[..., 2:4]
    a, b = warp(i0, F0), warp(i1, F1)
    S = static_src_mask(i0, i1)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    def half(F):
        xs = np.clip(np.floor((xx + F[..., 0]) * 0.5).astype(int), 0, W // 2 - 1)
        ys = np.clip(np.floor((yy + F[..., 1]) * 0.5).astype(int), 0, H // 2 - 1)
        return S[ys, xs]
    e = np.abs(a - b).sum(-1)
    dis = np.clip((e - 0.06) / 0.12, 0, 1)
    r0 = (1 - half(F0) * (np.abs(F0).sum(-1) > 1.0) * dis)[..., None]
    r1 = (1 - half(F1) * (np.abs(F1).sum(-1) > 1.0) * dis)[..., None]
    wa, wb = 0.5 * r0, 0.5 * r1
    mix = (wa * a + wb * b + 1e-3 * 0.5 * (a + b)) / (wa + wb + 1e-3)
    st = static_weight(i0, i1, OPTS["static_tau"])
    return st * (0.5 * (i0 + i1)) + (1 - st) * mix
