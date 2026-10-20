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
