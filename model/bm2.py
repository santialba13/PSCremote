"""Movimiento de NXFG-2: piramide RGB+luma a media resolucion y block-matching de 4 niveles.

Es la referencia exacta de los kernels nxfg_prep / nxfg_down / nxfg_bm / bm_median3
de shaders/nxflow.cl (mismo orden de suma, mismos desempates):

  L1  media resolucion (2x2 de la imagen), canales r g b y luma
  L2  1/4, L3  1/8 (entrada RGB del nivel B de la red), L4  1/16 (entrada del nivel A)
  busqueda simetrica de bloques de 8x8 en L4 (radio 4), L3 (3), L2 (3) y L1 (2), con
  mediana 3x3 de vectores despues de cada nivel; en L1 se agrega el ajuste sub-pixel
  (parabola sobre los costes vecinos) y la confianza (diferencia media de luma).

Alcance del vector simetrico v (I0(x - v) ~ I1(x + v)): +-104 px de resolucion completa,
es decir movimientos de hasta ~200 px entre dos fotogramas reales (30 fps).
La malla de L1 con bloques de 8 = 1/16 de la imagen (120x68 a 1080p, como VCFG).
"""
import numpy as np

RADII = (4, 3, 3, 2)          # L4, L3, L2, L1
LAM_D = 0.08                  # penalizacion por alejarse del vector del nivel grueso
LAM_V = 0.02                  # penalizacion por vector largo (zonas planas quietas)


def pyramid(img):
    """img (H, W, 3) en [0,1], H y W multiplos de 32 -> [L1, L2, L3, L4] (h, w, 4)."""
    H, W, _ = img.shape
    x = img.astype(np.float32)
    # 2x2 en el orden del kernel: (p00 + p01) + (p10 + p11)
    a = ((x[0::2, 0::2] + x[0::2, 1::2]) + (x[1::2, 0::2] + x[1::2, 1::2])) * np.float32(0.25)
    l = (a[..., 0] * np.float32(0.299) + a[..., 1] * np.float32(0.587)) + a[..., 2] * np.float32(0.114)
    out = [np.concatenate([a, l[..., None]], -1).astype(np.float32)]
    for _ in range(3):
        p = out[-1]
        out.append((((p[0::2, 0::2] + p[0::2, 1::2]) + (p[1::2, 0::2] + p[1::2, 1::2]))
                    * np.float32(0.25)).astype(np.float32))
    return out


def _search(l0, l1, init, R, finest):
    H, W = l0.shape
    By, Bx = init.shape[:2]
    ar = np.arange(8)
    by0 = (np.arange(By) * 8)[:, None, None]
    bx0 = (np.arange(Bx) * 8)[None, :, None]
    nc = 2 * R + 1
    costs = np.zeros((nc, nc, By, Bx), np.float32)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            vy = init[..., 0] + dy
            vx = init[..., 1] + dx
            ya = np.clip(by0 - vy[..., None] + ar, 0, H - 1)
            xa = np.clip(bx0 - vx[..., None] + ar, 0, W - 1)
            yb = np.clip(by0 + vy[..., None] + ar, 0, H - 1)
            xb = np.clip(bx0 + vx[..., None] + ar, 0, W - 1)
            a = l0[ya[..., :, None], xa[..., None, :]]
            b = l1[yb[..., :, None], xb[..., None, :]]
            d = np.abs(a - b).astype(np.float32)          # (By, Bx, 8, 8)
            # suma por filas, despues las 8 filas (orden del kernel)
            s = np.zeros((By, Bx), np.float32)
            for j in range(8):
                r = np.zeros((By, Bx), np.float32)
                for i in range(8):
                    r = r + d[..., j, i]
                s = s + r
            pen = np.float32(LAM_D) * np.float32(abs(dy) + abs(dx)) + \
                np.float32(LAM_V) * (np.abs(vy) + np.abs(vx)).astype(np.float32)
            costs[dy + R, dx + R] = s + pen
    flat = costs.reshape(nc * nc, By, Bx)
    k = np.argmin(flat, 0)                                # primer minimo: orden dy, dx
    ky, kx = k // nc, k % nc
    vec = np.stack([init[..., 0] + ky - R, init[..., 1] + kx - R], -1).astype(np.float32)
    if not finest:
        return vec, None
    yy, xx = np.meshgrid(np.arange(By), np.arange(Bx), indexing="ij")
    c0 = flat[k, yy, xx]

    def sub(cm, cp, inside):
        den = cm - np.float32(2.0) * c0 + cp
        ok = inside & (den > np.float32(1e-6))
        off = np.where(ok, np.float32(0.5) * (cm - cp) / np.where(ok, den, np.float32(1.0)), np.float32(0.0))
        return np.clip(off, -0.5, 0.5).astype(np.float32)

    iny = (ky > 0) & (ky < nc - 1)
    inx = (kx > 0) & (kx < nc - 1)
    cym = costs[np.clip(ky - 1, 0, nc - 1), kx, yy, xx]
    cyp = costs[np.clip(ky + 1, 0, nc - 1), kx, yy, xx]
    cxm = costs[ky, np.clip(kx - 1, 0, nc - 1), yy, xx]
    cxp = costs[ky, np.clip(kx + 1, 0, nc - 1), yy, xx]
    pen = np.float32(LAM_D) * (np.abs(ky - R) + np.abs(kx - R)).astype(np.float32) + \
        np.float32(LAM_V) * (np.abs(vec[..., 0]) + np.abs(vec[..., 1])).astype(np.float32)
    conf = (c0 - pen) * np.float32(1.0 / 64.0)
    vec[..., 0] += sub(cym, cyp, iny)
    vec[..., 1] += sub(cxm, cxp, inx)
    return vec, conf.astype(np.float32)


def median3(v):
    By, Bx = v.shape[:2]
    p = np.pad(v, ((1, 1), (1, 1), (0, 0)), mode="edge")
    stack = np.stack([p[i:i + By, j:j + Bx] for i in range(3) for j in range(3)])
    return np.median(stack, axis=0).astype(np.float32)


def motion(pyr0, pyr1):
    """-> (vec (By, Bx, 2) (dy, dx) en px de L1, conf (By, Bx, 1)) sobre la malla 1/16."""
    vec = None
    conf = None
    for lv, R in zip((3, 2, 1, 0), RADII):
        l0, l1 = pyr0[lv][..., 3], pyr1[lv][..., 3]
        By, Bx = -(-l0.shape[0] // 8), -(-l0.shape[1] // 8)
        if vec is None:
            init = np.zeros((By, Bx, 2), np.int64)
        else:
            cy = np.minimum(np.arange(By) // 2, vec.shape[0] - 1)
            cx = np.minimum(np.arange(Bx) // 2, vec.shape[1] - 1)
            init = np.rint(2.0 * vec[cy][:, cx]).astype(np.int64)
        v, c = _search(l0, l1, init, R, lv == 0)
        vec = median3(v)
        conf = c
    return vec, conf[..., None]


def features(i0, i1):
    """Todo lo que la red necesita de un par: (L3_0, L3_1, L4_0, L4_1, bm (By,Bx,3))."""
    p0, p1 = pyramid(i0), pyramid(i1)
    vec, conf = motion(p0, p1)
    return p0[2][..., :3], p1[2][..., :3], p0[3][..., :3], p1[3][..., :3], \
        np.concatenate([vec, conf], -1)
