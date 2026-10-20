"""Baseline que reproduce el enfoque de VCFG v2.14 (sin red neuronal).

VCFG hace: piramide de luma (pyr8) -> busqueda de movimiento por bloques en
3 niveles (me3, me2, me1) -> movimiento global (gm8) -> filtro de mediana de
vectores (mf8) -> warp + RCAS. Aqui se implementa lo mismo de forma simetrica
(bilateral): para cada bloque del fotograma intermedio se busca v tal que
I0(x - v) ~ I1(x + v), que es el truco clasico de FRUC en TVs.
"""
import numpy as np


def luma(img):
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


def down2(x):
    H, W = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
    x = x[:H, :W]
    return 0.25 * (x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2])


def symmetric_search(l0, l1, init, block, radius):
    """init: vectores por bloque (By, Bx, 2) en px de este nivel. Devuelve vectores refinados."""
    H, W = l0.shape
    By, Bx = H // block, W // block
    best = np.full((By, Bx), np.inf, np.float32)
    vec = init.copy()
    ar = np.arange(block)
    by0 = (np.arange(By) * block)[:, None, None]
    bx0 = (np.arange(Bx) * block)[None, :, None]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            # coste por bloque para el vector candidato init + (dy,dx), todo vectorizado
            vy = (init[..., 0] + dy).astype(int)
            vx = (init[..., 1] + dx).astype(int)
            ya = np.clip(by0 - vy[..., None] + ar, 0, H - 1)
            xa = np.clip(bx0 - vx[..., None] + ar, 0, W - 1)
            yb = np.clip(by0 + vy[..., None] + ar, 0, H - 1)
            xb = np.clip(bx0 + vx[..., None] + ar, 0, W - 1)
            a = l0[ya[..., :, None], xa[..., None, :]]
            b = l1[yb[..., :, None], xb[..., None, :]]
            cost = np.abs(a - b).sum((-1, -2))
            # pequena penalizacion a vectores largos (estabilidad en zonas planas)
            cost += 0.002 * block * block * (np.abs(vy) + np.abs(vx))
            m = cost < best
            best[m] = cost[m]
            vec[m, 0] = vy[m]
            vec[m, 1] = vx[m]
    return vec


def median3(v):
    p = np.pad(v, ((1, 1), (1, 1), (0, 0)), mode="edge")
    stack = np.stack([p[i:i + v.shape[0], j:j + v.shape[1]] for i in range(3) for j in range(3)])
    return np.median(stack, axis=0)


def block_vectors(i0, i1, block=8, levels=3, radius=3):
    """Vectores simetricos (dy, dx) por bloque de 8x8 -> malla de 1/8 de resolucion.

    El fotograma intermedio se reconstruye con I0(x - v) e I1(x + v).
    """
    l0, l1 = luma(i0), luma(i1)
    pyr = [(l0, l1)]
    for _ in range(levels - 1):
        pyr.append((down2(pyr[-1][0]), down2(pyr[-1][1])))
    vec = None
    for lv in range(levels - 1, -1, -1):
        a, b = pyr[lv]
        By, Bx = a.shape[0] // block, a.shape[1] // block
        if vec is None:
            init = np.zeros((By, Bx, 2), np.float32)
        else:
            init = np.repeat(np.repeat(vec, 2, 0), 2, 1)[:By, :Bx] * 2
            if init.shape[0] < By or init.shape[1] < Bx:
                init = np.pad(init, ((0, By - init.shape[0]), (0, Bx - init.shape[1]), (0, 0)), mode="edge")
        vec = median3(symmetric_search(a, b, np.round(init), block, radius))
    return vec.astype(np.float32)


def interpolate(i0, i1, block=8, levels=3, radius=3):
    vec = block_vectors(i0, i1, block, levels, radius)
    # vectores por bloque -> campo denso (bilineal) y warp simetrico t=0.5
    H, W = i0.shape[:2]
    By, Bx = vec.shape[:2]
    gy = (np.arange(H) + 0.5) / block - 0.5
    gx = (np.arange(W) + 0.5) / block - 0.5
    gy = np.clip(gy, 0, By - 1); gx = np.clip(gx, 0, Bx - 1)
    y0 = np.floor(gy).astype(int); x0 = np.floor(gx).astype(int)
    y1 = np.minimum(y0 + 1, By - 1); x1 = np.minimum(x0 + 1, Bx - 1)
    fy = (gy - y0)[:, None, None]; fx = (gx - x0)[None, :, None]
    dense = (vec[y0][:, x0] * (1 - fx) + vec[y0][:, x1] * fx) * (1 - fy) + \
            (vec[y1][:, x0] * (1 - fx) + vec[y1][:, x1] * fx) * fy
    w0 = _warp(i0, -dense)
    w1 = _warp(i1, dense)
    return 0.5 * (w0 + w1)


def _warp(img, flow):
    H, W, _ = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    y = np.clip(yy + flow[..., 0], 0, H - 1); x = np.clip(xx + flow[..., 1], 0, W - 1)
    y0 = np.floor(y).astype(int); x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, H - 1); x1 = np.minimum(x0 + 1, W - 1)
    fy = (y - y0)[..., None]; fx = (x - x0)[..., None]
    return (img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy) + \
           (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy
