"""Port en NumPy de AMD FidelityFX Super Resolution 1 (EASU + RCAS).

Referencia: GPUOpen-Effects/FidelityFX-FSR, ffx_fsr1.h (licencia MIT,
Copyright (c) 2021 Advanced Micro Devices, Inc.). Es la ruta en float
(FsrEasuF / FsrRcasF) traducida 1:1 y vectorizada; sirve como referencia
bit-a-bit aproximada para validar el shader de PS4 (shaders/fsr1.cl).

Imagenes float32 RGB en [0,1], forma (H, W, 3).
"""
import numpy as np

RCAS_LIMIT = 0.25 - 1.0 / 16.0


def _luma(c):
    return c[..., 2] * 0.5 + (c[..., 0] * 0.5 + c[..., 1])


def easu(img, out_h, out_w):
    H, W, _ = img.shape
    sy, sx = H / out_h, W / out_w
    oy, ox = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    ppx = ox * sx + 0.5 * sx - 0.5
    ppy = oy * sy + 0.5 * sy - 0.5
    fx, fy = np.floor(ppx), np.floor(ppy)
    ppx -= fx; ppy -= fy
    fx = fx.astype(np.int32); fy = fy.astype(np.int32)

    def tap(dx, dy):
        return img[np.clip(fy + dy, 0, H - 1), np.clip(fx + dx, 0, W - 1)]

    offs = {"b": (0, -1), "c": (1, -1), "e": (-1, 0), "f": (0, 0), "g": (1, 0),
            "h": (2, 0), "i": (-1, 1), "j": (0, 1), "k": (1, 1), "l": (2, 1),
            "n": (0, 2), "o": (1, 2)}
    C = {k: tap(*v) for k, v in offs.items()}
    L = {k: _luma(v) for k, v in C.items()}

    dirx = np.zeros_like(ppx); diry = np.zeros_like(ppx); ln = np.zeros_like(ppx)

    def setf(w, lA, lB, lC, lD, lE):
        nonlocal dirx, diry, ln
        dc = lD - lC; cb = lC - lB
        lenX = np.maximum(np.abs(dc), np.abs(cb))
        lenX = 1.0 / np.maximum(lenX, 1e-8)
        dX = lD - lB
        dirx = dirx + dX * w
        lenX = np.clip(np.abs(dX) * lenX, 0, 1) ** 2
        ln = ln + lenX * w
        ec = lE - lC; ca = lC - lA
        lenY = np.maximum(np.abs(ec), np.abs(ca))
        lenY = 1.0 / np.maximum(lenY, 1e-8)
        dY = lE - lA
        diry = diry + dY * w
        lenY = np.clip(np.abs(dY) * lenY, 0, 1) ** 2
        ln = ln + lenY * w

    setf((1 - ppx) * (1 - ppy), L["b"], L["e"], L["f"], L["g"], L["j"])
    setf(ppx * (1 - ppy), L["c"], L["f"], L["g"], L["h"], L["k"])
    setf((1 - ppx) * ppy, L["f"], L["i"], L["j"], L["k"], L["n"])
    setf(ppx * ppy, L["g"], L["j"], L["k"], L["l"], L["o"])

    dirR = dirx * dirx + diry * diry
    zro = dirR < 1.0 / 32768.0
    dirR = np.where(zro, 1.0, 1.0 / np.sqrt(np.maximum(dirR, 1e-20)))
    dirx = np.where(zro, 1.0, dirx)
    dirx *= dirR; diry *= dirR
    ln = (ln * 0.5) ** 2
    stretch = (dirx * dirx + diry * diry) / np.maximum(np.maximum(np.abs(dirx), np.abs(diry)), 1e-8)
    len2x = 1.0 + (stretch - 1.0) * ln
    len2y = 1.0 - 0.5 * ln
    lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * ln
    clp = 1.0 / lob

    aC = np.zeros(ppx.shape + (3,), np.float32); aW = np.zeros_like(ppx)
    for k, (dx, dy) in offs.items():
        ofx, ofy = dx - ppx, dy - ppy
        vx = (ofx * dirx + ofy * diry) * len2x
        vy = (ofx * -diry + ofy * dirx) * len2y
        d2 = np.minimum(vx * vx + vy * vy, clp)
        wB = (2.0 / 5.0 * d2 - 1.0) ** 2
        wA = (lob * d2 - 1.0) ** 2
        wB = 25.0 / 16.0 * wB - (25.0 / 16.0 - 1.0)
        w = wB * wA
        aC += C[k] * w[..., None]; aW += w
    mn = np.minimum(np.minimum(C["f"], C["g"]), np.minimum(C["j"], C["k"]))
    mx = np.maximum(np.maximum(C["f"], C["g"]), np.maximum(C["j"], C["k"]))
    return np.clip(aC / aW[..., None], mn, mx).astype(np.float32)


def _y601(c):
    return (c[..., 0] * np.float32(0.299) + c[..., 1] * np.float32(0.587)) + c[..., 2] * np.float32(0.114)


def rcas(img, sharpness=0.2, denoise=False, detail=0.0):
    """sharpness en 'stops' como en FSR: 0 = maximo, 2 = suave.

    detail > 0: FSR+ de VCFG-NX (ver shaders/fsr1.cl): realce de detalle de radio 2
    sobre la luma, limitado al rango 3x3 local (+-0.01) y sin tocar el grano."""
    con = 2.0 ** (-sharpness)
    p = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    b = p[:-2, 1:-1]; d = p[1:-1, :-2]; e = p[1:-1, 1:-1]; f = p[1:-1, 2:]; h = p[2:, 1:-1]
    mn4 = np.minimum(np.minimum(b, d), np.minimum(f, h))
    mx4 = np.maximum(np.maximum(b, d), np.maximum(f, h))
    hitMin = np.minimum(mn4, e) / np.maximum(4.0 * mx4, 1e-8)
    hitMax = (1.0 - np.maximum(mx4, e)) / np.minimum(4.0 * mn4 - 4.0, -1e-8)
    lobe = np.maximum(-hitMin, hitMax).max(-1)
    lobe = np.maximum(-RCAS_LIMIT, np.minimum(lobe, 0.0)) * con
    if denoise:
        bL, dL, eL, fL, hL = (_luma(x) for x in (b, d, e, f, h))
        nz = 0.25 * (bL + dL + fL + hL) - eL
        rng = np.maximum.reduce([bL, dL, eL, fL, hL]) - np.minimum.reduce([bL, dL, eL, fL, hL])
        nz = np.clip(np.abs(nz) / np.maximum(rng, 1e-8), 0, 1)
        lobe *= -0.5 * nz + 1.0
    lobe = lobe[..., None]
    o = (lobe * (b + d + f + h) + e) / (4.0 * lobe + 1.0)
    if detail > 0:
        q = np.pad(img, ((2, 2), (2, 2), (0, 0)), mode="edge")
        Y = _y601(q)
        H, W = img.shape[:2]
        t = lambda dy, dx: Y[2 + dy:2 + dy + H, 2 + dx:2 + dx + W]
        ya, yb, yc, yd, ye, yf, yg, yh, yi = (t(-1, -1), t(-1, 0), t(-1, 1), t(0, -1), t(0, 0), t(0, 1),
                                              t(1, -1), t(1, 0), t(1, 1))
        far = (t(-2, 0) + t(0, -2)) + (t(0, 2) + t(2, 0))
        blur = (((4.0 * ye + 2.0 * ((yb + yd) + (yf + yh))) + ((ya + yc) + (yg + yi))) + far) / 20.0
        dt = ye - blur
        wn = np.clip((np.abs(dt) - 0.004) / 0.016, 0, 1)
        nb = np.stack([ya, yb, yc, yd, ye, yf, yg, yh, yi])
        yo = _y601(o)
        tgt = np.clip(yo + detail * dt * wn, np.minimum(nb.min(0), yo) - 0.01, np.maximum(nb.max(0), yo) + 0.01)
        o = o + (tgt - yo)[..., None]
    return np.clip(o, 0, 1).astype(np.float32)


def fsr1(img, out_h, out_w, sharpness=0.2):
    return rcas(easu(img, out_h, out_w), sharpness)
