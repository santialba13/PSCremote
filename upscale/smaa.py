"""Referencia en Python de SMAA 1x (misma traduccion que shaders/smaa.cl).

SMAA: Jimenez, Echevarria, Masia, Navarro, Gutierrez (licencia MIT,
https://github.com/iryoku/smaa). Preset HIGH sin busqueda diagonal, en
coordenadas de pixel u = pixel + 0.5. Las texturas Area/Search se leen de
third_party/smaa/*.h.

Lento a proposito (bucle escalar en el pase 2): sirve para validar el shader
sobre imagenes chicas, no para procesar video.
"""
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLD = 0.05   # preset "Ultra" (igual que shaders/smaa.cl)
MAX_SEARCH_STEPS = 16
CORNER_ROUNDING_NORM = 0.25
LCA_FACTOR = 2.0


def _load_tex(name, var, w, h, c):
    text = open(os.path.join(ROOT, "third_party", "smaa", name)).read()
    body = text[text.index(var):]
    body = body[body.index("{") + 1:body.index("};")]
    vals = [int(v, 16) for v in re.findall(r"0x[0-9a-fA-F]+", body)]
    return np.array(vals, np.uint8).reshape(h, w, c)


AREA = _load_tex("AreaTex.h", "areaTexBytes", 160, 560, 2)
SEARCH = _load_tex("SearchTex.h", "searchTexBytes", 64, 16, 1)


def _bilerp(t, ux, uy):
    """Bilineal con clamp; t (H, W, C); ux, uy en coordenadas u (centro = i + 0.5)."""
    H, W = t.shape[:2]
    x, y = ux - 0.5, uy - 0.5
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - x0, y - y0
    xa, xb = min(max(x0, 0), W - 1), min(max(x0 + 1, 0), W - 1)
    ya, yb = min(max(y0, 0), H - 1), min(max(y0 + 1, 0), H - 1)
    t = t.astype(np.float64)
    return (t[ya, xa] * (1 - fx) + t[ya, xb] * fx) * (1 - fy) + (t[yb, xa] * (1 - fx) + t[yb, xb] * fx) * fy


def luma(img):
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def edges(img):
    """Bordes 0/1 (el shader los guarda como bytes 0/255: mismo valor al muestrear)."""
    L = luma(img.astype(np.float64))
    p = np.pad(L, 2, mode="edge")
    H, W = L.shape
    at = lambda dx, dy: p[2 + dy:2 + dy + H, 2 + dx:2 + dx + W]
    dl, dt = np.abs(L - at(-1, 0)), np.abs(L - at(0, -1))
    ex, ey = (dl >= THRESHOLD).astype(np.float64), (dt >= THRESHOLD).astype(np.float64)
    dr, db = np.abs(L - at(1, 0)), np.abs(L - at(0, 1))
    mx = np.maximum(np.maximum(dl, dr), np.abs(at(-1, 0) - at(-2, 0)))
    my = np.maximum(np.maximum(dt, db), np.abs(at(0, -1) - at(0, -2)))
    fin = np.maximum(mx, my)
    ex *= fin <= LCA_FACTOR * dl
    ey *= fin <= LCA_FACTOR * dt
    return np.stack([ex, ey], -1)


def _search_length(e, offset):
    u = 32.0 * e[0] + 66.0 * offset + 0.5
    v = -32.0 * e[1] + 32.5
    return _bilerp(SEARCH, u, v)[0] / 255.0


def _search(ed, ux, uy, end, axis, sign):
    e = np.array([0.0, 1.0]) if axis == 0 else np.array([1.0, 0.0])
    while True:
        pos = ux if axis == 0 else uy
        cont = (pos > end) if sign < 0 else (pos < end)
        main, cross = (e[1], e[0]) if axis == 0 else (e[0], e[1])
        if not (cont and main > 0.8281 and cross == 0.0):
            break
        e = _bilerp(ed, ux, uy)
        if axis == 0:
            ux += 2.0 * sign
        else:
            uy += 2.0 * sign
    ee = e if axis == 0 else e[::-1]
    off = -(255.0 / 127.0) * _search_length(ee, 0.0 if sign < 0 else 0.5) + 3.25
    pos = ux if axis == 0 else uy
    return pos + off if sign < 0 else pos - off


def _area(sd, e1, e2):
    tx = 16.0 * np.rint(4.0 * e1) + sd[0] + 0.5
    ty = 16.0 * np.rint(4.0 * e2) + sd[1] + 0.5
    return _bilerp(AREA, tx, ty) / 255.0


def weights(ed):
    H, W = ed.shape[:2]
    out = np.zeros((H, W, 4))
    S2 = 2.0 * MAX_SEARCH_STEPS
    for y, x in zip(*np.nonzero(ed.sum(-1))):
        ux, uy = x + 0.5, y + 0.5
        er, eg = ed[y, x]
        if eg > 0:
            cx = _search(ed, ux - 0.25, uy - 0.125, ux - 0.25 - S2, 0, -1)
            cy = uy - 0.25
            e1 = _bilerp(ed, cx, cy)[0]
            cz = _search(ed, ux + 1.25, uy - 0.125, ux + 1.25 + S2, 0, +1)
            d = np.abs(np.rint([cx - ux, cz - ux]))
            e2 = _bilerp(ed, cz + 1.0, cy)[0]
            a = _area(np.sqrt(d), e1, e2)
            lx, ly = float(d[0] <= d[1]), float(d[1] <= d[0])
            rx, ry = (1 - CORNER_ROUNDING_NORM) * lx / (lx + ly), (1 - CORNER_ROUNDING_NORM) * ly / (lx + ly)
            fx = 1 - rx * _bilerp(ed, cx, uy + 1)[0] - ry * _bilerp(ed, cz + 1, uy + 1)[0]
            fy = 1 - rx * _bilerp(ed, cx, uy - 2)[0] - ry * _bilerp(ed, cz + 1, uy - 2)[0]
            out[y, x, 0] = a[0] * np.clip(fx, 0, 1)
            out[y, x, 1] = a[1] * np.clip(fy, 0, 1)
        if er > 0:
            cy = _search(ed, ux - 0.125, uy - 0.25, uy - 0.25 - S2, 1, -1)
            cx = ux - 0.25
            e1 = _bilerp(ed, cx, cy)[1]
            cz = _search(ed, ux - 0.125, uy + 1.25, uy + 1.25 + S2, 1, +1)
            d = np.abs(np.rint([cy - uy, cz - uy]))
            e2 = _bilerp(ed, cx, cz + 1.0)[1]
            a = _area(np.sqrt(d), e1, e2)
            lx, ly = float(d[0] <= d[1]), float(d[1] <= d[0])
            rx, ry = (1 - CORNER_ROUNDING_NORM) * lx / (lx + ly), (1 - CORNER_ROUNDING_NORM) * ly / (lx + ly)
            fx = 1 - rx * _bilerp(ed, ux + 1, cy)[1] - ry * _bilerp(ed, ux + 1, cz + 1)[1]
            fy = 1 - rx * _bilerp(ed, ux - 2, cy)[1] - ry * _bilerp(ed, ux - 2, cz + 1)[1]
            out[y, x, 2] = a[0] * np.clip(fx, 0, 1)
            out[y, x, 3] = a[1] * np.clip(fy, 0, 1)
    return np.floor(out * 255.0 + 0.5) / 255.0   # el shader los guarda en 8 bits


def blend(img, w):
    H, W = img.shape[:2]
    out = img.astype(np.float64).copy()
    for y in range(H):
        for x in range(W):
            ax = w[y, min(x + 1, W - 1), 3]
            ay = w[min(y + 1, H - 1), x, 1]
            aw, az = w[y, x, 0], w[y, x, 2]
            if ax + ay + az + aw < 1e-5:
                continue
            h = max(ax, az) > max(ay, aw)
            off = (ax, 0.0, az, 0.0) if h else (0.0, ay, 0.0, aw)
            bw = np.array([ax, az] if h else [ay, aw])
            bw /= bw.sum()
            ux, uy = x + 0.5, y + 0.5
            out[y, x] = bw[0] * _bilerp(img, ux + off[0], uy + off[1]) + bw[1] * _bilerp(img, ux - off[2], uy - off[3])
    return out


def smaa(img):
    """img float (H, W, 3) en [0, 1], valores codificados en gamma (no lineales)."""
    return blend(img, weights(edges(img)))
