"""Escena de prueba para antialiasing: poligonos y circulos con bordes duros.

render(ss) muestrea cada pixel en una grilla ss x ss: ss=1 da la imagen
"aliased" (como sale de un juego sin AA) y ss=8 sirve de referencia (SSAA 8x8).
"""
import numpy as np


def render(W=256, H=160, ss=1, seed=0):
    rng = np.random.default_rng(seed)
    shapes = []
    for _ in range(14):
        kind = rng.integers(3)
        col = rng.random(3)
        cx, cy = rng.random(2) * [W, H]
        if kind == 0:   # circulo
            shapes.append(("c", col, cx, cy, rng.uniform(8, 40)))
        else:           # rectangulo rotado (lineas casi horizontales/verticales incluidas)
            shapes.append(("r", col, cx, cy, rng.uniform(10, 70), rng.uniform(4, 30), rng.uniform(-0.6, 0.6)))
    off = (np.arange(ss) + 0.5) / ss
    acc = np.zeros((H, W, 3))
    for oy in off:
        for ox in off:
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
            yy += oy; xx += ox
            img = np.zeros((H, W, 3)) + np.array([0.15, 0.18, 0.22]) + 0.1 * (xx / W)[..., None]
            for s in shapes:
                if s[0] == "c":
                    m = (xx - s[2]) ** 2 + (yy - s[3]) ** 2 < s[4] ** 2
                else:
                    c, sn = np.cos(s[6]), np.sin(s[6])
                    dx, dy = xx - s[2], yy - s[3]
                    m = (np.abs(c * dx + sn * dy) < s[4]) & (np.abs(-sn * dx + c * dy) < s[5])
                img[m] = s[1]
            acc += img
    return acc / (ss * ss)
