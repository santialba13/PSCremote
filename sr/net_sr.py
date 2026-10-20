"""NXSR: red de realce/reescalado del fotograma final para la GPU de PS4.

  x (RGB, resolucion de salida)
   -> space_to_depth(s)              s = 2: 12 canales a 1/2 de resolucion
   -> conv 3x3 (12 -> C) + lrelu     cabeza
   -> n x conv 3x3 (C -> C) + lrelu  cuerpo, con residual
   -> conv 1x1 (C -> 3*s*s)          cola
   -> depth_to_space(s) = residuo r (resolucion de salida)
   -> y = limitar(x + k * r)

El limitador acota cada canal de y al rango [minimo, maximo] de los 3x3 vecinos
de x, ampliado en MARGIN veces ese mismo rango (0 en la version final, igual
que el shader). Asi la red puede empinar bordes y recuperar detalle, pero no
sobreimpulsa y en zonas planas (minimo ~ maximo) no puede agregar grano. k (intensidad) va de 0 a 1:
nunca extrapola.

Tambien carga los modelos de GNMScale (SRNetFemto) desde su weights.bin para
compararlos en los mismos datos.
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np


def conv(x, w, b, act):
    y = jax.lax.conv_general_dilated(x, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")) + b
    return jnp.where(y > 0, y, 0.1 * y) if act else y


def s2d(x, s):
    B, H, W, C = x.shape
    x = x.reshape(B, H // s, s, W // s, s, C).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H // s, W // s, s * s * C)


def d2s(x, s):
    B, H, W, C = x.shape
    c = C // (s * s)
    x = x.reshape(B, H, W, s, s, c).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H * s, W * s, c)


WINDOW = 1      # radio del limitador: 1 = 3x3, 2 = 5x5


def local_minmax(x, r=None):
    r = WINDOW if r is None else r
    p = jnp.pad(x, ((0, 0), (r, r), (r, r), (0, 0)), mode="edge")
    H, W = x.shape[1:3]
    stack = jnp.stack([p[:, i:i + H, j:j + W] for i in range(2 * r + 1) for j in range(2 * r + 1)])
    return stack.min(0), stack.max(0)


def init(key, C=16, n_body=2, s=2):
    ks = jax.random.split(key, n_body + 2)
    cin = 3 * s * s
    he = lambda k, kh, ci, co: jax.random.normal(k, (kh, kh, ci, co)) * np.sqrt(2.0 / (kh * kh * ci))
    p = {"head": {"w": he(ks[0], 3, cin, C), "b": jnp.zeros(C)}}
    for i in range(n_body):
        p[f"body{i}"] = {"w": he(ks[1 + i], 3, C, C), "b": jnp.zeros(C)}
    p["tail"] = {"w": he(ks[-1], 1, C, cin) * 0.1, "b": jnp.zeros(cin)}
    return p


def residual(p, x):
    s = int(round(np.sqrt(p["tail"]["w"].shape[-1] / 3)))
    h = conv(s2d(x, s), p["head"]["w"], p["head"]["b"], True)
    r = h
    i = 0
    while f"body{i}" in p:
        h = conv(h, p[f"body{i}"]["w"], p[f"body{i}"]["b"], True)
        i += 1
    if i:
        h = h + r
    return d2s(conv(h, p["tail"]["w"], p["tail"]["b"], False), s)


MARGIN = 0.0    # sobreimpulso permitido, proporcional al contraste local (0 en zonas planas)


def apply(p, x, k=1.0, limit=True, margin=None):
    m = MARGIN if margin is None else margin
    y = x + k * residual(p, x)
    if limit:
        lo, hi = local_minmax(x)
        d = hi - lo
        y = jnp.clip(y, lo - m * d, hi + m * d)
    return y


def macs_per_pixel(p):
    s = int(round(np.sqrt(p["tail"]["w"].shape[-1] / 3)))
    tot = sum(int(np.prod(v["w"].shape)) for v in p.values())
    return tot / (s * s)


# ------------------------------------------------------------ GNMScale
def load_gnmscale(model_dir):
    """SRNetFemto de GNMScale: cabeza 3x3, cuerpo 3x3 (depth), cola 1x1, residual.
    Layout (manifest): pesos[((oc*k*k+tap)*Sin+si)*4+comp], ic = si*4+comp, luego bias[Cout_pad]."""
    man = json.load(open(os.path.join(model_dir, "manifest.json")))
    wts = np.fromfile(os.path.join(model_dir, "weights.bin"), np.float32)
    width, depth = man["width"], man["depth"]
    off = 0

    def layer(cin, cout, k):
        nonlocal off
        sin = (cin + 3) // 4
        cout_pad = (cout + 3) // 4 * 4
        n = cout_pad * k * k * sin * 4
        w = wts[off:off + n].reshape(cout_pad, k * k, sin * 4)[:cout, :, :cin]
        off += n
        b = wts[off:off + cout_pad][:cout]
        off += cout_pad
        # -> HWIO
        return {"w": jnp.asarray(w.reshape(cout, k, k, cin).transpose(1, 2, 3, 0)), "b": jnp.asarray(b)}

    p = {"head": layer(3, width, 3)}
    for i in range(depth):
        p[f"body{i}"] = layer(width, width, 3)
    p["tail"] = layer(width, 3, 1)
    assert off == len(wts), (off, len(wts))
    return p


def gnmscale_apply(p, x, k=1.0, slope=0.1):
    lr = lambda y: jnp.where(y > 0, y, slope * y)
    h = lr(jax.lax.conv_general_dilated(x, p["head"]["w"], (1, 1), "SAME",
                                        dimension_numbers=("NHWC", "HWIO", "NHWC")) + p["head"]["b"])
    i = 0
    while f"body{i}" in p:
        h = lr(jax.lax.conv_general_dilated(h, p[f"body{i}"]["w"], (1, 1), "SAME",
                                            dimension_numbers=("NHWC", "HWIO", "NHWC")) + p[f"body{i}"]["b"])
        i += 1
    t = jax.lax.conv_general_dilated(h, p["tail"]["w"], (1, 1), "SAME",
                                     dimension_numbers=("NHWC", "HWIO", "NHWC")) + p["tail"]["b"]
    return x + k * t
