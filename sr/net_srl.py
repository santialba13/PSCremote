"""NXSR-L ("ligero"): realce de detalle guiado por una red chica, pensado para
correr junto con la generacion de fotogramas en una PS4 base (~1/10 del calculo
de NXSR Rapido).

  Y      luma del fotograma (resolucion completa)
  b1     Y - G3(Y)           banda fina (radio 1)      G3 = binomial 3x3
  b2     G3(Y) - G5(Y)       banda media (radio 2)     G5 = binomial 5x5
  guia   red de 4 capas 3x3 a 1/8 de resolucion; entra el color medio de la celda
         y la energia de cada banda, sale por celda: ganancia de b1, ganancia de
         b2 y umbral de grano. Se sube con bilineal a resolucion completa.
  salida Y' = Y + k * (g1 * nucleo(b1, t) + g2 * nucleo(b2, t))
         nucleo(b, t) = b sin la parte |b| < t (el grano y el dithering no se realzan)
         Y' acotada al rango 3x3 local de Y (+ MARGIN de ese rango): sin halos
         RGB' = RGB + (Y' - Y)   (solo luma: sin bordes de color)

La red decide donde y cuanto realzar (bordes y texturas si, zonas planas y ruido
no); el realce en si es un filtro fijo barato. Es la referencia exacta de los
kernels srl_* de shaders/nxsrl.cl.
"""
import jax
import jax.numpy as jnp
import numpy as np

CELL = 8
C = 12
MARGIN = 0.25


def luma(x):
    return (x[..., 0] * 0.299 + x[..., 1] * 0.587) + x[..., 2] * 0.114


def _sep(Y, k):
    r = len(k) // 2
    p = jnp.pad(Y, ((0, 0), (r, r), (r, r)), mode="edge")
    H, W = Y.shape[1:]
    h = sum(k[i] * p[:, :, i:i + W] for i in range(len(k)))
    return sum(k[i] * h[:, i:i + H, :] for i in range(len(k)))


def bands(Y):
    g3 = _sep(Y, [0.25, 0.5, 0.25])
    g5 = _sep(Y, [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16])
    return Y - g3, g3 - g5


def avgpool(x, f):
    B, H, W, Cc = x.shape
    return x.reshape(B, H // f, f, W // f, f, Cc).mean((2, 4))


def conv(x, p, act=True):
    y = jax.lax.conv_general_dilated(x, p["w"], (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")) + p["b"]
    return jnp.where(y > 0, y, 0.1 * y) if act else y


def init(key, c=C):
    ks = jax.random.split(key, 4)
    he = lambda k, ci, co: jax.random.normal(k, (3, 3, ci, co)) * np.sqrt(2.0 / (9 * ci))
    p = {"c0": {"w": he(ks[0], 5, c), "b": jnp.zeros(c)},
         "c1": {"w": he(ks[1], c, c), "b": jnp.zeros(c)},
         "c2": {"w": he(ks[2], c, c), "b": jnp.zeros(c)},
         "out": {"w": he(ks[3], c, 3) * 0.1, "b": jnp.array([0.0, -0.5, -1.0])}}
    return p


def features(x):
    Y = luma(x)
    b1, b2 = bands(Y)
    f = jnp.concatenate([x, jnp.abs(b1)[..., None] * 8.0, jnp.abs(b2)[..., None] * 8.0], -1)
    return avgpool(f, CELL), Y, b1, b2


def guide(p, f):
    h = conv(f, p["c0"])
    r = h
    h = conv(h, p["c1"])
    h = conv(h, p["c2"]) + r
    o = conv(h, p["out"], act=False)
    g1 = jax.nn.softplus(o[..., 0])
    g2 = jax.nn.softplus(o[..., 1])
    t = jax.nn.softplus(o[..., 2]) * 0.02
    return jnp.stack([g1, g2, t], -1)


def local_minmax(Y):
    p = jnp.pad(Y, ((0, 0), (1, 1), (1, 1)), mode="edge")
    H, W = Y.shape[1:]
    s = jnp.stack([p[:, i:i + H, j:j + W] for i in range(3) for j in range(3)])
    return s.min(0), s.max(0)


def apply(p, x, k=1.0, margin=None):
    m = MARGIN if margin is None else margin
    f, Y, b1, b2 = features(x)
    g = guide(p, f)
    B, h, w, _ = g.shape
    G = jax.image.resize(g, (B, h * CELL, w * CELL, 3), "bilinear")
    t = G[..., 2]
    core = lambda b: b - jnp.clip(b, -t, t)
    Yn = Y + k * (G[..., 0] * core(b1) + G[..., 1] * core(b2))
    lo, hi = local_minmax(Y)
    d = hi - lo
    Yn = jnp.clip(Yn, lo - m * d, hi + m * d)
    return jnp.clip(x + (Yn - Y)[..., None], 0.0, 1.0)


def macs_per_pixel(c=C):
    return 9 * (5 * c + 2 * c * c + 3 * c) / CELL ** 2
