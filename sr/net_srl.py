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
MARGIN = 0.25    # Ligero; Calidad usa 0.35 (sr/train_srl.py --margin)
SUB = 2          # solo para FEAT = "sub" (experimento)


def luma(x):
    return (x[..., 0] * 0.299 + x[..., 1] * 0.587) + x[..., 2] * 0.114


def _sep(Y, k):
    r = len(k) // 2
    p = jnp.pad(Y, ((0, 0), (r, r), (r, r)), mode="edge")
    H, W = Y.shape[1:]
    h = sum(k[i] * p[:, :, i:i + W] for i in range(len(k)))
    return sum(k[i] * h[:, i:i + H, :] for i in range(len(k)))


BANDS = 2        # 3 = tambien la banda ancha b3 = G5 - G9 (Calidad)
K9 = [1 / 256, 8 / 256, 28 / 256, 56 / 256, 70 / 256, 56 / 256, 28 / 256, 8 / 256, 1 / 256]


def bands(Y, n=2):
    g3 = _sep(Y, [0.25, 0.5, 0.25])
    g5 = _sep(Y, [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16])
    if n == 3:
        return Y - g3, g3 - g5, g5 - _sep(Y, K9)
    return Y - g3, g3 - g5


def avgpool(x, f):
    B, H, W, Cc = x.shape
    return x.reshape(B, H // f, f, W // f, f, Cc).mean((2, 4))


def conv(x, p, act=True):
    y = jax.lax.conv_general_dilated(x, p["w"], (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")) + p["b"]
    return jnp.where(y > 0, y, 0.1 * y) if act else y


def init(key, c=C, n_mid=2, nb=None):
    """c0 (5 -> c), n_mid capas c -> c (la ultima suma el residual de c0), out (c -> 3).
    Ligero: c = 12, n_mid = 2. Calidad: c = 16, n_mid = 4."""
    ks = jax.random.split(key, n_mid + 2)
    he = lambda k, ci, co: jax.random.normal(k, (3, 3, ci, co)) * np.sqrt(2.0 / (9 * ci))
    nb = BANDS if nb is None else nb
    p = {"c0": {"w": he(ks[0], 3 + nb, c), "b": jnp.zeros(c)}}
    for i in range(n_mid):
        p[f"c{i + 1}"] = {"w": he(ks[i + 1], c, c), "b": jnp.zeros(c)}
    p["out"] = {"w": he(ks[-1], c, nb + 1) * 0.1, "b": jnp.array([0.0, -0.5, -1.0, -1.0][:nb] + [-1.0])}
    return p


def n_mid(p):
    return sum(1 for k in p if k != "c0" and k.startswith("c"))


# caracteristicas de la guia: "full" = todos los pixeles (la del plugin: medido, es la mas
# estable al mover la camara: 0.06/255 contra 0.14 de "half" y 0.28 de "sub")
FEAT = "full"


def features(x):
    Y = luma(x)
    b1, b2 = bands(Y)
    if FEAT == "full":
        bs = bands(Y, BANDS)
        f = jnp.concatenate([x] + [jnp.abs(b)[..., None] * 8.0 for b in bs], -1)
        return avgpool(f, CELL), Y, bs
    if FEAT == "half":
        # la guia mide la energia de detalle sobre la luma a media resolucion (promedio
        # 2x2, sin aliasing al moverse la camara) y el color medio exacto de la celda
        h1, h2 = bands(avgpool(Y[..., None], 2)[..., 0])
        e = jnp.stack([jnp.abs(h1) * 8.0, jnp.abs(h2) * 8.0], -1)
        return jnp.concatenate([avgpool(x, CELL), avgpool(e, CELL // 2)], -1), Y, b1, b2
    f = jnp.concatenate([x, jnp.abs(b1)[..., None] * 8.0, jnp.abs(b2)[..., None] * 8.0], -1)
    # promedio por celda de los pixeles (SUB*i, SUB*j)
    return avgpool(f[:, ::SUB, ::SUB], CELL // SUB), Y, b1, b2


def guide(p, f):
    h = conv(f, p["c0"])
    r = h
    n = n_mid(p)
    for i in range(n):
        h = conv(h, p[f"c{i + 1}"]) + (r if i == n - 1 else 0.0)
    o = conv(h, p["out"], act=False)
    n = o.shape[-1] - 1
    return jnp.concatenate([jax.nn.softplus(o[..., :n]), jax.nn.softplus(o[..., n:]) * 0.02], -1)


def local_minmax(Y):
    p = jnp.pad(Y, ((0, 0), (1, 1), (1, 1)), mode="edge")
    H, W = Y.shape[1:]
    s = jnp.stack([p[:, i:i + H, j:j + W] for i in range(3) for j in range(3)])
    return s.min(0), s.max(0)


def apply(p, x, k=1.0, margin=None):
    m = MARGIN if margin is None else margin
    f, Y, bs = features(x)
    g = guide(p, f)
    B, h, w, nc = g.shape
    G = jax.image.resize(g, (B, h * CELL, w * CELL, nc), "bilinear")
    t = G[..., nc - 1]
    core = lambda b: b - jnp.clip(b, -t, t)
    Yn = Y + k * sum(G[..., i] * core(b) for i, b in enumerate(bs))
    lo, hi = local_minmax(Y)
    d = hi - lo
    Yn = jnp.clip(Yn, lo - m * d, hi + m * d)
    return jnp.clip(x + (Yn - Y)[..., None], 0.0, 1.0)


def macs_per_pixel(c=C, n=2):
    return 9 * (5 * c + n * c * c + 3 * c) / CELL ** 2
