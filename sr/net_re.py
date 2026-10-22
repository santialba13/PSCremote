"""NXRE ("realismo"): transformacion fotorrealista aprendida, barata y estable.

Idea de "Deep Bilateral Learning" (HDRNet, Gharbi et al. 2017) aplicada a lo que
hacen los modelos de fotorrealismo para juegos (EPE, REGEN): una red chica mira el
fotograma entero a baja resolucion (~128 x 72) y decide, por zona de la imagen Y por
nivel de brillo, una transformacion de color afin (matriz 3x4). Cada pixel toma la
suya de esa malla (bilineal en el espacio, lineal entre niveles de brillo) y la aplica.

  baja   promedio f x f del fotograma (f = W / 128 redondeado): ~128 x 72 x 3
  red    conv 3x3 de paso 2: 3->16 (1/2), 16->16 (1/4), 16->24 (1/8), 24->24 (1/16)
         + 24->24 (contexto global de la escena: dia, noche, cielo, interior)
         rama local 16->24 a 1/4 + el contexto global subido a 1/4 (bilineal)
         24->24, 24->(D x 12) lineal: la malla (D = 8 niveles de brillo, 12 = afin 3x4)
  guia   g = clamp(m . rgb + c, 0, 1) por pixel (aprendida, empieza en la luma)
  salida T(x) = A(x, y, g) [r g b 1];  y = x + s * (T(x) - x),  s = intensidad

Al ser una transformacion de color suave por zonas no inventa texturas, no hace
halos ni parpadeos (la malla cambia lento entre fotogramas; el plugin ademas la
promedia con la del fotograma anterior). Es la referencia exacta de shaders/nxre.cl.
"""
import jax
import jax.numpy as jnp
import numpy as np

D = 8          # niveles de brillo de la malla
LOW_W = 128    # ancho de la imagen de baja resolucion que ve la red


def pool_factor(W):
    return max(1, int(W / LOW_W + 0.5))


def avgpool(x, f):
    B, H, W, C = x.shape
    h, w = H // f, W // f
    return x[:, :h * f, :w * f].reshape(B, h, f, w, f, C).mean((2, 4))


def conv(x, p, stride=1, act=True):
    y = jax.lax.conv_general_dilated(x, p["w"], (stride, stride), ((1, 1), (1, 1)),
                                     dimension_numbers=("NHWC", "HWIO", "NHWC")) + p["b"]
    return jnp.where(y > 0, y, 0.1 * y) if act else y


LAYERS = [  # nombre, cin, cout, paso, activacion
    ("s1", 3, 16, 2, True), ("s2", 16, 16, 2, True), ("s3", 16, 24, 2, True), ("s4", 24, 24, 2, True),
    ("g1", 24, 24, 1, True), ("l1", 16, 24, 1, True), ("l2", 24, 24, 1, True), ("out", 24, D * 12, 1, False),
]


def init(key):
    ks = jax.random.split(key, len(LAYERS))
    p = {}
    for k, (n, ci, co, _, _) in zip(ks, LAYERS):
        w = jax.random.normal(k, (3, 3, ci, co)) * np.sqrt(2.0 / (9 * ci))
        p[n] = {"w": w * (0.0 if n == "out" else 1.0), "b": jnp.zeros(co)}
    p["guide"] = {"m": jnp.array([0.299, 0.587, 0.114]), "c": jnp.zeros(())}
    return p


def grid(p, x):
    """x: B x H x W x 3 en [0, 1] -> malla B x gh x gw x D x 12 (afin residual)."""
    lo = avgpool(x, pool_factor(x.shape[2]))
    a = conv(lo, p["s1"], 2)
    b = conv(a, p["s2"], 2)
    c = conv(b, p["s3"], 2)
    d = conv(c, p["s4"], 2)
    g = conv(d, p["g1"])
    B, h, w, _ = b.shape
    u = jax.image.resize(g, (B, h, w, g.shape[-1]), "bilinear")
    e = conv(b, p["l1"]) + u
    f = conv(e, p["l2"])
    o = conv(f, p["out"], act=False)
    return o.reshape(B, h, w, D, 12)


def guide(p, x):
    return jnp.clip(x @ p["guide"]["m"] + p["guide"]["c"], 0.0, 1.0)


def slice_apply(G, gd, x, s=1.0):
    """G: B x gh x gw x D x 12, gd: B x H x W guia, x: B x H x W x 3."""
    B, H, W, _ = x.shape
    gh, gw = G.shape[1:3]
    # espacial: bilineal half-pixel con bordes repetidos (como jax.image.resize)
    Gs = jax.image.resize(G.reshape(B, gh, gw, D * 12), (B, H, W, D * 12), "bilinear").reshape(B, H, W, D, 12)
    z = gd * D - 0.5
    z0 = jnp.floor(z)
    fz = z - z0
    i0 = jnp.clip(z0, 0, D - 1).astype(jnp.int32)
    i1 = jnp.clip(z0 + 1, 0, D - 1).astype(jnp.int32)
    c0 = jnp.take_along_axis(Gs, i0[..., None, None], axis=3)[..., 0, :]
    c1 = jnp.take_along_axis(Gs, i1[..., None, None], axis=3)[..., 0, :]
    A = c0 * (1 - fz)[..., None] + c1 * fz[..., None]           # B x H x W x 12
    A = A.reshape(B, H, W, 3, 4)
    d = jnp.einsum("bhwij,bhwj->bhwi", A[..., :3], x) + A[..., 3]   # T(x) - x (afin residual)
    return jnp.clip(x + s * d, 0.0, 1.0)


def apply(p, x, s=1.0):
    return slice_apply(grid(p, x), guide(p, x), x, s)
