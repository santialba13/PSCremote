"""NXFlow: red de interpolacion de fotogramas pensada para la GPU de PS4.

Diseno hibrido: block-matching clasico + refinamiento aprendido.

  bm       busqueda de movimiento simetrica por bloques de 8x8 (la misma idea
           que los pases me3/me2/me1 de VCFG): vector v por bloque, malla 1/8.
  nivel A  entrada 1/8 (fotogramas + warps con v + v), interno 1/16
           -> corrige el flujo t->0 / t->1 y predice la mascara de oclusion
  nivel B  entrada 1/8 + warps del nivel A, interno 1/16 -> residuo
  nivel C  entrada 1/4, 16 canales, interno 1/8 -> residuo fino y mascara
           con resolucion de 4 px (bordes de objetos y HUD)
  salida   flujo y mascara a 1/4, se suben x4 y se hace backward-warp +
           mezcla a resolucion completa (un solo pase barato en la GPU).

A 1080p son ~1.2 GMAC por fotograma generado.
"""
import jax
import jax.numpy as jnp
import numpy as np

C_A = 32
C_B = 32
C_C = 16


def _conv_init(key, k, cin, cout):
    fan = k * k * cin
    w = jax.random.normal(key, (k, k, cin, cout)) * np.sqrt(2.0 / fan)
    return {"w": w.astype(jnp.float32), "b": jnp.zeros((cout,), jnp.float32)}


def conv(p, x, stride=1):
    y = jax.lax.conv_general_dilated(
        x, p["w"], (stride, stride), "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + p["b"]


def lrelu(x):
    return jnp.where(x > 0, x, 0.1 * x)


def block_init(key, cin, c, cout=5, n_mid=3):
    ks = jax.random.split(key, n_mid + 3)
    p = {"c0": _conv_init(ks[0], 3, cin, c), "c1": _conv_init(ks[1], 3, c, c)}
    for i in range(n_mid):
        p[f"m{i}"] = _conv_init(ks[2 + i], 3, c, c)
    p["out"] = _conv_init(ks[-1], 3, c, cout)
    p["out"]["w"] = p["out"]["w"] * 0.1
    return p


def block(p, x, down):
    """down: factor total respecto a la entrada (2 o 4) antes de las capas medias."""
    h = lrelu(conv(p["c0"], x, 2))
    h = lrelu(conv(p["c1"], h, 2 if down == 4 else 1))
    r = h
    i = 0
    while f"m{i}" in p:
        h = lrelu(conv(p[f"m{i}"], h))
        i += 1
    h = h + r
    B, H, W, C = h.shape
    h = jax.image.resize(h, (B, H * down, W * down, C), "bilinear")
    return conv(p["out"], h)


def init(key):
    ka, kb, kc = jax.random.split(key, 3)
    return {"A": block_init(ka, 6 + 6 + 2, C_A), "B": block_init(kb, 6 + 6 + 5, C_B),
            "C": block_init(kc, 6 + 6 + 5, C_C, n_mid=2)}


def avgpool(x, f):
    B, H, W, C = x.shape
    return x.reshape(B, H // f, f, W // f, f, C).mean((2, 4))


def warp(img, flow):
    """Backward warp bilineal: out(x) = img(x + flow(x)). flow (B,H,W,2) en px (dx,dy)."""
    B, H, W, C = img.shape
    gy, gx = jnp.meshgrid(jnp.arange(H, dtype=jnp.float32),
                          jnp.arange(W, dtype=jnp.float32), indexing="ij")
    x = jnp.clip(gx[None] + flow[..., 0], 0, W - 1)
    y = jnp.clip(gy[None] + flow[..., 1], 0, H - 1)
    x0 = jnp.floor(x); y0 = jnp.floor(y)
    fx = (x - x0)[..., None]; fy = (y - y0)[..., None]
    x0 = x0.astype(jnp.int32); y0 = y0.astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, W - 1); y1 = jnp.minimum(y0 + 1, H - 1)
    bi = jnp.arange(B)[:, None, None]
    a = img[bi, y0, x0]; b = img[bi, y0, x1]
    c = img[bi, y1, x0]; d = img[bi, y1, x1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def bm_init(bm):
    """bm: vectores (dy, dx) en px de resolucion completa a malla 1/8 -> (f0, f1) en px de 1/8."""
    v = bm[..., ::-1] / 8.0
    return -v, v


def upsample_flow(f, s):
    B, H, W, C = f.shape
    return jax.image.resize(f, (B, H * s, W * s, C), "bilinear") * s


def upsample5(x, s):
    """flow5 (f0, f1, mascara) a una malla s veces mas fina: el flujo se escala, la mascara no."""
    B, H, W, C = x.shape
    u = jax.image.resize(x, (B, H * s, W * s, C), "bilinear")
    return jnp.concatenate([u[..., :4] * s, u[..., 4:]], -1)


def _refine(p, a0, a1, f0, f1, m):
    x = jnp.concatenate([a0, a1, warp(a0, f0), warp(a1, f1), f0, f1, m], -1)
    d = block(p, x, 2)
    return f0 + d[..., 0:2], f1 + d[..., 2:4], m + d[..., 4:5]


def predict_lowres(params, i0, i1, bm):
    """Devuelve (flow t->0, flow t->1, mask_logit) a 1/4 de resolucion, flujo en px de 1/4."""
    a0, a1 = avgpool(i0, 8), avgpool(i1, 8)
    f0, f1 = bm_init(bm)
    x = jnp.concatenate([a0, a1, warp(a0, f0), warp(a1, f1), f1], -1)
    o = block(params["A"], x, 2)
    f0, f1, m = f0 + o[..., 0:2], f1 + o[..., 2:4], o[..., 4:5]
    f0, f1, m = _refine(params["B"], a0, a1, f0, f1, m)
    u = upsample5(jnp.concatenate([f0, f1, m], -1), 2)
    q0, q1 = avgpool(i0, 4), avgpool(i1, 4)
    return _refine(params["C"], q0, q1, u[..., 0:2], u[..., 2:4], u[..., 4:5])


def interpolate(params, i0, i1, bm):
    f0, f1, m = predict_lowres(params, i0, i1, bm)
    F0, F1 = upsample_flow(f0, 4), upsample_flow(f1, 4)
    B, H, W, _ = m.shape
    M = jax.nn.sigmoid(jax.image.resize(m, (B, H * 4, W * 4, 1), "bilinear"))
    w0, w1 = warp(i0, F0), warp(i1, F1)
    out = M * w0 + (1 - M) * w1
    return out, (F0, F1, M, w0, w1)


def count_params(params):
    return sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params))
