"""NXFG-2: generacion de fotogramas liviana y estable para la GPU de PS4.

Lo que mas pesa en la calidad es el movimiento (ver docs/resultados.md: la red
anterior apenas le ganaba al block-matching). NXFG-2 gasta la GPU ahi y deja la
red chica:

  movimiento  model/bm2.py: piramide a media resolucion, 4 niveles, vectores
              sub-pixel con confianza en una malla 1/16 (alcance ~200 px)
  nivel A     entrada 1/16 (fotogramas 1/16, warps, vector, confianza), interno
              1/32, 32 canales -> corrige los dos flujos y estima la oclusion
  nivel B     entrada 1/8 + flujo del nivel A, interno 1/16, 24 canales ->
              detalle de bordes a 8 px
  salida      flujo y mascara a 1/8, warp de los dos fotogramas reales a
              resolucion completa con bloques superpuestos (las 4 celdas vecinas
              con su peso bilineal) y eleccion por pixel del vector que mejor hace
              coincidir los dos reales (sin bloques ni halos), con protecciones
              fijas que no dependen de la red:
                * fantasmas de HUD: una muestra que cae en un pixel quieto con un
                  vector que se mueve no se usa (se toma la del otro fotograma)
                * pixel quieto (HUD, textos, zonas sin cambio): si I0 ~ I1 en ese
                  pixel se usa su promedio, sin warp -> sin temblor ni fantasmas
                * corte de escena (en el plugin): si la confianza media es mala
                  se repite el fotograma real en vez de inventar uno

A 1080p son ~0.25 GMAC por fotograma generado (la anterior: 1.17).
"""
import jax
import jax.numpy as jnp
import numpy as np

from net import _conv_init, conv, lrelu, avgpool, warp, upsample5  # noqa: F401

C_A = 32
C_B = 24
N_MID_A = 2
N_MID_B = 1
CONF_SCALE = 10.0
STATIC_TAU = 0.03      # diferencia media RGB por debajo de la cual el pixel se considera quieto


def block_init(key, cin, c, cout=5, n_mid=2):
    ks = jax.random.split(key, n_mid + 3)
    p = {"c0": _conv_init(ks[0], 3, cin, c), "c1": _conv_init(ks[1], 3, c, c)}
    for i in range(n_mid):
        p[f"m{i}"] = _conv_init(ks[2 + i], 3, c, c)
    p["out"] = _conv_init(ks[-1], 3, c, cout)
    p["out"]["w"] = p["out"]["w"] * 0.1
    return p


def block(p, x):
    """c0 stride 2, c1, capas medias (+ residual de c1), resize x2, salida."""
    h = lrelu(conv(p["c0"], x, 2))
    h = lrelu(conv(p["c1"], h))
    r = h
    i = 0
    while f"m{i}" in p:
        h = lrelu(conv(p[f"m{i}"], h))
        i += 1
    h = h + r
    B, H, W, C = h.shape
    h = jax.image.resize(h, (B, H * 2, W * 2, C), "bilinear")
    return conv(p["out"], h)


def init(key):
    ka, kb = jax.random.split(key)
    return {"A": block_init(ka, 15, C_A, n_mid=N_MID_A),
            "B": block_init(kb, 17, C_B, n_mid=N_MID_B)}


def bm_flow(bm):
    """bm (B, h, w, 3): (dy, dx) en px de media resolucion + confianza -> f0, f1 en px de 1/16."""
    v = bm[..., 1::-1] / 8.0
    return -v, v, bm[..., 2:3] * CONF_SCALE


def predict_lowres(params, i0, i1, bm):
    """-> (flujo t->0, flujo t->1, logit de mascara) a 1/8, flujo en px de 1/8."""
    a0, a1 = avgpool(i0, 16), avgpool(i1, 16)
    f0, f1, cf = bm_flow(bm)
    x = jnp.concatenate([a0, a1, warp(a0, f0), warp(a1, f1), f1, cf], -1)
    o = block(params["A"], x)
    f5 = jnp.concatenate([f0 + o[..., 0:2], f1 + o[..., 2:4], o[..., 4:5]], -1)
    u = upsample5(f5, 2)
    q0, q1 = avgpool(i0, 8), avgpool(i1, 8)
    g0, g1, m = u[..., 0:2], u[..., 2:4], u[..., 4:5]
    x = jnp.concatenate([q0, q1, warp(q0, g0), warp(q1, g1), g0, g1, m], -1)
    d = block(params["B"], x)
    return g0 + d[..., 0:2], g1 + d[..., 2:4], m + d[..., 4:5]


def static_weight(i0, i1):
    d = jnp.abs(i0 - i1).mean(-1, keepdims=True)
    return jnp.clip(1.0 - d / STATIC_TAU, 0.0, 1.0)


CAND_SIGMA = 0.1       # temperatura de la eleccion de vector por pixel
STATIC_SRC_TAU = 0.015 # diferencia de luma (media resolucion) de un pixel "quieto" (HUD, textos)
MIX_EPS = 1e-3


def static_src_mask(i0, i1):
    """1 donde el pixel (media resolucion) no cambia entre los dos reales."""
    def luma(x):
        return (x[..., 0] * 0.299 + x[..., 1] * 0.587) + x[..., 2] * 0.114
    d = jnp.abs(luma(avgpool(i0, 2)) - luma(avgpool(i1, 2)))
    s = (d < STATIC_SRC_TAU).astype(jnp.float32)
    # apertura 3x3 (erosion + dilatacion): las coincidencias sueltas de una textura
    # que se mueve no cuentan; el HUD (zonas quietas grandes) queda entero
    s = jax.lax.reduce_window(s, jnp.inf, jax.lax.min, (1, 3, 3), (1, 1, 1), "SAME")
    return jax.lax.reduce_window(s, -jnp.inf, jax.lax.max, (1, 3, 3), (1, 1, 1), "SAME")


def _sample_half(mask, F, H, W):
    """mask (B, H/2, W/2) en la posicion (x + F) / 2 (truncado), como el kernel."""
    B = F.shape[0]
    gy, gx = jnp.meshgrid(jnp.arange(H, dtype=jnp.float32), jnp.arange(W, dtype=jnp.float32), indexing="ij")
    xs = jnp.clip(jnp.floor((gx[None] + F[..., 0]) * 0.5).astype(jnp.int32), 0, W // 2 - 1)
    ys = jnp.clip(jnp.floor((gy[None] + F[..., 1]) * 0.5).astype(jnp.int32), 0, H // 2 - 1)
    return mask[jnp.arange(B)[:, None, None], ys, xs]


def _corners(low, s, H, W):
    """Las 4 celdas de la malla que rodean a cada pixel y sus pesos bilineales."""
    h, w = low.shape[1:3]
    gy = (jnp.arange(H) + 0.5) / s - 0.5
    gx = (jnp.arange(W) + 0.5) / s - 0.5
    fy = (gy - jnp.floor(gy))[None, :, None]
    fx = (gx - jnp.floor(gx))[None, None, :]
    y0 = jnp.clip(jnp.floor(gy).astype(jnp.int32), 0, h - 1)
    x0 = jnp.clip(jnp.floor(gx).astype(jnp.int32), 0, w - 1)
    y1 = jnp.clip(jnp.floor(gy).astype(jnp.int32) + 1, 0, h - 1)
    x1 = jnp.clip(jnp.floor(gx).astype(jnp.int32) + 1, 0, w - 1)
    r0, r1 = low[:, y0], low[:, y1]
    return [(r0[:, :, x0], (1 - fy) * (1 - fx)), (r0[:, :, x1], (1 - fy) * fx),
            (r1[:, :, x0], fy * (1 - fx)), (r1[:, :, x1], fy * fx)]


def cand_warp(i0, i1, low, s):
    """Warp por bloques superpuestos con eleccion por pixel (sin bloques ni halos):
    cada una de las 4 celdas vecinas aporta su warp con su peso bilineal (OBMC),
    multiplicado por lo bien que coinciden sus dos muestras (softmin de |w0 - w1|).
    Anti-fantasmas de HUD: si una muestra viene de un pixel quieto (HUD, texto), el
    vector no es cero y las dos muestras no coinciden, esa muestra no se usa (se
    toma la del otro fotograma)."""
    B, h, w, _ = low.shape
    H, W = h * s, w * s
    S = static_src_mask(i0, i1)
    outs, errs, bws = [], [], []
    for c, bw in _corners(low, s, H, W):
        F0, F1 = c[..., 0:2] * s, c[..., 2:4] * s
        a, b = warp(i0, F0), warp(i1, F1)
        m = jax.nn.sigmoid(c[..., 4:5])
        e = jnp.abs(a - b).sum(-1)
        # solo cuenta si ademas las dos muestras no coinciden (un fantasma de HUD)
        dis = jax.lax.stop_gradient(jnp.clip((e - 0.06) / 0.12, 0.0, 1.0))
        mv0 = (jnp.abs(F0).sum(-1) > 1.0).astype(jnp.float32)
        mv1 = (jnp.abs(F1).sum(-1) > 1.0).astype(jnp.float32)
        r0 = (1 - _sample_half(S, F0, H, W) * mv0 * dis)[..., None]
        r1 = (1 - _sample_half(S, F1, H, W) * mv1 * dis)[..., None]
        wa, wb = m * r0, (1 - m) * r1
        outs.append((wa * a + wb * b + MIX_EPS * 0.5 * (a + b)) / (wa + wb + MIX_EPS))
        errs.append(e)
        bws.append(jnp.broadcast_to(bw, (B, H, W)))
    E = jnp.stack(errs, 0)
    wts = jnp.stack(bws, 0) * jnp.exp(-(E - jax.lax.stop_gradient(E.min(0))) / CAND_SIGMA)
    wts = wts / wts.sum(0)
    return (wts[..., None] * jnp.stack(outs, 0)).sum(0)


def interpolate(params, i0, i1, bm):
    f0, f1, m = predict_lowres(params, i0, i1, bm)
    low = jnp.concatenate([f0, f1, m], -1)
    mix = cand_warp(i0, i1, low, 8)
    s = static_weight(i0, i1)
    out = s * (0.5 * (i0 + i1)) + (1 - s) * mix
    # para las perdidas: flujo y mascara del candidato bilineal
    B, H, W, _ = m.shape
    F0 = jax.image.resize(f0, (B, H * 8, W * 8, 2), "bilinear") * 8.0
    F1 = jax.image.resize(f1, (B, H * 8, W * 8, 2), "bilinear") * 8.0
    M = jax.nn.sigmoid(jax.image.resize(m, (B, H * 8, W * 8, 1), "bilinear"))
    return out, (F0, F1, M, warp(i0, F0), warp(i1, F1))


def count_params(params):
    return sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params))


def macs_1080p():
    """MAC por fotograma generado a 1920x1088 (solo convoluciones)."""
    g = 68 * 120
    a = g / 4 * 9 * (15 * C_A + C_A * C_A * (1 + N_MID_A)) + g * 9 * C_A * 5
    b = g * 9 * (17 * C_B + C_B * C_B * (1 + N_MID_B)) + 4 * g * 9 * C_B * 5
    return a + b
