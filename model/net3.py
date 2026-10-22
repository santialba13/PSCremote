"""NXFG-3: generacion de fotogramas mas barata, para ir junto con el realismo (NXRE).

Medido en GTA V (tripletes del juego a 30 fps, model/eval3.py):
  * las redes de NXFG-2 (niveles A y B) no mejoraban nada medible (+0.04 dB);
  * una red que decide el lado de cada pixel en las oclusiones tampoco aprende
    (y aun perfecta, a 1/4 de resolucion, valdria ~+0.9 dB);
  * mas candidatos, un nivel de bloques mas fino o un movimiento global de camara:
    +-0.05 dB.
El metodo por bloques esta en su techo de calidad para esta GPU, asi que NXFG-3 gana
en lo que si se puede: hace lo mismo con mucho menos trabajo.

  movimiento  model/bm2.py, sin cambios (bloques a 1/16, 4 niveles, sub-pixel)
  eleccion    a 1/4 de resolucion, sobre los promedios 4x4 de los dos reales: los 9
              vectores de la malla 1/8 alrededor de cada pixel (3x3), pesados por
              cercania (gaussiana, SEL_PRIOR celdas) y por lo bien que coinciden sus
              dos muestras (softmin, SEL_SIGMA) -> un flujo (f0, f1) por pixel de 1/4
  warp        a resolucion completa con ESE flujo (subido bilineal): 2 muestras por
              pixel en vez de 8 (NXFG-2: 4 candidatos x 2), con las protecciones de
              siempre: anti-fantasmas de HUD, pixeles quietos y corte de escena.

Misma calidad que NXFG-2 en GTA V; a 1080p ~40 % menos instrucciones de GPU.
"""
import jax
import jax.numpy as jnp

import net2
from net import avgpool, warp, upsample5

Q = 4              # resolucion de la eleccion (1/Q)
SEL_SIGMA = 0.05   # temperatura del softmin de la eleccion
SEL_PRIOR = 0.6    # ancho (en celdas de la malla 1/8) del peso por cercania
SCENE_CUT = 0.12   # confianza media del block-matching por encima de esto: se repite el real


def low_flow(bm):
    """bm (B, h, w, 3) de bm2 (malla 1/16) -> flow5 a 1/8 (f0, f1 en px de 1/8, mascara 0)."""
    f0, f1, cf = net2.bm_flow(bm)
    return upsample5(jnp.concatenate([f0, f1, jnp.zeros_like(cf)], -1), 2)


def select(i0, i1, low, sigma=None, prior=None):
    """Eleccion a 1/Q: flujo (f0x, f0y, f1x, f1y) en px de resolucion completa."""
    sigma = SEL_SIGMA if sigma is None else sigma
    prior = SEL_PRIOR if prior is None else prior
    q0, q1 = avgpool(i0, Q), avgpool(i1, Q)
    B, h, w, _ = q0.shape
    s = 8 // Q
    hl, wl = low.shape[1:3]
    gy = (jnp.arange(h) + 0.5) / s - 0.5
    gx = (jnp.arange(w) + 0.5) / s - 0.5
    cy = jnp.floor(gy + 0.5).astype(jnp.int32)
    cx = jnp.floor(gx + 0.5).astype(jnp.int32)
    Fs, errs, bws = [], [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy = jnp.clip(cy + dy, 0, hl - 1)
            xx = jnp.clip(cx + dx, 0, wl - 1)
            c = low[:, yy][:, :, xx]
            d2 = (gy - (cy + dy))[:, None] ** 2 + (gx - (cx + dx))[None, :] ** 2
            bws.append(jnp.broadcast_to(jnp.exp(-d2 / (2 * prior * prior))[None], (B, h, w)))
            F0, F1 = c[..., 0:2] * s, c[..., 2:4] * s
            a, b = warp(q0, F0), warp(q1, F1)
            errs.append(jnp.abs(a - b).sum(-1))
            Fs.append(jnp.concatenate([F0, F1], -1) * Q)
    E = jnp.stack(errs, 0)
    wt = jnp.stack(bws, 0) * jnp.exp(-(E - E.min(0)) / sigma)
    wt = wt / wt.sum(0)
    return (wt[..., None] * jnp.stack(Fs, 0)).sum(0)


def warp1(i0, i1, Fq):
    """Warp de los dos reales con el flujo elegido (subido bilineal) + protecciones."""
    B, H, W, _ = i0.shape
    F = jax.image.resize(Fq, (B, H, W, 4), "bilinear")
    F0, F1 = F[..., 0:2], F[..., 2:4]
    a, b = warp(i0, F0), warp(i1, F1)
    S_ = net2.static_src_mask(i0, i1)
    e = jnp.abs(a - b).sum(-1)
    dis = jnp.clip((e - 0.06) / 0.12, 0.0, 1.0)
    mv0 = (jnp.abs(F0).sum(-1) > 1.0).astype(jnp.float32)
    mv1 = (jnp.abs(F1).sum(-1) > 1.0).astype(jnp.float32)
    r0 = (1 - net2._sample_half(S_, F0, H, W) * mv0 * dis)[..., None]
    r1 = (1 - net2._sample_half(S_, F1, H, W) * mv1 * dis)[..., None]
    wa, wb = 0.5 * r0, 0.5 * r1
    mix = (wa * a + wb * b + net2.MIX_EPS * 0.5 * (a + b)) / (wa + wb + net2.MIX_EPS)
    s = net2.static_weight(i0, i1)
    return s * (0.5 * (i0 + i1)) + (1 - s) * mix


def interpolate(i0, i1, bm, sigma=None, prior=None):
    """i0, i1 (B, H, W, 3) con H, W multiplos de 32; bm de model/bm2.py (sin corte de escena)."""
    return warp1(i0, i1, select(i0, i1, low_flow(bm), sigma, prior))
