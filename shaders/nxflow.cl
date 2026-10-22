// VCFG-NX: kernels de computo de NXFG-2 (generacion de fotogramas) para la GPU de PS4 (GCN 1.1 "Liverpool").
//
// Compilacion (ver shaders/build_gcn.sh): OpenCL C -> LLVM IR -> enlace con
// gcn_image.ll -> code object gfx701. tools/gcn_pack.py lo convierte en
// plugin/include/nx_shaders.h (codigo + COMPUTE_PGM_RSRC1/2 + layout de SGPRs).
//
// Convenciones (identicas a model/reference_runtime.py y a JAX):
//   * tensores intermedios NHWC float en buffers lineales;
//   * fotogramas del juego como imagenes (T#), leidas con su tiling nativo;
//   * conv 3x3 "SAME": stride 1 -> pad 1/1; stride 2 con entrada par -> pad 0/1;
//   * resize bilineal con centros de pixel (half-pixel), sin antialias;
//   * el fotograma se procesa con alto/ancho rellenos a multiplo de 32
//     (1080 -> 1088): las mallas 1/4 y 1/8 cubren exactamente Wpad x Hpad.
#include "nx_common.h"

#define LRELU(x) ((x) > 0.0f ? (x) : 0.1f * (x))

// ---------------------------------------------------------------- conv 3x3
// Un work-item por pixel de salida. CIN/COUT son constantes de compilacion
// (una variante por forma de capa) para que el acumulador viva en VGPRs.
#define CONV3X3(NAME, CIN, COUT)                                                     \
NX_KERNEL void NAME(__global const float* in, int H, int W,                         \
                    __global float* out, int Ho, int Wo,                            \
                    __constant const float* w, /* [3][3][CIN][COUT] (HWIO) */       \
                    __constant const float* b, /* [COUT] */                         \
                    int stride, int pad_top, int pad_left, int act,                 \
                    __global const float* residual) /* NULL = sin suma */           \
{                                                                                   \
    int x = GID_X, y = GID_Y;                                                       \
    if (x >= Wo || y >= Ho) return;                                                 \
    float acc[COUT];                                                                \
    _Pragma("unroll") for (int o = 0; o < COUT; ++o) acc[o] = b[o];                 \
    for (int ky = 0; ky < 3; ++ky) {                                                \
        int iy = y * stride + ky - pad_top;                                         \
        if (iy < 0 || iy >= H) continue;                                            \
        for (int kx = 0; kx < 3; ++kx) {                                            \
            int ix = x * stride + kx - pad_left;                                    \
            if (ix < 0 || ix >= W) continue;                                        \
            __global const float* px = in + (iy * W + ix) * CIN;                    \
            __constant const float* wk = w + (ky * 3 + kx) * CIN * COUT;            \
            for (int i = 0; i < CIN; ++i) {                                         \
                float v = px[i];                                                    \
                _Pragma("unroll") for (int o = 0; o < COUT; ++o)                    \
                    acc[o] = __builtin_fmaf(v, wk[i * COUT + o], acc[o]);           \
            }                                                                       \
        }                                                                           \
    }                                                                               \
    __global float* dst = out + (y * Wo + x) * COUT;                                \
    _Pragma("unroll") for (int o = 0; o < COUT; ++o) {                              \
        float v = act ? LRELU(acc[o]) : acc[o];                                     \
        if (residual) v += residual[(y * Wo + x) * COUT + o];                       \
        dst[o] = v;                                                                 \
    }                                                                               \
}

// NXFG-2 (model/net2.py)
CONV3X3(conv3x3_15_32, 15, 32)  // A.c0 (entrada 1/16 -> interno 1/32)
CONV3X3(conv3x3_32_32, 32, 32)  // A.c1 y capas medias
CONV3X3(conv3x3_32_5, 32, 5)    // A.out (flujo t->0, t->1, mascara a 1/16)
CONV3X3(conv3x3_17_24, 17, 24)  // B.c0 (entrada 1/8 -> interno 1/16)
CONV3X3(conv3x3_24_24, 24, 24)  // B.c1 y capa media
CONV3X3(conv3x3_24_5, 24, 5)    // B.out (a 1/8)
// guia de NXSR-L (sr/net_srl.py, malla 1/8)
CONV3X3(conv3x3_5_12, 5, 12)
CONV3X3(conv3x3_12_12, 12, 12)
CONV3X3(conv3x3_12_3, 12, 3)

// ------------------------------------------------- utilidades de tensores
// Promedio fxf (f = 4 u 8) del fotograma del juego -> RGB a 1/f. Lee fuera de
// rango con clamp (filas 1080..1087 repiten la ultima), igual que el relleno "edge".
NX_KERNEL void avgpool_rgb(tdesc_t src, int H, int W, __global float* out, int Ho, int Wo, int f)
{
    int x = GID_X, y = GID_Y;
    if (x >= Wo || y >= Ho) return;
    float4 acc = (float4)(0.0f);
    for (int j = 0; j < f; ++j)
        for (int i = 0; i < f; ++i)
            acc += nx_load(src, nx_clampi(x * f + i, 0, W - 1), nx_clampi(y * f + j, 0, H - 1));
    acc *= 1.0f / (float)(f * f);
    __global float* d = out + (y * Wo + x) * 3;
    d[0] = acc.x; d[1] = acc.y; d[2] = acc.z;
}

static float fetch_c(__global const float* t, int H, int W, int C, int y, int x, int c)
{
    return t[(nx_clampi(y, 0, H - 1) * W + nx_clampi(x, 0, W - 1)) * C + c];
}

static float bilerp_c(__global const float* t, int H, int W, int C, float sy, float sx, int c)
{
    int y0 = (int)nx_floor(sy), x0 = (int)nx_floor(sx);
    float fy = sy - y0, fx = sx - x0;
    float a = fetch_c(t, H, W, C, y0, x0, c), b = fetch_c(t, H, W, C, y0, x0 + 1, c);
    float d = fetch_c(t, H, W, C, y0 + 1, x0, c), e = fetch_c(t, H, W, C, y0 + 1, x0 + 1, c);
    return (a * (1 - fx) + b * fx) * (1 - fy) + (d * (1 - fx) + e * fx) * fy;
}

// bilineal half-pixel: src(y) = (y + 0.5) * H / Ho - 0.5
NX_KERNEL void resize_bilinear(__global const float* in, int H, int W, int C,
                               __global float* out, int Ho, int Wo)
{
    int x = GID_X, y = GID_Y;
    if (x >= Wo || y >= Ho) return;
    float sy = (y + 0.5f) * ((float)H / Ho) - 0.5f;
    float sx = (x + 0.5f) * ((float)W / Wo) - 0.5f;
    for (int c = 0; c < C; ++c)
        out[(y * Wo + x) * C + c] = bilerp_c(in, H, W, C, sy, sx, c);
}

// flow5 (f0, f1, mascara) a una malla 2x mas fina: el flujo se escala x2, la mascara no
NX_KERNEL void upsample5(__global const float* in, int H, int W, __global float* out)
{
    int x = GID_X, y = GID_Y;
    if (x >= 2 * W || y >= 2 * H) return;
    float sy = (y + 0.5f) * 0.5f - 0.5f, sx = (x + 0.5f) * 0.5f - 0.5f;
    for (int c = 0; c < 5; ++c)
        out[(y * 2 * W + x) * 5 + c] = bilerp_c(in, H, W, 5, sy, sx, c) * (c < 4 ? 2.0f : 1.0f);
}

// promedio 2x2 de un tensor RGB (3 canales): 1/8 -> 1/16 (entrada del nivel A)
NX_KERNEL void pool2_rgb(__global const float* in, int H, int W, __global float* out)
{
    int x = GID_X, y = GID_Y;
    int Ho = H / 2, Wo = W / 2;
    if (x >= Wo || y >= Ho) return;
    for (int c = 0; c < 3; ++c) {
        __global const float* p = in + ((2 * y) * W + 2 * x) * 3 + c;
        out[(y * Wo + x) * 3 + c] = 0.25f * ((p[0] + p[3]) + (p[W * 3] + p[W * 3 + 3]));
    }
}

// Entrada del nivel A de NXFG-2 (malla 1/16) desde los vectores del block-matching
// (dy, dx en px de media resolucion, confianza): arma flow5 = (-v/8, +v/8, 0) y
// los 15 canales a0, a1, warp(a0, f0), warp(a1, f1), f1, 10 * confianza.
NX_KERNEL void nxfg_input_a(__global const float* a0, __global const float* a1, __global const float* vec,
                            int H, int W, __global float* flow5, __global float* out)
{
    int x = GID_X, y = GID_Y;
    if (x >= W || y >= H) return;
    __global const float* v = vec + (y * W + x) * 3;
    float vx = v[1] * 0.125f, vy = v[0] * 0.125f;
    __global float* f = flow5 + (y * W + x) * 5;
    f[0] = -vx; f[1] = -vy; f[2] = vx; f[3] = vy; f[4] = 0.0f;
    __global float* o = out + (y * W + x) * 15;
    for (int c = 0; c < 3; ++c) {
        o[c] = a0[(y * W + x) * 3 + c];
        o[3 + c] = a1[(y * W + x) * 3 + c];
    }
    float sx0 = nx_clampf(x - vx, 0.0f, W - 1.0f), sy0 = nx_clampf(y - vy, 0.0f, H - 1.0f);
    float sx1 = nx_clampf(x + vx, 0.0f, W - 1.0f), sy1 = nx_clampf(y + vy, 0.0f, H - 1.0f);
    for (int c = 0; c < 3; ++c) {
        o[6 + c] = bilerp_c(a0, H, W, 3, sy0, sx0, c);
        o[9 + c] = bilerp_c(a1, H, W, 3, sy1, sx1, c);
    }
    o[12] = vx; o[13] = vy;
    o[14] = v[2] * 10.0f;
}

// Entrada del nivel B (1/8): a0, a1, warp(a0, f0), warp(a1, f1), f0, f1, m -> 17 canales.
// flow5 (f0x f0y f1x f1y m) se actualiza in-place con delta5 (NULL = sin delta)
// antes de usarlo; con level = -1 solo acumula.
NX_KERNEL void build_level_input(__global const float* a0, __global const float* a1,
                                 __global float* flow5, __global const float* delta5,
                                 int H, int W, int level, __global float* out)
{
    int x = GID_X, y = GID_Y;
    if (x >= W || y >= H) return;
    __global float* f = flow5 + (y * W + x) * 5;
    if (delta5) {
        __global const float* d = delta5 + (y * W + x) * 5;
        for (int c = 0; c < 5; ++c) f[c] += d[c];
    }
    if (level < 0) return;
    __global float* o = out + (y * W + x) * 17;
    for (int c = 0; c < 3; ++c) {
        o[c] = a0[(y * W + x) * 3 + c];
        o[3 + c] = a1[(y * W + x) * 3 + c];
    }
    float sx0 = nx_clampf(x + f[0], 0.0f, W - 1.0f), sy0 = nx_clampf(y + f[1], 0.0f, H - 1.0f);
    float sx1 = nx_clampf(x + f[2], 0.0f, W - 1.0f), sy1 = nx_clampf(y + f[3], 0.0f, H - 1.0f);
    for (int c = 0; c < 3; ++c) {
        o[6 + c] = bilerp_c(a0, H, W, 3, sy0, sx0, c);
        o[9 + c] = bilerp_c(a1, H, W, 3, sy1, sx1, c);
    }
    for (int c = 0; c < 5; ++c) o[12 + c] = f[c];
}

// generacion rapida (sin la red): vectores (dy, dx, conf) -> flow5 = (-v/8, +v/8, 0) en la malla 1/16
NX_KERNEL void bm_to_flow(__global const float* vec, int H, int W, __global float* flow5)
{
    int x = GID_X, y = GID_Y;
    if (x >= W || y >= H) return;
    float vy = vec[(y * W + x) * 3] * 0.125f, vx = vec[(y * W + x) * 3 + 1] * 0.125f;
    __global float* f = flow5 + (y * W + x) * 5;
    f[0] = -vx; f[1] = -vy; f[2] = vx; f[3] = vy; f[4] = 0.0f;
}

// ------------------------------------------------ block-matching simetrico
// Equivalente exacto de model/bm2.py: piramide de luma a media resolucion (4
// niveles), bloques de 8x8, busqueda alrededor del vector del nivel grueso,
// mediana 3x3; en el nivel fino, ajuste sub-pixel y confianza.
// Vectores (dy, dx, conf) en px del nivel; en el nivel fino = px de media resolucion.

// luma de media resolucion: promedio 2x2 de la imagen (bordes repetidos) y luma del promedio
NX_KERNEL void luma_half(tdesc_t src, int Hsrc, int Wsrc, __global float* out, int H, int W)
{
    int x = GID_X, y = GID_Y;
    if (x >= W || y >= H) return;
    int x0 = nx_clampi(2 * x, 0, Wsrc - 1), x1 = nx_clampi(2 * x + 1, 0, Wsrc - 1);
    int y0 = nx_clampi(2 * y, 0, Hsrc - 1), y1 = nx_clampi(2 * y + 1, 0, Hsrc - 1);
    float4 a = ((nx_load(src, x0, y0) + nx_load(src, x1, y0)) + (nx_load(src, x0, y1) + nx_load(src, x1, y1))) * 0.25f;
    out[y * W + x] = (a.x * 0.299f + a.y * 0.587f) + a.z * 0.114f;
}

NX_KERNEL void luma_down2(__global const float* in, int H, int W, __global float* out)
{
    int x = GID_X, y = GID_Y;
    int Ho = H / 2, Wo = W / 2;
    if (x >= Wo || y >= Ho) return;
    __global const float* p = in + (2 * y) * W + 2 * x;
    out[y * Wo + x] = ((p[0] + p[1]) + (p[W] + p[W + 1])) * 0.25f;
}

// Un grupo de 64 hilos por bloque: las dos zonas de luma que se comparan se
// cargan una vez en LDS ((8 + 2R)^2 cada una) y cada hilo evalua candidatos desde
// ahi. Despacho: Bx * 8 x By * 8 hilos (un grupo por bloque). Radio maximo 4.
#define BM_RMAX 4
#define BM_SIDE (8 + 2 * BM_RMAX)
#define BM_LAM_D 0.08f   // por alejarse del vector del nivel grueso
#define BM_LAM_V 0.02f   // por vector largo
NX_KERNEL void wg_bm_search(__global const float* l0, __global const float* l1, int H, int W,
                            __global const float* coarse, int cBy, int cBx,
                            __global float* vec, int By, int Bx, int radius, int finest)
{
    __local float t0[BM_SIDE * BM_SIDE];
    __local float t1[BM_SIDE * BM_SIDE];
    __local float cost[(2 * BM_RMAX + 1) * (2 * BM_RMAX + 1)];
    const int bx = GROUP_X, by = GROUP_Y;
    if (bx >= Bx || by >= By) return;
    const int R = radius < BM_RMAX ? radius : BM_RMAX;
    const int side = 8 + 2 * R, nc = 2 * R + 1;
    int iy = 0, ix = 0;
    if (coarse) {
        int cy = by / 2 < cBy - 1 ? by / 2 : cBy - 1, cx = bx / 2 < cBx - 1 ? bx / 2 : cBx - 1;
        iy = (int)__builtin_rintf(2.0f * coarse[(cy * cBx + cx) * 3 + 0]);
        ix = (int)__builtin_rintf(2.0f * coarse[(cy * cBx + cx) * 3 + 1]);
    }
    // t0[r][c] = l0 en (by*8 - iy - R + r, bx*8 - ix - R + c); t1 con +iy, +ix
    WG_BEGIN
    for (int t = LID; t < side * side; t += NX_WG * NX_WG) {
        int r = t / side, c = t % side;
        int ya = nx_clampi(by * 8 - iy - R + r, 0, H - 1), xa = nx_clampi(bx * 8 - ix - R + c, 0, W - 1);
        int yb = nx_clampi(by * 8 + iy - R + r, 0, H - 1), xb = nx_clampi(bx * 8 + ix - R + c, 0, W - 1);
        t0[t] = l0[ya * W + xa];
        t1[t] = l1[yb * W + xb];
    }
    WG_END
    WG_BARRIER
    WG_BEGIN
    for (int ci = LID; ci < nc * nc; ci += NX_WG * NX_WG) {
        int dy = ci / nc - R, dx = ci % nc - R;
        float s = 0.0f;
        for (int j = 0; j < 8; ++j) {
            float r = 0.0f;
            for (int i = 0; i < 8; ++i)
                r += nx_absf(t0[(R - dy + j) * side + (R - dx + i)] - t1[(R + dy + j) * side + (R + dx + i)]);
            s += r;
        }
        int vy = iy + dy, vx = ix + dx;
        float pen = BM_LAM_D * (float)((dy < 0 ? -dy : dy) + (dx < 0 ? -dx : dx))
                  + BM_LAM_V * (float)((vy < 0 ? -vy : vy) + (vx < 0 ? -vx : vx));
        cost[ci] = s + pen;
    }
    WG_END
    WG_BARRIER
    WG_BEGIN
    if (LID == 0) {
        float best = 3.0e38f;
        int k = 0;
        for (int ci = 0; ci < nc * nc; ++ci)   // primer minimo en orden dy, despues dx
            if (cost[ci] < best) { best = cost[ci]; k = ci; }
        int ky = k / nc, kx = k % nc;
        int bvy = iy + ky - R, bvx = ix + kx - R;
        float oy = 0.0f, ox = 0.0f, conf = 0.0f;
        if (finest) {
            float c0 = cost[k];
            if (ky > 0 && ky < nc - 1) {
                float cm = cost[k - nc], cp = cost[k + nc], den = (cm - 2.0f * c0) + cp;
                if (den > 1e-6f) oy = nx_clampf(0.5f * (cm - cp) / den, -0.5f, 0.5f);
            }
            if (kx > 0 && kx < nc - 1) {
                float cm = cost[k - 1], cp = cost[k + 1], den = (cm - 2.0f * c0) + cp;
                if (den > 1e-6f) ox = nx_clampf(0.5f * (cm - cp) / den, -0.5f, 0.5f);
            }
            int ady = ky - R, adx = kx - R;
            float pen = BM_LAM_D * (float)((ady < 0 ? -ady : ady) + (adx < 0 ? -adx : adx))
                      + BM_LAM_V * (float)((bvy < 0 ? -bvy : bvy) + (bvx < 0 ? -bvx : bvx));
            conf = (c0 - pen) * (1.0f / 64.0f);
        }
        vec[(by * Bx + bx) * 3 + 0] = (float)bvy + oy;
        vec[(by * Bx + bx) * 3 + 1] = (float)bvx + ox;
        vec[(by * Bx + bx) * 3 + 2] = conf;
    }
    WG_END
}

// mediana 3x3 (por componente) de los vectores; la confianza pasa sin tocar
NX_KERNEL void bm_median3(__global const float* in, __global float* out, int By, int Bx)
{
    int bx = GID_X, by = GID_Y;
    if (bx >= Bx || by >= By) return;
    for (int c = 0; c < 2; ++c) {
        float a[9];
        int n = 0;
        for (int j = -1; j <= 1; ++j)
            for (int i = -1; i <= 1; ++i)
                a[n++] = in[(nx_clampi(by + j, 0, By - 1) * Bx + nx_clampi(bx + i, 0, Bx - 1)) * 3 + c];
        // red de ordenamiento: mediana de 9 en 19 comparaciones (Paeth)
#define S(p, q) { float t_ = nx_minf(a[p], a[q]); a[q] = nx_maxf(a[p], a[q]); a[p] = t_; }
        S(1, 2) S(4, 5) S(7, 8) S(0, 1) S(3, 4) S(6, 7) S(1, 2) S(4, 5) S(7, 8)
        S(0, 3) S(5, 8) S(4, 7) S(3, 6) S(1, 4) S(2, 5) S(4, 7) S(4, 2) S(6, 4) S(4, 2)
#undef S
        out[(by * Bx + bx) * 3 + c] = a[4];
    }
    out[(by * Bx + bx) * 3 + 2] = in[(by * Bx + bx) * 3 + 2];
}

// Corte de escena: confianza media de la malla fina (diferencia media de luma de
// los bloques ya alineados). Si es mala, ningun movimiento explica el cambio:
// el warp repite el fotograma real en vez de inventar uno (sin "fundido" raro).
// Un solo grupo de 64 hilos.
#define NXFG_CUT 0.12f
NX_KERNEL void wg_nxfg_scene(__global const float* vec, int n, __global float* flag)
{
    __local float part[NX_WG * NX_WG];
    WG_BEGIN
    float s = 0.0f;
    for (int i = LID; i < n; i += NX_WG * NX_WG) s += vec[i * 3 + 2];
    part[LID] = s;
    WG_END
    WG_BARRIER
    WG_BEGIN
    if (LID == 0) {
        float t = 0.0f;
        for (int i = 0; i < NX_WG * NX_WG; ++i) t += part[i];
        flag[0] = t > NXFG_CUT * (float)n ? 1.0f : 0.0f;
    }
    WG_END
}

// ------------------------------------- mascara de pixeles quietos (anti-fantasmas)
// 1 donde la luma de media resolucion no cambia entre los dos reales (HUD, textos,
// fondo fijo). Apertura 3x3 en dos pases (como model/net2.py): static_mask erosiona
// (las coincidencias sueltas de una textura que se mueve no cuentan) y mask_dilate
// dilata (el HUD vuelve a quedar entero, bordes incluidos).
#define NXFG_SRC_TAU 0.015f
NX_KERNEL void static_mask(__global const float* l0, __global const float* l1, int H, int W, __global float* out)
{
    int x = GID_X, y = GID_Y;
    if (x >= W || y >= H) return;
    float s = 1.0f;
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i) {
            int k = nx_clampi(y + j, 0, H - 1) * W + nx_clampi(x + i, 0, W - 1);
            if (!(nx_absf(l0[k] - l1[k]) < NXFG_SRC_TAU)) s = 0.0f;
        }
    out[y * W + x] = s;
}

NX_KERNEL void mask_dilate(__global const float* in, int H, int W, __global float* out)
{
    int x = GID_X, y = GID_Y;
    if (x >= W || y >= H) return;
    float s = 0.0f;
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i)
            s = nx_maxf(s, in[nx_clampi(y + j, 0, H - 1) * W + nx_clampi(x + i, 0, W - 1)]);
    out[y * W + x] = s;
}

// --------------------------------------------- pase final a resolucion completa
// Warp por bloques superpuestos con eleccion por pixel. Lee flujo+mascara de la
// malla 1/scale (5 canales: f0x f0y f1x f1y m); cada una de las 4 celdas vecinas
// del pixel hace backward warp de los dos fotogramas del juego (sampler bilineal
// del hardware) y aporta con su peso bilineal (OBMC: sin bordes de bloque)
// multiplicado por lo bien que coinciden sus dos muestras (softmin de |w0 - w1|:
// en los bordes de objetos gana el vector correcto, sin halos). Identico a
// model/net2.py. Protecciones fijas (no dependen de la red):
//   * fantasmas de HUD: una muestra que cae en un pixel quieto (smask) con un
//     vector que se mueve, y que no coincide con la otra, no se usa: se toma la
//     del otro fotograma;
//   * pixel quieto: si los dos reales casi coinciden en ese pixel (HUD, textos,
//     cielo, zonas sin cambio) se usa su promedio, sin warp: nada tiembla;
//   * corte de escena (flag != 0): se repite el fotograma actual.
#define NXFG_STATIC_TAU 0.03f
#define NXFG_CAND_SIGMA 0.1f
#define NXFG_MIX_EPS 1e-3f
NX_KERNEL void warp_blend(tdesc_t frame0, tdesc_t frame1, tdesc_t lin_sampler,
                          __global const float* low5, int hl, int wl, float scale,
                          __global const float* flag, __global const float* smask,
                          tdesc_t dst, int H, int W)
{
    int x = GID_X, y = GID_Y;
    if (x >= W || y >= H) return;
    float4 p1 = nx_load(frame1, x, y);
    if (flag && flag[0] > 0.5f) { nx_store(dst, x, y, p1); return; }
    float inv = 1.0f / scale;
    float sy = (y + 0.5f) * inv - 0.5f;
    float sx = (x + 0.5f) * inv - 0.5f;
    float fly = nx_floor(sy), flx = nx_floor(sx);
    float fy = sy - fly, fx = sx - flx;
    int y0 = (int)fly, x0 = (int)flx;
    int ya = nx_clampi(y0, 0, hl - 1), yb = nx_clampi(y0 + 1, 0, hl - 1);
    int xa = nx_clampi(x0, 0, wl - 1), xb = nx_clampi(x0 + 1, 0, wl - 1);
    int cell[4] = { ya * wl + xa, ya * wl + xb, yb * wl + xa, yb * wl + xb };
    float bw[4] = { (1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx };
    int W2 = ((W + 31) & ~31) / 2, H2 = ((H + 31) & ~31) / 2;   // la luma de media resolucion (rellena a 32)
    float4 o[4];
    float er[4], emin = 3.0e38f;
    for (int k = 0; k < 4; ++k) {
        __global const float* q = low5 + cell[k] * 5;
        float f0x = scale * q[0], f0y = scale * q[1], f1x = scale * q[2], f1y = scale * q[3];
        // el sampler usa coordenadas de pixel no normalizadas; +0.5 = centro de texel
        float u0 = nx_clampf(x + f0x, 0.0f, W - 1.0f) + 0.5f, v0 = nx_clampf(y + f0y, 0.0f, H - 1.0f) + 0.5f;
        float u1 = nx_clampf(x + f1x, 0.0f, W - 1.0f) + 0.5f, v1 = nx_clampf(y + f1y, 0.0f, H - 1.0f) + 0.5f;
        float4 a = nx_sample(frame0, lin_sampler, u0, v0);
        float4 b = nx_sample(frame1, lin_sampler, u1, v1);
        float m = nx_sigmoid(q[4]);
        float r0 = 1.0f, r1 = 1.0f;
        if (smask) {
            if (nx_absf(f0x) + nx_absf(f0y) > 1.0f) {
                int hx = nx_clampi((int)nx_floor((x + f0x) * 0.5f), 0, W2 - 1);
                int hy = nx_clampi((int)nx_floor((y + f0y) * 0.5f), 0, H2 - 1);
                r0 = 1.0f - smask[hy * W2 + hx];
            }
            if (nx_absf(f1x) + nx_absf(f1y) > 1.0f) {
                int hx = nx_clampi((int)nx_floor((x + f1x) * 0.5f), 0, W2 - 1);
                int hy = nx_clampi((int)nx_floor((y + f1y) * 0.5f), 0, H2 - 1);
                r1 = 1.0f - smask[hy * W2 + hx];
            }
        }
        er[k] = (nx_absf(a.x - b.x) + nx_absf(a.y - b.y)) + nx_absf(a.z - b.z);
        // solo cuenta si ademas las dos muestras no coinciden (un fantasma de HUD)
        float dis = nx_clampf((er[k] - 0.06f) * (1.0f / 0.12f), 0.0f, 1.0f);
        r0 = 1.0f - (1.0f - r0) * dis;
        r1 = 1.0f - (1.0f - r1) * dis;
        float wa = m * r0, wb = (1.0f - m) * r1;
        o[k] = (wa * a + wb * b + NXFG_MIX_EPS * 0.5f * (a + b)) * (1.0f / (wa + wb + NXFG_MIX_EPS));
        emin = nx_minf(emin, er[k]);
    }
    float4 mix = (float4)(0.0f);
    float ws = 0.0f;
    for (int k = 0; k < 4; ++k) {
        float wk = bw[k] * __builtin_exp2f(-(er[k] - emin) * (1.44269504f / NXFG_CAND_SIGMA));
        mix += wk * o[k];
        ws += wk;
    }
    mix *= 1.0f / ws;
    float4 p0 = nx_load(frame0, x, y);
    float d = (nx_absf(p0.x - p1.x) + nx_absf(p0.y - p1.y) + nx_absf(p0.z - p1.z)) * (1.0f / 3.0f);
    float st = nx_clampf(1.0f - d * (1.0f / NXFG_STATIC_TAU), 0.0f, 1.0f);
    nx_store(dst, x, y, st * (0.5f * (p0 + p1)) + (1.0f - st) * mix);
}
