// VCFG-NX: kernels de computo de NXFlow para la GPU de PS4 (GCN 1.1 "Liverpool").
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

CONV3X3(conv3x3_14_32, 14, 32)  // A.c0
CONV3X3(conv3x3_17_32, 17, 32)  // B.c0
CONV3X3(conv3x3_32_32, 32, 32)  // c1 y capas medias
CONV3X3(conv3x3_32_5, 32, 5)    // salida (flujo t->0, t->1, mascara)
CONV3X3(conv3x3_17_16, 17, 16)  // C.c0 (nivel a 1/4)
CONV3X3(conv3x3_16_16, 16, 16)  // C.c1 y capas medias
CONV3X3(conv3x3_16_5, 16, 5)    // C.out

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

// Arma la entrada de un nivel (concat de canales en un solo pase):
//   level 0 (A, 1/8): a0, a1, warp(a0, f0), warp(a1, f1), f1           -> 14 canales
//   level 1 (B 1/8, C 1/4): a0, a1, warp(a0, f0), warp(a1, f1), f0, f1, m -> 17 canales
// flow5 (f0x f0y f1x f1y m) se actualiza in-place con delta5 (NULL = sin delta)
// antes de usarlo; tras el nivel B, otro pase con level = -1 solo acumula.
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
    int C = level == 0 ? 14 : 17;
    __global float* o = out + (y * W + x) * C;
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
    if (level == 0) {
        o[12] = f[2]; o[13] = f[3];
    } else {
        for (int c = 0; c < 5; ++c) o[12 + c] = f[c];
    }
}

// vectores BM (dy, dx) en px completos -> flow5 = (-v/8, +v/8, mascara 0)
NX_KERNEL void bm_to_flow(__global const float* vec, int H, int W, __global float* flow5)
{
    int x = GID_X, y = GID_Y;
    if (x >= W || y >= H) return;
    float vy = vec[(y * W + x) * 2] * 0.125f, vx = vec[(y * W + x) * 2 + 1] * 0.125f;
    __global float* f = flow5 + (y * W + x) * 5;
    f[0] = -vx; f[1] = -vy; f[2] = vx; f[3] = vy; f[4] = 0.0f;
}

// ------------------------------------------------ block-matching simetrico
// Equivalente a model/baseline_bm.py: piramide de luma de 3 niveles, bloques
// de 8x8, busqueda +-3 alrededor del vector del nivel grueso, mediana 3x3.
// Vectores (dy, dx) en px del nivel; en el nivel 0 = px de resolucion completa.

NX_KERNEL void luma_full(tdesc_t src, int Hsrc, int Wsrc, __global float* out, int H, int W)
{
    int x = GID_X, y = GID_Y;
    if (x >= W || y >= H) return;
    float4 c = nx_load(src, nx_clampi(x, 0, Wsrc - 1), nx_clampi(y, 0, Hsrc - 1));
    out[y * W + x] = c.x * 0.299f + c.y * 0.587f + c.z * 0.114f;
}

NX_KERNEL void luma_down2(__global const float* in, int H, int W, __global float* out)
{
    int x = GID_X, y = GID_Y;
    int Ho = H / 2, Wo = W / 2;
    if (x >= Wo || y >= Ho) return;
    __global const float* p = in + (2 * y) * W + 2 * x;
    out[y * Wo + x] = 0.25f * (p[0] + p[1] + p[W] + p[W + 1]);
}

// coarse == NULL en el nivel mas grueso; si no, init = round(2 * coarse[by/2][bx/2])
NX_KERNEL void bm_search(__global const float* l0, __global const float* l1, int H, int W,
                         __global const float* coarse, int cBy, int cBx,
                         __global float* vec, int By, int Bx, int radius)
{
    int bx = GID_X, by = GID_Y;
    if (bx >= Bx || by >= By) return;
    int iy = 0, ix = 0;
    if (coarse) {
        int cy = by / 2 < cBy - 1 ? by / 2 : cBy - 1, cx = bx / 2 < cBx - 1 ? bx / 2 : cBx - 1;
        iy = (int)__builtin_rintf(2.0f * coarse[(cy * cBx + cx) * 2 + 0]);
        ix = (int)__builtin_rintf(2.0f * coarse[(cy * cBx + cx) * 2 + 1]);
    }
    float best = 3.0e38f;
    int bvy = iy, bvx = ix;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
            int vy = iy + dy, vx = ix + dx;
            float cost = 0.0f;
            for (int j = 0; j < 8; ++j) {
                int ya = nx_clampi(by * 8 - vy + j, 0, H - 1), yb = nx_clampi(by * 8 + vy + j, 0, H - 1);
                for (int i = 0; i < 8; ++i) {
                    int xa = nx_clampi(bx * 8 - vx + i, 0, W - 1), xb = nx_clampi(bx * 8 + vx + i, 0, W - 1);
                    cost += nx_absf(l0[ya * W + xa] - l1[yb * W + xb]);
                }
            }
            cost += 0.002f * 64.0f * (float)((vy < 0 ? -vy : vy) + (vx < 0 ? -vx : vx));
            if (cost < best) { best = cost; bvy = vy; bvx = vx; }
        }
    vec[(by * Bx + bx) * 2 + 0] = (float)bvy;
    vec[(by * Bx + bx) * 2 + 1] = (float)bvx;
}

// Igual que bm_search (mismas comparaciones, mismo orden de suma y de desempate),
// pero un grupo de 64 hilos por bloque: las dos zonas de luma que se comparan se
// cargan una vez en LDS ((8 + 2R)^2 cada una) y cada hilo evalua candidatos desde
// ahi. bm_search lee 2 x 64 valores de memoria por candidato y por bloque (~270
// millones por fotograma a 1080p con radio 3): aca son 2 x (8 + 2R)^2 por bloque.
// Despacho: Bx * 8 x By * 8 hilos (un grupo por bloque). Radio maximo 4.
#define BM_RMAX 4
#define BM_SIDE (8 + 2 * BM_RMAX)
NX_KERNEL void wg_bm_search(__global const float* l0, __global const float* l1, int H, int W,
                            __global const float* coarse, int cBy, int cBx,
                            __global float* vec, int By, int Bx, int radius)
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
        iy = (int)__builtin_rintf(2.0f * coarse[(cy * cBx + cx) * 2 + 0]);
        ix = (int)__builtin_rintf(2.0f * coarse[(cy * cBx + cx) * 2 + 1]);
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
        for (int j = 0; j < 8; ++j)
            for (int i = 0; i < 8; ++i)
                s += nx_absf(t0[(R - dy + j) * side + (R - dx + i)] - t1[(R + dy + j) * side + (R + dx + i)]);
        int vy = iy + dy, vx = ix + dx;
        s += 0.002f * 64.0f * (float)((vy < 0 ? -vy : vy) + (vx < 0 ? -vx : vx));
        cost[ci] = s;
    }
    WG_END
    WG_BARRIER
    WG_BEGIN
    if (LID == 0) {
        float best = 3.0e38f;
        int bvy = iy, bvx = ix;
        for (int ci = 0; ci < nc * nc; ++ci)   // mismo orden que bm_search: dy, despues dx
            if (cost[ci] < best) { best = cost[ci]; bvy = iy + ci / nc - R; bvx = ix + ci % nc - R; }
        vec[(by * Bx + bx) * 2 + 0] = (float)bvy;
        vec[(by * Bx + bx) * 2 + 1] = (float)bvx;
    }
    WG_END
}

NX_KERNEL void bm_median3(__global const float* in, __global float* out, int By, int Bx)
{
    int bx = GID_X, by = GID_Y;
    if (bx >= Bx || by >= By) return;
    for (int c = 0; c < 2; ++c) {
        float a[9];
        int n = 0;
        for (int j = -1; j <= 1; ++j)
            for (int i = -1; i <= 1; ++i)
                a[n++] = in[(nx_clampi(by + j, 0, By - 1) * Bx + nx_clampi(bx + i, 0, Bx - 1)) * 2 + c];
        // red de ordenamiento: mediana de 9 en 19 comparaciones (Paeth)
#define S(p, q) { float t_ = nx_minf(a[p], a[q]); a[q] = nx_maxf(a[p], a[q]); a[p] = t_; }
        S(1, 2) S(4, 5) S(7, 8) S(0, 1) S(3, 4) S(6, 7) S(1, 2) S(4, 5) S(7, 8)
        S(0, 3) S(5, 8) S(4, 7) S(3, 6) S(1, 4) S(2, 5) S(4, 7) S(4, 2) S(6, 4) S(4, 2)
#undef S
        out[(by * Bx + bx) * 2 + c] = a[4];
    }
}

// --------------------------------------------- pase final a resolucion completa
// Lee flujo+mascara a 1/4 (5 canales: f0x f0y f1x f1y m), los sube con bilineal
// half-pixel (x4 en valor para el flujo), hace backward warp de los dos
// fotogramas del juego con el sampler bilineal del hardware y mezcla.
// El HUD estatico no necesita mascara aparte: la red aprende flujo ~0 donde no hay cambio.
NX_KERNEL void warp_blend(tdesc_t frame0, tdesc_t frame1, tdesc_t lin_sampler,
                          __global const float* lowres5, int h4, int w4,
                          tdesc_t dst, int H, int W)
{
    int x = GID_X, y = GID_Y;
    if (x >= W || y >= H) return;
    // relacion exacta 1/4 con la malla (alto relleno a multiplo de 32)
    float sy = (y + 0.5f) * 0.25f - 0.5f;
    float sx = (x + 0.5f) * 0.25f - 0.5f;
    float v[5];
    for (int c = 0; c < 5; ++c) v[c] = bilerp_c(lowres5, h4, w4, 5, sy, sx, c);
    float m = nx_sigmoid(v[4]);
    // el sampler usa coordenadas de pixel no normalizadas; +0.5 = centro de texel
    float x0 = nx_clampf(x + 4.0f * v[0], 0.0f, W - 1.0f) + 0.5f;
    float y0 = nx_clampf(y + 4.0f * v[1], 0.0f, H - 1.0f) + 0.5f;
    float x1 = nx_clampf(x + 4.0f * v[2], 0.0f, W - 1.0f) + 0.5f;
    float y1 = nx_clampf(y + 4.0f * v[3], 0.0f, H - 1.0f) + 0.5f;
    float4 c0 = nx_sample(frame0, lin_sampler, x0, y0);
    float4 c1 = nx_sample(frame1, lin_sampler, x1, y1);
    nx_store(dst, x, y, m * c0 + (1.0f - m) * c1);
}
