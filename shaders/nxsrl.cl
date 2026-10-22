// NXSR-L ("ligero"): realce de detalle guiado por una red chica (sr/net_srl.py).
//
//   wg_srl_feat   un grupo de 64 hilos por celda de 8x8: luma con borde de 2 en LDS,
//                 bandas b1 = Y - G3(Y), b2 = G3(Y) - G5(Y); por celda: RGB medio,
//                 8 * media |b1|, 8 * media |b2|  -> 5 canales a 1/8
//   (red)         conv3x3 5->12, 12->12, 12->12 (+ residual), 12->3 con los kernels
//                 genericos de nxflow.cl
//   srl_act       softplus de las 3 salidas (ganancias g1, g2 y umbral t * 0.02)
//   wg_srl_apply  un grupo por 8x8 pixeles: bandas desde LDS, ganancias bilineales
//                 de la malla 1/8, nucleo (sin la parte |b| < t), limite al rango
//                 3x3 de Y (+ margen) y Y' - Y sumado a RGB.
// Bordes repetidos (como el padding "edge" de JAX). Alto/ancho no multiplos de 8:
// la ultima celda lee con bordes repetidos (igual a rellenar la imagen a multiplo de 8).
#include "nx_common.h"

#define SRL_T 12   // 8 + 2 * 2
#define SRL_MARGIN 0.25f

static float srl_y(float4 c) { return (c.x * 0.299f + c.y * 0.587f) + c.z * 0.114f; }

// Filtros separables en LDS (como sr/net_srl.py: primero horizontal, despues
// vertical): el tile de luma tiene 12x12 (borde de 2); la pasada horizontal deja,
// para las 8 columnas del centro y las 12 filas, G3 y G5 horizontales y el min/max
// de 3; la vertical, por pixel, termina G3, G5 y el rango 3x3. 14 lecturas de LDS
// por pixel en vez de 43.
typedef struct { __local float* t; __local float* h3; __local float* h5; __local float* hmn; __local float* hmx; } srl_lds_t;

static void srl_hpass(srl_lds_t L)
{
    WG_BEGIN
    for (int k = LID; k < SRL_T * 8; k += NX_WG * NX_WG) {
        int r = k / 8, cc = k % 8 + 2;
        __local const float* row = L.t + r * SRL_T;
        float a = row[cc - 2], b = row[cc - 1], c = row[cc], d = row[cc + 1], e = row[cc + 2];
        L.h3[k] = (0.25f * b + 0.5f * c) + 0.25f * d;
        L.h5[k] = (((1.0f / 16) * a + (4.0f / 16) * b) + (6.0f / 16) * c + (4.0f / 16) * d) + (1.0f / 16) * e;
        L.hmn[k] = nx_minf(nx_minf(b, c), d);
        L.hmx[k] = nx_maxf(nx_maxf(b, c), d);
    }
    WG_END
}

// pixel (lx, ly) del bloque de 8x8 (fila ly + 2 del tile)
static void srl_vpass(srl_lds_t L, int lx, int ly, float* b1, float* b2, float* yc, float* lo, float* hi)
{
    __local const float* h3 = L.h3 + lx;
    __local const float* h5 = L.h5 + lx;
    int r = ly + 2;
    float g3 = (0.25f * h3[(r - 1) * 8] + 0.5f * h3[r * 8]) + 0.25f * h3[(r + 1) * 8];
    float g5 = (((1.0f / 16) * h5[(r - 2) * 8] + (4.0f / 16) * h5[(r - 1) * 8]) + (6.0f / 16) * h5[r * 8]
                + (4.0f / 16) * h5[(r + 1) * 8]) + (1.0f / 16) * h5[(r + 2) * 8];
    float y = L.t[r * SRL_T + lx + 2];
    *b1 = y - g3;
    *b2 = g3 - g5;
    *yc = y;
    *lo = nx_minf(nx_minf(L.hmn[(r - 1) * 8 + lx], L.hmn[r * 8 + lx]), L.hmn[(r + 1) * 8 + lx]);
    *hi = nx_maxf(nx_maxf(L.hmx[(r - 1) * 8 + lx], L.hmx[r * 8 + lx]), L.hmx[(r + 1) * 8 + lx]);
}

static void srl_tile(__local float* t, tdesc_t src, int W, int H, int x0, int y0)
{
    WG_BEGIN
    for (int k = LID; k < SRL_T * SRL_T; k += NX_WG * NX_WG) {
        int r = k / SRL_T, c = k % SRL_T;
        t[k] = srl_y(nx_load(src, nx_clampi(x0 - 2 + c, 0, W - 1), nx_clampi(y0 - 2 + r, 0, H - 1)));
    }
    WG_END
}

#define SRL_LDS_DECL                                                                  \
    __local float t_[SRL_T * SRL_T], h3_[SRL_T * 8], h5_[SRL_T * 8], mn_[SRL_T * 8], mx_[SRL_T * 8]; \
    srl_lds_t L = { t_, h3_, h5_, mn_, mx_ };

// Despacho: wc * 8 x hc * 8 hilos (un grupo por celda). out: hc x wc x 5
NX_KERNEL void wg_srl_feat(tdesc_t src, int W, int H, __global float* out, int wc, int hc)
{
    SRL_LDS_DECL
    __local float acc[5 * NX_WG * NX_WG];
    const int cx = GROUP_X, cy = GROUP_Y;
    if (cx >= wc || cy >= hc) return;
    srl_tile(L.t, src, W, H, cx * 8, cy * 8);
    WG_BARRIER
    srl_hpass(L);
    WG_BARRIER
    WG_BEGIN
    int lx = LID % 8, ly = LID / 8;
    float b1, b2, yc, lo, hi;
    srl_vpass(L, lx, ly, &b1, &b2, &yc, &lo, &hi);
    float4 c = nx_load(src, nx_clampi(cx * 8 + lx, 0, W - 1), nx_clampi(cy * 8 + ly, 0, H - 1));
    acc[0 * 64 + LID] = c.x;
    acc[1 * 64 + LID] = c.y;
    acc[2 * 64 + LID] = c.z;
    acc[3 * 64 + LID] = nx_absf(b1) * 8.0f;
    acc[4 * 64 + LID] = nx_absf(b2) * 8.0f;
    WG_END
    // suma en arbol (6 pasos) de los 5 canales
    for (int s = 32; s >= 1; s >>= 1) {
        WG_BARRIER
        WG_BEGIN
        if (LID < s)
            for (int k = 0; k < 5; ++k) acc[k * 64 + LID] += acc[k * 64 + LID + s];
        WG_END
    }
    WG_BARRIER
    WG_BEGIN
    if (LID < 5) out[(cy * wc + cx) * 5 + LID] = acc[LID * 64] * (1.0f / 64.0f);
    WG_END
}

static float srl_softplus(float x)
{
    // log(1 + e^x) estable: max(x, 0) + log(1 + e^-|x|)
    float a = nx_absf(x);
    return nx_maxf(x, 0.0f) + 0.69314718f * __builtin_log2f(1.0f + __builtin_exp2f(-1.44269504f * a));
}

NX_KERNEL void srl_act(__global float* g, int n)
{
    int i = GID_Y * 1024 + GID_X;   // despacho 1024 x ceil(n / 1024)
    if (GID_X >= 1024 || i >= n) return;
    __global float* p = g + i * 3;
    p[0] = srl_softplus(p[0]);
    p[1] = srl_softplus(p[1]);
    p[2] = srl_softplus(p[2]) * 0.02f;
}

// Despacho: W x H hilos (grupos de 8x8). g: hc x wc x 3 (g1, g2, t). k = intensidad.
NX_KERNEL void wg_srl_apply(tdesc_t src, tdesc_t dst, int W, int H, __global const float* g, int wc, int hc, float k)
{
    SRL_LDS_DECL
    const int gx0 = GROUP_X * 8, gy0 = GROUP_Y * 8;
    srl_tile(L.t, src, W, H, gx0, gy0);
    WG_BARRIER
    srl_hpass(L);
    WG_BARRIER
    WG_BEGIN
    int lx = LID % 8, ly = LID / 8;
    int x = gx0 + lx, y = gy0 + ly;
    if (x < W && y < H) {
        float b1, b2, yc, lo, hi;
        srl_vpass(L, lx, ly, &b1, &b2, &yc, &lo, &hi);
        // ganancias: bilineal half-pixel de la malla 1/8 (bordes repetidos)
        float sy = (y + 0.5f) * 0.125f - 0.5f, sx = (x + 0.5f) * 0.125f - 0.5f;
        float fy0 = nx_floor(sy), fx0 = nx_floor(sx);
        float fy = sy - fy0, fx = sx - fx0;
        int ya = nx_clampi((int)fy0, 0, hc - 1), yb = nx_clampi((int)fy0 + 1, 0, hc - 1);
        int xa = nx_clampi((int)fx0, 0, wc - 1), xb = nx_clampi((int)fx0 + 1, 0, wc - 1);
        float G[3];
        for (int c = 0; c < 3; ++c) {
            float a = g[(ya * wc + xa) * 3 + c], b = g[(ya * wc + xb) * 3 + c];
            float d = g[(yb * wc + xa) * 3 + c], e = g[(yb * wc + xb) * 3 + c];
            G[c] = (a * (1 - fx) + b * fx) * (1 - fy) + (d * (1 - fx) + e * fx) * fy;
        }
        float th = G[2];
        float c1 = b1 - nx_clampf(b1, -th, th), c2 = b2 - nx_clampf(b2, -th, th);
        float yn = yc + k * (G[0] * c1 + G[1] * c2);
        float m = SRL_MARGIN * (hi - lo);
        yn = nx_clampf(yn, lo - m, hi + m);
        float dy = yn - yc;
        float4 c = nx_load(src, x, y);
        float4 o = nx_clamp4(c + (float4)(dy, dy, dy, 0.0f), (float4)(0.0f), (float4)(1.0f));
        o.w = c.w;
        nx_store(dst, x, y, o);
    }
    WG_END
}
