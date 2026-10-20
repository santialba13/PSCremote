// NXSR: realce / reescalado del fotograma final (ver sr/net_sr.py).
// Toda la red en un pase por grupo de 8x8 celdas de media resolucion (16x16
// pixeles): la cabeza se calcula sobre la celda + halo en memoria local (LDS),
// el cuerpo y la cola en registros. Sin buffers intermedios a pantalla completa.
//
//   celdas (s2d x2, 12 canales) -> cabeza 3x3 (12->C) -> [cuerpo 3x3 (C->C) + residual]
//   -> cola 1x1 (C->12) -> d2s -> y = limitar(x + k * r)
//
// Igual que JAX: el conv "SAME" rellena con ceros fuera de la imagen (tambien
// la salida de la cabeza fuera de la imagen vale 0 para el cuerpo), y el
// limitador usa el min/max 3x3 de x con los bordes repetidos.
#include "nx_common.h"

#ifndef SR_C
#define SR_C 16
#endif
#define SR_T (NX_WG + 2)            // celdas de la cabeza por lado (tile + halo de 1)
#define LRELU(v) ((v) > 0.0f ? (v) : 0.1f * (v))

// layout de pesos (sr/export_sr.py): cabeza HWIO [3][3][12][C], b[C];
// cuerpo [3][3][C][C], b[C]; cola [C][12], b[12]
#define W_HEAD 0
#define B_HEAD (W_HEAD + 9 * 12 * SR_C)
#define W_BODY (B_HEAD + SR_C)
#define B_BODY (W_BODY + 9 * SR_C * SR_C)
#define W_TAIL (B_BODY + SR_C)
#define B_TAIL (W_TAIL + SR_C * 12)

// canal ch de la celda (cy, cx) del tensor s2d: pixel (2cy+sy, 2cx+sx), color c
static float cell(tdesc_t src, int cy, int cx, int ch)
{
    int s = ch / 3, c = ch - 3 * s;
    float4 p = nx_load(src, 2 * cx + (s & 1), 2 * cy + (s >> 1));
    return c == 0 ? p.x : (c == 1 ? p.y : p.z);
}

NX_KERNEL void wg_nxsr(tdesc_t src, int W, int H, __constant const float* w, tdesc_t dst, float k)
{
    __local float head[SR_T * SR_T * SR_C];
    const int Wc = W / 2, Hc = H / 2;
    const int cy0 = GROUP_Y * NX_WG - 1, cx0 = GROUP_X * NX_WG - 1;

    // fase 1: cabeza en la celda + halo (ceros fuera de la imagen)
    WG_BEGIN
    for (int t = LID; t < SR_T * SR_T; t += NX_WG * NX_WG) {
        int cy = cy0 + t / SR_T, cx = cx0 + t % SR_T;
        float acc[SR_C];
        for (int o = 0; o < SR_C; ++o) acc[o] = 0.0f;
        if (cy >= 0 && cx >= 0 && cy < Hc && cx < Wc) {
            for (int o = 0; o < SR_C; ++o) acc[o] = w[B_HEAD + o];
            for (int ky = 0; ky < 3; ++ky) {
                int iy = cy + ky - 1;
                if (iy < 0 || iy >= Hc) continue;
                for (int kx = 0; kx < 3; ++kx) {
                    int ix = cx + kx - 1;
                    if (ix < 0 || ix >= Wc) continue;
                    for (int i = 0; i < 12; ++i) {
                        float v = cell(src, iy, ix, i);
                        __constant const float* wk = w + W_HEAD + ((ky * 3 + kx) * 12 + i) * SR_C;
                        for (int o = 0; o < SR_C; ++o) acc[o] = __builtin_fmaf(v, wk[o], acc[o]);
                    }
                }
            }
            for (int o = 0; o < SR_C; ++o) acc[o] = LRELU(acc[o]);
        }
        for (int o = 0; o < SR_C; ++o) head[t * SR_C + o] = acc[o];
    }
    WG_END
    WG_BARRIER

    // fase 2: cuerpo + residual + cola + limitador, una celda (4 pixeles) por hilo
    WG_BEGIN
    int ly = LID / NX_WG, lx = LID % NX_WG;
    int cy = cy0 + 1 + ly, cx = cx0 + 1 + lx;
    if (cy < Hc && cx < Wc) {
        float h[SR_C];
        for (int o = 0; o < SR_C; ++o) h[o] = w[B_BODY + o];
        for (int ky = 0; ky < 3; ++ky)
            for (int kx = 0; kx < 3; ++kx) {
                __local const float* hv = head + ((ly + ky) * SR_T + (lx + kx)) * SR_C;
                for (int i = 0; i < SR_C; ++i) {
                    float v = hv[i];
                    __constant const float* wk = w + W_BODY + ((ky * 3 + kx) * SR_C + i) * SR_C;
                    for (int o = 0; o < SR_C; ++o) h[o] = __builtin_fmaf(v, wk[o], h[o]);
                }
            }
        __local const float* hc = head + ((ly + 1) * SR_T + (lx + 1)) * SR_C;
        for (int o = 0; o < SR_C; ++o) h[o] = LRELU(h[o]) + hc[o];
        for (int s = 0; s < 4; ++s) {
            int px = 2 * cx + (s & 1), py = 2 * cy + (s >> 1);
            float r[3];
            for (int c = 0; c < 3; ++c) {
                int ch = s * 3 + c;
                float a = w[B_TAIL + ch];
                for (int i = 0; i < SR_C; ++i) a = __builtin_fmaf(h[i], w[W_TAIL + i * 12 + ch], a);
                r[c] = a;
            }
            float4 x = nx_load(src, px, py);
            float4 lo = x, hi = x;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    float4 q = nx_load(src, nx_clampi(px + dx, 0, W - 1), nx_clampi(py + dy, 0, H - 1));
                    lo = nx_min4(lo, q); hi = nx_max4(hi, q);
                }
            float4 y = (float4)(x.x + k * r[0], x.y + k * r[1], x.z + k * r[2], 1.0f);
            y = nx_clamp4(y, lo, hi);
            y.w = 1.0f;
            nx_store(dst, px, py, y);
        }
    }
    WG_END
}
