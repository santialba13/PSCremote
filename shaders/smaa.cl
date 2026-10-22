// SMAA 1x para PS4 (OpenCL C -> ISA GCN gfx701).
// Traduccion de SMAA.hlsl (Jimenez, Echevarria, Masia, Navarro, Gutierrez,
// licencia MIT, https://github.com/iryoku/smaa) a compute y a coordenadas en
// pixeles: u = pixel + 0.5 (centro). Sin busqueda diagonal
// (umbral 0.05, 8 pasos de busqueda, redondeo de esquinas 25 %).
// Referencia numerica: upscale/smaa.py.
//
// Tres pases: smaa_edges (bordes por luma) -> smaa_weights (busqueda de
// lineas + areas precalculadas) -> smaa_blend (mezcla con el vecino).
#include "nx_common.h"

#define SMAA_THRESHOLD 0.05f   // preset "Ultra": los juegos ya traen FXAA/TAA y sus bordes quedan con poco contraste
#define SMAA_MAX_SEARCH_STEPS 8   // preset Medium: la mitad de busqueda (la parte cara); en bordes de juego casi no cambia
#define SMAA_CORNER_ROUNDING_NORM 0.25f
#define SMAA_LCA_FACTOR 2.0f
#define AREATEX_W 160
#define AREATEX_H 560
#define SEARCHTEX_W 64
#define SEARCHTEX_H 16

static float lumaw(float4 c) { return c.x * 0.2126f + c.y * 0.7152f + c.z * 0.0722f; }

static float L(tdesc_t src, int W, int H, int x, int y)
{
    return lumaw(nx_load(src, nx_clampi(x, 0, W - 1), nx_clampi(y, 0, H - 1)));
}

// ---------------------------------------------------------------- pase 1
// bordes como imagen RGBA8 (r = oeste, g = norte; 0 o 1), como la textura RG8 del
// SMAA original: el pase 2 la lee con el filtro bilineal del hardware (2.75; antes
// era un buffer de bytes con el bilineal hecho a mano: 8 lecturas por muestra)
NX_KERNEL void smaa_edges(tdesc_t src, int W, int H, tdesc_t edges)
{
    int x = GID_X, y = GID_Y;
    if (x >= W || y >= H) return;
    float l = L(src, W, H, x, y);
    float dl = nx_absf(l - L(src, W, H, x - 1, y)), dt = nx_absf(l - L(src, W, H, x, y - 1));
    float ex = dl >= SMAA_THRESHOLD ? 1.0f : 0.0f, ey = dt >= SMAA_THRESHOLD ? 1.0f : 0.0f;
    if (ex + ey == 0.0f) { nx_store(edges, x, y, (float4)(0.0f)); return; }
    float dr = nx_absf(l - L(src, W, H, x + 1, y)), db = nx_absf(l - L(src, W, H, x, y + 1));
    float mx = nx_maxf(dl, dr), my = nx_maxf(dt, db);
    float dll = nx_absf(L(src, W, H, x - 1, y) - L(src, W, H, x - 2, y));
    float dtt = nx_absf(L(src, W, H, x, y - 1) - L(src, W, H, x, y - 2));
    mx = nx_maxf(mx, dll); my = nx_maxf(my, dtt);
    float fin = nx_maxf(mx, my);
    // adaptacion al contraste local
    ex *= fin <= SMAA_LCA_FACTOR * dl ? 1.0f : 0.0f;
    ey *= fin <= SMAA_LCA_FACTOR * dt ? 1.0f : 0.0f;
    nx_store(edges, x, y, (float4)(ex, ey, 0.0f, 0.0f));
}

// ---------------------------------------------------------------- pase 2
// muestreo bilineal (clamp al borde, coordenadas en texels: u = pixel + 0.5) con el
// sampler del hardware, como SMAA.hlsl
static float2 samp_e(tdesc_t t, tdesc_t s, float ux, float uy)
{
    float4 v = nx_sample(t, s, ux, uy);
    return (float2)(v.x, v.y);
}

static float search_length(tdesc_t st, tdesc_t s, float2 e, float offset)
{
    // SMAASearchLength en texels de la textura empaquetada (64x16)
    float sx = 66.0f * 0.5f - 1.0f, sy = -33.0f + 1.0f;
    float bx = 66.0f * offset + 0.5f, by = 33.0f - 0.5f;
    return nx_sample(st, s, sx * e.x + bx, sy * e.y + by).x;
}

static float search_x_left(tdesc_t ed, tdesc_t s, tdesc_t st,
                           float ux, float uy, float end)
{
    float2 e = (float2)(0.0f, 1.0f);
    while (ux > end && e.y > 0.8281f && e.x == 0.0f) {
        e = samp_e(ed, s, ux, uy);
        ux -= 2.0f;
    }
    float off = -(255.0f / 127.0f) * search_length(st, s, e, 0.0f) + 3.25f;
    return ux + off;
}

static float search_x_right(tdesc_t ed, tdesc_t s, tdesc_t st,
                            float ux, float uy, float end)
{
    float2 e = (float2)(0.0f, 1.0f);
    while (ux < end && e.y > 0.8281f && e.x == 0.0f) {
        e = samp_e(ed, s, ux, uy);
        ux += 2.0f;
    }
    float off = -(255.0f / 127.0f) * search_length(st, s, e, 0.5f) + 3.25f;
    return ux - off;
}

static float search_y_up(tdesc_t ed, tdesc_t s, tdesc_t st,
                         float ux, float uy, float end)
{
    float2 e = (float2)(1.0f, 0.0f);
    while (uy > end && e.x > 0.8281f && e.y == 0.0f) {
        e = samp_e(ed, s, ux, uy);
        uy -= 2.0f;
    }
    float off = -(255.0f / 127.0f) * search_length(st, s, e.yx, 0.0f) + 3.25f;
    return uy + off;
}

static float search_y_down(tdesc_t ed, tdesc_t s, tdesc_t st,
                           float ux, float uy, float end)
{
    float2 e = (float2)(1.0f, 0.0f);
    while (uy < end && e.x > 0.8281f && e.y == 0.0f) {
        e = samp_e(ed, s, ux, uy);
        uy += 2.0f;
    }
    float off = -(255.0f / 127.0f) * search_length(st, s, e.yx, 0.5f) + 3.25f;
    return uy - off;
}

static float2 area(tdesc_t at, tdesc_t s, float2 sqrt_d, float e1, float e2)
{
    float tx = 16.0f * __builtin_rintf(4.0f * e1) + sqrt_d.x + 0.5f;
    float ty = 16.0f * __builtin_rintf(4.0f * e2) + sqrt_d.y + 0.5f;
    return samp_e(at, s, tx, ty);
}

static float sat(float v) { return nx_clampf(v, 0.0f, 1.0f); }

// pesos en bytes, como la textura RGBA8 del SMAA original
NX_KERNEL void smaa_weights(tdesc_t edges, tdesc_t s, int W, int H, tdesc_t area_tex, tdesc_t search_tex,
                            __global uchar* weights)
{
    int x = GID_X, y = GID_Y;
    if (x >= W || y >= H) return;
    float ux = x + 0.5f, uy = y + 0.5f;
    float4 w = (float4)(0.0f);
    float4 ec = nx_load(edges, x, y);
    float er = ec.x, eg = ec.y;   // 0 o 1
    const float S2 = 2.0f * SMAA_MAX_SEARCH_STEPS;

    if (eg > 0.0f) {   // borde al norte: linea horizontal
        float cx = search_x_left(edges, s, search_tex, ux - 0.25f, uy - 0.125f, ux - 0.25f - S2);
        float cy = uy - 0.25f;
        float e1 = samp_e(edges, s, cx, cy).x;
        float cz = search_x_right(edges, s, search_tex, ux + 1.25f, uy - 0.125f, ux + 1.25f + S2);
        float2 d = (float2)(nx_absf(__builtin_rintf(cx - ux)), nx_absf(__builtin_rintf(cz - ux)));
        float2 sd = (float2)(__builtin_sqrtf(d.x), __builtin_sqrtf(d.y));
        float e2 = samp_e(edges, s, cz + 1.0f, cy).x;
        float2 a = area(area_tex, s, sd, e1, e2);
        // esquinas
        float lx = d.x <= d.y ? 1.0f : 0.0f, ly = d.y <= d.x ? 1.0f : 0.0f;
        float rx = (1.0f - SMAA_CORNER_ROUNDING_NORM) * lx / (lx + ly);
        float ry = (1.0f - SMAA_CORNER_ROUNDING_NORM) * ly / (lx + ly);
        float fx = 1.0f - rx * samp_e(edges, s, cx, uy + 1.0f).x - ry * samp_e(edges, s, cz + 1.0f, uy + 1.0f).x;
        float fy = 1.0f - rx * samp_e(edges, s, cx, uy - 2.0f).x - ry * samp_e(edges, s, cz + 1.0f, uy - 2.0f).x;
        w.x = a.x * sat(fx);
        w.y = a.y * sat(fy);
    }
    if (er > 0.0f) {   // borde al oeste: linea vertical
        float cy = search_y_up(edges, s, search_tex, ux - 0.125f, uy - 0.25f, uy - 0.25f - S2);
        float cx = ux - 0.25f;
        float e1 = samp_e(edges, s, cx, cy).y;
        float cz = search_y_down(edges, s, search_tex, ux - 0.125f, uy + 1.25f, uy + 1.25f + S2);
        float2 d = (float2)(nx_absf(__builtin_rintf(cy - uy)), nx_absf(__builtin_rintf(cz - uy)));
        float2 sd = (float2)(__builtin_sqrtf(d.x), __builtin_sqrtf(d.y));
        float e2 = samp_e(edges, s, cx, cz + 1.0f).y;
        float2 a = area(area_tex, s, sd, e1, e2);
        float lx = d.x <= d.y ? 1.0f : 0.0f, ly = d.y <= d.x ? 1.0f : 0.0f;
        float rx = (1.0f - SMAA_CORNER_ROUNDING_NORM) * lx / (lx + ly);
        float ry = (1.0f - SMAA_CORNER_ROUNDING_NORM) * ly / (lx + ly);
        float fx = 1.0f - rx * samp_e(edges, s, ux + 1.0f, cy).y - ry * samp_e(edges, s, ux + 1.0f, cz + 1.0f).y;
        float fy = 1.0f - rx * samp_e(edges, s, ux - 2.0f, cy).y - ry * samp_e(edges, s, ux - 2.0f, cz + 1.0f).y;
        w.z = a.x * sat(fx);
        w.w = a.y * sat(fy);
    }
    __global uchar* o = weights + (y * W + x) * 4;
    o[0] = (uchar)(w.x * 255.0f + 0.5f); o[1] = (uchar)(w.y * 255.0f + 0.5f);
    o[2] = (uchar)(w.z * 255.0f + 0.5f); o[3] = (uchar)(w.w * 255.0f + 0.5f);
}

// ---------------------------------------------------------------- pase 3
NX_KERNEL void smaa_blend(tdesc_t src, tdesc_t lin_sampler, __global const uchar* weights,
                          int W, int H, tdesc_t dst)
{
    int x = GID_X, y = GID_Y;
    if (x >= W || y >= H) return;
    int xr = x + 1 < W ? x + 1 : W - 1, yb = y + 1 < H ? y + 1 : H - 1;
    const float k = 1.0f / 255.0f;
    float ax = weights[(y * W + xr) * 4 + 3] * k;   // derecha (.a)
    float ay = weights[(yb * W + x) * 4 + 1] * k;   // vecino en +y (.g)
    float aw = weights[(y * W + x) * 4 + 0] * k;    // propio (.r)
    float az = weights[(y * W + x) * 4 + 2] * k;    // propio (.b)
    float ux = x + 0.5f, uy = y + 0.5f;
    float4 col;
    if (ax + ay + az + aw < 1e-5f) {
        col = nx_load(src, x, y);
    } else {
        bool h = nx_maxf(ax, az) > nx_maxf(ay, aw);
        float4 off = h ? (float4)(ax, 0.0f, az, 0.0f) : (float4)(0.0f, ay, 0.0f, aw);
        float2 bw = h ? (float2)(ax, az) : (float2)(ay, aw);
        bw /= bw.x + bw.y;
        col = bw.x * nx_sample(src, lin_sampler, ux + off.x, uy + off.y)
            + bw.y * nx_sample(src, lin_sampler, ux - off.z, uy - off.w);
    }
    nx_store(dst, x, y, col);
}
