// FSR 1 (EASU + RCAS) para PS4 en OpenCL C -> ISA GCN gfx701 con clang.
// Traduccion de la ruta float de ffx_fsr1.h (AMD FidelityFX FSR, licencia MIT,
// Copyright (c) 2021 Advanced Micro Devices, Inc.). Referencia numerica:
// upscale/fsr1.py.
#include "nx_common.h"

static float lum(float4 c) { return c.z * 0.5f + (c.x * 0.5f + c.y); }

static void easu_set(float2* dir, float* len, float w,
                     float lA, float lB, float lC, float lD, float lE)
{
    float dc = lD - lC, cb = lC - lB;
    float lenX = nx_rcp(nx_maxf(nx_maxf(nx_absf(dc), nx_absf(cb)), 1e-8f));
    float dX = lD - lB;
    dir->x += dX * w;
    lenX = nx_clampf(nx_absf(dX) * lenX, 0.0f, 1.0f); lenX *= lenX;
    *len += lenX * w;
    float ec = lE - lC, ca = lC - lA;
    float lenY = nx_rcp(nx_maxf(nx_maxf(nx_absf(ec), nx_absf(ca)), 1e-8f));
    float dY = lE - lA;
    dir->y += dY * w;
    lenY = nx_clampf(nx_absf(dY) * lenY, 0.0f, 1.0f); lenY *= lenY;
    *len += lenY * w;
}

static void easu_tap(float4* aC, float* aW, float2 off, float2 dir, float2 len2,
                     float lob, float clp, float4 c)
{
    float2 v = (float2)(off.x * dir.x + off.y * dir.y, off.x * -dir.y + off.y * dir.x) * len2;
    float d2 = nx_minf(v.x * v.x + v.y * v.y, clp);
    float wB = 0.4f * d2 - 1.0f, wA = lob * d2 - 1.0f;
    wB *= wB; wA *= wA;
    wB = 25.0f / 16.0f * wB - (25.0f / 16.0f - 1.0f);
    float w = wB * wA;
    *aC += c * w; *aW += w;
}

// src: imagen de entrada (render del juego o fotograma generado), dst: salida
NX_KERNEL void easu(tdesc_t src, int H, int W, tdesc_t dst, int Ho, int Wo)
{
    int ox = GID_X, oy = GID_Y;
    if (ox >= Wo || oy >= Ho) return;
    float sx = (float)W / Wo, sy = (float)H / Ho;
    float2 pp = (float2)(ox * sx + 0.5f * sx - 0.5f, oy * sy + 0.5f * sy - 0.5f);
    float2 fp = (float2)(nx_floor(pp.x), nx_floor(pp.y));
    pp -= fp;
    int fx = (int)fp.x, fy = (int)fp.y;
#define T(dx, dy) nx_load(src, nx_clampi(fx + (dx), 0, W - 1), nx_clampi(fy + (dy), 0, H - 1))
    float4 b = T(0, -1), c = T(1, -1);
    float4 e = T(-1, 0), f = T(0, 0), g = T(1, 0), h = T(2, 0);
    float4 i = T(-1, 1), j = T(0, 1), k = T(1, 1), l = T(2, 1);
    float4 n = T(0, 2), o = T(1, 2);
#undef T
    float bL = lum(b), cL = lum(c), eL = lum(e), fL = lum(f), gL = lum(g), hL = lum(h);
    float iL = lum(i), jL = lum(j), kL = lum(k), lL = lum(l), nL = lum(n), oL = lum(o);
    float2 dir = (float2)(0.0f);
    float len = 0.0f;
    easu_set(&dir, &len, (1 - pp.x) * (1 - pp.y), bL, eL, fL, gL, jL);
    easu_set(&dir, &len, pp.x * (1 - pp.y), cL, fL, gL, hL, kL);
    easu_set(&dir, &len, (1 - pp.x) * pp.y, fL, iL, jL, kL, nL);
    easu_set(&dir, &len, pp.x * pp.y, gL, jL, kL, lL, oL);
    float dirR = dir.x * dir.x + dir.y * dir.y;
    bool zro = dirR < 1.0f / 32768.0f;
    dirR = zro ? 1.0f : nx_rsqrt(dirR);
    dir.x = zro ? 1.0f : dir.x;
    dir *= dirR;
    len = len * 0.5f;
    len *= len;
    float stretch = (dir.x * dir.x + dir.y * dir.y) * nx_rcp(nx_maxf(nx_maxf(nx_absf(dir.x), nx_absf(dir.y)), 1e-8f));
    float2 len2 = (float2)(1.0f + (stretch - 1.0f) * len, 1.0f - 0.5f * len);
    float lob = 0.5f + ((1.0f / 4.0f - 0.04f) - 0.5f) * len;
    float clp = nx_rcp(lob);
    float4 aC = (float4)(0.0f);
    float aW = 0.0f;
    easu_tap(&aC, &aW, (float2)(0, -1) - pp, dir, len2, lob, clp, b);
    easu_tap(&aC, &aW, (float2)(1, -1) - pp, dir, len2, lob, clp, c);
    easu_tap(&aC, &aW, (float2)(-1, 1) - pp, dir, len2, lob, clp, i);
    easu_tap(&aC, &aW, (float2)(0, 1) - pp, dir, len2, lob, clp, j);
    easu_tap(&aC, &aW, (float2)(0, 0) - pp, dir, len2, lob, clp, f);
    easu_tap(&aC, &aW, (float2)(-1, 0) - pp, dir, len2, lob, clp, e);
    easu_tap(&aC, &aW, (float2)(1, 1) - pp, dir, len2, lob, clp, k);
    easu_tap(&aC, &aW, (float2)(2, 1) - pp, dir, len2, lob, clp, l);
    easu_tap(&aC, &aW, (float2)(2, 0) - pp, dir, len2, lob, clp, h);
    easu_tap(&aC, &aW, (float2)(1, 0) - pp, dir, len2, lob, clp, g);
    easu_tap(&aC, &aW, (float2)(1, 2) - pp, dir, len2, lob, clp, o);
    easu_tap(&aC, &aW, (float2)(0, 2) - pp, dir, len2, lob, clp, n);
    float4 mn = nx_min4(nx_min4(f, g), nx_min4(j, k)), mx = nx_max4(nx_max4(f, g), nx_max4(j, k));
    nx_store(dst, ox, oy, nx_clamp4(aC * nx_rcp(aW), mn, mx));
}

#define RCAS_LIMIT (0.25f - 1.0f / 16.0f)

static float y601(float4 c) { return (c.x * 0.299f + c.y * 0.587f) + c.z * 0.114f; }

// sharp = exp2(-nitidez_en_stops); denoise = 1 activa FSR_RCAS_DENOISE.
// detail > 0: "FSR+" (VCFG-NX), despues de RCAS un realce de detalle de radio 2
// sobre la luma (recupera lo que suavizan TAA/FXAA, que RCAS con su limite no
// alcanza), sin halos (el realce no saca la luma del rango 3x3 local +-0.01) y sin
// subir el grano (diferencias de menos de 0.004 no se tocan). Referencia:
// upscale/fsr1.py:rcas.
NX_KERNEL void rcas(tdesc_t src, tdesc_t dst, int W, int H, float sharp, int denoise, float detail)
{
    int x = GID_X, y = GID_Y;
    if (x >= W || y >= H) return;
#define L(dx, dy) nx_load(src, nx_clampi(x + (dx), 0, W - 1), nx_clampi(y + (dy), 0, H - 1))
    float4 b = L(0, -1), d = L(-1, 0), e = L(0, 0), f = L(1, 0), h = L(0, 1);
    float4 mn4 = nx_min4(nx_min4(b, d), nx_min4(f, h)), mx4 = nx_max4(nx_max4(b, d), nx_max4(f, h));
    float lobe = -1.0f;
    for (int c = 0; c < 3; ++c) {
        float mn = c == 0 ? mn4.x : (c == 1 ? mn4.y : mn4.z);
        float mx = c == 0 ? mx4.x : (c == 1 ? mx4.y : mx4.z);
        float ec = c == 0 ? e.x : (c == 1 ? e.y : e.z);
        float hitMin = nx_minf(mn, ec) * nx_rcp(nx_maxf(4.0f * mx, 1e-8f));
        float hitMax = (1.0f - nx_maxf(mx, ec)) * nx_rcp(nx_minf(4.0f * mn - 4.0f, -1e-8f));
        lobe = nx_maxf(lobe, nx_maxf(-hitMin, hitMax));
    }
    lobe = nx_maxf(-RCAS_LIMIT, nx_minf(lobe, 0.0f)) * sharp;
    if (denoise) {
        float bL = lum(b), dL = lum(d), eL = lum(e), fL = lum(f), hL = lum(h);
        float nz = 0.25f * (bL + dL + fL + hL) - eL;
        float r = nx_maxf(nx_maxf(nx_maxf(bL, dL), nx_maxf(eL, fL)), hL)
                - nx_minf(nx_minf(nx_minf(bL, dL), nx_minf(eL, fL)), hL);
        nz = nx_clampf(nx_absf(nz) * nx_rcp(nx_maxf(r, 1e-8f)), 0.0f, 1.0f);
        lobe *= -0.5f * nz + 1.0f;
    }
    float4 o = (lobe * (b + d + f + h) + e) * nx_rcp(4.0f * lobe + 1.0f);
    if (detail > 0.0f) {
        float4 a = L(-1, -1), c = L(1, -1), g = L(-1, 1), i = L(1, 1);
        float yb = y601(b), yd = y601(d), ye = y601(e), yf = y601(f), yh = y601(h);
        float ya = y601(a), yc = y601(c), yg = y601(g), yi = y601(i);
        float far = (y601(L(0, -2)) + y601(L(-2, 0))) + (y601(L(2, 0)) + y601(L(0, 2)));
        float blur = (((4.0f * ye + 2.0f * ((yb + yd) + (yf + yh))) + ((ya + yc) + (yg + yi))) + far) * (1.0f / 20.0f);
        float dt = ye - blur;
        float wn = nx_clampf((nx_absf(dt) - 0.004f) * (1.0f / 0.016f), 0.0f, 1.0f);
        float mnY = nx_minf(nx_minf(nx_minf(ya, yb), nx_minf(yc, yd)), nx_minf(nx_minf(ye, yf), nx_minf(nx_minf(yg, yh), yi)));
        float mxY = nx_maxf(nx_maxf(nx_maxf(ya, yb), nx_maxf(yc, yd)), nx_maxf(nx_maxf(ye, yf), nx_maxf(nx_maxf(yg, yh), yi)));
        float yo = y601(o);
        // el limite nunca recorta lo que ya hizo RCAS: solo acota el realce extra
        float tgt = nx_clampf(yo + detail * dt * wn, nx_minf(mnY, yo) - 0.01f, nx_maxf(mxY, yo) + 0.01f);
        float dy = tgt - yo;
        o = o + (float4)(dy, dy, dy, 0.0f);
    }
#undef L
    nx_store(dst, x, y, nx_clamp4(o, (float4)(0.0f), (float4)(1.0f)));
}
