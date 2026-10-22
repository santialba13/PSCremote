// NXRE ("realismo"): transformacion de color fotorrealista por zonas y niveles de
// brillo (sr/net_re.py). La red (convs genericas de nxflow.cl) deja una malla
// gh x gw x D x 12 (afin 3x4 residual por celda y nivel de brillo):
//   re_ema    promedio con la malla del fotograma anterior (estabilidad temporal)
//   re_apply  por pixel: guia g = m . rgb + c, malla bilineal en el espacio y lineal
//             entre los dos niveles de brillo vecinos, y = x + s * (A [rgb 1])
#include "nx_common.h"

#define RE_D 8

// a = 0: copia (primer fotograma); si no, state = a * state + (1 - a) * g
NX_KERNEL void re_ema(__global const float* g, __global float* state, int n, float a)
{
    int i = GID_Y * 1024 + GID_X;   // despacho 1024 x ceil(n / 1024)
    if (GID_X >= 1024 || i >= n) return;
    state[i] = a * state[i] + (1.0f - a) * g[i];
}

static float4 re_ld4(__global const float* p) { return (float4)(p[0], p[1], p[2], p[3]); }

NX_KERNEL void re_apply(tdesc_t src, tdesc_t dst, int W, int H, __global const float* grid, int gw, int gh,
                        float m0, float m1, float m2, float mc, float s)
{
    int x = GID_X, y = GID_Y;
    if (x >= W || y >= H) return;
    float4 c = nx_load(src, x, y);
    float g = nx_clampf((c.x * m0 + c.y * m1) + c.z * m2 + mc, 0.0f, 1.0f);
    float z = g * RE_D - 0.5f;
    float z0 = nx_floor(z), fz = z - z0;
    int i0 = nx_clampi((int)z0, 0, RE_D - 1), i1 = nx_clampi((int)z0 + 1, 0, RE_D - 1);
    // bilineal half-pixel de la malla con bordes repetidos (jax.image.resize)
    float sy = (y + 0.5f) * ((float)gh / H) - 0.5f, sx = (x + 0.5f) * ((float)gw / W) - 0.5f;
    float fy0 = nx_floor(sy), fx0 = nx_floor(sx);
    float fy = sy - fy0, fx = sx - fx0;
    int ya = nx_clampi((int)fy0, 0, gh - 1), yb = nx_clampi((int)fy0 + 1, 0, gh - 1);
    int xa = nx_clampi((int)fx0, 0, gw - 1), xb = nx_clampi((int)fx0 + 1, 0, gw - 1);
    float w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy), w10 = (1 - fx) * fy, w11 = fx * fy;
    int cell[4] = { ya * gw + xa, ya * gw + xb, yb * gw + xa, yb * gw + xb };
    float ws[4] = { w00, w01, w10, w11 };
    float4 r0 = (float4)(0.0f), r1 = (float4)(0.0f), r2 = (float4)(0.0f);
    for (int k = 0; k < 4; ++k) {
        __global const float* q0 = grid + (cell[k] * RE_D + i0) * 12;
        __global const float* q1 = grid + (cell[k] * RE_D + i1) * 12;
        float a = ws[k] * (1.0f - fz), b = ws[k] * fz;
        r0 += a * re_ld4(q0) + b * re_ld4(q1);
        r1 += a * re_ld4(q0 + 4) + b * re_ld4(q1 + 4);
        r2 += a * re_ld4(q0 + 8) + b * re_ld4(q1 + 8);
    }
    // filas de la afin: [a_r0 a_r1 a_r2 t_r] [a_g0 ...] [a_b0 ...]
    float dr = (r0.x * c.x + r0.y * c.y) + r0.z * c.z + r0.w;
    float dg = (r1.x * c.x + r1.y * c.y) + r1.z * c.z + r1.w;
    float db = (r2.x * c.x + r2.y * c.y) + r2.z * c.z + r2.w;
    float4 o = nx_clamp4(c + s * (float4)(dr, dg, db, 0.0f), (float4)(0.0f), (float4)(1.0f));
    o.w = c.w;
    nx_store(dst, x, y, o);
}
