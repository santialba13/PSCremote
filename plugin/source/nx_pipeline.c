#include "nx_pipeline.h"

#include <math.h>
#include <string.h>

#include "../include/nx_shaders.h"
#include "../include/nxflow_weights.h"
#include "../include/nxsr_weights.h"
#include "../include/nxsr_hq_weights.h"
#include "../../third_party/smaa/AreaTex.h"
#include "../../third_party/smaa/SearchTex.h"

// 2^x sin libm: exp2f no esta en todas las versiones de libSceLibcInternal y una
// importacion que el juego no resuelve impide que GoldHEN cargue el plugin entero
static float nx_exp2(float x)
{
    int k = (int)(x < 0 ? x - 1.0f : x);   // parte entera hacia abajo
    float f = (x - (float)k) * 0.69314718f, t = 1.0f, e = 1.0f;
    for (int i = 1; i < 10; ++i) { t *= f / (float)i; e += t; }   // e^(f ln2), f en [0, 1)
    for (; k > 0; --k) e *= 2.0f;
    for (; k < 0; ++k) e *= 0.5f;
    return e;
}

enum {
    K_CONV_14_32, K_CONV_17_32, K_CONV_32_32, K_CONV_32_5, K_CONV_17_16, K_CONV_16_16, K_CONV_16_5,
    K_AVGPOOL, K_RESIZE, K_UPSAMPLE5, K_BUILD_INPUT, K_BM_TO_FLOW, K_LUMA_FULL, K_LUMA_DOWN2,
    K_BM_SEARCH, K_BM_MEDIAN3, K_WARP_BLEND, K_EASU, K_RCAS,
    K_SMAA_EDGES, K_SMAA_WEIGHTS, K_SMAA_BLEND, K_NXSR, K_NXSR_HQ, K_COUNT
};
static const char* const k_names[K_COUNT] = {
    "conv3x3_14_32", "conv3x3_17_32", "conv3x3_32_32", "conv3x3_32_5",
    "conv3x3_17_16", "conv3x3_16_16", "conv3x3_16_5",
    "avgpool_rgb", "resize_bilinear", "upsample5", "build_level_input", "bm_to_flow", "luma_full",
    "luma_down2", "bm_search", "bm_median3", "warp_blend", "easu", "rcas",
    "smaa_edges", "smaa_weights", "smaa_blend", "wg_nxsr", "wg_nxsr_hq",
};

// primera capa y numero de capas de cada bloque en nx_layers[] (model/export.py)
#define BLOCK_A 0, 6
#define BLOCK_B 6, 6
#define BLOCK_C 12, 5

#define KERNARG_BYTES (16 * 1024)
#define DESC_BYTES 4096
#define WG 8

// --- layouts de kernargs: mismo orden y alineacion natural que los .cl
typedef struct { uint64_t in; int32_t H, W; uint64_t out; int32_t Ho, Wo; uint64_t w, b;
                 int32_t stride, pad_top, pad_left, act; uint64_t residual; } ka_conv_t;
typedef struct { uint64_t src; int32_t H, W; uint64_t out; int32_t Ho, Wo, f; } ka_avgpool_t;
typedef struct { uint64_t in; int32_t H, W, C, _pad; uint64_t out; int32_t Ho, Wo; } ka_resize_t;
typedef struct { uint64_t a0, a1, flow5, delta5; int32_t H, W, level, _pad; uint64_t out; } ka_build_t;
typedef struct { uint64_t vec; int32_t H, W; uint64_t flow5; } ka_bm2flow_t;
typedef struct { uint64_t in; int32_t H, W; uint64_t out; } ka_up5_t;
typedef struct { uint64_t src; int32_t Hs, Ws; uint64_t out; int32_t H, W; } ka_luma_t;
typedef struct { uint64_t in; int32_t H, W; uint64_t out; } ka_down2_t;
typedef struct { uint64_t l0, l1; int32_t H, W; uint64_t coarse; int32_t cBy, cBx;
                 uint64_t vec; int32_t By, Bx, radius; } ka_bm_t;
typedef struct { uint64_t in, out; int32_t By, Bx; } ka_median_t;
typedef struct { uint64_t f0, f1, samp, low5; int32_t h4, w4; uint64_t dst; int32_t H, W; } ka_warp_t;
typedef struct { uint64_t src; int32_t H, W; uint64_t dst; int32_t Ho, Wo; } ka_easu_t;
typedef struct { uint64_t src, dst; int32_t W, H; float sharp; int32_t denoise; } ka_rcas_t;
typedef struct { uint64_t src; int32_t W, H; uint64_t edges; } ka_smaa_edges_t;
typedef struct { uint64_t edges; int32_t W, H; uint64_t area, search, weights; } ka_smaa_weights_t;
typedef struct { uint64_t src, samp, weights; int32_t W, H; uint64_t dst; } ka_smaa_blend_t;
typedef struct { uint64_t src; int32_t W, H; uint64_t w; uint64_t dst; float k; } ka_nxsr_t;

static size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

static uint64_t arena_alloc(nx_arena_t* a, size_t bytes, size_t align)
{
    size_t off = align_up(a->used, align);
    if (off + bytes > a->size)
        return 0;
    a->used = off + bytes;
    return a->gpu + off;
}

static void* arena_cpu(const nx_arena_t* a, uint64_t gpu) { return a->cpu + (gpu - a->gpu); }

static size_t tensor_bytes(int h, int w, int c) { return align_up((size_t)h * w * c * 4, 256); }

static size_t max_sz(size_t a, size_t b) { return a > b ? a : b; }

// Reparto de la arena. Los temporales del FG (flujo, entradas de nivel, capas
// ocultas) y los del post-proceso (SMAA, intermedios) se usan en fases
// distintas del mismo trabajo, separadas por barreras de computo, y nunca
// sobreviven al trabajo: comparten una sola region ("scratch"). Lo que pasa de
// un fotograma al siguiente (lumas, pooling del fotograma anterior, tmp_img,
// que lee el post-proceso despues del FG) va aparte.
static void alloc_fg_scratch(nx_pipeline_t* p, nx_arena_t* a)
{
    for (int lv = 0; lv < 3; ++lv)
        p->vec[lv] = arena_alloc(a, tensor_bytes(p->h8 >> lv, p->w8 >> lv, 2), 256);
    p->vtmp = arena_alloc(a, tensor_bytes(p->h8, p->w8, 2), 256);
    p->flow5 = arena_alloc(a, tensor_bytes(p->h8, p->w8, 5), 256);
    p->delta5 = arena_alloc(a, tensor_bytes(p->h8, p->w8, 5), 256);
    p->flow4 = arena_alloc(a, tensor_bytes(p->h4, p->w4, 5), 256);
    p->delta4 = arena_alloc(a, tensor_bytes(p->h4, p->w4, 5), 256);
    p->lvl_in = arena_alloc(a, tensor_bytes(p->h4, p->w4, 17), 256);
    size_t hid = max_sz(tensor_bytes(p->h16, p->w16, 32), tensor_bytes(p->h8, p->w8, 16));
    p->hA = arena_alloc(a, hid, 256);
    p->hB = arena_alloc(a, hid, 256);
    p->hR = arena_alloc(a, hid, 256);
    p->up = arena_alloc(a, max_sz(tensor_bytes(p->h8, p->w8, 32), tensor_bytes(p->h4, p->w4, 16)), 256);
}

static void alloc_post_scratch(nx_pipeline_t* p, nx_arena_t* a)
{
    p->tmp_img2 = arena_alloc(a, tensor_bytes(p->Ho, (int)align_up(p->Wo, 64), 1), 64 * 1024);
    p->tmp_aa = arena_alloc(a, tensor_bytes(p->H, (int)align_up(p->W, 64), 1), 64 * 1024);
    p->smaa_edges = arena_alloc(a, (size_t)p->W * p->H * 2, 256);
    p->smaa_weights = arena_alloc(a, (size_t)p->W * p->H * 4, 256);
}

// dry = 1: solo mide (la arena no tiene memoria detras)
static int layout(nx_pipeline_t* p, int W, int H, int Wo, int Ho, nx_arena_t arena, int dry)
{
    memset(p, 0, sizeof(*p));
    p->W = W; p->H = H; p->Wo = Wo; p->Ho = Ho;
    p->Wp = (int)align_up(W, 32); p->Hp = (int)align_up(H, 32);
    p->w4 = p->Wp / 4; p->h4 = p->Hp / 4;
    p->w8 = p->Wp / 8; p->h8 = p->Hp / 8; p->w16 = p->w8 / 2; p->h16 = p->h8 / 2;
    p->arena = arena;
    nx_arena_t* a = &p->arena;

    for (int k = 0; k < K_COUNT; ++k) {
        p->shader[k] = -1;
        for (int i = 0; i < NX_NUM_SHADERS; ++i)
            if (!strcmp(nx_shaders[i].name, k_names[k])) p->shader[k] = i;
        if (p->shader[k] < 0) return -1;
        const nx_shader_t* s = &nx_shaders[p->shader[k]];
        p->shader_gpu[k] = arena_alloc(a, s->code_dwords * 4, 256);
        if (!p->shader_gpu[k]) return -2;
        if (!dry) memcpy(arena_cpu(a, p->shader_gpu[k]), s->code, s->code_dwords * 4);
    }
    for (int l = 0; l < NX_NUM_LAYERS; ++l) {
        const nx_layer_t* L = &nx_layers[l];
        size_t wn = (size_t)9 * L->cin * L->cout;
        p->w_gpu[l] = arena_alloc(a, wn * 4, 256);
        p->b_gpu[l] = arena_alloc(a, L->cout * 4, 256);
        if (!p->w_gpu[l] || !p->b_gpu[l]) return -3;
        if (!dry) {
            memcpy(arena_cpu(a, p->w_gpu[l]), L->w, wn * 4);
            memcpy(arena_cpu(a, p->b_gpu[l]), L->b, L->cout * 4);
        }
    }
    for (int s = 0; s < 2; ++s)
        for (int lv = 0; lv < 3; ++lv)
            p->luma[s][lv] = arena_alloc(a, tensor_bytes(p->Hp >> lv, p->Wp >> lv, 1), 256);
    for (int s = 0; s < 2; ++s) {
        p->a8[s] = arena_alloc(a, tensor_bytes(p->h8, p->w8, 3), 256);
        p->a4[s] = arena_alloc(a, tensor_bytes(p->h4, p->w4, 3), 256);
    }
    p->tmp_img = arena_alloc(a, tensor_bytes(H, (int)align_up(W, 64), 1), 64 * 1024);

    // region compartida: del tamano del mayor de los dos grupos
    nx_arena_t m1 = { NULL, 1u << 20, (size_t)-1 >> 1, 0 }, m2 = m1;   // base != 0: 0 es "sin memoria"
    nx_pipeline_t probe = *p;
    alloc_fg_scratch(&probe, &m1);
    alloc_post_scratch(&probe, &m2);
    size_t scratch_sz = max_sz(m1.used, m2.used) + 64 * 1024;
    uint64_t scratch = arena_alloc(a, scratch_sz, 64 * 1024);
    if (!scratch) return -6;
    nx_arena_t sa = { dry ? NULL : (uint8_t*)arena_cpu(a, scratch), scratch, scratch_sz, 0 };
    alloc_fg_scratch(p, &sa);
    sa.used = 0;
    alloc_post_scratch(p, &sa);

    p->smaa_area = arena_alloc(a, sizeof areaTexBytes, 256);
    p->smaa_search = arena_alloc(a, sizeof searchTexBytes, 256);
    p->nxsr_w = arena_alloc(a, sizeof nxsr_weights, 256);
    p->nxsr_hq_w = arena_alloc(a, sizeof nxsr_hq_weights, 256);
    if (!p->smaa_search || !p->nxsr_w || !p->nxsr_hq_w) return -5;
    if (!dry) {
        memcpy(arena_cpu(a, p->nxsr_w), nxsr_weights, sizeof nxsr_weights);
        memcpy(arena_cpu(a, p->nxsr_hq_w), nxsr_hq_weights, sizeof nxsr_hq_weights);
        memcpy(arena_cpu(a, p->smaa_area), areaTexBytes, sizeof areaTexBytes);
        memcpy(arena_cpu(a, p->smaa_search), searchTexBytes, sizeof searchTexBytes);
    }
    for (int s = 0; s < 2; ++s) {
        p->kernarg_base[s] = arena_alloc(a, KERNARG_BYTES, 256);
        p->desc_base[s] = arena_alloc(a, DESC_BYTES, 256);
    }
    p->label_gpu = arena_alloc(a, 256, 256);
    if (!p->label_gpu || !p->tmp_img2 || !p->up || !p->smaa_weights) return -4;
    if (!dry) {
        p->label_cpu = (volatile uint32_t*)arena_cpu(a, p->label_gpu);
        *p->label_cpu = 0;
    }
    return 0;
}

size_t nx_pipeline_mem_size(int W, int H, int Wo, int Ho)
{
    static nx_pipeline_t p;   // grande: fuera de la pila
    nx_arena_t m = { NULL, 1u << 20, (size_t)-1 >> 1, 0 };
    if (layout(&p, W, H, Wo, Ho, m, 1) != 0) return 0;
    return p.arena.used + 64 * 1024;
}

int nx_pipeline_init(nx_pipeline_t* p, int W, int H, int Wo, int Ho, nx_arena_t arena)
{
    return layout(p, W, H, Wo, Ho, arena, 0);
}

// ------------------------------------------------------------ emision

typedef struct {
    nx_pipeline_t* p;
    pm4_t* c;
    uint64_t ka, ka_end, desc, desc_end;
    int err;
} ctx_t;

static uint64_t put_kernargs(ctx_t* x, const void* data, size_t n)
{
    uint64_t at = align_up(x->ka, 64);
    if (at + n > x->ka_end) { x->err = 1; return 0; }
    memcpy(arena_cpu(&x->p->arena, at), data, n);
    x->ka = at + n;
    return at;
}

static uint64_t put_desc(ctx_t* x, const uint32_t* d, int ndw)
{
    uint64_t at = align_up(x->desc, 32);
    if (at + ndw * 4 > x->desc_end) { x->err = 1; return 0; }
    memcpy(arena_cpu(&x->p->arena, at), d, ndw * 4);
    x->desc = at + ndw * 4;
    return at;
}

static uint64_t tsharp(ctx_t* x, const nx_image_t* img)
{
    uint32_t t[8];
    nx_make_tsharp(t, img);
    return put_desc(x, t, 8);
}

static void run(ctx_t* x, int k, const void* args, size_t n, int tx, int ty)
{
    const nx_shader_t* s = &nx_shaders[x->p->shader[k]];
    // el struct C puede tener relleno al final (sizeof redondea a 8): se copian
    // exactamente los bytes que declara el kernel; si faltan, el layout no coincide
    if (n < s->kernarg_size || n > s->kernarg_size + 7) { x->err = 2; return; }
    uint64_t ka = put_kernargs(x, args, s->kernarg_size);
    uint64_t pgm = x->p->shader_gpu[k];
    uint32_t pg[2] = { (uint32_t)(pgm >> 8), (uint32_t)(pgm >> 40) };
    uint32_t rs[2] = { s->rsrc1, s->rsrc2 };
    uint32_t nt[3] = { WG, WG, 1 };
    // ABI HSA: s[0:3] = private segment buffer (sin scratch -> 0), s[4:5] = kernarg ptr
    uint32_t ud[6] = { 0, 0, 0, 0, (uint32_t)ka, (uint32_t)(ka >> 32) };
    pm4_set_sh(x->c, R_COMPUTE_PGM_LO, pg, 2);
    pm4_set_sh(x->c, R_COMPUTE_PGM_RSRC1, rs, 2);
    pm4_set_sh(x->c, R_COMPUTE_NUM_THREAD_X, nt, 3);
    pm4_set_sh(x->c, R_COMPUTE_USER_DATA_0, ud, s->user_sgprs < 6 ? s->user_sgprs : 6);
    pm4_dispatch(x->c, (uint32_t)(tx + WG - 1) / WG, (uint32_t)(ty + WG - 1) / WG, 1);
    pm4_cs_barrier(x->c);
}

static void same_pads(int n, int stride, int* before)
{
    int out = (n + stride - 1) / stride;
    int total = (out - 1) * stride + 3 - n;
    if (total < 0) total = 0;
    *before = total / 2;
}

static int conv_kernel(int cin, int cout)
{
    if (cout == 32) return cin == 14 ? K_CONV_14_32 : cin == 17 ? K_CONV_17_32 : K_CONV_32_32;
    if (cout == 16) return cin == 17 ? K_CONV_17_16 : K_CONV_16_16;
    return cin == 32 ? K_CONV_32_5 : K_CONV_16_5;
}

static void conv(ctx_t* x, int layer, uint64_t in, int H, int W, uint64_t out, uint64_t residual)
{
    nx_pipeline_t* p = x->p;
    const nx_layer_t* L = &nx_layers[layer];
    int s = L->stride, Ho = (H + s - 1) / s, Wo = (W + s - 1) / s, pt, pl;
    same_pads(H, s, &pt);
    same_pads(W, s, &pl);
    ka_conv_t a = { in, H, W, out, Ho, Wo, p->w_gpu[layer], p->b_gpu[layer], s, pt, pl, L->act, residual };
    run(x, conv_kernel(L->cin, L->cout), &a, sizeof a, Wo, Ho);
}

// Bloque de NXFlow (ver model/net.py: block): c0 s2, c1, capas medias (la ultima
// suma el residual de c1), resize x2 y conv de salida. Entrada en lvl_in (H x W),
// salida (5 canales) en "delta".
static void nx_block(ctx_t* x, int first, int n_layers, int H, int W, uint64_t delta)
{
    nx_pipeline_t* p = x->p;
    int h = (H + 1) / 2, w = (W + 1) / 2;
    conv(x, first + 0, p->lvl_in, H, W, p->hA, 0);
    conv(x, first + 1, p->hA, h, w, p->hR, 0);
    uint64_t src = p->hR;
    int n_mid = n_layers - 3;
    for (int i = 0; i < n_mid; ++i) {
        uint64_t dst = src == p->hA ? p->hB : p->hA;
        conv(x, first + 2 + i, src, h, w, dst, i == n_mid - 1 ? p->hR : 0);
        src = dst;
    }
    ka_resize_t r = { src, h, w, nx_layers[first].cout, 0, p->up, H, W };
    run(x, K_RESIZE, &r, sizeof r, W, H);
    conv(x, first + n_layers - 1, p->up, H, W, delta, 0);
}

static void luma_pyramid(ctx_t* x, const nx_image_t* img, int s)
{
    nx_pipeline_t* p = x->p;
    ka_luma_t l = { tsharp(x, img), p->H, p->W, p->luma[s][0], p->Hp, p->Wp };
    run(x, K_LUMA_FULL, &l, sizeof l, p->Wp, p->Hp);
    for (int lv = 1; lv < 3; ++lv) {
        ka_down2_t d = { p->luma[s][lv - 1], p->Hp >> (lv - 1), p->Wp >> (lv - 1), p->luma[s][lv] };
        run(x, K_LUMA_DOWN2, &d, sizeof d, p->Wp >> lv, p->Hp >> lv);
    }
    ka_avgpool_t a = { tsharp(x, img), p->H, p->W, p->a8[s], p->h8, p->w8, 8 };
    run(x, K_AVGPOOL, &a, sizeof a, p->w8, p->h8);
    ka_avgpool_t q = { tsharp(x, img), p->H, p->W, p->a4[s], p->h4, p->w4, 4 };
    run(x, K_AVGPOOL, &q, sizeof q, p->w4, p->h4);
}

static nx_image_t linear_img(uint64_t gpu, int w, int h)
{
    nx_image_t i = { gpu, (uint32_t)w, (uint32_t)h, (uint32_t)align_up(w, 64),
                     NX_FMT_RGBA8_UNORM, NX_TILE_DISPLAY_LINEAR_ALIGNED };
    return i;
}

// SMAA 1x de "src" a "dst" (misma resolucion: la del juego)
static void smaa(ctx_t* x, const nx_image_t* src, const nx_image_t* dst)
{
    nx_pipeline_t* p = x->p;
    uint32_t samp[4];
    nx_make_ssharp(samp, 1);
    ka_smaa_edges_t e = { tsharp(x, src), p->W, p->H, p->smaa_edges };
    run(x, K_SMAA_EDGES, &e, sizeof e, p->W, p->H);
    ka_smaa_weights_t w = { p->smaa_edges, p->W, p->H, p->smaa_area, p->smaa_search, p->smaa_weights };
    run(x, K_SMAA_WEIGHTS, &w, sizeof w, p->W, p->H);
    ka_smaa_blend_t b = { tsharp(x, src), put_desc(x, samp, 4), p->smaa_weights, p->W, p->H, tsharp(x, dst) };
    run(x, K_SMAA_BLEND, &b, sizeof b, p->W, p->H);
}

// post-proceso de "src" (resolucion del juego) hacia "dst" (salida):
//   [SMAA] -> [EASU si la salida es mas grande] -> [NXSR o RCAS]; el ultimo pase escribe en dst
static void post(ctx_t* x, const nx_job_t* job, const nx_image_t* src, const nx_image_t* dst)
{
    nx_pipeline_t* p = x->p;
    enum { ST_SMAA, ST_EASU, ST_NXSR, ST_RCAS };
    int st[4], n = 0;
    if (job->aa) st[n++] = ST_SMAA;
    if (job->post == NX_POST_EASU) st[n++] = ST_EASU;
    if (job->sr) st[n++] = ST_NXSR;
    else if (job->rcas || job->post == NX_POST_RCAS) st[n++] = ST_RCAS;
    if (n == 0) { x->err = 3; return; }
    nx_image_t taa = linear_img(p->tmp_aa, p->W, p->H);
    nx_image_t t2 = linear_img(p->tmp_img2, p->Wo, p->Ho);
    const nx_image_t* cur = src;
    for (int i = 0; i < n; ++i) {
        // cada tamano tiene su intermedio: W x H solo lo usa SMAA, Wo x Ho solo EASU
        const nx_image_t* o = i == n - 1 ? dst : (st[i] == ST_EASU ? &t2 : &taa);
        int w = (int)o->width, h = (int)o->height;
        if (st[i] == ST_SMAA) {
            smaa(x, cur, o);
        } else if (st[i] == ST_EASU) {
            ka_easu_t e = { tsharp(x, cur), p->H, p->W, tsharp(x, o), p->Ho, p->Wo };
            run(x, K_EASU, &e, sizeof e, p->Wo, p->Ho);
        } else if (st[i] == ST_NXSR) {
            ka_nxsr_t r = { tsharp(x, cur), w, h, job->sr_hq ? p->nxsr_hq_w : p->nxsr_w, tsharp(x, o), job->sr_k };
            run(x, job->sr_hq ? K_NXSR_HQ : K_NXSR, &r, sizeof r, w / 2, h / 2);
        } else {
            ka_rcas_t r = { tsharp(x, cur), tsharp(x, o), w, h, nx_exp2(-job->sharpness_stops), 0 };
            run(x, K_RCAS, &r, sizeof r, w, h);
        }
        cur = o;
    }
}

static void draw_osd(ctx_t* x, const nx_job_t* job, const nx_image_t* img)
{
    if (!job->osd_ctx) return;
    if (nx_osd_draw(job->osd_ctx, x->c, job->slot, img, job->osd, 2) != 0) x->err = 4;
}

int nx_pipeline_build_osd_only(nx_osd_ctx_t* o, pm4_t* c, const nx_job_t* job, uint64_t label_gpu)
{
    nx_osd_begin(o, job->slot);
    if (nx_osd_draw(o, c, job->slot, job->cur, job->osd, 2) != 0) return -14;
    pm4_end_of_job(c, label_gpu, job->seq);
    pm4_nop_pad(c, 8);
    return c->overflow ? -20 : 0;
}

static void frame_generation(ctx_t* x, const nx_job_t* job, const nx_image_t* prev, const nx_image_t* cur,
                             const nx_image_t* warp_dst)
{
    nx_pipeline_t* p = x->p;
    int sc = job->slot, sp = job->slot ^ 1;

    // 1. luma (piramide) y RGB a 1/8: el del fotograma previo se reutiliza
    if (!p->luma_valid[sp])
        luma_pyramid(x, prev, sp);
    luma_pyramid(x, cur, sc);
    p->luma_valid[sc] = 1;

    // 2. block-matching simetrico de grueso a fino (prev -> cur)
    for (int lv = 2; lv >= 0; --lv) {
        int By = p->h8 >> lv, Bx = p->w8 >> lv;
        ka_bm_t b = { p->luma[sp][lv], p->luma[sc][lv], p->Hp >> lv, p->Wp >> lv,
                      lv == 2 ? 0 : p->vec[lv + 1], p->h8 >> (lv + 1), p->w8 >> (lv + 1),
                      p->vtmp, By, Bx, 3 };
        run(x, K_BM_SEARCH, &b, sizeof b, Bx, By);
        ka_median_t m = { p->vtmp, p->vec[lv], By, Bx };
        run(x, K_BM_MEDIAN3, &m, sizeof m, Bx, By);
    }

    // 3. NXFlow: niveles A y B a 1/8, nivel C a 1/4
    ka_bm2flow_t f = { p->vec[0], p->h8, p->w8, p->flow5 };
    run(x, K_BM_TO_FLOW, &f, sizeof f, p->w8, p->h8);
    ka_build_t bi = { p->a8[sp], p->a8[sc], p->flow5, 0, p->h8, p->w8, 0, 0, p->lvl_in };
    run(x, K_BUILD_INPUT, &bi, sizeof bi, p->w8, p->h8);
    nx_block(x, BLOCK_A, p->h8, p->w8, p->delta5);
    bi.delta5 = p->delta5; bi.level = 1;                 // flow5 += A; entrada de B
    run(x, K_BUILD_INPUT, &bi, sizeof bi, p->w8, p->h8);
    nx_block(x, BLOCK_B, p->h8, p->w8, p->delta5);
    bi.level = -1;                                       // flow5 += B
    run(x, K_BUILD_INPUT, &bi, sizeof bi, p->w8, p->h8);
    ka_up5_t u = { p->flow5, p->h8, p->w8, p->flow4 };
    run(x, K_UPSAMPLE5, &u, sizeof u, p->w4, p->h4);
    ka_build_t bc = { p->a4[sp], p->a4[sc], p->flow4, 0, p->h4, p->w4, 1, 0, p->lvl_in };
    run(x, K_BUILD_INPUT, &bc, sizeof bc, p->w4, p->h4);
    nx_block(x, BLOCK_C, p->h4, p->w4, p->delta4);
    bc.delta5 = p->delta4; bc.level = -1;                // flow4 += C
    run(x, K_BUILD_INPUT, &bc, sizeof bc, p->w4, p->h4);

    // 4. warp + mezcla a resolucion completa (con el sampler bilineal del hardware)
    uint32_t samp[4];
    nx_make_ssharp(samp, 1);
    ka_warp_t w = { tsharp(x, prev), tsharp(x, cur), put_desc(x, samp, 4), p->flow4,
                    p->h4, p->w4, tsharp(x, warp_dst), p->H, p->W };
    run(x, K_WARP_BLEND, &w, sizeof w, p->W, p->H);
}

int nx_pipeline_build(nx_pipeline_t* p, pm4_t* c, const nx_job_t* job)
{
    ctx_t x = { p, c, p->kernarg_base[job->slot], p->kernarg_base[job->slot] + KERNARG_BYTES,
                p->desc_base[job->slot], p->desc_base[job->slot] + DESC_BYTES, 0 };
    if (job->osd_ctx) nx_osd_begin(job->osd_ctx, job->slot);

    pm4_set_sh1(c, R_COMPUTE_TMPRING_SIZE, 0);
    uint32_t mgmt[2] = { 0xFFFFFFFFu, 0xFFFFFFFFu };
    pm4_set_sh(c, R_COMPUTE_STATIC_THREAD_MGMT_SE0, mgmt, 2);
    pm4_set_sh1(c, R_COMPUTE_RESOURCE_LIMITS, 0);

    int any_post = job->post != NX_POST_NONE || job->aa || job->sr || job->rcas;
    if (any_post && !job->out_real) return -15;
    if (job->fg) {
        if (job->enhance_first) {
            // FG sobre fotogramas ya procesados: el post-proceso corre una sola vez por
            // fotograma del juego y el intermedio hereda la calidad del real
            if (!job->prev_real_valid) {
                post(&x, job, job->prev, job->prev_real);
                p->luma_valid[job->slot ^ 1] = 0;
            }
            post(&x, job, job->cur, job->out_real);
            frame_generation(&x, job, job->prev_real, job->out_real, job->out_mid);
        } else {
            nx_image_t t1 = linear_img(p->tmp_img, p->W, p->H);
            frame_generation(&x, job, job->prev, job->cur, any_post ? &t1 : job->out_mid);
            if (any_post) {
                post(&x, job, &t1, job->out_mid);
                post(&x, job, job->cur, job->out_real);
            }
        }
    } else if (any_post) {
        post(&x, job, job->cur, job->out_real);
    }
    if (job->compare && job->out_real && job->osd_ctx
        && nx_osd_copy_left(job->osd_ctx, c, job->slot, job->cur, job->out_real) != 0)
        x.err = 5;
    if (job->fg) draw_osd(&x, job, job->out_mid);
    draw_osd(&x, job, job->out_real ? job->out_real : job->cur);

    pm4_end_of_job(c, p->label_gpu, job->seq);
    pm4_nop_pad(c, 8);
    return x.err ? -10 - x.err : (c->overflow ? -20 : 0);
}
