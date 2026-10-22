#include "nx_pipeline.h"

#include <math.h>
#include <string.h>

#include "../include/nx_shaders.h"
#include "../include/nxsr_weights.h"
#include "../include/nxsr_hq_weights.h"
#include "../include/nxsrl_weights.h"
#include "../include/nxre_weights.h"
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
    K_CONV_24_24,
    K_AVGPOOL, K_RESIZE, K_UPSAMPLE5, K_BM_TO_FLOW, K_LUMA_HALF,
    K_LUMA_DOWN2, K_BM_SEARCH, K_BM_MEDIAN3, K_SCENE, K_SMASK, K_DILATE, K_WARP_BLEND, K_EASU, K_RCAS,
    K_SMAA_EDGES, K_SMAA_WEIGHTS, K_SMAA_BLEND, K_NXSR, K_NXSR_HQ,
    K_CONV_5_12, K_CONV_12_12, K_CONV_12_3, K_SRL_FEAT, K_SRL_ACT, K_SRL_APPLY,
    K_CONV_5_16, K_CONV_16_16, K_CONV_16_3,
    K_CONV_3_16, K_CONV_16_24, K_CONV_24_96, K_RE_EMA, K_RE_APPLY, K_FG3_SELECT, K_FG3_WARP, K_COUNT
};
static const char* const k_names[K_COUNT] = {
    "conv3x3_24_24",
    "avgpool_rgb", "resize_bilinear", "upsample5",
    "bm_to_flow", "luma_half", "luma_down2", "wg_bm_search", "bm_median3", "wg_nxfg_scene", "static_mask", "mask_dilate", "warp_blend",
    "easu", "rcas", "smaa_edges", "smaa_weights", "smaa_blend", "wg_nxsr", "wg_nxsr_hq",
    "conv3x3_5_12", "conv3x3_12_12", "conv3x3_12_3", "wg_srl_feat", "srl_act", "wg_srl_apply",
    "conv3x3_5_16", "conv3x3_16_16", "conv3x3_16_3",
    "conv3x3_3_16", "conv3x3_16_24", "conv3x3_24_96", "re_ema", "re_apply", "fg3_select", "fg3_warp",
};

// radios del block-matching por nivel (L1 fino .. L4 grueso), como model/bm2.py
static const int bm_radius[4] = { 2, 3, 3, 4 };

#define KERNARG_BYTES (16 * 1024)
#define DESC_BYTES 4096
#define WG 8

// --- layouts de kernargs: mismo orden y alineacion natural que los .cl
typedef struct { uint64_t in; int32_t H, W; uint64_t out; int32_t Ho, Wo; uint64_t w, b;
                 int32_t stride, pad_top, pad_left, act; uint64_t residual; } ka_conv_t;
typedef struct { uint64_t src; int32_t H, W; uint64_t out; int32_t Ho, Wo, f; } ka_avgpool_t;
typedef struct { uint64_t in; int32_t H, W, C, _pad; uint64_t out; int32_t Ho, Wo; } ka_resize_t;
typedef struct { uint64_t vec; int32_t n, _pad; uint64_t flag; } ka_scene_t;
typedef struct { uint64_t vec; int32_t H, W; uint64_t flow5; } ka_bm2flow_t;
typedef struct { uint64_t in; int32_t H, W; uint64_t out; } ka_up5_t;
typedef struct { uint64_t src; int32_t Hs, Ws; uint64_t out; int32_t H, W; } ka_luma_t;
typedef struct { uint64_t in; int32_t H, W; uint64_t out; } ka_down2_t;
typedef struct { uint64_t l0, l1; int32_t H, W; uint64_t coarse; int32_t cBy, cBx;
                 uint64_t vec; int32_t By, Bx, radius, finest; } ka_bm_t;
typedef struct { uint64_t in, out; int32_t By, Bx; } ka_median_t;
typedef struct { uint64_t f0, f1, samp, low5; int32_t hl, wl; float scale; int32_t _pad; uint64_t flag, smask, dst;
                 int32_t H, W; } ka_warp_t;
typedef struct { uint64_t q0, q1; int32_t H, W; uint64_t low5; int32_t hl, wl; uint64_t out; } ka_fg3_select_t;
typedef struct { uint64_t f0, f1, samp, fq; int32_t hq, wq; uint64_t flag, smask, dst; int32_t H, W; } ka_fg3_warp_t;
typedef struct { uint64_t l0, l1; int32_t H, W; uint64_t out; } ka_smask_t;
typedef struct { uint64_t src; int32_t H, W; uint64_t dst; int32_t Ho, Wo; } ka_easu_t;
typedef struct { uint64_t src, dst; int32_t W, H; float sharp; int32_t denoise; float detail; } ka_rcas_t;
typedef struct { uint64_t src; int32_t W, H; uint64_t edges; } ka_smaa_edges_t;
typedef struct { uint64_t edges, samp; int32_t W, H; uint64_t area, search, weights; } ka_smaa_weights_t;
typedef struct { uint64_t src, samp, weights; int32_t W, H; uint64_t dst; } ka_smaa_blend_t;
typedef struct { uint64_t src; int32_t W, H; uint64_t w; uint64_t dst; float k; } ka_nxsr_t;
typedef struct { uint64_t src; int32_t W, H; uint64_t out; int32_t wc, hc; } ka_srl_feat_t;
typedef struct { uint64_t g; int32_t n; } ka_srl_act_t;
typedef struct { uint64_t src, dst; int32_t W, H; uint64_t g; int32_t wc, hc; float k, margin; } ka_srl_apply_t;
typedef struct { uint64_t g, state; int32_t n; float a; } ka_re_ema_t;
typedef struct { uint64_t src, dst; int32_t W, H; uint64_t grid; int32_t gw, gh; float m0, m1, m2, mc, s; } ka_re_apply_t;

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
    for (int lv = 0; lv < 4; ++lv)
        p->vec[lv] = arena_alloc(a, tensor_bytes(p->bm_h[lv], p->bm_w[lv], 3), 256);
    p->vtmp = arena_alloc(a, tensor_bytes(p->h16, p->w16, 3), 256);
    p->flowG = arena_alloc(a, tensor_bytes(p->h16, p->w16, 5), 256);
    p->flow8 = arena_alloc(a, tensor_bytes(p->h8, p->w8, 5), 256);
    p->fq = arena_alloc(a, tensor_bytes(p->h4, p->w4, 4), 256);   // NXFG-3: flujo elegido a 1/4
    p->scene = arena_alloc(a, 256, 256);
    p->smask = arena_alloc(a, tensor_bytes(p->lum_h[0], p->lum_w[0], 1), 256);
    p->smask_e = arena_alloc(a, tensor_bytes(p->lum_h[0], p->lum_w[0], 1), 256);
}

static void alloc_post_scratch(nx_pipeline_t* p, nx_arena_t* a)
{
    // NXSR-L: guia a 1/8 de la etapa mas grande (la salida)
    int wc = (p->Wo + 7) / 8, hc = (p->Ho + 7) / 8;
    p->srl_feat = arena_alloc(a, tensor_bytes(hc, wc, 5), 256);
    for (int i = 0; i < 3; ++i) p->srl_h[i] = arena_alloc(a, tensor_bytes(hc, wc, 16), 256);
    p->srl_g = arena_alloc(a, tensor_bytes(hc, wc, 3), 256);
    // NXRE: imagen chica y 4 tensores de la red (todos caben en el del primer nivel)
    p->re_lo = arena_alloc(a, tensor_bytes(p->re_hl, p->re_wl, 3), 256);
    for (int i = 0; i < 4; ++i) p->re_t[i] = arena_alloc(a, tensor_bytes(p->re_h[0], p->re_w_[0], 24), 256);
    p->re_grid = arena_alloc(a, tensor_bytes(p->re_gh, p->re_gw, NX_RE_D * 12), 256);
    p->tmp_img2 = arena_alloc(a, tensor_bytes(p->Ho, (int)align_up(p->Wo, 64), 1), 64 * 1024);
    p->tmp_aa = arena_alloc(a, tensor_bytes(p->H, (int)align_up(p->W, 64), 1), 64 * 1024);
    p->smaa_edges = arena_alloc(a, tensor_bytes(p->H, (int)align_up(p->W, 64), 1), 256);   // imagen RGBA8 lineal
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
    for (int lv = 0; lv < 4; ++lv) {   // block-matching: bloques de 8 sobre la luma L1..L4 (media resolucion)
        p->lum_h[lv] = (p->Hp / 2) >> lv; p->lum_w[lv] = (p->Wp / 2) >> lv;
        p->bm_h[lv] = (p->lum_h[lv] + 7) / 8; p->bm_w[lv] = (p->lum_w[lv] + 7) / 8;
    }
    // NXRE: imagen de ~128 de ancho (promedio f x f) y 4 convs de paso 2 (padding 1)
    p->re_f = (int)((float)W / NX_RE_LOW_W + 0.5f);
    if (p->re_f < 1) p->re_f = 1;
    p->re_hl = H / p->re_f; p->re_wl = W / p->re_f;
    for (int i = 0; i < 4; ++i) {
        p->re_h[i] = ((i ? p->re_h[i - 1] : p->re_hl) - 1) / 2 + 1;
        p->re_w_[i] = ((i ? p->re_w_[i - 1] : p->re_wl) - 1) / 2 + 1;
    }
    p->re_gh = p->re_h[1]; p->re_gw = p->re_w_[1];   // la malla: 1/4 de la imagen chica
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
    for (int s = 0; s < 2; ++s)
        for (int lv = 0; lv < 4; ++lv)
            p->luma[s][lv] = arena_alloc(a, tensor_bytes(p->lum_h[lv], p->lum_w[lv], 1), 256);
    for (int s = 0; s < 2; ++s) p->a4[s] = arena_alloc(a, tensor_bytes(p->h4, p->w4, 3), 256);   // NXFG-3: promedios 4x4
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

    // texturas de area (160x560, RG) y busqueda (64x16, R) de SMAA como imagenes RGBA8
    // lineales: el pase de pesos las lee con el filtro bilineal del hardware
    p->smaa_area = arena_alloc(a, (size_t)align_up(AREATEX_WIDTH, 64) * AREATEX_HEIGHT * 4, 256);
    p->smaa_search = arena_alloc(a, (size_t)align_up(SEARCHTEX_WIDTH, 64) * SEARCHTEX_HEIGHT * 4, 256);
    p->nxsr_w = arena_alloc(a, sizeof nxsr_weights, 256);
    for (int l = 0; l < NX_RE_LAYERS; ++l) {
        const nx_re_layer_t* L = &nx_re_layers[l];
        size_t wn = (size_t)9 * L->cin * L->cout;
        p->re_w[l] = arena_alloc(a, wn * 4, 256);
        p->re_b[l] = arena_alloc(a, L->cout * 4, 256);
        if (!p->re_w[l] || !p->re_b[l]) return -8;
        if (!dry) {
            memcpy(arena_cpu(a, p->re_w[l]), L->w, wn * 4);
            memcpy(arena_cpu(a, p->re_b[l]), L->b, L->cout * 4);
        }
    }
    p->re_state = arena_alloc(a, tensor_bytes(p->re_gh, p->re_gw, NX_RE_D * 12), 256);
    if (!p->re_state) return -8;
    for (int m = 0; m < 2; ++m) {
        int n = m ? NX_SRL_HQ_LAYERS : NX_SRL_LITE_LAYERS;
        for (int l = 0; l < n; ++l) {
            const nx_srl_layer_t* L = m ? &nx_srl_hq_layers[l] : &nx_srl_lite_layers[l];
            size_t wn = (size_t)9 * L->cin * L->cout;
            p->srl_w[m][l] = arena_alloc(a, wn * 4, 256);
            p->srl_b[m][l] = arena_alloc(a, L->cout * 4, 256);
            if (!p->srl_w[m][l] || !p->srl_b[m][l]) return -7;
            if (!dry) {
                memcpy(arena_cpu(a, p->srl_w[m][l]), L->w, wn * 4);
                memcpy(arena_cpu(a, p->srl_b[m][l]), L->b, L->cout * 4);
            }
        }
    }
    p->nxsr_hq_w = arena_alloc(a, sizeof nxsr_hq_weights, 256);
    if (!p->smaa_search || !p->nxsr_w || !p->nxsr_hq_w) return -5;
    if (!dry) {
        memcpy(arena_cpu(a, p->nxsr_w), nxsr_weights, sizeof nxsr_weights);
        memcpy(arena_cpu(a, p->nxsr_hq_w), nxsr_hq_weights, sizeof nxsr_hq_weights);
        uint8_t* at = arena_cpu(a, p->smaa_area);
        for (int y = 0; y < AREATEX_HEIGHT; ++y)
            for (int x = 0; x < AREATEX_WIDTH; ++x) {
                uint8_t* d = at + ((size_t)y * align_up(AREATEX_WIDTH, 64) + x) * 4;
                d[0] = areaTexBytes[(y * AREATEX_WIDTH + x) * 2];
                d[1] = areaTexBytes[(y * AREATEX_WIDTH + x) * 2 + 1];
                d[2] = d[3] = 0;
            }
        uint8_t* stx = arena_cpu(a, p->smaa_search);
        for (int y = 0; y < SEARCHTEX_HEIGHT; ++y)
            for (int x = 0; x < SEARCHTEX_WIDTH; ++x) {
                uint8_t* d = stx + ((size_t)y * align_up(SEARCHTEX_WIDTH, 64) + x) * 4;
                d[0] = searchTexBytes[y * SEARCHTEX_WIDTH + x];
                d[1] = d[2] = d[3] = 0;
            }
    }
    for (int s = 0; s < NX_PIPE_SLOTS; ++s) {
        p->kernarg_base[s] = arena_alloc(a, KERNARG_BYTES, 256);
        p->desc_base[s] = arena_alloc(a, DESC_BYTES, 256);
    }
    p->label_gpu = arena_alloc(a, 256, 256);
    if (!p->label_gpu || !p->tmp_img2 || !p->fq || !p->smaa_weights) return -4;
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

static int conv_kernel_name(const char* name)
{
    for (int k = 0; k < K_COUNT; ++k)
        if (k_names[k][0] == 'c' && k_names[k][4] == '3' && !strcmp(k_names[k], name)) return k;   // conv3x3_*
    return -1;
}

// NXSR-L (sr/net_srl.py): guia a 1/8 + realce de bandas con limite, de cur a o (w x h).
// model: 0 = Ligero, 1 = Calidad (la misma idea con una guia mas grande)
static void srl(ctx_t* x, const nx_image_t* cur, const nx_image_t* o, int w, int h, float k, int model)
{
    nx_pipeline_t* p = x->p;
    int wc = (w + 7) / 8, hc = (h + 7) / 8;
    ka_srl_feat_t f = { tsharp(x, cur), w, h, p->srl_feat, wc, hc };
    run(x, K_SRL_FEAT, &f, sizeof f, wc * 8, hc * 8);   // un grupo por celda (todos los pixeles: estable al moverse)
    int n = model ? NX_SRL_HQ_LAYERS : NX_SRL_LITE_LAYERS;
    uint64_t src = p->srl_feat;
    for (int l = 0; l < n; ++l) {
        const nx_srl_layer_t* L = model ? &nx_srl_hq_layers[l] : &nx_srl_lite_layers[l];
        int kk = conv_kernel_name(L->kernel);
        if (kk < 0) { x->err = 9; return; }
        // c0 -> h0; capas medias alternando h1/h2 (la ultima suma c0); salida -> g
        uint64_t dst = l == 0 ? p->srl_h[0] : l == n - 1 ? p->srl_g : src == p->srl_h[1] ? p->srl_h[2] : p->srl_h[1];
        ka_conv_t a = { src, hc, wc, dst, hc, wc, p->srl_w[model][l], p->srl_b[model][l], 1, 1, 1, L->act,
                        L->residual ? p->srl_h[0] : 0 };
        run(x, kk, &a, sizeof a, wc, hc);
        src = dst;
    }
    ka_srl_act_t ac = { p->srl_g, wc * hc };
    run(x, K_SRL_ACT, &ac, sizeof ac, 1024, (wc * hc + 1023) / 1024);
    ka_srl_apply_t ap = { tsharp(x, cur), tsharp(x, o), w, h, p->srl_g, wc, hc, k, model ? 0.35f : 0.25f };
    run(x, K_SRL_APPLY, &ap, sizeof ap, w, h);
}

// NXRE (sr/net_re.py): red sobre la imagen chica -> malla de afines por zona y nivel
// de brillo -> promedio con la del fotograma anterior -> por pixel, de cur a o.
static void re_conv(ctx_t* x, int l, uint64_t in, int H, int W, uint64_t out, uint64_t residual)
{
    nx_pipeline_t* p = x->p;
    const nx_re_layer_t* L = &nx_re_layers[l];
    int Ho = L->stride == 2 ? (H - 1) / 2 + 1 : H, Wo = L->stride == 2 ? (W - 1) / 2 + 1 : W;
    int k = conv_kernel_name(L->kernel);
    if (k < 0) { x->err = 9; return; }
    ka_conv_t a = { in, H, W, out, Ho, Wo, p->re_w[l], p->re_b[l], L->stride, 1, 1, L->act, residual };
    run(x, k, &a, sizeof a, Wo, Ho);
}

static void re(ctx_t* x, const nx_image_t* cur, const nx_image_t* o, int w, int h, float s)
{
    nx_pipeline_t* p = x->p;
    enum { S1, S2, S3, S4, G1, L1, L2, OUT };   // orden de nx_re_layers (sr/net_re.py LAYERS)
    uint64_t* T = p->re_t;
    ka_avgpool_t ap = { tsharp(x, cur), h, w, p->re_lo, p->re_hl, p->re_wl, p->re_f };
    run(x, K_AVGPOOL, &ap, sizeof ap, p->re_wl, p->re_hl);
    re_conv(x, S1, p->re_lo, p->re_hl, p->re_wl, T[0], 0);
    re_conv(x, S2, T[0], p->re_h[0], p->re_w_[0], T[1], 0);
    re_conv(x, S3, T[1], p->re_h[1], p->re_w_[1], T[2], 0);
    re_conv(x, S4, T[2], p->re_h[2], p->re_w_[2], T[3], 0);
    re_conv(x, G1, T[3], p->re_h[3], p->re_w_[3], T[2], 0);        // contexto global
    ka_resize_t r = { T[2], p->re_h[3], p->re_w_[3], 24, 0, T[0], p->re_gh, p->re_gw };
    run(x, K_RESIZE, &r, sizeof r, p->re_gw, p->re_gh);
    re_conv(x, L1, T[1], p->re_gh, p->re_gw, T[3], T[0]);           // local + global
    re_conv(x, L2, T[3], p->re_gh, p->re_gw, T[0], 0);
    re_conv(x, OUT, T[0], p->re_gh, p->re_gw, p->re_grid, 0);
    int n = p->re_gh * p->re_gw * NX_RE_D * 12;
    ka_re_ema_t e = { p->re_grid, p->re_state, n, p->re_valid ? 0.5f : 0.0f };
    run(x, K_RE_EMA, &e, sizeof e, 1024, (n + 1023) / 1024);
    p->re_valid = 1;
    ka_re_apply_t a = { tsharp(x, cur), tsharp(x, o), w, h, p->re_state, p->re_gw, p->re_gh,
                        nx_re_guide[0], nx_re_guide[1], nx_re_guide[2], nx_re_guide[3], s };
    run(x, K_RE_APPLY, &a, sizeof a, w, h);
}

// piramide de un fotograma: luma a media resolucion y 3 mitades (block-matching) y,
// para la eleccion de NXFG-3, RGB a 1/4
static void luma_pyramid(ctx_t* x, const nx_image_t* img, int s, int fast)
{
    nx_pipeline_t* p = x->p;
    ka_luma_t l = { tsharp(x, img), p->H, p->W, p->luma[s][0], p->lum_h[0], p->lum_w[0] };
    run(x, K_LUMA_HALF, &l, sizeof l, p->lum_w[0], p->lum_h[0]);
    for (int lv = 1; lv < 4; ++lv) {
        ka_down2_t d = { p->luma[s][lv - 1], p->lum_h[lv - 1], p->lum_w[lv - 1], p->luma[s][lv] };
        run(x, K_LUMA_DOWN2, &d, sizeof d, p->lum_w[lv], p->lum_h[lv]);
    }
    if (fast) return;   // los promedios RGB solo los usa la eleccion a 1/4
    ka_avgpool_t a = { tsharp(x, img), p->H, p->W, p->a4[s], p->h4, p->w4, 4 };
    run(x, K_AVGPOOL, &a, sizeof a, p->w4, p->h4);
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
    nx_image_t ei = linear_img(p->smaa_edges, p->W, p->H);
    nx_image_t ai = linear_img(p->smaa_area, AREATEX_WIDTH, AREATEX_HEIGHT);
    nx_image_t si = linear_img(p->smaa_search, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT);
    ka_smaa_edges_t e = { tsharp(x, src), p->W, p->H, tsharp(x, &ei) };
    run(x, K_SMAA_EDGES, &e, sizeof e, p->W, p->H);
    ka_smaa_weights_t w = { tsharp(x, &ei), put_desc(x, samp, 4), p->W, p->H, tsharp(x, &ai), tsharp(x, &si),
                            p->smaa_weights };
    run(x, K_SMAA_WEIGHTS, &w, sizeof w, p->W, p->H);
    ka_smaa_blend_t b = { tsharp(x, src), put_desc(x, samp, 4), p->smaa_weights, p->W, p->H, tsharp(x, dst) };
    run(x, K_SMAA_BLEND, &b, sizeof b, p->W, p->H);
}

// post-proceso de "src" (resolucion del juego) hacia "dst" (salida):
//   [SMAA] -> [EASU si la salida es mas grande] -> [NXSR o RCAS]; el ultimo pase escribe en dst
static void post(ctx_t* x, const nx_job_t* job, const nx_image_t* src, const nx_image_t* dst)
{
    nx_pipeline_t* p = x->p;
    enum { ST_RE, ST_SMAA, ST_EASU, ST_NXSR, ST_RCAS };
    int st[5], n = 0;
    if (job->re > 0.0f) st[n++] = ST_RE;   // realismo primero: el resto trabaja sobre la imagen final
    else p->re_valid = 0;                  // al volver a prenderlo no se mezcla con una malla vieja
    if (job->aa) st[n++] = ST_SMAA;
    if (job->post == NX_POST_EASU) st[n++] = ST_EASU;
    if (job->sr) {
        st[n++] = ST_NXSR;
        // nitidez tambien despues del reescalado (con EASU no queda otro buffer de salida)
        if (job->rcas && job->post != NX_POST_EASU) st[n++] = ST_RCAS;
    } else if (job->rcas || job->post == NX_POST_RCAS) {
        st[n++] = ST_RCAS;
    }
    if (n == 0) { x->err = 3; return; }
    nx_image_t taa = linear_img(p->tmp_aa, p->W, p->H);
    nx_image_t t2 = linear_img(p->tmp_img2, p->Wo, p->Ho);
    nx_image_t t1 = linear_img(p->tmp_img, p->W, p->H);
    const nx_image_t* cur = src;
    int big = 0;   // despues de EASU las etapas son de Wo x Ho
    for (int i = 0; i < n; ++i) {
        if (st[i] == ST_EASU) big = 1;
        // intermedio del tamano de la etapa, distinto del que se lee (y de la fuente)
        const nx_image_t* o = dst;
        if (i < n - 1) {
            const nx_image_t* cand[3];
            int nc = 0;
            if (big) { cand[nc++] = &t2; if (p->Wo == p->W && p->Ho == p->H) cand[nc++] = &taa; }
            else { cand[nc++] = &taa; if (p->Wo == p->W && p->Ho == p->H) cand[nc++] = &t2; cand[nc++] = &t1; }
            o = NULL;
            for (int k = 0; k < nc && !o; ++k)
                if (cand[k]->gpu_addr != cur->gpu_addr && cand[k]->gpu_addr != src->gpu_addr) o = cand[k];
            if (!o) { x->err = 8; return; }
        }
        int w = (int)o->width, h = (int)o->height;
        if (st[i] == ST_RE) {
            re(x, cur, o, w, h, job->re);
        } else if (st[i] == ST_SMAA) {
            smaa(x, cur, o);
        } else if (st[i] == ST_EASU) {
            ka_easu_t e = { tsharp(x, cur), p->H, p->W, tsharp(x, o), p->Ho, p->Wo };
            run(x, K_EASU, &e, sizeof e, p->Wo, p->Ho);
        } else if (st[i] == ST_NXSR && job->sr_lite) {
            srl(x, cur, o, w, h, job->sr_k, job->sr_lite == 2);
        } else if (st[i] == ST_NXSR) {
            ka_nxsr_t r = { tsharp(x, cur), w, h, job->sr_hq ? p->nxsr_hq_w : p->nxsr_w, tsharp(x, o), job->sr_k };
            run(x, job->sr_hq ? K_NXSR_HQ : K_NXSR, &r, sizeof r, w / 2, h / 2);
        } else {
            ka_rcas_t r = { tsharp(x, cur), tsharp(x, o), w, h, nx_exp2(-job->sharpness_stops), 0, job->sharp_detail };
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

// NXFG-2 (model/net2.py + model/bm2.py)
static void frame_generation(ctx_t* x, const nx_job_t* job, const nx_image_t* prev, const nx_image_t* cur,
                             const nx_image_t* warp_dst)
{
    nx_pipeline_t* p = x->p;
    int sc = job->slot & 1, sp = sc ^ 1;   // piramides por paridad (los slots consecutivos alternan)
    // la rapida no calcula los promedios RGB: al pasar a la IA hay que rehacer el anterior
    if (!job->fg_fast && p->luma_valid[sp] == 2) p->luma_valid[sp] = 0;

    // 1. piramides: la del fotograma previo se reutiliza
    if (!p->luma_valid[sp])
        luma_pyramid(x, prev, sp, job->fg_fast);
    luma_pyramid(x, cur, sc, job->fg_fast);
    p->luma_valid[sc] = job->fg_fast ? 2 : 1;

    // 2. block-matching simetrico de grueso (L4, 1/16) a fino (L1, media resolucion)
    for (int lv = 3; lv >= 0; --lv) {
        ka_bm_t b = { p->luma[sp][lv], p->luma[sc][lv], p->lum_h[lv], p->lum_w[lv],
                      lv == 3 ? 0 : p->vec[lv + 1], lv == 3 ? 0 : p->bm_h[lv + 1], lv == 3 ? 0 : p->bm_w[lv + 1],
                      p->vtmp, p->bm_h[lv], p->bm_w[lv], bm_radius[lv], lv == 0 };
        run(x, K_BM_SEARCH, &b, sizeof b, p->bm_w[lv] * 8, p->bm_h[lv] * 8);   // un grupo por bloque
        ka_median_t m = { p->vtmp, p->vec[lv], p->bm_h[lv], p->bm_w[lv] };
        run(x, K_BM_MEDIAN3, &m, sizeof m, p->bm_w[lv], p->bm_h[lv]);
    }
    ka_scene_t sn = { p->vec[0], p->h16 * p->w16, 0, p->scene };
    run(x, K_SCENE, &sn, sizeof sn, WG, WG);
    // pixeles quietos (HUD, textos) para que el warp no los copie a otro lado
    ka_smask_t sm = { p->luma[sp][0], p->luma[sc][0], p->lum_h[0], p->lum_w[0], p->smask_e };
    run(x, K_SMASK, &sm, sizeof sm, p->lum_w[0], p->lum_h[0]);
    ka_down2_t dl = { p->smask_e, p->lum_h[0], p->lum_w[0], p->smask };
    run(x, K_DILATE, &dl, sizeof dl, p->lum_w[0], p->lum_h[0]);

    uint32_t samp[4];
    nx_make_ssharp(samp, 1);
    ka_bm2flow_t f = { p->vec[0], p->h16, p->w16, p->flowG };
    run(x, K_BM_TO_FLOW, &f, sizeof f, p->w16, p->h16);
    if (job->fg_fast) {
        // generacion rapida: el flujo del block-matching a 1/16 con los 4 candidatos por pixel
        ka_warp_t w = { tsharp(x, prev), tsharp(x, cur), put_desc(x, samp, 4), p->flowG, p->h16, p->w16, 16.0f, 0,
                        p->scene, p->smask, tsharp(x, warp_dst), p->H, p->W };
        run(x, K_WARP_BLEND, &w, sizeof w, p->W, p->H);
        return;
    }
    // NXFG-3 (model/net3.py): flujo a 1/8, eleccion a 1/4 entre los 9 vectores vecinos
    // (promedios 4x4 de los dos reales) y un solo warp por pixel a resolucion completa
    ka_up5_t u = { p->flowG, p->h16, p->w16, p->flow8 };
    run(x, K_UPSAMPLE5, &u, sizeof u, p->w8, p->h8);
    ka_fg3_select_t sl = { p->a4[sp], p->a4[sc], p->h4, p->w4, p->flow8, p->h8, p->w8, p->fq };
    run(x, K_FG3_SELECT, &sl, sizeof sl, p->w4, p->h4);
    ka_fg3_warp_t w = { tsharp(x, prev), tsharp(x, cur), put_desc(x, samp, 4), p->fq, p->h4, p->w4, p->scene,
                        p->smask, tsharp(x, warp_dst), p->H, p->W };
    run(x, K_FG3_WARP, &w, sizeof w, p->W, p->H);
}

int nx_pipeline_build(nx_pipeline_t* p, pm4_t* c, const nx_job_t* job)
{
    ctx_t x = { p, c, p->kernarg_base[job->slot], p->kernarg_base[job->slot] + KERNARG_BYTES,
                p->desc_base[job->slot], p->desc_base[job->slot] + DESC_BYTES, 0 };
    if (job->osd_ctx) nx_osd_begin(job->osd_ctx, job->slot);

    // No se tocan COMPUTE_TMPRING_SIZE / STATIC_THREAD_MGMT / RESOURCE_LIMITS: son
    // estado del anillo y quedan puestos para lo que el juego envie despues (con
    // TMPRING_SIZE = 0, los shaders del juego que usan scratch colgaban la GPU: GTA V
    // aguanto la pantalla de carga y se colgo al empezar la partida). VCFG tampoco
    // los toca; nuestros kernels no usan scratch.

    int any_post = job->post != NX_POST_NONE || job->aa || job->sr || job->rcas || job->re > 0.0f;
    if (any_post && !job->out_real) return -15;
    if (job->fg) {
        if (job->enhance_first) {
            // FG sobre fotogramas ya procesados: el post-proceso corre una sola vez por
            // fotograma del juego y el intermedio hereda la calidad del real
            if (!job->prev_real_valid) {
                post(&x, job, job->prev, job->prev_real);
                p->luma_valid[(job->slot & 1) ^ 1] = 0;
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
    // comparar: mitad izquierda sin procesar y una linea divisoria clara en el medio
    if (job->compare && job->out_real && job->osd_ctx
        && (nx_osd_copy_left(job->osd_ctx, c, job->slot, job->cur, job->out_real) != 0
            || nx_osd_fill_rect(job->osd_ctx, c, job->slot, job->out_real, (int)job->out_real->width / 2 - 1, 0, 3,
                                (int)job->out_real->height, 0.95f, 0.95f, 0.95f) != 0))
        x.err = 5;
    if (job->fg) {
        // el intermedio sale de deformar fotogramas que pueden traer el menu del
        // anterior grabado: bajo cada panel va la imagen real limpia, asi el panel se
        // ve igual en el intermedio y en el real (si no, parpadeaba)
        const nx_image_t* clean = job->out_real ? job->out_real : job->cur;
        for (int i = 0; i < 2 && job->osd_ctx; ++i) {
            const nx_osd_t* d = job->osd[i];
            if (!d) continue;
            int pw, ph;
            nx_osd_panel_size(d->cols, d->rows, d->scale, &pw, &ph);
            if (nx_osd_copy_rect(job->osd_ctx, c, job->slot, clean, job->out_mid, d->x0, d->y0, pw, ph) != 0) x.err = 7;
        }
        draw_osd(&x, job, job->out_mid);
    }
    draw_osd(&x, job, job->out_real ? job->out_real : job->cur);
    // el real procesado vuelve al buffer del juego, que es el que se presenta: el
    // juego sigue viendo sus propios buffers en pantalla (como VCFG)
    if (job->copy_back && job->out_real && job->osd_ctx
        && nx_osd_copy(job->osd_ctx, c, job->slot, job->out_real, job->copy_back) != 0)
        x.err = 6;

    if (!job->no_end) {
        pm4_end_of_job(c, p->label_gpu, job->seq);
        pm4_nop_pad(c, 8);
    }
    return x.err ? -10 - x.err : (c->overflow ? -20 : 0);
}
