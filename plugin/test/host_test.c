// Test en PC (Linux/macOS) de la parte portable del plugin:
//   * los structs de kernargs coinciden con los offsets que emitio el compilador;
//   * el DCB de un fotograma 1080p se arma sin errores y es PM4 bien formado;
//   * resumen de dispatches y memoria.
//
//   make -C plugin/test   (o ver el Makefile de esta carpeta)
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../source/nx_pipeline.h"
#include "../include/nx_shaders.h"

// --- copia de los layouts de nx_pipeline.c (si se tocan alli, tocar aca)
typedef struct { uint64_t in; int32_t H, W; uint64_t out; int32_t Ho, Wo; uint64_t w, b;
                 int32_t stride, pad_top, pad_left, act; uint64_t residual; } ka_conv_t;
typedef struct { uint64_t src; int32_t H, W; uint64_t out; int32_t Ho, Wo, f; } ka_avgpool_t;
typedef struct { uint64_t in; int32_t H, W; uint64_t out; } ka_up5_t;
typedef struct { uint64_t in; int32_t H, W, C, _pad; uint64_t out; int32_t Ho, Wo; } ka_resize_t;
typedef struct { uint64_t a0, a1, flow5, delta5; int32_t H, W, level, _pad; uint64_t out; } ka_build_t;
typedef struct { uint64_t vec; int32_t H, W; uint64_t flow5; } ka_bm2flow_t;
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
typedef struct { uint64_t dst; int32_t W, H; uint64_t text; int32_t cols, rows; uint64_t font;
                 int32_t x0, y0, scale, sel_row; } ka_osd_t;
typedef struct { uint64_t src, dst; int32_t x0, y0, w, h; } ka_copy_t;

#define O(t, f) (int)offsetof(t, f)
typedef struct { const char* name; int n; int offs[16]; } layout_t;
static const layout_t layouts[] = {
    { "conv3x3_14_32", 13, { O(ka_conv_t, in), O(ka_conv_t, H), O(ka_conv_t, W), O(ka_conv_t, out), O(ka_conv_t, Ho),
        O(ka_conv_t, Wo), O(ka_conv_t, w), O(ka_conv_t, b), O(ka_conv_t, stride), O(ka_conv_t, pad_top),
        O(ka_conv_t, pad_left), O(ka_conv_t, act), O(ka_conv_t, residual) } },
    { "avgpool_rgb", 7, { O(ka_avgpool_t, src), O(ka_avgpool_t, H), O(ka_avgpool_t, W), O(ka_avgpool_t, out),
        O(ka_avgpool_t, Ho), O(ka_avgpool_t, Wo), O(ka_avgpool_t, f) } },
    { "upsample5", 4, { O(ka_up5_t, in), O(ka_up5_t, H), O(ka_up5_t, W), O(ka_up5_t, out) } },
    { "resize_bilinear", 7, { O(ka_resize_t, in), O(ka_resize_t, H), O(ka_resize_t, W), O(ka_resize_t, C),
        O(ka_resize_t, out), O(ka_resize_t, Ho), O(ka_resize_t, Wo) } },
    { "build_level_input", 8, { O(ka_build_t, a0), O(ka_build_t, a1), O(ka_build_t, flow5), O(ka_build_t, delta5),
        O(ka_build_t, H), O(ka_build_t, W), O(ka_build_t, level), O(ka_build_t, out) } },
    { "bm_to_flow", 4, { O(ka_bm2flow_t, vec), O(ka_bm2flow_t, H), O(ka_bm2flow_t, W), O(ka_bm2flow_t, flow5) } },
    { "luma_full", 6, { O(ka_luma_t, src), O(ka_luma_t, Hs), O(ka_luma_t, Ws), O(ka_luma_t, out),
        O(ka_luma_t, H), O(ka_luma_t, W) } },
    { "luma_down2", 4, { O(ka_down2_t, in), O(ka_down2_t, H), O(ka_down2_t, W), O(ka_down2_t, out) } },
    { "bm_search", 11, { O(ka_bm_t, l0), O(ka_bm_t, l1), O(ka_bm_t, H), O(ka_bm_t, W), O(ka_bm_t, coarse),
        O(ka_bm_t, cBy), O(ka_bm_t, cBx), O(ka_bm_t, vec), O(ka_bm_t, By), O(ka_bm_t, Bx), O(ka_bm_t, radius) } },
    { "bm_median3", 4, { O(ka_median_t, in), O(ka_median_t, out), O(ka_median_t, By), O(ka_median_t, Bx) } },
    { "warp_blend", 9, { O(ka_warp_t, f0), O(ka_warp_t, f1), O(ka_warp_t, samp), O(ka_warp_t, low5),
        O(ka_warp_t, h4), O(ka_warp_t, w4), O(ka_warp_t, dst), O(ka_warp_t, H), O(ka_warp_t, W) } },
    { "easu", 6, { O(ka_easu_t, src), O(ka_easu_t, H), O(ka_easu_t, W), O(ka_easu_t, dst),
        O(ka_easu_t, Ho), O(ka_easu_t, Wo) } },
    { "rcas", 6, { O(ka_rcas_t, src), O(ka_rcas_t, dst), O(ka_rcas_t, W), O(ka_rcas_t, H),
        O(ka_rcas_t, sharp), O(ka_rcas_t, denoise) } },
    { "smaa_edges", 4, { O(ka_smaa_edges_t, src), O(ka_smaa_edges_t, W), O(ka_smaa_edges_t, H),
        O(ka_smaa_edges_t, edges) } },
    { "smaa_weights", 6, { O(ka_smaa_weights_t, edges), O(ka_smaa_weights_t, W), O(ka_smaa_weights_t, H),
        O(ka_smaa_weights_t, area), O(ka_smaa_weights_t, search), O(ka_smaa_weights_t, weights) } },
    { "smaa_blend", 6, { O(ka_smaa_blend_t, src), O(ka_smaa_blend_t, samp), O(ka_smaa_blend_t, weights),
        O(ka_smaa_blend_t, W), O(ka_smaa_blend_t, H), O(ka_smaa_blend_t, dst) } },
    { "wg_nxsr", 6, { O(ka_nxsr_t, src), O(ka_nxsr_t, W), O(ka_nxsr_t, H), O(ka_nxsr_t, w), O(ka_nxsr_t, dst),
        O(ka_nxsr_t, k) } },
    { "wg_nxsr_hq", 6, { O(ka_nxsr_t, src), O(ka_nxsr_t, W), O(ka_nxsr_t, H), O(ka_nxsr_t, w), O(ka_nxsr_t, dst),
        O(ka_nxsr_t, k) } },
    { "osd_draw", 11, { O(ka_osd_t, dst), O(ka_osd_t, W), O(ka_osd_t, H), O(ka_osd_t, text), O(ka_osd_t, cols),
        O(ka_osd_t, rows), O(ka_osd_t, font), O(ka_osd_t, x0), O(ka_osd_t, y0), O(ka_osd_t, scale),
        O(ka_osd_t, sel_row) } },
    { "copy_rect", 6, { O(ka_copy_t, src), O(ka_copy_t, dst), O(ka_copy_t, x0), O(ka_copy_t, y0), O(ka_copy_t, w),
        O(ka_copy_t, h) } },
};

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FALLO: " __VA_ARGS__); printf("\n"); ++fails; } } while (0)

static const nx_shader_t* find(const char* n)
{
    for (int i = 0; i < NX_NUM_SHADERS; ++i)
        if (!strcmp(nx_shaders[i].name, n)) return &nx_shaders[i];
    return NULL;
}

static void check_layouts(void)
{
    for (size_t i = 0; i < sizeof layouts / sizeof layouts[0]; ++i) {
        const layout_t* l = &layouts[i];
        const nx_shader_t* s = find(l->name);
        CHECK(s, "kernel %s no esta en nx_shaders.h", l->name);
        if (!s) continue;
        CHECK(s->num_args == l->n, "%s: %d args en el .cl, %d en el struct", l->name, s->num_args, l->n);
        for (int a = 0; a < l->n && a < s->num_args; ++a)
            CHECK(s->arg_offsets[a] == l->offs[a], "%s arg %d: offset %d en el .cl, %d en el struct",
                  l->name, a, s->arg_offsets[a], l->offs[a]);
        CHECK(s->user_sgprs == 6, "%s: user_sgprs=%d", l->name, s->user_sgprs);
    }
    // las 4 variantes de conv comparten layout
    const char* convs[] = { "conv3x3_17_32", "conv3x3_32_32", "conv3x3_32_5",
                            "conv3x3_17_16", "conv3x3_16_16", "conv3x3_16_5" };
    for (int i = 0; i < 6; ++i) {
        const nx_shader_t* s = find(convs[i]);
        CHECK(s && s->num_args == 13, "%s", convs[i]);
        for (int a = 0; s && a < 13; ++a)
            CHECK(s->arg_offsets[a] == layouts[0].offs[a], "%s arg %d", convs[i], a);
    }
}

// recorre el DCB: todos los paquetes deben ser tipo 3 (o tipo 2 de relleno) y terminar justo al final
static void walk_pm4(const uint32_t* b, size_t n, int* dispatches, int* sh_writes)
{
    size_t i = 0;
    *dispatches = *sh_writes = 0;
    while (i < n) {
        uint32_t h = b[i];
        if (h == 0x80000000u) { ++i; continue; }
        CHECK((h >> 30) == 3, "paquete no tipo 3 en dword %zu: %08x", i, h);
        if ((h >> 30) != 3) return;
        uint32_t op = (h >> 8) & 0xFF, cnt = ((h >> 16) & 0x3FFF) + 1;
        if (op == PM4_DISPATCH_DIRECT) {
            ++*dispatches;
            CHECK(b[i + 1] > 0 && b[i + 2] > 0 && b[i + 3] == 1, "dispatch con grilla invalida en %zu", i);
        }
        if (op == PM4_SET_SH_REG) {
            ++*sh_writes;
            uint32_t reg = SH_REG_BASE + b[i + 1] * 4;
            CHECK(reg >= 0xB800 && reg < 0xBA00, "SET_SH_REG fuera del rango de compute: %x", reg);
        }
        i += 1 + cnt;
    }
    CHECK(i == n, "el DCB no termina en un limite de paquete (%zu vs %zu)", i, n);
}

// perfil: hilos lanzados por kernel en un DCB (PROFILE=1 en el entorno)
static void profile(const nx_pipeline_t* p, const uint32_t* b, size_t n)
{
    uint64_t pgm = 0, thr[64] = { 0 };
    int cnt[64] = { 0 };
    for (size_t i = 0; i < n;) {
        uint32_t h = b[i];
        if (h == 0x80000000u) { ++i; continue; }
        uint32_t op = (h >> 8) & 0xFF, c = ((h >> 16) & 0x3FFF) + 1;
        if (op == PM4_SET_SH_REG && SH_REG_BASE + b[i + 1] * 4 == R_COMPUTE_PGM_LO) pgm = (uint64_t)b[i + 2] << 8;
        if (op == PM4_DISPATCH_DIRECT)
            for (int k = 0; k < 32; ++k)
                if (p->shader_gpu[k] == pgm) { thr[k] += (uint64_t)b[i + 1] * b[i + 2] * 64; cnt[k]++; }
        i += 1 + c;
    }
    for (int k = 0; k < 32; ++k)
        if (cnt[k]) printf("    %-20s x%-3d %10llu hilos\n", nx_shaders[p->shader[k]].name, cnt[k], (unsigned long long)thr[k]);
}

static void run_case(int W, int H, int Wo, int Ho, nx_post_t post, int aa, int sr, const char* label)
{
    size_t sz = nx_pipeline_mem_size(W, H, Wo, Ho);
    nx_arena_t a = { calloc(1, sz), 0x0000004000000000ull, sz, 0 };
    nx_pipeline_t* p = calloc(1, sizeof *p);
    int r = nx_pipeline_init(p, W, H, Wo, Ho, a);
    CHECK(r == 0, "%s: init devolvio %d", label, r);
    if (r) return;
    nx_image_t f0 = { 0x0000005000000000ull, W, H, (W + 63) / 64 * 64, NX_FMT_RGBA8_SRGB, NX_TILE_DISPLAY_2D_THIN };
    nx_image_t f1 = f0, mid = f0, real = f0, real2 = f0;
    f1.gpu_addr += 0x1000000; mid.gpu_addr += 0x2000000; real.gpu_addr += 0x3000000;
    real2.gpu_addr += 0x4000000;
    mid.width = real.width = real2.width = Wo; mid.height = real.height = real2.height = Ho;
    mid.pitch = real.pitch = real2.pitch = (Wo + 63) / 64 * 64;
    static uint32_t dcb[64 * 1024];
    int total_disp = 0;
    for (int frame = 0; frame < 3; ++frame) {
        pm4_t c;
        pm4_init(&c, dcb, sizeof dcb / 4);
        nx_job_t job = { .prev = frame & 1 ? &f1 : &f0, .cur = frame & 1 ? &f0 : &f1, .out_mid = &mid,
                         .out_real = post == NX_POST_NONE && !aa && !sr ? NULL : &real, .slot = frame & 1,
                         .post = post, .aa = aa, .sharpness_stops = 0.25f, .seq = 100 + frame,
                         .fg = 1, .sr = sr, .sr_k = 0.75f, .enhance_first = sr && W == Wo,
                         .prev_real = &real2, .prev_real_valid = frame > 0 };
        r = nx_pipeline_build(p, &c, &job);
        CHECK(r == 0, "%s: build devolvio %d", label, r);
        int d, s;
        walk_pm4(dcb, c.n, &d, &s);
        if (frame == 0) total_disp = d;
        if (frame == 2 && getenv("PROFILE")) profile(p, dcb, c.n);
        if (frame == 2)
            printf("%-28s arena %6.1f MiB, DCB %5zu dwords, dispatches %d (1er frame %d)\n",
                   label, p->arena.used / 1048576.0, c.n, d, total_disp);
    }
    free(a.cpu);
    free(p);
}

int main(void)
{
    check_layouts();
    run_case(1920, 1080, 1920, 1080, NX_POST_NONE, 0, 0, "1080p FG");
    run_case(1920, 1080, 1920, 1080, NX_POST_RCAS, 0, 0, "1080p FG + RCAS");
    run_case(1920, 1080, 1920, 1080, NX_POST_NONE, 1, 0, "1080p FG + SMAA");
    run_case(1920, 1080, 1920, 1080, NX_POST_NONE, 1, 1, "1080p FG + SMAA + NXSR");
    run_case(1600, 900, 1920, 1080, NX_POST_EASU, 1, 1, "900p->1080p FG+SMAA+EASU+NXSR");
    run_case(1280, 720, 1920, 1080, NX_POST_EASU, 1, 0, "720p->1080p FG+SMAA+FSR1");
    printf(fails ? "\n%d FALLOS\n" : "\nOK: todo en orden\n", fails);
    return fails != 0;
}
