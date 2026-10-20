// Emulador de DCB para validar en el PC el camino completo de la GPU:
//   nx_pipeline_build() arma el DCB -> este programa lo interpreta (PM4), lee los
//   kernargs y descriptores T#/S# de la memoria "GPU" y ejecuta los MISMOS kernels
//   de shaders/*.cl compilados para x86 (-DNX_HOST).
// La salida se compara con model/reference_runtime.py (ver test_gpu_path.py).
//
//   gpu_emu W H frame0.rgba frame1.rgba out_mid.rgba [post: 0 none, 1 rcas, 2 smaa] [out_real.rgba]
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../source/nx_pipeline.h"
#include "../include/nx_shaders.h"

typedef float float4 __attribute__((ext_vector_type(4)));
typedef const uint32_t* tdesc_t;

// ------------------------------------------------------------- memoria
typedef struct { uint64_t gpu; uint8_t* host; size_t size; } region_t;
static region_t regions[16];
static int nregions;

static void map_region(uint64_t gpu, void* host, size_t size)
{
    regions[nregions++] = (region_t){ gpu, host, size };
}

static void* H(uint64_t gpu)
{
    if (!gpu) return NULL;
    for (int i = 0; i < nregions; ++i)
        if (gpu >= regions[i].gpu && gpu < regions[i].gpu + regions[i].size)
            return regions[i].host + (gpu - regions[i].gpu);
    fprintf(stderr, "direccion GPU sin mapear: %llx\n", (unsigned long long)gpu);
    exit(2);
}

// --------------------------------------------- servicios para los kernels
static int gx, gy;
int nx_host_gid_x(void) { return gx; }
int nx_host_gid_y(void) { return gy; }

typedef struct { uint8_t* base; int w, h, pitch; } img_t;

static img_t decode_t(tdesc_t t)
{
    uint64_t base = ((uint64_t)t[0] | (uint64_t)(t[1] & 0xFF) << 32) << 8;
    uint32_t dfmt = (t[1] >> 20) & 0x3F, tile = (t[3] >> 20) & 0x1F;
    if (dfmt != 10 || tile != NX_TILE_DISPLAY_LINEAR_ALIGNED) {
        fprintf(stderr, "el emulador solo soporta RGBA8 lineal (dfmt %u tile %u)\n", dfmt, tile);
        exit(3);
    }
    img_t i = { H(base), (int)(t[2] & 0x3FFF) + 1, (int)((t[2] >> 14) & 0x3FFF) + 1,
                (int)((t[4] >> 13) & 0x3FFF) + 1 };
    return i;
}

static float4 texel(img_t i, int x, int y)
{
    x = x < 0 ? 0 : (x >= i.w ? i.w - 1 : x);
    y = y < 0 ? 0 : (y >= i.h ? i.h - 1 : y);
    uint8_t* p = i.base + ((size_t)y * i.pitch + x) * 4;
    return (float4){ p[0], p[1], p[2], p[3] } * (1.0f / 255.0f);
}

float4 nx_load(tdesc_t t, int x, int y) { return texel(decode_t(t), x, y); }

float4 nx_sample(tdesc_t t, tdesc_t s, float u, float v)
{
    (void)s;  // el plugin solo usa un S#: bilineal, clamp al borde, coordenadas en texels
    img_t i = decode_t(t);
    float x = u - 0.5f, y = v - 0.5f;
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    float fx = x - x0, fy = y - y0;
    return (texel(i, x0, y0) * (1 - fx) + texel(i, x0 + 1, y0) * fx) * (1 - fy) +
           (texel(i, x0, y0 + 1) * (1 - fx) + texel(i, x0 + 1, y0 + 1) * fx) * fy;
}

void nx_store(tdesc_t t, int x, int y, float4 v)
{
    img_t i = decode_t(t);
    uint8_t* p = i.base + ((size_t)y * i.pitch + x) * 4;
    for (int c = 0; c < 4; ++c) {
        float f = v[c] < 0 ? 0 : (v[c] > 1 ? 1 : v[c]);
        p[c] = (uint8_t)(f * 255.0f + 0.5f);
    }
}

// --------------------------------------- kernels (compilados de shaders/*.cl)
typedef void* P;
#define CONV_DECL(n) void n(P, int, int, P, int, int, P, P, int, int, int, int, P);
CONV_DECL(conv3x3_14_32) CONV_DECL(conv3x3_17_32) CONV_DECL(conv3x3_32_32) CONV_DECL(conv3x3_32_5)
CONV_DECL(conv3x3_17_16) CONV_DECL(conv3x3_16_16) CONV_DECL(conv3x3_16_5)
void avgpool_rgb(P, int, int, P, int, int, int);
void resize_bilinear(P, int, int, int, P, int, int);
void upsample5(P, int, int, P);
void build_level_input(P, P, P, P, int, int, int, P);
void bm_to_flow(P, int, int, P);
void luma_full(P, int, int, P, int, int);
void luma_down2(P, int, int, P);
void bm_search(P, P, int, int, P, int, int, P, int, int, int);
void bm_median3(P, P, int, int);
void warp_blend(P, P, P, P, int, int, P, int, int);
void easu(P, int, int, P, int, int);
void rcas(P, P, int, int, float, int);
void smaa_edges(P, int, int, P);
void smaa_weights(P, int, int, P, P, P);
void smaa_blend(P, P, P, int, int, P);

static const uint8_t* ka;
static const uint8_t* offs;
static P ap(int i) { uint64_t v; memcpy(&v, ka + offs[i], 8); return H(v); }
static int ai(int i) { int v; memcpy(&v, ka + offs[i], 4); return v; }
static float af(int i) { float v; memcpy(&v, ka + offs[i], 4); return v; }

static void call(const char* n)
{
#define IS(s) (!strcmp(n, s))
#define CONV(f) f(ap(0), ai(1), ai(2), ap(3), ai(4), ai(5), ap(6), ap(7), ai(8), ai(9), ai(10), ai(11), ap(12))
    if (IS("conv3x3_14_32")) CONV(conv3x3_14_32);
    else if (IS("conv3x3_17_32")) CONV(conv3x3_17_32);
    else if (IS("conv3x3_32_32")) CONV(conv3x3_32_32);
    else if (IS("conv3x3_32_5")) CONV(conv3x3_32_5);
    else if (IS("conv3x3_17_16")) CONV(conv3x3_17_16);
    else if (IS("conv3x3_16_16")) CONV(conv3x3_16_16);
    else if (IS("conv3x3_16_5")) CONV(conv3x3_16_5);
    else if (IS("avgpool_rgb")) avgpool_rgb(ap(0), ai(1), ai(2), ap(3), ai(4), ai(5), ai(6));
    else if (IS("resize_bilinear")) resize_bilinear(ap(0), ai(1), ai(2), ai(3), ap(4), ai(5), ai(6));
    else if (IS("upsample5")) upsample5(ap(0), ai(1), ai(2), ap(3));
    else if (IS("build_level_input")) build_level_input(ap(0), ap(1), ap(2), ap(3), ai(4), ai(5), ai(6), ap(7));
    else if (IS("bm_to_flow")) bm_to_flow(ap(0), ai(1), ai(2), ap(3));
    else if (IS("luma_full")) luma_full(ap(0), ai(1), ai(2), ap(3), ai(4), ai(5));
    else if (IS("luma_down2")) luma_down2(ap(0), ai(1), ai(2), ap(3));
    else if (IS("bm_search"))
        bm_search(ap(0), ap(1), ai(2), ai(3), ap(4), ai(5), ai(6), ap(7), ai(8), ai(9), ai(10));
    else if (IS("bm_median3")) bm_median3(ap(0), ap(1), ai(2), ai(3));
    else if (IS("warp_blend")) warp_blend(ap(0), ap(1), ap(2), ap(3), ai(4), ai(5), ap(6), ai(7), ai(8));
    else if (IS("easu")) easu(ap(0), ai(1), ai(2), ap(3), ai(4), ai(5));
    else if (IS("rcas")) rcas(ap(0), ap(1), ai(2), ai(3), af(4), ai(5));
    else if (IS("smaa_edges")) smaa_edges(ap(0), ai(1), ai(2), ap(3));
    else if (IS("smaa_weights")) smaa_weights(ap(0), ai(1), ai(2), ap(3), ap(4), ap(5));
    else if (IS("smaa_blend")) smaa_blend(ap(0), ap(1), ap(2), ai(3), ai(4), ap(5));
    else { fprintf(stderr, "kernel desconocido %s\n", n); exit(4); }
}

// --------------------------------------------------------- interprete PM4
static int run_dcb(const nx_pipeline_t* p, const uint32_t* b, size_t n)
{
    uint32_t sh[0x400] = { 0 };
    int dispatches = 0;
    for (size_t i = 0; i < n;) {
        uint32_t h = b[i];
        if (h == 0x80000000u) { ++i; continue; }
        uint32_t op = (h >> 8) & 0xFF, cnt = ((h >> 16) & 0x3FFF) + 1;
        const uint32_t* body = b + i + 1;
        if (op == PM4_SET_SH_REG) {
            for (uint32_t k = 1; k < cnt; ++k) sh[body[0] + k - 1] = body[k];
        } else if (op == PM4_DISPATCH_DIRECT) {
#define SH(r) sh[((r) - SH_REG_BASE) >> 2]
            uint64_t pgm = ((uint64_t)SH(R_COMPUTE_PGM_LO) << 8) | ((uint64_t)SH(R_COMPUTE_PGM_LO + 4) << 40);
            const nx_shader_t* s = NULL;
            for (int k = 0; k < 32; ++k)
                if (p->shader_gpu[k] == pgm) s = &nx_shaders[p->shader[k]];
            if (!s) { fprintf(stderr, "PGM %llx no corresponde a ningun kernel\n", (unsigned long long)pgm); return -1; }
            uint64_t kptr = SH(R_COMPUTE_USER_DATA_0 + 16) | (uint64_t)SH(R_COMPUTE_USER_DATA_0 + 20) << 32;
            ka = H(kptr);
            offs = s->arg_offsets;
            uint32_t tx = SH(R_COMPUTE_NUM_THREAD_X), ty = SH(R_COMPUTE_NUM_THREAD_X + 4);
            for (gy = 0; gy < (int)(body[1] * ty); ++gy)
                for (gx = 0; gx < (int)(body[0] * tx); ++gx)
                    call(s->name);
            ++dispatches;
        } else if (op == PM4_EVENT_WRITE_EOP) {
            uint64_t addr = body[1] | (uint64_t)(body[2] & 0xFFFF) << 32;
            *(uint32_t*)H(addr) = body[3];
        }
        i += 1 + cnt;
    }
    return dispatches;
}

static uint8_t* load_rgba(const char* path, size_t bytes)
{
    uint8_t* b = malloc(bytes);
    FILE* f = fopen(path, "rb");
    if (!f || fread(b, 1, bytes, f) != bytes) { fprintf(stderr, "no pude leer %s\n", path); exit(1); }
    fclose(f);
    return b;
}

static void save_rgba(const char* path, const uint8_t* img, int W, int Hh, int pitch)
{
    FILE* f = fopen(path, "wb");
    for (int y = 0; y < Hh; ++y) fwrite(img + (size_t)y * pitch * 4, 4, W, f);
    fclose(f);
}

int main(int argc, char** argv)
{
    if (argc < 6) { fprintf(stderr, "uso: %s W H f0.rgba f1.rgba out_mid.rgba [post] [out_real.rgba]\n", argv[0]); return 1; }
    int W = atoi(argv[1]), Hh = atoi(argv[2]), post = argc > 6 ? atoi(argv[6]) : 0;
    int pitch = (W + 63) / 64 * 64;
    size_t img_bytes = (size_t)pitch * Hh * 4;
    // los fotogramas del juego, con pitch de 64 px como los buffers de VideoOut
    uint8_t* frames = calloc(4, img_bytes);
    for (int k = 0; k < 2; ++k) {
        uint8_t* src = load_rgba(argv[3 + k], (size_t)W * Hh * 4);
        for (int y = 0; y < Hh; ++y)
            memcpy(frames + k * img_bytes + (size_t)y * pitch * 4, src + (size_t)y * W * 4, (size_t)W * 4);
        free(src);
    }
    uint64_t fgpu = 0x0000005000000000ull;
    map_region(fgpu, frames, 4 * img_bytes);

    size_t sz = nx_pipeline_mem_size(W, Hh, W, Hh);
    nx_arena_t a = { calloc(1, sz), 0x0000004000000000ull, sz, 0 };
    map_region(a.gpu, a.cpu, sz);
    static nx_pipeline_t p;
    if (nx_pipeline_init(&p, W, Hh, W, Hh, a) != 0) { fprintf(stderr, "init fallo\n"); return 1; }

    nx_image_t img[4];
    for (int k = 0; k < 4; ++k)
        img[k] = (nx_image_t){ fgpu + k * img_bytes, (uint32_t)W, (uint32_t)Hh, (uint32_t)pitch,
                               NX_FMT_RGBA8_UNORM, NX_TILE_DISPLAY_LINEAR_ALIGNED };
    static uint32_t dcb[64 * 1024];
    pm4_t c;
    pm4_init(&c, dcb, sizeof dcb / 4);
    nx_job_t job = { .prev = &img[0], .cur = &img[1], .out_mid = &img[2], .out_real = post ? &img[3] : NULL,
                     .slot = 1, .post = post == 1 ? NX_POST_RCAS : NX_POST_NONE, .aa = post == 2,
                     .sharpness_stops = 0.25f, .seq = 7 };
    if (nx_pipeline_build(&p, &c, &job) != 0) { fprintf(stderr, "build fallo\n"); return 1; }
    int d = run_dcb(&p, dcb, c.n);
    if (d < 0) return 1;
    printf("dispatches ejecutados: %d, label = %u\n", d, *p.label_cpu);
    save_rgba(argv[5], frames + 2 * img_bytes, W, Hh, pitch);
    if (post && argc > 7) save_rgba(argv[7], frames + 3 * img_bytes, W, Hh, pitch);
    return *p.label_cpu == 7 ? 0 : 1;
}
