// VCFG-NX: generador de fotogramas (30 -> 60) + FSR1 para PS4 como plugin de GoldHEN.
//
// Rediseno del VCFG v2.14 de la comunidad: mismo punto de enganche (el flip
// del juego), pero el movimiento lo estima un modelo hibrido entrenado
// (block-matching + red NXFlow) y el post-proceso es FSR1 (EASU + RCAS) tanto
// en fotogramas reales como generados.
//
// ESTADO: pasa un chequeo de sintaxis contra los headers de OpenOrbis y del
// GoldHEN SDK, pero todavia NO se compilo con el toolchain completo ni se probo en
// una consola. Los puntos a validar en hardware estan marcados con "VALIDAR".
#include <Common.h>
#include <orbis/GnmDriver.h>
#include <orbis/Pad.h>
#include <orbis/VideoOut.h>
#include <orbis/libkernel.h>
#include <string.h>

#include "config.h"
#include "nx_pipeline.h"
#include "nx_sys.h"

attr_public const char* g_pluginName = "vcfg_nx";
attr_public const char* g_pluginDesc = "Frame generation (NXFlow) + SMAA + FSR1 para juegos a 30 fps";
attr_public const char* g_pluginAuth = "VCFG-NX";
attr_public u32 g_pluginVersion = 0x00000100; // 1.00

#define MAX_MID 2
#define DCB_DWORDS 4096
#define PAD_L3 0x0002
#define PAD_R3 0x0004

HOOK_INIT(sceVideoOutRegisterBuffers);
HOOK_INIT(sceVideoOutSetFlipRate);
HOOK_INIT(sceGnmSubmitAndFlipCommandBuffers);
HOOK_INIT(scePadReadState);

static struct {
    nx_config_t cfg;
    char title[16];
    int ready;            // memoria y buffers extra listos
    int active;           // FG activo ahora (toggle + auto-desactivacion)
    int handle;
    int game_first, game_num;
    nx_image_t game_img[16];
    int mid_index[MAX_MID], real_index[MAX_MID];
    nx_image_t mid_img[MAX_MID], real_img[MAX_MID];
    int prev_idx;         // indice del buffer del juego del fotograma anterior
    int slot;
    uint32_t seq, slot_seq[2];
    uint32_t* dcb[2];
    uint32_t* nop_dcb;
    nx_pipeline_t pipe;
    uint64_t last_flip_us;
    float game_fps;
    uint64_t combo_since_us;
} g;

// ---------------------------------------------------------------- memoria
static void* gpu_alloc(size_t len, size_t align)
{
    off_t phys = 0;
    void* va = NULL;
    len = (len + 0xFFFF) & ~(size_t)0xFFFF;
    if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), len, align,
                                      ORBIS_KERNEL_WC_GARLIC, &phys) != 0)
        return NULL;
    if (sceKernelMapDirectMemory(&va, len, ORBIS_KERNEL_PROT_CPU_RW | ORBIS_KERNEL_PROT_GPU_RW,
                                 0, phys, align) != 0)
        return NULL;
    return va;   // en PS4 la direccion virtual de CPU es la misma que ve la GPU
}

static nx_format_t map_format(int32_t fmt, int* ok)
{
    *ok = 1;
    switch ((uint32_t)fmt) {
    case 0x80000000u: /* A8R8G8B8_SRGB */
    case 0x80002200u: /* A8B8G8R8_SRGB */
        // se leen y escriben como UNORM a proposito: el modelo trabaja sobre los
        // valores codificados en sRGB (como los videos de entrenamiento); con
        // NFMT_SRGB el hardware los linealizaria al leer
        return NX_FMT_RGBA8_UNORM;
    case 0x88060000u: /* A2R10G10B10 */
        return NX_FMT_RGB10A2_UNORM;
    default:
        *ok = 0;  // HDR PQ u otros: el modelo se entreno en SDR
        return NX_FMT_RGBA8_UNORM;
    }
}

static uint64_t now_us(void) { return sceKernelGetProcessTime(); }

// ----------------------------------------------------------------- hooks
s32 sceVideoOutRegisterBuffers_hook(s32 handle, s32 start, void* const* addrs, s32 num,
                                    const OrbisVideoOutBufferAttribute* attr)
{
    s32 r = HOOK_CONTINUE(sceVideoOutRegisterBuffers,
                          s32 (*)(s32, s32, void* const*, s32, const OrbisVideoOutBufferAttribute*),
                          handle, start, addrs, num, attr);
    if (r < 0 || g.ready || num <= 0 || num > 8)
        return r;
    int ok;
    nx_format_t fmt = map_format(attr->format, &ok);
    if (!ok) {
        final_printf("formato 0x%08x no soportado: VCFG-NX queda inactivo\n", attr->format);
        return r;
    }
    int W = (int)attr->width, H = (int)attr->height;
    // VALIDAR: indice de tile mode de los backbuffers "TILE" (ver gcn_desc.h)
    uint32_t tile = attr->tmode == ORBIS_VIDEO_OUT_TILING_MODE_LINEAR ? NX_TILE_DISPLAY_LINEAR_ALIGNED
                                                                      : NX_TILE_DISPLAY_2D_THIN;
    g.handle = handle;
    g.game_first = start;
    g.game_num = num;
    for (int i = 0; i < num; ++i) {
        nx_image_t im = { (uint64_t)addrs[i], (uint32_t)W, (uint32_t)H, attr->pixelPitch, fmt, tile };
        g.game_img[i] = im;
    }

    int post = g.cfg.post;
    int Wo = W, Ho = H;
    if (post == NX_POST_EASU && (g.cfg.out_w > W || g.cfg.out_h > H)) {
        Wo = g.cfg.out_w;
        Ho = g.cfg.out_h;
    } else if (post == NX_POST_EASU) {
        post = NX_POST_RCAS;   // el juego ya entrega la resolucion de salida
    }
    g.cfg.post = post;

    size_t need = nx_pipeline_mem_size(W, H, Wo, Ho);
    void* arena = gpu_alloc(need, 64 * 1024);
    // buffers de salida propios: mismos atributos que el juego (salvo tamano con EASU)
    size_t pitch_o = (size_t)((Wo + 63) / 64 * 64);
    size_t buf_bytes = (pitch_o * Ho * 4 + 0xFFFF) & ~(size_t)0xFFFF;
    int own_real = post != NX_POST_NONE || g.cfg.aa;   // el real tambien pasa por SMAA/FSR1
    int nbuf = own_real ? 2 * MAX_MID : MAX_MID;
    uint8_t* outs = gpu_alloc(buf_bytes * nbuf, 64 * 1024);
    uint32_t* dcbs = gpu_alloc((2 * DCB_DWORDS + 64) * 4, 4096);
    if (!arena || !outs || !dcbs) {
        final_printf("sin memoria directa (%zu MiB): VCFG-NX inactivo esta sesion\n", (need + buf_bytes * nbuf) >> 20);
        return r;
    }
    nx_arena_t a = { arena, (uint64_t)arena, need, 0 };
    if (nx_pipeline_init(&g.pipe, W, H, Wo, Ho, a) != 0) {
        final_printf("nx_pipeline_init fallo\n");
        return r;
    }

    // registrar los buffers extra a continuacion de los del juego
    OrbisVideoOutBufferAttribute oa = *attr;
    oa.width = Wo; oa.height = Ho; oa.pixelPitch = (uint32_t)pitch_o;
    oa.tmode = ORBIS_VIDEO_OUT_TILING_MODE_LINEAR;  // nuestros kernels escriben lineal
    void* extra[2 * MAX_MID];
    for (int i = 0; i < nbuf; ++i) extra[i] = outs + buf_bytes * i;
    int first = start + num;
    s32 rr = HOOK_CONTINUE(sceVideoOutRegisterBuffers,
                           s32 (*)(s32, s32, void* const*, s32, const OrbisVideoOutBufferAttribute*),
                           handle, first, extra, nbuf, &oa);
    if (rr < 0) {
        final_printf("RegisterBuffers de buffers extra fallo 0x%08x\n", rr);
        return r;
    }
    for (int i = 0; i < MAX_MID; ++i) {
        nx_image_t im = { (uint64_t)extra[i], (uint32_t)Wo, (uint32_t)Ho, (uint32_t)pitch_o, fmt,
                          NX_TILE_DISPLAY_LINEAR_ALIGNED };
        g.mid_index[i] = first + i;
        g.mid_img[i] = im;
        if (own_real) {
            im.gpu_addr = (uint64_t)extra[MAX_MID + i];
            g.real_index[i] = first + MAX_MID + i;
            g.real_img[i] = im;
        }
    }
    g.dcb[0] = dcbs;
    g.dcb[1] = dcbs + DCB_DWORDS;
    g.nop_dcb = dcbs + 2 * DCB_DWORDS;
    for (int i = 0; i < 64; ++i) g.nop_dcb[i] = 0x80000000u;  // relleno tipo 2
    g.prev_idx = -1;
    g.ready = 1;
    g.active = g.cfg.framegen;
    final_printf("listo: %dx%d -> %dx%d post=%d aa=%d, %zu MiB\n", W, H, Wo, Ho, post, g.cfg.aa,
                 (need + buf_bytes * nbuf) >> 20);
    Notify(TEX_ICON_SYSTEM, "VCFG-NX listo. L3+R3 para activar/desactivar.");
    return r;
}

s32 sceVideoOutSetFlipRate_hook(s32 handle, s32 rate)
{
    // con FG activo la salida es siempre 60 Hz; el juego sigue a su ritmo
    if (g.ready && g.active)
        rate = 0;
    return HOOK_CONTINUE(sceVideoOutSetFlipRate, s32 (*)(s32, s32), handle, rate);
}

typedef s32 (*submit_flip_t)(u32, void**, u32*, void**, u32*, u32, u32, u32, u64);

static s32 flip_only(u32 handle, int index, u32 mode, u64 arg)
{
    void* d[1] = { g.nop_dcb };
    u32 sz[1] = { 64 * 4 };
    return HOOK_CONTINUE(sceGnmSubmitAndFlipCommandBuffers, submit_flip_t, 1, d, sz, NULL, NULL,
                         handle, (u32)index, mode, arg);
}

s32 sceGnmSubmitAndFlipCommandBuffers_hook(u32 count, void** dcb, u32* dcb_sz, void** ccb, u32* ccb_sz,
                                           u32 handle, u32 index, u32 mode, u64 arg)
{
    uint64_t t = now_us();
    if (g.last_flip_us) {
        float fps = 1e6f / (float)(t - g.last_flip_us + 1);
        g.game_fps = g.game_fps * 0.95f + fps * 0.05f;
    }
    g.last_flip_us = t;

    int gi = (int)index - g.game_first;
    int usable = g.ready && g.active && (int)handle == g.handle && gi >= 0 && gi < g.game_num
                 && g.game_fps < (float)g.cfg.max_game_fps;
    if (!usable || g.prev_idx < 0) {
        g.prev_idx = usable ? gi : -1;
        g.pipe.luma_valid[0] = g.pipe.luma_valid[1] = 0;
        return HOOK_CONTINUE(sceGnmSubmitAndFlipCommandBuffers, submit_flip_t,
                             count, dcb, dcb_sz, ccb, ccb_sz, handle, index, mode, arg);
    }

    // el slot anterior tiene que haber terminado en la GPU antes de reusar sus kernargs
    int s = g.slot;
    for (int spin = 0; *g.pipe.label_cpu < g.slot_seq[s] && spin < 2000; ++spin)
        sceKernelUsleep(10);
    if (*g.pipe.label_cpu < g.slot_seq[s]) {   // GPU atrasada: este fotograma va nativo
        g.prev_idx = gi;
        g.pipe.luma_valid[0] = g.pipe.luma_valid[1] = 0;
        return HOOK_CONTINUE(sceGnmSubmitAndFlipCommandBuffers, submit_flip_t,
                             count, dcb, dcb_sz, ccb, ccb_sz, handle, index, mode, arg);
    }

    // 1. trabajo del juego, sin flip
    s32 r = sceGnmSubmitCommandBuffers(count, dcb, dcb_sz, ccb, ccb_sz);
    if (r != 0)
        return r;

    // 2. nuestro DCB: genera el intermedio (y post-procesa el real) y flipea el intermedio
    int m = (int)(g.seq & 1);
    pm4_t c;
    pm4_init(&c, g.dcb[s], DCB_DWORDS);
    int own_real = g.cfg.post != NX_POST_NONE || g.cfg.aa;
    nx_job_t job = { .prev = &g.game_img[g.prev_idx], .cur = &g.game_img[gi], .out_mid = &g.mid_img[m],
                     .out_real = own_real ? &g.real_img[m] : NULL, .slot = s,
                     .post = (nx_post_t)g.cfg.post, .aa = g.cfg.aa,
                     .sharpness_stops = g.cfg.sharpness, .seq = ++g.seq };
    if (nx_pipeline_build(&g.pipe, &c, &job) != 0) {
        g.active = 0;
        Notify(TEX_ICON_SYSTEM, "VCFG-NX: error armando el DCB, desactivado");
        return flip_only(handle, (int)index, mode, arg);
    }
    g.slot_seq[s] = job.seq;
    void* our[1] = { g.dcb[s] };
    u32 our_sz[1] = { (u32)(c.n * 4) };
    // VALIDAR: el juego pide flipMode VSYNC; ambos flips quedan en orden de GPU
    r = HOOK_CONTINUE(sceGnmSubmitAndFlipCommandBuffers, submit_flip_t, 1, our, our_sz, NULL, NULL,
                      handle, (u32)g.mid_index[m], 1 /* VSYNC */, arg);

    // 3. el fotograma real, un vblank despues
    int real = own_real ? g.real_index[m] : (int)index;
    flip_only(handle, real, 1, arg);

    g.prev_idx = gi;
    g.slot ^= 1;
    return r;
}

s32 scePadReadState_hook(s32 handle, OrbisPadData* data)
{
    s32 r = HOOK_CONTINUE(scePadReadState, s32 (*)(s32, OrbisPadData*), handle, data);
    if (r == 0 && data && g.ready) {
        int combo = (data->buttons & (PAD_L3 | PAD_R3)) == (PAD_L3 | PAD_R3);
        uint64_t t = now_us();
        if (!combo) {
            g.combo_since_us = 0;
        } else if (!g.combo_since_us) {
            g.combo_since_us = t;
        } else if (g.combo_since_us != 1 && t - g.combo_since_us > (uint64_t)g.cfg.toggle_hold_ms * 1000) {
            g.active = !g.active;
            g.combo_since_us = 1;   // no repetir hasta soltar
            sceVideoOutSetFlipRate(g.handle, 0);
            Notify(TEX_ICON_SYSTEM, g.active ? "VCFG-NX: frame generation ON" : "VCFG-NX: frame generation OFF");
        }
    }
    return r;
}

s32 attr_public plugin_load(s32 argc, const char* argv[])
{
    final_printf("[GoldHEN] <%s\\Ver.0x%08x> %s\n", g_pluginName, g_pluginVersion, __func__);
    struct proc_info info;
    if (sys_sdk_proc_info(&info) == 0)
        memcpy(g.title, info.titleid, sizeof g.title - 1);
    if (strncmp(g.title, "CUSA", 4) != 0) {
        final_printf("proceso '%s' no es un juego: sin hooks\n", g.title);
        return 0;
    }
    nx_config_load(&g.cfg, g.title);
    HOOK32(sceVideoOutRegisterBuffers);
    HOOK32(sceVideoOutSetFlipRate);
    HOOK32(sceGnmSubmitAndFlipCommandBuffers);
    HOOK32(scePadReadState);
    return 0;
}

s32 attr_public plugin_unload(s32 argc, const char* argv[])
{
    UNHOOK(sceVideoOutRegisterBuffers);
    UNHOOK(sceVideoOutSetFlipRate);
    UNHOOK(sceGnmSubmitAndFlipCommandBuffers);
    UNHOOK(scePadReadState);
    return 0;
}

s32 attr_module_hidden module_start(s64 argc, const void* args) { return 0; }
s32 attr_module_hidden module_stop(s64 argc, const void* args) { return 0; }
