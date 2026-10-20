// VCFG-NX: frame generation (30 -> 60) + reescalado IA NXSR + SMAA/FSR1 para
// PS4 como plugin de GoldHEN, con menu en pantalla estilo VCFG (L3+R3).
//
// Este archivo solo conecta los hooks con el motor (nx_engine.c) y le da las
// funciones reales del sistema. Toda la logica (cuando reservar memoria,
// protecciones, menu) esta en el motor y se prueba en el PC con
// plugin/test/console_sim.c.
//
// ESTADO: compila con OpenOrbis + GoldHEN SDK; NO probado en consola. Los
// puntos a validar en hardware estan marcados con "VALIDAR".
#include <Common.h>
#include <orbis/GnmDriver.h>
#include <orbis/Pad.h>
#include <orbis/VideoOut.h>
#include <orbis/libkernel.h>
#include <stdio.h>
#include <string.h>

#include "nx_engine.h"
#include "nx_sys.h"

attr_public const char* g_pluginName = "vcfg_nx";
attr_public const char* g_pluginDesc = "Frame generation + reescalado IA (NXSR) + SMAA/FSR1, menu L3+R3";
attr_public const char* g_pluginAuth = "VCFG-NX";
attr_public u32 g_pluginVersion = 0x00000200; // 2.00

// prototipos de las variantes ForWorkload (OpenOrbis las declara sin argumentos)
typedef s32 (*submit_wl_t)(u64, u32, void**, u32*, void**, u32*);
typedef s32 (*submit_flip_wl_t)(u64, u32, void**, u32*, void**, u32*, u32, u32, u32, s64);
typedef s32 (*submit_flip_t)(u32, void**, u32*, void**, u32*, u32, u32, u32, u64);
typedef s32 (*vo_register_fn)(s32, s32, void* const*, s32, const OrbisVideoOutBufferAttribute*);

HOOK_INIT(sceVideoOutRegisterBuffers);
HOOK_INIT(sceVideoOutSetFlipRate);
HOOK_INIT(sceGnmSubmitAndFlipCommandBuffers);
HOOK_INIT(sceGnmSubmitAndFlipCommandBuffersForWorkload);
HOOK_INIT(sceVideoOutSubmitFlip);
HOOK_INIT(scePadReadState);
HOOK_INIT(scePadRead);

static int g_hooked;

// ------------------------------------------------------------- plataforma
static void* pf_alloc_gpu(size_t len, size_t align)
{
    off_t phys = 0;
    void* va = NULL;
    len = (len + 0xFFFF) & ~(size_t)0xFFFF;
    if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), len, align,
                                      ORBIS_KERNEL_WC_GARLIC, &phys) != 0)
        return NULL;
    if (sceKernelMapDirectMemory(&va, len, ORBIS_KERNEL_PROT_CPU_RW | ORBIS_KERNEL_PROT_GPU_RW,
                                 0, phys, align) != 0) {
        sceKernelReleaseDirectMemory(phys, len);
        return NULL;
    }
    return va;   // en PS4 la direccion virtual de CPU es la misma que ve la GPU
}

static size_t pf_avail_direct(void)
{
    off_t start = 0;
    size_t size = 0;
    // el bloque contiguo mas grande que queda libre. OpenOrbis declara los dos
    // ultimos parametros por valor; en el sistema son punteros de salida.
    typedef int32_t (*avail_fn)(off_t, off_t, size_t, off_t*, size_t*);
    if (((avail_fn)sceKernelAvailableDirectMemorySize)(0, sceKernelGetDirectMemorySize(), 64 * 1024, &start, &size) != 0)
        return 0;
    return size;
}

static int pf_register(int handle, int start, void* const* addrs, int num, const nx_vo_attr_t* a)
{
    OrbisVideoOutBufferAttribute oa;
    memset(&oa, 0, sizeof oa);
    oa.format = a->format; oa.tmode = a->tmode; oa.aspect = a->aspect;
    oa.width = a->width; oa.height = a->height; oa.pixelPitch = a->pitch;
    return HOOK_CONTINUE(sceVideoOutRegisterBuffers, vo_register_fn, handle, start, addrs, num, &oa);
}

static int pf_set_flip_rate(int handle, int rate)
{
    return HOOK_CONTINUE(sceVideoOutSetFlipRate, s32 (*)(s32, s32), handle, rate);
}

static int pf_submit(const nx_submit_t* s)
{
    if (s->for_workload)   // VALIDAR: firma de la variante ForWorkload
        return ((submit_wl_t)sceGnmSubmitCommandBuffersForWorkload)(s->workload, s->count, s->dcb, s->dcb_sz,
                                                                      s->ccb, s->ccb_sz);
    return sceGnmSubmitCommandBuffers(s->count, s->dcb, s->dcb_sz, s->ccb, s->ccb_sz);
}

static int pf_submit_flip(const nx_submit_t* s)
{
    if (s->cpu_flip && s->count == 0)   // el flip de CPU original del juego
        return HOOK_CONTINUE(sceVideoOutSubmitFlip, s32 (*)(s32, s32, u32, s64), (s32)s->handle, (s32)s->index,
                             s->mode, s->arg);
    if (s->for_workload)
        return HOOK_CONTINUE(sceGnmSubmitAndFlipCommandBuffersForWorkload, submit_flip_wl_t, s->workload,
                             s->count, s->dcb, s->dcb_sz, s->ccb, s->ccb_sz, s->handle, s->index, s->mode, s->arg);
    return HOOK_CONTINUE(sceGnmSubmitAndFlipCommandBuffers, submit_flip_t, s->count, s->dcb, s->dcb_sz,
                         s->ccb, s->ccb_sz, s->handle, s->index, s->mode, (u64)s->arg);
}

static uint64_t pf_now_us(void) { return sceKernelGetProcessTime(); }
static void pf_sleep_us(uint32_t us) { sceKernelUsleep(us); }
static void pf_notify(const char* msg) { Notify(TEX_ICON_SYSTEM, "%s", msg); }
static void pf_log(const char* msg) { final_printf("%s", msg); }

static int pf_read_file(const char* path, char* buf, int cap)
{
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    int n = (int)fread(buf, 1, (size_t)cap, f);
    fclose(f);
    return n;
}

static int pf_write_file(const char* path, const char* data, int len)
{
    mkdir("/data/vcfg_nx", 0777);
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    int n = (int)fwrite(data, 1, (size_t)len, f);
    fclose(f);
    return n;
}

static const nx_platform_t g_pf = {
    pf_alloc_gpu, pf_avail_direct, pf_register, pf_set_flip_rate, pf_submit, pf_submit_flip,
    pf_now_us, pf_sleep_us, pf_notify, pf_log, pf_read_file, pf_write_file,
};

// ----------------------------------------------------------------- hooks
s32 sceVideoOutRegisterBuffers_hook(s32 handle, s32 start, void* const* addrs, s32 num,
                                    const OrbisVideoOutBufferAttribute* attr)
{
    s32 r = HOOK_CONTINUE(sceVideoOutRegisterBuffers, vo_register_fn, handle, start, addrs, num, attr);
    if (attr) {
        nx_vo_attr_t a = { attr->format, attr->tmode, attr->aspect, attr->width, attr->height, attr->pixelPitch };
        nx_engine_on_register(handle, start, addrs, num, &a, r);
    }
    return r;
}

s32 sceVideoOutSetFlipRate_hook(s32 handle, s32 rate)
{
    return HOOK_CONTINUE(sceVideoOutSetFlipRate, s32 (*)(s32, s32), handle, nx_engine_on_set_flip_rate(handle, rate));
}

s32 sceGnmSubmitAndFlipCommandBuffers_hook(u32 count, void** dcb, u32* dcb_sz, void** ccb, u32* ccb_sz,
                                           u32 handle, u32 index, u32 mode, u64 arg)
{
    nx_submit_t s = { count, dcb, dcb_sz, ccb, ccb_sz, handle, index, mode, (int64_t)arg, 0, 0, 0 };
    return nx_engine_on_submit_flip(&s);
}

// juegos que presentan desde la CPU: su trabajo de GPU ya esta en la cola, asi
// que nuestro DCB + flip queda ordenado despues. VALIDAR en consola.
s32 sceVideoOutSubmitFlip_hook(s32 handle, s32 index, u32 mode, s64 arg)
{
    nx_submit_t s = { 0, NULL, NULL, NULL, NULL, (u32)handle, (u32)index, mode, arg, 0, 0, 1 };
    return nx_engine_on_submit_flip(&s);
}

s32 sceGnmSubmitAndFlipCommandBuffersForWorkload_hook(u64 workload, u32 count, void** dcb, u32* dcb_sz, void** ccb,
                                                      u32* ccb_sz, u32 handle, u32 index, u32 mode, s64 arg)
{
    nx_submit_t s = { count, dcb, dcb_sz, ccb, ccb_sz, handle, index, mode, arg, workload, 1, 0 };
    return nx_engine_on_submit_flip(&s);
}

static void pad_filter(OrbisPadData* d)
{
    uint8_t sticks[4] = { d->leftStick.x, d->leftStick.y, d->rightStick.x, d->rightStick.y };
    uint8_t trig[2] = { d->analogButtons.l2, d->analogButtons.r2 };
    nx_engine_on_pad(&d->buttons, sticks, trig);
    d->leftStick.x = sticks[0]; d->leftStick.y = sticks[1];
    d->rightStick.x = sticks[2]; d->rightStick.y = sticks[3];
    d->analogButtons.l2 = trig[0]; d->analogButtons.r2 = trig[1];
}

s32 scePadReadState_hook(s32 handle, OrbisPadData* data)
{
    s32 r = HOOK_CONTINUE(scePadReadState, s32 (*)(s32, OrbisPadData*), handle, data);
    if (r == 0 && data) pad_filter(data);
    return r;
}

s32 scePadRead_hook(s32 handle, OrbisPadData* data, s32 count)
{
    s32 r = HOOK_CONTINUE(scePadRead, s32 (*)(s32, OrbisPadData*, s32), handle, data, count);
    for (s32 i = 0; data && i < r && i < count; ++i) pad_filter(&data[i]);
    return r;
}

s32 attr_public plugin_load(s32 argc, const char* argv[])
{
    final_printf("<%s ver 0x%08x> %s\n", g_pluginName, g_pluginVersion, __func__);
    struct proc_info info;
    char title[16] = { 0 };
    if (sys_sdk_proc_info(&info) == 0)
        memcpy(title, info.titleid, sizeof title - 1);
    nx_engine_init(&g_pf, title);
    if (nx_engine_info()->state == NX_ST_OFF) {
        final_printf("proceso '%s': sin hooks (%s)\n", title, nx_engine_info()->status);
        return 0;
    }
    HOOK32(sceVideoOutRegisterBuffers);
    HOOK32(sceVideoOutSetFlipRate);
    HOOK32(sceGnmSubmitAndFlipCommandBuffers);
    HOOK32(sceGnmSubmitAndFlipCommandBuffersForWorkload);
    HOOK32(sceVideoOutSubmitFlip);
    HOOK32(scePadReadState);
    HOOK32(scePadRead);
    g_hooked = 1;
    return 0;
}

s32 attr_public plugin_unload(s32 argc, const char* argv[])
{
    if (!g_hooked) return 0;
    UNHOOK(sceVideoOutRegisterBuffers);
    UNHOOK(sceVideoOutSetFlipRate);
    UNHOOK(sceGnmSubmitAndFlipCommandBuffers);
    UNHOOK(sceGnmSubmitAndFlipCommandBuffersForWorkload);
    UNHOOK(sceVideoOutSubmitFlip);
    UNHOOK(scePadReadState);
    UNHOOK(scePadRead);
    return 0;
}

s32 attr_module_hidden module_start(s64 argc, const void* args) { return 0; }
s32 attr_module_hidden module_stop(s64 argc, const void* args) { return 0; }
