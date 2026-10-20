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
#include <orbis/UserService.h>
#include <orbis/libkernel.h>
#include <stdio.h>
#include <string.h>

#include "nx_engine.h"
#include "nx_sys.h"

attr_public const char* g_pluginName = "vcfg_nx";
attr_public const char* g_pluginDesc = "Frame generation + reescalado IA (NXSR) + SMAA/FSR1, menu L3+R3";
attr_public const char* g_pluginAuth = "VCFG-NX";
attr_public u32 g_pluginVersion = 0x00000240; // 2.40

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
// las variantes Ext se enganchan solo si el sistema las tiene (sin importarlas)
HOOK_INIT(scePadReadStateExt);
HOOK_INIT(scePadReadExt);
static int g_ext_state, g_ext_read;

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

// archivos con las llamadas del kernel (como v2.23): no dependen de la libc
// del juego ni de su estado
#define K_RDONLY 0x0000
#define K_WRONLY 0x0001
#define K_APPEND 0x0008
#define K_CREAT 0x0200
#define K_TRUNC 0x0400

static void make_dirs(void)
{
    sceKernelMkdir("/data/vcfg_nx", 0777);
    sceKernelMkdir("/data/vcfg_nx/logs", 0777);
}

static int pf_read_file(const char* path, char* buf, int cap)
{
    int fd = sceKernelOpen(path, K_RDONLY, 0);
    if (fd < 0) return -1;
    int n = 0;
    while (n < cap) {
        int r = (int)sceKernelRead(fd, buf + n, (size_t)(cap - n));
        if (r <= 0) break;
        n += r;
    }
    sceKernelClose(fd);
    return n;
}

static int write_flags(const char* path, const char* data, int len, int flags)
{
    int fd = sceKernelOpen(path, K_WRONLY | K_CREAT | flags, 0777);
    if (fd < 0) {
        make_dirs();
        fd = sceKernelOpen(path, K_WRONLY | K_CREAT | flags, 0777);
        if (fd < 0) return -1;
    }
    int n = 0;
    while (n < len) {
        int r = (int)sceKernelWrite(fd, data + n, (size_t)(len - n));
        if (r <= 0) break;
        n += r;
    }
    sceKernelClose(fd);
    return n;
}

static int pf_write_file(const char* path, const char* data, int len) { return write_flags(path, data, len, K_TRUNC); }
static int pf_append_file(const char* path, const char* data, int len) { return write_flags(path, data, len, K_APPEND); }

// el juego corre con la GPU completa de PS4 Pro (modo Neo): alcanza para NXSR Calidad
static int pf_is_pro(void) { return sceKernelIsNeoMode() == 1; }

// temperatura del procesador. Solo algunos modulos del sistema la exportan: se
// busca por nombre en los modulos cargados en vez de importarla (una importacion
// que el proceso del juego no resuelve puede impedir que el plugin cargue).
typedef int (*temp_fn)(int*);
static temp_fn g_temp;

// busca una funcion por nombre en todos los modulos cargados (NULL si no esta)
static void* find_sym(const char* name)
{
    OrbisKernelModule mods[256];
    size_t n = 0;
    if (sceKernelGetModuleList(mods, sizeof mods / sizeof mods[0], &n) != 0) return NULL;
    if (n > sizeof mods / sizeof mods[0]) n = sizeof mods / sizeof mods[0];
    for (size_t i = 0; i < n; ++i) {
        void* f = NULL;
        if (sceKernelDlsym((int32_t)mods[i], name, &f) == 0 && f) return f;
    }
    return NULL;
}

static int pf_temp_c(void)
{
    int t = -1;
    if (!g_temp || g_temp(&t) != 0) return -1;
    return t;
}

static const nx_platform_t g_pf = {
    pf_alloc_gpu, pf_avail_direct, pf_register, pf_set_flip_rate, pf_submit, pf_submit_flip,
    pf_now_us, pf_sleep_us, pf_notify, pf_log, pf_read_file, pf_write_file, pf_append_file, pf_is_pro, pf_temp_c,
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

typedef s32 (*pad_state_fn)(s32, OrbisPadData*);
typedef s32 (*pad_read_fn)(s32, OrbisPadData*, s32);

s32 scePadReadState_hook(s32 handle, OrbisPadData* data)
{
    s32 r = HOOK_CONTINUE(scePadReadState, pad_state_fn, handle, data);
    if (r == 0 && data) pad_filter(data);
    return r;
}

s32 scePadRead_hook(s32 handle, OrbisPadData* data, s32 count)
{
    s32 r = HOOK_CONTINUE(scePadRead, pad_read_fn, handle, data, count);
    for (s32 i = 0; data && i < r && i < count; ++i) pad_filter(&data[i]);
    return r;
}

// variantes Ext (las usan muchos juegos grandes): mismo formato de datos al principio
s32 scePadReadStateExt_hook(s32 handle, OrbisPadData* data)
{
    s32 r = HOOK_CONTINUE(scePadReadStateExt, pad_state_fn, handle, data);
    if (r == 0 && data) pad_filter(data);
    return r;
}

s32 scePadReadExt_hook(s32 handle, OrbisPadData* data, s32 count)
{
    s32 r = HOOK_CONTINUE(scePadReadExt, pad_read_fn, handle, data, count);
    for (s32 i = 0; data && i < r && i < count; ++i) pad_filter(&data[i]);
    return r;
}

// hilo propio: lee el mando del usuario cada ~16 ms con la funcion original
// (sin pasar por nuestros hooks), asi L3+R3 funciona aunque el juego lea el
// mando de una forma que no vemos. Tambien hace el diagnostico periodico.
static volatile int g_pad_run;
static OrbisPthread g_pad_thread;

static void* pad_thread(void* arg)
{
    (void)arg;
    s32 handle = -1;
    int fails = 0, ok_logged = 0;
    uint64_t next_tick = 0, next_try = 0;
    while (g_pad_run) {
        uint64_t t = sceKernelGetProcessTime();
        if (t >= next_tick) { nx_engine_tick(); next_tick = t + 1000000; }
        if (handle < 0 && t >= next_try) {
            next_try = t + 500000;
            s32 user = -1;
            if (sceUserServiceGetInitialUser(&user) == 0 && user >= 0) {
                s32 h = scePadGetHandle(user, 0, 0);   // el que abrio el juego
                if (h >= 0) { handle = h; fails = 0; }
            }
        }
        if (handle >= 0) {
            OrbisPadData d;
            memset(&d, 0, sizeof d);
            s32 r = HOOK_CONTINUE(scePadReadState, pad_state_fn, handle, &d);
            if (r == 0) {
                if (!ok_logged) { ok_logged = 1; nx_engine_pad_polled(1); }
                fails = 0;
                nx_engine_pad_logic(d.connected ? d.buttons : 0);
            } else if (++fails > 60) {
                handle = -1;   // el juego cerro/reabrio el mando: buscarlo otra vez
                if (ok_logged) { ok_logged = 0; nx_engine_pad_polled(0); }
            }
        }
        sceKernelUsleep(16000);
    }
    return NULL;
}

s32 attr_public plugin_load(s32 argc, const char* argv[])
{
    final_printf("<%s ver 0x%08x> %s\n", g_pluginName, g_pluginVersion, __func__);
    struct proc_info info;
    char title[16] = { 0 };
    if (sys_sdk_proc_info(&info) == 0)
        memcpy(title, info.titleid, sizeof title - 1);
    make_dirs();
    nx_engine_init(&g_pf, title);
    if (nx_engine_info()->state == NX_ST_OFF) {
        final_printf("proceso '%s': sin hooks (%s)\n", title, nx_engine_info()->status);
        return 0;
    }
    g_temp = (temp_fn)find_sym("sceKernelGetCpuTemperature");
    HOOK32(sceVideoOutRegisterBuffers);
    HOOK32(sceVideoOutSetFlipRate);
    HOOK32(sceGnmSubmitAndFlipCommandBuffers);
    HOOK32(sceGnmSubmitAndFlipCommandBuffersForWorkload);
    HOOK32(sceVideoOutSubmitFlip);
    HOOK32(scePadReadState);
    HOOK32(scePadRead);
    void* ext = find_sym("scePadReadStateExt");
    if (ext) {
        Detour_Construct(&Detour_scePadReadStateExt, DetourMode_x32);
        Detour_DetourFunction(&Detour_scePadReadStateExt, (uint64_t)ext, (void*)&scePadReadStateExt_hook);
        g_ext_state = 1;
    }
    ext = find_sym("scePadReadExt");
    if (ext) {
        Detour_Construct(&Detour_scePadReadExt, DetourMode_x32);
        Detour_DetourFunction(&Detour_scePadReadExt, (uint64_t)ext, (void*)&scePadReadExt_hook);
        g_ext_read = 1;
    }
    g_hooked = 1;
    g_pad_run = 1;
    int tr = scePthreadCreate(&g_pad_thread, NULL, pad_thread, NULL, "vcfg_nx_pad");
    if (tr != 0) g_pad_run = 0;
    char m[160];
    snprintf(m, sizeof m, "[main] plugin %x.%02x cargado en %s; hooks listos (pad Ext: %d/%d); hilo del mando rc=0x%08x; temperatura %s\n",
             g_pluginVersion >> 8, g_pluginVersion & 0xFF, title, g_ext_state, g_ext_read, (unsigned)tr,
             g_temp ? "disponible" : "no disponible");
    pf_log(m);
    pf_append_file("/data/vcfg_nx/logs/cargas.txt", m, (int)strlen(m));
    nx_engine_flush_log();
    // aviso visible: si no aparece al abrir el juego, GoldHEN no cargo el plugin
    pf_notify("VCFG-NX cargado. Mantener L3+R3 para el menu");
    return 0;
}

s32 attr_public plugin_unload(s32 argc, const char* argv[])
{
    if (!g_hooked) return 0;
    if (g_pad_run) { g_pad_run = 0; scePthreadJoin(g_pad_thread, NULL); }
    nx_engine_flush_log();
    UNHOOK(sceVideoOutRegisterBuffers);
    UNHOOK(sceVideoOutSetFlipRate);
    UNHOOK(sceGnmSubmitAndFlipCommandBuffers);
    UNHOOK(sceGnmSubmitAndFlipCommandBuffersForWorkload);
    UNHOOK(sceVideoOutSubmitFlip);
    UNHOOK(scePadReadState);
    UNHOOK(scePadRead);
    if (g_ext_state) UNHOOK(scePadReadStateExt);
    if (g_ext_read) UNHOOK(scePadReadExt);
    return 0;
}

s32 attr_module_hidden module_start(s64 argc, const void* args) { return 0; }
s32 attr_module_hidden module_stop(s64 argc, const void* args) { return 0; }
