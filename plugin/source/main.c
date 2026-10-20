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
attr_public u32 g_pluginVersion = 0x00000250; // 2.50

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

static int g_hooked, g_hooked_dm;
static char g_boot_path[64];
static char g_title_s[16];
static int g_level;   // nivel de arranque (ver boot_level_pick)
static void boot_stage(const char* title, const char* what);
static void boot_level_ok(void);   // /data/vcfg_nx/boot-<TITLE>.txt: el arranque con reserva llego a jugarse

// ------------------------------------------------------------- plataforma
// Memoria. Como VCFG v2.23: se reserva apenas carga el plugin (antes de que el
// juego se quede con toda la memoria directa, como hace GTA V). Dos bloques:
//   base (4 MiB): menu, DCBs, OSD. Nunca se devuelve.
//   trabajo (POOL_MB): arena y buffers de los efectos. Si el juego pide memoria
//   y no hay, mientras no lo hayamos usado se le devuelve ("valvula") y la
//   llamada del juego se repite: el juego nunca se queda sin memoria por nosotros.
HOOK_INIT(sceKernelAllocateDirectMemory);
typedef int32_t (*alloc_dm_fn)(off_t, off_t, size_t, size_t, int, off_t*);
typedef int32_t (*avail_fn)(off_t, off_t, size_t, off_t*, size_t*);

#define BASE_POOL (4u << 20)
#define WORK_POOL_MB 112
typedef struct { off_t phys; uint8_t* va; size_t len, used; } pool_t;
static pool_t g_base, g_work;
static volatile int g_work_given_back, g_valve_count;

static int32_t dm_alloc(off_t lo, off_t hi, size_t len, size_t align, int type, off_t* out)
{
    if (g_hooked_dm)
        return HOOK_CONTINUE(sceKernelAllocateDirectMemory, alloc_dm_fn, lo, hi, len, align, type, out);
    return sceKernelAllocateDirectMemory(lo, hi, len, align, type, out);
}

static void* map_new(size_t len, size_t align, off_t* phys_out)
{
    off_t phys = 0;
    void* va = NULL;
    len = (len + 0xFFFF) & ~(size_t)0xFFFF;
    if (dm_alloc(0, sceKernelGetDirectMemorySize(), len, align, ORBIS_KERNEL_WC_GARLIC, &phys) != 0) return NULL;
    if (sceKernelMapDirectMemory(&va, len, ORBIS_KERNEL_PROT_CPU_RW | ORBIS_KERNEL_PROT_GPU_RW, 0, phys, align) != 0) {
        sceKernelReleaseDirectMemory(phys, len);
        return NULL;
    }
    if (phys_out) *phys_out = phys;
    return va;   // en PS4 la direccion virtual de CPU es la misma que ve la GPU
}

static int pool_init(pool_t* p, size_t len)
{
    p->va = map_new(len, 2u << 20, &p->phys);
    p->len = p->va ? len : 0;
    p->used = 0;
    return p->va != NULL;
}

static void* pool_take(pool_t* p, size_t len, size_t align)
{
    if (!p->va) return NULL;
    size_t at = (p->used + align - 1) & ~(align - 1);
    if (at + len > p->len) return NULL;
    p->used = at + len;
    return p->va + at;
}

static void* pf_alloc_gpu(size_t len, size_t align)
{
    len = (len + 0xFFFF) & ~(size_t)0xFFFF;
    if (align < 65536) align = 65536;
    void* r = len <= (2u << 20) ? pool_take(&g_base, len, align) : NULL;
    if (!r && !g_work_given_back) r = pool_take(&g_work, len, align);
    return r ? r : map_new(len, align, NULL);
}

static size_t raw_avail(void)
{
    off_t start = 0;
    size_t size = 0;
    // el bloque contiguo mas grande que queda libre. OpenOrbis declara los dos
    // ultimos parametros por valor; en el sistema son punteros de salida.
    if (((avail_fn)sceKernelAvailableDirectMemorySize)(0, sceKernelGetDirectMemorySize(), 64 * 1024, &start, &size) != 0)
        return 0;
    return size;
}

static size_t pf_avail_direct(void) { return raw_avail(); }

// lo que queda en la reserva propia (base = 1: la del menu)
static size_t pf_pool_free(int base)
{
    if (base) return g_base.va ? g_base.len - g_base.used : 0;
    return g_work.va && !g_work_given_back ? g_work.len - g_work.used : 0;
}

static int give_back_work(void)
{
    if (!g_work.va || g_work_given_back || g_work.used) return 0;
    g_work_given_back = 1;
    sceKernelMunmap(g_work.va, g_work.len);
    sceKernelReleaseDirectMemory(g_work.phys, g_work.len);
    g_valve_count++;
    return 1;
}

// el juego pide memoria directa: si no hay y nuestra reserva de trabajo sigue
// sin usar, se la devolvemos y se repite su pedido
s32 sceKernelAllocateDirectMemory_hook(off_t lo, off_t hi, size_t len, size_t align, int type, off_t* out)
{
    s32 r = HOOK_CONTINUE(sceKernelAllocateDirectMemory, alloc_dm_fn, lo, hi, len, align, type, out);
    if (r == 0 || !give_back_work()) return r;
    return HOOK_CONTINUE(sceKernelAllocateDirectMemory, alloc_dm_fn, lo, hi, len, align, type, out);
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

typedef int (*fsync_fn)(int);
static fsync_fn g_fsync;

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
    // al disco ya: si la consola se cuelga y hay que apagarla a la fuerza, la
    // marca de arranque y las etapas tienen que quedar (fsync se busca por
    // nombre: ningun PRX que funcione lo importa)
    if (g_fsync) g_fsync(fd);
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
    pf_pool_free,
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
    int fails = 0, ok_logged = 0, boot_ok = 0, valve_seen = 0, seen = 0;
    uint64_t next_tick = 0, next_try = 0;
    while (g_pad_run) {
        uint64_t t = sceKernelGetProcessTime();
        if (t >= next_tick) {
            nx_engine_tick();
            next_tick = t + 1000000;
            const nx_engine_info_t* inf = nx_engine_info();
            if (inf->flips && !(seen & 1)) { seen |= 1; boot_stage(g_title_s, "primer fotograma del juego"); }
            if (inf->state >= NX_ST_WARMUP && !(seen & 2)) { seen |= 2; boot_stage(g_title_s, "pantalla del juego registrada"); }
            if (inf->state == NX_ST_RUN && !(seen & 4)) { seen |= 4; boot_stage(g_title_s, "menu listo"); }
            if (inf->full_ready && !(seen & 8)) { seen |= 8; boot_stage(g_title_s, "efectos activos"); }
            if (!boot_ok && (nx_engine_info()->flips > 900 || (g_level >= 2 && t > 60000000))) {
                boot_ok = 1;
                boot_level_ok();
                boot_stage(g_title_s, "jugando: arranque correcto");
            }
            if (g_valve_count != valve_seen) {
                valve_seen = g_valve_count;
                nx_engine_note("[mem] el juego necesito memoria: se le devolvio la reserva de trabajo");
            }
        }
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
            s32 r = scePadReadState(handle, &d);
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

// Arranque por niveles (como el "crash guard" de VCFG v2.23). Si un arranque
// no llega a jugarse (~30 s de fotogramas), el siguiente usa un nivel mas
// prudente; el nivel que funciono queda guardado para ese juego:
//   0  4 MiB tempranos para el menu; los efectos piden memoria tarde (si hay)
//   1  sin reserva temprana: todo se pide tarde
//   2  pasivo: sin hooks; el juego corre como si el plugin no estuviera
// (la reserva de efectos de 112 MiB al arrancar hacia cerrar GTA V: quedo
// apagada; WORK_POOL_MB y la valvula siguen para probar en otros juegos)
#define BOOT_LEVELS 3
#define EARLY_WORK 0
static int g_level;
static const char* const LEVEL_NAME[BOOT_LEVELS] = { "normal", "sin reserva temprana", "pasivo (sin efectos)" };

static void boot_stage(const char* title, const char* what)
{
    char m[160];
    int n = snprintf(m, sizeof m, "[etapa] %s nivel %d: %s\n", title, g_level, what);
    if (n > 0) pf_append_file("/data/vcfg_nx/logs/cargas.txt", m, n);
}

static void boot_level_pick(const char* title)
{
    snprintf(g_boot_path, sizeof g_boot_path, "/data/vcfg_nx/boot-%.12s.txt", title);
    char prev[64] = { 0 };
    int n = pf_read_file(g_boot_path, prev, sizeof prev - 1);
    int lv = 0, pending = 0;
    if (n > 0) {
        const char* l = strstr(prev, "level=");
        lv = l ? atoi(l + 6) : 0;
        pending = l && strstr(prev, "state=pending") != NULL;   // los archivos de la 2.48 o antes no cuentan
    }
    if (lv < 0 || lv >= BOOT_LEVELS) lv = 0;
    if (pending && lv < BOOT_LEVELS - 1) lv++;   // el anterior se cerro antes de jugar: mas prudente

    g_level = lv;
    char t[48];
    int k = snprintf(t, sizeof t, "level=%d state=pending\n", lv);
    pf_write_file(g_boot_path, t, k);
}

static void boot_level_ok(void)
{
    char t[48];
    int k = snprintf(t, sizeof t, "level=%d state=ok\n", g_level);
    pf_write_file(g_boot_path, t, k);
}

static void early_reserve(const char* title)
{
    if (g_level >= 1) return;
    pool_init(&g_base, BASE_POOL);
    if (EARLY_WORK) {
        pool_init(&g_work, (size_t)WORK_POOL_MB << 20);
        HOOK32(sceKernelAllocateDirectMemory);   // la valvula (v2.23 engancha la misma funcion)
        g_hooked_dm = 1;
    }
    char m[160];
    int k = snprintf(m, sizeof m, "[mem] reserva temprana: base %s, trabajo %s; libre del sistema %zu MiB\n",
                     g_base.va ? "ok" : "FALLO", g_work.va ? "ok" : "no", raw_avail() >> 20);
    if (k > 0) pf_append_file("/data/vcfg_nx/logs/cargas.txt", m, k);
}

// marca en logs/cargas.txt apenas GoldHEN abre el plugin: si el juego no deja
// ninguna, GoldHEN no lo cargo; si queda module_start sin "cargado en", algo
// fallo en el medio
static void boot_mark(const char* what, const char* title)
{
    char m[96];
    int n = snprintf(m, sizeof m, "[boot] %s %x.%02x %s\n", what, g_pluginVersion >> 8, g_pluginVersion & 0xFF, title);
    if (n > 0) pf_append_file("/data/vcfg_nx/logs/cargas.txt", m, n);
}

s32 attr_public plugin_load(s32 argc, const char* argv[])
{
    final_printf("<%s ver 0x%08x> %s\n", g_pluginName, g_pluginVersion, __func__);
    struct proc_info info;
    char title[16] = { 0 };
    if (sys_sdk_proc_info(&info) == 0)
        memcpy(title, info.titleid, sizeof title - 1);
    make_dirs();
    g_fsync = (fsync_fn)find_sym("sceKernelFsync");
    memcpy(g_title_s, title, sizeof g_title_s);
    boot_mark("plugin_load", title);
    // aviso visible lo antes posible: si no aparece al abrir el juego, GoldHEN no cargo el plugin
    {
        char n[96];
        snprintf(n, sizeof n, "VCFG-NX %x.%02x cargado en %s. Mantener L3+R3: menu", g_pluginVersion >> 8,
                 g_pluginVersion & 0xFF, title);
        pf_notify(n);
    }
    nx_engine_init(&g_pf, title);
    if (nx_engine_info()->state == NX_ST_OFF) {
        final_printf("proceso '%s': sin hooks (%s)\n", title, nx_engine_info()->status);
        return 0;
    }
    boot_level_pick(title);
    {
        char n[128];
        snprintf(n, sizeof n, "VCFG-NX: arranque en nivel %d (%s)", g_level, LEVEL_NAME[g_level]);
        if (g_level) pf_notify(n);
    }
    boot_stage(title, "inicio");
    if (g_level >= 2) {   // pasivo: solo se deja constancia; si el juego igual se cierra, no es el plugin
        boot_stage(title, "pasivo: sin hooks");
        g_pad_run = 1;
        if (scePthreadCreate(&g_pad_thread, NULL, pad_thread, NULL, "vcfg_nx_pad") != 0) g_pad_run = 0;
        return 0;
    }
    if (g_level == 0) g_temp = (temp_fn)find_sym("sceKernelGetCpuTemperature");
    early_reserve(title);
    boot_stage(title, "memoria lista");
    HOOK32(sceVideoOutRegisterBuffers);
    HOOK32(sceVideoOutSetFlipRate);
    HOOK32(sceGnmSubmitAndFlipCommandBuffers);
    HOOK32(sceGnmSubmitAndFlipCommandBuffersForWorkload);
    HOOK32(sceVideoOutSubmitFlip);
    boot_stage(title, "hooks de video listos");
    // el mando NO se engancha (como VCFG v2.23: "pad entry hooks disabled"):
    // solo lo lee el hilo propio. El juego sigue recibiendo los botones con el
    // menu abierto, pero no se toca nada de su lectura del mando.
    g_hooked = 1;
    g_pad_run = 1;
    int tr = scePthreadCreate(&g_pad_thread, NULL, pad_thread, NULL, "vcfg_nx_pad");
    if (tr != 0) g_pad_run = 0;
    char m[160];
    snprintf(m, sizeof m, "[main] plugin %x.%02x cargado en %s; hooks listos; hilo del mando rc=0x%08x; temperatura %s\n",
             g_pluginVersion >> 8, g_pluginVersion & 0xFF, title, (unsigned)tr,
             g_temp ? "disponible" : "no disponible");
    pf_log(m);
    pf_append_file("/data/vcfg_nx/logs/cargas.txt", m, (int)strlen(m));
    nx_engine_flush_log();
    boot_stage(title, "plugin_load terminado");
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
    if (g_hooked_dm) { g_hooked_dm = 0; UNHOOK(sceKernelAllocateDirectMemory); }

    return 0;
}

s32 attr_module_hidden module_start(s64 argc, const void* args) { return 0; }
s32 attr_module_hidden module_stop(s64 argc, const void* args) { return 0; }
