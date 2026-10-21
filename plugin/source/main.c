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
#include "nx_hook.h"
#include "nx_sys.h"

// Enganches con nx_hook (puente cerca de la funcion, instrucciones relativas
// corregidas) en vez del HOOK32 del SDK, cuyo puente en memoria de malloc puede
// quedar a mas de 2 GB y hacer saltar al juego a cualquier lado al arrancar.
// Mismos nombres de macro que el SDK para no cambiar el resto del archivo.
#undef HOOK_INIT
#undef HOOK32
#undef HOOK
#undef HOOK_CONTINUE
#undef UNHOOK
#define HOOK_INIT(name) static nx_hook_t NXH_##name
#define HOOK32(name) hook_one(&NXH_##name, (void*)name, (void*)&name##_hook, #name)
#define HOOK_CONTINUE(name, type, ...) (((type)(NXH_##name.stub ? NXH_##name.stub : (void*)name))(__VA_ARGS__))
#define UNHOOK(name) nx_hook_remove(&NXH_##name)
static int hook_one(nx_hook_t* h, void* fn, void* hk, const char* name);

// Traza de respaldo: una linea por paso en NX_LOGDIR/traza.txt y en
// /data/vcfg_nx/traza.txt, con llamadas del kernel como hace VCFG (abrir,
// agregar, cerrar; 0666; nada de stdio, que falla dentro del arranque del juego)
static int g_trace_ok = -1;   // -1 sin probar, 0 fallo, 1 escribio
static void trace_one(const char* path, const char* line, int len)
{
    int fd = sceKernelOpen(path, 0x209 /* O_WRONLY|O_CREAT|O_APPEND */, 0666);
    if (fd < 0) return;
    if (sceKernelWrite(fd, line, (size_t)len) == len) g_trace_ok = 1;
    sceKernelClose(fd);
}
static void trace(const char* line)
{
    int len = (int)strlen(line);
    g_trace_ok = 0;
    sceKernelMkdir(NX_LOGDIR, 0777);
    sceKernelMkdir("/data/vcfg_nx", 0777);
    trace_one(NX_LOGDIR "/traza.txt", line, len);
    trace_one("/data/vcfg_nx/traza.txt", line, len);
}

attr_public const char* g_pluginName = "vcfg_nx";
attr_public const char* g_pluginDesc = "Frame generation + reescalado IA (NXSR) + SMAA/FSR1, menu L3+R3";
attr_public const char* g_pluginAuth = "VCFG-NX";
attr_public u32 g_pluginVersion = 0x00000261; // 2.61

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

// --- plataforma de nx_hook: memoria ejecutable cerca de la funcion
void* nx_hook_map_near(void* near, size_t len)
{
    uintptr_t base = (uintptr_t)near & ~(uintptr_t)0xFFFF;
    for (int i = 1; i < 96; ++i) {
        // alterna abajo / arriba de la funcion, de a 16 MiB
        uintptr_t step = (uintptr_t)((i + 1) / 2) * (16u << 20);
        uintptr_t hint = (i & 1) ? base - step : base + step;
        void* p = NULL;
        if (sceKernelMmap((void*)hint, len, 7 /* VM_PROT_ALL */, 0x1000 | 0x2, -1, 0, &p) < 0 || !p) continue;
        int64_t d = (int64_t)(uintptr_t)p - (int64_t)(uintptr_t)near;
        if (d < 0x7FF00000LL && d > -0x7FF00000LL) return p;
        sceKernelMunmap(p, len);
    }
    return NULL;
}
void nx_hook_unmap(void* p, size_t len) { sceKernelMunmap(p, len); }
int nx_hook_make_writable(void* p, size_t len) { return sceKernelMprotect(p, len, 7) < 0 ? -1 : 0; }

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
#define K_FSYNC 0   // sin O_FSYNC: VCFG escribe sin el y le funciona en GTA V

static void make_dirs(void)
{
    sceKernelMkdir("/data/vcfg_nx", 0777);
    sceKernelMkdir(NX_LOGDIR, 0777);
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


static int g_write_err;   // ultimo error al abrir para escribir (se muestra en el aviso)
static int write_flags(const char* path, const char* data, int len, int flags)
{
    int fd = sceKernelOpen(path, K_WRONLY | K_CREAT | flags, 0666);
    if (fd < 0) {
        make_dirs();
        fd = sceKernelOpen(path, K_WRONLY | K_CREAT | flags, 0666);
        if (fd < 0) { g_write_err = fd; return -1; }
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
static int write_sync(const char* path, const char* data, int len) { return write_flags(path, data, len, K_TRUNC | K_FSYNC); }
static int append_sync(const char* path, const char* data, int len) { return write_flags(path, data, len, K_APPEND | K_FSYNC); }

// el juego corre con la GPU completa de PS4 Pro (modo Neo): alcanza para NXSR Calidad
static int pf_is_pro(void) { return sceKernelIsNeoMode() == 1; }

// temperatura del procesador. Solo algunos modulos del sistema la exportan: se
// busca por nombre en los modulos cargados en vez de importarla (una importacion
// que el proceso del juego no resuelve puede impedir que el plugin cargue).
typedef int (*temp_fn)(int*);
static temp_fn g_temp;

// (No se buscan funciones por nombre recorriendo los modulos del juego: hacerlo
// mientras el juego arranca lo cerraba. La temperatura queda sin leer.)

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

// El hilo propio (mando, avisos, etapas) recien arranca cuando el juego ya
// presento ~10 s de imagen, como VCFG v2.23 ("pad poll thread" desde la
// activacion, no desde plugin_load). Arrancarlo en plugin_load llamaba a
// UserService/Pad mientras el juego todavia se inicializaba.
static volatile int g_pad_run;
static OrbisPthread g_pad_thread;
static int g_thread_tried;
static int g_level_max = 4;   // BOOT_TOP
static int g_engine_live;        // nivel 3+: el motor procesa los flips; si no, pasan derecho
static uint64_t g_game_flips;    // contados aca (el motor puede no estar iniciado)
static void* pad_thread(void* arg);
static void boot_level_ok(void);
static void after_game_flip(void)
{
    uint64_t f = ++g_game_flips;
    if (f == 1) boot_stage(g_title_s, "primer fotograma del juego");
    // ~15-30 s de juego: este nivel anda. Con el motor procesando (nivel 3+), recien
    // cuando nuestro trabajo de GPU lleva ~600 fotogramas (o el juego ~20 min sin
    // activarlo): un cierre con los efectos recien encendidos no cuenta como correcto
    static int ok_done;
    if (!ok_done && f >= 900 && (!g_engine_live || nx_engine_info()->our_jobs >= 600 || f >= 36000)) {
        ok_done = 1;
        boot_level_ok();
        boot_stage(g_title_s, g_level < g_level_max ? "jugando: arranque correcto (el proximo va al nivel completo)"
                                                    : "jugando: arranque correcto");
    }
    // ningun hilo propio antes de que el juego este andando (como VCFG: sus
    // hilos arrancan desde los flips, nunca desde plugin_load)
    if (g_thread_tried || !g_engine_live || f < 300) return;
    g_thread_tried = 1;
    g_pad_run = 1;
    if (scePthreadCreate(&g_pad_thread, NULL, pad_thread, NULL, "vcfg_nx_pad") != 0) g_pad_run = 0;
    boot_stage(g_title_s, g_pad_run ? "300 fotogramas: hilo del plugin iniciado" : "300 fotogramas: el hilo no se pudo crear");
}

s32 sceGnmSubmitAndFlipCommandBuffers_hook(u32 count, void** dcb, u32* dcb_sz, void** ccb, u32* ccb_sz,
                                           u32 handle, u32 index, u32 mode, u64 arg)
{
    nx_submit_t s = { count, dcb, dcb_sz, ccb, ccb_sz, handle, index, mode, (int64_t)arg, 0, 0, 0 };
    s32 r = g_engine_live ? nx_engine_on_submit_flip(&s) : pf_submit_flip(&s);
    after_game_flip();
    return r;
}

// juegos que presentan desde la CPU: su trabajo de GPU ya esta en la cola, asi
// que nuestro DCB + flip queda ordenado despues. VALIDAR en consola.
s32 sceVideoOutSubmitFlip_hook(s32 handle, s32 index, u32 mode, s64 arg)
{
    nx_submit_t s = { 0, NULL, NULL, NULL, NULL, (u32)handle, (u32)index, mode, arg, 0, 0, 1 };
    s32 r = g_engine_live ? nx_engine_on_submit_flip(&s) : pf_submit_flip(&s);
    after_game_flip();
    return r;
}

s32 sceGnmSubmitAndFlipCommandBuffersForWorkload_hook(u64 workload, u32 count, void** dcb, u32* dcb_sz, void** ccb,
                                                      u32* ccb_sz, u32 handle, u32 index, u32 mode, s64 arg)
{
    nx_submit_t s = { count, dcb, dcb_sz, ccb, ccb_sz, handle, index, mode, arg, workload, 1, 0 };
    s32 r = g_engine_live ? nx_engine_on_submit_flip(&s) : pf_submit_flip(&s);
    after_game_flip();
    return r;
}

typedef s32 (*pad_state_fn)(s32, OrbisPadData*);

// hilo propio: lee el mando del usuario cada ~16 ms con la funcion original
// (sin pasar por nuestros hooks), asi L3+R3 funciona aunque el juego lea el
// mando de una forma que no vemos. Tambien hace el diagnostico periodico.
static void* pad_thread(void* arg)
{
    (void)arg;
    s32 handle = -1;
    int fails = 0, ok_logged = 0, valve_seen = 0, seen = 0;
    uint64_t next_tick = 0, next_try = 0;
    while (g_pad_run) {
        uint64_t t = sceKernelGetProcessTime();
        if (t >= next_tick) {
            nx_engine_tick();
            next_tick = t + 1000000;
            const nx_engine_info_t* inf = nx_engine_info();
            if (inf->state >= NX_ST_WARMUP && !(seen & 2)) { seen |= 2; boot_stage(g_title_s, "pantalla del juego registrada"); }
            if (inf->state == NX_ST_RUN && !(seen & 4)) { seen |= 4; boot_stage(g_title_s, "menu listo"); }
            if (inf->full_ready && !(seen & 8)) { seen |= 8; boot_stage(g_title_s, "efectos activos"); }

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

// Arranque ASCENDENTE por niveles. Cada juego empieza sin que el plugin toque
// nada y, por cada arranque que llega a jugarse, se suma una parte. Si un
// nivel hace cerrar el juego, el siguiente arranque vuelve al ultimo que
// funciono y ese queda como tope: a lo sumo un cierre, y el nivel en el que
// paso dice que parte es la culpable.
//  -1  apagado (hasta el nivel 0 cerro el juego)
//   0  el aviso y los flips solo se cuentan (para saber que llego a jugar)
//   1  + configuracion y motor
//   2  + engancha el registro de pantalla (RegisterBuffers, SetFlipRate)
//   3  + engancha la presentacion (flips): menu L3+R3 y efectos
//   4  + 4 MiB tempranos para el menu
#define BOOT_TOP 4
#define EARLY_WORK 0
static const char* const LEVEL_NAME[BOOT_TOP + 2] = { "apagado", "solo aviso y conteo", "configuracion", "registro de pantalla",
                                                     "menu y efectos", "completo" };
#define LVNAME(l) LEVEL_NAME[(l) + 1]

static void boot_stage(const char* title, const char* what)
{
    char m[160];
    int n = snprintf(m, sizeof m, "[etapa] %s nivel %d: %s\n", title, g_level, what);
    if (n > 0) { append_sync(NX_LOGDIR "/cargas.txt", m, n); trace(m); }
}

static void boot_write(const char* state)
{
    char t[96];
    int k = snprintf(t, sizeof t, "ver=%x level=%d max=%d state=%s\n", (unsigned)g_pluginVersion, g_level, g_level_max, state);
    write_sync(g_boot_path, t, k);
}

static void boot_level_pick(const char* title)
{
    snprintf(g_boot_path, sizeof g_boot_path, NX_LOGDIR "/boot-%.12s.txt", title);
    char prev[96] = { 0 };
    int n = pf_read_file(g_boot_path, prev, sizeof prev - 1);
    int lv = 0, mx = BOOT_TOP, pending = 0, ok = 0;
    // vale el registro de la 2.59 en adelante (desde ahi la pila se realinea);
    // uno mas viejo no sirve y se empieza desde 0
    unsigned pv = 0;
    if (n > 0 && !strncmp(prev, "ver=", 4)) {
        for (const char* q = prev + 4; (*q >= '0' && *q <= '9') || (*q >= 'a' && *q <= 'f'); ++q)
            pv = pv * 16 + (unsigned)(*q <= '9' ? *q - '0' : *q - 'a' + 10);
    }
    if (n > 0 && pv >= 0x259 && pv <= (unsigned)g_pluginVersion) {
        const char* l = strstr(prev, "level=");
        const char* m = strstr(prev, "max=");
        lv = l ? atoi(l + 6) : 0;
        mx = m ? atoi(m + 4) : BOOT_TOP;
        pending = strstr(prev, "state=pending") != NULL;
        ok = strstr(prev, "state=ok") != NULL;
    }
    if (pending && lv <= 0) {   // nivel 0 no toca nada: se cerro por otra cosa (o se salio antes)
        lv = 0;
    } else if (pending) {       // el arranque anterior se cerro en este nivel: el tope queda abajo
        mx = lv - 1;
        lv = mx;
    } else if (ok && lv < mx) {
        lv = mx;                // salio bien: directo al tope (si se cierra, baja de a uno)
    }
    if (mx > BOOT_TOP) mx = BOOT_TOP;
    if (mx < -1) mx = -1;
    if (lv > mx) lv = mx;
    if (lv < -1) lv = -1;
    g_level = lv;
    g_level_max = mx;
    boot_write("pending");
}

static void boot_level_ok(void) { if (g_boot_path[0]) boot_write("ok"); }

// engancha y anota el resultado; si no se puede de forma segura, la funcion
// queda intacta (el plugin sigue sin esa parte, el juego no se toca)
static int hook_one(nx_hook_t* h, void* fn, void* hk, const char* name)
{
    int r = nx_hook_install(h, fn, hk);
    char m[200];
    int n = snprintf(m, sizeof m, "[hook] %s: %s%s%s (fn %p puente %p)\n", name, r == 0 ? "ok" : "NO enganchada: ",
                     r == 0 ? "" : h->why, "", h->target, (void*)h->mem);
    if (n > 0) append_sync(NX_LOGDIR "/cargas.txt", m, n);
    return r;
}

static void early_reserve(const char* title)
{
    pool_init(&g_base, BASE_POOL);
    if (EARLY_WORK) {
        pool_init(&g_work, (size_t)WORK_POOL_MB << 20);
        HOOK32(sceKernelAllocateDirectMemory);   // la valvula (v2.23 engancha la misma funcion)
        g_hooked_dm = 1;
    }
    char m[160];
    int k = snprintf(m, sizeof m, "[mem] reserva temprana: base %s, trabajo %s; libre del sistema %zu MiB\n",
                     g_base.va ? "ok" : "FALLO", g_work.va ? "ok" : "no", raw_avail() >> 20);
    if (k > 0) pf_append_file(NX_LOGDIR "/cargas.txt", m, k);
}

static void hook_flips(void)
{
    HOOK32(sceGnmSubmitAndFlipCommandBuffers);
    HOOK32(sceGnmSubmitAndFlipCommandBuffersForWorkload);
    HOOK32(sceVideoOutSubmitFlip);
    g_hooked = 1;
}

// marca en logs/cargas.txt apenas GoldHEN abre el plugin: si el juego no deja
// ninguna, GoldHEN no lo cargo; si queda module_start sin "cargado en", algo
// fallo en el medio
static void boot_mark(const char* what, const char* title)
{
    char m[96];
    int n = snprintf(m, sizeof m, "[boot] %s %x.%02x %s\n", what, g_pluginVersion >> 8, g_pluginVersion & 0xFF, title);
    if (n > 0) pf_append_file(NX_LOGDIR "/cargas.txt", m, n);
}

s32 attr_public plugin_load(s32 argc, const char* argv[])
{
    trace("[traza] plugin_load: inicio\n");
    struct proc_info info;
    char title[16] = { 0 };
    if (sys_sdk_proc_info(&info) == 0)
        memcpy(title, info.titleid, sizeof title - 1);
    {
        char t[80];
        snprintf(t, sizeof t, "[traza] plugin_load: titulo '%.15s' version %x\n", title, (unsigned)g_pluginVersion);
        trace(t);
    }
    make_dirs();
    memcpy(g_title_s, title, sizeof g_title_s);
    int is_game = !strncmp(title, "CUSA", 4) || !strncmp(title, "PLAS", 4);
    // lo primero de todo: la marca de arranque "pendiente". Si el juego se
    // cierra en cualquier punto de aca en adelante, el proximo arranque usa un
    // nivel mas prudente
    if (!is_game) return 0;
    boot_level_pick(title);
    boot_mark("plugin_load", title);
    if (g_level < 0) {   // hasta el nivel 0 cerro este juego: el plugin no hace nada
        boot_stage(title, "apagado para este juego (Limpiar todo en la app lo vuelve a probar)");
        return 0;
    }
    {   // aviso visible: si no aparece al abrir el juego, GoldHEN no cargo el plugin
        char n[128];
        char w[40];
        if (g_write_err) snprintf(w, sizeof w, "registro ERROR %08X", (unsigned)g_write_err);
        else snprintf(w, sizeof w, "registro ok");
        snprintf(n, sizeof n, "VCFG-NX %x.%02x en %s - nivel %d de %d: %s\n%s \xc2\xb7 traza %s", g_pluginVersion >> 8,
                 g_pluginVersion & 0xFF, title, g_level, g_level_max, LVNAME(g_level), w,
                 g_trace_ok == 1 ? "ok" : g_trace_ok == 0 ? "ERROR" : "-");
        pf_notify(n);
    }
    boot_stage(title, "inicio");
    // nivel 0: los flips solo se cuentan (pasan derecho al sistema) para saber
    // que el juego llego a jugar. Ningun hilo propio aca.
    if (g_level < 1) {
        hook_flips();
        boot_stage(title, "conteo de fotogramas enganchado: plugin_load terminado");
        return 0;
    }
    nx_engine_init(&g_pf, title);
    boot_stage(title, "configuracion leida");
    if (nx_engine_info()->state == NX_ST_OFF || g_level < 2) {
        hook_flips();
        boot_stage(title, "conteo de fotogramas enganchado: plugin_load terminado");
        return 0;
    }
    if (g_level >= 4) {
        early_reserve(title);
        boot_stage(title, "memoria temprana lista");
    }
    HOOK32(sceVideoOutRegisterBuffers);
    HOOK32(sceVideoOutSetFlipRate);
    g_hooked = 1;
    boot_stage(title, "registro de pantalla enganchado");
    // el mando NO se engancha (como VCFG v2.23): lo lee el hilo propio, que
    // arranca recien con el juego presentando imagen (after_game_flip)
    g_engine_live = g_level >= 3;
    hook_flips();
    if (!g_engine_live) {
        boot_stage(title, "conteo de fotogramas enganchado: plugin_load terminado");
        return 0;
    }
    nx_engine_flush_log();
    boot_stage(title, "presentacion enganchada: plugin_load terminado");
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

// la app (Diagnostico) carga el plugin con el argumento "vcfgapp" para
// probarlo: asi la traza distingue esa carga de la del juego. En el juego,
// el aviso sale ya aca, antes de plugin_load: si aparece, GoldHEN abrio el
// plugin dentro del juego aunque despues algo falle.
s32 attr_module_hidden module_start(s64 argc, const void* args)
{
    if (argc >= 7 && args && !memcmp(args, "vcfgapp", 7)) {
        trace("[traza] module_start (prueba de la app)\n");
        return 0;
    }
    trace("[traza] module_start (juego)\n");
    pf_notify(g_trace_ok == 1 ? "VCFG-NX: GoldHEN abri\xc3\xb3 el plugin (traza ok)" : "VCFG-NX: GoldHEN abri\xc3\xb3 el plugin (traza ERROR)");
    return 0;
}
s32 attr_module_hidden module_stop(s64 argc, const void* args) { return 0; }
