// Motor de VCFG-NX (ver nx_engine.h). Reglas de seguridad que sigue:
//
//  * Arranque sin tocar nada: en plugin_load y RegisterBuffers no se reserva
//    memoria. Se espera `warmup_flips` fotogramas (el juego termina de cargar)
//    y recien ahi se mira cuanta memoria directa queda libre.
//  * Memoria: solo se reserva si queda libre lo necesario + `reserve_mb`
//    para el juego. Si no alcanza, los efectos quedan apagados (el menu
//    sigue funcionando) en vez de arriesgar que el juego se quede sin memoria.
//  * Proteccion anti-cierre: antes de activar los efectos se marca la sesion
//    como "pendiente"; tras ~1 minuto estable se marca "ok". Si el juego se
//    cerro con la sesion pendiente, la proxima arranca en modo seguro.
//  * GPU: si nuestro trabajo no termina a tiempo, ese fotograma va intacto y
//    la calidad baja un escalon (NXSR Calidad -> Rapido -> sin SMAA -> sin
//    NXSR -> sin FG). Tras ~1 minuto sin atrasos prueba subir uno; cada subida
//    que falla duplica la espera para la siguiente (hasta ~4 minutos). Solo si ni sin efectos da
//    abasto pasa a "solo pasar fotogramas", y lo reintenta pasados unos minutos.
//  * Temperatura: cada ~2 s se lee la del procesador. Por encima de
//    `temp_limit` (80 C) baja un escalon cada ~15 s; 6 C por encima apaga
//    todos los efectos al instante. Solo vuelve a subir con 6 C de margen.
//  * Re-registro de buffers con la misma geometria (pasa al volver del modo
//    reposo o al cambiar opciones en algunos juegos): se adoptan las nuevas
//    direcciones y se sigue. Con otra geometria, o cualquier cosa inesperada
//    (otro handle, indice fuera de rango, formato HDR) -> el fotograma del
//    juego pasa intacto.
#include "nx_engine.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NX_OSD_FONT_DATA
#include "../include/nx_osd_font.h"
#include "nx_fg8.h"
#include "nx_pipeline.h"

#define MAX_MID 2
#define DCB_DWORDS 6144
#define NOP_DWORDS 64
#define STABLE_FLIPS 1800       // ~1 min a 30 fps con efectos activos -> sesion "ok"
#define STALL_LIMIT 30          // fotogramas seguidos con la GPU atrasada -> degradado
#define WAIT_BUDGET_US 4000     // espera maxima por el trabajo anterior en cada flip
#define RETRY_FLIPS 900         // reintento de reserva de memoria
#define MAX_RETRIES 4
#define DOWN_COOLDOWN 30        // flips entre dos bajadas de calidad seguidas
#define UP_FLIPS 1800           // ~1 min sin atrasos -> probar subir un escalon
#define PROBE_FLIPS 600         // si se atrasa en este lapso tras subir, la subida fallo
#define MAX_UP_FAILS 2          // espera para subir: UP_FLIPS << min(fallos, MAX_UP_FAILS)
#define SLOW_WAIT_US 1500       // esperar mas que esto por la GPU frena al juego...
#define SLOW_LIMIT 15           // ...y si pasa tantos fotogramas seguidos, cuenta como atraso
#define FG8_BUDGET_MS 8.0f      // con la generacion: mas GPU que esto por fotograma le quita al juego
// lo aplicado en el ultimo fotograma (linea de estado del menu)
#define FX_FG_VCFG 1u
#define FX_FG_IA 2u
#define FX_SR 4u
#define FX_AA 8u
#define FX_RCAS 16u
#define FX_EASU 32u
#define DEGRADED_RETRY_FLIPS 5400   // ~3 min en "solo pasar" -> reintentar con lo minimo
#define MAX_DEGRADED_RETRIES 2
#define OSD_COLS 40
#define OSD_ROWS 14
#define TEMP_FLIPS 60           // lectura de temperatura cada ~2 s
#define TEMP_STEP_FLIPS 450     // con calor, un escalon cada ~15 s
#define TEMP_HOT_READS 3        // lecturas seguidas por encima del limite antes de bajar
#define TEMP_CRIT_DELTA 6       // limite + esto: todo apagado ya
#define TEMP_COOL_DELTA 6       // para volver a subir hay que estar este margen por debajo
#define WAIT_STEP_US 250        // sondeo del label mientras se espera a la GPU

#define BTN_L3 0x0002
#define BTN_R3 0x0004
#define BTN_UP 0x0010
#define BTN_RIGHT 0x0020
#define BTN_DOWN 0x0040
#define BTN_LEFT 0x0080
#define BTN_CIRCLE 0x2000
#define BTN_CROSS 0x4000

enum { ROW_ALL, ROW_FG, ROW_SR, ROW_MODEL, ROW_ECO, ROW_AA, ROW_RCAS, ROW_FPS, ROW_CMP, ROW_SAVE, ROW_CLOSE, ROW_COUNT };

static struct {
    const nx_platform_t* pf;
    nx_config_t cfg;
    char title[16];
    nx_engine_info_t info;
    // framebuffers del juego
    int handle;
    uint32_t game_mask;         // indices de VideoOut que registro el juego (puede hacerlo de a uno)
    nx_vo_attr_t attr;
    nx_image_t game_img[16];
    int W, H, Wo, Ho;
    nx_post_t post;
    // recursos propios
    uint8_t* base_mem;          // DCBs + OSD (pocos KiB)
    uint32_t* dcb[2];
    uint32_t* tail[2];          // copia de la "cola de flip" (64 dwords) para los flips extra
    uint8_t* osd_text[2];       // menu, contador
    uint8_t* osd_attr[2];       // color de cada celda (paleta de shaders/osd.cl)
    uint8_t* osd_font;          // atlas suavizado (nx_osd_font.h)
    nx_osd_ctx_t osd;
    volatile uint32_t* label_cpu;   // un solo label para todo nuestro trabajo de GPU
    volatile uint32_t* label2_cpu;  // label del prologo (el trabajo del juego termino)
    uint64_t label2_gpu;
    volatile uint64_t* ts_cpu;      // reloj de la GPU al empezar/terminar nuestro trabajo, por slot
    uint64_t ts_gpu, ts_first, ts_cpu_first;
    float ts_unit;                  // tics de la GPU por microsegundo (calibrado contra el reloj de la CPU)
    uint64_t label_gpu;
    nx_pipeline_t pipe;
    nx_image_t mid_img[MAX_MID], real_img[MAX_MID];
    int mid_index[MAX_MID], real_index[MAX_MID];
    int retries;
    uint64_t next_try_flip;
    // estado por fotograma
    int prev_idx, prev_real_valid, last_fg;
    int slot;
    uint32_t seq, slot_seq[2];
    uint64_t last_flip_us, stable_flips;
    int stall_run, guard_marked;
    int game_rate, rate_forced, nofg_run;
    // mando
    uint64_t combo_since;
    uint32_t last_buttons;
    int crashes;
    // diagnostico: por que cada fotograma paso intacto
    uint64_t why[8];
    uint64_t flips_gnm, flips_wl, flips_cpu;
    char logbuf[12288];
    int loglen, log_flushes;
    char log_path[64];
    int pad_polled, tick_warned, tick_status;
    uint64_t tick_start;             // el hilo de sondeo del mando funciona: los hooks solo filtran
    int nofmt_notified;
    // calidad dinamica
    int pro, announced, hdr;
    int effects_blocked;             // el arranque escalonado todavia no habilito los efectos
    // generacion de fotogramas de VCFG (nx_fg8): 3 intermedios con el formato del juego
    nx_fg8_t fg8;
    int fg8_ready, fg8_failed, fg8_slot, fg8_midk, fg8_first;
    uint32_t fg8_seq, fg8_slotseq[NX_FG8_SLOTS];
    uint64_t fg8_next_try, pace_last_us;
    nx_image_t fg8_mid[3];
    int fg8_mid_index[3];
    uint32_t own_mask;               // indices de VideoOut que registro el plugin
    int fx_cap;                      // tope de efectos (nx_engine_allow_effects)
    int fx_first_job, fx_first_fg, fx_post_first;
    int have_mids;                   // intermedios propios (generacion del camino propio, flips de CPU)
    int ia_fallback_said;            // ya se aviso que la IA paso a VCFG por carga
    unsigned fx_active;              // lo que se aplico en el ultimo fotograma (FX_*), para el menu
    int mem_need_mb, mem_free_mb;    // la ultima reserva de los efectos que no alcanzo
    uint64_t level_since, last_stall, probe_until, degraded_at;
    int up_fails, degraded_gpu, degraded_retries, slow_run;
    // temperatura
    uint64_t next_temp, last_temp_step;
    int hot_reads;
} E;

// registro: va a la consola de depuracion y a /data/vcfg_nx/logs/<TITLE>.log
// (en memoria y escrito de a bloques: nunca un archivo por fotograma)
// el hilo del mando tambien escribe: un candado simple alcanza (es poco y corto)
static volatile int log_lock;
static void lock_log(void) { while (__atomic_exchange_n(&log_lock, 1, __ATOMIC_ACQUIRE)) {} }
static void unlock_log(void) { __atomic_store_n(&log_lock, 0, __ATOMIC_RELEASE); }

static void log_flush_locked(void)
{
    if (!E.loglen || !E.pf || !E.pf->append_file || !E.log_path[0]) { E.loglen = 0; return; }
    if (E.log_flushes < 200) E.pf->append_file(E.log_path, E.logbuf, E.loglen);   // tope ~2 MB por sesion
    E.log_flushes++;
    E.loglen = 0;
}

static void log_flush(void)
{
    lock_log();
    log_flush_locked();
    unlock_log();
}

static void logf_(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void logf_(const char* fmt, ...)
{
    char b[320];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if (n >= (int)sizeof b) n = (int)sizeof b - 1;
    if (E.pf && E.pf->log) E.pf->log(b);
    char t[24];
    int tn = snprintf(t, sizeof t, "[%7llu] ", (unsigned long long)E.info.flips);
    lock_log();
    if (E.loglen + tn + n + 1 > (int)sizeof E.logbuf) log_flush_locked();
    memcpy(E.logbuf + E.loglen, t, (size_t)tn);
    memcpy(E.logbuf + E.loglen + tn, b, (size_t)n);
    E.loglen += tn + n;
    unlock_log();
}

enum { WHY_STATE, WHY_NOBASE, WHY_HANDLE, WHY_INDEX, WHY_COUNT, WHY_NOTHING, WHY_STALL, WHY_TAIL, WHY_N };
static const char* const WHY_NAME[WHY_N] = { "estado", "sin_memoria_menu", "otro_handle", "indice_desconocido",
                                             "count_raro", "nada_que_hacer", "gpu_atrasada", "sin_cola_de_flip" };

// los textos del motor son Latin-1 (los usa el menu); las notificaciones, UTF-8
static void notify(const char* msg)
{
    if (!E.pf || !E.pf->notify) return;
    char u[320];
    size_t o = 0;
    for (const unsigned char* p = (const unsigned char*)msg; *p && o + 3 < sizeof u; ++p) {
        if (*p < 0x80) { u[o++] = (char)*p; continue; }
        u[o++] = (char)(0xC0 | (*p >> 6));
        u[o++] = (char)(0x80 | (*p & 0x3F));
    }
    u[o] = 0;
    E.pf->notify(u);
}

static void set_status(const char* s)
{
    snprintf(E.info.status, sizeof E.info.status, "%s", s);
}

// ------------------------------------------------------- proteccion anti-cierre
static void guard_path(char* p, size_t n) { snprintf(p, n, NX_LOGDIR "/guard-%s.txt", E.title); }

static void guard_write(const char* state)
{
    char p[96], t[96];
    guard_path(p, sizeof p);
    int n = snprintf(t, sizeof t, "state=%s\ncrashes=%d\n", state, E.crashes);
    if (E.pf->write_file) E.pf->write_file(p, t, n);
}

static void guard_check(void)
{
    char p[96], t[128];
    guard_path(p, sizeof p);
    int n = E.pf->read_file ? E.pf->read_file(p, t, sizeof t - 1) : -1;
    if (n <= 0) return;
    t[n] = 0;
    const char* c = strstr(t, "crashes=");
    E.crashes = c ? atoi(c + 8) : 0;
    if (strstr(t, "state=pending")) {
        E.crashes++;
        E.info.safe_mode = 1;
        guard_write("safe");
        logf_("[guard] la sesion anterior se cerro con efectos activos (%d vez/veces): modo seguro\n", E.crashes);
    } else if (strstr(t, "state=safe") && E.crashes >= 3) {
        E.info.safe_mode = 1;   // tres cierres: queda apagado hasta reactivar desde el menu
    }
}

// --------------------------------------------------------------- formatos
static int map_format(int32_t fmt, nx_format_t* out)
{
    switch ((uint32_t)fmt) {
    case 0x80000000u: /* A8R8G8B8_SRGB */
    case 0x80002200u: /* A8B8G8R8_SRGB */
        // UNORM a proposito: los modelos trabajan sobre valores codificados en sRGB
        *out = NX_FMT_RGBA8_UNORM;
        return 1;
    case 0x88060000u: /* A2R10G10B10 (SDR de 10 bits) */
    case 0x88000000u: /* A2R10G10B10_SRGB: igual que arriba, se lee codificado */
        *out = NX_FMT_RGB10A2_UNORM;
        return 1;
    case 0x88740000u: /* A2R10G10B10_BT2020_PQ (HDR): solo el menu, sin efectos */
        *out = NX_FMT_RGB10A2_UNORM;
        return 2;
    default:
        return 0;   // flotante u otros: no se toca
    }
}

// --------------------------------------------------------------- hooks
void nx_engine_init(const nx_platform_t* pf, const char* title_id)
{
    memset(&E, 0, sizeof E);
    E.fx_cap = 3;
    E.pf = pf;
    snprintf(E.title, sizeof E.title, "%s", title_id ? title_id : "");
    E.prev_idx = -1;
    E.game_rate = -1;
    nx_config_load(&E.cfg, E.title, pf->read_file);
    if (E.title[0]) {
        snprintf(E.log_path, sizeof E.log_path, NX_LOGDIR "/%.12s.log", E.title);
        if (pf->write_file) {   // un registro por juego: se reemplaza en cada arranque
            char h[160];
            int n = snprintf(h, sizeof h, "VCFG-NX plugin: arranque de %s\n", E.title);
            pf->write_file(E.log_path, h, n);
        }
    }
    logf_("[engine] plugin_load titulo='%s' config: efectos=%d fg=%d sr=%d modelo=%d aa=%d menu=%dms\n", E.title,
          E.cfg.effects, E.cfg.framegen, E.cfg.sr, E.cfg.sr_model, E.cfg.aa, E.cfg.menu_hold_ms);
    if (strncmp(E.title, "CUSA", 4) != 0 && strncmp(E.title, "PLAS", 4) != 0) {
        E.info.state = NX_ST_OFF;
        set_status("no es un juego");
        return;
    }
    guard_check();
    E.pro = pf->is_pro ? pf->is_pro() != 0 : 0;
    nx_desc_neo = E.pro;
    E.info.temp_c = -1;
    E.info.state = NX_ST_WAIT_BUFFERS;
    set_status(E.info.safe_mode ? "modo seguro" : "esperando el juego");
    logf_("[engine] estado: esperando RegisterBuffers (modo seguro=%d, pro=%d)\n", E.info.safe_mode, E.pro);
    log_flush();
}

static void adopt_buffers(int handle, int start, void* const* addrs, int num, const nx_vo_attr_t* a, nx_format_t fmt)
{
    uint32_t tile = a->tmode == 1 ? NX_TILE_DISPLAY_LINEAR_ALIGNED : NX_TILE_DISPLAY_2D_THIN;
    for (int i = 0; i < num; ++i) {
        nx_image_t im = { (uint64_t)(uintptr_t)addrs[i], a->width, a->height, a->pitch, fmt, tile };
        E.game_img[start + i] = im;
        E.game_mask |= 1u << (start + i);
    }
    E.handle = handle;
    E.attr = *a;
    E.W = (int)a->width;
    E.H = (int)a->height;
}

void nx_engine_on_register(int handle, int start, void* const* addrs, int num, const nx_vo_attr_t* a, int rc)
{
    if (a)
        logf_("[vo] RegisterBuffers handle=%d start=%d num=%d fmt=0x%08x %ux%u pitch=%u tmode=%d rc=0x%08x\n", handle,
              start, num, (unsigned)a->format, a->width, a->height, a->pitch, a->tmode, (unsigned)rc);
    if (rc < 0 || !a || !addrs) return;
    if (E.info.state == NX_ST_OFF) return;
    if (num < 1 || start < 0 || start + num > 16 || (a->width & 1) || (a->height & 1)) return;
    nx_format_t fmt;
    int fk = map_format(a->format, &fmt);
    if (!fk || a->width < 640 || a->height < 360 || a->width > 3840 || a->height > 2160) {
        logf_("[vo] formato 0x%08x %ux%u no soportado: el plugin no toca este juego\n", (unsigned)a->format, a->width,
              a->height);
        // no se apaga del todo: el juego puede registrar despues su pantalla real
        if (!E.game_mask) { E.nofmt_notified = 1; set_status("formato de video no soportado"); }
        log_flush();
        return;
    }
    if (E.game_mask && handle != E.handle) {
        logf_("[vo] otro handle (%d, el del juego es %d): se ignora\n", handle, E.handle);
        return;
    }
    int same = E.game_mask && a->width == E.attr.width && a->height == E.attr.height && a->format == E.attr.format
               && a->pitch == E.attr.pitch && a->tmode == E.attr.tmode;
    if (E.game_mask && same) {
        // mas buffers con la misma geometria (el juego los registra de a uno) o el
        // mismo indice otra vez (vuelta del reposo): se adoptan
        int again = (E.game_mask >> start) & ((1u << num) - 1);
        adopt_buffers(handle, start, addrs, num, a, fmt);
        if (again) {
            E.prev_idx = -1;
            E.prev_real_valid = 0;
            E.pipe.luma_valid[0] = E.pipe.luma_valid[1] = 0;
            E.info.readopts++;
        }
        logf_("[vo] buffers del juego: mascara 0x%04x\n", E.game_mask);
        log_flush();
        return;
    }
    if (E.game_mask && E.info.full_ready) {
        // nuestros buffers de salida tienen la geometria vieja
        logf_("[vo] el juego cambio la geometria de sus buffers: pasa a inactivo\n");
        E.info.state = NX_ST_DEGRADED;
        set_status("el juego cambio sus buffers");
        log_flush();
        return;
    }
    if (E.game_mask) logf_("[vo] otra geometria antes de activar efectos: se vuelve a empezar\n");
    E.game_mask = 0;
    adopt_buffers(handle, start, addrs, num, a, fmt);
    E.hdr = fk == 2;
    E.prev_idx = -1;
    if (E.info.state == NX_ST_WAIT_BUFFERS) {
        E.info.state = NX_ST_WARMUP;
        E.next_try_flip = (uint64_t)E.cfg.warmup_flips;
        if (E.next_try_flip < E.info.flips + 60) E.next_try_flip = E.info.flips + 60;
        set_status(E.hdr ? "HDR: solo el men\xfa" : "juego cargando");
    }
    logf_("[vo] buffers del juego: %dx%d mascara 0x%04x%s\n", E.W, E.H, E.game_mask, E.hdr ? " (HDR: sin efectos)" : "");
    log_flush();
}

int nx_engine_on_set_flip_rate(int handle, int rate)
{
    if (handle == E.handle && !E.rate_forced) E.game_rate = rate;
    if (handle == E.handle && E.rate_forced) return 0;
    return rate;
}

// ------------------------------------------------------------- cola de flip
// sceGnmSubmitAndFlipCommandBuffers exige que el ultimo DCB termine en el
// paquete que deja DrawCommandBuffer::prepareFlip: un NOP de 64 dwords con la
// marca 0x687507xx en el segundo (0x68750777; GTA V usa 0x68750778). El sistema
// lo reescribe al presentar (la escritura de fin de trabajo que espera el flip).
// Igual que VCFG: el juego se envia sin esos 64 dwords, nuestro DCB va despues
// en el MISMO envio y termina con una copia de la cola. Antes nuestro DCB iba
// en un envio aparte sin cola y el sistema escribia sobre sus ultimas
// instrucciones (el label no llegaba: "carga alta" y la GPU colgada).
#define TAIL_HDR 0xC03E1000u   // NOP tipo 3 de 63 dwords de cuerpo
static void make_tail(uint32_t* t)
{
    t[0] = TAIL_HDR;
    t[1] = 0x68750777u;
    for (int i = 2; i < NOP_DWORDS; ++i) t[i] = 0;
}

static const uint32_t* find_tail(const nx_submit_t* s)
{
    if (!s->count || !s->dcb || !s->dcb_sz || !s->dcb[s->count - 1]) return NULL;
    uint32_t nd = s->dcb_sz[s->count - 1] / 4;
    if (nd < NOP_DWORDS) return NULL;
    const uint32_t* t = (const uint32_t*)s->dcb[s->count - 1] + (nd - NOP_DWORDS);
    return t[0] == TAIL_HDR && (t[1] >> 8) == 0x687507u ? t : NULL;
}

// ------------------------------------------------------------- reservas
static int alloc_base(void)
{
    size_t osd_sz = nx_osd_mem_size();
    size_t need = (size_t)(2 * DCB_DWORDS + 2 * NOP_DWORDS) * 4 + 4 * 4096 + sizeof nx_osd_font + osd_sz + 8192;
    // la base son pocos KiB (menu + DCBs): alcanza con dejar 32 MiB libres
    size_t pool = E.pf->pool_free ? E.pf->pool_free(1) : 0;
    if (pool < need && E.pf->avail_direct() < need + 32u * 1048576) return 0;
    E.base_mem = E.pf->alloc_gpu(need, 65536);
    if (!E.base_mem) return 0;
    uint8_t* p = E.base_mem;
#define ALIGN256(q) (uint8_t*)(((uintptr_t)(q) + 255) & ~(uintptr_t)255)
    E.dcb[0] = (uint32_t*)p; p += DCB_DWORDS * 4;
    E.dcb[1] = (uint32_t*)p; p += DCB_DWORDS * 4;
    for (int k = 0; k < 2; ++k) {
        E.tail[k] = (uint32_t*)p; p += NOP_DWORDS * 4;
        make_tail(E.tail[k]);
    }
    p = ALIGN256(p);
    E.label_cpu = (volatile uint32_t*)p; E.label_gpu = (uint64_t)(uintptr_t)p; *E.label_cpu = 0;
    E.label2_cpu = (volatile uint32_t*)(p + 64); E.label2_gpu = E.label_gpu + 64; *E.label2_cpu = 0;
    E.ts_cpu = (volatile uint64_t*)(p + 128); E.ts_gpu = E.label_gpu + 128;   // [slot][inicio, fin]
    for (int i = 0; i < 4; ++i) E.ts_cpu[i] = 0;
    p += 256;
    E.osd_text[0] = p; p += 4096;
    E.info.menu_text = E.osd_text[0];
    E.info.menu_cols = OSD_COLS;
    E.info.menu_rows = ROW_COUNT + 3;
    E.osd_text[1] = p; p += 4096;
    E.osd_attr[0] = p; p += 4096;
    E.info.menu_attr = E.osd_attr[0];
    E.osd_attr[1] = p; p += 4096;
    p = ALIGN256(p);
    E.osd_font = p; p += sizeof nx_osd_font;
    memcpy(E.osd_font, nx_osd_font, sizeof nx_osd_font);
    p = ALIGN256(p);
    if (nx_osd_init(&E.osd, p, (uint64_t)(uintptr_t)p) != 0) return 0;
    return 1;
}

void nx_engine_allow_effects(int cap)
{
    E.fx_cap = cap < 0 ? 0 : cap > 3 ? 3 : cap;
    E.effects_blocked = E.fx_cap == 0;
}

// hito al instante en la traza del plugin (si el juego se cuelga justo despues,
// queda dicho hasta donde se llego) y en el registro
static void stage(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void stage(const char* fmt, ...)
{
    char m[160];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(m, sizeof m, fmt, ap);
    va_end(ap);
    logf_("[etapa] %s\n", m);
    log_flush();
    if (E.pf && E.pf->stage) E.pf->stage(m);
}

static int effects_wanted(void)
{
    return !E.hdr && !E.effects_blocked && E.cfg.effects && (E.cfg.framegen || E.cfg.sr || E.cfg.aa || E.cfg.rcas);
}

static int alloc_full(void)
{
    E.post = E.cfg.sr || E.cfg.rcas ? NX_POST_RCAS : NX_POST_NONE;
    E.Wo = E.W; E.Ho = E.H;
    if (E.cfg.out_w > E.W && E.cfg.out_h > E.H) {  // el juego entrega menos que la salida: EASU
        E.Wo = E.cfg.out_w; E.Ho = E.cfg.out_h; E.post = NX_POST_EASU;
    }
    size_t arena_sz = nx_pipeline_mem_size(E.W, E.H, E.Wo, E.Ho);
    size_t pitch_o = (size_t)((E.Wo + 63) / 64 * 64);
    size_t buf = (pitch_o * E.Ho * 4 + 0xFFFF) & ~(size_t)0xFFFF;
    // buffers de salida: los intermedios propios solo los usa la generacion del camino
    // propio (juegos con flip de CPU; los demas usan la presentacion de VCFG, con sus
    // intermedios); el real procesado vuelve al buffer del juego, asi que basta uno
    // sin registrar, salvo con EASU (salida mas grande: se presenta el propio)
    int nmid = E.flips_cpu ? MAX_MID : 0;
    int easu = E.Wo != E.W || E.Ho != E.H;
    int nreal = easu || nmid ? MAX_MID : 1;   // con generacion propia el anterior procesado se reusa
    int nreg = nmid + (easu ? MAX_MID : 0);
    size_t total = arena_sz + (size_t)(nmid + nreal) * buf;
    // nuestros indices: el bloque libre mas alto (el juego suele usar los primeros)
    int extra_first = -1;
    for (int s0 = 16 - nreg; nreg && s0 >= 0 && extra_first < 0; --s0)
        if (!(((E.game_mask | E.own_mask) >> s0) & ((1u << nreg) - 1))) extra_first = s0;
    if (nreg && extra_first < 0) {
        set_status("sin indices libres de VideoOut");
        logf_("[mem] sin indices libres (mascara del juego 0x%04x)\n", E.game_mask);
        return -1;
    }
    size_t avail = E.pf->avail_direct();
    size_t pool = E.pf->pool_free ? E.pf->pool_free(0) : 0;
    // con la reserva temprana alcanza: es nuestra, no se le quita nada al juego
    if (pool < total + 2 * 65536 && avail < total + (size_t)E.cfg.reserve_mb * 1048576) {
        logf_("[mem] hacen falta %zu MiB + %d de reserva, libres %zu MiB (reserva propia %zu MiB): se reintenta mas tarde\n",
              total >> 20, E.cfg.reserve_mb, avail >> 20, pool >> 20);
        E.info.alloc_fail++;
        E.mem_need_mb = (int)(total >> 20) + E.cfg.reserve_mb;
        E.mem_free_mb = (int)(avail >> 20);
        return 0;
    }
    uint8_t* arena = E.pf->alloc_gpu(arena_sz, 65536);
    uint8_t* outs = arena ? E.pf->alloc_gpu((size_t)(nmid + nreal) * buf, 65536) : NULL;
    if (!arena || !outs) { E.info.alloc_fail++; logf_("[mem] la reserva fallo (arena %p, salidas %p)\n", (void*)arena, (void*)outs); return 0; }
    nx_arena_t a = { arena, (uint64_t)(uintptr_t)arena, arena_sz, 0 };
    if (nx_pipeline_init(&E.pipe, E.W, E.H, E.Wo, E.Ho, a) != 0) { set_status("error interno (pipeline)"); return -1; }
    E.pipe.label_gpu = E.label_gpu;   // el mismo label que el OSD: la secuencia nunca retrocede
    E.pipe.label_cpu = E.label_cpu;
    void* addrs[2 * MAX_MID];
    for (int i = 0; i < nmid + nreal; ++i) addrs[i] = outs + buf * (size_t)i;
    if (nreg) {
        nx_vo_attr_t oa = E.attr;
        oa.width = (uint32_t)E.Wo; oa.height = (uint32_t)E.Ho; oa.pitch = (uint32_t)pitch_o;
        oa.tmode = 1;   // nuestros kernels escriben en lineal
        int rc = E.pf->register_buffers(E.handle, extra_first, addrs, nreg, &oa);
        if (rc < 0) { logf_("[engine] RegisterBuffers propio = 0x%08x\n", (unsigned)rc); set_status("VideoOut rechazo los buffers"); return -1; }
        E.own_mask |= ((1u << nreg) - 1) << extra_first;
    }
    nx_format_t fmt = E.game_img[__builtin_ctz(E.game_mask)].format;
    for (int i = 0; i < MAX_MID; ++i) {
        nx_image_t im = { 0, (uint32_t)E.Wo, (uint32_t)E.Ho, (uint32_t)pitch_o, fmt, NX_TILE_DISPLAY_LINEAR_ALIGNED };
        im.gpu_addr = nmid ? (uint64_t)(uintptr_t)addrs[i] : 0;
        E.mid_img[i] = im; E.mid_index[i] = nmid ? extra_first + i : -1;
        im.gpu_addr = (uint64_t)(uintptr_t)addrs[nmid + (nreal > 1 ? i : 0)];
        E.real_img[i] = im; E.real_index[i] = easu ? extra_first + nmid + i : -1;
    }
    E.have_mids = nmid > 0;
    E.info.full_ready = 1;
    E.mem_need_mb = 0;
    logf_("[engine] listo: %dx%d -> %dx%d, %zu MiB\n", E.W, E.H, E.Wo, E.Ho, total >> 20);
    stage("efectos: memoria lista %dx%d, %zu MiB, buffers propios desde el indice %d", E.Wo, E.Ho, total >> 20, extra_first);
    return 1;
}

// lo que necesita la memoria grande de los efectos (alloc_full): reescalado, SMAA,
// nitidez o la generacion con IA. La generacion de VCFG tiene la suya (alloc_fg8).
static int full_wanted(void)
{
    if (E.hdr || E.effects_blocked || !E.cfg.effects) return 0;
    if (E.cfg.framegen == 2) return 1;
    // con la generacion de VCFG la memoria grande solo hace falta si hay post-proceso
    // (desde la 2.73 se combina: el real se procesa antes de generar el intermedio)
    return E.cfg.sr || E.cfg.aa || E.cfg.rcas;
}

static int fg8_wanted(void)
{
    // la presentacion (intermedios, colas, slots) la usan las dos generaciones
    return !E.hdr && !E.effects_blocked && E.fx_cap >= 3 && E.cfg.effects && E.cfg.framegen >= 1;
}

// generacion de VCFG: tablas/DCBs (memoria compartida), trabajo, y 3 intermedios
// con el mismo formato, tiling y pitch que los buffers del juego (como VCFG)
static int alloc_fg8(void)
{
    if (!E.game_mask) return 0;
    const nx_image_t* g = &E.game_img[__builtin_ctz(E.game_mask)];
    if (g->width != 1920 && g->width != 1280 && g->width != 1600 && g->width != 2560 && g->width != 3840) {
        stage("generacion VCFG: resolucion %ux%u no soportada", g->width, g->height);
        return -1;
    }
    size_t one = (size_t)g->pitch * ((g->height + 63u) & ~63u) * 4;
    one = (one + 0x7FFFF) & ~(size_t)0x7FFFF;   // 512 KiB, como VCFG
    size_t total = NX_FG8_SLAB + NX_FG8_WORK + 3 * one;
    size_t avail = E.pf->avail_direct();
    size_t pool = E.pf->pool_free ? E.pf->pool_free(0) : 0;
    // el mismo margen que los demas efectos (con reserva temprana propia alcanza)
    if (pool < total + 2 * 65536 && avail < total + (size_t)E.cfg.reserve_mb * 1048576) {
        logf_("[fg8] hacen falta %zu MiB + %d de margen, libres %zu MiB: se reintenta\n", total >> 20, E.cfg.reserve_mb,
              avail >> 20);
        return 0;
    }
    int first = -1;
    for (int s0 = 13; s0 >= 0 && first < 0; --s0)
        if (!(((E.game_mask | E.own_mask) >> s0) & 7u)) first = s0;
    if (first < 0) { stage("generacion VCFG: sin indices libres de VideoOut (0x%04x)", E.game_mask | E.own_mask); return -1; }
    uint8_t* slab = E.pf->alloc_shared ? E.pf->alloc_shared(NX_FG8_SLAB, 65536) : E.pf->alloc_gpu(NX_FG8_SLAB, 65536);
    uint8_t* work = slab ? E.pf->alloc_gpu(NX_FG8_WORK, 65536) : NULL;
    uint8_t* mids = work ? E.pf->alloc_gpu(3 * one, 65536) : NULL;
    if (!mids) { logf_("[fg8] la reserva fallo (%p %p %p)\n", (void*)slab, (void*)work, (void*)mids); return 0; }
    if (nx_fg8_init(&E.fg8, slab, work) != 0) { stage("generacion VCFG: los shaders no entran"); return -1; }
    void* addrs[3];
    for (int i = 0; i < 3; ++i) {
        addrs[i] = mids + one * (size_t)i;
        E.fg8_mid[i] = *g;
        E.fg8_mid[i].gpu_addr = (uint64_t)(uintptr_t)addrs[i];
        E.fg8_mid_index[i] = first + i;
    }
    int rc = E.pf->register_buffers(E.handle, first, addrs, 3, &E.attr);
    if (rc < 0) { stage("generacion VCFG: VideoOut rechazo los intermedios (0x%08x)", (unsigned)rc); return -1; }
    E.own_mask |= 7u << first;
    E.fg8_ready = 1;
    stage("generacion VCFG: lista, 3 intermedios %ux%u desde el indice %d (%zu MiB)", g->width, g->height, first,
          total >> 20);
    return 1;
}

static void try_allocations(void)
{
    if (!E.base_mem && !alloc_base()) {
        set_status("sin memoria para el menu");
        return;
    }
    if (E.info.state == NX_ST_WARMUP) {
        E.info.state = NX_ST_RUN;
        set_status("listo");
        if (!E.announced++)
            notify(E.info.safe_mode ? "VCFG-NX en modo seguro (el juego se cerro la ultima vez). L3+R3: menu"
                                : "VCFG-NX listo. Mantener L3+R3: menu");
    }
    if (!E.info.safe_mode && fg8_wanted() && !E.fg8_ready && !E.fg8_failed && E.info.flips >= E.fg8_next_try) {
        int r8 = alloc_fg8();
        if (r8 < 0) E.fg8_failed = 1;
        else if (r8 == 0) E.fg8_next_try = E.info.flips + RETRY_FLIPS;
        else set_status("activo");
    }
    if (E.info.full_ready || E.info.safe_mode || !full_wanted() || E.retries >= MAX_RETRIES) return;
    int r = alloc_full();
    if (r == 1) {
        set_status("activo");
    } else if (r == 0) {
        E.retries++;
        E.next_try_flip = E.info.flips + RETRY_FLIPS;
        set_status(E.retries >= MAX_RETRIES ? "memoria insuficiente: efectos apagados" : "esperando memoria libre");
        if (E.retries >= MAX_RETRIES) notify("VCFG-NX: el juego usa casi toda la memoria; efectos apagados");
    } else {
        E.retries = MAX_RETRIES;
    }
}

// -------------------------------------------------------------- menu (OSD)

// atributos (colores) de la paleta de shaders/osd.cl
enum { A_TEXT, A_DIM, A_ON, A_OFF, A_ACCENT, A_WARN, A_TITLE, A_ERROR };

static void osd_clear(uint8_t* t, uint8_t* a, int row)
{
    memset(t + row * OSD_COLS, ' ', OSD_COLS);
    memset(a + row * OSD_COLS, A_TEXT, OSD_COLS);
}

static void osd_put(uint8_t* t, uint8_t* a, int row, int col, int attr, const char* s)
{
    for (int i = 0; s[i] && col + i < OSD_COLS; ++i) {
        if (col + i < 0) continue;
        t[row * OSD_COLS + col + i] = (uint8_t)s[i];
        a[row * OSD_COLS + col + i] = (uint8_t)attr;
    }
}

// fila "etiqueta ........ valor" con el valor alineado a la derecha
static void osd_item(uint8_t* t, uint8_t* a, int row, const char* label, const char* value, int vattr, int dim)
{
    osd_clear(t, a, row);
    osd_put(t, a, row, 1, dim ? A_DIM : A_TEXT, label);
    if (value) osd_put(t, a, row, OSD_COLS - 1 - (int)strlen(value), dim && vattr != A_ERROR ? A_OFF : vattr, value);
}

static void osd_switch(uint8_t* t, uint8_t* a, int row, const char* label, int on, int dim)
{
    osd_item(t, a, row, label, on ? "ON" : "OFF", on ? A_ON : A_OFF, dim);
}

static void build_menu_text(void)
{
    static const char* sr_names[] = { "OFF", "Suave", "Normal", "Fuerte" };
    static const char* rc_names[] = { "OFF", "Baja", "Media", "Alta" };
    uint8_t *t = E.osd_text[0], *a = E.osd_attr[0];
    char l[64];
    // titulo + fps/temperatura
    osd_clear(t, a, 0);
    osd_put(t, a, 0, 1, A_TITLE, "VCFG-NX");
    char g[40] = "";
    if (E.info.gpu_ms > 0.0f) snprintf(g, sizeof g, "  GPU %.1f ms", (double)E.info.gpu_ms);
    // tasa de salida: 60 Hz forzada (generacion) o la que pidio el juego (1 = 30 Hz)
    int hz = E.rate_forced || E.game_rate <= 0 ? 60 : E.game_rate == 1 ? 30 : 20;
    snprintf(g + strlen(g), sizeof g - strlen(g), "  %dHz", hz);
    if (E.info.temp_c > 0) snprintf(l, sizeof l, "%.0f fps%s  %d\xb0" "C", E.info.game_fps, g, E.info.temp_c);
    else snprintf(l, sizeof l, "%.0f fps%s", E.info.game_fps, g);
    osd_put(t, a, 0, OSD_COLS - 1 - (int)strlen(l), A_DIM, l);
    // estado
    osd_clear(t, a, 1);
    int warn = E.info.safe_mode || E.info.hot || E.info.level > 0 || E.info.state == NX_ST_DEGRADED || !E.info.full_ready;
    if (E.effects_blocked) snprintf(l, sizeof l, "solo men\xfa (efectos: pr\xf3x. arranque)");
    else if (E.fx_cap < 3 && E.cfg.effects && !E.info.hot && !E.info.level)
        snprintf(l, sizeof l, "%.20s \xb7 parte %d de 3", E.info.status, E.fx_cap);
    else if (!E.cfg.effects) snprintf(l, sizeof l, "efectos apagados");
    else if (E.info.hot) snprintf(l, sizeof l, "%s: consola caliente (-%d)", E.info.hot == 2 ? "pausa" : "ahorro", E.info.level);
    else if ((E.cfg.sr || E.cfg.aa || E.cfg.rcas) && !E.info.full_ready && E.mem_need_mb)
        snprintf(l, sizeof l, "efectos: falta memoria (%d de %d MiB)", E.mem_free_mb, E.mem_need_mb);
    else if (E.info.state == NX_ST_RUN && E.fx_active) {
        unsigned f = E.fx_active;
        int n = snprintf(l, sizeof l, "%s", E.info.level > 0 ? "carga: " : "activo: ");
        const char* sep = "";
        if (f & (FX_FG_VCFG | FX_FG_IA)) { n += snprintf(l + n, sizeof l - (size_t)n, "%s", f & FX_FG_IA ? "IA" : "VCFG"); sep = "+"; }
        if (f & FX_EASU) { n += snprintf(l + n, sizeof l - (size_t)n, "%sEASU", sep); sep = "+"; }
        if (f & FX_SR) { n += snprintf(l + n, sizeof l - (size_t)n, "%sNXSR%s", sep, E.info.lite == 2 ? "-LC" : E.info.lite ? "-L" : ""); sep = "+"; }
        if (f & FX_AA) { n += snprintf(l + n, sizeof l - (size_t)n, "%sSMAA", sep); sep = "+"; }
        if (f & FX_RCAS) { n += snprintf(l + n, sizeof l - (size_t)n, "%sFSR", sep); sep = "+"; }
        if (E.info.level > 0 && n < (int)sizeof l - 6) snprintf(l + n, sizeof l - (size_t)n, " (-%d)", E.info.level);
    }
    else if (E.info.level > 0) snprintf(l, sizeof l, "%.26s (carga -%d)", E.info.status, E.info.level);
    else snprintf(l, sizeof l, "%.36s", E.info.status);
    osd_put(t, a, 1, 1, !E.cfg.effects ? A_OFF : (warn || E.effects_blocked ? A_WARN : A_ACCENT), l);
    int dim = !E.cfg.effects || E.effects_blocked;
    // desde la 2.73 el post-proceso tambien se combina con la generacion
    int pdim = dim;
    // interruptor general
    osd_item(t, a, 2 + ROW_ALL, "Todos los efectos", E.cfg.effects ? "ON" : "OFF", E.cfg.effects ? A_ON : A_ERROR, 0);
    osd_item(t, a, 2 + ROW_FG, "Generaci\xf3n de fotogramas",
             E.cfg.framegen == 2 ? "IA NXFG-2" : E.cfg.framegen ? "VCFG" : "OFF", E.cfg.framegen ? A_ON : A_OFF, dim);
    osd_item(t, a, 2 + ROW_SR, "Reescalado IA (NXSR)", sr_names[E.cfg.sr & 3], E.cfg.sr ? A_ACCENT : A_OFF, pdim);
    // modelo: lo que eligio el usuario y lo que se usa de verdad
    const char* mv;
    if (E.cfg.eco) mv = "Ligero (ahorro)";
    else if (E.cfg.sr_model == 2) mv = "Calidad";
    else if (E.cfg.sr_model == 1) mv = "Cl\xe1sico";
    else mv = "Ligero";
    osd_item(t, a, 2 + ROW_MODEL, "Modelo IA", mv, A_ACCENT, pdim || !E.cfg.sr);
    osd_switch(t, a, 2 + ROW_ECO, "Modo ahorro (menos calor)", E.cfg.eco, dim);
    osd_switch(t, a, 2 + ROW_AA, "Antialiasing SMAA", E.cfg.aa, pdim);
    osd_item(t, a, 2 + ROW_RCAS, "Nitidez FSR+ (RCAS)", rc_names[E.cfg.rcas & 3], E.cfg.rcas ? A_ACCENT : A_OFF,
             pdim);
    osd_switch(t, a, 2 + ROW_FPS, "Contador de FPS", E.cfg.fps_counter, 0);
    osd_switch(t, a, 2 + ROW_CMP, "Comparar (izq. original)", E.cfg.compare, dim);
    osd_item(t, a, 2 + ROW_SAVE, "Guardar para este juego", NULL, 0, 0);
    osd_item(t, a, 2 + ROW_CLOSE, "Cerrar men\xfa", NULL, 0, 0);
    osd_clear(t, a, 2 + ROW_COUNT);
    osd_put(t, a, 2 + ROW_COUNT, 1, A_DIM, "\xab \xbb cambiar   X aceptar   O cerrar");
    // fila elegida: sobre la pastilla azul, lo azul y lo gris se leen mejor en blanco
    uint8_t* ra = a + (2 + E.info.menu_row) * OSD_COLS;
    for (int i = 0; i < OSD_COLS; ++i)
        if (ra[i] == A_ACCENT || ra[i] == A_DIM || ra[i] == A_OFF) ra[i] = A_TITLE;
}

static void build_fps_text(int fg_on)
{
    char l[64];
    uint8_t *t = E.osd_text[1], *a = E.osd_attr[1];
    osd_clear(t, a, 0);
    snprintf(l, sizeof l, "%.0f fps", E.info.game_fps * (fg_on ? 2.0f : 1.0f));
    int x = (int)strlen(l);
    osd_put(t, a, 0, 0, A_TITLE, l);
    if (fg_on) { osd_put(t, a, 0, x + 1, A_ON, "FG"); x += 3; }
    if (E.info.temp_c > 0) {
        snprintf(l, sizeof l, "%d\xb0" "C", E.info.temp_c);
        osd_put(t, a, 0, x + 1, E.info.hot ? A_WARN : A_DIM, l);
    }
}

static void menu_change(int row, int dir)
{
    nx_config_t* c = &E.cfg;
    switch (row) {
    case ROW_ALL:
        c->effects = !c->effects;
        if (c->effects && !E.info.full_ready) E.next_try_flip = E.info.flips;   // reservar ya si hace falta
        break;
    case ROW_FG: c->framegen = (c->framegen + (dir > 0 ? 1 : 2)) % 3; break;   // OFF, Rapida, IA
    case ROW_SR: c->sr = (c->sr + (dir > 0 ? 1 : 3)) % 4; break;
    case ROW_MODEL: c->sr_model = (c->sr_model + (dir > 0 ? 1 : 2)) % 3; break;
    case ROW_ECO: c->eco = !c->eco; break;
    case ROW_AA: c->aa = !c->aa; break;
    case ROW_RCAS: c->rcas = (c->rcas + (dir > 0 ? 1 : 3)) % 4; break;
    case ROW_FPS: c->fps_counter = !c->fps_counter; break;
    case ROW_CMP: c->compare = !c->compare; break;
    default: break;
    }
    // el usuario eligio otra cosa: la calidad dinamica vuelve a empezar desde
    // arriba (salvo que la baja sea por temperatura)
    if (!E.info.hot) E.info.level = 0;
    E.info.level_locked = 0;
    E.up_fails = 0;
    E.level_since = E.info.flips;
    if (E.info.safe_mode && effects_wanted()) {
        // el usuario reactiva a mano: salimos del modo seguro
        E.info.safe_mode = 0;
        E.crashes = 0;
        E.retries = 0;
    }
}

// por que no abre el menu: se avisa una vez por pulsacion
static void menu_unavailable(void)
{
    const char* why;
    switch (E.info.state) {
    case NX_ST_OFF: why = E.info.status[0] ? E.info.status : "este juego no es compatible"; break;
    case NX_ST_WAIT_BUFFERS:
        why = E.nofmt_notified ? "formato de video no soportado" : "el juego todavia no registro su pantalla";
        break;
    case NX_ST_WARMUP:
        why = !strcmp(E.info.status, "sin memoria para el menu") ? "el juego ocupa toda la memoria; no queda ni para el menu"
                                                                 : "el juego esta cargando, proba en unos segundos";
        break;
    default: why = "sin memoria para el menu"; break;
    }
    char m[160];
    snprintf(m, sizeof m, "VCFG-NX: menu no disponible (%s)", why);
    notify(m);
    logf_("[pad] L3+R3 sin menu: estado=%d %s\n", (int)E.info.state, why);
    log_flush();
}

// logica del mando: combo L3+R3 y navegacion. La llama el hilo de sondeo del
// plugin (si funciona) o, si no, los hooks de lectura del juego.
void nx_engine_pad_logic(uint32_t b)
{
    if (!E.pf) return;
    uint64_t t = E.pf->now_us();
    int combo = (b & (BTN_L3 | BTN_R3)) == (BTN_L3 | BTN_R3);
    if (!combo) {
        E.combo_since = 0;
    } else if (!E.combo_since) {
        E.combo_since = t;
    } else if (E.combo_since != 1 && t - E.combo_since > (uint64_t)E.cfg.menu_hold_ms * 1000) {
        E.combo_since = 1;
        if (E.info.menu_open) {
            E.info.menu_open = 0;
        } else if (E.base_mem && (E.info.state == NX_ST_RUN || E.info.state == NX_ST_DEGRADED)) {
            E.info.menu_open = 1;
            E.info.menu_row = 0;
            logf_("[pad] menu abierto\n");
        } else {
            menu_unavailable();
        }
    }
    if (E.info.menu_open) {
        uint32_t pressed = b & ~E.last_buttons;
        if (pressed & BTN_UP) E.info.menu_row = (E.info.menu_row + ROW_COUNT - 1) % ROW_COUNT;
        if (pressed & BTN_DOWN) E.info.menu_row = (E.info.menu_row + 1) % ROW_COUNT;
        if (pressed & (BTN_LEFT | BTN_RIGHT)) menu_change(E.info.menu_row, (pressed & BTN_RIGHT) ? 1 : -1);
        if (pressed & BTN_CROSS) {
            if (E.info.menu_row == ROW_SAVE) {
                int ok = nx_config_save(&E.cfg, E.title, E.pf->write_file) > 0;
                notify(ok ? "VCFG-NX: ajustes guardados para este juego" : "VCFG-NX: no se pudo guardar");
            } else if (E.info.menu_row == ROW_CLOSE) {
                E.info.menu_open = 0;
            } else {
                menu_change(E.info.menu_row, 1);
            }
        }
        if (pressed & BTN_CIRCLE) E.info.menu_open = 0;
    }
    E.last_buttons = b;
}

void nx_engine_pad_polled(int on)
{
    if (on != E.pad_polled) logf_("[pad] sondeo propio del mando: %s\n", on ? "si" : "no (se usan los hooks)");
    E.pad_polled = on;
}

void nx_engine_on_pad(uint32_t* buttons, uint8_t sticks[4], uint8_t triggers[2])
{
    if (!buttons || !E.pf) return;
    if (!E.pad_polled) nx_engine_pad_logic(*buttons);
    if (E.info.menu_open) {
        // el juego no recibe nada mientras el menu esta abierto
        *buttons = 0;
        if (sticks) sticks[0] = sticks[1] = sticks[2] = sticks[3] = 0x80;
        if (triggers) triggers[0] = triggers[1] = 0;
    }
}

// ------------------------------------------------------------- flips
static int passthrough_why(const nx_submit_t* s, int why)
{
    E.info.passthrough++;
    E.why[why]++;
    return E.pf->submit_flip(s);
}

static int flip_index(const nx_submit_t* game, int index, uint32_t mode)
{
    void* d[1] = { E.tail[E.slot] };
    uint32_t sz[1] = { NOP_DWORDS * 4 };
    nx_submit_t f = { 1, d, sz, NULL, NULL, game->handle, (uint32_t)index, mode, game->arg, 0, 0, 0 };
    return E.pf->submit_flip(&f);
}

static void set_rate_forced(int on)
{
    if (on) E.nofg_run = 0;
    if (on == E.rate_forced || !E.pf->set_flip_rate) return;
    E.rate_forced = on;
    int r = E.pf->set_flip_rate(E.handle, on ? 0 : (E.game_rate >= 0 ? E.game_rate : 0));
    logf_("[flip] tasa de salida %s (juego pidio %d) = 0x%08x\n", on ? "60 Hz (forzada)" : "la del juego", E.game_rate,
          (unsigned)r);
}

// un fotograma sin generacion: la salida a 60 Hz se suelta recien tras ~2 s
// seguidos sin ella. Cambiar la tasa en cada fotograma (pasaba con cada atraso
// de la GPU) hacia saltar la salida entre 30 y 60 Hz: tirones y menos fps.
static void rate_idle(void)
{
    if (E.rate_forced && ++E.nofg_run < 120) return;
    set_rate_forced(0);
}

const nx_engine_info_t* nx_engine_info(void) { return &E.info; }
void nx_engine_flush_log(void) { log_flush(); }

void nx_engine_note(const char* line)
{
    logf_("%s\n", line);
    log_flush();
}

// lo llama el hilo del plugin ~1 vez por segundo: deja constancia si el juego
// nunca llega a los hooks (otra API de presentacion, registro antes de cargar...)
void nx_engine_tick(void)
{
    if (!E.pf || E.info.state == NX_ST_OFF) return;
    uint64_t t = E.pf->now_us();
    if (!E.tick_start) E.tick_start = t;
    uint64_t secs = (t - E.tick_start) / 1000000;
    // a los 25 s, un aviso en pantalla con el estado: sirve para saber que paso
    // sin tener que buscar el registro
    if (!E.tick_status && secs >= 25) {
        E.tick_status = 1;
        char m[200];
        if (E.info.state == NX_ST_WAIT_BUFFERS)
            snprintf(m, sizeof m, "VCFG-NX: %s (fotogramas vistos: %llu)",
                     E.nofmt_notified ? "formato de video no soportado" : "no vio la pantalla del juego",
                     (unsigned long long)E.info.flips);
        else if (!E.info.flips)
            snprintf(m, sizeof m, "VCFG-NX: el juego no paso por los hooks de presentacion");
        else
            snprintf(m, sizeof m, "VCFG-NX: %s \xb7 %dx%d \xb7 %.0f fps \xb7 L3+R3: men\xfa", E.info.status, E.W, E.H,
                     (double)E.info.game_fps);
        notify(m);
        logf_("[diag] aviso a los 25 s: %s\n", m);
        log_flush();
    }
    if (!E.tick_warned && secs >= 30) {
        if (!E.info.flips) {
            E.tick_warned = 1;
            logf_("[diag] 30 s sin ningun flip por los hooks (gnm/workload/cpu): el juego presenta por otra via\n");
            log_flush();
        } else if (E.info.state == NX_ST_WAIT_BUFFERS) {
            E.tick_warned = 1;
            logf_("[diag] %llu flips pero ningun RegisterBuffers visto: el juego registro antes de cargar el plugin\n",
                  (unsigned long long)E.info.flips);
            log_flush();
        }
    }
}

// ------------------------------------------------------- calidad dinamica
// escalones, del mas caro al mas barato; solo cuentan los que estan en uso
// modelo de reescalado (2.75): 0 = NXSR-L Ligero (por defecto y en modo ahorro),
// 1 = Clasico (NXSR Rapido, el primero), 2 = NXSR-L Calidad (guia mas grande, el mismo
// costo a resolucion completa: se combina con la generacion). El NXSR Calidad viejo
// (~20 ms de GPU a 1080p) ya no se usa.
static void sr_choice(int sr, int* hq, int* lite)
{
    *hq = 0;
    *lite = !sr ? 0 : (E.cfg.eco || E.cfg.sr_model == 0) ? 1 : E.cfg.sr_model == 2 ? 2 : 0;
}

// escalones de la calidad dinamica, de lo mas pesado a lo mas liviano: primero el
// modelo pesado pasa a NXSR-L, despues SMAA, el reescalado, la nitidez y la generacion
// ia: generacion IA con la presentacion de VCFG; su escalon la cambia por la de VCFG
// (nunca se apaga la generacion en ese camino: fg = NULL)
static void quality_ladder(int* fg, int* ia, int* sr, int* hq, int* lite, int* aa, int* rcas)
{
    int steps = 0, lv = E.info.level;   // los escalones por atraso solo se suman con `dynamic`; los de calor siempre
    if (*sr && *lite != 1) { if (lv > steps) { *hq = 0; *lite = 1; } steps++; }
    if (*aa) { if (lv > steps) *aa = 0; steps++; }
    if (*sr) { if (lv > steps) *sr = *hq = *lite = 0; steps++; }
    if (*rcas) { if (lv > steps) *rcas = 0; steps++; }
    if (ia && *ia) { if (lv > steps) *ia = 0; steps++; }
    if (fg && *fg) { if (lv > steps) *fg = 0; steps++; }
    E.info.level_max = steps;
    if (E.info.level > steps) E.info.level = steps;
}

static void on_stall(void)
{
    uint64_t f = E.info.flips;
    E.last_stall = f;
    if (E.probe_until && f < E.probe_until) {   // acababa de subir: no alcanzo
        E.probe_until = 0;
        ++E.up_fails;
        E.info.level_locked = E.up_fails >= MAX_UP_FAILS;
        logf_("[engine] calidad dinamica: la subida no se sostuvo (%d)\n", E.up_fails);
    }
    if (!E.cfg.dynamic || E.info.level >= E.info.level_max || f < E.level_since + DOWN_COOLDOWN) return;
    if (E.info.level == 0) notify("VCFG-NX: carga alta, bajando calidad para no trabar el juego");
    E.info.level++;
    E.level_since = f;
    E.stall_run = 0;     // a "solo pasar" se llega solo si ni con lo minimo alcanza
    logf_("[engine] calidad dinamica: baja a -%d\n", E.info.level);
}

static int temp_limit(void)
{
    int l = E.cfg.temp_limit;
    if (E.cfg.eco) l = l && l < 75 ? l : 75;
    return l;
}

// lee la temperatura cada ~2 s y baja la calidad si hace falta
static void thermal_check(void)
{
    uint64_t f = E.info.flips;
    if (!E.pf->temp_c || f < E.next_temp) return;
    E.next_temp = f + TEMP_FLIPS;
    int t = E.pf->temp_c();
    if (t < 15 || t > 120) { E.info.temp_c = -1; return; }   // lectura imposible: sin control termico
    E.info.temp_c = t;
    int lim = temp_limit();
    if (!lim) { E.info.hot = 0; return; }
    if (t >= lim + TEMP_CRIT_DELTA) {
        if (E.info.hot != 2) {
            notify("VCFG-NX: consola muy caliente, efectos en pausa hasta que enfrie");
            logf_("[engine] temperatura critica %d C: todo apagado\n", t);
        }
        E.info.hot = 2;
        E.info.level = 99;   // quality_ladder lo acota a lo que este en uso
        E.level_since = E.last_temp_step = f;
        return;
    }
    if (t >= lim) {
        if (++E.hot_reads >= TEMP_HOT_READS && f >= E.last_temp_step + TEMP_STEP_FLIPS
            && E.info.level < E.info.level_max) {
            if (!E.info.hot) notify("VCFG-NX: consola caliente, bajando calidad");
            E.info.hot = 1;
            E.info.level++;
            E.level_since = E.last_temp_step = f;
            logf_("[engine] temperatura %d C: baja a -%d\n", t, E.info.level);
        }
        return;
    }
    E.hot_reads = 0;
    if (t <= lim - TEMP_COOL_DELTA) E.info.hot = 0;   // con margen: se puede volver a subir
}

static void maybe_level_up(void)
{
    uint64_t f = E.info.flips;
    if (E.probe_until && f >= E.probe_until) E.probe_until = 0;   // la subida se sostuvo
    if (E.info.level == 0 || E.info.hot) return;
    if (E.info.temp_c > 0 && temp_limit() && E.info.temp_c > temp_limit() - TEMP_COOL_DELTA) return;
    uint64_t since = E.level_since > E.last_stall ? E.level_since : E.last_stall;
    int sh = E.up_fails < MAX_UP_FAILS ? E.up_fails : MAX_UP_FAILS;
    if (f < since + ((uint64_t)UP_FLIPS << sh)) return;
    E.info.level--;
    E.level_since = f;
    E.probe_until = f + PROBE_FLIPS;
    logf_("[engine] calidad dinamica: prueba subir a -%d\n", E.info.level);
}

// tiempo de GPU del trabajo anterior en este slot (ya termino): la unidad del
// reloj de la GPU se calibra contra el de la CPU con los inicios de cada trabajo
// waits = 1: ts[2..3] es la espera a que la pantalla suelte el intermedio (generacion
// con la presentacion de VCFG), que se descuenta: no es carga de la GPU. Hasta la
// 2.74 contaba, y con la generacion cualquier efecto parecia "carga alta".
static void gpu_time_read_ts(volatile uint64_t* ts, int waits)
{
    uint64_t t0 = ts[0], t1 = ts[1];
    if (!t0 || t1 <= t0) return;
    uint64_t idle = 0;
    if (waits && ts[2] && ts[3] >= ts[2] && ts[2] >= t0 && ts[3] <= t1) idle = ts[3] - ts[2];
    else if (waits) return;   // falto alguna marca: este fotograma no se mide
    uint64_t now = E.pf->now_us();
    if (!E.ts_first) { E.ts_first = t0; E.ts_cpu_first = now; return; }
    if (now > E.ts_cpu_first + 2000000 && t0 > E.ts_first)
        E.ts_unit = (float)(t0 - E.ts_first) / (float)(now - E.ts_cpu_first);
    if (E.ts_unit <= 0.0f) return;
    float ms = (float)(t1 - t0 - idle) / E.ts_unit / 1000.0f;
    if (ms > 0.0f && ms < 1000.0f) E.info.gpu_ms = E.info.gpu_ms ? E.info.gpu_ms * 0.9f + ms * 0.1f : ms;
    if (waits) {
        float w = (float)idle / E.ts_unit / 1000.0f;
        if (w >= 0.0f && w < 1000.0f) E.info.wait_ms = E.info.wait_ms ? E.info.wait_ms * 0.9f + w * 0.1f : w;
    }
}

static void gpu_time_read(int sl) { gpu_time_read_ts(E.ts_cpu + sl * 2, 0); }

// resumen periodico al registro: cuantos fotogramas y por que pasaron intactos
static void flip_summary(const nx_submit_t* s)
{
    char w[200];
    int n = 0;
    for (int i = 0; i < WHY_N && n < (int)sizeof w - 32; ++i)
        if (E.why[i]) n += snprintf(w + n, sizeof w - (size_t)n, " %s=%llu", WHY_NAME[i], (unsigned long long)E.why[i]);
    if (!n) snprintf(w, sizeof w, " -");
    logf_("[flip] estado=%d handle=%u idx=%u count=%u gnm=%llu wl=%llu cpu=%llu nuestros=%llu fps=%.1f gpu=%.1fms espera=%.1fms nivel=%d intactos:%s\n",
          (int)E.info.state, s->handle, s->index, s->count, (unsigned long long)E.flips_gnm,
          (unsigned long long)E.flips_wl, (unsigned long long)E.flips_cpu, (unsigned long long)E.info.our_jobs,
          (double)E.info.game_fps, (double)E.info.gpu_ms, (double)E.info.wait_ms, E.info.level, w);
    log_flush();
}

// paneles del OSD (contador y menu) para una imagen de ow x oh; rect (si no es
// NULL) = el rectangulo que cubre a los dos (x, y, w, h), 0 si no hay
static int make_osds(const nx_osd_t** osds, nx_osd_t* fps, nx_osd_t* menu, int osd_fps, int osd_menu, int fg_on,
                     int ow, int oh, int* rect)
{
    int no = 0, x0 = 1 << 30, y0 = 1 << 30, x1 = 0, y1 = 0;
    osds[0] = osds[1] = NULL;
    if (osd_fps) {
        build_fps_text(fg_on);
        int sc = ow >= 3000 ? 2 : 1, pw, ph;
        *fps = (nx_osd_t){ (uint64_t)(uintptr_t)E.osd_text[1], (uint64_t)(uintptr_t)E.osd_attr[1],
                           (uint64_t)(uintptr_t)E.osd_font, 16, 1, 24, 24, sc, -1 };
        nx_osd_panel_size(16, 1, sc, &pw, &ph);
        x0 = 24; y0 = 24; x1 = 24 + pw; y1 = 24 + ph;
        osds[no++] = fps;
    }
    if (osd_menu) {
        build_menu_text();
        int scale = ow >= 3000 ? 2 : 1, pw, ph;   // la fuente ya es de 15x30: x2 solo en 4K
        nx_osd_panel_size(OSD_COLS, ROW_COUNT + 3, scale, &pw, &ph);
        *menu = (nx_osd_t){ (uint64_t)(uintptr_t)E.osd_text[0], (uint64_t)(uintptr_t)E.osd_attr[0],
                            (uint64_t)(uintptr_t)E.osd_font, OSD_COLS, ROW_COUNT + 3, ow > pw ? (ow - pw) / 2 : 0,
                            oh > ph ? (oh - ph) / 2 : 0, scale, 2 + E.info.menu_row };
        E.info.menu_x = menu->x0; E.info.menu_y = menu->y0; E.info.menu_w = pw; E.info.menu_h = ph;
        if (menu->x0 < x0) x0 = menu->x0;
        if (menu->y0 < y0) y0 = menu->y0;
        if (menu->x0 + pw > x1) x1 = menu->x0 + pw;
        if (menu->y0 + ph > y1) y1 = menu->y0 + ph;
        osds[no++] = menu;
    }
    if (rect) {
        rect[0] = no ? x0 : 0; rect[1] = no ? y0 : 0;
        rect[2] = no ? x1 - x0 : 0; rect[3] = no ? y1 - y0 : 0;
    }
    return no;
}

// con la salida forzada a 60 Hz el juego queda a 30 parejo: entre un envio y el
// siguiente pasan al menos 33 ms (como pace_hz(30) / fps_lock 30 de VCFG; sin esto
// GTA V se destraba a 45-52 fps y la generacion tironea)
static void pace30(void)
{
    uint64_t now = E.pf->now_us();
    if (E.pace_last_us && now < E.pace_last_us + 33000) {
        uint64_t d = E.pace_last_us + 33000 - now;
        if (d <= 34000) E.pf->sleep_us((uint32_t)d);
        now = E.pf->now_us();
    }
    E.pace_last_us = now;
}

static void gpu_time_read_ts(volatile uint64_t* ts, int waits);

// nitidez FSR+ (2.74, mas fuerte que en la 2.73): RCAS y despues el realce de detalle
// (Baja/Media/Alta = 0.5/1.0/1.6, con un sobreimpulso permitido que crece con la intensidad)
static float sharp_stops(void) { return E.cfg.rcas == 1 ? 0.4f : E.cfg.rcas == 2 ? 0.1f : E.cfg.rcas ? 0.0f : 2.0f; }
static float sharp_detail(void) { return E.cfg.rcas == 1 ? 0.5f : E.cfg.rcas == 2 ? 1.0f : E.cfg.rcas ? 1.6f : 0.0f; }
// NXSR a la resolucion nativa es realce de detalle (no hay escalado que hacer): con
// k = 1 cambia ~0.7/255 en promedio y no se notaba. El limitador 3x3 del shader
// impide halos, asi que se puede subir (Suave/Normal/Fuerte = 1/2/3)
static float sr_gain(int lite)
{
    // NXSR-L: en validacion (sr/net_srl.py) a 1.6 es el mejor en SSIM; se usa bastante
    // mas para que se vea a 1080p nativos (2.74 con 1.2/2/3 "no cambiaba nada"). El
    // nucleo deja afuera el grano y el limite al rango 3x3 acota los halos aun a 4.5.
    if (lite) return E.cfg.sr == 1 ? 2.0f : E.cfg.sr == 2 ? 3.0f : 4.5f;   // Ligero y Calidad
    return E.cfg.sr == 1 ? 1.0f : E.cfg.sr == 2 ? 2.0f : 3.0f;
}

// post-proceso del real de la generacion: en un buffer propio y de vuelta al del
// juego, ANTES de generar el intermedio (que asi hereda la misma calidad y no hace
// falta procesarlo aparte: una sola vez por fotograma del juego). Slots 6-9 del
// pipeline y del OSD (los de la generacion son 0-3 y 2-5).
static int fg8_post(pm4_t* c, int k, const nx_image_t* C, int sr, int hq, int lite, int aa, int rcas, uint32_t seq)
{
    nx_job_t pj;
    memset(&pj, 0, sizeof pj);
    pj.cur = C;
    pj.out_real = &E.real_img[k & 1];
    pj.slot = 6 + k;
    pj.post = NX_POST_NONE;
    pj.aa = aa;
    pj.sharpness_stops = sharp_stops();
    pj.sharp_detail = rcas ? sharp_detail() : 0.0f;
    pj.seq = seq;
    pj.sr = sr;
    pj.sr_k = sr ? sr_gain(lite) : 0.0f;
    pj.sr_hq = hq;
    pj.sr_lite = lite;
    pj.rcas = rcas;
    pj.copy_back = C;
    pj.osd_ctx = &E.osd;
    pj.no_end = 1;
    int r = nx_pipeline_build(&E.pipe, c, &pj);
    if (!r && !E.fx_post_first) {
        E.fx_post_first = 1;
        stage("efectos: primer post-proceso con generacion (nxsr %d%s, smaa %d, nitidez %d)", sr,
              lite ? " ligero" : hq ? " calidad" : "", aa, rcas);
    }
    return r;
}

// un fotograma con la generacion de VCFG
static int fg8_frame(const nx_submit_t* s, int gi)
{
    const uint32_t* gtail = find_tail(s);   // ya verificada al admitir el fotograma
    int k = E.fg8_slot;
    volatile uint32_t* lab = nx_fg8_label(&E.fg8, k);
    uint64_t waited = 0;
    while (*lab < E.fg8_slotseq[k] && waited < WAIT_BUDGET_US) {   // 4 slots: casi nunca espera
        E.pf->sleep_us(WAIT_STEP_US);
        waited += WAIT_STEP_US;
    }
    if (*lab < E.fg8_slotseq[k]) {
        E.info.stalls++;
        E.prev_idx = -1;
        if (++E.stall_run >= STALL_LIMIT) {
            E.info.state = NX_ST_DEGRADED;
            E.degraded_gpu = 1;
            E.degraded_at = E.info.flips;
            set_status("GPU saturada: desactivado");
            set_rate_forced(0);
            notify("VCFG-NX: la GPU no da abasto en este juego; en pausa");
            stage("generacion VCFG: GPU atrasada %d fotogramas seguidos: en pausa", E.stall_run);
        }
        return passthrough_why(s, WHY_STALL);
    }
    E.stall_run = 0;
    gpu_time_read_ts(nx_fg8_ts(&E.fg8, k), 1);
    // calidad dinamica por tiempo de GPU: los 4 slots casi nunca esperan, asi que el
    // atraso no se ve en la espera sino en lo que tarda nuestro trabajo
    if (E.info.gpu_ms > FG8_BUDGET_MS) {
        if (++E.slow_run >= SLOW_LIMIT) { E.slow_run = 0; on_stall(); }
    } else {
        E.slow_run = 0;
        if (E.info.gpu_ms < 0.6f * FG8_BUDGET_MS) maybe_level_up();
    }
    pace30();
    set_rate_forced(1);
    uint32_t seq = ++E.fg8_seq;
    int m = E.fg8_midk;
    int ia = E.cfg.framegen == 2 && E.info.full_ready;   // generacion propia (NXFG-2) con la presentacion de VCFG
    // post-proceso del real (reescalado, SMAA, nitidez), con la calidad dinamica: el
    // ultimo escalon cambia la IA por la generacion de VCFG; la generacion no se apaga
    int sr = 0, hq = 0, lite = 0, aa = 0, rcas = 0;
    if (E.info.full_ready && E.Wo == E.W && E.Ho == E.H) {
        sr = E.cfg.sr > 0;
        sr_choice(sr, &hq, &lite);
        aa = E.cfg.aa;
        rcas = E.cfg.rcas > 0;
    }
    int want_ia = ia;
    quality_ladder(NULL, &ia, &sr, &hq, &lite, &aa, &rcas);
    if (want_ia && !ia && !E.ia_fallback_said) {
        E.ia_fallback_said = 1;
        notify("VCFG-NX: la generacion IA no entra en la GPU con este juego: usa la de VCFG");
        stage("generacion IA -> VCFG por carga (nivel %d)", E.info.level);
    }
    E.info.hq = hq;
    E.info.lite = lite;
    int post_on = sr || aa || rcas;
    int tag = ia ? 4 : 3;
    int repeat = E.prev_idx < 0 || E.prev_idx == gi || E.prev_idx >= 16 || !((E.game_mask >> E.prev_idx) & 1)
                 || E.last_fg != tag;
    if (E.last_fg != tag) E.pipe.luma_valid[0] = E.pipe.luma_valid[1] = 0;
    E.last_fg = tag;
    const nx_image_t* C = &E.game_img[gi];
    const nx_image_t* P = repeat ? C : &E.game_img[E.prev_idx];
    nx_osd_t fps = { 0 }, menu = { 0 };
    const nx_osd_t* osds[2];
    int rect[4];
    int no = make_osds(osds, &fps, &menu, E.cfg.fps_counter, E.info.menu_open, 1, E.W, E.H, rect);
    pm4_t c;
    pm4_init(&c, nx_fg8_dcb(&E.fg8, k), NX_FG8_DCB_DWORDS - NX_FG8_TAIL_DWORDS);
    nx_fg8_job_t j = { P, C, &E.fg8_mid[m], (uint32_t)E.fg8_mid_index[m], E.handle, seq, 0x8000, repeat,
                       rect[0], rect[1], rect[2], rect[3], E.pf->wait_flip_done,
                       (uint64_t)(uintptr_t)nx_fg8_ts(&E.fg8, k), 0 };
    nx_osd_begin(&E.osd, 2 + k);   // una sola vez: la copia y el OSD comparten el anillo del slot
    // prologo de VCFG (espera a que el juego termine el fotograma); despues el
    // post-proceso del real y recien ahi la generacion, sobre los reales ya procesados
    nx_fg8_prologue(&E.fg8, &c, k, seq, C, j.ts0);
    j.no_prologue = 1;
    int br = post_on ? fg8_post(&c, k, C, sr, hq, lite, aa, rcas, seq) : 0;
    if (br) {
        logf_("[fg8] post = %d: post-proceso apagado\n", br);
        E.cfg.sr = E.cfg.aa = E.cfg.rcas = 0;
        post_on = sr = aa = rcas = 0;
        br = 0;
        pm4_init(&c, nx_fg8_dcb(&E.fg8, k), NX_FG8_DCB_DWORDS - NX_FG8_TAIL_DWORDS);
        nx_osd_begin(&E.osd, 2 + k);
        nx_fg8_prologue(&E.fg8, &c, k, seq, C, j.ts0);
    }
    if (!ia) {
        br = nx_fg8_build(&E.fg8, &c, k, &j);
    } else {
        // espera de pantalla de VCFG y la cadena propia: piramides + block-matching (LDS)
        // + NXFG-2 + warp, escribiendo directo en el intermedio
        nx_fg8_wait_mid(&c, &j);
        if (repeat) {   // sin fotograma anterior: el intermedio repite el actual
            br = nx_osd_copy(&E.osd, &c, 2 + k, C, &E.fg8_mid[m]) != 0 ? -4 : 0;
            // la piramide de esta paridad no es la de C: que el proximo la rehaga (si no,
            // el siguiente intermedio saldria del movimiento contra un fotograma viejo)
            E.pipe.luma_valid[k & 1] = 0;
        } else {
            nx_job_t pj;
            memset(&pj, 0, sizeof pj);
            pj.prev = P;
            pj.cur = C;
            pj.out_mid = &E.fg8_mid[m];
            pj.slot = k;
            pj.post = NX_POST_NONE;
            pj.seq = seq;
            pj.fg = 1;
            pj.no_end = 1;
            br = nx_pipeline_build(&E.pipe, &c, &pj);
        }
    }
    if (!br && no) {   // menu/contador en el intermedio y en el real, en el mismo trabajo
        if (nx_osd_draw(&E.osd, &c, 2 + k, &E.fg8_mid[m], osds, no) != 0
            || nx_osd_draw(&E.osd, &c, 2 + k, C, osds, no) != 0) br = -3;
    }
    if (!br) {
        pm4_timestamp(&c, (uint64_t)(uintptr_t)(nx_fg8_ts(&E.fg8, k) + 1));
        nx_fg8_end(&E.fg8, &c, k, seq);
    }
    if (br || c.overflow) {
        logf_("[fg8] build = %d (overflow %d): generacion VCFG apagada\n", br, c.overflow);
        stage("generacion VCFG: error al armar el trabajo (%d)", br);
        E.fg8_ready = 0;
        E.fg8_failed = 1;
        return E.pf->submit_flip(s);
    }
    memcpy(c.buf + c.n, gtail, NX_FG8_TAIL_DWORDS * 4);
    c.n += NX_FG8_TAIL_DWORDS;
    memcpy(nx_fg8_tail(&E.fg8, k), gtail, NX_FG8_TAIL_DWORDS * 4);
    // envio 1: los DCB del juego (el ultimo sin su cola) + el nuestro, flip al intermedio
    void* d[65];
    uint32_t dsz[65];
    void* cb[65];
    uint32_t cbsz[65];
    uint32_t n = 0;
    for (uint32_t i = 0; i < s->count; ++i) {
        d[n] = s->dcb[i];
        dsz[n] = s->dcb_sz[i];
        cb[n] = s->ccb ? s->ccb[i] : NULL;
        cbsz[n] = s->ccb_sz ? s->ccb_sz[i] : 0;
        ++n;
    }
    if (n) {
        dsz[n - 1] -= NX_FG8_TAIL_DWORDS * 4;
        if (!dsz[n - 1]) --n;
    }
    d[n] = c.buf;
    dsz[n] = (uint32_t)(c.n * 4);
    cb[n] = NULL;
    cbsz[n] = 0;
    ++n;
    int has_ccb = s->ccb != NULL;
    nx_submit_t mid = { n, d, dsz, has_ccb ? cb : NULL, has_ccb ? cbsz : NULL, s->handle, (uint32_t)E.fg8_mid_index[m],
                        s->mode, s->arg, s->workload, s->for_workload, 0 };
    int rc = E.pf->submit_flip(&mid);
    // envio 2: el real, en el buffer del juego, con la cola sola
    void* d2[1] = { nx_fg8_tail(&E.fg8, k) };
    uint32_t z2[1] = { NX_FG8_TAIL_DWORDS * 4 };
    nx_submit_t real = { 1, d2, z2, NULL, NULL, s->handle, (uint32_t)gi, s->mode, s->arg, 0, 0, 0 };
    int rc2 = E.pf->submit_flip(&real);
    E.fg8_slotseq[k] = seq;
    E.fg8_slot = (k + 1) % NX_FG8_SLOTS;
    E.fg8_midk = (m + 1) % 3;
    E.prev_idx = gi;
    E.info.our_jobs++;
    E.fx_active = (ia ? FX_FG_IA : FX_FG_VCFG) | (sr ? FX_SR : 0) | (aa ? FX_AA : 0) | (rcas ? FX_RCAS : 0);
    // proteccion anti-cierre, igual que en el otro camino
    if (!E.guard_marked) { guard_write("pending"); E.guard_marked = 1; E.stable_flips = 0; }
    if (++E.stable_flips == STABLE_FLIPS) { E.crashes = 0; guard_write("ok"); }
    if (!E.fg8_first) {
        E.fg8_first = 1;
        stage("generacion VCFG: primer fotograma generado (flips 0x%08x 0x%08x)", (unsigned)rc, (unsigned)rc2);
    }
    return rc;
}

int nx_engine_on_submit_flip(const nx_submit_t* s)
{
    E.info.flips++;
    uint64_t t = E.pf->now_us();
    if (E.last_flip_us && t > E.last_flip_us) {
        float fps = 1e6f / (float)(t - E.last_flip_us);
        E.info.game_fps = E.info.game_fps ? E.info.game_fps * 0.95f + fps * 0.05f : fps;
    }
    E.last_flip_us = t;

    nx_state_t st = E.info.state;
    if (st == NX_ST_DEGRADED && E.degraded_gpu && E.degraded_retries < MAX_DEGRADED_RETRIES
        && E.info.flips >= E.degraded_at + DEGRADED_RETRY_FLIPS) {
        // reintento con lo minimo; la calidad dinamica sube sola si ahora alcanza
        E.degraded_gpu = 0;
        E.degraded_retries++;
        E.info.state = st = NX_ST_RUN;
        E.info.level = 99;
        E.level_since = E.last_stall = E.info.flips;
        E.stall_run = 0;
        set_status("activo");
        logf_("[engine] reintento tras degradado (%d)\n", E.degraded_retries);
    }
    if (s->cpu_flip) E.flips_cpu++; else if (s->for_workload) E.flips_wl++; else E.flips_gnm++;
    if (E.info.flips <= 3 || (E.info.flips % 1800) == 0) flip_summary(s);
    if (st == NX_ST_OFF || st == NX_ST_WAIT_BUFFERS) return passthrough_why(s, WHY_STATE);
    if (st == NX_ST_RUN && E.info.full_ready) thermal_check();
    if ((st == NX_ST_WARMUP || st == NX_ST_RUN) && E.info.flips >= E.next_try_flip) try_allocations();
    st = E.info.state;
    int gi = (int)s->index;
    int why = st != NX_ST_RUN ? WHY_STATE
            : !E.base_mem ? WHY_NOBASE
            : (int)s->handle != E.handle ? WHY_HANDLE
            : (gi < 0 || gi >= 16 || !((E.game_mask >> gi) & 1)) ? WHY_INDEX
            : ((s->count == 0 && !s->cpu_flip) || s->count > 64) ? WHY_COUNT
            : (!s->cpu_flip && !find_tail(s)) ? WHY_TAIL : -1;
    if (why >= 0) {
        if (why != WHY_STATE && !E.why[why]) {
            logf_("[flip] primer fotograma sin tocar por %s (handle=%u idx=%u count=%u)\n", WHY_NAME[why],
                  s->handle, s->index, s->count);
            if (why == WHY_TAIL && s->count && s->dcb_sz && s->dcb && s->dcb[s->count - 1]) {
                uint32_t nd = s->dcb_sz[s->count - 1] / 4;
                const uint32_t* w = (const uint32_t*)s->dcb[s->count - 1];
                logf_("[flip] ultimo DCB %u dwords; final: %08x %08x %08x %08x\n", nd, nd >= 64 ? w[nd - 64] : 0,
                      nd >= 63 ? w[nd - 63] : 0, nd >= 2 ? w[nd - 2] : 0, nd >= 1 ? w[nd - 1] : 0);
            }
            log_flush();
        }
        E.prev_idx = -1;
        rate_idle();
        return passthrough_why(s, why);
    }

    int fx_mem = E.info.full_ready || E.fg8_ready;
    int effects = fx_mem && !E.info.safe_mode && !E.effects_blocked && E.cfg.effects;
    int full = effects && E.info.full_ready;
    // las dos generaciones se presentan como VCFG (fg8_frame); la IA ademas usa la
    // memoria de la cadena propia (piramides, NXFlow)
    // la IA necesita la memoria de los efectos; sin ella va la de VCFG (fg8_frame)
    int want_fg = effects && E.cfg.framegen && E.fg8_ready
                  && !E.cfg.compare && E.info.game_fps > 1.0f && E.info.game_fps < (float)E.cfg.max_game_fps;
    if (E.fx_cap < 3) want_fg = 0;   // arranque escalonado: los efectos se habilitan por partes
    // generacion (VCFG o IA) con la presentacion de VCFG: su propio camino (mismo envio
    // que el juego, flip al intermedio y flip del real al buffer del juego); el
    // post-proceso del real va dentro de ese trabajo, antes de generar. Su calidad
    // dinamica no apaga la generacion (ver fg8_frame)
    if (want_fg && !s->cpu_flip) return fg8_frame(s, gi);
    int sr = full && E.cfg.sr > 0;
    int hq, lite;
    sr_choice(sr, &hq, &lite);
    int aa = full && E.cfg.aa;
    int rcas = full && E.cfg.rcas;   // tambien despues del reescalado
    quality_ladder(&want_fg, NULL, &sr, &hq, &lite, &aa, &rcas);
    if (E.fx_cap < 2) { sr = hq = lite = aa = 0; rcas = full; }
    E.info.hq = hq;
    E.info.lite = lite;
    int want_post = effects && (sr || aa || rcas || E.post == NX_POST_EASU);
    int osd_menu = E.info.menu_open, osd_fps = E.cfg.fps_counter;
    if (want_fg && !E.have_mids) want_fg = 0;   // flips de CPU vistos recien despues de reservar
    if (!want_fg && !want_post && !osd_menu && !osd_fps) {
        E.fx_active = 0;
        E.prev_idx = -1;
        rate_idle();
        // con todo bajado por la calidad dinamica no hay trabajo propio: se mira
        // si la GPU ya se puso al dia para volver a subir
        if (E.info.level > 0) {
            if (*E.label_cpu >= E.seq) maybe_level_up();
            else E.last_stall = E.info.flips;
        }
        return passthrough_why(s, WHY_NOTHING);
    }

    // el trabajo anterior en este slot tiene que haber terminado (presupuesto acotado)
    int sl = E.slot;
    uint64_t waited = 0;
    while (*E.label_cpu < E.slot_seq[sl] && waited < WAIT_BUDGET_US) {
        E.pf->sleep_us(WAIT_STEP_US);
        waited += WAIT_STEP_US;
    }
    if (*E.label_cpu < E.slot_seq[sl]) {
        E.info.stalls++;
        E.prev_idx = -1;
        ++E.stall_run;
        on_stall();
        if (E.stall_run >= STALL_LIMIT) {
            E.info.state = NX_ST_DEGRADED;
            E.degraded_gpu = 1;
            E.degraded_at = E.info.flips;
            set_status("GPU saturada: desactivado");
            set_rate_forced(0);
            notify("VCFG-NX: la GPU no da abasto en este juego; en pausa");
            logf_("[engine] %d fotogramas seguidos con la GPU atrasada: degradado\n", E.stall_run);
            stage("efectos: GPU atrasada %d fotogramas seguidos: en pausa", E.stall_run);
            log_flush();
        }
        return passthrough_why(s, WHY_STALL);
    }
    E.stall_run = 0;
    gpu_time_read(sl);
    // la GPU llego, pero esperarla frena al juego: si es sostenido, bajar calidad
    if (waited >= SLOW_WAIT_US) {
        if (++E.slow_run >= SLOW_LIMIT) { E.slow_run = 0; on_stall(); }
    } else {
        E.slow_run = 0;
        maybe_level_up();
    }

    // nuestro DCB (va en el mismo envio que el trabajo del juego, despues)
    const uint32_t* gtail = s->cpu_flip ? NULL : find_tail(s);
    int rc;
    if (want_fg && E.prev_idx < 0) want_fg = 0;   // falta el fotograma anterior: este va sin FG
    int m = (int)(E.seq & 1);
    nx_osd_t menu = { 0 }, fps = { 0 };
    const nx_osd_t* osds[2] = { NULL, NULL };
    int no = 0;
    // tamano de la imagen donde va el OSD: la de salida si hay efectos reservados,
    // si no la del juego (en el nivel 3 Wo/Ho nunca se calculaban: valian 0 y el
    // menu quedaba centrado en (0,0), con 3/4 fuera de la pantalla)
    int ow = E.info.full_ready && E.Wo ? E.Wo : E.W, oh = E.info.full_ready && E.Ho ? E.Ho : E.H;
    no = make_osds(osds, &fps, &menu, osd_fps, osd_menu, want_fg, ow, oh, NULL);
    (void)no;
    int fg_mode = want_fg ? (E.cfg.framegen == 2 ? 2 : 1) : 0;   // cambiar de rapida a IA rehace la piramide
    if (fg_mode != E.last_fg) { E.prev_real_valid = 0; E.pipe.luma_valid[0] = E.pipe.luma_valid[1] = 0; }
    E.last_fg = fg_mode;
    int enhance_first = want_fg && want_post && E.Wo == E.W;
    pm4_t c;
    pm4_init(&c, E.dcb[sl], DCB_DWORDS - NOP_DWORDS);   // los ultimos 64: la cola de flip
    // antes de leer la imagen: que el juego haya terminado de dibujarla (EOP +
    // espera + caches), como el prologo de los DCB de VCFG
    {
        const nx_image_t* g = &E.game_img[gi];
        uint32_t bytes = g->pitch * g->height * 4;
        *E.label2_cpu = 0;
        pm4_wait_previous_work(&c, E.label2_gpu, E.seq + 1, g->gpu_addr, bytes);
    }
    E.ts_cpu[sl * 2] = E.ts_cpu[sl * 2 + 1] = 0;
    pm4_timestamp(&c, E.ts_gpu + (uint64_t)sl * 16);
    nx_job_t job = {
        .prev = want_fg ? &E.game_img[E.prev_idx] : NULL,
        .cur = &E.game_img[gi],
        .out_mid = want_fg ? &E.mid_img[m] : NULL,
        .out_real = want_post ? &E.real_img[m] : NULL,
        .slot = sl, .post = want_post ? E.post : NX_POST_NONE, .aa = aa,
        .sharpness_stops = sharp_stops(), .sharp_detail = rcas ? sharp_detail() : 0.0f,
        .seq = E.seq + 1,
        .fg = want_fg, .fg_fast = E.cfg.framegen != 2, .sr = sr, .sr_k = sr ? sr_gain(lite) : 0.0f, .sr_hq = hq, .sr_lite = lite,
        .rcas = rcas,
        .enhance_first = enhance_first,
        .prev_real = enhance_first ? &E.real_img[m ^ 1] : NULL,
        .prev_real_valid = E.prev_real_valid,
        .compare = effects && E.cfg.compare && E.Wo == E.W,
        .copy_back = want_post && E.Wo == E.W && E.Ho == E.H ? &E.game_img[gi] : NULL,
        .osd_ctx = &E.osd,
        .osd = { osds[0], osds[1] },
    };
    if (!E.info.full_ready) {   // sin arena solo se puede dibujar el OSD sobre el buffer del juego
        job.fg = 0; job.out_real = NULL; job.post = NX_POST_NONE; job.aa = 0; job.sr = 0; job.rcas = 0;
    }
    int br = E.info.full_ready ? nx_pipeline_build(&E.pipe, &c, &job)
                          : nx_pipeline_build_osd_only(&E.osd, &c, &job, E.label_gpu);
    if (br != 0) {
        logf_("[engine] build = %d: degradado\n", br);
        E.info.state = NX_ST_DEGRADED;
        set_status("error interno: desactivado");
        return E.pf->submit_flip(s);   // nada se envio todavia: el fotograma del juego tal cual
    }
    pm4_timestamp(&c, E.ts_gpu + (uint64_t)sl * 16 + 8);
    // cola de flip al final de nuestro DCB y copia para el flip extra
    if (gtail) {
        memcpy(E.dcb[sl] + c.n, gtail, NOP_DWORDS * 4);
        memcpy(E.tail[sl], gtail, NOP_DWORDS * 4);
    } else {
        make_tail(E.dcb[sl] + c.n);
        make_tail(E.tail[sl]);
    }
    c.n += NOP_DWORDS;
    E.seq++;
    E.slot_seq[sl] = E.seq;
    E.info.our_jobs++;
    if (!E.fx_first_job && job.out_real) {
        E.fx_first_job = 1;
        stage("efectos: primer trabajo (nitidez %d, reescalado %d, smaa %d, generacion %d, copia al juego %d)", job.rcas,
              job.sr, job.aa, job.fg, job.copy_back != NULL);
    }
    // un solo envio: los DCB del juego (el ultimo sin su cola) + el nuestro con la cola
    void* d[65];
    uint32_t dsz[65];
    void* cb[65];
    uint32_t cbsz[65];
    uint32_t n = 0;
    for (uint32_t k = 0; k < s->count; ++k) {
        d[n] = s->dcb[k];
        dsz[n] = s->dcb_sz[k];
        cb[n] = s->ccb ? s->ccb[k] : NULL;
        cbsz[n] = s->ccb_sz ? s->ccb_sz[k] : 0;
        ++n;
    }
    if (n) {
        dsz[n - 1] -= NOP_DWORDS * 4;
        if (!dsz[n - 1]) --n;   // el ultimo era solo la cola: lo reemplaza el nuestro (y su CCB se descarta)
    }
    d[n] = E.dcb[sl];
    dsz[n] = (uint32_t)(c.n * 4);
    cb[n] = NULL;
    cbsz[n] = 0;
    ++n;
    int has_ccb = s->ccb != NULL;
    // el real se presenta en el buffer del juego (copiado de vuelta); solo con EASU
    // (salida mas grande que el juego) se presenta el buffer propio
    int real_idx = job.out_real && !job.copy_back ? E.real_index[m] : (int)s->index;
    nx_submit_t all = { n, d, dsz, has_ccb ? cb : NULL, has_ccb ? cbsz : NULL, s->handle, 0, s->mode, s->arg,
                        s->workload, s->for_workload, 0 };
    if (job.fg) {
        set_rate_forced(1);
        all.index = (uint32_t)E.mid_index[m];
        all.mode = 1;
        rc = E.pf->submit_flip(&all);
        int rc2 = flip_index(s, real_idx, 1);
        if (!E.fx_first_fg) {
            E.fx_first_fg = 1;
            stage("efectos: primer fotograma generado (flips 0x%08x 0x%08x)", (unsigned)rc, (unsigned)rc2);
        }
    } else {
        rate_idle();
        all.index = (uint32_t)real_idx;
        rc = E.pf->submit_flip(&all);
    }
    E.prev_idx = gi;
    E.prev_real_valid = job.out_real != NULL;
    E.slot ^= 1;
    E.fx_active = (job.fg ? (job.fg_fast ? FX_FG_VCFG : FX_FG_IA) : 0) | (job.sr ? FX_SR : 0) | (job.aa ? FX_AA : 0)
                  | (job.rcas ? FX_RCAS : 0) | (job.post == NX_POST_EASU ? FX_EASU : 0);

    // proteccion anti-cierre: sesion "pendiente" mientras no lleve un rato estable
    if (effects && (job.fg || job.out_real)) {
        if (!E.guard_marked) { guard_write("pending"); E.guard_marked = 1; E.stable_flips = 0; }
        if (++E.stable_flips == STABLE_FLIPS) { E.crashes = 0; guard_write("ok"); }
    }
    return rc;
}
