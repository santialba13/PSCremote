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
    int handle, game_first, game_num;
    nx_vo_attr_t attr;
    nx_image_t game_img[16];
    int W, H, Wo, Ho;
    nx_post_t post;
    // recursos propios
    uint8_t* base_mem;          // DCBs + OSD (pocos KiB)
    uint32_t* dcb[2];
    uint32_t* nop_dcb;
    uint8_t* osd_text[2];       // menu, contador
    uint8_t* osd_attr[2];       // color de cada celda (paleta de shaders/osd.cl)
    uint8_t* osd_font;          // atlas suavizado (nx_osd_font.h)
    nx_osd_ctx_t osd;
    volatile uint32_t* label_cpu;   // un solo label para todo nuestro trabajo de GPU
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
    int game_rate, rate_forced;
    // mando
    uint64_t combo_since;
    uint32_t last_buttons;
    int crashes;
    // calidad dinamica
    int pro, announced;
    uint64_t level_since, last_stall, probe_until, degraded_at;
    int up_fails, degraded_gpu, degraded_retries, slow_run;
    // temperatura
    uint64_t next_temp, last_temp_step;
    int hot_reads;
} E;

static void logf_(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void logf_(const char* fmt, ...)
{
    char b[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    if (E.pf && E.pf->log) E.pf->log(b);
}

static void notify(const char* msg)
{
    if (E.pf && E.pf->notify) E.pf->notify(msg);
}

static void set_status(const char* s)
{
    snprintf(E.info.status, sizeof E.info.status, "%s", s);
}

// ------------------------------------------------------- proteccion anti-cierre
static void guard_path(char* p, size_t n) { snprintf(p, n, "/data/vcfg_nx/guard-%s.txt", E.title); }

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
    default:
        return 0;   // HDR (PQ / BT.2020) u otros: no se toca
    }
}

// --------------------------------------------------------------- hooks
void nx_engine_init(const nx_platform_t* pf, const char* title_id)
{
    memset(&E, 0, sizeof E);
    E.pf = pf;
    snprintf(E.title, sizeof E.title, "%s", title_id ? title_id : "");
    E.prev_idx = -1;
    E.game_rate = -1;
    nx_config_load(&E.cfg, E.title, pf->read_file);
    if (strncmp(E.title, "CUSA", 4) != 0 && strncmp(E.title, "PLAS", 4) != 0) {
        E.info.state = NX_ST_OFF;
        set_status("no es un juego");
        return;
    }
    guard_check();
    E.pro = pf->is_pro ? pf->is_pro() != 0 : 0;
    E.info.temp_c = -1;
    E.info.state = NX_ST_WAIT_BUFFERS;
    set_status(E.info.safe_mode ? "modo seguro" : "esperando el juego");
}

void nx_engine_on_register(int handle, int start, void* const* addrs, int num, const nx_vo_attr_t* a, int rc)
{
    if (rc < 0 || !a || !addrs) return;
    if (E.info.state == NX_ST_OFF) return;
    if (E.info.state != NX_ST_WAIT_BUFFERS) {
        if (handle != E.handle) return;
        // el juego re-registra buffers en nuestro handle (vuelta del reposo,
        // cambio de opciones...). Misma geometria: se adoptan las direcciones.
        int same = start == E.game_first && num == E.game_num && a->width == E.attr.width
                   && a->height == E.attr.height && a->format == E.attr.format && a->pitch == E.attr.pitch
                   && a->tmode == E.attr.tmode;
        if (same && E.info.state != NX_ST_OFF) {
            for (int i = 0; i < num; ++i) E.game_img[i].gpu_addr = (uint64_t)(uintptr_t)addrs[i];
            E.prev_idx = -1;
            E.prev_real_valid = 0;
            E.pipe.luma_valid[0] = E.pipe.luma_valid[1] = 0;
            E.info.readopts++;
            logf_("[engine] re-registro de buffers con la misma geometria: adoptado\n");
            return;
        }
        if (E.info.full_ready || E.info.state == NX_ST_OFF) {
            // nuestros buffers ya estan registrados con la geometria vieja
            logf_("[engine] re-registro de buffers con otra geometria: pasa a inactivo\n");
            E.info.state = E.info.full_ready ? NX_ST_DEGRADED : NX_ST_OFF;
            set_status("el juego cambio sus buffers");
            return;
        }
        // todavia no reservamos nada de efectos: se toma como un registro nuevo
        logf_("[engine] re-registro antes de activar efectos: se vuelve a empezar\n");
        E.info.state = NX_ST_WAIT_BUFFERS;
        E.prev_idx = -1;
    }
    if (num < 2 || num > 8 || start < 0 || start + num > 16 || (a->width & 1) || (a->height & 1)) return;
    nx_format_t fmt;
    if (!map_format(a->format, &fmt) || a->width < 640 || a->height < 360 || a->width > 3840 || a->height > 2160) {
        logf_("[engine] formato 0x%08x %ux%u no soportado: inactivo\n", (unsigned)a->format, a->width, a->height);
        E.info.state = NX_ST_OFF;
        set_status("formato no soportado (HDR?)");
        return;
    }
    E.handle = handle;
    E.game_first = start;
    E.game_num = num;
    E.attr = *a;
    E.W = (int)a->width;
    E.H = (int)a->height;
    uint32_t tile = a->tmode == 1 ? NX_TILE_DISPLAY_LINEAR_ALIGNED : NX_TILE_DISPLAY_2D_THIN;
    for (int i = 0; i < num; ++i) {
        nx_image_t im = { (uint64_t)(uintptr_t)addrs[i], (uint32_t)E.W, (uint32_t)E.H, a->pitch, fmt, tile };
        E.game_img[i] = im;
    }
    E.info.state = NX_ST_WARMUP;
    E.next_try_flip = (uint64_t)E.cfg.warmup_flips;
    if (E.next_try_flip < E.info.flips + 60) E.next_try_flip = E.info.flips + 60;
    set_status("juego cargando");
    logf_("[engine] buffers del juego: %dx%d x%d, formato 0x%08x\n", E.W, E.H, num, (unsigned)a->format);
}

int nx_engine_on_set_flip_rate(int handle, int rate)
{
    if (handle == E.handle && !E.rate_forced) E.game_rate = rate;
    if (handle == E.handle && E.rate_forced) return 0;
    return rate;
}

// ------------------------------------------------------------- reservas
static int alloc_base(void)
{
    size_t osd_sz = nx_osd_mem_size();
    size_t need = (size_t)(2 * DCB_DWORDS + NOP_DWORDS) * 4 + 4 * 4096 + sizeof nx_osd_font + osd_sz + 8192;
    // la base son pocos KiB (menu + DCBs): alcanza con dejar 32 MiB libres
    if (E.pf->avail_direct() < need + 32u * 1048576) return 0;
    E.base_mem = E.pf->alloc_gpu(need, 65536);
    if (!E.base_mem) return 0;
    uint8_t* p = E.base_mem;
#define ALIGN256(q) (uint8_t*)(((uintptr_t)(q) + 255) & ~(uintptr_t)255)
    E.dcb[0] = (uint32_t*)p; p += DCB_DWORDS * 4;
    E.dcb[1] = (uint32_t*)p; p += DCB_DWORDS * 4;
    E.nop_dcb = (uint32_t*)p; p += NOP_DWORDS * 4;
    for (int i = 0; i < NOP_DWORDS; ++i) E.nop_dcb[i] = 0x80000000u;
    p = ALIGN256(p);
    E.label_cpu = (volatile uint32_t*)p; E.label_gpu = (uint64_t)(uintptr_t)p; *E.label_cpu = 0; p += 256;
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

static int effects_wanted(void)
{
    return E.cfg.effects && (E.cfg.framegen || E.cfg.sr || E.cfg.aa || E.cfg.rcas);
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
    size_t total = arena_sz + 2 * MAX_MID * buf;
    int extra_first = E.game_first + E.game_num;
    if (extra_first + 2 * MAX_MID > 16) {
        set_status("sin indices libres de VideoOut");
        return -1;
    }
    size_t avail = E.pf->avail_direct();
    if (avail < total + (size_t)E.cfg.reserve_mb * 1048576) {
        logf_("[engine] memoria: hacen falta %zu MiB + %d de reserva, libres %zu MiB\n",
              total >> 20, E.cfg.reserve_mb, avail >> 20);
        E.info.alloc_fail++;
        return 0;
    }
    uint8_t* arena = E.pf->alloc_gpu(arena_sz, 65536);
    uint8_t* outs = arena ? E.pf->alloc_gpu(2 * MAX_MID * buf, 65536) : NULL;
    if (!arena || !outs) { E.info.alloc_fail++; return 0; }
    nx_arena_t a = { arena, (uint64_t)(uintptr_t)arena, arena_sz, 0 };
    if (nx_pipeline_init(&E.pipe, E.W, E.H, E.Wo, E.Ho, a) != 0) { set_status("error interno (pipeline)"); return -1; }
    E.pipe.label_gpu = E.label_gpu;   // el mismo label que el OSD: la secuencia nunca retrocede
    E.pipe.label_cpu = E.label_cpu;
    void* addrs[2 * MAX_MID];
    for (int i = 0; i < 2 * MAX_MID; ++i) addrs[i] = outs + buf * i;
    nx_vo_attr_t oa = E.attr;
    oa.width = (uint32_t)E.Wo; oa.height = (uint32_t)E.Ho; oa.pitch = (uint32_t)pitch_o;
    oa.tmode = 1;   // nuestros kernels escriben en lineal
    int rc = E.pf->register_buffers(E.handle, extra_first, addrs, 2 * MAX_MID, &oa);
    if (rc < 0) { logf_("[engine] RegisterBuffers propio = 0x%08x\n", (unsigned)rc); set_status("VideoOut rechazo los buffers"); return -1; }
    nx_format_t fmt = E.game_img[0].format;
    for (int i = 0; i < MAX_MID; ++i) {
        nx_image_t im = { (uint64_t)(uintptr_t)addrs[i], (uint32_t)E.Wo, (uint32_t)E.Ho, (uint32_t)pitch_o, fmt,
                          NX_TILE_DISPLAY_LINEAR_ALIGNED };
        E.mid_img[i] = im; E.mid_index[i] = extra_first + i;
        im.gpu_addr = (uint64_t)(uintptr_t)addrs[MAX_MID + i];
        E.real_img[i] = im; E.real_index[i] = extra_first + MAX_MID + i;
    }
    E.info.full_ready = 1;
    logf_("[engine] listo: %dx%d -> %dx%d, %zu MiB\n", E.W, E.H, E.Wo, E.Ho, total >> 20);
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
    if (E.info.full_ready || E.info.safe_mode || !effects_wanted() || E.retries >= MAX_RETRIES) return;
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
    if (E.info.temp_c > 0) snprintf(l, sizeof l, "%.0f fps  %d\xb0" "C", E.info.game_fps, E.info.temp_c);
    else snprintf(l, sizeof l, "%.0f fps", E.info.game_fps);
    osd_put(t, a, 0, OSD_COLS - 1 - (int)strlen(l), A_DIM, l);
    // estado
    osd_clear(t, a, 1);
    int warn = E.info.safe_mode || E.info.hot || E.info.level > 0 || E.info.state == NX_ST_DEGRADED || !E.info.full_ready;
    if (!E.cfg.effects) snprintf(l, sizeof l, "efectos apagados");
    else if (E.info.hot) snprintf(l, sizeof l, "%s: consola caliente (-%d)", E.info.hot == 2 ? "pausa" : "ahorro", E.info.level);
    else if (E.info.level > 0) snprintf(l, sizeof l, "%.26s (carga -%d)", E.info.status, E.info.level);
    else snprintf(l, sizeof l, "%.36s", E.info.status);
    osd_put(t, a, 1, 1, !E.cfg.effects ? A_OFF : (warn ? A_WARN : A_ACCENT), l);
    int dim = !E.cfg.effects;
    // interruptor general
    osd_item(t, a, 2 + ROW_ALL, "Todos los efectos", E.cfg.effects ? "ON" : "OFF", E.cfg.effects ? A_ON : A_ERROR, 0);
    osd_switch(t, a, 2 + ROW_FG, "Generaci\xf3n de fotogramas", E.cfg.framegen, dim);
    osd_item(t, a, 2 + ROW_SR, "Reescalado IA (NXSR)", sr_names[E.cfg.sr & 3], E.cfg.sr ? A_ACCENT : A_OFF, dim);
    // modelo: lo que eligio el usuario y lo que se usa de verdad
    const char* mv;
    if (E.cfg.eco) mv = "R\xe1pido (ahorro)";
    else if (E.cfg.sr_model == 2) mv = "Calidad";
    else if (E.cfg.sr_model == 1) mv = "R\xe1pido";
    else mv = E.pro ? "Auto: Calidad" : "Auto: R\xe1pido";
    osd_item(t, a, 2 + ROW_MODEL, "Modelo IA", mv, A_ACCENT, dim || !E.cfg.sr);
    osd_switch(t, a, 2 + ROW_ECO, "Modo ahorro (menos calor)", E.cfg.eco, dim);
    osd_switch(t, a, 2 + ROW_AA, "Antialiasing SMAA", E.cfg.aa, dim || E.cfg.eco);
    osd_item(t, a, 2 + ROW_RCAS, "Nitidez FSR1 (RCAS)", rc_names[E.cfg.rcas & 3], E.cfg.rcas ? A_ACCENT : A_OFF,
             dim || E.cfg.sr);
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
    case ROW_FG: c->framegen = !c->framegen; break;
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

void nx_engine_on_pad(uint32_t* buttons, uint8_t sticks[4], uint8_t triggers[2])
{
    if (!buttons || E.info.state == NX_ST_OFF || E.info.state == NX_ST_WAIT_BUFFERS) return;
    uint32_t b = *buttons;
    uint64_t t = E.pf->now_us();
    int combo = (b & (BTN_L3 | BTN_R3)) == (BTN_L3 | BTN_R3);
    if (!combo) {
        E.combo_since = 0;
    } else if (!E.combo_since) {
        E.combo_since = t;
    } else if (E.combo_since != 1 && t - E.combo_since > (uint64_t)E.cfg.menu_hold_ms * 1000) {
        E.combo_since = 1;
        E.info.menu_open = !E.info.menu_open && E.base_mem != NULL;
        E.info.menu_row = 0;
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
        E.last_buttons = b;
        // el juego no recibe nada mientras el menu esta abierto
        *buttons = 0;
        if (sticks) sticks[0] = sticks[1] = sticks[2] = sticks[3] = 0x80;
        if (triggers) triggers[0] = triggers[1] = 0;
        return;
    }
    E.last_buttons = b;
}

// ------------------------------------------------------------- flips
static int passthrough(const nx_submit_t* s)
{
    E.info.passthrough++;
    return E.pf->submit_flip(s);
}

static int flip_index(const nx_submit_t* game, int index, uint32_t mode)
{
    void* d[1] = { E.nop_dcb };
    uint32_t sz[1] = { NOP_DWORDS * 4 };
    nx_submit_t f = { 1, d, sz, NULL, NULL, game->handle, (uint32_t)index, mode, game->arg, 0, 0, 0 };
    return E.pf->submit_flip(&f);
}

static void set_rate_forced(int on)
{
    if (on == E.rate_forced || !E.pf->set_flip_rate) return;
    E.rate_forced = on;
    E.pf->set_flip_rate(E.handle, on ? 0 : (E.game_rate >= 0 ? E.game_rate : 0));
}

const nx_engine_info_t* nx_engine_info(void) { return &E.info; }

// ------------------------------------------------------- calidad dinamica
// escalones, del mas caro al mas barato; solo cuentan los que estan en uso
static void quality_ladder(int* fg, int* sr, int* hq, int* aa, int* rcas)
{
    int steps = 0, lv = E.info.level;   // los escalones por atraso solo se suman con `dynamic`; los de calor siempre
    if (*sr && *hq) { if (lv > steps) *hq = 0; steps++; }
    if (*aa) { if (lv > steps) *aa = 0; steps++; }
    if (*sr || *rcas) { if (lv > steps) *sr = *rcas = *hq = 0; steps++; }
    if (*fg) { if (lv > steps) *fg = 0; steps++; }
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
    if (st == NX_ST_OFF || st == NX_ST_WAIT_BUFFERS) return passthrough(s);
    if (st == NX_ST_RUN && E.info.full_ready) thermal_check();
    if ((st == NX_ST_WARMUP || st == NX_ST_RUN) && E.info.flips >= E.next_try_flip) try_allocations();
    st = E.info.state;
    int gi = (int)s->index - E.game_first;
    if (st != NX_ST_RUN || !E.base_mem || (int)s->handle != E.handle || gi < 0 || gi >= E.game_num
        || (s->count == 0 && !s->cpu_flip) || s->count > 64) {
        E.prev_idx = -1;
        set_rate_forced(0);
        return passthrough(s);
    }

    int effects = E.info.full_ready && !E.info.safe_mode && E.cfg.effects;
    int want_fg = effects && E.cfg.framegen && !E.cfg.compare && E.info.game_fps > 1.0f
                  && E.info.game_fps < (float)E.cfg.max_game_fps;
    int sr = effects && E.cfg.sr > 0;
    int hq = sr && !E.cfg.eco && (E.cfg.sr_model == 2 || (E.cfg.sr_model == 0 && E.pro));
    int aa = effects && E.cfg.aa && !E.cfg.eco;
    int rcas = effects && E.cfg.rcas && !sr;
    quality_ladder(&want_fg, &sr, &hq, &aa, &rcas);
    E.info.hq = hq;
    int want_post = effects && (sr || aa || rcas || E.post == NX_POST_EASU);
    int osd_menu = E.info.menu_open, osd_fps = E.cfg.fps_counter;
    if (!want_fg && !want_post && !osd_menu && !osd_fps) {
        E.prev_idx = -1;
        set_rate_forced(0);
        // con todo bajado por la calidad dinamica no hay trabajo propio: se mira
        // si la GPU ya se puso al dia para volver a subir
        if (E.info.level > 0) {
            if (*E.label_cpu >= E.seq) maybe_level_up();
            else E.last_stall = E.info.flips;
        }
        return passthrough(s);
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
        }
        return passthrough(s);
    }
    E.stall_run = 0;
    // la GPU llego, pero esperarla frena al juego: si es sostenido, bajar calidad
    if (waited >= SLOW_WAIT_US) {
        if (++E.slow_run >= SLOW_LIMIT) { E.slow_run = 0; on_stall(); }
    } else {
        E.slow_run = 0;
        maybe_level_up();
    }

    // 1. el trabajo del juego, sin flip (en un flip de CPU ya lo envio el juego)
    int rc = s->count ? E.pf->submit(s) : 0;
    if (rc != 0) return rc;

    // 2. nuestro DCB
    if (want_fg && E.prev_idx < 0) want_fg = 0;   // falta el fotograma anterior: este va sin FG
    int m = (int)(E.seq & 1);
    nx_osd_t menu = { 0 }, fps = { 0 };
    const nx_osd_t* osds[2] = { NULL, NULL };
    int no = 0;
    if (osd_fps) {
        build_fps_text(want_fg);
        fps = (nx_osd_t){ (uint64_t)(uintptr_t)E.osd_text[1], (uint64_t)(uintptr_t)E.osd_attr[1],
                          (uint64_t)(uintptr_t)E.osd_font, 16, 1, 24, 24, E.Wo >= 3000 ? 2 : 1, -1 };
        osds[no++] = &fps;
    }
    if (osd_menu) {
        build_menu_text();
        int scale = E.Wo >= 3000 ? 2 : 1, pw, ph;   // la fuente ya es de 15x30: x2 solo en 4K
        nx_osd_panel_size(OSD_COLS, ROW_COUNT + 3, scale, &pw, &ph);
        menu = (nx_osd_t){ (uint64_t)(uintptr_t)E.osd_text[0], (uint64_t)(uintptr_t)E.osd_attr[0],
                           (uint64_t)(uintptr_t)E.osd_font, OSD_COLS, ROW_COUNT + 3, (E.Wo - pw) / 2, (E.Ho - ph) / 2,
                           scale, 2 + E.info.menu_row };
        osds[no++] = &menu;
    }
    int fg_mode = want_fg ? 1 : 0;
    if (fg_mode != E.last_fg) { E.prev_real_valid = 0; E.pipe.luma_valid[0] = E.pipe.luma_valid[1] = 0; }
    E.last_fg = fg_mode;
    int enhance_first = want_fg && want_post && E.Wo == E.W;
    pm4_t c;
    pm4_init(&c, E.dcb[sl], DCB_DWORDS);
    nx_job_t job = {
        .prev = want_fg ? &E.game_img[E.prev_idx] : NULL,
        .cur = &E.game_img[gi],
        .out_mid = want_fg ? &E.mid_img[m] : NULL,
        .out_real = want_post ? &E.real_img[m] : NULL,
        .slot = sl, .post = want_post ? E.post : NX_POST_NONE, .aa = aa,
        .sharpness_stops = E.cfg.rcas ? 0.75f - 0.25f * (float)E.cfg.rcas : 2.0f,
        .seq = E.seq + 1,
        .fg = want_fg, .sr = sr, .sr_k = sr ? 0.5f + 0.25f * (float)(E.cfg.sr - 1) : 0.0f, .sr_hq = hq,
        .rcas = rcas,
        .enhance_first = enhance_first,
        .prev_real = enhance_first ? &E.real_img[m ^ 1] : NULL,
        .prev_real_valid = E.prev_real_valid,
        .compare = effects && E.cfg.compare && E.Wo == E.W,
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
        // el juego ya se envio sin flip: solo falta presentar su fotograma
        return flip_index(s, (int)s->index, s->mode);
    }
    E.seq++;
    E.slot_seq[sl] = E.seq;
    E.info.our_jobs++;
    void* d[1] = { E.dcb[sl] };
    uint32_t dsz[1] = { (uint32_t)(c.n * 4) };
    int real_idx = job.out_real ? E.real_index[m] : (int)s->index;
    if (job.fg) {
        set_rate_forced(1);
        nx_submit_t mid = { 1, d, dsz, NULL, NULL, s->handle, (uint32_t)E.mid_index[m], 1, s->arg, 0, 0, 0 };
        rc = E.pf->submit_flip(&mid);
        flip_index(s, real_idx, 1);
    } else {
        set_rate_forced(0);
        nx_submit_t one = { 1, d, dsz, NULL, NULL, s->handle, (uint32_t)real_idx, s->mode, s->arg, 0, 0, 0 };
        rc = E.pf->submit_flip(&one);
    }
    E.prev_idx = gi;
    E.prev_real_valid = job.out_real != NULL;
    E.slot ^= 1;

    // proteccion anti-cierre: sesion "pendiente" mientras no lleve un rato estable
    if (effects && (job.fg || job.out_real)) {
        if (!E.guard_marked) { guard_write("pending"); E.guard_marked = 1; E.stable_flips = 0; }
        if (++E.stable_flips == STABLE_FLIPS) { E.crashes = 0; guard_write("ok"); }
    }
    return rc;
}
