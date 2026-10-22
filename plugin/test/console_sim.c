// Consola simulada para probar nx_engine (la logica del plugin) en el PC, sin
// arriesgar una PS4: memoria directa limitada, VideoOut con 16 indices, GPU que
// completa (o no) nuestros DCB, reloj, archivos y mando.
//
// Cada escenario simula un juego a 30 o 60 fps y verifica invariantes:
//   * cada fotograma del juego se envia a la GPU exactamente una vez y se presenta;
//   * nunca se presenta un indice no registrado ni se registran indices ocupados;
//   * no se reserva memoria durante el arranque del juego ni si no queda margen;
//   * ante GPU trabada, formato HDR, re-registro o datos raros: el juego sigue intacto.
// Compilar con -fsanitize=address,undefined (ver Makefile: make sim).
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../source/nx_engine.h"
#include "../source/pm4.h"
#include "../include/nx_shaders.h"

// --------------------------------------------------------------- consola
typedef struct { char path[96]; char data[1024]; int len; } file_t;

static struct {
    size_t pool, used;                // memoria directa
    void* allocs[64];
    int nallocs;
    int registered[16];               // indices de VideoOut ocupados
    uint32_t reg_w[16];
    uint64_t clock_us;
    int gpu_stuck;                    // la GPU nunca completa nuestros DCB
    // modelo de tiempo de la GPU (0 = instantanea): cola serie, el juego cuesta
    // game_cost_us por fotograma y lo nuestro ns_per_kinst por cada 1000 instrucciones
    // de wavefront (kernel_costs.h: medidas con el codigo maquina real, tools/isa_costs.py)
    uint32_t game_cost_us, ns_per_kinst;
    uint64_t gpu_free_at;
    struct { volatile uint32_t* addr; uint32_t val; uint64_t at; int ts; } pend[64];
    uint32_t flip_wait_us;   // lo que tarda WaitFlipDone (la pantalla suelta el intermedio): GPU ociosa
    int npend;
    int pro;
    size_t early_base, early_work;    // reserva temprana del plugin (la hace main.c al cargar)
    int temp;                         // grados C (0 = el sistema no la informa)
    uint64_t groups_total;
    int game_dcb_submits, flips, bad_flips, our_dcbs, bad_dcbs, bad_tail, plain_with_ours;
    int mid_flips, pairs, last_mid, game_nbuf;
    int last_flip_index;
    file_t files[16];
    int nfiles;
    char last_notify[160];
    int notifies;
    int verbose;
} C;

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("    FALLO: " __VA_ARGS__); printf("\n"); ++fails; } } while (0)

// "DCB del juego": un NOP corto y, como en la consola, la cola de flip de
// DrawCommandBuffer::prepareFlip (NOP de 64 dwords con la marca 0x687507xx)
#define GAME_SZ (68 * 4)
static uint32_t game_dcb_mark[68] = { 0xC0021000u, 0, 0, 0, 0xC03E1000u, 0x68750777u };
static int tail_ok(const uint32_t* d, uint32_t bytes)
{
    uint32_t n = bytes / 4;
    return n >= 64 && d[n - 64] == 0xC03E1000u && (d[n - 63] >> 8) == 0x687507u;
}

static void* sim_alloc(size_t len, size_t align)
{
    (void)align;
    len = (len + 0xFFFF) & ~(size_t)0xFFFF;
    size_t* early = len <= (2u << 20) && C.early_base >= len ? &C.early_base : C.early_work >= len ? &C.early_work : NULL;
    if (early && C.nallocs < 64) {   // sale de la reserva temprana: no cuenta contra lo libre
        void* p = aligned_alloc(65536, len);
        memset(p, 0, len);
        *early -= len;
        C.allocs[C.nallocs++] = p;
        return p;
    }
    if (C.used + len > C.pool || C.nallocs >= 64) return NULL;
    void* p = aligned_alloc(65536, len);
    memset(p, 0, len);
    C.used += len;
    C.allocs[C.nallocs++] = p;
    return p;
}

static size_t sim_avail(void) { return C.pool - C.used; }

static int sim_register(int handle, int start, void* const* addrs, int num, const nx_vo_attr_t* a)
{
    (void)handle; (void)addrs;
    if (start < 0 || start + num > 16) return -1;
    for (int i = start; i < start + num; ++i)
        if (C.registered[i]) return -2;   // VideoOut no deja pisar indices
    for (int i = start; i < start + num; ++i) { C.registered[i] = 1; C.reg_w[i] = a->width; }
    return 0;
}

static int rate_calls;
static int sim_rate(int h, int r) { (void)h; (void)r; ++rate_calls; return 0; }

static void gpu_resolve(void)
{
    int k = 0;
    for (int i = 0; i < C.npend; ++i) {
        if (C.pend[i].at <= C.clock_us) {
            if (C.pend[i].ts) *(volatile uint64_t*)C.pend[i].addr = C.pend[i].at * 100;   // reloj de la GPU: 100 MHz
            else *C.pend[i].addr = C.pend[i].val;
        }
        else C.pend[k++] = C.pend[i];
    }
    C.npend = k;
}

static uint64_t gpu_enqueue(uint64_t cost_us)
{
    uint64_t start = C.gpu_free_at > C.clock_us ? C.gpu_free_at : C.clock_us;
    C.gpu_free_at = start + cost_us;
    return C.gpu_free_at;
}

#include "kernel_costs.h"

// instrucciones de wavefront por grupo del kernel que esta en la direccion pgm
static unsigned pgm_cost(uint64_t pgm)
{
    const uint32_t* code = (const uint32_t*)(uintptr_t)pgm;
    for (int i = 0; i < NX_NUM_SHADERS; ++i) {
        const nx_shader_t* s = &nx_shaders[i];
        if (s->code_dwords >= 8 && !memcmp(code, s->code, 32)) {
            for (size_t k = 0; k < sizeof kernel_costs / sizeof kernel_costs[0]; ++k)
                if (!strcmp(kernel_costs[k].name, s->name)) return kernel_costs[k].ipg;
            return 300;   // sin medir (OSD, copias): liviano
        }
    }
    return 300;   // codigo de VCFG (nx_fg8)
}

// "ejecuta" un DCB: valida PM4 y procesa la escritura del label al final
static void run_dcb(uint32_t* b, uint32_t bytes)
{
    size_t n = bytes / 4, i = 0;
    int ok = 1;
    uint64_t groups = 0, kinst = 0, pgm = 0;
    while (i < n) {
        uint32_t h = b[i];
        if (h == 0x80000000u) { ++i; continue; }
        if ((h >> 30) != 3) { ok = 0; break; }
        uint32_t op = (h >> 8) & 0xFF, cnt = ((h >> 16) & 0x3FFF) + 1;
        if (i + 1 + cnt > n) { ok = 0; break; }
        if (op == PM4_SET_SH_REG && b[i + 1] == ((R_COMPUTE_PGM_LO - SH_REG_BASE) >> 2))
            pgm = ((uint64_t)b[i + 2] << 8) | (cnt > 2 ? (uint64_t)b[i + 3] << 40 : 0);
        if (op == 0x10 && cnt == 6 && C.flip_wait_us && !C.gpu_stuck) {
            // espera a la pantalla (WaitFlipDone; sin la funcion del sistema queda un NOP de 7 dwords)
            gpu_enqueue(kinst * C.ns_per_kinst / 1000);
            kinst = 0;
            gpu_enqueue(C.flip_wait_us);
        }
        if (op == PM4_DISPATCH_DIRECT) {
            uint64_t g = (uint64_t)b[i + 1] * b[i + 2] * b[i + 3];
            groups += g;
            C.groups_total += g;
            if (C.ns_per_kinst && pgm) kinst += g * pgm_cost(pgm) / 1000;
        }
        if (op == PM4_EVENT_WRITE_EOP) {   // GPU trabada: queda colgado hasta que se destrabe
            uint64_t addr = b[i + 2] | (uint64_t)(b[i + 3] & 0xFFFF) << 32;
            uint64_t at = C.gpu_stuck ? UINT64_MAX : gpu_enqueue(kinst * C.ns_per_kinst / 1000);
            groups = 0;
            kinst = 0;
            if (C.npend < 64) {
                C.pend[C.npend].addr = (volatile uint32_t*)(uintptr_t)addr;
                C.pend[C.npend].val = b[i + 4];
                C.pend[C.npend].ts = ((b[i + 3] >> 29) & 7) == 3;   // DATA_SEL = 3: reloj de 64 bits
                C.pend[C.npend++].at = at;
            }
            gpu_resolve();
        }
        i += 1 + cnt;
    }
    if (!ok || i != n) C.bad_dcbs++;
}

static void process(const nx_submit_t* s)
{
    for (uint32_t k = 0; k < s->count; ++k) {
        if (s->dcb[k] == (void*)game_dcb_mark) { C.game_dcb_submits++; gpu_enqueue(C.game_cost_us); }
        else { C.our_dcbs++; run_dcb((uint32_t*)s->dcb[k], s->dcb_sz[k]); }
    }
}

static int sim_submit(const nx_submit_t* s)
{
    for (uint32_t k = 0; k < s->count; ++k) if (s->dcb[k] != (void*)game_dcb_mark) C.plain_with_ours++;
    process(s);
    return 0;
}

static int cpu_flips;

static int sim_submit_flip(const nx_submit_t* s)
{
    if (s->cpu_flip && s->count == 0) cpu_flips++;
    // todo flip de Gnm: el ultimo DCB termina en la cola de flip (si no, el sistema
    // escribe sobre las ultimas instrucciones de ese DCB)
    if (s->count && !tail_ok((const uint32_t*)s->dcb[s->count - 1], s->dcb_sz[s->count - 1])) C.bad_tail++;
    process(s);
    C.flips++;
    if (s->index >= 16 || !C.registered[s->index]) C.bad_flips++;
    if (C.game_nbuf && (int)s->index >= C.game_nbuf) { C.mid_flips++; C.last_mid = 1; }
    else { if (C.last_mid) C.pairs++; C.last_mid = 0; }
    C.last_flip_index = (int)s->index;
    return 0;
}

static uint64_t sim_now(void) { gpu_resolve(); return C.clock_us; }
static void sim_sleep(uint32_t us) { C.clock_us += us; gpu_resolve(); }
static int sim_is_pro(void) { return C.pro; }
static int sim_temp(void) { return C.temp ? C.temp : -1; }
static size_t sim_pool_free(int base) { return base ? C.early_base : C.early_work; }
static void sim_notify(const char* m) { snprintf(C.last_notify, sizeof C.last_notify, "%s", m); C.notifies++; }
static void sim_log(const char* m) { if (C.verbose) printf("      log: %s", m); }

static file_t* find_file(const char* p)
{
    for (int i = 0; i < C.nfiles; ++i) if (!strcmp(C.files[i].path, p)) return &C.files[i];
    return NULL;
}

static int sim_read(const char* p, char* buf, int cap)
{
    file_t* f = find_file(p);
    if (!f) return -1;
    int n = f->len < cap ? f->len : cap;
    memcpy(buf, f->data, n);
    return n;
}

static int sim_write(const char* p, const char* d, int len)
{
    file_t* f = find_file(p);
    if (!f) { f = &C.files[C.nfiles++]; snprintf(f->path, sizeof f->path, "%s", p); }
    f->len = len < 1023 ? len : 1023;
    memcpy(f->data, d, f->len);
    f->data[f->len] = 0;
    return len;
}

// registro del plugin: un solo archivo grande en memoria
static char sim_logfile[1 << 20];
static int sim_loglen;
static char sim_logpath[96];

static int sim_append(const char* p, const char* d, int len)
{
    snprintf(sim_logpath, sizeof sim_logpath, "%s", p);
    if (sim_loglen + len >= (int)sizeof sim_logfile - 1) return -1;
    memcpy(sim_logfile + sim_loglen, d, (size_t)len);
    sim_loglen += len;
    sim_logfile[sim_loglen] = 0;
    return len;
}

static const nx_platform_t PF = { sim_alloc, sim_avail, sim_register, sim_rate, sim_submit, sim_submit_flip,
                                  sim_now, sim_sleep, sim_notify, sim_log, sim_read, sim_write, sim_append,
                                  sim_is_pro, sim_temp, sim_pool_free, NULL, NULL, NULL };

// ------------------------------------------------------------- juego
typedef struct {
    const char* title;
    int W, H, fps, nbuf;
    int32_t format;
    size_t pool_mb, game_mb;
} game_t;

static uint8_t* g_fb;

static int boot_pro;

static void boot(const game_t* g, int keep_files)
{
    file_t files[16];
    int nfiles = C.nfiles;
    if (keep_files) memcpy(files, C.files, sizeof files);
    for (int i = 0; i < C.nallocs; ++i) free(C.allocs[i]);
    memset(&C, 0, sizeof C);
    cpu_flips = 0;
    sim_loglen = 0; sim_logfile[0] = 0;
    if (keep_files) { memcpy(C.files, files, sizeof files); C.nfiles = nfiles; }
    C.pool = g->pool_mb << 20;
    C.used = g->game_mb << 20;
    C.pro = boot_pro;
    // los escenarios se escribieron con 600 fotogramas de arranque (el valor por
    // defecto ahora es 1200: ver el escenario 24)
    // y sin generacion (framegen = 0): la calidad dinamica es del post-proceso. Las
    // generaciones (VCFG = 1, IA = 2) se presentan como VCFG: escenarios 28 y 29
    if (!find_file("/data/vcfg_nx/config.ini")) sim_write("/data/vcfg_nx/config.ini", "warmup_flips = 600\nframegen = 0\n", 32);
    nx_engine_init(&PF, g->title);
    free(g_fb);
    g_fb = calloc(1, 4096);
    void* addrs[8];
    for (int i = 0; i < g->nbuf; ++i) addrs[i] = g_fb + i * 256;
    nx_vo_attr_t a = { g->format, 0, 0, (uint32_t)g->W, (uint32_t)g->H, (uint32_t)((g->W + 63) / 64 * 64) };
    int rc = sim_register(0, 0, addrs, g->nbuf, &a);
    C.game_nbuf = g->nbuf;
    nx_engine_on_register(0, 0, addrs, g->nbuf, &a, rc);
}

static uint32_t pad_buttons;

static void play(const game_t* g, int frames, int workload)
{
    void* dcb[1] = { game_dcb_mark };
    uint32_t sz[1] = { GAME_SZ };
    for (int f = 0; f < frames; ++f) {
        uint32_t b = pad_buttons;
        uint8_t st[4] = { 10, 20, 30, 40 }, tr[2] = { 5, 6 };
        nx_engine_on_pad(&b, st, tr);
        nx_submit_t s = { 1, dcb, sz, NULL, NULL, 0, (uint32_t)(f % g->nbuf), 1, f, 77, workload == 1, 0 };
        if (workload == 2) {   // juego que envia su trabajo aparte y presenta con un flip de CPU
            nx_submit_t w = { 1, dcb, sz, NULL, NULL, 0, 0, 0, 0, 0, 0, 0 };
            sim_submit(&w);
            s.count = 0; s.dcb = NULL; s.dcb_sz = NULL; s.cpu_flip = 1;
        }
        int before = C.game_dcb_submits;
        nx_engine_on_submit_flip(&s);
        if (workload != 2 && C.game_dcb_submits != before + 1) { CHECK(0, "fotograma %d: el DCB del juego se envio %d veces", f, C.game_dcb_submits - before); break; }
        C.clock_us += 1000000 / g->fps;
        gpu_resolve();
    }
}

static const nx_engine_info_t* I(void) { return nx_engine_info(); }

static void report(const char* name)
{
    const nx_engine_info_t* i = I();
    printf("  %-44s estado=%d mem=%3zu MiB jobs=%llu pass=%llu stalls=%llu nivel=-%d/%d | %s\n", name, i->state,
           C.used >> 20, (unsigned long long)i->our_jobs, (unsigned long long)i->passthrough,
           (unsigned long long)i->stalls, i->level, i->level_max, i->status);
}

static void common_checks(void)
{
    CHECK(C.bad_flips == 0, "%d flips a indices no registrados", C.bad_flips);
    CHECK(C.bad_dcbs == 0, "%d DCB mal formados", C.bad_dcbs);
    CHECK(C.bad_tail == 0, "%d flips sin la cola de flip al final", C.bad_tail);
    CHECK(C.plain_with_ours == 0, "%d DCB propios enviados sin flip", C.plain_with_ours);
}

int main(int argc, char** argv)
{
    C.verbose = argc > 1;
    setvbuf(stdout, NULL, _IONBF, 0);
    game_t bb = { "CUSA00207", 1920, 1080, 30, 2, (int32_t)0x80000000, 5120, 4600 };
    printf("Consola simulada: escenarios\n");

    // 1. arranque normal con memoria de sobra
    boot(&bb, 0);
    play(&bb, 300, 0);
    CHECK(C.used == (size_t)bb.game_mb << 20, "se reservo memoria durante el arranque del juego");
    CHECK(I()->state == NX_ST_WARMUP, "deberia seguir en warmup");
    play(&bb, 2400, 0);
    report("1. 30 fps, memoria de sobra");
    CHECK(I()->full_ready, "no se activaron los efectos");
    CHECK(I()->our_jobs > 1500, "pocos trabajos propios (%llu)", (unsigned long long)I()->our_jobs);
    CHECK(C.flips >= 2700, "faltan flips: %d", C.flips);   // sin generacion: un flip por fotograma
    CHECK(strstr(find_file("/data/GoldHEN/vcfg_nx/guard-CUSA00207.txt")->data, "state=ok") != NULL, "la sesion no quedo marcada ok");
    common_checks();

    // 2. poca memoria: el juego deja 150 MiB libres (menos que la reserva + lo necesario)
    game_t low = bb; low.game_mb = bb.pool_mb - 150;
    boot(&low, 0);
    play(&low, 5000, 0);
    report("2. memoria insuficiente");
    CHECK(!I()->full_ready, "reservo los efectos sin margen");
    CHECK(C.used <= ((size_t)low.game_mb << 20) + (1 << 20), "reservo mas de 1 MiB con poca memoria");
    CHECK(C.flips == 5000, "con efectos apagados cada fotograma deberia ser un flip (%d)", C.flips);
    common_checks();

    // 3. GPU trabada a mitad de partida
    boot(&bb, 0);
    play(&bb, 1200, 0);
    C.gpu_stuck = 1;
    play(&bb, 200, 0);
    report("3. GPU trabada");
    CHECK(I()->level == I()->level_max && I()->level_max >= 2, "no bajo todos los escalones (%d/%d)", I()->level, I()->level_max);
    int flips_before = C.flips;
    uint64_t jobs_before = I()->our_jobs;
    play(&bb, 100, 0);
    CHECK(C.flips == flips_before + 100 && I()->our_jobs == jobs_before, "sin efectos deberia ser un flip por fotograma");
    common_checks();

    // 3b. GPU trabada con el contador de FPS (queda trabajo propio): pausa y reintento
    boot(&bb, 0);
    sim_write("/data/vcfg_nx/config.ini", "fps_counter = 1\nwarmup_flips = 600\nframegen = 0\n", 48);
    nx_engine_init(&PF, bb.title);
    {
        void* ad[2] = { g_fb, g_fb + 256 };
        nx_vo_attr_t at = { bb.format, 0, 0, 1920, 1080, 1920 };
        nx_engine_on_register(0, 0, ad, 2, &at, 0);
    }
    play(&bb, 1200, 0);
    C.gpu_stuck = 1;
    play(&bb, 300, 0);
    CHECK(I()->state == NX_ST_DEGRADED, "no se pauso con la GPU trabada y el OSD activo");
    C.gpu_stuck = 0;
    for (int k = 0; k < C.npend; ++k) C.pend[k].at = C.clock_us;
    play(&bb, 5600, 0);
    CHECK(I()->state == NX_ST_RUN, "no reintento tras la pausa");
    play(&bb, 12000, 0);
    report("3b. GPU trabada -> pausa -> reintento");
    CHECK(I()->level == 0, "no recupero toda la calidad tras el reintento (nivel -%d)", I()->level);
    common_checks();

    // 4. cierre del juego con efectos activos -> la proxima sesion arranca en modo seguro
    boot(&bb, 0);
    play(&bb, 900, 0);   // efectos activos, todavia no "estable"
    CHECK(strstr(find_file("/data/GoldHEN/vcfg_nx/guard-CUSA00207.txt")->data, "pending") != NULL, "sin marca pendiente");
    boot(&bb, 1);        // "se cerro": nueva sesion conservando /data
    play(&bb, 1500, 0);
    report("4. sesion tras un cierre: modo seguro");
    CHECK(I()->safe_mode, "no entro en modo seguro");
    CHECK(!I()->full_ready, "en modo seguro no deberia reservar los efectos");
    // el usuario abre el menu (L3+R3 1,6 s) y reactiva frame generation
    pad_buttons = 0x0002 | 0x0004;
    play(&bb, 60, 0);
    CHECK(I()->menu_open, "el menu no se abrio");
    pad_buttons = 0; play(&bb, 2, 0);
    pad_buttons = 0x0020; play(&bb, 1, 0);   // derecha: cambia "Generacion de fotogramas"
    pad_buttons = 0; play(&bb, 2, 0);
    pad_buttons = 0x0020; play(&bb, 1, 0);   // y otra vez (queda como estaba, pero activo)
    pad_buttons = 0; play(&bb, 1000, 0);
    report("   ...reactivado desde el menu");
    CHECK(!I()->safe_mode && I()->full_ready, "no salio del modo seguro al reactivar");
    common_checks();

    // 5. menu: el juego no recibe botones, guardar escribe el ini
    boot(&bb, 0);
    play(&bb, 700, 0);
    pad_buttons = 0x0002 | 0x0004; play(&bb, 60, 0);
    uint32_t b = 0x1000; uint8_t st[4] = { 1, 2, 3, 4 }, tr[2] = { 9, 9 };   // triangulo: el menu no lo usa
    nx_engine_on_pad(&b, st, tr);
    CHECK(b == 0 && st[0] == 0x80 && tr[0] == 0, "el juego recibio input con el menu abierto");
    pad_buttons = 0; play(&bb, 1, 0);
    for (int k = 0; k < 4; ++k) { pad_buttons = 0x0040; play(&bb, 1, 0); pad_buttons = 0; play(&bb, 1, 0); }
    {   // grilla del menu (fila "Modelo IA" elegida) para gpu_emu osd
        FILE* mf = fopen("menu_grid.bin", "wb");
        if (mf && I()->menu_text) {
            fwrite(I()->menu_text, 1, (size_t)I()->menu_cols * I()->menu_rows, mf);
            fwrite(I()->menu_attr, 1, (size_t)I()->menu_cols * I()->menu_rows, mf);
        }
        if (mf) fclose(mf);
    }
    CHECK(I()->menu_row == 4, "fila del menu (%d)", I()->menu_row);
    pad_buttons = 0x0020; play(&bb, 1, 0); pad_buttons = 0; play(&bb, 1, 0);   // derecha: modelo Auto -> Rapido
    for (int k = 0; k < 4; ++k) { pad_buttons = 0x0010; play(&bb, 1, 0); pad_buttons = 0; play(&bb, 1, 0); }
    pad_buttons = 0x4000; play(&bb, 1, 0); pad_buttons = 0; play(&bb, 1, 0);   // X en "Todos los efectos": OFF
    for (int k = 0; k < 10; ++k) { pad_buttons = 0x0040; play(&bb, 1, 0); pad_buttons = 0; play(&bb, 1, 0); }
    pad_buttons = 0x4000; play(&bb, 1, 0); pad_buttons = 0; play(&bb, 1, 0);   // X en "Guardar"
    file_t* ini = find_file("/data/vcfg_nx/CUSA00207.ini");
    CHECK(ini && strstr(ini->data, "effects = 0") && strstr(ini->data, "framegen = 0") && strstr(ini->data, "sr_model = 1"),
          "no se guardo el ini del juego (efectos OFF, modelo Rapido)");
    pad_buttons = 0x2000; play(&bb, 1, 0); pad_buttons = 0; play(&bb, 1, 0);   // O cierra
    CHECK(!I()->menu_open, "O no cerro el menu");
    report("5. menu y guardado");
    common_checks();

    // 6. juego HDR: solo el menu (sin efectos que alteren la imagen)
    game_t hdr = bb; hdr.format = (int32_t)0x88740000;
    boot(&hdr, 0);
    play(&hdr, 2000, 0);
    report("6. juego HDR");
    CHECK(I()->state == NX_ST_RUN && !I()->full_ready && C.used <= ((size_t)hdr.game_mb << 20) + (1 << 20)
          && C.flips == 2000 && I()->our_jobs == 0, "HDR deberia quedar intacto salvo el menu");
    pad_buttons = 0x0002 | 0x0004; play(&hdr, 60, 0); pad_buttons = 0; play(&hdr, 10, 0);
    CHECK(I()->menu_open && I()->our_jobs > 0, "en HDR el menu deberia abrirse");
    common_checks();

    // 7. juego a 60 fps: sin frame generation (pero con reescalado)
    game_t g60 = bb; g60.fps = 60;
    boot(&g60, 0);
    play(&g60, 3000, 0);
    report("7. juego a 60 fps");
    CHECK(C.flips <= 3000 + 5, "uso frame generation en un juego a 60 fps (%d flips)", C.flips);
    CHECK(I()->our_jobs > 2000, "no aplico el reescalado");
    common_checks();

    // 8. el juego re-registra sus buffers en medio de la partida
    boot(&bb, 0);
    play(&bb, 1200, 0);
    void* addrs[2] = { g_fb, g_fb + 256 };
    nx_vo_attr_t a = { bb.format, 0, 0, 1280, 720, 1280 };
    nx_engine_on_register(0, 0, addrs, 2, &a, 0);
    flips_before = C.flips;
    play(&bb, 300, 0);
    report("8. re-registro de buffers");
    CHECK(C.flips == flips_before + 300, "tras el re-registro deberia pasar todo intacto");
    common_checks();

    // 9. variante ForWorkload y llamadas raras (indice fuera de rango, count 0, otro handle)
    boot(&bb, 0);
    play(&bb, 1000, 1);
    void* dcb[3] = { game_dcb_mark, game_dcb_mark, game_dcb_mark };
    uint32_t sz[3] = { GAME_SZ, GAME_SZ, GAME_SZ };
    srand(1);
    for (int k = 0; k < 2000; ++k) {
        nx_submit_t s = { (uint32_t)(rand() % 3), dcb, sz, NULL, NULL, (uint32_t)(rand() % 3),
                          (uint32_t)(rand() % 20), (uint32_t)(rand() % 3), k, 0, rand() & 1, 0 };
        nx_engine_on_submit_flip(&s);
        C.clock_us += 33333;
    }
    report("9. ForWorkload + 2000 llamadas aleatorias");
    CHECK(C.bad_dcbs == 0, "DCB mal formados");

    // 11. juego que presenta con sceVideoOutSubmitFlip (flip de CPU)
    boot(&bb, 0);
    play(&bb, 3000, 2);
    report("11. juego con flips de CPU");
    CHECK(I()->full_ready && I()->our_jobs > 1500, "no proceso los fotogramas del juego con flip de CPU");
    CHECK(cpu_flips >= 590 && cpu_flips <= 610, "durante el arranque deberia pasar sus flips de CPU intactos (%d)", cpu_flips);
    common_checks();

    // 12. GPU justa: el juego usa 24 ms de GPU por fotograma (de 33): con todo
    // activo no alcanza; la calidad dinamica baja hasta que alcanza y no se apaga.
    // Con NXSR Rapido: el costo del simulador es por grupo de GPU y NXSR-L tiene
    // muchos grupos livianos (en la consola es ~20 veces mas barato que Rapido)
    boot(&bb, 0);   // archivos limpios
    sim_write("/data/vcfg_nx/CUSA00207.ini", "sr_model = 1\n", 13);
    boot(&bb, 1);
    C.game_cost_us = 24000; C.ns_per_kinst = 100;   // SMAA + NXSR + RCAS caros: no alcanza con todo
    play(&bb, 4000, 0);
    report("12. GPU justa: calidad dinamica");
    CHECK(I()->state == NX_ST_RUN && I()->level > 0 && I()->level < I()->level_max,
          "deberia quedar activo con calidad parcial (nivel -%d de %d)", I()->level, I()->level_max);
    uint64_t st12 = I()->stalls;
    {   // con atrasos sueltos la salida no salta entre 30 y 60 Hz en cada fotograma
        int rc0 = rate_calls;
        play(&bb, 1000, 0);
        CHECK(rate_calls - rc0 <= 4, "la tasa de salida cambio %d veces en 1000 fotogramas", rate_calls - rc0);
        st12 = I()->stalls;
    }
    CHECK(st12 < 60, "demasiados fotogramas atrasados (%llu)", (unsigned long long)st12);
    jobs_before = I()->our_jobs;
    play(&bb, 3000, 0);
    CHECK(I()->our_jobs > jobs_before + 2500, "dejo de procesar");
    // (cada prueba de subida que no se sostiene cuesta 2-3 fotogramas atrasados;
    // la copia de vuelta al buffer del juego suma un pase completo por fotograma)
    CHECK(I()->stalls - st12 <= 3 * 3, "sigue atrasandose (%llu)", (unsigned long long)(I()->stalls - st12));
    // la escena se aligera (8 ms de GPU): vuelve a subir a la calidad completa
    C.game_cost_us = 8000;
    play(&bb, 16000, 0);
    report("    ...escena liviana: sube de nuevo");
    CHECK(I()->level == 0, "no volvio a la calidad completa (nivel -%d)", I()->level);
    common_checks();
    sim_write("/data/vcfg_nx/CUSA00207.ini", "sr_model = 0\n", 13);

    // 13. re-registro con la misma geometria (vuelta del reposo): sigue activo
    boot(&bb, 0);
    play(&bb, 1200, 0);
    {
        void* ad[2] = { g_fb + 512, g_fb + 768 };
        nx_vo_attr_t at = { bb.format, 0, 0, 1920, 1080, 1920 };
        nx_engine_on_register(0, 0, ad, 2, &at, 0);
    }
    jobs_before = I()->our_jobs;
    play(&bb, 600, 0);
    report("13. re-registro con la misma geometria");
    CHECK(I()->state == NX_ST_RUN && I()->readopts == 1 && I()->our_jobs > jobs_before + 500, "no adopto los buffers nuevos");
    common_checks();

    // 14. re-registro con otra resolucion antes de activar efectos: vuelve a empezar
    boot(&bb, 0);
    play(&bb, 200, 0);
    {
        void* ad[3] = { g_fb, g_fb + 256, g_fb + 512 };
        nx_vo_attr_t at = { bb.format, 0, 0, 1280, 720, 1280 };
        int rc14 = sim_register(0, 4, ad, 3, &at);
        nx_engine_on_register(0, 4, ad, 3, &at, rc14);
    }
    game_t g14 = bb; g14.nbuf = 3;
    for (int f = 0; f < 2000; ++f) {   // ahora presenta en los indices 4..6
        void* dcb1[1] = { game_dcb_mark };
        uint32_t sz1[1] = { GAME_SZ };
        nx_submit_t s14 = { 1, dcb1, sz1, NULL, NULL, 0, (uint32_t)(4 + f % 3), 1, f, 0, 0, 0 };
        nx_engine_on_submit_flip(&s14);
        C.clock_us += 33333;
        gpu_resolve();
    }
    report("14. re-registro antes de activar efectos");
    CHECK(I()->state == NX_ST_RUN && I()->full_ready && I()->our_jobs > 1000, "no se activo con los buffers nuevos");
    (void)g14;
    common_checks();

    // 15. modelo de reescalado: por defecto NXSR-L (ligero) en Pro y en PS4 normal
    boot_pro = 1;
    boot(&bb, 0);
    play(&bb, 1200, 0);
    CHECK(I()->lite == 1 && I()->hq == 0, "por defecto deberia usar NXSR-L (Pro)");
    boot_pro = 0;
    boot(&bb, 0);
    play(&bb, 1200, 0);
    CHECK(I()->lite == 1 && I()->hq == 0, "por defecto deberia usar NXSR-L");
    sim_write("/data/vcfg_nx/CUSA00207.ini", "sr_model = 2\n", 13);
    sim_write("/data/GoldHEN/vcfg_nx/guard-CUSA00207.txt", "state=ok\ncrashes=0\n", 19);   // salio bien
    boot(&bb, 1);
    play(&bb, 1200, 0);
    report("15. modelo IA (Auto / Calidad)");
    CHECK(I()->lite == 2 && I()->hq == 0, "sr_model = 2 deberia usar NXSR-L Calidad");
    common_checks();

    // 16. temperatura: sube por encima del limite, luego critica, luego enfria
    sim_write("/data/vcfg_nx/CUSA00207.ini", "fps_counter = 1\n", 16);
    sim_write("/data/GoldHEN/vcfg_nx/guard-CUSA00207.txt", "state=ok\ncrashes=0\n", 19);
    boot(&bb, 1);
    C.temp = 65;
    play(&bb, 1200, 0);
    CHECK(I()->temp_c == 65 && I()->level == 0, "lectura de temperatura (%d) o nivel (-%d)", I()->temp_c, I()->level);
    C.temp = 82;   // caliente: baja de a un escalon
    play(&bb, 1000, 0);
    int lv_hot = I()->level;
    CHECK(lv_hot >= 1 && I()->hot == 1, "con 82 C no bajo la calidad (nivel -%d)", lv_hot);
    C.temp = 90;   // critica: todo apagado
    play(&bb, 100, 0);
    CHECK(I()->hot == 2 && I()->level == I()->level_max, "con 90 C no apago todo");
    jobs_before = I()->our_jobs;
    C.temp = 77;   // bajo el limite pero sin margen: no sube
    play(&bb, 3000, 0);
    CHECK(I()->level == I()->level_max, "subio sin margen de temperatura");
    C.temp = 68;   // enfrio: sube de a un escalon por minuto
    play(&bb, 9000, 0);
    report("16. temperatura: caliente -> critica -> enfria");
    CHECK(I()->level == 0 && I()->hot == 0, "no recupero la calidad al enfriar (nivel -%d)", I()->level);
    C.temp = 300;  // lectura imposible: se ignora
    play(&bb, 100, 0);
    CHECK(I()->temp_c == -1, "acepto una temperatura imposible");
    common_checks();

    // 17. modo ahorro: con el modelo Calidad elegido, usa NXSR-L (SMAA sigue)
    sim_write("/data/vcfg_nx/CUSA00207.ini", "eco = 1\nsr_model = 2\n", 21);
    sim_write("/data/GoldHEN/vcfg_nx/guard-CUSA00207.txt", "state=ok\ncrashes=0\n", 19);
    boot_pro = 1;
    boot(&bb, 1);
    play(&bb, 1200, 0);
    uint64_t g0 = C.groups_total, j0 = I()->our_jobs;
    play(&bb, 100, 0);
    uint64_t eco_groups = (C.groups_total - g0) / (I()->our_jobs - j0);
    CHECK(I()->hq == 0 && I()->lite == 1, "modo ahorro uso el modelo Calidad");
    sim_write("/data/vcfg_nx/CUSA00207.ini", "eco = 0\nsr_model = 2\n", 21);
    sim_write("/data/GoldHEN/vcfg_nx/guard-CUSA00207.txt", "state=ok\ncrashes=0\n", 19);
    boot(&bb, 1);
    play(&bb, 1200, 0);
    g0 = C.groups_total; j0 = I()->our_jobs;
    play(&bb, 100, 0);
    uint64_t full_groups = (C.groups_total - g0) / (I()->our_jobs - j0);
    boot_pro = 0;
    printf("  17. modo ahorro: %llu grupos de GPU por fotograma (normal en Pro: %llu, -%.0f%%)\n",
           (unsigned long long)eco_groups, (unsigned long long)full_groups,
           100.0 * (1.0 - (double)eco_groups / (double)full_groups));
    CHECK(eco_groups != full_groups, "modo ahorro no cambio el trabajo");
    sim_write("/data/vcfg_nx/CUSA00207.ini", "eco = 0\nsr_model = 0\n", 21);
    common_checks();

    // 18. interruptor general desde el menu: todo apagado y despues vuelve
    sim_write("/data/GoldHEN/vcfg_nx/guard-CUSA00207.txt", "state=ok\ncrashes=0\n", 19);
    sim_write("/data/vcfg_nx/CUSA00207.ini", "effects = 1\n", 12);
    boot(&bb, 1);
    play(&bb, 1200, 0);
    CHECK(I()->full_ready && I()->our_jobs > 500, "no arranco activo");
    pad_buttons = 0x0002 | 0x0004; play(&bb, 60, 0); pad_buttons = 0; play(&bb, 1, 0);   // abrir menu
    pad_buttons = 0x4000; play(&bb, 1, 0); pad_buttons = 0; play(&bb, 1, 0);           // X: todos OFF
    pad_buttons = 0x2000; play(&bb, 1, 0); pad_buttons = 0; play(&bb, 1, 0);           // O: cerrar
    jobs_before = I()->our_jobs;
    flips_before = C.flips;
    play(&bb, 300, 0);
    CHECK(I()->our_jobs == jobs_before && C.flips == flips_before + 300, "con todo apagado sigue procesando");
    pad_buttons = 0x0002 | 0x0004; play(&bb, 60, 0); pad_buttons = 0; play(&bb, 1, 0);
    pad_buttons = 0x4000; play(&bb, 1, 0); pad_buttons = 0; play(&bb, 1, 0);           // X: todos ON
    pad_buttons = 0x2000; play(&bb, 1, 0); pad_buttons = 0; play(&bb, 1, 0);
    jobs_before = I()->our_jobs;
    flips_before = C.flips;
    play(&bb, 300, 0);
    report("18. interruptor general (menu L3+R3)");
    CHECK(I()->our_jobs > jobs_before + 250 && C.flips > flips_before + 250, "no volvio a procesar");
    common_checks();

    // 19. registro de a un buffer por llamada (y un handle de otra salida que se ignora)
    sim_write("/data/GoldHEN/vcfg_nx/guard-CUSA00207.txt", "state=ok\ncrashes=0\n", 19);
    sim_write("/data/vcfg_nx/CUSA00207.ini", "effects = 1\n", 12);
    boot(&bb, 1);
    {
        // boot() ya registro 0..1 de una; se reinicia y se registra de a uno
        for (int i = 0; i < 16; ++i) C.registered[i] = 0;
        nx_engine_init(&PF, bb.title);
        nx_vo_attr_t at = { bb.format, 0, 0, 1920, 1080, 1920 };
        for (int i = 0; i < 3; ++i) {
            void* ad[1] = { g_fb + 256 * i };
            int rc = sim_register(0, i, ad, 1, &at);
            nx_engine_on_register(0, i, ad, 1, &at, rc);
        }
        void* ad2[1] = { g_fb + 1024 };   // otra salida (p. ej. la social): no cambia nada
        nx_vo_attr_t small = { bb.format, 0, 0, 640, 360, 640 };
        nx_engine_on_register(7, 0, ad2, 1, &small, 0);
    }
    game_t g19 = bb; g19.nbuf = 3;
    play(&g19, 2400, 0);
    report("19. registro de a un buffer");
    CHECK(I()->state == NX_ST_RUN && I()->full_ready && I()->our_jobs > 1000, "no se activo con buffers registrados de a uno");
    CHECK(sim_loglen > 0 && strstr(sim_logfile, "RegisterBuffers") && strstr(sim_logfile, "[flip]"),
          "el registro del juego no se escribio");
    CHECK(!strcmp(sim_logpath, "/data/GoldHEN/vcfg_nx/CUSA00207.log"), "ruta del registro: %s", sim_logpath);
    common_checks();

    // 20. sondeo propio del mando: el combo funciona aunque el juego no lea con
    // los hooks; con el menu abierto los hooks siguen filtrando
    sim_write("/data/GoldHEN/vcfg_nx/guard-CUSA00207.txt", "state=ok\ncrashes=0\n", 19);
    boot(&bb, 1);
    nx_engine_pad_polled(1);
    play(&bb, 1200, 0);
    for (int k = 0; k < 60; ++k) { nx_engine_pad_logic(0x0002 | 0x0004); C.clock_us += 33333; }
    nx_engine_pad_logic(0);
    CHECK(I()->menu_open, "el sondeo propio no abrio el menu");
    pad_buttons = 0x0002 | 0x0004;   // por los hooks no cuenta (no se cierra solo)
    play(&bb, 120, 0);
    pad_buttons = 0;
    CHECK(I()->menu_open, "los hooks no deberian manejar el combo con sondeo propio");
    b = 0x4000; nx_engine_on_pad(&b, NULL, NULL);
    CHECK(b == 0, "con el menu abierto el hook deberia filtrar");
    nx_engine_pad_logic(0x2000); nx_engine_pad_logic(0);
    CHECK(!I()->menu_open, "O no cerro el menu (sondeo)");
    report("20. mando por sondeo propio");
    common_checks();

    // 21. L3+R3 antes de que el juego registre la pantalla: avisa por que no abre
    nx_engine_init(&PF, bb.title);
    C.notifies = 0;
    for (int k = 0; k < 60; ++k) { nx_engine_pad_logic(0x0002 | 0x0004); C.clock_us += 33333; }
    CHECK(C.notifies == 1 && strstr(C.last_notify, "no disponible"), "sin aviso al pedir el menu sin pantalla (%s)", C.last_notify);
    printf("  21. L3+R3 sin pantalla: \"%s\"\n", C.last_notify);

    // 22. primero registra una superficie en formato no soportado y despues la pantalla real
    sim_write("/data/GoldHEN/vcfg_nx/guard-CUSA00207.txt", "state=ok\ncrashes=0\n", 19);
    boot(&bb, 1);
    for (int i = 0; i < 16; ++i) C.registered[i] = 0;
    nx_engine_init(&PF, bb.title);
    {
        void* ad[2] = { g_fb, g_fb + 256 };
        nx_vo_attr_t fl = { (int32_t)0xC1060000, 0, 0, 1920, 1080, 1920 };
        nx_engine_on_register(0, 0, ad, 2, &fl, 0);   // flotante: no sirve, pero no apaga
        nx_vo_attr_t at = { bb.format, 0, 0, 1920, 1080, 1920 };
        int rc = sim_register(0, 0, ad, 2, &at);
        nx_engine_on_register(0, 0, ad, 2, &at, rc);
    }
    play(&bb, 2400, 0);
    report("22. formato raro y despues el real");
    CHECK(I()->state == NX_ST_RUN && I()->full_ready, "no se activo tras un formato no soportado");
    common_checks();

    // 23. GTA V: el juego se queda con casi toda la memoria directa al arrancar
    //     (quedan 20 MiB). Sin la reserva temprana no habria ni menu; con ella
    //     (4 MiB base + 112 de trabajo, hecha al cargar el plugin) anda todo.
    sim_write("/data/GoldHEN/vcfg_nx/guard-CUSA00411.txt", "state=ok\ncrashes=0\n", 19);
    game_t gta = bb; gta.title = "CUSA00411"; gta.game_mb = bb.pool_mb - 20;
    boot(&gta, 1);
    play(&gta, 2400, 0);
    CHECK(!I()->full_ready, "sin reserva temprana no deberia poder reservar");
    pad_buttons = 0x0002 | 0x0004; play(&gta, 60, 0); pad_buttons = 0; play(&gta, 2, 0);
    CHECK(!I()->menu_open, "sin memoria no deberia abrir el menu");
    printf("  23a. GTA V sin reserva temprana: \"%s\"\n", C.last_notify);
    boot(&gta, 1);
    C.early_base = 4u << 20; C.early_work = 112u << 20;
    play(&gta, 2400, 0);
    report("23. GTA V con reserva temprana");
    CHECK(I()->state == NX_ST_RUN && I()->full_ready && I()->our_jobs > 1000, "no se activo con la reserva temprana");
    CHECK(C.used == (size_t)gta.game_mb << 20, "uso memoria del juego teniendo reserva propia");
    pad_buttons = 0x0002 | 0x0004; play(&gta, 60, 0); pad_buttons = 0; play(&gta, 2, 0);
    CHECK(I()->menu_open, "con reserva temprana el menu deberia abrir");
    // la valvula le devolvio la reserva de trabajo al juego: solo menu
    boot(&gta, 1);
    C.early_base = 4u << 20; C.early_work = 0;
    play(&gta, 2400, 0);
    pad_buttons = 0x0002 | 0x0004; play(&gta, 60, 0); pad_buttons = 0; play(&gta, 2, 0);
    report("    ...reserva de trabajo devuelta al juego");
    CHECK(I()->menu_open && !I()->full_ready, "con solo la base deberia quedar el menu");
    common_checks();

    // 24. por defecto el arranque dura 1200 fotogramas: el plugin no usa la GPU
    //     antes de que el arranque quede confirmado (900)
    {
        nx_config_t c;
        nx_config_defaults(&c);
        CHECK(c.warmup_flips == 1200, "warmup por defecto (%d)", c.warmup_flips);
    }

    // 25. cola de flip: GTA V usa la marca 0x68750778 (anda igual); un juego
    //     cuyo ultimo DCB no termina en la cola pasa intacto (nunca se toca)
    {
        game_t g25 = bb; g25.title = "CUSA00411";
        boot(&g25, 0);
        game_dcb_mark[5] = 0x68750778u;
        play(&g25, 2400, 0);
        report("25a. GTA V: cola con 0x68750778");
        CHECK(I()->full_ready && I()->our_jobs > 1000, "con la marca de GTA V deberia activarse");
        common_checks();
        game_dcb_mark[5] = 0x68750777u;
        boot(&g25, 0);
        game_dcb_mark[4] = 0xC0001000u;   // sin cola
        int f0 = C.flips;
        play(&g25, 2400, 0);
        game_dcb_mark[4] = 0xC03E1000u;
        report("25b. ultimo DCB sin cola de flip");
        CHECK(I()->our_jobs == 0 && C.flips == f0 + 2400, "sin cola no deberia tocar nada (trabajos %llu)",
              (unsigned long long)I()->our_jobs);
    }

    // 26. nivel 3 del arranque escalonado: solo menu, sin efectos
    {
        boot(&bb, 0);
        nx_engine_allow_effects(0);
        play(&bb, 2400, 0);
        CHECK(I()->state == NX_ST_RUN && !I()->full_ready, "con efectos bloqueados no deberia reservar efectos");
        pad_buttons = 0x0002 | 0x0004; play(&bb, 60, 0); pad_buttons = 0; play(&bb, 30, 0);
        report("26. nivel 3: solo menu");
        CHECK(I()->menu_open && I()->our_jobs > 0, "el menu deberia abrir y dibujarse");
        CHECK(I()->menu_x > 0 && I()->menu_y > 0 && I()->menu_x + I()->menu_w <= bb.W && I()->menu_y + I()->menu_h <= bb.H,
              "menu fuera de pantalla: (%d,%d) %dx%d en %dx%d", I()->menu_x, I()->menu_y, I()->menu_w, I()->menu_h, bb.W, bb.H);
        common_checks();
    }

    // 27. efectos por partes (niveles 4-6) y la imagen real en el buffer del juego
    {
        int caps[3] = { 1, 2, 3 };
        for (int k = 0; k < 3; ++k) {
            boot(&bb, 0);
            nx_engine_allow_effects(caps[k]);
            play(&bb, 2400, 0);
            char n[64];
            snprintf(n, sizeof n, "27. efectos parte %d de 3", caps[k]);
            report(n);
            CHECK(I()->full_ready && I()->our_jobs > 1000, "parte %d: no se activaron los efectos", caps[k]);
            CHECK(C.last_flip_index >= 0 && C.last_flip_index < bb.nbuf,
                  "parte %d: el ultimo flip deberia ser a un buffer del juego (fue %d)", caps[k], C.last_flip_index);
            common_checks();
        }
    }

    // 28. generacion de VCFG (framegen = 1, la de por defecto): por fotograma del
    //     juego, un envio con el trabajo del juego + el nuestro y flip al intermedio, y
    //     despues el flip del real al buffer del juego (los dos terminan en la cola)
    {
        const char* c28 = "warmup_flips = 600\nframegen = 1\nsr = 0\naa = 0\nrcas = 0\nre = 0\n";
        sim_write("/data/vcfg_nx/config.ini", c28, strlen(c28));
        boot(&bb, 1);
        int rc0 = rate_calls;
        play(&bb, 2400, 0);
        report("28. generacion VCFG");
        CHECK(I()->our_jobs > 1500, "pocos fotogramas generados (%llu)", (unsigned long long)I()->our_jobs);
        CHECK(C.mid_flips > 1500 && C.pairs > 1500, "faltan flips intermedio + real (%d, %d)", C.mid_flips, C.pairs);
        CHECK(C.last_flip_index >= 0 && C.last_flip_index < bb.nbuf, "el ultimo flip deberia ser al buffer del juego (%d)",
              C.last_flip_index);
        CHECK(rate_calls - rc0 >= 1, "no forzo la salida a 60 Hz");
        CHECK(!I()->full_ready, "con solo la generacion VCFG no hace falta la memoria grande de los efectos");
        common_checks();
        // con la GPU trabada se pausa (y no se cuelga)
        C.gpu_stuck = 1;
        play(&bb, 200, 0);
        CHECK(I()->state == NX_ST_DEGRADED, "con la GPU trabada deberia pausarse");
        C.gpu_stuck = 0;
        // con poca memoria no se activa
        game_t lowm = bb; lowm.game_mb = bb.pool_mb - 100;
        boot(&lowm, 1);
        play(&lowm, 2400, 0);
        report("    ...poca memoria");
        CHECK(I()->our_jobs < 10 && C.mid_flips == 0, "con poca memoria no deberia generar (%llu)", (unsigned long long)I()->our_jobs);
        sim_write("/data/vcfg_nx/config.ini", "warmup_flips = 600\nframegen = 0\n", 32);
    }

    // 30. generacion VCFG + reescalado + SMAA + nitidez (2.73): el real se procesa en
    //     su buffer antes de generar; la memoria grande se reserva y sin intermedios propios
    {
        const char* c30 = "warmup_flips = 600\nframegen = 1\nsr = 2\naa = 1\nrcas = 2\n";
        sim_write("/data/vcfg_nx/config.ini", c30, strlen(c30));
        boot(&bb, 1);
        uint64_t g0 = C.groups_total;
        play(&bb, 2400, 0);
        report("30. VCFG + NXSR + SMAA + FSR");
        CHECK(I()->full_ready, "con post-proceso y generacion se reserva la memoria de los efectos");
        CHECK(I()->our_jobs > 1500 && C.mid_flips > 1500 && C.pairs > 1500, "VCFG+post: faltan flips (%llu, %d, %d)",
              (unsigned long long)I()->our_jobs, C.mid_flips, C.pairs);
        CHECK(C.groups_total - g0 > 1000000, "VCFG+post: el post-proceso no corrio");
        CHECK(strstr(I()->status, "activo") != NULL, "estado: %s", I()->status);
        common_checks();
        sim_write("/data/vcfg_nx/config.ini", "warmup_flips = 600\nframegen = 0\n", 32);
    }

    // 29. generacion IA (NXFG-2) con la presentacion de VCFG: los mismos dos flips por
    //     fotograma, la cadena propia escribe el intermedio
    {
        sim_write("/data/vcfg_nx/config.ini", "warmup_flips = 600\nframegen = 2\nfps_counter = 1\n", 46);
        boot(&bb, 1);
        uint64_t g0 = C.groups_total;
        play(&bb, 2400, 0);
        report("29. generacion IA");
        CHECK(I()->our_jobs > 1500 && C.mid_flips > 1500 && C.pairs > 1500, "IA: faltan flips intermedio + real (%llu, %d, %d)",
              (unsigned long long)I()->our_jobs, C.mid_flips, C.pairs);
        CHECK(C.groups_total - g0 > 1000000, "IA: casi no hubo trabajo de GPU (la red no corrio)");
        common_checks();
        sim_write("/data/vcfg_nx/config.ini", "warmup_flips = 600\nframegen = 0\n", 32);
    }

    // 31. generacion VCFG + nitidez con la GPU justa y la pantalla tardando en soltar
    //     el intermedio: esa espera no es carga (hasta la 2.74 contaba y la calidad
    //     dinamica apagaba los efectos)
    {
        const char* c31 = "warmup_flips = 600\nframegen = 1\nsr = 0\naa = 0\nrcas = 2\n";
        sim_write("/data/vcfg_nx/config.ini", c31, strlen(c31));
        boot(&bb, 1);
        C.game_cost_us = 24000; C.ns_per_kinst = 100; C.flip_wait_us = 9000;
        play(&bb, 3000, 0);
        printf("  31. VCFG + FSR, espera de pantalla 9 ms: GPU propia %.1f ms, espera %.1f ms, nivel -%d\n",
               (double)I()->gpu_ms, (double)I()->wait_ms, I()->level);
        CHECK(I()->gpu_ms > 0.0f && I()->gpu_ms < 8.0f, "el tiempo de GPU medido incluye la espera (%.1f ms)", (double)I()->gpu_ms);
        CHECK(I()->wait_ms > 5.0f, "no se midio la espera a la pantalla (%.1f ms)", (double)I()->wait_ms);
        CHECK(I()->level == 0, "bajo la calidad por una espera que no es carga (nivel -%d)", I()->level);
        CHECK(C.mid_flips > 2000, "la generacion no siguio (%d)", C.mid_flips);
        common_checks();
    }
    // 32. IA + NXSR-L + SMAA + FSR con la GPU justa: la calidad dinamica baja efectos
    //     y, si hace falta, pasa la IA a VCFG, pero la generacion nunca se apaga
    {
        const char* c32 = "warmup_flips = 600\nframegen = 2\nsr = 2\naa = 1\nrcas = 2\n";
        sim_write("/data/vcfg_nx/config.ini", c32, strlen(c32));
        boot(&bb, 1);
        C.game_cost_us = 24000; C.ns_per_kinst = 100; C.flip_wait_us = 2000;
        play(&bb, 6000, 0);
        int m0 = C.mid_flips;
        play(&bb, 1000, 0);
        printf("  32. IA + todo con la GPU justa: GPU propia %.1f ms, nivel -%d de %d, intermedios %d/1000\n",
               (double)I()->gpu_ms, I()->level, I()->level_max, C.mid_flips - m0);
        CHECK(C.mid_flips - m0 > 900, "la carga apago la generacion (%d intermedios)", C.mid_flips - m0);
        CHECK(I()->state == NX_ST_RUN, "quedo en estado %d", (int)I()->state);
        CHECK(I()->gpu_ms < 9.0f, "no bajo la carga lo suficiente (%.1f ms)", (double)I()->gpu_ms);
        common_checks();
        C.flip_wait_us = 0; C.ns_per_kinst = 0;
        sim_write("/data/vcfg_nx/config.ini", "warmup_flips = 600\nframegen = 0\n", 32);
    }

    // 33. todo a la vez (2.76): IA + realismo NXRE + NXSR-L + SMAA + FSR+ con un juego
    //     tipo GTA V (20 ms de GPU por fotograma): entra todo, sin "carga alta"
    {
        const char* c33 = "warmup_flips = 600\nframegen = 2\nre = 2\nsr = 2\naa = 1\nrcas = 1\n";
        sim_write("/data/vcfg_nx/config.ini", c33, strlen(c33));
        boot(&bb, 1);
        C.game_cost_us = 20000; C.ns_per_kinst = 100; C.flip_wait_us = 2000;
        C.notifies = 0; C.last_notify[0] = 0;
        play(&bb, 6000, 0);
        int m0 = C.mid_flips;
        play(&bb, 1000, 0);
        printf("  33. todo a la vez (juego 20 ms): GPU propia %.1f ms, nivel -%d de %d, intermedios %d/1000, efectos 0x%x\n",
               (double)I()->gpu_ms, I()->level, I()->level_max, C.mid_flips - m0, I()->menu_open);
        CHECK(I()->level == 0, "bajo la calidad sin que el juego lo necesite (nivel -%d, %s)", I()->level, C.last_notify);
        CHECK(!strstr(C.last_notify, "carga alta"), "aviso de carga alta: %s", C.last_notify);
        CHECK(C.mid_flips - m0 > 950, "faltan intermedios (%d)", C.mid_flips - m0);
        CHECK(I()->lite == 1, "deberia usar NXSR-L");
        common_checks();
        // el mismo juego con 26 ms de GPU: ya no entra todo; baja escalones pero la
        // generacion sigue
        C.game_cost_us = 26000;
        play(&bb, 6000, 0);
        m0 = C.mid_flips;
        play(&bb, 1000, 0);
        printf("      ...juego de 26 ms: GPU propia %.1f ms, nivel -%d de %d, intermedios %d/1000\n",
               (double)I()->gpu_ms, I()->level, I()->level_max, C.mid_flips - m0);
        CHECK(C.mid_flips - m0 > 900, "la carga apago la generacion (%d)", C.mid_flips - m0);
        common_checks();
        C.flip_wait_us = 0; C.ns_per_kinst = 0; C.game_cost_us = 0;
        sim_write("/data/vcfg_nx/config.ini", "warmup_flips = 600\nframegen = 0\n", 32);
    }

    // 34. lo que pediste en la 2.77: generacion NXFG-3 + realismo NXRE con un juego pesado
    //     (28 ms de GPU de los 33 que hay a 30 fps): entra sin bajar nada
    {
        const char* c34 = "warmup_flips = 600\nframegen = 2\nre = 2\nsr = 0\naa = 0\nrcas = 0\n";
        sim_write("/data/vcfg_nx/config.ini", c34, strlen(c34));
        boot(&bb, 1);
        C.game_cost_us = 28000; C.ns_per_kinst = 100; C.flip_wait_us = 2000;
        C.notifies = 0; C.last_notify[0] = 0;
        play(&bb, 6000, 0);
        int m0 = C.mid_flips;
        play(&bb, 1000, 0);
        printf("  34. NXFG-3 + realismo, juego de 28 ms: GPU propia %.1f ms, nivel -%d de %d, intermedios %d/1000\n",
               (double)I()->gpu_ms, I()->level, I()->level_max, C.mid_flips - m0);
        CHECK(I()->level == 0, "bajo la calidad (nivel -%d, %s)", I()->level, C.last_notify);
        CHECK(!C.notifies || !strstr(C.last_notify, "carga"), "aviso: %s", C.last_notify);
        CHECK(C.mid_flips - m0 > 950, "faltan intermedios (%d)", C.mid_flips - m0);
        CHECK(I()->gpu_ms < 6.0f, "NXFG-3 + NXRE (con las copias de la presentacion) deberia costar menos de 6 ms (%.1f)", (double)I()->gpu_ms);
        common_checks();
        C.flip_wait_us = 0; C.ns_per_kinst = 0; C.game_cost_us = 0;
        sim_write("/data/vcfg_nx/config.ini", "warmup_flips = 600\nframegen = 0\n", 32);
    }

    // 10. juego que arranca en un proceso que no es un juego
    game_t sh = bb; sh.title = "NPXS20001";
    boot(&sh, 0);
    play(&sh, 1000, 0);
    report("10. proceso que no es un juego");
    CHECK(I()->state == NX_ST_OFF && C.flips == 1000, "deberia quedar inactivo");

    printf(fails ? "\n%d FALLOS\n" : "\nOK: todos los escenarios pasaron\n", fails);
    return fails != 0;
}
