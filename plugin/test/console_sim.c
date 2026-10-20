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
    // game_cost_us por fotograma y lo nuestro ns_per_group por grupo despachado
    uint32_t game_cost_us, ns_per_group;
    uint64_t gpu_free_at;
    struct { volatile uint32_t* addr; uint32_t val; uint64_t at; } pend[16];
    int npend;
    int pro;
    int temp;                         // grados C (0 = el sistema no la informa)
    uint64_t groups_total;
    int game_dcb_submits, flips, bad_flips, our_dcbs, bad_dcbs;
    int last_flip_index;
    file_t files[16];
    int nfiles;
    char last_notify[160];
    int notifies;
    int verbose;
} C;

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("    FALLO: " __VA_ARGS__); printf("\n"); ++fails; } } while (0)

static uint32_t game_dcb_mark[4] = { 0xC0001000u, 0, 0, 0 };   // NOP con un dword: "DCB del juego"

static void* sim_alloc(size_t len, size_t align)
{
    (void)align;
    len = (len + 0xFFFF) & ~(size_t)0xFFFF;
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

static int sim_rate(int h, int r) { (void)h; (void)r; return 0; }

static void gpu_resolve(void)
{
    int k = 0;
    for (int i = 0; i < C.npend; ++i) {
        if (C.pend[i].at <= C.clock_us) *C.pend[i].addr = C.pend[i].val;
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

// "ejecuta" un DCB: valida PM4 y procesa la escritura del label al final
static void run_dcb(uint32_t* b, uint32_t bytes)
{
    size_t n = bytes / 4, i = 0;
    int ok = 1;
    uint64_t groups = 0;
    while (i < n) {
        uint32_t h = b[i];
        if (h == 0x80000000u) { ++i; continue; }
        if ((h >> 30) != 3) { ok = 0; break; }
        uint32_t op = (h >> 8) & 0xFF, cnt = ((h >> 16) & 0x3FFF) + 1;
        if (i + 1 + cnt > n) { ok = 0; break; }
        if (op == PM4_DISPATCH_DIRECT) { groups += (uint64_t)b[i + 1] * b[i + 2] * b[i + 3]; C.groups_total += (uint64_t)b[i + 1] * b[i + 2] * b[i + 3]; }
        if (op == PM4_EVENT_WRITE_EOP) {   // GPU trabada: queda colgado hasta que se destrabe
            uint64_t addr = b[i + 2] | (uint64_t)(b[i + 3] & 0xFFFF) << 32;
            uint64_t at = C.gpu_stuck ? UINT64_MAX : gpu_enqueue(groups * C.ns_per_group / 1000);
            groups = 0;
            if (C.npend < 16) {
                C.pend[C.npend].addr = (volatile uint32_t*)(uintptr_t)addr;
                C.pend[C.npend].val = b[i + 4];
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

static int sim_submit(const nx_submit_t* s) { process(s); return 0; }

static int cpu_flips;

static int sim_submit_flip(const nx_submit_t* s)
{
    if (s->cpu_flip && s->count == 0) cpu_flips++;
    process(s);
    C.flips++;
    if (s->index >= 16 || !C.registered[s->index]) C.bad_flips++;
    C.last_flip_index = (int)s->index;
    return 0;
}

static uint64_t sim_now(void) { gpu_resolve(); return C.clock_us; }
static void sim_sleep(uint32_t us) { C.clock_us += us; gpu_resolve(); }
static int sim_is_pro(void) { return C.pro; }
static int sim_temp(void) { return C.temp ? C.temp : -1; }
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

static const nx_platform_t PF = { sim_alloc, sim_avail, sim_register, sim_rate, sim_submit, sim_submit_flip,
                                  sim_now, sim_sleep, sim_notify, sim_log, sim_read, sim_write, sim_is_pro, sim_temp };

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
    if (keep_files) { memcpy(C.files, files, sizeof files); C.nfiles = nfiles; }
    C.pool = g->pool_mb << 20;
    C.used = g->game_mb << 20;
    C.pro = boot_pro;
    nx_engine_init(&PF, g->title);
    free(g_fb);
    g_fb = calloc(1, 4096);
    void* addrs[8];
    for (int i = 0; i < g->nbuf; ++i) addrs[i] = g_fb + i * 256;
    nx_vo_attr_t a = { g->format, 0, 0, (uint32_t)g->W, (uint32_t)g->H, (uint32_t)((g->W + 63) / 64 * 64) };
    int rc = sim_register(0, 0, addrs, g->nbuf, &a);
    nx_engine_on_register(0, 0, addrs, g->nbuf, &a, rc);
}

static uint32_t pad_buttons;

static void play(const game_t* g, int frames, int workload)
{
    void* dcb[1] = { game_dcb_mark };
    uint32_t sz[1] = { 8 };
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
    CHECK(C.flips >= 2700 + (int)I()->our_jobs - 10, "faltan flips: %d", C.flips);
    CHECK(strstr(find_file("/data/vcfg_nx/guard-CUSA00207.txt")->data, "state=ok") != NULL, "la sesion no quedo marcada ok");
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
    CHECK(I()->level == I()->level_max && I()->level_max == 3, "no bajo todos los escalones (%d/%d)", I()->level, I()->level_max);
    int flips_before = C.flips;
    uint64_t jobs_before = I()->our_jobs;
    play(&bb, 100, 0);
    CHECK(C.flips == flips_before + 100 && I()->our_jobs == jobs_before, "sin efectos deberia ser un flip por fotograma");
    common_checks();

    // 3b. GPU trabada con el contador de FPS (queda trabajo propio): pausa y reintento
    boot(&bb, 0);
    sim_write("/data/vcfg_nx/config.ini", "fps_counter = 1\n", 16);
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
    play(&bb, 6000, 0);
    report("3b. GPU trabada -> pausa -> reintento");
    CHECK(I()->level == 0, "no recupero toda la calidad tras el reintento (nivel -%d)", I()->level);
    common_checks();

    // 4. cierre del juego con efectos activos -> la proxima sesion arranca en modo seguro
    boot(&bb, 0);
    play(&bb, 900, 0);   // efectos activos, todavia no "estable"
    CHECK(strstr(find_file("/data/vcfg_nx/guard-CUSA00207.txt")->data, "pending") != NULL, "sin marca pendiente");
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
    uint32_t b = 0x4000; uint8_t st[4] = { 1, 2, 3, 4 }, tr[2] = { 9, 9 };
    nx_engine_on_pad(&b, st, tr);
    CHECK(b == 0 && st[0] == 0x80 && tr[0] == 0, "el juego recibio input con el menu abierto");
    pad_buttons = 0; play(&bb, 1, 0);
    {   // grilla del menu para el preview (plugin/test/osd_preview.py)
        FILE* mf = fopen("menu_grid.bin", "wb");
        if (mf && I()->menu_text) { fwrite(I()->menu_text, 1, (size_t)I()->menu_cols * I()->menu_rows, mf); }
        if (mf) fclose(mf);
    }
    for (int k = 0; k < 8; ++k) { pad_buttons = 0x0040; play(&bb, 1, 0); pad_buttons = 0; play(&bb, 1, 0); }
    pad_buttons = 0x4000; play(&bb, 1, 0); pad_buttons = 0; play(&bb, 1, 0);   // X en "Guardar"
    file_t* ini = find_file("/data/vcfg_nx/CUSA00207.ini");
    CHECK(ini && strstr(ini->data, "framegen = 0") && strstr(ini->data, "sr = 2"), "no se guardo el ini del juego (con FG apagado por la X)");
    pad_buttons = 0x2000; play(&bb, 1, 0); pad_buttons = 0; play(&bb, 1, 0);   // O cierra
    CHECK(!I()->menu_open, "O no cerro el menu");
    report("5. menu y guardado");
    common_checks();

    // 6. juego HDR: no se toca nada
    game_t hdr = bb; hdr.format = (int32_t)0x88740000;
    boot(&hdr, 0);
    play(&hdr, 2000, 0);
    report("6. juego HDR");
    CHECK(I()->state == NX_ST_OFF && C.used == (size_t)hdr.game_mb << 20 && C.flips == 2000, "HDR no quedo intacto");
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
    uint32_t sz[3] = { 8, 8, 8 };
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
    // activo no alcanza; la calidad dinamica baja hasta que alcanza y no se apaga
    boot(&bb, 0);
    C.game_cost_us = 24000; C.ns_per_group = 60;   // FG + SMAA + NXSR a 1080p ~ 12 ms
    play(&bb, 4000, 0);
    report("12. GPU justa: calidad dinamica");
    CHECK(I()->state == NX_ST_RUN && I()->level > 0 && I()->level < I()->level_max,
          "deberia quedar activo con calidad parcial (nivel -%d de %d)", I()->level, I()->level_max);
    uint64_t st12 = I()->stalls;
    CHECK(st12 < 60, "demasiados fotogramas atrasados (%llu)", (unsigned long long)st12);
    jobs_before = I()->our_jobs;
    play(&bb, 3000, 0);
    CHECK(I()->our_jobs > jobs_before + 2500, "dejo de procesar");
    CHECK(I()->stalls - st12 <= 3 * 2, "sigue atrasandose (%llu)", (unsigned long long)(I()->stalls - st12));
    // la escena se aligera (8 ms de GPU): vuelve a subir a la calidad completa
    C.game_cost_us = 8000;
    play(&bb, 16000, 0);
    report("    ...escena liviana: sube de nuevo");
    CHECK(I()->level == 0, "no volvio a la calidad completa (nivel -%d)", I()->level);
    common_checks();

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
        uint32_t sz1[1] = { 8 };
        nx_submit_t s14 = { 1, dcb1, sz1, NULL, NULL, 0, (uint32_t)(4 + f % 3), 1, f, 0, 0, 0 };
        nx_engine_on_submit_flip(&s14);
        C.clock_us += 33333;
        gpu_resolve();
    }
    report("14. re-registro antes de activar efectos");
    CHECK(I()->state == NX_ST_RUN && I()->full_ready && I()->our_jobs > 1000, "no se activo con los buffers nuevos");
    (void)g14;
    common_checks();

    // 15. modelo NXSR: Auto en PS4 Pro usa Calidad; en PS4 normal, Rapido
    boot_pro = 1;
    boot(&bb, 0);
    play(&bb, 1200, 0);
    CHECK(I()->hq == 1, "Auto en Pro deberia usar el modelo Calidad");
    boot_pro = 0;
    boot(&bb, 0);
    play(&bb, 1200, 0);
    CHECK(I()->hq == 0, "Auto en PS4 normal deberia usar el modelo Rapido");
    sim_write("/data/vcfg_nx/CUSA00207.ini", "sr_model = 2\n", 13);
    sim_write("/data/vcfg_nx/guard-CUSA00207.txt", "state=ok\ncrashes=0\n", 19);   // salio bien
    boot(&bb, 1);
    play(&bb, 1200, 0);
    report("15. modelo IA (Auto / Calidad)");
    CHECK(I()->hq == 1, "sr_model = 2 deberia usar Calidad");
    common_checks();

    // 16. temperatura: sube por encima del limite, luego critica, luego enfria
    sim_write("/data/vcfg_nx/CUSA00207.ini", "fps_counter = 1\n", 16);
    sim_write("/data/vcfg_nx/guard-CUSA00207.txt", "state=ok\ncrashes=0\n", 19);
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

    // 17. modo ahorro: sin modelo Calidad ni SMAA aunque sea PS4 Pro
    sim_write("/data/vcfg_nx/CUSA00207.ini", "eco = 1\n", 8);
    sim_write("/data/vcfg_nx/guard-CUSA00207.txt", "state=ok\ncrashes=0\n", 19);
    boot_pro = 1;
    boot(&bb, 1);
    play(&bb, 1200, 0);
    uint64_t g0 = C.groups_total, j0 = I()->our_jobs;
    play(&bb, 100, 0);
    uint64_t eco_groups = (C.groups_total - g0) / (I()->our_jobs - j0);
    CHECK(I()->hq == 0, "modo ahorro uso el modelo Calidad");
    sim_write("/data/vcfg_nx/CUSA00207.ini", "eco = 0\n", 8);
    sim_write("/data/vcfg_nx/guard-CUSA00207.txt", "state=ok\ncrashes=0\n", 19);
    boot(&bb, 1);
    play(&bb, 1200, 0);
    g0 = C.groups_total; j0 = I()->our_jobs;
    play(&bb, 100, 0);
    uint64_t full_groups = (C.groups_total - g0) / (I()->our_jobs - j0);
    boot_pro = 0;
    printf("  17. modo ahorro: %llu grupos de GPU por fotograma (normal en Pro: %llu, -%.0f%%)\n",
           (unsigned long long)eco_groups, (unsigned long long)full_groups,
           100.0 * (1.0 - (double)eco_groups / (double)full_groups));
    CHECK(eco_groups < full_groups, "modo ahorro no redujo el trabajo");
    common_checks();

    // 10. juego que arranca en un proceso que no es un juego
    game_t sh = bb; sh.title = "NPXS20001";
    boot(&sh, 0);
    play(&sh, 1000, 0);
    report("10. proceso que no es un juego");
    CHECK(I()->state == NX_ST_OFF && C.flips == 1000, "deberia quedar inactivo");

    printf(fails ? "\n%d FALLOS\n" : "\nOK: todos los escenarios pasaron\n", fails);
    return fails != 0;
}
