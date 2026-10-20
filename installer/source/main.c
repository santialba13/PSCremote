// VCFG-NX 2.2 - app de configuracion e instalacion (estilo del lanzador de VCFG).
//
// Cuadricula con los juegos instalados (icono, titulo, ON/OFF). Cada juego
// puede usar los ajustes por defecto o tener los suyos (/data/vcfg_nx/<ID>.ini,
// el mismo archivo que guarda el menu L3+R3 dentro del juego). Al guardar:
//   * copia /app0/plugins/vcfg_nx.prx a /data/GoldHEN/plugins/
//   * hace una copia de seguridad de plugins.ini (una por sesion) y lo reescribe
//     solo con los cambios de VCFG-NX, verificando que se haya escrito bien
//   * escribe config.ini (por defecto) y los .ini de los juegos que cambiaron
// La logica de archivos esta en inst_core.c y ../plugin/source/config.c
// (testeadas en el PC). La pantalla solo se redibuja cuando algo cambia: sin
// tocar nada la app no usa CPU.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include <orbis/Pad.h>
#include <orbis/SystemService.h>
#include <orbis/UserService.h>
#include <orbis/VideoOut.h>
#include <orbis/libkernel.h>

#include <GoldHEN.h>

#include "../../plugin/include/nx_font.h"
#include "inst_core.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wsign-compare"
#include "../../third_party/stb/stb_image.h"
#pragma clang diagnostic pop

#define W 1920
#define H 1080
// ROOT: prefijo de todas las rutas (vacio en la consola; una carpeta temporal en
// installer/test/test_app.c, que maneja la app entera en el PC)
#ifndef ROOT
#define ROOT ""
#endif
#define PLUGIN_SRC ROOT "/app0/plugins/vcfg_nx.prx"
#define GOLDHEN_DIR ROOT "/data/GoldHEN"
#define PLUGIN_DIR ROOT "/data/GoldHEN/plugins"
#define PLUGINS_INI ROOT "/data/GoldHEN/plugins.ini"
#define CFG_DIR ROOT "/data/vcfg_nx"
#define VERSION "2.2"

#define C_BG 0xFF0E1220u
#define C_PANEL 0xFF161C2Cu
#define C_CARD 0xFF1A2236u
#define C_SEL 0xFF1E4A9Eu
#define C_TEXT 0xFFF0F4FFu
#define C_DIM 0xFF8A96B4u
#define C_FAINT 0xFF4A5470u
#define C_ACCENT 0xFF5AA0FFu
#define C_OK 0xFF3CB46Eu
#define C_WARN 0xFFFFB45Au
#define C_OFF 0xFF3A4258u
// colores de los botones del mando
#define C_BTN_X 0xFF5A8CFFu
#define C_BTN_O 0xFFFF5A64u
#define C_BTN_SQ 0xFFE68CC8u
#define C_BTN_TRI 0xFF46D2A0u
#define C_BTN_SYS 0xFF5A6482u

static uint32_t* fb[2];
static int video, cur_fb;
static int pad;

// ------------------------------------------------------------- dibujo
static void rect(int x, int y, int w, int h, uint32_t c)
{
    int x0 = x < 0 ? 0 : x, y0 = y < 0 ? 0 : y, x1 = x + w > W ? W : x + w, y1 = y + h > H ? H : y + h;
    uint32_t* p = fb[cur_fb];
    for (int j = y0; j < y1; ++j)
        for (int i = x0; i < x1; ++i) p[j * W + i] = c;
}

static void frame(int x, int y, int w, int h, int t, uint32_t c)
{
    rect(x, y, w, t, c);
    rect(x, y + h - t, w, t, c);
    rect(x, y, t, h, c);
    rect(x + w - t, y, t, h, c);
}

static int text(int x, int y, int s, uint32_t c, const char* fmt, ...)
{
    char b[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    uint32_t* p = fb[cur_fb];
    for (const unsigned char* t = (const unsigned char*)b; *t; ++t, x += 8 * s) {
        if (*t < NX_FONT_FIRST) continue;
        const uint8_t* g = nx_font8x16 + (*t - NX_FONT_FIRST) * 16;
        for (int r = 0; r < 16; ++r)
            for (int k = 0; k < 8; ++k)
                if ((g[r] >> (7 - k)) & 1)
                    for (int dy = 0; dy < s; ++dy)
                        for (int dx = 0; dx < s; ++dx) {
                            int px = x + k * s + dx, py = y + r * s + dy;
                            if (px >= 0 && px < W && py >= 0 && py < H) p[py * W + px] = c;
                        }
    }
    return x;
}

// etiqueta con fondo (boton del mando, estado); devuelve el x siguiente
static int chip(int x, int y, int s, uint32_t bg, uint32_t fg, const char* label)
{
    int w = (int)strlen(label) * 8 * s + 12 * s;
    rect(x, y, w, 16 * s + 6, bg);
    text(x + 6 * s, y + 3, s, fg, "%s", label);
    return x + w;
}

// boton + accion para el pie de pantalla
static int hint(int x, int y, uint32_t bg, const char* btn, const char* what)
{
    x = chip(x, y, 2, bg, 0xFF0E1220u, btn);
    return text(x + 10, y + 3, 2, C_TEXT, "%s", what) + 34;
}

static int video_init(void)
{
    video = sceVideoOutOpen(ORBIS_VIDEO_USER_MAIN, ORBIS_VIDEO_OUT_BUS_MAIN, 0, 0);
    if (video < 0) return -1;
    size_t sz = (size_t)W * H * 4, total = (2 * sz + 0x1FFFFF) & ~(size_t)0x1FFFFF;
    off_t phys;
    void* mem = NULL;
    if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), total, 0x200000, 3, &phys) != 0) return -2;
    if (sceKernelMapDirectMemory(&mem, total, 0x33, 0, phys, 0x200000) != 0) return -3;
    fb[0] = mem;
    fb[1] = (uint32_t*)((uint8_t*)mem + sz);
    OrbisVideoOutBufferAttribute attr;
    sceVideoOutSetBufferAttribute(&attr, 0x80000000, 1, 0, W, H, W);   // A8R8G8B8_SRGB lineal
    sceVideoOutSetFlipRate(video, 0);
    return sceVideoOutRegisterBuffers(video, 0, (void**)fb, 2, &attr) == 0 ? 0 : -4;
}

static void present(void)
{
    sceVideoOutSubmitFlip(video, cur_fb, ORBIS_VIDEO_OUT_FLIP_VSYNC, 0);
    cur_fb ^= 1;
    sceKernelUsleep(17000);   // el flip ocurre en el proximo vsync: recien ahi se puede dibujar en el otro buffer
}

// ------------------------------------------------------------ archivos
static char* read_text(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) { char* e = malloc(1); e[0] = 0; return e; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* b = malloc((size_t)(n > 0 ? n : 0) + 1);
    size_t r = n > 0 ? fread(b, 1, (size_t)n, f) : 0;
    b[r] = 0;
    fclose(f);
    return b;
}

static int write_text_verified(const char* path, const char* text)
{
    char tmp[160];
    snprintf(tmp, sizeof tmp, "%s.vcfg_tmp", path);
    FILE* f = fopen(tmp, "wb");
    if (!f) return -1;
    size_t n = strlen(text);
    int ok = fwrite(text, 1, n, f) == n;
    if (fclose(f) != 0) ok = 0;
    char* back = ok ? read_text(tmp) : NULL;
    ok = ok && back && strcmp(back, text) == 0;
    free(back);
    if (!ok) { remove(tmp); return -2; }
    remove(path);
    return rename(tmp, path) == 0 ? 0 : -3;
}

static int copy_file(const char* src, const char* dst)
{
    FILE* in = fopen(src, "rb");
    if (!in) return -1;
    char tmp[160];
    snprintf(tmp, sizeof tmp, "%s.vcfg_tmp", dst);
    FILE* out = fopen(tmp, "wb");
    if (!out) { fclose(in); return -2; }
    static char buf[64 * 1024];
    size_t n;
    int err = 0;
    while ((n = fread(buf, 1, sizeof buf, in)) > 0)
        if (fwrite(buf, 1, n, out) != n) { err = -3; break; }
    fclose(in);
    if (fclose(out) != 0) err = -4;
    if (err) { remove(tmp); return err; }
    remove(dst);
    return rename(tmp, dst) == 0 ? 0 : -5;
}

// --------------------------------------------------------------- estado
static inst_game_t games[INST_MAX_GAMES];
static int ngames, dirty, backup_done;
static nx_config_t defaults;
static char msg[160];
static uint32_t msg_color = C_DIM;

enum { F_ALL, F_ON, F_CUSTOM, F_COUNT };
static int filter, vis[INST_MAX_GAMES], nvis, sel, top_row;

static void set_msg(uint32_t c, const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    msg_color = c;
}

static void rebuild_visible(void)
{
    int keep = nvis ? vis[sel] : -1;
    nvis = 0;
    for (int i = 0; i < ngames; ++i)
        if (filter == F_ALL || (filter == F_ON && games[i].enabled) || (filter == F_CUSTOM && games[i].custom))
            vis[nvis++] = i;
    sel = 0;
    for (int k = 0; k < nvis; ++k)
        if (vis[k] == keep) sel = k;
}

static void load_defaults(void)
{
    nx_config_defaults(&defaults);
    char* t = read_text(CFG_DIR "/config.ini");
    nx_config_parse(&defaults, t);
    free(t);
}

static int save_all(int uninstall)
{
    mkdir(GOLDHEN_DIR, 0777);
    mkdir(PLUGIN_DIR, 0777);
    mkdir(CFG_DIR, 0777);
    char* ini = read_text(PLUGINS_INI);
    if (!backup_done && ini[0]) {
        char bak[128];
        snprintf(bak, sizeof bak, PLUGINS_INI ".vcfg_nx_backup_%ld", (long)time(NULL));
        if (copy_file(PLUGINS_INI, bak) != 0) { free(ini); return -10; }
        backup_done = 1;
    }
    char* cur = ini;
    if (uninstall) {
        char* n = inst_ini_remove_all(cur);
        if (n) { free(cur); cur = n; }
    } else {
        for (int i = 0; i < ngames; ++i) {
            char* n = games[i].enabled ? inst_ini_add(cur, games[i].id) : inst_ini_remove(cur, games[i].id);
            if (n) { free(cur); cur = n; }
        }
    }
    int r = write_text_verified(PLUGINS_INI, cur);
    free(cur);
    if (r != 0) return -20 + r;
    if (uninstall) {
        remove(PLUGIN_DIR "/vcfg_nx.prx");
        return 0;
    }
    if (copy_file(PLUGIN_SRC, PLUGIN_DIR "/vcfg_nx.prx") != 0) return -30;
    char cfg[768];
    nx_config_format(&defaults, "VCFG-NX: ajustes por defecto (los juegos con ajustes propios usan su <ID>.ini)",
                     cfg, sizeof cfg);
    if (write_text_verified(CFG_DIR "/config.ini", cfg) != 0) return -40;
    for (int i = 0; i < ngames; ++i) {
        inst_game_t* g = &games[i];
        if (!g->cfg_dirty) continue;
        char path[96];
        snprintf(path, sizeof path, CFG_DIR "/%s.ini", g->id);
        if (g->custom) {
            char head[96];
            snprintf(head, sizeof head, "VCFG-NX: ajustes propios de %.40s", g->title);
            nx_config_format(&g->cfg, head, cfg, sizeof cfg);
            if (write_text_verified(path, cfg) != 0) return -50;
            g->custom_saved = 1;
        } else if (g->custom_saved) {
            remove(path);
            g->custom_saved = 0;
        }
        g->cfg_dirty = 0;
    }
    return 0;
}

// --------------------------------------------------------------- iconos
// icon0.png de cada juego (512x512) reducido a ICON x ICON; se decodifican solo
// los que se ven y se guardan los ultimos ICON_CACHE.
#define ICON 176
#define ICON_CACHE 32
typedef struct { int game; uint32_t* px; int failed; unsigned used; } icon_t;
static icon_t icons[ICON_CACHE];
static unsigned icon_clock;

static uint32_t* decode_icon(const char* path)
{
    int w, h, n;
    unsigned char* src = stbi_load(path, &w, &h, &n, 4);
    if (!src) return NULL;
    uint32_t* out = malloc((size_t)ICON * ICON * 4);
    if (!out) { stbi_image_free(src); return NULL; }
    // promedio por area: cada pixel de salida junta su bloque de la imagen original
    for (int y = 0; y < ICON; ++y) {
        int y0 = y * h / ICON, y1 = (y + 1) * h / ICON;
        if (y1 <= y0) y1 = y0 + 1;
        for (int x = 0; x < ICON; ++x) {
            int x0 = x * w / ICON, x1 = (x + 1) * w / ICON;
            if (x1 <= x0) x1 = x0 + 1;
            unsigned r = 0, g = 0, b = 0, a = 0, cnt = 0;
            for (int j = y0; j < y1; ++j)
                for (int i = x0; i < x1; ++i) {
                    const unsigned char* p = src + ((size_t)j * w + i) * 4;
                    r += p[0]; g += p[1]; b += p[2]; a += p[3]; ++cnt;
                }
            r /= cnt; g /= cnt; b /= cnt; a /= cnt;
            // sobre el fondo de la tarjeta (los iconos con transparencia)
            uint32_t bg = C_CARD;
            r = (r * a + ((bg >> 16) & 0xFF) * (255 - a)) / 255;
            g = (g * a + ((bg >> 8) & 0xFF) * (255 - a)) / 255;
            b = (b * a + (bg & 0xFF) * (255 - a)) / 255;
            out[y * ICON + x] = 0xFF000000u | r << 16 | g << 8 | b;
        }
    }
    stbi_image_free(src);
    return out;
}

static const uint32_t* game_icon(int gi)
{
    icon_t* slot = &icons[0];
    for (int k = 0; k < ICON_CACHE; ++k) {
        if (icons[k].used && icons[k].game == gi) {
            icons[k].used = ++icon_clock;
            return icons[k].px;
        }
        if (icons[k].used < slot->used) slot = &icons[k];
    }
    free(slot->px);
    slot->game = gi;
    slot->px = NULL;
    slot->used = ++icon_clock;
    if (games[gi].meta[0]) {
        char p[192];
        snprintf(p, sizeof p, "%s/icon0.png", games[gi].meta);
        slot->px = decode_icon(p);
    }
    return slot->px;
}

static void draw_icon(int gi, int x, int y)
{
    const uint32_t* px = game_icon(gi);
    if (!px) {   // sin icono: tarjeta con las iniciales
        rect(x, y, ICON, ICON, C_OFF);
        const char* t = games[gi].title;
        text(x + ICON / 2 - 32, y + ICON / 2 - 32, 4, C_DIM, "%c%c", t[0] ? t[0] : '?', t[0] && t[1] ? t[1] : ' ');
        return;
    }
    uint32_t* p = fb[cur_fb];
    for (int j = 0; j < ICON; ++j)
        if (y + j >= 0 && y + j < H) memcpy(p + (y + j) * W + x, px + j * ICON, ICON * 4);
}

// ---------------------------------------------------------------- ajustes
static const char* SR_N[] = { "OFF", "Suave", "Normal", "Fuerte" };
static const char* RC_N[] = { "OFF", "Baja", "Media", "Alta" };
static const char* MD_N[] = { "Auto", "R\xe1pido", "Calidad" };
static const int TEMPS[] = { 0, 70, 75, 80, 85, 90 };
#define NTEMPS 6

enum { R_ENABLED, R_MODE, R_FG, R_SR, R_MODEL, R_AA, R_RCAS, R_DYN, R_ECO, R_TEMP, R_FPS, R_COUNT };
static const char* ROW_LABEL[R_COUNT] = {
    "VCFG-NX en este juego", "Ajustes", "Generaci\xf3n de fotogramas (30 -> 60)", "Reescalado IA (NXSR)",
    "Modelo IA", "Antialiasing SMAA", "Nitidez FSR1 (RCAS, sin NXSR)", "Calidad din\xe1mica",
    "Modo ahorro (menos calor)", "L\xedmite de temperatura", "Contador de FPS",
};
static const char* ROW_HELP[R_COUNT] = {
    "Agrega o quita VCFG-NX de plugins.ini para este juego.",
    "Por defecto: sigue los ajustes generales. Propios: este juego guarda los suyos.",
    "Genera un fotograma intermedio: 30 fps se ven como 60. Se apaga solo si el juego ya va a m\xe1s de 48.",
    "Red entrenada que recupera detalle sin agregar grano ni halos (salida acotada a los p\xedxeles vecinos).",
    "Auto: Calidad en PS4 Pro y R\xe1pido en PS4 normal. Calidad cuesta ~3 veces m\xe1s de GPU.",
    "Suaviza los bordes dentados (SMAA 1x).",
    "Nitidez cl\xe1sica de AMD; solo se usa con NXSR apagado.",
    "Si la GPU no da abasto baja un escal\xf3n (Calidad > R\xe1pido > sin SMAA > sin NXSR > sin FG) y despu\xe9s sube.",
    "Modelo R\xe1pido, sin SMAA y l\xedmite de temperatura de 75\xb0" "C.",
    "Por encima baja la calidad; 6\xb0" "C m\xe1s pausa los efectos hasta que enfr\xed" "e.",
    "Muestra los FPS (y la temperatura) en una esquina.",
};

static int temp_index(int t)
{
    for (int i = 0; i < NTEMPS; ++i)
        if (TEMPS[i] == t) return i;
    return 3;
}

static void value_text(const nx_config_t* c, int row, char* out, size_t n)
{
    switch (row) {
    case R_FG: snprintf(out, n, "%s", c->framegen ? "ON" : "OFF"); break;
    case R_SR: snprintf(out, n, "%s", SR_N[c->sr & 3]); break;
    case R_MODEL: snprintf(out, n, "%s", MD_N[c->sr_model % 3]); break;
    case R_AA: snprintf(out, n, "%s", c->aa ? "ON" : "OFF"); break;
    case R_RCAS: snprintf(out, n, "%s", RC_N[c->rcas & 3]); break;
    case R_DYN: snprintf(out, n, "%s", c->dynamic ? "ON" : "OFF"); break;
    case R_ECO: snprintf(out, n, "%s", c->eco ? "ON" : "OFF"); break;
    case R_TEMP: if (c->temp_limit) snprintf(out, n, "%d\xb0" "C", c->temp_limit); else snprintf(out, n, "OFF"); break;
    case R_FPS: snprintf(out, n, "%s", c->fps_counter ? "ON" : "OFF"); break;
    default: out[0] = 0;
    }
}

static void change(nx_config_t* c, int row, int dir)
{
    switch (row) {
    case R_FG: c->framegen = !c->framegen; break;
    case R_SR: c->sr = (c->sr + (dir > 0 ? 1 : 3)) % 4; break;
    case R_MODEL: c->sr_model = (c->sr_model + (dir > 0 ? 1 : 2)) % 3; break;
    case R_AA: c->aa = !c->aa; break;
    case R_RCAS: c->rcas = (c->rcas + (dir > 0 ? 1 : 3)) % 4; break;
    case R_DYN: c->dynamic = !c->dynamic; break;
    case R_ECO: c->eco = !c->eco; break;
    case R_TEMP: c->temp_limit = TEMPS[(temp_index(c->temp_limit) + (dir > 0 ? 1 : NTEMPS - 1)) % NTEMPS]; break;
    case R_FPS: c->fps_counter = !c->fps_counter; break;
    default: break;
    }
}

static const nx_config_t* effective(const inst_game_t* g) { return g->custom ? &g->cfg : &defaults; }

static void summary(const nx_config_t* c, char* out, size_t n)
{
    char t[16];
    value_text(c, R_TEMP, t, sizeof t);
    snprintf(out, n, "FG %s  \xb7  NXSR %s (%s)  \xb7  SMAA %s  \xb7  Din\xe1mica %s  \xb7  Ahorro %s  \xb7  L\xedmite %s",
             c->framegen ? "ON" : "OFF", SR_N[c->sr & 3], MD_N[c->sr_model % 3], c->aa ? "ON" : "OFF",
             c->dynamic ? "ON" : "OFF", c->eco ? "ON" : "OFF", t);
}

// ---------------------------------------------------------------- vistas
#define COLS 7
#define ROWS 2
#define CARD_W 232
#define CARD_H 288
#define GRID_X 82
#define GRID_Y 190

static void header(void)
{
    rect(0, 0, W, H, C_BG);
    for (int y = 0; y < 112; ++y) rect(0, y, W, 1, 0xFF000000u | ((14 + y / 5) << 16) | ((18 + y / 3) << 8) | (32 + y / 2));
    text(60, 24, 4, C_TEXT, "VCFG-NX");
    chip(60 + 8 * 4 * 7 + 18, 40, 2, C_ACCENT, 0xFF0E1220u, VERSION);
    text(560, 46, 2, C_DIM, "Generaci\xf3n de fotogramas  \xb7  Reescalado IA (NXSR)  \xb7  SMAA / FSR1");
    rect(0, 112, W, 3, C_ACCENT);
}

static void footer(int (*hints)(int x, int y))
{
    rect(0, H - 112, W, 112, C_PANEL);
    text(60, H - 98, 2, msg_color, "%s", msg);
    hints(60, H - 52);
}

static int hints_grid(int x, int y)
{
    x = hint(x, y, C_BTN_X, "X", "activar");
    x = hint(x, y, C_BTN_SQ, "CUADRADO", "ajustes del juego");
    x = hint(x, y, C_BTN_TRI, "TRI\xc1NGULO", "por defecto");
    x = hint(x, y, C_BTN_SYS, "L1/R1", "filtro");
    x = hint(x, y, C_BTN_SYS, "OPTIONS", "m\xe1s");
    return hint(x, y, C_BTN_O, "O", "guardar y salir");
}

static void draw_grid(void)
{
    header();
    // pestanas de filtro
    static const char* names[F_COUNT] = { "Todos", "Activos", "Con ajustes propios" };
    int counts[F_COUNT] = { ngames, 0, 0 };
    for (int i = 0; i < ngames; ++i) { counts[F_ON] += games[i].enabled; counts[F_CUSTOM] += games[i].custom; }
    int x = 60;
    for (int f = 0; f < F_COUNT; ++f) {
        char l[48];
        snprintf(l, sizeof l, "%s (%d)", names[f], counts[f]);
        x = chip(x, 132, 2, f == filter ? C_ACCENT : C_PANEL, f == filter ? 0xFF0E1220u : C_DIM, l) + 14;
    }

    if (sel / COLS < top_row) top_row = sel / COLS;
    if (sel / COLS >= top_row + ROWS) top_row = sel / COLS - ROWS + 1;
    for (int r = 0; r < ROWS; ++r)
        for (int c = 0; c < COLS; ++c) {
            int k = (top_row + r) * COLS + c;
            if (k >= nvis) break;
            const inst_game_t* g = &games[vis[k]];
            int cx = GRID_X + c * (CARD_W + 22), cy = GRID_Y + r * (CARD_H + 16);
            int is_sel = k == sel;
            rect(cx, cy, CARD_W, CARD_H, is_sel ? C_SEL : C_CARD);
            if (is_sel) frame(cx - 4, cy - 4, CARD_W + 8, CARD_H + 8, 4, C_ACCENT);
            int ix = cx + (CARD_W - ICON) / 2, iy = cy + 12;
            draw_icon(vis[k], ix, iy);
            if (!g->enabled) {   // apagado: icono atenuado con una franja
                rect(ix, iy + ICON - 30, ICON, 30, 0xFF10141Eu);
                text(ix + 8, iy + ICON - 23, 1, C_DIM, "sin VCFG-NX");
            }
            chip(ix + 6, iy + 6, 1, g->enabled ? C_OK : C_OFF, g->enabled ? 0xFF0E1220u : C_DIM,
                 g->enabled ? "ON" : "OFF");
            if (g->custom) chip(ix + ICON - 60, iy + 6, 1, C_WARN, 0xFF0E1220u, "PROPIO");
            // titulo en dos lineas de 13 caracteres
            char l1[16] = "", l2[16] = "";
            const char* t = g->title;
            int len = (int)strlen(t);
            if (len <= 13) {
                snprintf(l1, sizeof l1, "%s", t);
            } else {
                int cut = 13;
                for (int i = 13; i > 6; --i)
                    if (t[i] == ' ') { cut = i; break; }
                snprintf(l1, sizeof l1, "%.*s", cut, t);
                const char* rest = t + cut + (t[cut] == ' ');
                if ((int)strlen(rest) > 13) snprintf(l2, sizeof l2, "%.11s..", rest);
                else snprintf(l2, sizeof l2, "%s", rest);
            }
            text(cx + 12, iy + ICON + 6, 2, C_TEXT, "%s", l1);
            text(cx + 12, iy + ICON + 38, 2, C_TEXT, "%s", l2);
            text(cx + 12, cy + CARD_H - 19, 1, is_sel ? C_TEXT : C_FAINT, "%s", g->id);
        }
    if (!nvis)
        text(GRID_X, GRID_Y + 40, 2, C_WARN, "%s", ngames ? "No hay juegos en este filtro (L1/R1 para cambiar)."
                                                         : "No se encontraron juegos en /user/app ni en el disco externo.");
    // barra de desplazamiento
    int total_rows = (nvis + COLS - 1) / COLS;
    if (total_rows > ROWS) {
        int bh = ROWS * (CARD_H + 16) - 16, th = bh * ROWS / total_rows;
        rect(W - 40, GRID_Y, 6, bh, C_PANEL);
        rect(W - 40, GRID_Y + (bh - th) * top_row / (total_rows - ROWS), 6, th, C_ACCENT);
    }
    // detalle del juego elegido
    rect(0, 794, W, 2, C_PANEL);
    if (nvis) {
        const inst_game_t* g = &games[vis[sel]];
        text(60, 808, 3, C_TEXT, "%.50s", g->title);
        int x2 = chip(60, 864, 2, g->enabled ? C_OK : C_OFF, g->enabled ? 0xFF0E1220u : C_DIM,
                      g->enabled ? "VCFG-NX activo" : "VCFG-NX apagado");
        x2 = chip(x2 + 12, 864, 2, g->custom ? C_WARN : C_PANEL, g->custom ? 0xFF0E1220u : C_DIM,
                  g->custom ? "ajustes propios" : "ajustes por defecto");
        text(x2 + 16, 867, 2, C_FAINT, "%s", g->id);
        char s[200];
        summary(effective(g), s, sizeof s);
        text(60, 914, 2, g->enabled ? C_DIM : C_FAINT, "%s", s);
    }
    footer(hints_grid);
}

static int settings_game = -1;

static int hints_settings(int x, int y)
{
    x = hint(x, y, C_BTN_SYS, "ARRIBA/ABAJO", "elegir");
    x = hint(x, y, C_BTN_SYS, "IZQ/DER", "cambiar");
    if (settings_game >= 0 && games[settings_game].custom) x = hint(x, y, C_BTN_SQ, "CUADRADO", "volver a por defecto");
    return hint(x, y, C_BTN_O, "O", "listo");
}

static int hints_defaults(int x, int y)
{
    x = hint(x, y, C_BTN_SYS, "ARRIBA/ABAJO", "elegir");
    x = hint(x, y, C_BTN_SYS, "IZQ/DER", "cambiar");
    return hint(x, y, C_BTN_O, "O", "listo");
}

// gi < 0: ajustes por defecto
static void draw_settings(int gi, int row)
{
    header();
    settings_game = gi;
    int first = gi < 0 ? R_FG : R_ENABLED;
    int y0;
    if (gi >= 0) {
        const inst_game_t* g = &games[gi];
        draw_icon(gi, 60, 136);
        text(260, 150, 3, C_TEXT, "%.48s", g->title);
        text(260, 204, 2, C_DIM, "%s", g->id);
        text(260, 244, 2, C_DIM, "%s", g->custom ? "Este juego usa sus propios ajustes."
                                                 : "Este juego sigue los ajustes por defecto (cambi\xe1 algo para tener propios).");
        y0 = 336;
    } else {
        text(60, 140, 3, C_TEXT, "Ajustes por defecto");
        text(60, 194, 2, C_DIM, "Los usan todos los juegos que no tienen ajustes propios.");
        y0 = 250;
    }
    const nx_config_t* c = gi >= 0 ? effective(&games[gi]) : &defaults;
    int inherited = gi >= 0 && !games[gi].custom;
    for (int r = first, i = 0; r < R_COUNT; ++r, ++i) {
        int y = y0 + i * 52;
        if (r == row) rect(40, y - 8, W - 80, 48, C_SEL);
        char v[32];
        uint32_t vc = C_ACCENT;
        if (r == R_ENABLED) {
            snprintf(v, sizeof v, "%s", games[gi].enabled ? "ON" : "OFF");
            vc = games[gi].enabled ? C_OK : C_DIM;
        } else if (r == R_MODE) {
            snprintf(v, sizeof v, "%s", games[gi].custom ? "Propios" : "Por defecto");
            vc = games[gi].custom ? C_WARN : C_DIM;
        } else {
            value_text(c, r, v, sizeof v);
            if (inherited) vc = C_DIM;
        }
        text(80, y, 2, r == row ? C_TEXT : (inherited && r > R_MODE ? C_DIM : C_TEXT), "%s", ROW_LABEL[r]);
        int vx = 1360;
        text(vx, y, 2, r == row ? C_TEXT : C_FAINT, "<");
        text(vx + 40, y, 2, vc, "%s", v);
        text(vx + 40 + 16 * 13, y, 2, r == row ? C_TEXT : C_FAINT, ">");
        if (r == R_MODE) rect(60, y + 44, W - 120, 1, C_PANEL);
    }
    set_msg(C_DIM, "%s", ROW_HELP[row]);
    footer(gi < 0 ? hints_defaults : hints_settings);
}

enum { O_ALL_ON, O_ALL_OFF, O_RESET_ALL, O_UNINSTALL, O_EXIT, O_BACK, O_COUNT };
static const char* OPT_LABEL[O_COUNT] = {
    "Activar VCFG-NX en todos los juegos", "Desactivar VCFG-NX en todos los juegos",
    "Quitar los ajustes propios de todos los juegos", "Desinstalar VCFG-NX", "Salir sin guardar", "Volver",
};

static int hints_menu(int x, int y)
{
    x = hint(x, y, C_BTN_X, "X", "elegir");
    return hint(x, y, C_BTN_O, "O", "volver");
}

static const char* OPT_HELP[O_COUNT] = {
    "Agrega VCFG-NX a plugins.ini en todos los juegos de la lista.",
    "Quita VCFG-NX de plugins.ini en todos los juegos (los ajustes se conservan).",
    "Todos los juegos pasan a usar los ajustes por defecto.",
    "Quita el plugin de GoldHEN. Los ajustes en /data/vcfg_nx se conservan.",
    "Cierra la app sin aplicar los cambios de esta sesi\xf3n.",
    "",
};

static void draw_options(int row)
{
    header();
    set_msg(row == O_UNINSTALL ? C_WARN : C_DIM, "%s", OPT_HELP[row]);
    text(60, 140, 3, C_TEXT, "Opciones");
    for (int i = 0; i < O_COUNT; ++i) {
        int y = 230 + i * 64;
        if (i == row) rect(40, y - 10, 1100, 52, C_SEL);
        text(80, y, 2, i == O_UNINSTALL ? C_WARN : C_TEXT, "%s", OPT_LABEL[i]);
    }
    footer(hints_menu);
}

static int hints_confirm(int x, int y)
{
    x = hint(x, y, C_BTN_X, "X", "confirmar");
    return hint(x, y, C_BTN_O, "O", "cancelar");
}

static void draw_confirm(int what)
{
    header();
    set_msg(C_DIM, "%s", "");
    if (what == O_UNINSTALL) {
        text(60, 260, 3, C_WARN, "\xbf" "Desinstalar VCFG-NX?");
        text(60, 340, 2, C_TEXT, "Se quita el plugin de plugins.ini (con copia de seguridad) y se borra vcfg_nx.prx.");
        text(60, 380, 2, C_TEXT, "Los ajustes en /data/vcfg_nx se conservan.");
    } else if (what == O_RESET_ALL) {
        text(60, 260, 3, C_WARN, "\xbf" "Quitar los ajustes propios de todos los juegos?");
        text(60, 340, 2, C_TEXT, "Todos pasan a usar los ajustes por defecto. Se aplica al guardar (O).");
    } else {
        text(60, 260, 3, C_WARN, "\xbf" "Salir sin guardar?");
        text(60, 340, 2, C_TEXT, "Se pierden los cambios de esta sesi\xf3n.");
    }
    footer(hints_confirm);
}

// ----------------------------------------------------------------- mando
#define DPAD (ORBIS_PAD_BUTTON_UP | ORBIS_PAD_BUTTON_DOWN | ORBIS_PAD_BUTTON_LEFT | ORBIS_PAD_BUTTON_RIGHT)

// botones recien pulsados; las flechas se repiten si se mantienen
static uint32_t read_pad(void)
{
    static uint32_t prev;
    static uint64_t held_at, last_rep;
    OrbisPadData d;
    if (pad < 0 || scePadReadState(pad, &d) != 0) return 0;
    uint64_t now = sceKernelGetProcessTime();
    uint32_t pressed = d.buttons & ~prev;
    if ((d.buttons & DPAD) && (d.buttons & DPAD) == (prev & DPAD)) {
        if (now - held_at > 380000 && now - last_rep > 70000) {
            pressed |= d.buttons & DPAD;
            last_rep = now;
        }
    } else {
        held_at = now;
    }
    prev = d.buttons;
    return pressed;
}

static void notify_exit(const char* m)
{
    OrbisNotificationRequest req;
    memset(&req, 0, sizeof req);
    snprintf(req.message, sizeof req.message, "%s", m);
    req.type = NotificationRequest;
    req.useIconImageUri = 1;
    req.targetId = -1;
    strcpy(req.iconUri, "cxml://psnotification/tex_icon_system");
    sceKernelSendNotificationRequest(0, &req, sizeof req, 0);
}

static void make_custom(inst_game_t* g)
{
    if (g->custom) return;
    g->cfg = defaults;
    g->custom = 1;
    g->cfg_dirty = 1;
}

static int load_games(void)
{
    static const char* const metas[] = { ROOT "/user/appmeta", ROOT "/user/appmeta/external",
                                         ROOT "/system_data/priv/appmeta", NULL };
    ngames = inst_list_games(ROOT "/user/app", metas, games, 0, INST_MAX_GAMES);
    ngames = inst_list_games(ROOT "/mnt/ext0/user/app", metas, games, ngames, INST_MAX_GAMES);
    ngames = inst_list_games(ROOT "/mnt/ext1/user/app", metas, games, ngames, INST_MAX_GAMES);
    char* ini = read_text(PLUGINS_INI);
    for (int i = 0; i < ngames; ++i) {
        games[i].enabled = inst_ini_has(ini, games[i].id);
        inst_load_game_cfg(&games[i], &defaults, CFG_DIR);
    }
    free(ini);
    return ngames;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    if (video_init() != 0) {
        notify_exit("VCFG-NX: no se pudo abrir la pantalla");
        sceSystemServiceLoadExec("exit", NULL);
        return 1;
    }
    int user = -1;
    sceUserServiceInitialize(NULL);
    sceUserServiceGetInitialUser(&user);
    scePadInit();
    pad = scePadOpen(user, 0, 0, NULL);

    // fuera del sandbox para leer /user/app y escribir /data/GoldHEN (se restaura al salir)
    static struct jailbreak_backup jb;
    int jailed = sys_sdk_jailbreak(&jb) != 0;
    load_defaults();
    load_games();
    rebuild_visible();
    struct stat st;
    if (jailed || (stat(PLUGIN_DIR, &st) != 0 && stat(GOLDHEN_DIR, &st) != 0))
        set_msg(C_WARN, "No se encontr\xf3 /data/GoldHEN: \xbf" "est\xe1 GoldHEN activo?");
    else
        set_msg(C_DIM, "Eleg\xed un juego: X lo activa, CUADRADO abre sus ajustes. O guarda y sale.");

    enum { V_GRID, V_GAME, V_DEFAULTS, V_OPTIONS, V_CONFIRM } view = V_GRID;
    int row = 0, orow = 0, confirm = 0, redraw = 1;
    for (;;) {
        uint32_t b = read_pad();
        if (b) redraw = 1;
        if (view == V_GRID) {
            if (nvis) {
                if (b & ORBIS_PAD_BUTTON_LEFT) sel = sel > 0 ? sel - 1 : 0;
                if (b & ORBIS_PAD_BUTTON_RIGHT) sel = sel + 1 < nvis ? sel + 1 : nvis - 1;
                if (b & ORBIS_PAD_BUTTON_UP) sel = sel >= COLS ? sel - COLS : sel;
                if (b & ORBIS_PAD_BUTTON_DOWN) sel = sel + COLS < nvis ? sel + COLS : nvis - 1;
                if (b & ORBIS_PAD_BUTTON_CROSS) {
                    inst_game_t* g = &games[vis[sel]];
                    g->enabled ^= 1;
                    dirty = 1;
                    set_msg(C_DIM, "%s: VCFG-NX %s (se aplica al guardar con O).", g->title,
                            g->enabled ? "activado" : "desactivado");
                    if (filter == F_ON) rebuild_visible();
                }
                if (b & ORBIS_PAD_BUTTON_SQUARE) { view = V_GAME; row = R_ENABLED; }
            }
            if (b & ORBIS_PAD_BUTTON_TRIANGLE) { view = V_DEFAULTS; row = R_FG; }
            if (b & ORBIS_PAD_BUTTON_L1) { filter = (filter + F_COUNT - 1) % F_COUNT; rebuild_visible(); }
            if (b & ORBIS_PAD_BUTTON_R1) { filter = (filter + 1) % F_COUNT; rebuild_visible(); }
            if (b & ORBIS_PAD_BUTTON_OPTIONS) { view = V_OPTIONS; orow = 0; }
            if (b & ORBIS_PAD_BUTTON_CIRCLE) {
                int r = save_all(0);
                if (r == 0) { notify_exit("VCFG-NX: ajustes guardados. Mantener L3+R3 en el juego: men\xfa"); break; }
                set_msg(C_WARN, "Error al guardar (%d). No se cambi\xf3 nada; reintent\xe1 con O.", r);
            }
            if (redraw) draw_grid();
        } else if (view == V_GAME || view == V_DEFAULTS) {
            int gi = view == V_GAME ? vis[sel] : -1, first = gi < 0 ? R_FG : R_ENABLED;
            if (b & ORBIS_PAD_BUTTON_UP) row = row > first ? row - 1 : R_COUNT - 1;
            if (b & ORBIS_PAD_BUTTON_DOWN) row = row < R_COUNT - 1 ? row + 1 : first;
            int dir = (b & ORBIS_PAD_BUTTON_LEFT) ? -1 : ((b & (ORBIS_PAD_BUTTON_RIGHT | ORBIS_PAD_BUTTON_CROSS)) ? 1 : 0);
            if (dir) {
                dirty = 1;
                if (gi < 0) {
                    change(&defaults, row, dir);
                } else {
                    inst_game_t* g = &games[gi];
                    if (row == R_ENABLED) {
                        g->enabled ^= 1;
                    } else if (row == R_MODE) {
                        if (g->custom) { g->custom = 0; g->cfg_dirty = 1; }
                        else make_custom(g);
                    } else {
                        make_custom(g);   // tocar un ajuste lo vuelve propio de este juego
                        change(&g->cfg, row, dir);
                        g->cfg_dirty = 1;
                    }
                }
            }
            if (gi >= 0 && (b & ORBIS_PAD_BUTTON_SQUARE) && games[gi].custom) {
                games[gi].custom = 0;
                games[gi].cfg_dirty = 1;
                dirty = 1;
            }
            if (b & ORBIS_PAD_BUTTON_CIRCLE) {
                view = V_GRID;
                rebuild_visible();
                set_msg(C_DIM, "%s", dirty ? "Cambios pendientes: O guarda y sale." : "");
                draw_grid();
            } else if (redraw) {
                draw_settings(gi, row);
            }
        } else if (view == V_OPTIONS) {
            if (b & ORBIS_PAD_BUTTON_UP) orow = (orow + O_COUNT - 1) % O_COUNT;
            if (b & ORBIS_PAD_BUTTON_DOWN) orow = (orow + 1) % O_COUNT;
            if (b & ORBIS_PAD_BUTTON_CIRCLE) view = V_GRID;
            if (b & ORBIS_PAD_BUTTON_CROSS) {
                if (orow == O_ALL_ON || orow == O_ALL_OFF) {
                    for (int i = 0; i < ngames; ++i) games[i].enabled = orow == O_ALL_ON;
                    dirty = 1;
                    set_msg(C_DIM, "VCFG-NX %s en todos los juegos (se aplica al guardar).",
                            orow == O_ALL_ON ? "activado" : "desactivado");
                    view = V_GRID;
                    rebuild_visible();
                } else if (orow == O_BACK) {
                    view = V_GRID;
                } else {
                    confirm = orow;
                    view = V_CONFIRM;
                }
            }
            if (view == V_GRID) draw_grid();
            else if (view == V_CONFIRM) draw_confirm(confirm);
            else if (redraw) draw_options(orow);
        } else {
            if (b & ORBIS_PAD_BUTTON_CIRCLE) { view = V_OPTIONS; draw_options(orow); }
            if (b & ORBIS_PAD_BUTTON_CROSS) {
                if (confirm == O_UNINSTALL) {
                    int r = save_all(1);
                    notify_exit(r == 0 ? "VCFG-NX desinstalado" : "VCFG-NX: error al desinstalar");
                    break;
                }
                if (confirm == O_EXIT) break;
                for (int i = 0; i < ngames; ++i)   // O_RESET_ALL
                    if (games[i].custom) { games[i].custom = 0; games[i].cfg_dirty = 1; }
                dirty = 1;
                set_msg(C_DIM, "Ajustes propios quitados (se aplica al guardar).");
                view = V_GRID;
                rebuild_visible();
                draw_grid();
            } else if (redraw && view == V_CONFIRM) {
                draw_confirm(confirm);
            }
        }
        if (redraw || (b & (ORBIS_PAD_BUTTON_CIRCLE | ORBIS_PAD_BUTTON_CROSS))) {
            present();
            redraw = 0;
        } else {
            sceKernelUsleep(16000);   // nada que dibujar: solo mirar el mando
        }
    }
    sceKernelUsleep(500000);
    if (!jailed) sys_sdk_unjailbreak(&jb);
    sceSystemServiceLoadExec("exit", NULL);
    for (;;) sceKernelUsleep(1000000);
    return 0;
}
