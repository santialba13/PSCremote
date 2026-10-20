// VCFG-NX 2.3 - app de configuracion e instalacion.
//
// Cuadricula con los juegos instalados (icono, titulo, estado). Cada juego usa
// los ajustes por defecto o los suyos (/data/vcfg_nx/<ID>.ini, el mismo archivo
// que guarda el menu L3+R3 dentro del juego). Al guardar:
//   * escribe /data/GoldHEN/plugins/vcfg_nx.prx (leido de la app al arrancar)
//   * hace una copia de seguridad de plugins.ini (una por sesion) y lo reescribe
//     solo con los cambios de VCFG-NX, verificando lo escrito
//   * escribe config.ini y los .ini de los juegos que cambiaron
//
// Dibujo: ui_gfx.c sobre un lienzo en memoria; cada cambio redibuja solo su
// zona (recorte) y solo esa zona se copia al framebuffer. Los iconos se
// decodifican de a uno mientras no se toca el mando. Sin entrada no se dibuja
// nada: la app no usa CPU en reposo.
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include <orbis/AudioOut.h>
#include <orbis/Pad.h>
#include <orbis/SystemService.h>
#include <orbis/UserService.h>
#include <orbis/VideoOut.h>
#include <orbis/libkernel.h>

#include <GoldHEN.h>

#include "inst_core.h"
#include "ui_font_data.h"
#include "ui_gfx.h"
#include "ui_sound.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wsign-compare"
#include "../../third_party/stb/stb_image.h"
#pragma clang diagnostic pop

// ROOT: prefijo de todas las rutas (vacio en la consola; una carpeta temporal en
// installer/test/test_app.c, que maneja la app entera en el PC)
#ifndef ROOT
#define ROOT ""
#endif
#define GOLDHEN_DIR ROOT "/data/GoldHEN"
#define PLUGIN_DIR ROOT "/data/GoldHEN/plugins"
#define PLUGIN_DST ROOT "/data/GoldHEN/plugins/vcfg_nx.prx"
#define PLUGINS_INI ROOT "/data/GoldHEN/plugins.ini"
#define CFG_DIR ROOT "/data/vcfg_nx"
#define TITLE_ID "VCNX00001"
#define CONTENT_ID "IV0000-VCNX00001_00-VCFGNXINSTALL000"
#define VERSION "2.4.0"

#define W 1920
#define H 1080

// ------------------------------------------------------------- paleta
// "Aurora": grafito casi negro, luz turquesa y violeta, vidrio esmerilado
#define C_TEXT 0xFFF4F6FAu
#define C_DIM 0xFFA7AFBEu
#define C_FAINT 0xFF6C7383u
#define C_ACCENT 0xFF3FE0CFu    // turquesa
#define C_ACCENT2 0xFFA78BFAu   // violeta
#define C_OK 0xFF4ADE9Au
#define C_WARN 0xFFFBC062u
#define C_DANGER 0xFFFB7185u
#define C_INK 0xFF07080Cu
#define C_GLASS 0x0FFFFFFFu
#define C_STROKE 0x1AFFFFFFu
#define C_EDGE_TOP 0x40FFFFFFu  // borde del vidrio: brilla arriba...
#define C_EDGE_BOT 0x0AFFFFFFu  // ...y casi desaparece abajo

// ----------------------------------------------------------- pantalla
static uint32_t* fb[2];
static int video, cur_fb;
static ui_surf_t canvas, bg;
static ui_surf_t frost;   // el fondo esmerilado (desenfocado y aclarado): lo que se ve a traves del vidrio
static ui_surf_t under;   // lo que queda detras de un dialogo, desenfocado
static int under_ok;
static struct { int x0, y0, x1, y1; } dirty[2];   // zona del lienzo que falta copiar a cada framebuffer

static void mark(int x, int y, int w, int h)
{
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x + w > W) w = W - x;
    if (y + h > H) h = H - y;
    if (w <= 0 || h <= 0) return;
    for (int k = 0; k < 2; ++k) {
        if (dirty[k].x1 <= dirty[k].x0) {
            dirty[k].x0 = x; dirty[k].y0 = y; dirty[k].x1 = x + w; dirty[k].y1 = y + h;
            continue;
        }
        if (x < dirty[k].x0) dirty[k].x0 = x;
        if (y < dirty[k].y0) dirty[k].y0 = y;
        if (x + w > dirty[k].x1) dirty[k].x1 = x + w;
        if (y + h > dirty[k].y1) dirty[k].y1 = y + h;
    }
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
    canvas = (ui_surf_t){ malloc(sz), W, H, W };
    bg = (ui_surf_t){ malloc(sz), W, H, W };
    frost = (ui_surf_t){ malloc(sz), W, H, W };
    under = (ui_surf_t){ malloc(sz), W, H, W };
    if (!canvas.px || !bg.px || !frost.px || !under.px) return -5;
    OrbisVideoOutBufferAttribute attr;
    sceVideoOutSetBufferAttribute(&attr, 0x80000000, 1, 0, W, H, W);   // A8R8G8B8_SRGB lineal
    sceVideoOutSetFlipRate(video, 0);
    return sceVideoOutRegisterBuffers(video, 0, (void**)fb, 2, &attr) == 0 ? 0 : -4;
}

// copia al framebuffer de atras lo que cambio y lo presenta
static void present(void)
{
    int k = cur_fb;
    if (dirty[k].x1 > dirty[k].x0) {
        size_t n = (size_t)(dirty[k].x1 - dirty[k].x0) * 4;
        for (int y = dirty[k].y0; y < dirty[k].y1; ++y)
            memcpy(fb[k] + (size_t)y * W + dirty[k].x0, canvas.px + (size_t)y * W + dirty[k].x0, n);
        dirty[k].x1 = dirty[k].x0 = 0;
    }
    sceVideoOutSubmitFlip(video, k, ORBIS_VIDEO_OUT_FLIP_VSYNC, 0);
    cur_fb ^= 1;
    sceKernelUsleep(16700);   // el flip ocurre en el proximo vsync: recien ahi se puede tocar el otro buffer
}

// fondo: grafito con tres auroras difusas (turquesa, violeta, rosa) y grano
// fino para que no haya bandas. Despues, la version esmerilada para el vidrio.
static void build_background(void)
{
    static const struct { float x, y, rx, ry, k, r, g, b; } A[] = {
        { 330, -140, 1050, 760, 0.42f, 20, 184, 166 },    // turquesa arriba a la izquierda
        { 1700, 300, 900, 820, 0.36f, 124, 92, 246 },     // violeta a la derecha
        { 1040, 1240, 1150, 560, 0.20f, 236, 72, 153 },   // rosa abajo, muy suave
        { 120, 1080, 700, 520, 0.14f, 56, 120, 255 },     // azul en la esquina
    };
    uint32_t seed = 12345;
    for (int y = 0; y < H; ++y) {
        uint32_t* p = bg.px + (size_t)y * W;
        float t = (float)y / H;
        float br = 7 + 4 * t, bgc = 8 + 4 * t, bb = 12 + 6 * t;
        for (int x = 0; x < W; ++x) {
            float r = br, g = bgc, b = bb;
            for (int k = 0; k < 4; ++k) {
                float dx = (x - A[k].x) / A[k].rx, dy = (y - A[k].y) / A[k].ry;
                float q = 1.0f - (dx * dx + dy * dy);
                if (q <= 0) continue;
                q = q * q * A[k].k;
                r += A[k].r * q; g += A[k].g * q; b += A[k].b * q;
            }
            // vineta: los bordes un poco mas oscuros
            float vx = (x - W * 0.5f) / (W * 0.75f), vy = (y - H * 0.45f) / (H * 0.9f);
            float v = 1.0f - 0.35f * (vx * vx + vy * vy);
            seed = seed * 1664525u + 1013904223u;
            float n = (float)((seed >> 24) & 3) - 1.5f;
            r = r * v + n; g = g * v + n; b = b * v + n;
            p[x] = 0xFF000000u | (uint32_t)(r < 0 ? 0 : r > 255 ? 255 : r) << 16 |
                   (uint32_t)(g < 0 ? 0 : g > 255 ? 255 : g) << 8 | (uint32_t)(b < 0 ? 0 : b > 255 ? 255 : b);
        }
    }
    // vidrio: fondo desenfocado, con mas luz y color (como un cristal que junta la luz)
    memcpy(frost.px, bg.px, (size_t)W * H * 4);
    ui_blur(&frost, 0, 0, W, H, 48);
    for (size_t i = 0; i < (size_t)W * H; ++i) {
        uint32_t c = frost.px[i];
        int r = c >> 16 & 0xFF, g = c >> 8 & 0xFF, b = c & 0xFF;
        int l = (r + g + b) / 3;
        r = l + (r - l) * 3 / 2 + 14; g = l + (g - l) * 3 / 2 + 15; b = l + (b - l) * 3 / 2 + 19;
        frost.px[i] = 0xFF000000u | (uint32_t)(r > 255 ? 255 : r < 0 ? 0 : r) << 16 |
                      (uint32_t)(g > 255 ? 255 : g < 0 ? 0 : g) << 8 | (uint32_t)(b > 255 ? 255 : b < 0 ? 0 : b);
    }
}

// ------------------------------------------------------------ archivos
static char* read_text(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) { char* e = malloc(1); e[0] = 0; return e; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0 || n > (64L << 20)) n = 0;   // carpeta o algo raro: como vacio
    char* b = malloc((size_t)n + 1);
    size_t r = n > 0 ? fread(b, 1, (size_t)n, f) : 0;
    b[r] = 0;
    fclose(f);
    return b;
}

static uint8_t* read_bytes(const char* path, size_t* len)
{
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* b = n > 0 && n <= (64L << 20) ? malloc((size_t)n) : NULL;   // carpeta o algo raro: falla
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); b = NULL; }
    fclose(f);
    if (b) *len = (size_t)n;
    return b;
}

// escribe en <path>.vcfg_tmp, relee y compara, y recien ahi reemplaza
static int write_verified(const char* path, const void* data, size_t n)
{
    char tmp[200];
    snprintf(tmp, sizeof tmp, "%s.vcfg_tmp", path);
    FILE* f = fopen(tmp, "wb");
    if (!f) return -1;
    int ok = fwrite(data, 1, n, f) == n;
    if (fclose(f) != 0) ok = 0;
    size_t bl = 0;
    uint8_t* back = ok ? read_bytes(tmp, &bl) : NULL;
    ok = ok && (n == 0 || (back && bl == n && memcmp(back, data, n) == 0));
    free(back);
    if (!ok) { remove(tmp); return -2; }
    remove(path);
    return rename(tmp, path) == 0 ? 0 : -3;
}

static int write_text_verified(const char* path, const char* text) { return write_verified(path, text, strlen(text)); }

// ----------------------------------------------------------- estado
static inst_game_t games[INST_MAX_GAMES];
static int ngames, dirty_cfg, backup_done, jailed;
static nx_config_t defaults;
static char msg[200];
static uint32_t msg_color = C_DIM;

// el plugin se lee de la app ANTES de salir del sandbox: afuera, /app0 ya no
// es la carpeta de la app (era la causa de "Error al guardar" en la 2.2)
static uint8_t* prx;
static size_t prx_len;
static char prx_from[200];

static struct {
    uint32_t sdk;
    int icons_ok, icons_fail, icons_from_pkg;
    char icon_fail_path[200];
    char save_error[200];
} diag;

enum { V_GRID, V_GAME, V_MODAL };
static int view = V_GRID, modal_of = V_GRID;   // modal_of: la vista que queda detras del dialogo

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

static void load_prx(int after_jailbreak)
{
    static const char* const paths[] = {
        ROOT "/app0/plugins/vcfg_nx.prx",
        ROOT "/mnt/sandbox/pfsmnt/" CONTENT_ID "-app0/plugins/vcfg_nx.prx",
        ROOT "/mnt/sandbox/" TITLE_ID "_000/app0/plugins/vcfg_nx.prx",
        NULL,
    };
    if (prx) return;
    for (int i = 0; paths[i]; ++i) {
        if (!after_jailbreak && i > 0) break;
        prx = read_bytes(paths[i], &prx_len);
        if (prx) { snprintf(prx_from, sizeof prx_from, "%s", paths[i]); return; }
    }
}

// codigo de error -> texto para el usuario
static const char* save_error_text(int r)
{
    switch (r) {
    case -10: return "No se pudo hacer la copia de seguridad de plugins.ini.";
    case -21: case -22: case -23: return "No se pudo escribir /data/GoldHEN/plugins.ini.";
    case -30: return "La app no encontr\xf3 el plugin en su propio paquete (reinstal\xe1 el PKG).";
    case -31: return "No se pudo escribir /data/GoldHEN/plugins/vcfg_nx.prx.";
    case -40: return "No se pudo escribir /data/vcfg_nx/config.ini.";
    case -50: return "No se pudieron escribir los ajustes de un juego.";
    default: return "Error desconocido al guardar.";
    }
}

static int save_all(int uninstall)
{
    // sin el plugin no se toca nada: plugins.ini no puede quedar apuntando a un .prx que no esta
    if (!uninstall && !prx) load_prx(1);
    if (!uninstall && !prx) return -30;
    mkdir(ROOT "/data", 0777);
    mkdir(GOLDHEN_DIR, 0777);
    mkdir(PLUGIN_DIR, 0777);
    mkdir(CFG_DIR, 0777);
    char* ini = read_text(PLUGINS_INI);
    if (!backup_done && ini[0]) {
        char bak[200];
        snprintf(bak, sizeof bak, PLUGINS_INI ".vcfg_nx_backup_%ld", (long)time(NULL));
        if (write_text_verified(bak, ini) != 0) { free(ini); return -10; }
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
        remove(PLUGIN_DST);
        return 0;
    }
    size_t have_len = 0;
    uint8_t* have = read_bytes(PLUGIN_DST, &have_len);
    int same = have && have_len == prx_len && memcmp(have, prx, prx_len) == 0;
    free(have);
    if (!same && write_verified(PLUGIN_DST, prx, prx_len) != 0) return -31;
    char cfg[768];
    nx_config_format(&defaults, "VCFG-NX: ajustes por defecto (los juegos con ajustes propios usan su <ID>.ini)",
                     cfg, sizeof cfg);
    if (write_text_verified(CFG_DIR "/config.ini", cfg) != 0) return -40;
    for (int i = 0; i < ngames; ++i) {
        inst_game_t* g = &games[i];
        if (!g->cfg_dirty) continue;
        char path[160];
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
// icon0.png (512x512) reducido por promedio de area y con esquinas redondeadas.
// Los de la cuadricula se guardan (ICON_CACHE), el grande de la pantalla de
// ajustes es uno solo. Se decodifican de a uno cuando no se toca el mando.
#define ICON 248
#define BIG_ICON 360
#define ICON_CACHE 36
typedef struct { int game; ui_img_t img; int state; unsigned used; } icon_t;   // state: 1 ok, -1 sin icono
static icon_t icons[ICON_CACHE], big_icon = { -1 };
static unsigned icon_clock;

static int icon_path(int gi, int k, char* p, size_t n)
{
    const char* id = games[gi].id;
    switch (k) {
    case 0: if (!games[gi].meta[0]) return 0; snprintf(p, n, "%s/icon0.png", games[gi].meta); return 1;
    case 1: snprintf(p, n, ROOT "/user/appmeta/%s/icon0.png", id); return 1;
    case 2: snprintf(p, n, ROOT "/user/appmeta/external/%s/icon0.png", id); return 1;
    case 3: snprintf(p, n, ROOT "/system_data/priv/appmeta/%s/icon0.png", id); return 1;
    case 4: snprintf(p, n, ROOT "/user/app/%s/sce_sys/icon0.png", id); return 1;
    default: return -1;
    }
}

static ui_img_t decode_icon(int gi, int size, int radius)
{
    ui_img_t out = { 0 };
    char p[200];
    unsigned char* src = NULL;
    int w = 0, h = 0, n;
    for (int k = 0; !src; ++k) {
        int r = icon_path(gi, k, p, sizeof p);
        if (r < 0) break;
        if (r) src = stbi_load(p, &w, &h, &n, 4);
    }
    if (!src && games[gi].app[0]) {   // respaldo: el icon0.png que viene dentro del app.pkg instalado
        size_t len = 0;
        snprintf(p, sizeof p, "%s/app.pkg", games[gi].app);
        unsigned char* png = inst_pkg_entry(p, PKG_ENTRY_ICON0, &len);
        if (png) src = stbi_load_from_memory(png, (int)len, &w, &h, &n, 4);
        free(png);
        if (src) diag.icons_from_pkg++;
    }
    if (!src) {
        diag.icons_fail++;
        icon_path(gi, games[gi].meta[0] ? 0 : 1, diag.icon_fail_path, sizeof diag.icon_fail_path);
        return out;
    }
    out = ui_img_new(size, size);
    if (!out.px) { stbi_image_free(src); return out; }
    for (int y = 0; y < size; ++y) {
        int y0 = y * h / size, y1 = (y + 1) * h / size;
        if (y1 <= y0) y1 = y0 + 1;
        for (int x = 0; x < size; ++x) {
            int x0 = x * w / size, x1 = (x + 1) * w / size;
            if (x1 <= x0) x1 = x0 + 1;
            unsigned r = 0, g = 0, b = 0, a = 0, cnt = 0;
            for (int j = y0; j < y1; ++j)
                for (int i = x0; i < x1; ++i) {
                    const unsigned char* q = src + ((size_t)j * w + i) * 4;
                    r += q[0]; g += q[1]; b += q[2]; a += q[3]; ++cnt;
                }
            out.px[y * size + x] = (a / cnt) << 24 | (r / cnt) << 16 | (g / cnt) << 8 | (b / cnt);
        }
    }
    stbi_image_free(src);
    ui_img_round(&out, radius);
    ui_img_spans(&out);
    diag.icons_ok++;
    return out;
}

// icono de la cuadricula si ya esta decodificado (NULL si falta o no hay)
static icon_t* icon_lookup(int gi)
{
    for (int k = 0; k < ICON_CACHE; ++k)
        if (icons[k].used && icons[k].game == gi) {
            icons[k].used = ++icon_clock;
            return &icons[k];
        }
    return NULL;
}

static void icon_load(int gi)
{
    icon_t* slot = &icons[0];
    for (int k = 1; k < ICON_CACHE; ++k)
        if (icons[k].used < slot->used) slot = &icons[k];
    ui_img_free(&slot->img);
    slot->game = gi;
    slot->img = decode_icon(gi, ICON, 30);
    slot->state = slot->img.px ? 1 : -1;
    slot->used = ++icon_clock;
}

// ---------------------------------------------------------------- ajustes
static const char* SR_N[] = { "OFF", "Suave", "Normal", "Fuerte" };
static const char* RC_N[] = { "OFF", "Baja", "Media", "Alta" };
static const char* MD_N[] = { "Auto", "R\xe1pido", "Calidad" };
static const int TEMPS[] = { 0, 70, 75, 80, 85, 90 };
#define NTEMPS 6

enum { R_ENABLED, R_MODE, R_FG, R_SR, R_MODEL, R_AA, R_RCAS, R_DYN, R_ECO, R_TEMP, R_FPS, R_COUNT };
static const char* ROW_LABEL[R_COUNT] = {
    "VCFG-NX en este juego", "Ajustes", "Generaci\xf3n de fotogramas (30 \xbb 60)", "Reescalado IA (NXSR)",
    "Modelo IA", "Antialiasing SMAA", "Nitidez FSR1 (RCAS)", "Calidad din\xe1mica",
    "Modo ahorro (menos calor)", "L\xedmite de temperatura", "Contador de FPS",
};
static const char* ROW_HELP[R_COUNT] = {
    "Agrega o quita VCFG-NX de plugins.ini para este juego.",
    "Por defecto: sigue los ajustes generales. Propios: este juego guarda los suyos.",
    "Genera un fotograma intermedio: 30 fps se ven como 60. Se apaga solo si el juego ya va a m\xe1s de 48.",
    "Recupera detalle sin agregar grano ni halos: la salida queda acotada a los p\xedxeles vecinos.",
    "Auto: Calidad en PS4 Pro y R\xe1pido en PS4 normal. Calidad usa unas 3 veces m\xe1s GPU.",
    "Suaviza los bordes dentados (SMAA 1x).",
    "Nitidez cl\xe1sica de AMD; solo se usa con NXSR apagado.",
    "Si la GPU no da abasto baja un escal\xf3n en vez de trabar el juego, y despu\xe9s vuelve a subir.",
    "Modelo R\xe1pido, sin SMAA y l\xedmite de temperatura de 75\xb0" "C.",
    "Por encima baja la calidad; 6\xb0" "C m\xe1s pausa los efectos hasta que enfr\xed" "e.",
    "Muestra los FPS y la temperatura en una esquina.",
};

static int is_switch(int row) { return row == R_ENABLED || row == R_FG || row == R_AA || row == R_DYN || row == R_ECO || row == R_FPS; }

static int temp_index(int t)
{
    for (int i = 0; i < NTEMPS; ++i)
        if (TEMPS[i] == t) return i;
    return 3;
}

static int row_on(const nx_config_t* c, int row)
{
    switch (row) {
    case R_FG: return c->framegen;
    case R_AA: return c->aa;
    case R_DYN: return c->dynamic;
    case R_ECO: return c->eco;
    case R_FPS: return c->fps_counter;
    default: return 0;
    }
}

static void value_text(const nx_config_t* c, int row, char* out, size_t n)
{
    switch (row) {
    case R_SR: snprintf(out, n, "%s", SR_N[c->sr & 3]); break;
    case R_MODEL: snprintf(out, n, "%s", MD_N[c->sr_model % 3]); break;
    case R_RCAS: snprintf(out, n, "%s", RC_N[c->rcas & 3]); break;
    case R_TEMP: if (c->temp_limit) snprintf(out, n, "%d\xb0" "C", c->temp_limit); else snprintf(out, n, "OFF"); break;
    default: snprintf(out, n, "%s", row_on(c, row) ? "ON" : "OFF");
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

// --------------------------------------------------------- piezas visuales
// cuadricula de portadas: 6 x 2, cada una con el titulo y el estado debajo
#define COLS 6
#define ROWS 2
#define CARD_W ICON
#define CARD_H (ICON + 86)
#define GAP_X 62
#define GAP_Y 30
#define GRID_X ((W - COLS * CARD_W - (COLS - 1) * GAP_X) / 2)
#define GRID_Y 168
#define LIFT 10
#define RING 7     // separacion del anillo de la portada elegida

static ui_img_t spr_shadow, spr_glow, spr_ring, spr_sheen, spr_big_glow, spr_mark;

// recolorea un sprite con un degradado diagonal turquesa -> violeta (conserva el alfa)
static void aurora_tint(ui_img_t* im, uint32_t c0, uint32_t c1)
{
    for (int y = 0; y < im->h; ++y)
        for (int x = 0; x < im->w; ++x) {
            uint32_t* p = &im->px[y * im->w + x];
            if (!(*p >> 24)) continue;
            int t = (x + y) * 255 / (im->w + im->h);
            uint32_t c = 0;
            for (int sh = 0; sh < 24; sh += 8) {
                int a = (int)(c0 >> sh & 0xFF), b = (int)(c1 >> sh & 0xFF);
                c |= (uint32_t)(a + (b - a) * t / 255) << sh;
            }
            *p = (*p & 0xFF000000u) | c;
        }
}

static void build_sprites(void)
{
    // sombra suave de cada portada y brillo de color de la elegida
    spr_shadow = ui_make_shadow(ICON, ICON, 30, 24, 0x90000000u);
    ui_img_hole(&spr_shadow, 24, 10, ICON, ICON, 30);   // lo que la portada tapa siempre
    ui_img_spans(&spr_shadow);
    spr_glow = ui_make_shadow(ICON + 2 * RING, ICON + 2 * RING, 36, 40, 0xC8FFFFFFu);
    aurora_tint(&spr_glow, C_ACCENT, C_ACCENT2);
    ui_img_hole(&spr_glow, 40 + RING, 40 + RING, ICON, ICON, 30);
    ui_img_spans(&spr_glow);
    // anillo con degradado alrededor de la elegida
    spr_ring = ui_img_new(ICON + 2 * RING, ICON + 2 * RING);
    if (spr_ring.px) {
        ui_surf_t s = { spr_ring.px, spr_ring.w, spr_ring.h, spr_ring.w };
        ui_target(&s);
        ui_rrect_stroke(0, 0, spr_ring.w, spr_ring.h, 30 + RING, 3, 0xFFFFFFFFu);
        aurora_tint(&spr_ring, 0xFF7CF5E6u, 0xFFC4B5FDu);
        ui_img_spans(&spr_ring);
        ui_target(&canvas);
    }
    // reflejo diagonal sobre la portada elegida
    spr_sheen = ui_img_new(ICON, ICON);
    for (int y = 0; spr_sheen.px && y < ICON; ++y)
        for (int x = 0; x < ICON; ++x) {
            float t = 1.0f - (x * 0.6f + y) / (ICON * 0.8f);
            if (t > 0) spr_sheen.px[y * ICON + x] = (uint32_t)(t * t * 64) << 24 | 0xFFFFFF;
        }
    if (spr_sheen.px) { ui_img_round(&spr_sheen, 30); ui_img_spans(&spr_sheen); }
    spr_big_glow = ui_make_shadow(BIG_ICON, BIG_ICON, 40, 56, 0x90FFFFFFu);
    aurora_tint(&spr_big_glow, C_ACCENT, C_ACCENT2);
    ui_img_hole(&spr_big_glow, 56, 46, BIG_ICON, BIG_ICON, 40);
    ui_img_spans(&spr_big_glow);
    // marca del logo: gota de luz con degradado
    spr_mark = ui_make_shadow(18, 18, 9, 12, 0xFFFFFFFFu);
    aurora_tint(&spr_mark, C_ACCENT, C_ACCENT2);
    ui_img_spans(&spr_mark);
}

// panel de vidrio: el fondo esmerilado, un velo claro y un borde que brilla arriba
static void glass(int x, int y, int w, int h, int r, uint32_t veil)
{
    ui_rrect_src(x, y, w, h, r, under_ok && view == V_MODAL ? &under : &frost);
    if (veil >> 24) ui_rrect(x, y, w, h, r, veil);
    ui_rrect_stroke_grad(x, y, w, h, r, 1, C_EDGE_TOP, C_EDGE_BOT);
}

// pastilla con texto; devuelve el x siguiente
static int pill(int x, int y, int font, uint32_t bgc, uint32_t fg, const char* label)
{
    int w = ui_text_w(font, label) + 24, h = ui_line_h(font) + 10;
    ui_rrect(x, y, w, h, h / 2, bgc);
    ui_rrect_stroke_grad(x, y, w, h, h / 2, 1, 0x30FFFFFFu, 0x08FFFFFFu);
    ui_text(font, x + 12, y + 5, fg, label);
    return x + w;
}

// estado: punto de color con halo y texto (sin caja: minimalista)
static int status_dot(int x, int y, int font, uint32_t dot, uint32_t fg, const char* label)
{
    int h = ui_line_h(font), cy = y + h / 2 + 1;
    ui_circle(x + 5, cy, 9, (dot & 0xFFFFFFu) | 0x30000000u);
    ui_circle(x + 5, cy, 4, dot);
    return ui_text(font, x + 18, y, fg, label);
}

// pastilla de vidrio con punto de color
static int status_pill(int x, int y, uint32_t dot, const char* label)
{
    int w = ui_text_w(FONT_SMALL, label) + 46, h = ui_line_h(FONT_SMALL) + 14;
    ui_rrect(x, y, w, h, h / 2, 0x12FFFFFFu);
    ui_rrect_stroke_grad(x, y, w, h, h / 2, 1, 0x2EFFFFFFu, 0x08FFFFFFu);
    ui_circle(x + 20, y + h / 2, 8, (dot & 0xFFFFFFu) | 0x38000000u);
    ui_circle(x + 20, y + h / 2, 4, dot);
    ui_text(FONT_SMALL, x + 34, y + 7, C_TEXT, label);
    return x + w;
}

typedef struct { int btn; const char* chip; const char* label; } hint_t;   // btn < 0: chip de texto

static void draw_hints(const hint_t* h, int n)
{
    int x = 60, y = 1018;
    for (int i = 0; i < n; ++i) {
        if (h[i].btn >= 0) {
            ui_button(h[i].btn, x, y);
            x += ui_button_size() + 10;
        } else {
            x = pill(x, y + 6, FONT_TINY, 0x14FFFFFFu, C_DIM, h[i].chip) + 10;
        }
        x = ui_text(FONT_SMALL, x, y + 9, C_DIM, h[i].label) + 36;
    }
    char fit[200];
    ui_text_fit(FONT_SMALL, msg, W - 60 - x, fit, sizeof fit);
    ui_text(FONT_SMALL, W - 60 - ui_text_w(FONT_SMALL, fit), y + 9, msg_color, fit);
}

static void draw_tabs(void);

static void draw_topbar(void)
{
    ui_blit(&spr_mark, 60, 46, 255);
    int x = ui_text(FONT_LOGO, 100, 36, C_TEXT, "vcfg");
    x = ui_text(FONT_LOGO, x + 2, 36, C_ACCENT, "nx");
    ui_text(FONT_TINY, x + 14, 62, C_FAINT, "v" VERSION);
    int on = 0;
    for (int i = 0; i < ngames; ++i) on += games[i].enabled;
    char t[64];
    snprintf(t, sizeof t, "%d de %d juegos activos", on, ngames);
    const char* gh = jailed ? "GoldHEN sin acceso" : "GoldHEN listo";
    int w1 = ui_text_w(FONT_SMALL, t) + 18, w2 = ui_text_w(FONT_SMALL, gh) + 18;
    status_dot(W - 60 - w1, 54, FONT_SMALL, C_ACCENT2, C_DIM, t);
    status_dot(W - 60 - w1 - 40 - w2, 54, FONT_SMALL, jailed ? C_WARN : C_OK, C_DIM, gh);
}

// ------------------------------------------------------------- cuadricula
static void card_rect(int k, int* x, int* y)
{
    int r = k / COLS - top_row, c = k % COLS;
    *x = GRID_X + c * (CARD_W + GAP_X);
    *y = GRID_Y + r * (CARD_H + GAP_Y) - (k == sel ? LIFT : 0);
}

static void draw_card(int k)
{
    int x, y;
    card_rect(k, &x, &y);
    const inst_game_t* g = &games[vis[k]];
    int is_sel = k == sel;
    if (is_sel) ui_blit(&spr_glow, x - RING - 40, y - RING - 40, 255);
    ui_blit(&spr_shadow, x - 24, y - 24 + 14, 255);
    icon_t* ic = icon_lookup(vis[k]);
    if (ic && ic->state == 1) {
        ui_blit(&ic->img, x, y, 255);
    } else {   // sin icono (o todavia cargando): vidrio con las iniciales
        ui_rrect_grad(x, y, ICON, ICON, 30, 0xFF1C1F2Au, 0xFF121319u);
        ui_rrect_stroke_grad(x, y, ICON, ICON, 30, 1, 0x30FFFFFFu, 0x06FFFFFFu);
        char ini[3] = { g->title[0] ? g->title[0] : '?', g->title[0] && g->title[1] ? g->title[1] : 0, 0 };
        ui_text(FONT_LOGO, x + (ICON - ui_text_w(FONT_LOGO, ini)) / 2, y + ICON / 2 - 30,
                ic ? C_FAINT : 0x40FFFFFFu, ini);
    }
    // las que no estan elegidas quedan un poco atras; las apagadas, bastante
    if (!g->enabled) ui_rrect(x, y, ICON, ICON, 30, is_sel ? 0x70070810u : 0xA8070810u);
    else if (!is_sel) ui_rrect(x, y, ICON, ICON, 30, 0x30070810u);
    if (is_sel) {
        ui_blit(&spr_sheen, x, y, 255);
        ui_blit(&spr_ring, x - RING, y - RING, 255);
    }
    if (g->custom) {   // ajustes propios: chip de vidrio en la esquina
        int w = ui_text_w(FONT_TINY, "PROPIO") + 22;
        ui_rrect(x + ICON - 12 - w, y + 12, w, 26, 13, 0xB80D0F16u);
        ui_rrect_stroke_grad(x + ICON - 12 - w, y + 12, w, 26, 13, 1, 0x50FBC062u, 0x18FBC062u);
        ui_text(FONT_TINY, x + ICON - 1 - w, y + 17, C_WARN, "PROPIO");
    }
    // titulo en una linea y estado debajo (el titulo entero esta en la barra de detalle)
    char t[96];
    ui_text_fit(FONT_CARD, g->title, CARD_W - 4, t, sizeof t);
    ui_text(FONT_CARD, x + 2, y + ICON + 16, is_sel ? C_TEXT : 0xFFC9CED8u, t);
    status_dot(x + 2, y + ICON + 20 + ui_line_h(FONT_CARD), FONT_TINY, g->enabled ? C_OK : C_FAINT,
               is_sel ? C_DIM : C_FAINT, g->enabled ? "ACTIVO" : "APAGADO");
}

static void card_region(int k, int* x, int* y, int* w, int* h)
{
    int cx, cy;
    card_rect(k, &cx, &cy);
    *x = cx - RING - 42; *y = cy - RING - 42 - LIFT; *w = CARD_W + 2 * (RING + 42);
    *h = CARD_H + RING + 42 + 2 * LIFT + 8;
}

// filtros: control segmentado de vidrio, centrado arriba
static void draw_tabs(void)
{
    static const char* names[F_COUNT] = { "Todos", "Activos", "Propios" };
    int counts[F_COUNT] = { ngames, 0, 0 };
    for (int i = 0; i < ngames; ++i) { counts[F_ON] += games[i].enabled; counts[F_CUSTOM] += games[i].custom; }
    char l[F_COUNT][48], n[F_COUNT][8];
    int tw = 0, seg[F_COUNT];
    for (int f = 0; f < F_COUNT; ++f) {
        snprintf(l[f], sizeof l[f], "%s", names[f]);
        snprintf(n[f], sizeof n[f], "%d", counts[f]);
        seg[f] = ui_text_w(FONT_BODY, l[f]) + ui_text_w(FONT_TINY, n[f]) + 64;
        tw += seg[f];
    }
    int h = 54, w = tw + 12 + 2 * 50, x = (W - w) / 2, y = 36;
    glass(x, y, w, h, h / 2, 0x0AFFFFFFu);
    ui_text(FONT_TINY, x + 20, y + 19, C_FAINT, "L1");
    ui_text(FONT_TINY, x + w - 20 - ui_text_w(FONT_TINY, "R1"), y + 19, C_FAINT, "R1");
    int sx = x + 56;
    for (int f = 0; f < F_COUNT; ++f) {
        int on = f == filter;
        if (on) {
            ui_rrect_grad(sx, y + 6, seg[f], h - 12, (h - 12) / 2, 0x3D3FE0CFu, 0x3DA78BFAu);
            ui_rrect_stroke_grad(sx, y + 6, seg[f], h - 12, (h - 12) / 2, 1, 0x80FFFFFFu, 0x18FFFFFFu);
        }
        int tx = ui_text(FONT_BODY, sx + 26, y + 13, on ? C_TEXT : C_DIM, l[f]);
        ui_text(FONT_TINY, tx + 10, y + 20, on ? C_ACCENT : C_FAINT, n[f]);
        sx += seg[f];
    }
}

static void summary(const nx_config_t* c, char* out, size_t n)
{
    char t[16];
    value_text(c, R_TEMP, t, sizeof t);
    snprintf(out, n, "Fotogramas %s   \xb7   NXSR %s (%s)   \xb7   SMAA %s   \xb7   Din\xe1mica %s   \xb7   Ahorro %s   \xb7   L\xedmite %s",
             c->framegen ? "ON" : "OFF", SR_N[c->sr & 3], MD_N[c->sr_model % 3], c->aa ? "ON" : "OFF",
             c->dynamic ? "ON" : "OFF", c->eco ? "ON" : "OFF", t);
}

// barra de detalle: vidrio flotante con el juego elegido
#define DETAIL_Y 900
#define DETAIL_H 92
static void draw_details(void)
{
    glass(60, DETAIL_Y, W - 120, DETAIL_H, 28, 0x08FFFFFFu);
    if (!nvis) {
        ui_text(FONT_H2, 96, DETAIL_Y + 30, C_WARN, ngames ? "No hay juegos en este filtro (L1 / R1 para cambiarlo)."
                                                            : "No se encontraron juegos instalados.");
        return;
    }
    const inst_game_t* g = &games[vis[sel]];
    char t[96];
    ui_text_fit(FONT_H2, g->title, 860, t, sizeof t);
    int x = ui_text(FONT_H2, 96, DETAIL_Y + 14, C_TEXT, t) + 28;
    x = status_dot(x, DETAIL_Y + 20, FONT_SMALL, g->enabled ? C_OK : C_FAINT, C_DIM,
                   g->enabled ? "VCFG-NX activo" : "VCFG-NX apagado") + 28;
    x = status_dot(x, DETAIL_Y + 20, FONT_SMALL, g->custom ? C_WARN : C_ACCENT2, C_DIM,
                   g->custom ? "Ajustes propios" : "Ajustes por defecto") + 28;
    ui_text(FONT_SMALL, x, DETAIL_Y + 20, C_FAINT, g->id);
    char s[220];
    summary(effective(g), s, sizeof s);
    ui_text(FONT_SMALL, 96, DETAIL_Y + 56, g->enabled ? C_DIM : C_FAINT, s);
}

static void draw_scrollbar(void)
{
    int total_rows = (nvis + COLS - 1) / COLS;
    if (total_rows <= ROWS) return;
    int bh = ROWS * (CARD_H + GAP_Y) - GAP_Y, th = bh * ROWS / total_rows;
    int x = W - 30;
    ui_rrect(x, GRID_Y, 4, bh, 2, 0x14FFFFFFu);
    ui_rrect_grad(x, GRID_Y + (bh - th) * top_row / (total_rows - ROWS), 4, th, 2, C_ACCENT, C_ACCENT2);
}

static const hint_t HINTS_GRID[] = {
    { UI_BTN_CROSS, NULL, "Activar" }, { UI_BTN_SQUARE, NULL, "Ajustes del juego" },
    { UI_BTN_TRIANGLE, NULL, "Por defecto" }, { -1, "OPTIONS", "M\xe1s" }, { UI_BTN_CIRCLE, NULL, "Salir" },
};

static void draw_grid_view(void)
{
    draw_topbar();
    draw_tabs();
    if (sel / COLS < top_row) top_row = sel / COLS;
    if (sel / COLS >= top_row + ROWS) top_row = sel / COLS - ROWS + 1;
    for (int r = 0; r < ROWS; ++r)
        for (int c = 0; c < COLS; ++c) {
            int k = (top_row + r) * COLS + c;
            if (k < nvis && k != sel) draw_card(k);
        }
    if (nvis) draw_card(sel);   // la elegida al final: su brillo va por encima de las vecinas
    draw_scrollbar();
    draw_details();
    if (view == V_GRID) draw_hints(HINTS_GRID, 5);
}

// --------------------------------------------------------------- ajustes
#define PANEL_X 600
#define PANEL_Y 128
#define PANEL_W (W - 60 - PANEL_X)
#define PANEL_H 864
#define ROW_Y0 (PANEL_Y + 118)
#define ROW_H 60

static int settings_game = -1;   // -1: ajustes por defecto
static int settings_row;

static void draw_switch(int x, int y, int on, int dim)
{
    if (on) {
        ui_rrect_grad(x, y, 74, 38, 19, dim ? 0xFF2A6B62u : 0xFF3FE0CFu, dim ? 0xFF2A5E58u : 0xFF2BB8A9u);
        if (!dim) ui_rrect_stroke_grad(x, y, 74, 38, 19, 1, 0x70FFFFFFu, 0x10FFFFFFu);
    } else {
        ui_rrect(x, y, 74, 38, 19, 0x1CFFFFFFu);
        ui_rrect_stroke_grad(x, y, 74, 38, 19, 1, 0x30FFFFFFu, 0x08FFFFFFu);
    }
    int kx = on ? x + 55 : x + 19;
    ui_circle(kx, y + 21, 15, 0x40000000u);   // sombra del boton
    ui_circle(kx, y + 19, 15, dim ? 0xFFB9BFCAu : 0xFFFFFFFFu);
}

static void draw_selector(int x, int y, const char* v, int selected, int dim)
{
    int w = 236;
    ui_rrect(x, y, w, 42, 21, selected ? 0x1EFFFFFFu : 0x0EFFFFFFu);
    ui_rrect_stroke_grad(x, y, w, 42, 21, 1, selected ? 0x50FFFFFFu : 0x22FFFFFFu, 0x06FFFFFFu);
    uint32_t ch = selected ? C_ACCENT : C_FAINT;
    ui_text(FONT_BODY, x + 16, y + 7, ch, "\xab");
    ui_text(FONT_BODY, x + w - 30, y + 7, ch, "\xbb");
    ui_text(FONT_BODY, x + (w - ui_text_w(FONT_BODY, v)) / 2, y + 7, dim ? C_DIM : C_TEXT, v);
}

static void row_region(int row, int* x, int* y, int* w, int* h)
{
    int first = settings_game < 0 ? R_FG : R_ENABLED;
    *x = PANEL_X + 8; *y = ROW_Y0 + (row - first) * ROW_H - 4; *w = PANEL_W - 16; *h = ROW_H;
}

static void draw_settings_row(int row)
{
    int gi = settings_game, first = gi < 0 ? R_FG : R_ENABLED;
    if (row < first) return;
    const nx_config_t* c = gi >= 0 ? effective(&games[gi]) : &defaults;
    int inherited = gi >= 0 && !games[gi].custom && row > R_MODE;
    int y = ROW_Y0 + (row - first) * ROW_H, s = row == settings_row;
    if (s) {   // fila elegida: vidrio mas claro con una linea de luz a la izquierda
        ui_rrect(PANEL_X + 16, y - 2, PANEL_W - 32, ROW_H - 8, 18, 0x14FFFFFFu);
        ui_rrect_stroke_grad(PANEL_X + 16, y - 2, PANEL_W - 32, ROW_H - 8, 18, 1, 0x3CFFFFFFu, 0x0AFFFFFFu);
        ui_rrect_grad(PANEL_X + 26, y + 12, 4, ROW_H - 32, 2, C_ACCENT, C_ACCENT2);
    }
    ui_text(FONT_BODY, PANEL_X + 46, y + 10, inherited && !s ? C_DIM : C_TEXT, ROW_LABEL[row]);
    int cx = PANEL_X + PANEL_W - 40;
    if (inherited) ui_text(FONT_TINY, cx - 330, y + 18, C_FAINT, "POR DEFECTO");
    if (row == R_ENABLED) {
        draw_switch(cx - 74, y + 6, games[gi].enabled, 0);
    } else if (row == R_MODE) {
        draw_selector(cx - 236, y + 4, games[gi].custom ? "Propios" : "Por defecto", s, 0);
    } else if (is_switch(row)) {
        draw_switch(cx - 74, y + 6, row_on(c, row), inherited);
    } else {
        char v[32];
        value_text(c, row, v, sizeof v);
        draw_selector(cx - 236, y + 4, v, s, inherited);
    }
    if (row == R_MODE) ui_fill(PANEL_X + 40, y + ROW_H - 6, PANEL_W - 80, 1, 0x14FFFFFFu);
}

static void draw_settings_help(void)
{
    ui_text(FONT_SMALL, PANEL_X + 46, PANEL_Y + PANEL_H - 54, C_DIM, ROW_HELP[settings_row]);
}

static const hint_t HINTS_SETTINGS[] = {
    { -1, "ARRIBA / ABAJO", "Elegir" }, { -1, "IZQ. / DER.", "Cambiar" }, { UI_BTN_CIRCLE, NULL, "Listo" },
};
static const hint_t HINTS_SETTINGS_CUSTOM[] = {
    { -1, "ARRIBA / ABAJO", "Elegir" }, { -1, "IZQ. / DER.", "Cambiar" }, { UI_BTN_SQUARE, NULL, "Volver a por defecto" },
    { UI_BTN_CIRCLE, NULL, "Listo" },
};

static void draw_settings_view(void)
{
    int gi = settings_game;
    draw_topbar();
    // columna izquierda
    if (gi >= 0) {
        const inst_game_t* g = &games[gi];
        int ix = 60 + (PANEL_X - 60 - 40 - BIG_ICON) / 2, iy = 150;
        ui_blit(&spr_big_glow, ix - 56, iy - 56 + 10, 255);
        if (big_icon.game == gi && big_icon.state == 1) {
            ui_blit(&big_icon.img, ix, iy, 255);
        } else {
            ui_rrect_grad(ix, iy, BIG_ICON, BIG_ICON, 40, 0xFF1C1F2Au, 0xFF121319u);
            char ini[3] = { g->title[0], g->title[0] ? g->title[1] : 0, 0 };
            ui_text(FONT_LOGO, ix + (BIG_ICON - ui_text_w(FONT_LOGO, ini)) / 2, iy + BIG_ICON / 2 - 30, C_FAINT, ini);
        }
        ui_rrect_stroke_grad(ix, iy, BIG_ICON, BIG_ICON, 40, 1, 0x50FFFFFFu, 0x08FFFFFFu);
        char l1[96], l2[96];
        ui_text_wrap2(FONT_H1, g->title, PANEL_X - 100, l1, l2, sizeof l1);
        int y = iy + BIG_ICON + 34;
        ui_text(FONT_H1, 60, y, C_TEXT, l1);
        if (l2[0]) { y += ui_line_h(FONT_H1); ui_text(FONT_H1, 60, y, C_TEXT, l2); }
        y += ui_line_h(FONT_H1) + 8;
        ui_text(FONT_SMALL, 60, y, C_FAINT, g->id);
        y += 44;
        int x = status_dot(62, y, FONT_SMALL, g->enabled ? C_OK : C_FAINT, C_DIM, g->enabled ? "Activo" : "Apagado") + 28;
        status_dot(x, y, FONT_SMALL, g->custom ? C_WARN : C_ACCENT2, C_DIM, g->custom ? "Ajustes propios" : "Ajustes por defecto");
    } else {
        ui_text(FONT_H1, 60, 170, C_TEXT, "Ajustes por");
        ui_text(FONT_H1, 60, 170 + ui_line_h(FONT_H1), C_TEXT, "defecto");
        ui_text(FONT_BODY, 60, 296, C_DIM, "Los usan todos los juegos");
        ui_text(FONT_BODY, 60, 330, C_DIM, "que no tienen ajustes propios.");
        ui_text(FONT_SMALL, 60, 404, C_FAINT, "Cada juego puede tener los suyos:");
        ui_text(FONT_SMALL, 60, 434, C_FAINT, "en la cuadr\xed" "cula, CUADRADO sobre el juego,");
        ui_text(FONT_SMALL, 60, 464, C_FAINT, "o L3+R3 dentro del juego.");
    }
    // panel derecho de vidrio
    glass(PANEL_X, PANEL_Y, PANEL_W, PANEL_H, 32, 0x08FFFFFFu);
    ui_text(FONT_H2, PANEL_X + 46, PANEL_Y + 32, C_TEXT, gi >= 0 ? "Ajustes de este juego" : "Ajustes generales");
    ui_text(FONT_SMALL, PANEL_X + 46, PANEL_Y + 72, C_FAINT,
            gi >= 0 ? (games[gi].custom ? "Este juego guarda sus propios ajustes."
                                        : "Sigue los ajustes por defecto: cambi\xe1 cualquier valor para tener propios.")
                    : "Se aplican al guardar y salir.");
    for (int r = gi >= 0 ? R_ENABLED : R_FG; r < R_COUNT; ++r) draw_settings_row(r);
    draw_settings_help();
    if (view != V_GAME) return;
    if (gi >= 0 && games[gi].custom) draw_hints(HINTS_SETTINGS_CUSTOM, 4);
    else draw_hints(HINTS_SETTINGS, 3);
}

// ---------------------------------------------------------------- dialogos
#define MODAL_W 860
#define MODAL_H 580
#define MODAL_X ((W - MODAL_W) / 2)
#define MODAL_Y ((H - MODAL_H) / 2)

typedef struct {
    const char* title;
    uint32_t title_color;
    const char* body[5];
    const char* opts[6];
    int nopts, row;
} modal_t;
static modal_t modal;

static void draw_view(void);

// lo de atras del dialogo se dibuja una vez, se desenfoca y se oscurece: el
// dialogo es un vidrio de verdad sobre la pantalla anterior
static void snapshot_under(void)
{
    if (!under.px) return;
    ui_target(&under);
    ui_copy(&bg, 0, 0, W, H);
    under_ok = 0;
    int v = view;
    view = V_MODAL;   // sin las ayudas de abajo
    if (modal_of == V_GRID) draw_grid_view();
    else draw_settings_view();
    view = v;
    ui_blur(&under, 0, 0, W, H, 30);
    for (size_t i = 0; i < (size_t)W * H; ++i) {
        uint32_t c = under.px[i];
        unsigned r = (c >> 16 & 0xFF) * 120 / 255 + 3, g = (c >> 8 & 0xFF) * 120 / 255 + 4, b = (c & 0xFF) * 120 / 255 + 7;
        under.px[i] = 0xFF000000u | r << 16 | g << 8 | b;
    }
    under_ok = 1;
    ui_target(&canvas);
}

static void draw_modal(void)
{
    if (!under_ok) ui_fill(0, 0, W, H, 0xA004060Cu);
    glass(MODAL_X, MODAL_Y, MODAL_W, MODAL_H, 34, 0x12FFFFFFu);
    ui_rrect_grad(MODAL_X + 48, MODAL_Y + 36, 40, 4, 2, C_ACCENT, C_ACCENT2);
    ui_text(FONT_H1, MODAL_X + 48, MODAL_Y + 52, modal.title_color, modal.title);
    int y = MODAL_Y + 120;
    for (int i = 0; i < 5 && modal.body[i]; ++i) {
        char l1[200], l2[200];
        ui_text_wrap2(FONT_SMALL, modal.body[i], MODAL_W - 96, l1, l2, sizeof l1);
        ui_text(FONT_SMALL, MODAL_X + 48, y, C_DIM, l1);
        y += 29;
        if (l2[0]) { ui_text(FONT_SMALL, MODAL_X + 48, y, C_DIM, l2); y += 29; }
        y += 6;
    }
    int oy = MODAL_Y + MODAL_H - 40 - modal.nopts * 58;
    for (int i = 0; i < modal.nopts; ++i, oy += 58) {
        int s = i == modal.row;
        if (s) {
            ui_rrect_grad(MODAL_X + 32, oy, MODAL_W - 64, 50, 25, 0x4D3FE0CFu, 0x4DA78BFAu);
            ui_rrect_stroke_grad(MODAL_X + 32, oy, MODAL_W - 64, 50, 25, 1, 0x90FFFFFFu, 0x20FFFFFFu);
        } else {
            ui_rrect(MODAL_X + 32, oy, MODAL_W - 64, 50, 25, 0x0AFFFFFFu);
        }
        ui_text(FONT_BODY, MODAL_X + 64, oy + 11, s ? C_TEXT : C_DIM, modal.opts[i]);
    }
}

static const hint_t HINTS_MODAL[] = { { UI_BTN_CROSS, NULL, "Aceptar" }, { UI_BTN_CIRCLE, NULL, "Volver" } };

// ------------------------------------------------------------ vistas y redibujo

static void draw_view(void)
{
    if (view == V_MODAL) {
        if (!under_ok) {   // sin memoria para el desenfoque: la vista de atras, oscurecida
            if (modal_of == V_GRID) draw_grid_view();
            else draw_settings_view();
        }
        draw_modal();
        draw_hints(HINTS_MODAL, 2);
        return;
    }
    if (view == V_GRID) draw_grid_view();
    else draw_settings_view();
}

// redibuja una zona: fondo + toda la vista recortada a esa zona
static void repaint(int x, int y, int w, int h)
{
    ui_target(&canvas);
    ui_clip(x, y, w, h);
    ui_copy(view == V_MODAL && under_ok ? &under : &bg, x, y, w, h);
    draw_view();
    ui_clip_all();
    mark(x, y, w, h);
}

static void repaint_all(void) { repaint(0, 0, W, H); }
static void repaint_card(int k)
{
    if (k < 0 || k >= nvis) return;
    int x, y, w, h;
    card_region(k, &x, &y, &w, &h);
    repaint(x, y, w, h);
}
static void repaint_modal_options(void)
{
    repaint(MODAL_X, MODAL_Y + MODAL_H - 40 - modal.nopts * 58, MODAL_W, modal.nopts * 58);
}

// -------------------------------------------------------------- dialogos (logica)
enum { M_NONE, M_EXIT, M_OPTIONS, M_CONFIRM_UNINSTALL, M_CONFIRM_RESET, M_SAVE_ERROR, M_DIAG, M_SAVED };
static int modal_kind;
static char modal_lines[5][200];

static void open_modal(int kind)
{
    memset(&modal, 0, sizeof modal);
    modal_kind = kind;
    modal.title_color = C_TEXT;
    if (view != V_MODAL) {
        modal_of = view;
        snapshot_under();
    }
    view = V_MODAL;
    switch (kind) {
    case M_EXIT:
        modal.title = "\xbfSalir de VCFG-NX?";
        modal.body[0] = dirty_cfg ? "Hay cambios sin guardar." : "No hay cambios pendientes.";
        modal.body[1] = "Guardar instala el plugin y actualiza plugins.ini (con copia de seguridad).";
        modal.opts[0] = "Guardar y salir"; modal.opts[1] = "Salir sin guardar"; modal.opts[2] = "Cancelar";
        modal.nopts = 3;
        break;
    case M_OPTIONS:
        modal.title = "Opciones";
        modal.opts[0] = "Activar VCFG-NX en todos los juegos";
        modal.opts[1] = "Desactivar VCFG-NX en todos los juegos";
        modal.opts[2] = "Quitar los ajustes propios de todos";
        modal.opts[3] = "Diagn\xf3stico";
        modal.opts[4] = "Desinstalar VCFG-NX";
        modal.opts[5] = "Volver";
        modal.nopts = 6;
        break;
    case M_CONFIRM_UNINSTALL:
        modal.title = "\xbf" "Desinstalar VCFG-NX?"; modal.title_color = C_WARN;
        modal.body[0] = "Se quita el plugin de plugins.ini (con copia de seguridad)";
        modal.body[1] = "y se borra /data/GoldHEN/plugins/vcfg_nx.prx.";
        modal.body[2] = "Los ajustes de /data/vcfg_nx se conservan.";
        modal.opts[0] = "Desinstalar"; modal.opts[1] = "Cancelar"; modal.nopts = 2; modal.row = 1;
        break;
    case M_CONFIRM_RESET:
        modal.title = "\xbfQuitar los ajustes propios?"; modal.title_color = C_WARN;
        modal.body[0] = "Todos los juegos pasan a usar los ajustes por defecto.";
        modal.body[1] = "Se aplica al guardar y salir.";
        modal.opts[0] = "Quitar"; modal.opts[1] = "Cancelar"; modal.nopts = 2; modal.row = 1;
        break;
    case M_SAVE_ERROR:
        modal.title = "No se pudo guardar"; modal.title_color = C_DANGER;
        modal.body[0] = diag.save_error;
        modal.body[1] = "Cada archivo se verifica antes de reemplazarlo: no queda nada a medias.";
        modal.opts[0] = "Reintentar"; modal.opts[1] = "Ver diagn\xf3stico"; modal.opts[2] = "Salir sin guardar";
        modal.opts[3] = "Volver"; modal.nopts = 4;
        break;
    case M_DIAG:
        modal.title = "Diagn\xf3stico";
        snprintf(modal_lines[0], 200, "GoldHEN SDK %x.%02x \xb7 acceso fuera del sandbox: %s", diag.sdk >> 8, diag.sdk & 0xFF,
                 jailed ? "NO" : "s\xed");
        snprintf(modal_lines[1], 200, "Plugin en la app: %s", prx ? prx_from : "NO ENCONTRADO");
        snprintf(modal_lines[2], 200, "Juegos: %d \xb7 iconos: %d (del app.pkg: %d) \xb7 sin icono: %d%s%s", ngames,
                 diag.icons_ok, diag.icons_from_pkg, diag.icons_fail, diag.icons_fail ? " \xb7 primero: " : "",
                 diag.icons_fail ? diag.icon_fail_path : "");
        snprintf(modal_lines[3], 200, "\xdaltimo error al guardar: %s", diag.save_error[0] ? diag.save_error : "ninguno");
        {   // lo que dejo el plugin la ultima vez que se abrio el juego elegido
            const inst_game_t* g = nvis ? &games[vis[sel]] : NULL;
            char path[160], sum[160];
            char* log = NULL;
            if (g) {
                snprintf(path, sizeof path, CFG_DIR "/logs/%s.log", g->id);
                log = read_text(path);
            }
            inst_log_summary(log, sum, sizeof sum);
            snprintf(modal_lines[4], 200, "%.40s: %s", g ? g->title : "Juego", g ? sum : "no hay juegos");
            free(log);
        }
        for (int i = 0; i < 5; ++i) modal.body[i] = modal_lines[i];
        modal.opts[0] = "Volver"; modal.nopts = 1;
        break;
    case M_SAVED:
        modal.title = "Guardado"; modal.title_color = C_OK;
        modal.body[0] = "Mantener L3 + R3 dentro del juego abre el men\xfa de VCFG-NX.";
        modal.nopts = 0;
        break;
    }
    repaint_all();
}

static void close_modal(void)
{
    under_ok = 0;
    view = modal_of;
    modal_kind = M_NONE;
    repaint_all();
}

// ----------------------------------------------------------------- mando
#define DPAD (ORBIS_PAD_BUTTON_UP | ORBIS_PAD_BUTTON_DOWN | ORBIS_PAD_BUTTON_LEFT | ORBIS_PAD_BUTTON_RIGHT)
static int pad;

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
        if (now - held_at > 380000 && now - last_rep > 85000) {
            pressed |= d.buttons & DPAD;
            last_rep = now;
        }
    } else {
        held_at = now;
    }
    prev = d.buttons;
    return pressed;
}

static void notify(const char* m)
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

// ----------------------------------------------------------------- audio
static int audio_port = -1;
static volatile int audio_run;
static OrbisPthread audio_th;

static void* audio_thread(void* arg)
{
    (void)arg;
    static int16_t buf[256 * 2];
    while (audio_run) {
        if (ui_sound_busy()) {
            ui_sound_mix(buf, 256);
            sceAudioOutOutput(audio_port, buf);   // bloquea ~5 ms
        } else {
            sceKernelUsleep(6000);
        }
    }
    return NULL;
}

static void audio_start(void)
{
    ui_sound_init();
    sceAudioOutInit();   // puede devolver "ya inicializado": no importa
    audio_port = sceAudioOutOpen(0xFF, ORBIS_AUDIO_OUT_PORT_TYPE_MAIN, 0, 256, UI_SOUND_RATE,
                                 ORBIS_AUDIO_OUT_PARAM_FORMAT_S16_STEREO);
    if (audio_port < 0) return;
    audio_run = 1;
    if (scePthreadCreate(&audio_th, NULL, audio_thread, NULL, "vcfg_audio") != 0) {
        audio_run = 0;
        sceAudioOutClose(audio_port);
        audio_port = -1;
    }
}

static void audio_stop(void)
{
    if (!audio_run) return;
    for (int i = 0; i < 100 && ui_sound_busy(); ++i) sceKernelUsleep(10000);   // que termine el ultimo sonido
    audio_run = 0;
    scePthreadJoin(audio_th, NULL);
    sceAudioOutClose(audio_port);
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

// un icono que falte de los que se ven (o el grande de la pantalla de ajustes)
static int load_one_icon(void)
{
    if (view != V_GRID && !(view == V_MODAL && modal_of == V_GRID)) {
        int gi = settings_game;
        if (gi >= 0 && big_icon.game != gi) {
            ui_img_free(&big_icon.img);
            big_icon.game = gi;
            big_icon.img = decode_icon(gi, BIG_ICON, 40);
            big_icon.state = big_icon.img.px ? 1 : -1;
            if (view == V_GAME) repaint(60, 100, PANEL_X - 60, BIG_ICON + 120);
            return 1;
        }
        return 0;
    }
    int first = top_row * COLS, last = first + COLS * ROWS;
    // primero la elegida, despues el resto de la pagina, despues la pagina siguiente
    int order[COLS * ROWS * 2 + 1], n = 0;
    order[n++] = sel;
    for (int k = first; k < last + COLS * ROWS; ++k) order[n++] = k;
    for (int i = 0; i < n; ++i) {
        int k = order[i];
        if (k < 0 || k >= nvis || icon_lookup(vis[k])) continue;
        icon_load(vis[k]);
        if (k >= first && k < last && view == V_GRID) repaint_card(k);
        return 1;
    }
    return 0;
}

// ------------------------------------------------------------------ main
static void grid_move(int nsel)
{
    if (nsel == sel || nsel < 0 || nsel >= nvis) return;
    int old = sel, old_top = top_row;
    sel = nsel;
    if (sel / COLS < top_row) top_row = sel / COLS;
    if (sel / COLS >= top_row + ROWS) top_row = sel / COLS - ROWS + 1;
    ui_sound_play(SND_MOVE);
    if (top_row != old_top) {
        repaint(0, GRID_Y - 60, W, DETAIL_Y - GRID_Y + 60);   // cambio de pagina: la cuadricula entera
    } else {
        repaint_card(old);
        repaint_card(sel);
    }
    repaint(0, DETAIL_Y - 4, W, 110);
}

static int handle_modal(uint32_t b, int* quit)
{
    if (modal_kind == M_SAVED) return 0;
    if (b & ORBIS_PAD_BUTTON_UP && modal.nopts) {
        modal.row = (modal.row + modal.nopts - 1) % modal.nopts;
        ui_sound_play(SND_MOVE);
        repaint_modal_options();
    }
    if (b & ORBIS_PAD_BUTTON_DOWN && modal.nopts) {
        modal.row = (modal.row + 1) % modal.nopts;
        ui_sound_play(SND_MOVE);
        repaint_modal_options();
    }
    if (b & ORBIS_PAD_BUTTON_CIRCLE) {
        ui_sound_play(SND_BACK);
        if (modal_kind == M_DIAG || modal_kind == M_CONFIRM_RESET) open_modal(M_OPTIONS);
        else close_modal();
        return 0;
    }
    if (!(b & ORBIS_PAD_BUTTON_CROSS)) return 0;
    int r = modal.row;
    switch (modal_kind) {
    case M_EXIT:
    case M_SAVE_ERROR:
        if ((modal_kind == M_EXIT && r == 0) || (modal_kind == M_SAVE_ERROR && r == 0)) {
            errno = 0;
            int e = save_all(0);
            if (e == 0) {
                dirty_cfg = 0;
                diag.save_error[0] = 0;
                ui_sound_play(SND_SAVE);
                open_modal(M_SAVED);
                present();
                notify("VCFG-NX: ajustes guardados. Mantener L3+R3 en el juego abre el men\xfa.");
                sceKernelUsleep(700000);
                *quit = 1;
            } else {
                snprintf(diag.save_error, sizeof diag.save_error, "%s (c\xf3" "digo %d%s%s)", save_error_text(e), e,
                         errno ? ", " : "", errno ? strerror(errno) : "");
                ui_sound_play(SND_ERROR);
                open_modal(M_SAVE_ERROR);
            }
        } else if ((modal_kind == M_EXIT && r == 1) || (modal_kind == M_SAVE_ERROR && r == 2)) {
            ui_sound_play(SND_BACK);
            *quit = 1;
        } else if (modal_kind == M_SAVE_ERROR && r == 1) {
            ui_sound_play(SND_SELECT);
            open_modal(M_DIAG);
        } else {
            ui_sound_play(SND_BACK);
            close_modal();
        }
        break;
    case M_OPTIONS:
        ui_sound_play(SND_SELECT);
        if (r == 0 || r == 1) {
            for (int i = 0; i < ngames; ++i) games[i].enabled = r == 0;
            dirty_cfg = 1;
            set_msg(C_DIM, "VCFG-NX %s en todos los juegos (se aplica al guardar).", r == 0 ? "activado" : "desactivado");
            rebuild_visible();
            close_modal();
        } else if (r == 2) {
            open_modal(M_CONFIRM_RESET);
        } else if (r == 3) {
            open_modal(M_DIAG);
        } else if (r == 4) {
            open_modal(M_CONFIRM_UNINSTALL);
        } else {
            close_modal();
        }
        break;
    case M_CONFIRM_UNINSTALL:
        if (r == 0) {
            int e = save_all(1);
            notify(e == 0 ? "VCFG-NX desinstalado" : "VCFG-NX: error al desinstalar");
            *quit = 1;
        } else {
            ui_sound_play(SND_BACK);
            open_modal(M_OPTIONS);
        }
        break;
    case M_CONFIRM_RESET:
        if (r == 0) {
            for (int i = 0; i < ngames; ++i)
                if (games[i].custom) { games[i].custom = 0; games[i].cfg_dirty = 1; }
            dirty_cfg = 1;
            set_msg(C_DIM, "Ajustes propios quitados (se aplica al guardar).");
            rebuild_visible();
            ui_sound_play(SND_SELECT);
            close_modal();
        } else {
            ui_sound_play(SND_BACK);
            open_modal(M_OPTIONS);
        }
        break;
    case M_DIAG:
        ui_sound_play(SND_BACK);
        open_modal(M_OPTIONS);
        break;
    }
    return 0;
}

static void handle_grid(uint32_t b)
{
    if (nvis) {
        if (b & ORBIS_PAD_BUTTON_LEFT) grid_move(sel - 1);
        if (b & ORBIS_PAD_BUTTON_RIGHT) grid_move(sel + 1);
        if (b & ORBIS_PAD_BUTTON_UP) grid_move(sel - COLS);
        if (b & ORBIS_PAD_BUTTON_DOWN) {
            if (sel + COLS < nvis) grid_move(sel + COLS);
            else if (sel / COLS < (nvis - 1) / COLS) grid_move(nvis - 1);   // ultima fila incompleta
        }
        if (b & ORBIS_PAD_BUTTON_CROSS) {
            inst_game_t* g = &games[vis[sel]];
            g->enabled ^= 1;
            dirty_cfg = 1;
            ui_sound_play(SND_SELECT);
            set_msg(C_DIM, "%s: VCFG-NX %s", g->title, g->enabled ? "activado" : "desactivado");
            if (filter == F_ON) { rebuild_visible(); repaint_all(); }
            else { repaint_card(sel); repaint(0, 0, W, 200); repaint(0, DETAIL_Y - 4, W, H - DETAIL_Y + 4); }
        }
        if (b & ORBIS_PAD_BUTTON_SQUARE) {
            ui_sound_play(SND_SELECT);
            settings_game = vis[sel];
            settings_row = R_ENABLED;
            view = V_GAME;
            set_msg(C_DIM, "%s", "");
            repaint_all();
            return;
        }
    }
    if (b & ORBIS_PAD_BUTTON_TRIANGLE) {
        ui_sound_play(SND_SELECT);
        settings_game = -1;
        settings_row = R_FG;
        view = V_GAME;
        set_msg(C_DIM, "%s", "");
        repaint_all();
        return;
    }
    if (b & (ORBIS_PAD_BUTTON_L1 | ORBIS_PAD_BUTTON_R1)) {
        filter = (filter + ((b & ORBIS_PAD_BUTTON_R1) ? 1 : F_COUNT - 1)) % F_COUNT;
        rebuild_visible();
        top_row = 0;
        ui_sound_play(SND_TAB);
        repaint_all();
    }
    if (b & ORBIS_PAD_BUTTON_OPTIONS) { ui_sound_play(SND_SELECT); open_modal(M_OPTIONS); }
    if (b & ORBIS_PAD_BUTTON_CIRCLE) { ui_sound_play(SND_BACK); open_modal(M_EXIT); }
}

static void handle_settings(uint32_t b)
{
    int gi = settings_game, first = gi < 0 ? R_FG : R_ENABLED, old = settings_row;
    if (b & ORBIS_PAD_BUTTON_UP) settings_row = settings_row > first ? settings_row - 1 : R_COUNT - 1;
    if (b & ORBIS_PAD_BUTTON_DOWN) settings_row = settings_row < R_COUNT - 1 ? settings_row + 1 : first;
    if (settings_row != old) {
        ui_sound_play(SND_MOVE);
        int x, y, w, h;
        row_region(old, &x, &y, &w, &h); repaint(x, y, w, h);
        row_region(settings_row, &x, &y, &w, &h); repaint(x, y, w, h);
        repaint(PANEL_X, PANEL_Y + PANEL_H - 70, PANEL_W, 60);   // texto de ayuda
    }
    int dir = (b & ORBIS_PAD_BUTTON_LEFT) ? -1 : ((b & (ORBIS_PAD_BUTTON_RIGHT | ORBIS_PAD_BUTTON_CROSS)) ? 1 : 0);
    if (dir) {
        dirty_cfg = 1;
        ui_sound_play(SND_TAB);
        int was_custom = gi >= 0 && games[gi].custom;
        if (gi < 0) {
            change(&defaults, settings_row, dir);
        } else {
            inst_game_t* g = &games[gi];
            if (settings_row == R_ENABLED) g->enabled ^= 1;
            else if (settings_row == R_MODE) { if (g->custom) { g->custom = 0; g->cfg_dirty = 1; } else make_custom(g); }
            else { make_custom(g); change(&g->cfg, settings_row, dir); g->cfg_dirty = 1; }
        }
        if (gi >= 0 && (was_custom != games[gi].custom || settings_row == R_ENABLED)) repaint_all();
        else { int x, y, w, h; row_region(settings_row, &x, &y, &w, &h); repaint(x, y, w, h); }
    }
    if (gi >= 0 && (b & ORBIS_PAD_BUTTON_SQUARE) && games[gi].custom) {
        games[gi].custom = 0;
        games[gi].cfg_dirty = 1;
        dirty_cfg = 1;
        ui_sound_play(SND_BACK);
        repaint_all();
    }
    if (b & ORBIS_PAD_BUTTON_CIRCLE) {
        ui_sound_play(SND_BACK);
        view = V_GRID;
        rebuild_visible();
        set_msg(C_DIM, "%s", dirty_cfg ? "Cambios pendientes: se guardan al salir." : "");
        repaint_all();
    }
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    if (video_init() != 0) {
        notify("VCFG-NX: no se pudo abrir la pantalla");
        sceSystemServiceLoadExec("exit", NULL);
        return 1;
    }
    int user = -1;
    sceUserServiceInitialize(NULL);
    sceUserServiceGetInitialUser(&user);
    scePadInit();
    pad = scePadOpen(user, 0, 0, NULL);
    audio_start();

    // pantalla de carga mientras se arma todo
    build_background();
    ui_target(&canvas);
    ui_copy(&bg, 0, 0, W, H);
    ui_text(FONT_LOGO, 60, 28, C_TEXT, "VCFG-NX");
    ui_text(FONT_BODY, 60, 140, C_DIM, "Cargando juegos...");
    mark(0, 0, W, H);
    present();

    load_prx(0);   // antes de salir del sandbox: ahi /app0 es la app
    static struct jailbreak_backup jb;
    diag.sdk = sys_sdk_version();
    jailed = sys_sdk_jailbreak(&jb) != 0;
    load_prx(1);
    load_defaults();
    load_games();
    rebuild_visible();
    build_sprites();
    if (jailed)
        set_msg(C_WARN, "GoldHEN no dio acceso fuera del sandbox: \xbf" "est\xe1 activo y actualizado?");
    else if (!prx)
        set_msg(C_WARN, "No se encontr\xf3 el plugin dentro de la app: reinstal\xe1 el PKG.");
    else
        set_msg(C_DIM, "Eleg\xed un juego con las flechas.");
    repaint_all();
    present();

    int quit = 0;
    while (!quit) {
        uint32_t b = read_pad();
        if (b) {
            if (view == V_MODAL) handle_modal(b, &quit);
            else if (view == V_GRID) handle_grid(b);
            else handle_settings(b);
        } else if (!load_one_icon()) {
            sceKernelUsleep(16000);   // nada que hacer: solo mirar el mando
            continue;
        }
        int k = cur_fb;
        if (dirty[k].x1 > dirty[k].x0 || dirty[k ^ 1].x1 > dirty[k ^ 1].x0) present();
    }
    audio_stop();
    if (!jailed) sys_sdk_unjailbreak(&jb);
    sceSystemServiceLoadExec("exit", NULL);
    for (;;) sceKernelUsleep(1000000);
    return 0;
}
