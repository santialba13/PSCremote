// VCFG-NX 2.0 - app de configuracion e instalacion (estilo del lanzador de VCFG).
//
// Muestra los juegos instalados y permite activar VCFG-NX en cada uno, editar
// los ajustes por defecto, desinstalar, y guardar. Al guardar:
//   * copia /app0/plugins/vcfg_nx.prx a /data/GoldHEN/plugins/
//   * hace una copia de seguridad de plugins.ini (una por sesion) y lo reescribe
//     solo con los cambios de VCFG-NX, verificando que se haya escrito bien
//   * escribe /data/vcfg_nx/config.ini con los ajustes por defecto
// La logica de archivos esta en inst_core.c (testeada en el PC).
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

#define W 1920
#define H 1080
#define PLUGIN_SRC "/app0/plugins/vcfg_nx.prx"
#define PLUGIN_DIR "/data/GoldHEN/plugins"
#define PLUGINS_INI "/data/GoldHEN/plugins.ini"
#define CFG_DIR "/data/vcfg_nx"
#define VERSION "2.1"

#define C_BG 0xFF0E1220u
#define C_PANEL 0xFF161C2Cu
#define C_SEL 0xFF1E4A9Eu
#define C_TEXT 0xFFF0F4FFu
#define C_DIM 0xFF8A96B4u
#define C_ACCENT 0xFF5AA0FFu
#define C_OK 0xFF5AD27Au
#define C_WARN 0xFFFFB45Au

static uint32_t* fb[2];
static int video, cur_fb;
static int pad;

// ------------------------------------------------------------- dibujo
static void rect(int x, int y, int w, int h, uint32_t c)
{
    uint32_t* p = fb[cur_fb];
    for (int j = y < 0 ? 0 : y; j < y + h && j < H; ++j)
        for (int i = x < 0 ? 0 : x; i < x + w && i < W; ++i) p[j * W + i] = c;
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
    sceKernelUsleep(33000);
}

// ------------------------------------------------------------ archivos
static char* read_text(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) { char* e = malloc(1); e[0] = 0; return e; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* b = malloc((size_t)n + 1);
    size_t r = fread(b, 1, (size_t)n, f);
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
static int ngames, sel, top, dirty, backup_done;
static int set_fg = 1, set_sr = 2, set_model = 0, set_dyn = 1, set_aa = 1, set_rcas = 0;
static char msg[128];
static uint32_t msg_color = C_DIM;

static void load_defaults(void)
{
    char* t = read_text(CFG_DIR "/config.ini");
    const char* p;
    if ((p = strstr(t, "framegen ="))) set_fg = atoi(p + 10) != 0;
    if ((p = strstr(t, "\nsr ="))) set_sr = atoi(p + 5);
    if ((p = strstr(t, "\naa ="))) set_aa = atoi(p + 5) != 0;
    if ((p = strstr(t, "\nrcas ="))) set_rcas = atoi(p + 7);
    if ((p = strstr(t, "\nsr_model ="))) set_model = atoi(p + 11);
    if ((p = strstr(t, "\ndynamic ="))) set_dyn = atoi(p + 10) != 0;
    free(t);
    // un ini editado a mano no puede dejar indices fuera de rango
    if (set_sr < 0 || set_sr > 3) set_sr = 2;
    if (set_rcas < 0 || set_rcas > 3) set_rcas = 0;
    if (set_model < 0 || set_model > 2) set_model = 0;
}

static int save_all(int uninstall)
{
    mkdir("/data/GoldHEN", 0777);
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
    char cfg[512];
    snprintf(cfg, sizeof cfg,
             "; VCFG-NX: ajustes por defecto (cada juego puede guardar los suyos desde el menu L3+R3)\n"
             "framegen = %d\nsr = %d\nsr_model = %d\ndynamic = %d\naa = %d\nrcas = %d\n", set_fg, set_sr, set_model,
             set_dyn, set_aa, set_rcas);
    return write_text_verified(CFG_DIR "/config.ini", cfg) == 0 ? 0 : -40;
}

// ---------------------------------------------------------------- vistas
static const char* SR_N[] = { "OFF", "Suave", "Normal", "Fuerte" };
static const char* RC_N[] = { "OFF", "Baja", "Media", "Alta" };
static const char* MD_N[] = { "Auto", "R\xe1pido", "Calidad" };
#define SET_ROWS 6

static void header(void)
{
    rect(0, 0, W, H, C_BG);
    for (int y = 0; y < 150; ++y) rect(0, y, W, 1, 0xFF000000u | ((14 + y / 6) << 16) | ((18 + y / 4) << 8) | (32 + y / 2));
    text(80, 34, 5, C_TEXT, "VCFG-NX");
    text(80 + 8 * 5 * 8, 58, 3, C_ACCENT, VERSION);
    text(80, 112, 2, C_DIM, "Generaci\xf3n de fotogramas  +  Reescalado IA (NXSR)  +  SMAA / FSR1");
    rect(0, 150, W, 3, C_ACCENT);
}

static void footer(const char* keys)
{
    rect(0, H - 120, W, 120, C_PANEL);
    text(80, H - 104, 2, msg_color, "%s", msg);
    text(80, H - 60, 2, C_TEXT, "%s", keys);
}

static void draw_games(void)
{
    header();
    text(80, 180, 3, C_TEXT, "Juegos instalados");
    int on = 0;
    for (int i = 0; i < ngames; ++i) on += games[i].enabled;
    text(1180, 192, 2, C_DIM, "%d juegos, %d con VCFG-NX", ngames, on);
    const int rows = 14, y0 = 250, rh = 46;
    if (sel < top) top = sel;
    if (sel >= top + rows) top = sel - rows + 1;
    for (int r = 0; r < rows && top + r < ngames; ++r) {
        inst_game_t* g = &games[top + r];
        int y = y0 + r * rh;
        if (top + r == sel) rect(60, y - 6, W - 120, rh - 4, C_SEL);
        text(90, y, 2, g->enabled ? C_OK : C_DIM, g->enabled ? "[ON ]" : "[OFF]");
        text(220, y, 2, C_TEXT, "%.60s", g->title);
        text(W - 320, y, 2, C_DIM, "%s", g->id);
    }
    if (!ngames) text(90, y0, 2, C_WARN, "No se encontraron juegos en /user/app");
    rect(60, 900, W - 120, 2, C_PANEL);
    text(80, 920, 2, C_DIM, "Ajustes por defecto:  FG %s   NXSR %s (%s)   SMAA %s   RCAS %s   din\xe1mica %s",
         set_fg ? "ON" : "OFF", SR_N[set_sr], MD_N[set_model], set_aa ? "ON" : "OFF", RC_N[set_rcas],
         set_dyn ? "ON" : "OFF");
    footer("X activar/desactivar   TRI todos   CUAD ajustes   OPTIONS desinstalar   O guardar y salir");
}

static void draw_settings(int row)
{
    header();
    text(80, 180, 3, C_TEXT, "Ajustes por defecto");
    const char* labels[SET_ROWS] = { "Generaci\xf3n de fotogramas (30 -> 60)", "Reescalado IA (NXSR)",
                                     "Modelo IA (Calidad: mejor en PS4 Pro)", "Antialiasing SMAA",
                                     "Nitidez FSR1 (RCAS, sin NXSR)", "Calidad din\xe1mica (evita tirones)" };
    const char* vals[SET_ROWS] = { set_fg ? "ON" : "OFF", SR_N[set_sr], MD_N[set_model], set_aa ? "ON" : "OFF",
                                   RC_N[set_rcas], set_dyn ? "ON" : "OFF" };
    for (int i = 0; i < SET_ROWS; ++i) {
        int y = 270 + i * 70;
        if (i == row) rect(60, y - 10, W - 120, 56, C_SEL);
        text(100, y, 3, C_TEXT, "%s", labels[i]);
        text(1300, y, 3, C_ACCENT, "< %s >", vals[i]);
    }
    text(100, 720, 2, C_DIM, "Dentro del juego: mantener L3+R3 abre el men\xfa de VCFG-NX (cada juego guarda lo suyo).");
    text(100, 760, 2, C_DIM, "NXSR nunca agrega grano: su salida queda acotada al rango de los p\xedxeles vecinos.");
    text(100, 800, 2, C_DIM, "Calidad din\xe1mica: si la GPU no da abasto baja un escal\xf3n en vez de trabar el juego.");
    text(100, 840, 2, C_WARN, "Beta: si un juego se cierra con VCFG-NX activo, la pr\xf3xima vez arranca en modo seguro.");
    footer("arriba/abajo elegir   izq/der cambiar   O volver");
}

static void draw_confirm(void)
{
    header();
    text(80, 300, 3, C_WARN, "\xbf" "Desinstalar VCFG-NX?");
    text(80, 380, 2, C_TEXT, "Se quita el plugin de plugins.ini (con copia de seguridad) y se borra vcfg_nx.prx.");
    text(80, 420, 2, C_TEXT, "Los ajustes en /data/vcfg_nx se conservan.");
    footer("X desinstalar   O cancelar");
}

// ----------------------------------------------------------------- main
static uint32_t read_pad(uint32_t* prev)
{
    OrbisPadData d;
    if (pad < 0 || scePadReadState(pad, &d) != 0) return 0;
    uint32_t pressed = d.buttons & ~*prev;
    *prev = d.buttons;
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
    ngames = inst_list_games("/user/app", "/user/appmeta", games, INST_MAX_GAMES);
    ngames += inst_list_games("/mnt/ext1/user/app", "/user/appmeta", games + ngames, INST_MAX_GAMES - ngames);
    char* ini = read_text(PLUGINS_INI);
    for (int i = 0; i < ngames; ++i) games[i].enabled = inst_ini_has(ini, games[i].id);
    free(ini);
    load_defaults();
    struct stat st;
    if (jailed || (stat(PLUGIN_DIR, &st) != 0 && stat("/data/GoldHEN", &st) != 0)) {
        snprintf(msg, sizeof msg, "No se encontr\xf3 /data/GoldHEN: \xbf" "est\xe1 GoldHEN activo?");
        msg_color = C_WARN;
    } else {
        snprintf(msg, sizeof msg, "Eleg\xed los juegos donde usar VCFG-NX y guard\xe1 con O.");
    }

    uint32_t prev = 0;
    int view = 0, srow = 0;
    for (;;) {
        uint32_t b = read_pad(&prev);
        if (view == 0) {
            if ((b & ORBIS_PAD_BUTTON_UP) && ngames) sel = (sel + ngames - 1) % ngames;
            if ((b & ORBIS_PAD_BUTTON_DOWN) && ngames) sel = (sel + 1) % ngames;
            if ((b & ORBIS_PAD_BUTTON_LEFT) && ngames) sel = sel > 13 ? sel - 14 : 0;
            if ((b & ORBIS_PAD_BUTTON_RIGHT) && ngames) sel = sel + 14 < ngames ? sel + 14 : ngames - 1;
            if ((b & ORBIS_PAD_BUTTON_CROSS) && ngames) { games[sel].enabled ^= 1; dirty = 1; }
            if (b & ORBIS_PAD_BUTTON_TRIANGLE) {
                int any_off = 0;
                for (int i = 0; i < ngames; ++i) any_off |= !games[i].enabled;
                for (int i = 0; i < ngames; ++i) games[i].enabled = any_off;
                dirty = 1;
            }
            if (b & ORBIS_PAD_BUTTON_SQUARE) { view = 1; srow = 0; }
            if (b & ORBIS_PAD_BUTTON_OPTIONS) view = 2;
            if (b & ORBIS_PAD_BUTTON_CIRCLE) {
                int r = save_all(0);
                if (r == 0) { notify_exit("VCFG-NX: ajustes guardados. Mantener L3+R3 en el juego: men\xfa"); break; }
                snprintf(msg, sizeof msg, "Error al guardar (%d). No se cambi\xf3 nada; reintent\xe1 con O.", r);
                msg_color = C_WARN;
            }
            draw_games();
        } else if (view == 1) {
            if (b & ORBIS_PAD_BUTTON_UP) srow = (srow + SET_ROWS - 1) % SET_ROWS;
            if (b & ORBIS_PAD_BUTTON_DOWN) srow = (srow + 1) % SET_ROWS;
            if (b & (ORBIS_PAD_BUTTON_LEFT | ORBIS_PAD_BUTTON_RIGHT | ORBIS_PAD_BUTTON_CROSS)) {
                int d = (b & ORBIS_PAD_BUTTON_LEFT) ? 3 : 1;
                if (srow == 0) set_fg ^= 1;
                if (srow == 1) set_sr = (set_sr + d) % 4;
                if (srow == 2) set_model = (set_model + (d == 3 ? 2 : 1)) % 3;
                if (srow == 3) set_aa ^= 1;
                if (srow == 4) set_rcas = (set_rcas + d) % 4;
                if (srow == 5) set_dyn ^= 1;
                dirty = 1;
            }
            if (b & ORBIS_PAD_BUTTON_CIRCLE) view = 0;
            draw_settings(srow);
        } else {
            if (b & ORBIS_PAD_BUTTON_CROSS) {
                int r = save_all(1);
                notify_exit(r == 0 ? "VCFG-NX desinstalado" : "VCFG-NX: error al desinstalar");
                break;
            }
            if (b & ORBIS_PAD_BUTTON_CIRCLE) view = 0;
            draw_confirm();
        }
        present();
    }
    sceKernelUsleep(500000);
    if (!jailed) sys_sdk_unjailbreak(&jb);
    sceSystemServiceLoadExec("exit", NULL);
    for (;;) sceKernelUsleep(1000000);
    return 0;
}
