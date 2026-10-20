// Instalador de VCFG-NX (app homebrew para PS4 con GoldHEN).
//
// Al abrirla:
//   1. copia /app0/plugins/vcfg_nx.prx a /data/GoldHEN/plugins/
//   2. crea /data/vcfg_nx/config.ini y /data/vcfg_nx/titles.txt si no existen
//   3. agrega el plugin a /data/GoldHEN/plugins.ini para cada juego de
//      titles.txt que este instalado (/user/app/<TITLE_ID>), con copia de
//      seguridad del plugins.ini anterior
//   4. avisa con una notificacion y se cierra
// Es idempotente: se puede volver a abrir despues de editar titles.txt.
#include <dirent.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include <orbis/SystemService.h>
#include <orbis/libkernel.h>

#define PLUGIN_SRC "/app0/plugins/vcfg_nx.prx"
#define PLUGIN_DIR "/data/GoldHEN/plugins"
#define PLUGIN_DST PLUGIN_DIR "/vcfg_nx.prx"
#define PLUGINS_INI "/data/GoldHEN/plugins.ini"
#define CFG_DIR "/data/vcfg_nx"
#define MAX_TITLES 64

static void notify(const char* fmt, ...)
{
    OrbisNotificationRequest req;
    memset(&req, 0, sizeof req);
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(req.message, sizeof req.message, fmt, ap);
    va_end(ap);
    printf("[vcfg_nx installer] %s\n", req.message);
    req.type = NotificationRequest;
    req.useIconImageUri = 1;
    req.targetId = -1;
    strcpy(req.iconUri, "cxml://psnotification/tex_icon_system");
    sceKernelSendNotificationRequest(0, &req, sizeof req, 0);
}

static int file_exists(const char* p)
{
    struct stat st;
    return stat(p, &st) == 0;
}

static int copy_file(const char* src, const char* dst)
{
    FILE* in = fopen(src, "rb");
    if (!in) return -1;
    char tmp[256];
    snprintf(tmp, sizeof tmp, "%s.tmp", dst);
    FILE* out = fopen(tmp, "wb");
    if (!out) { fclose(in); return -2; }
    char buf[64 * 1024];
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

static char* read_all(const char* path, size_t* len)
{
    FILE* f = fopen(path, "rb");
    if (!f) { *len = 0; return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* b = malloc((size_t)n + 1);
    *len = fread(b, 1, (size_t)n, f);
    b[*len] = 0;
    fclose(f);
    return b;
}

static void write_text(const char* path, const char* text)
{
    FILE* f = fopen(path, "w");
    if (f) { fputs(text, f); fclose(f); }
}

static const char* DEFAULT_CONFIG =
    "; VCFG-NX: configuracion global. Para un juego concreto crear /data/vcfg_nx/<TITLE_ID>.ini\n"
    "framegen = 1        ; generar fotogramas intermedios (30 -> 60 fps)\n"
    "aa = 1              ; SMAA 1x: antialiasing de bordes en fotogramas reales y generados\n"
    "post = 1            ; 0 nada, 1 RCAS (nitidez), 2 FSR1 completo (EASU+RCAS) si el juego renderiza < salida\n"
    "sharpness = 0.25    ; 0 = maxima nitidez, 2 = suave\n"
    "max_game_fps = 48   ; si el juego ya corre mas rapido, frame generation se apaga solo\n"
    "toggle_hold_ms = 1500 ; mantener L3+R3 este tiempo para activar/desactivar\n";

// Lista inicial: los juegos que soportaba el VCFG original. Se pueden agregar
// mas Title IDs (uno por linea) y volver a abrir el instalador.
static const char* DEFAULT_TITLES =
    "; Juegos donde se activa VCFG-NX (un Title ID por linea). Volver a abrir el instalador tras editar.\n"
    "CUSA00207\n"
    "CUSA00411\n"
    "CUSA07439\n"
    "CUSA08519\n";

static int load_titles(char titles[][10], int max)
{
    size_t len;
    char* b = read_all(CFG_DIR "/titles.txt", &len);
    if (!b) return 0;
    int n = 0;
    for (char* line = strtok(b, "\r\n"); line && n < max; line = strtok(NULL, "\r\n")) {
        while (*line == ' ' || *line == '\t') ++line;
        if (strncmp(line, "CUSA", 4) != 0 || strlen(line) < 9) continue;
        memcpy(titles[n], line, 9);
        titles[n][9] = 0;
        ++n;
    }
    free(b);
    return n;
}

// Devuelve 1 si la seccion [title] de "ini" ya contiene la linea del plugin.
static int section_has_plugin(const char* ini, const char* title)
{
    char hdr[16];
    snprintf(hdr, sizeof hdr, "[%s]", title);
    const char* s = strstr(ini, hdr);
    if (!s) return 0;
    const char* end = strstr(s + 1, "\n[");
    const char* p = strstr(s, PLUGIN_DST);
    return p && (!end || p < end);
}

// Inserta la linea del plugin en la seccion [title] (la crea si no existe).
static char* add_to_section(char* ini, const char* title)
{
    char hdr[16];
    snprintf(hdr, sizeof hdr, "[%s]", title);
    size_t len = strlen(ini);
    char* out = malloc(len + 128);
    const char* s = strstr(ini, hdr);
    if (!s) {
        snprintf(out, len + 128, "%s%s\n%s\n%s\n", ini, (len && ini[len - 1] != '\n') ? "\n" : "", hdr, PLUGIN_DST);
    } else {
        const char* eol = strchr(s, '\n');
        size_t at = eol ? (size_t)(eol - ini) + 1 : len;
        memcpy(out, ini, at);
        int k = sprintf(out + at, "%s%s\n", eol ? "" : "\n", PLUGIN_DST);
        strcpy(out + at + k, ini + at);
    }
    free(ini);
    return out;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    mkdir("/data/GoldHEN", 0777);
    mkdir(PLUGIN_DIR, 0777);
    mkdir(CFG_DIR, 0777);

    int r = copy_file(PLUGIN_SRC, PLUGIN_DST);
    if (r != 0) {
        notify("VCFG-NX: no se pudo copiar el plugin (%d). Esta GoldHEN activo?", r);
        goto done;
    }
    if (!file_exists(CFG_DIR "/config.ini")) write_text(CFG_DIR "/config.ini", DEFAULT_CONFIG);
    if (!file_exists(CFG_DIR "/titles.txt")) write_text(CFG_DIR "/titles.txt", DEFAULT_TITLES);

    char titles[MAX_TITLES][10];
    int nt = load_titles(titles, MAX_TITLES);
    size_t len;
    char* ini = read_all(PLUGINS_INI, &len);
    if (!ini) {
        ini = malloc(1);
        ini[0] = 0;
    } else {
        char bak[128];
        snprintf(bak, sizeof bak, PLUGINS_INI ".vcfg_nx_backup_%ld", (long)time(NULL));
        copy_file(PLUGINS_INI, bak);
    }
    int added = 0, installed = 0;
    for (int i = 0; i < nt; ++i) {
        char app[64];
        snprintf(app, sizeof app, "/user/app/%s", titles[i]);
        if (!file_exists(app)) continue;
        ++installed;
        if (section_has_plugin(ini, titles[i])) continue;
        ini = add_to_section(ini, titles[i]);
        ++added;
    }
    if (added) write_text(PLUGINS_INI, ini);
    free(ini);

    if (installed)
        notify("VCFG-NX instalado: %d juego(s) configurado(s) (%d nuevo(s)). L3+R3 en el juego: on/off.",
               installed, added);
    else
        notify("VCFG-NX copiado, pero ningun juego de /data/vcfg_nx/titles.txt esta instalado.");

done:
    sceKernelUsleep(4 * 1000 * 1000);
    sceSystemServiceLoadExec("exit", NULL);
    for (;;) sceKernelUsleep(1000 * 1000);
    return 0;
}
