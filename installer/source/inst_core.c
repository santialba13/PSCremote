#include "inst_core.h"

#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rd32(const unsigned char* p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t rd16(const unsigned char* p) { return (uint16_t)(p[0] | p[1] << 8); }

int inst_sfo_title(const unsigned char* sfo, size_t len, char* out, size_t cap)
{
    if (len < 20 || memcmp(sfo, "\0PSF", 4) != 0) return -1;
    uint32_t keys = rd32(sfo + 8), data = rd32(sfo + 12), n = rd32(sfo + 16);
    if (keys >= len || data >= len || 20 + (size_t)n * 16 > len) return -1;
    for (uint32_t i = 0; i < n; ++i) {
        const unsigned char* e = sfo + 20 + i * 16;
        uint32_t ko = keys + rd16(e), dlen = rd32(e + 4), dof = data + rd32(e + 12);
        if (ko >= len || dof + dlen > len) continue;
        if (strncmp((const char*)sfo + ko, "TITLE", 6) == 0) {
            size_t m = dlen < cap - 1 ? dlen : cap - 1;
            memcpy(out, sfo + dof, m);
            out[m] = 0;
            return 0;
        }
    }
    return -1;
}

static int cmp_game(const void* a, const void* b)
{
    const unsigned char* p = (const unsigned char*)((const inst_game_t*)a)->title;
    const unsigned char* q = (const unsigned char*)((const inst_game_t*)b)->title;
    for (;; ++p, ++q) {
        int x = *p >= 'a' && *p <= 'z' ? *p - 32 : *p, y = *q >= 'a' && *q <= 'z' ? *q - 32 : *q;
        if (x != y || !x) return x - y;
    }
}

int inst_list_games(const char* dir_app, const char* const* metas, inst_game_t* games, int n, int max)
{
    DIR* d = opendir(dir_app);
    if (!d) return n;
    struct dirent* e;
    while ((e = readdir(d)) && n < max) {
        if (strncmp(e->d_name, "CUSA", 4) != 0 || strlen(e->d_name) != 9) continue;
        int dup = 0;
        for (int i = 0; i < n; ++i) dup |= !strcmp(games[i].id, e->d_name);
        if (dup) continue;
        inst_game_t* g = &games[n++];
        memset(g, 0, sizeof *g);
        snprintf(g->id, sizeof g->id, "%.9s", e->d_name);
        snprintf(g->title, sizeof g->title, "%.9s", e->d_name);
        snprintf(g->app, sizeof g->app, "%.110s/%.9s", dir_app, e->d_name);
        for (int k = 0; metas && metas[k]; ++k) {
            char p[256];
            snprintf(p, sizeof p, "%.200s/%.9s/param.sfo", metas[k], e->d_name);
            FILE* f = fopen(p, "rb");
            if (!f) continue;
            unsigned char buf[8192];
            size_t len = fread(buf, 1, sizeof buf, f);
            fclose(f);
            inst_sfo_title(buf, len, g->title, sizeof g->title);
            snprintf(g->meta, sizeof g->meta, "%.110s/%.9s", metas[k], e->d_name);
            break;
        }
        if (!g->meta[0]) {   // sin appmeta: el nombre sale del param.sfo dentro del app.pkg
            char p[256];
            size_t len = 0;
            snprintf(p, sizeof p, "%.200s/app.pkg", g->app);
            unsigned char* sfo = inst_pkg_entry(p, PKG_ENTRY_PARAM_SFO, &len);
            if (sfo) inst_sfo_title(sfo, len, g->title, sizeof g->title);
            free(sfo);
        }
    }
    closedir(d);
    qsort(games, n, sizeof *games, cmp_game);
    return n;
}

static uint32_t be32(const unsigned char* p) { return (uint32_t)p[0] << 24 | p[1] << 16 | p[2] << 8 | p[3]; }

unsigned char* inst_pkg_entry(const char* pkg_path, unsigned id, size_t* len)
{
    FILE* f = fopen(pkg_path, "rb");
    if (!f) return NULL;
    unsigned char hdr[0x20], ent[32];
    unsigned char* out = NULL;
    if (fread(hdr, 1, sizeof hdr, f) != sizeof hdr || be32(hdr) != 0x7F434E54u) goto done;   // "\x7FCNT"
    uint32_t count = be32(hdr + 0x10), table = be32(hdr + 0x18);
    if (count == 0 || count > 4096) goto done;
    for (uint32_t i = 0; i < count; ++i) {
        if (fseek(f, (long)table + (long)i * 32, SEEK_SET) != 0 || fread(ent, 1, 32, f) != 32) break;
        if (be32(ent) != id) continue;
        uint32_t flags = be32(ent + 8), off = be32(ent + 16), size = be32(ent + 20);
        if ((flags & 0x80000000u) || size == 0 || size > (8u << 20)) break;   // cifrada o absurda
        out = malloc(size);
        if (out && (fseek(f, (long)off, SEEK_SET) != 0 || fread(out, 1, size, f) != size)) { free(out); out = NULL; }
        if (out) *len = size;
        break;
    }
done:
    fclose(f);
    return out;
}

void inst_load_game_cfg(inst_game_t* g, const nx_config_t* defaults, const char* cfg_dir)
{
    char p[160];
    snprintf(p, sizeof p, "%s/%s.ini", cfg_dir, g->id);
    g->cfg = *defaults;
    g->custom = g->custom_saved = g->cfg_dirty = 0;
    FILE* f = fopen(p, "rb");
    if (!f) return;
    static char buf[4096];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    nx_config_parse(&g->cfg, buf);
    g->custom = g->custom_saved = 1;
}

// devuelve el inicio de la seccion [s] (la linea del encabezado) o NULL
static const char* find_section(const char* ini, const char* s)
{
    char hdr[32];
    snprintf(hdr, sizeof hdr, "[%s]", s);
    size_t hl = strlen(hdr);
    for (const char* p = ini; p && *p;) {
        const char* q = p;
        while (*q == ' ' || *q == '\t') ++q;
        if (!strncmp(q, hdr, hl)) return p;
        p = strchr(p, '\n');
        if (p) ++p;
    }
    return NULL;
}

static const char* section_end(const char* sec)
{
    const char* p = strchr(sec, '\n');
    while (p) {
        const char* q = p + 1;
        while (*q == ' ' || *q == '\t') ++q;
        if (*q == '[') return p + 1;
        p = strchr(p + 1, '\n');
    }
    return sec + strlen(sec);
}

int inst_ini_has(const char* ini, const char* section)
{
    const char* s = find_section(ini, section);
    if (!s) return 0;
    const char* end = section_end(s);
    const char* p = strstr(s, PLUGIN_LINE);
    return p && p < end;
}

char* inst_ini_add(const char* ini, const char* section)
{
    if (inst_ini_has(ini, section)) return NULL;
    size_t len = strlen(ini);
    char* out = malloc(len + 128);
    const char* s = find_section(ini, section);
    if (!s) {
        sprintf(out, "%s%s\n[%s]\n%s\n", ini, (len && ini[len - 1] != '\n') ? "\n" : "", section, PLUGIN_LINE);
    } else {
        const char* eol = strchr(s, '\n');
        size_t at = eol ? (size_t)(eol - ini) + 1 : len;
        memcpy(out, ini, at);
        int k = sprintf(out + at, "%s%s\n", eol ? "" : "\n", PLUGIN_LINE);
        strcpy(out + at + k, ini + at);
    }
    return out;
}

// quita las lineas del plugin entre [from, to) del texto
static char* remove_lines(const char* ini, const char* from, const char* to)
{
    size_t len = strlen(ini);
    char* out = malloc(len + 1);
    size_t o = 0;
    for (const char* p = ini; *p;) {
        const char* eol = strchr(p, '\n');
        size_t n = eol ? (size_t)(eol - p) + 1 : strlen(p);
        int drop = p >= from && p < to && strstr(p, PLUGIN_LINE) && strstr(p, PLUGIN_LINE) < p + n;
        if (!drop) { memcpy(out + o, p, n); o += n; }
        p += n;
    }
    out[o] = 0;
    return out;
}

char* inst_ini_remove(const char* ini, const char* section)
{
    if (!inst_ini_has(ini, section)) return NULL;
    const char* s = find_section(ini, section);
    return remove_lines(ini, s, section_end(s));
}

char* inst_ini_remove_all(const char* ini)
{
    if (!strstr(ini, PLUGIN_LINE)) return NULL;
    return remove_lines(ini, ini, ini + strlen(ini));
}

// ultima linea de `log` que contiene `key` (NULL si no hay); la copia en out
static const char* last_line_with(const char* log, const char* key, char* out, size_t n)
{
    const char* hit = NULL;
    for (const char* p = strstr(log, key); p; p = strstr(p + 1, key)) hit = p;
    if (!hit) return NULL;
    const char* b = hit;
    while (b > log && b[-1] != '\n') --b;
    const char* e = strchr(hit, '\n');
    size_t len = e ? (size_t)(e - b) : strlen(b);
    if (len >= n) len = n - 1;
    memcpy(out, b, len);
    out[len] = 0;
    return out;
}

static unsigned long long field_u(const char* line, const char* key)
{
    const char* p = strstr(line, key);
    return p ? strtoull(p + strlen(key), NULL, 0) : 0;
}

int inst_log_summary(const char* log, char* out, size_t n)
{
    char l[400];
    if (!log || !*log) {
        snprintf(out, n, "sin registro: el plugin no carg\xf3 en este juego (revis\xe1 GoldHEN y plugins.ini)");
        return 0;
    }
    if (last_line_with(log, "[flip] estado=", l, sizeof l)) {
        int st = (int)field_u(l, "estado=");
        unsigned long long ours = field_u(l, "nuestros="), flips = 0;
        const char* q = strchr(l, '[');
        if (q) flips = strtoull(q + 1, NULL, 10);
        if (st == 3 && ours) { snprintf(out, n, "activo: %llu de %llu fotogramas procesados", ours, flips); return 3; }
        if (st == 3) { snprintf(out, n, "carg\xf3 y est\xe1 listo, sin efectos aplicados todav\xed" "a"); return 2; }
        if (st == 4) { snprintf(out, n, "carg\xf3 pero qued\xf3 en pausa (GPU saturada o cambio de pantalla)"); return 2; }
    }
    if (strstr(log, "ningun RegisterBuffers")) {
        snprintf(out, n, "carg\xf3 tarde: el juego registr\xf3 su pantalla antes que el plugin");
        return 1;
    }
    if (strstr(log, "sin ningun flip")) {
        snprintf(out, n, "carg\xf3, pero el juego presenta la imagen por otra v\xed" "a (a\xfa" "n no compatible)");
        return 1;
    }
    if (last_line_with(log, "no soportado", l, sizeof l) && !strstr(log, "buffers del juego:")) {
        const char* f = strstr(l, "0x");
        snprintf(out, n, "carg\xf3, pero el formato de video %.10s no es compatible", f ? f : "?");
        return 1;
    }
    if (strstr(log, "buffers del juego:")) { snprintf(out, n, "carg\xf3 y vio la pantalla del juego (arrancando)"); return 2; }
    snprintf(out, n, "carg\xf3, esperando que el juego registre su pantalla");
    return 1;
}

// plugins listados en la seccion [section]: 1 si esta el nuestro; en `others`
// los nombres de los demas .prx (separados por ", ")
int inst_ini_section_plugins(const char* ini, const char* section, char* others, size_t n)
{
    if (n) others[0] = 0;
    const char* s = find_section(ini, section);
    if (!s) return 0;
    const char* end = section_end(s);
    int ours = 0;
    size_t o = 0;
    const char* p = strchr(s, '\n');
    while (p && p < end) {
        ++p;
        const char* eol = strchr(p, '\n');
        size_t len = eol ? (size_t)(eol - p) : strlen(p);
        if (p + len > end) len = (size_t)(end - p);
        while (len && (p[len - 1] == '\r' || p[len - 1] == ' ')) --len;
        const char* q = p;
        while (q < p + len && (*q == ' ' || *q == '\t')) ++q;
        if (q < p + len && *q == '/') {
            if ((size_t)(p + len - q) == strlen(PLUGIN_LINE) && !strncmp(q, PLUGIN_LINE, strlen(PLUGIN_LINE))) {
                ours = 1;
            } else {
                const char* base = q;
                for (const char* t = q; t < p + len; ++t) if (*t == '/') base = t + 1;
                int bl = (int)(p + len - base);
                int k = snprintf(others + o, n > o ? n - o : 0, "%s%.*s", o ? ", " : "", bl, base);
                if (k > 0) o += (size_t)k;
                if (o >= n) o = n ? n - 1 : 0;
            }
        }
        p = eol;
    }
    return ours;
}
