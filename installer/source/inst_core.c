#include "inst_core.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
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

// quita las lineas de VCFG-NX (plugin y sonda) entre [from, to) del texto
static int our_line(const char* p, size_t n)
{
    const char* a = strstr(p, PLUGIN_LINE);
    const char* b = strstr(p, PROBE_LINE);
    return (a && a < p + n) || (b && b < p + n);
}

static char* remove_lines(const char* ini, const char* from, const char* to)
{
    size_t len = strlen(ini);
    char* out = malloc(len + 1);
    size_t o = 0;
    for (const char* p = ini; *p;) {
        const char* eol = strchr(p, '\n');
        size_t n = eol ? (size_t)(eol - p) + 1 : strlen(p);
        int drop = p >= from && p < to && our_line(p, n);
        if (!drop) { memcpy(out + o, p, n); o += n; }
        p += n;
    }
    out[o] = 0;
    return out;
}

// deja en [section] la sonda y el plugin (en ese orden) justo debajo del
// encabezado. NULL si ya estaba exactamente asi
char* inst_ini_add(const char* ini, const char* section)
{
    const char* s = find_section(ini, section);
    char* base;
    if (s) {
        base = remove_lines(ini, s, section_end(s));
    } else {
        size_t len = strlen(ini);
        base = malloc(len + 64);
        sprintf(base, "%s%s[%s]\n", ini, (len && ini[len - 1] != '\n') ? "\n" : "", section);
    }
    s = find_section(base, section);
    size_t len = strlen(base);
    char* out = malloc(len + 128);
    const char* eol = strchr(s, '\n');
    size_t at = eol ? (size_t)(eol - base) + 1 : len;
    memcpy(out, base, at);
    int k = sprintf(out + at, "%s%s\n%s\n", eol ? "" : "\n", PROBE_LINE, PLUGIN_LINE);
    strcpy(out + at + k, base + at);
    free(base);
    if (!strcmp(out, ini)) { free(out); return NULL; }
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
    if (!strstr(ini, PLUGIN_LINE) && !strstr(ini, PROBE_LINE)) return NULL;
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
            } else if ((size_t)(p + len - q) == strlen(PROBE_LINE) && !strncmp(q, PROBE_LINE, strlen(PROBE_LINE))) {
                // la sonda es nuestra
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

// borra una carpeta con todo lo que tenga adentro; devuelve cuantas cosas borro
int inst_rm_tree(const char* path)
{
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    if (!S_ISDIR(st.st_mode)) return remove(path) == 0;
    int n = 0;
    DIR* d = opendir(path);
    if (d) {
        struct dirent* e;
        char p[512];
        while ((e = readdir(d))) {
            if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
            snprintf(p, sizeof p, "%s/%s", path, e->d_name);
            n += inst_rm_tree(p);
        }
        closedir(d);
    }
    return n + (rmdir(path) == 0);
}

// borra los archivos de `dir` cuyo nombre empieza con `prefix` o (si no es
// NULL) termina en `suffix`; devuelve cuantos borro
int inst_rm_matching(const char* dir, const char* prefix, const char* suffix)
{
    DIR* d = opendir(dir);
    if (!d) return 0;
    int n = 0;
    struct dirent* e;
    char p[512];
    while ((e = readdir(d))) {
        const char* nm = e->d_name;
        size_t l = strlen(nm), sl = suffix ? strlen(suffix) : 0;
        int hit = (prefix && !strncmp(nm, prefix, strlen(prefix))) || (suffix && l >= sl && !strcmp(nm + l - sl, suffix));
        if (!hit || !strcmp(nm, ".") || !strcmp(nm, "..")) continue;
        snprintf(p, sizeof p, "%s/%s", dir, nm);
        n += inst_rm_tree(p);
    }
    closedir(d);
    return n;
}

// plugins que tocan lo mismo que VCFG-NX (presentacion de fotogramas, escalado,
// framerate) o versiones viejas de VCFG: en un juego con VCFG-NX se quitan
static const char* const CONFLICTS[] = {
    "gnmscale", "vcfg", "bb_probe", "fliprate_remover", "force_30_fps", "force_1080p_display", "frame_logger",
    "fsr", "upscal", "framegen", "frame_gen", "60fps", NULL,
};

int inst_is_conflict(const char* name)
{
    if (!strcmp(name, "vcfg_nx.prx") || !strcmp(name, "vcfg_nx_probe.prx")) return 0;
    char low[128];
    size_t i = 0;
    for (; name[i] && i < sizeof low - 1; ++i) low[i] = (char)(name[i] >= 'A' && name[i] <= 'Z' ? name[i] + 32 : name[i]);
    low[i] = 0;
    for (int k = 0; CONFLICTS[k]; ++k)
        if (strstr(low, CONFLICTS[k])) return 1;
    return 0;
}

char* inst_ini_strip_conflicts(const char* ini, const char* section, char* removed, size_t n)
{
    const char* s = find_section(ini, section);
    if (!s) return NULL;
    const char* end = section_end(s);
    size_t len = strlen(ini), o = 0, ro = removed && n ? strlen(removed) : 0;
    char* out = malloc(len + 1);
    int changed = 0;
    for (const char* p = ini; *p;) {
        const char* eol = strchr(p, '\n');
        size_t l = eol ? (size_t)(eol - p) + 1 : strlen(p);
        int drop = 0;
        if (p > s && p < end) {
            const char* q = p;
            while (q < p + l && (*q == ' ' || *q == '\t')) ++q;
            if (*q == '/') {
                const char* base = q, *e = p + l;
                while (e > q && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' ')) --e;
                for (const char* t = q; t < e; ++t) if (*t == '/') base = t + 1;
                char nm[128];
                snprintf(nm, sizeof nm, "%.*s", (int)(e - base), base);
                if (inst_is_conflict(nm)) {
                    drop = 1;
                    if (removed && n > ro + 1) {
                        int k = snprintf(removed + ro, n - ro, "%s%s", ro ? ", " : "", nm);
                        if (k > 0) ro += (size_t)k;
                        if (ro >= n) ro = n - 1;
                    }
                }
            }
        }
        if (!drop) { memcpy(out + o, p, l); o += l; } else changed = 1;
        p += l;
    }
    out[o] = 0;
    if (!changed) { free(out); return NULL; }
    return out;
}
