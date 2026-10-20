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
    return strcmp(((const inst_game_t*)a)->title, ((const inst_game_t*)b)->title);
}

int inst_list_games(const char* dir_app, const char* dir_meta, inst_game_t* games, int max)
{
    DIR* d = opendir(dir_app);
    if (!d) return 0;
    int n = 0;
    struct dirent* e;
    while ((e = readdir(d)) && n < max) {
        if (strncmp(e->d_name, "CUSA", 4) != 0 || strlen(e->d_name) != 9) continue;
        inst_game_t* g = &games[n++];
        memset(g, 0, sizeof *g);
        snprintf(g->id, sizeof g->id, "%s", e->d_name);
        snprintf(g->title, sizeof g->title, "%s", e->d_name);
        char p[256];
        snprintf(p, sizeof p, "%s/%s/param.sfo", dir_meta, e->d_name);
        FILE* f = fopen(p, "rb");
        if (f) {
            unsigned char buf[8192];
            size_t len = fread(buf, 1, sizeof buf, f);
            fclose(f);
            inst_sfo_title(buf, len, g->title, sizeof g->title);
        }
    }
    closedir(d);
    qsort(games, n, sizeof *games, cmp_game);
    return n;
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
