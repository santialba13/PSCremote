#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONFIG_DIR "/data/vcfg_nx"

void nx_config_defaults(nx_config_t* c)
{
    c->effects = 1;
    c->framegen = 1;
    c->sr = 2;
    c->sr_model = 0;
    c->dynamic = 1;
    c->eco = 0;
    c->temp_limit = 80;
    c->aa = 1;
    c->rcas = 1;          // FSR+ Baja por defecto: el realce de NXSR-L + un toque de nitidez se nota a 1080p
    c->fps_counter = 0;
    c->compare = 0;
    c->out_w = 1920;
    c->out_h = 1080;
    c->max_game_fps = 48;
    c->menu_hold_ms = 1500;
    c->warmup_flips = 1200;    // ~40 s a 30 fps: el arranque ya quedo confirmado (900) antes de usar la GPU
    c->reserve_mb = 192;
}

static int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

void nx_config_parse(nx_config_t* c, const char* text)
{
    char line[160];
    while (text && *text) {
        int n = 0;
        while (*text && *text != '\n' && n < (int)sizeof line - 1) line[n++] = *text++;
        while (*text && *text != '\n') ++text;
        if (*text == '\n') ++text;
        line[n] = 0;
        char* s = line;
        while (*s == ' ' || *s == '\t') ++s;
        if (!*s || *s == ';' || *s == '#' || *s == '[') continue;
        char* eq = strchr(s, '=');
        if (!eq) continue;
        *eq = 0;
        char* k = s;
        char* v = eq + 1;
        char* e = eq;
        while (e > k && (e[-1] == ' ' || e[-1] == '\t')) *--e = 0;
        while (*v == ' ' || *v == '\t') ++v;
        int iv = atoi(v);
        if (!strcmp(k, "effects")) c->effects = iv != 0;
        else if (!strcmp(k, "framegen")) c->framegen = clampi(iv, 0, 2);   // 0 no, 1 rapida, 2 IA
        else if (!strcmp(k, "sr")) c->sr = clampi(iv, 0, 3);
        else if (!strcmp(k, "sr_model")) c->sr_model = clampi(iv, 0, 2);
        else if (!strcmp(k, "dynamic")) c->dynamic = iv != 0;
        else if (!strcmp(k, "eco")) c->eco = iv != 0;
        else if (!strcmp(k, "temp_limit")) c->temp_limit = iv == 0 ? 0 : clampi(iv, 60, 95);
        else if (!strcmp(k, "aa")) c->aa = iv != 0;
        else if (!strcmp(k, "rcas")) c->rcas = clampi(iv, 0, 3);
        else if (!strcmp(k, "fps_counter")) c->fps_counter = iv != 0;
        else if (!strcmp(k, "compare")) c->compare = iv != 0;
        else if (!strcmp(k, "out_w")) c->out_w = clampi(iv, 640, 3840);
        else if (!strcmp(k, "out_h")) c->out_h = clampi(iv, 360, 2160);
        else if (!strcmp(k, "max_game_fps")) c->max_game_fps = clampi(iv, 20, 120);
        else if (!strcmp(k, "menu_hold_ms")) c->menu_hold_ms = clampi(iv, 300, 10000);
        else if (!strcmp(k, "warmup_flips")) c->warmup_flips = clampi(iv, 60, 100000);
        else if (!strcmp(k, "reserve_mb")) c->reserve_mb = clampi(iv, 64, 4096);
    }
}

void nx_config_load(nx_config_t* c, const char* title_id, nx_read_fn rd)
{
    static char buf[4096];
    char path[96];
    nx_config_defaults(c);
    if (!rd) return;
    int n = rd(CONFIG_DIR "/config.ini", buf, sizeof buf - 1);
    if (n > 0) { buf[n] = 0; nx_config_parse(c, buf); }
    snprintf(path, sizeof path, CONFIG_DIR "/%s.ini", title_id);
    n = rd(path, buf, sizeof buf - 1);
    if (n > 0) { buf[n] = 0; nx_config_parse(c, buf); }
}

int nx_config_format(const nx_config_t* c, const char* comment, char* out, int cap)
{
    return snprintf(out, (size_t)cap,
                    "; %s\n"
                    "effects = %d\nframegen = %d\nsr = %d\nsr_model = %d\naa = %d\nrcas = %d\ndynamic = %d\neco = %d\n"
                    "temp_limit = %d\nfps_counter = %d\ncompare = %d\n",
                    comment, c->effects, c->framegen, c->sr, c->sr_model, c->aa, c->rcas, c->dynamic, c->eco, c->temp_limit,
                    c->fps_counter, c->compare);
}

int nx_config_save(const nx_config_t* c, const char* title_id, nx_write_fn wr)
{
    char path[96], text[512];
    snprintf(path, sizeof path, CONFIG_DIR "/%s.ini", title_id);
    int n = nx_config_format(c, "guardado desde el menu de VCFG-NX", text, sizeof text);
    return wr ? wr(path, text, n) : -1;
}
