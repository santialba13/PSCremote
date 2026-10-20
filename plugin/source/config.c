#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CONFIG_DIR "/data/vcfg_nx"

void nx_config_defaults(nx_config_t* c)
{
    c->framegen = 1;
    c->post = 1;
    c->aa = 1;
    c->sharpness = 0.25f;
    c->out_w = 1920;
    c->out_h = 1080;
    c->max_game_fps = 48;
    c->toggle_hold_ms = 1500;
}

static char* trim(char* s)
{
    while (*s == ' ' || *s == '\t') ++s;
    char* e = s + strlen(s);
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r' || e[-1] == '\n')) *--e = 0;
    return s;
}

static void parse(nx_config_t* c, const char* path)
{
    FILE* f = fopen(path, "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        char* s = trim(line);
        if (!*s || *s == ';' || *s == '#' || *s == '[') continue;
        char* eq = strchr(s, '=');
        if (!eq) continue;
        *eq = 0;
        char* k = trim(s);
        char* v = trim(eq + 1);
        if (!strcmp(k, "framegen")) c->framegen = atoi(v);
        else if (!strcmp(k, "post")) c->post = atoi(v);
        else if (!strcmp(k, "aa")) c->aa = atoi(v) != 0;
        else if (!strcmp(k, "sharpness")) c->sharpness = (float)atof(v);
        else if (!strcmp(k, "out_w")) c->out_w = atoi(v);
        else if (!strcmp(k, "out_h")) c->out_h = atoi(v);
        else if (!strcmp(k, "max_game_fps")) c->max_game_fps = atoi(v);
        else if (!strcmp(k, "toggle_hold_ms")) c->toggle_hold_ms = atoi(v);
    }
    fclose(f);
}

void nx_config_load(nx_config_t* c, const char* title_id)
{
    char path[128];
    nx_config_defaults(c);
    parse(c, CONFIG_DIR "/config.ini");
    snprintf(path, sizeof path, CONFIG_DIR "/%s.ini", title_id);
    parse(c, path);
    if (c->post < 0 || c->post > 2) c->post = 1;
    if (c->sharpness < 0) c->sharpness = 0;
    if (c->sharpness > 2) c->sharpness = 2;
}
