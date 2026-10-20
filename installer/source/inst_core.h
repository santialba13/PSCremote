// Logica del instalador sin dependencias de la PS4 (se testea en el PC):
// lista de juegos, titulo desde param.sfo y edicion segura de plugins.ini.
#pragma once
#include <stddef.h>

#define INST_MAX_GAMES 256
#define PLUGIN_LINE "/data/GoldHEN/plugins/vcfg_nx.prx"

typedef struct {
    char id[16];
    char title[64];
    int enabled;      // el plugin figura en la seccion [id] de plugins.ini
} inst_game_t;

// titulo (clave TITLE) de un param.sfo en memoria; 0 si lo encontro
int inst_sfo_title(const unsigned char* sfo, size_t len, char* out, size_t cap);
// lista /user/app/<CUSAxxxxx> (dir_app) y lee el titulo de dir_meta/<id>/param.sfo
int inst_list_games(const char* dir_app, const char* dir_meta, inst_game_t* games, int max);

// plugins.ini en memoria (texto). Devuelven texto nuevo (malloc) o NULL si no hay cambios.
int inst_ini_has(const char* ini, const char* section);
char* inst_ini_add(const char* ini, const char* section);
char* inst_ini_remove(const char* ini, const char* section);
// quita la linea del plugin de todas las secciones
char* inst_ini_remove_all(const char* ini);
