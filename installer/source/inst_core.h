// Logica del instalador sin dependencias de la PS4 (se testea en el PC):
// lista de juegos, titulo desde param.sfo y edicion segura de plugins.ini.
#pragma once
#include <stddef.h>

#include "../../plugin/source/config.h"

#define INST_MAX_GAMES 256
#define PLUGIN_LINE "/data/GoldHEN/plugins/vcfg_nx.prx"

typedef struct {
    char id[16];
    char title[64];
    char meta[128];   // carpeta con su param.sfo e icon0.png ("" si no se encontro)
    char app[128];    // carpeta del juego instalado (<...>/user/app/<id>, con su app.pkg)
    int enabled;      // el plugin figura en la seccion [id] de plugins.ini
    int custom;       // tiene ajustes propios (/data/vcfg_nx/<id>.ini)
    int custom_saved; // ...y el archivo existe en disco
    int cfg_dirty;    // hay que reescribir (o borrar) su .ini al guardar
    nx_config_t cfg;  // ajustes propios (si custom)
} inst_game_t;

// titulo (clave TITLE) de un param.sfo en memoria; 0 si lo encontro
int inst_sfo_title(const unsigned char* sfo, size_t len, char* out, size_t cap);
// agrega los juegos de dir_app (/user/app/<CUSAxxxxx>) a games[0..n) sin repetir
// ids; el titulo sale del primer metas[k]/<id>/param.sfo que exista (metas
// termina en NULL). Devuelve el total, ordenado por titulo sin mayusculas.
int inst_list_games(const char* dir_app, const char* const* metas, inst_game_t* games, int n, int max);
// una entrada sin cifrar de un PKG de PS4 (tabla de entradas de la cabecera):
// 0x1000 = param.sfo, 0x1200 = icon0.png. Lee solo lo necesario aunque el PKG
// pese decenas de GB. Devuelve malloc (len en *len) o NULL.
#define PKG_ENTRY_PARAM_SFO 0x1000
#define PKG_ENTRY_ICON0 0x1200
unsigned char* inst_pkg_entry(const char* pkg_path, unsigned id, size_t* len);

// lee cfg_dir/<id>.ini sobre los ajustes por defecto: custom = existe
void inst_load_game_cfg(inst_game_t* g, const nx_config_t* defaults, const char* cfg_dir);

// plugins.ini en memoria (texto). Devuelven texto nuevo (malloc) o NULL si no hay cambios.
int inst_ini_has(const char* ini, const char* section);
char* inst_ini_add(const char* ini, const char* section);
char* inst_ini_remove(const char* ini, const char* section);
// quita la linea del plugin de todas las secciones
char* inst_ini_remove_all(const char* ini);

// resume /data/vcfg_nx/logs/<ID>.log (lo escribe el plugin dentro del juego) en
// una frase. Devuelve 0 = no cargo, 1 = cargo sin poder actuar, 2 = cargo, 3 = activo
int inst_log_summary(const char* log, char* out, size_t n);
