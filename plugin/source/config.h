#pragma once

typedef struct {
    int effects;           // interruptor general: 0 = todos los efectos apagados (sin perder los ajustes)
    int framegen;          // 1 = generar fotogramas intermedios (30 -> 60)
    int sr;                // reescalado IA NXSR: 0 off, 1 suave, 2 normal, 3 fuerte
    int sr_model;          // modelo de reescalado: 0 NXSR-L (ligero), 1 Rapido, 2 Calidad
    int dynamic;           // 1 = calidad dinamica: baja escalones si la GPU no da abasto
    int eco;               // 1 = modo ahorro: modelo Rapido, sin SMAA y limite de temperatura mas bajo
    int temp_limit;        // grados C: por encima baja la calidad (0 = sin control termico)
    int aa;                // 1 = SMAA 1x (antialiasing de bordes)
    int rcas;              // nitidez FSR1 RCAS: 0 off, 1 baja, 2 media, 3 alta (sin efecto con NXSR activo)
    int fps_counter;       // 1 = contador de FPS en pantalla
    int compare;           // 1 = mitad izquierda sin procesar (desactiva FG mientras esta activo)
    int out_w, out_h;      // salida de EASU cuando el juego registra buffers mas chicos
    int max_game_fps;      // si el juego ya corre por encima, FG se apaga solo
    int menu_hold_ms;      // L3+R3 mantenido este tiempo abre/cierra el menu
    int warmup_flips;      // fotogramas del juego antes de reservar memoria (no molestar el arranque)
    int reserve_mb;        // memoria directa que siempre se deja libre para el juego
} nx_config_t;

typedef int (*nx_read_fn)(const char* path, char* buf, int cap);
typedef int (*nx_write_fn)(const char* path, const char* data, int len);

void nx_config_defaults(nx_config_t* c);
// aplica un texto ini ("clave = valor" por linea) sobre c
void nx_config_parse(nx_config_t* c, const char* text);
// lee /data/vcfg_nx/config.ini y luego /data/vcfg_nx/<TITLE_ID>.ini (si existe)
void nx_config_load(nx_config_t* c, const char* title_id, nx_read_fn rd);
// texto ini con todos los ajustes que el usuario puede cambiar (menu y app)
int nx_config_format(const nx_config_t* c, const char* comment, char* out, int cap);
// guarda esos ajustes en /data/vcfg_nx/<TITLE_ID>.ini
int nx_config_save(const nx_config_t* c, const char* title_id, nx_write_fn wr);
