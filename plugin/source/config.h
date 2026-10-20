#pragma once

typedef struct {
    int framegen;          // 1 = generar fotogramas intermedios (30 -> 60)
    int post;              // 0 = nada, 1 = RCAS, 2 = FSR1 (EASU + RCAS)
    int aa;                // 1 = SMAA 1x (antialiasing de bordes) en reales y generados
    float sharpness;       // stops de FSR: 0 = maxima nitidez, 2 = suave
    int out_w, out_h;      // salida de EASU cuando el juego registra buffers mas chicos
    int max_game_fps;      // si el juego ya corre por encima, FG se desactiva solo
    int toggle_hold_ms;    // L3+R3 mantenido este tiempo -> activar/desactivar
} nx_config_t;

void nx_config_defaults(nx_config_t* c);
// lee /data/vcfg_nx/config.ini y luego /data/vcfg_nx/<TITLE_ID>.ini (si existe)
void nx_config_load(nx_config_t* c, const char* title_id);
