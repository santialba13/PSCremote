// Pipeline de GPU de VCFG-NX: arma el DCB que genera un fotograma intermedio
// (block-matching + NXFlow niveles A/B a 1/8 y C a 1/4 + warp) y opcionalmente
// lo pasa por FSR1.
// C portable: la parte especifica de PS4 (memoria, hooks, flips) esta en main.c.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "gcn_desc.h"
#include "pm4.h"

typedef enum {
    NX_POST_NONE = 0,   // el fotograma generado va directo al buffer de salida
    NX_POST_RCAS = 1,   // nitidez adaptativa (FSR1 RCAS) en reales y generados
    NX_POST_EASU = 2,   // FSR1 completo: EASU (escalado) + RCAS
} nx_post_t;

typedef struct {
    uint8_t* cpu;       // memoria compartida CPU/GPU (WC garlic)
    uint64_t gpu;
    size_t size, used;
} nx_arena_t;

typedef struct {
    int W, H;           // resolucion del juego (buffers que registra en VideoOut)
    int Wp, Hp;         // rellenada a multiplo de 32
    int w4, h4, w8, h8, w16, h16;
    int Wo, Ho;         // resolucion de salida (== W, H salvo con EASU)
    nx_arena_t arena;
    int shader[32];     // indice en nx_shaders[] por kernel usado
    uint64_t shader_gpu[32];
    uint64_t w_gpu[24], b_gpu[24];
    // tensores
    uint64_t luma[2][3], a8[2], a4[2];
    uint64_t vec[3], vtmp, flow5, delta5, flow4, delta4, lvl_in, hA, hB, hR, up;
    uint64_t tmp_img, tmp_img2;     // imagenes intermedias para el post-proceso
    uint64_t kernarg_base[2], desc_base[2];
    uint64_t label_gpu;
    volatile uint32_t* label_cpu;
    int luma_valid[2];
} nx_pipeline_t;

typedef struct {
    const nx_image_t* prev;     // fotograma N-1 del juego
    const nx_image_t* cur;      // fotograma N del juego
    const nx_image_t* out_mid;  // destino del fotograma generado (buffer del plugin)
    const nx_image_t* out_real; // destino del real post-procesado (NULL si post = NONE)
    int slot;                   // 0/1: alterna por fotograma (doble buffer de kernargs)
    nx_post_t post;
    float sharpness_stops;      // 0 = maxima nitidez, 2 = suave (como FSR)
    uint32_t seq;               // valor que el GPU escribe en label al terminar
} nx_job_t;

size_t nx_pipeline_mem_size(int W, int H, int Wo, int Ho);
// arena: memoria GPU ya mapeada de al menos nx_pipeline_mem_size() bytes
int nx_pipeline_init(nx_pipeline_t* p, int W, int H, int Wo, int Ho, nx_arena_t arena);
// escribe el DCB del trabajo en "c"; devuelve 0 si todo bien
int nx_pipeline_build(nx_pipeline_t* p, pm4_t* c, const nx_job_t* job);
