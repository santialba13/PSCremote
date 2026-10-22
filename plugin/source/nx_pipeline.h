// Pipeline de GPU de VCFG-NX: arma el DCB que genera un fotograma intermedio
// (NXFG-2: block-matching de 4 niveles + red a 1/16 y 1/8 + warp con protecciones)
// y el post-proceso (SMAA 1x, NXSR, FSR1 EASU/RCAS).
// C portable: la parte especifica de PS4 (memoria, hooks, flips) esta en main.c.
#pragma once

#define NX_PIPE_SLOTS 10  // kernargs por trabajo: motor 0-1, generacion con presentacion VCFG 0-3, su post-proceso 6-9
#include <stddef.h>
#include <stdint.h>

#include "gcn_desc.h"
#include "nx_osd.h"
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
    int lum_h[4], lum_w[4];   // luma L1..L4 (media resolucion y mitades)
    int bm_h[4], bm_w[4];     // mallas de bloques de 8 de cada nivel (L1 = 1/16 de la imagen)
    int Wo, Ho;         // resolucion de salida (== W, H salvo con EASU)
    nx_arena_t arena;
    int shader[48];     // indice en nx_shaders[] por kernel usado
    uint64_t shader_gpu[48];
    uint64_t w_gpu[24], b_gpu[24];
    // tensores
    uint64_t luma[2][4], a8[2], a16[2];
    uint64_t vec[4], vtmp, flowG, deltaG, flow8, delta8, lvl_in, hA, hB, hR, up, scene, smask, smask_e;
    uint64_t tmp_img, tmp_img2, tmp_aa;   // imagenes intermedias para el post-proceso
    uint64_t smaa_edges, smaa_weights, smaa_area, smaa_search, nxsr_w, nxsr_hq_w;
    uint64_t srl_w[2][8], srl_b[2][8], srl_feat, srl_h[3], srl_g;   // NXSR-L: [0] Ligero, [1] Calidad
    // NXRE (realismo): red a ~128 x 72 -> malla re_gh x re_gw x D x 12; re_state es la
    // malla promediada entre fotogramas (persiste entre trabajos)
    uint64_t re_w[8], re_b[8], re_lo, re_t[4], re_grid, re_state;
    int re_f, re_hl, re_wl, re_h[5], re_w_[5], re_gh, re_gw, re_valid;
    uint64_t kernarg_base[NX_PIPE_SLOTS], desc_base[NX_PIPE_SLOTS];   // por slot (piramides: por paridad)
    uint64_t label_gpu;
    volatile uint32_t* label_cpu;
    int luma_valid[2];          // 1 = piramide completa, 2 = solo luma (generacion rapida)
} nx_pipeline_t;

typedef struct {
    const nx_image_t* prev;     // fotograma N-1 del juego
    const nx_image_t* cur;      // fotograma N del juego
    const nx_image_t* out_mid;  // destino del fotograma generado (buffer del plugin)
    const nx_image_t* out_real; // destino del real post-procesado (NULL si post = NONE y aa = 0)
    int slot;                   // 0/1: alterna por fotograma (doble buffer de kernargs)
    nx_post_t post;
    int aa;                     // 1 = SMAA 1x antes del escalado / nitidez
    float sharpness_stops;      // 0 = maxima nitidez, 2 = suave (como FSR)
    float sharp_detail;         // FSR+: realce de detalle despues de RCAS (0 = RCAS puro)
    uint32_t seq;               // valor que el GPU escribe en label al terminar
    int fg;                     // 0 = solo post-proceso del fotograma real
    int fg_fast;                // 1 = sin la red: vectores del block-matching + warp (mas liviano)
    int sr;                     // NXSR (reemplaza a RCAS)
    float sr_k;                 // intensidad de NXSR (1 = la del entrenamiento; el limitador 3x3 evita halos)
    int sr_hq;                  // 1 = modelo Calidad (wg_nxsr_hq), 0 = Rapido
    int sr_lite;                // NXSR-L (guia a 1/8, se combina con la generacion): 1 = Ligero, 2 = Calidad
    int rcas;                   // nitidez RCAS
    float re;                   // NXRE (realismo por IA): intensidad 0..1 (0 = apagado)
    int enhance_first;          // FG sobre los reales ya procesados (misma resolucion)
    const nx_image_t* prev_real;  // real procesado del fotograma anterior
    int prev_real_valid;
    int compare;                // mitad izquierda sin procesar
    const nx_image_t* copy_back; // si no es NULL: out_real se copia aca al final (el buffer del juego)
    int no_end;                 // 1 = sin el fin de trabajo (lo agrega el que llama: generacion con presentacion VCFG)
    nx_osd_ctx_t* osd_ctx;      // menu / contador (NULL = nada)
    const nx_osd_t* osd[2];
} nx_job_t;

size_t nx_pipeline_mem_size(int W, int H, int Wo, int Ho);
// arena: memoria GPU ya mapeada de al menos nx_pipeline_mem_size() bytes
int nx_pipeline_init(nx_pipeline_t* p, int W, int H, int Wo, int Ho, nx_arena_t arena);
// solo OSD sobre job->cur (sin memoria de efectos: menu en modo seguro)
int nx_pipeline_build_osd_only(nx_osd_ctx_t* o, pm4_t* c, const nx_job_t* job, uint64_t label_gpu);
// escribe el DCB del trabajo en "c"; devuelve 0 si todo bien
int nx_pipeline_build(nx_pipeline_t* p, pm4_t* c, const nx_job_t* job);
