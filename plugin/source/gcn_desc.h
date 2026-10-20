// Descriptores de recursos GCN 1.1 (T# imagen, S# sampler) para los kernels.
// Campos segun SQ_IMG_RSRC_WORD0..7 / SQ_IMG_SAMP_WORD0..3 (sid.h de Mesa, GFX7).
#pragma once
#include <stdint.h>

// formato de los buffers de VideoOut que soporta el plugin
typedef enum {
    NX_FMT_RGBA8_UNORM,   // A8B8G8R8 / A8R8G8B8 (el orden de canales no importa: la
                          // red se entreno con permutacion de canales)
    NX_FMT_RGBA8_SRGB,
    NX_FMT_RGB10A2_UNORM, // A2R10G10B10 (HDR10 sin PQ: los juegos SDR no lo usan)
} nx_format_t;

// indices de tile mode del PS4 (GpuAddress::TileMode); "display 2D thin" es el
// habitual de los backbuffers. A VALIDAR en hardware antes de activar por defecto.
#define NX_TILE_DISPLAY_LINEAR_ALIGNED 8
#define NX_TILE_DISPLAY_2D_THIN        10

typedef struct {
    uint64_t gpu_addr;     // alineado a 256 B
    uint32_t width, height;
    uint32_t pitch;        // en pixeles
    nx_format_t format;
    uint32_t tile_index;
} nx_image_t;

void nx_make_tsharp(uint32_t out[8], const nx_image_t* img);
// sampler con clamp al borde, coordenadas en pixeles, filtro bilineal o nearest
void nx_make_ssharp(uint32_t out[4], int bilinear);
