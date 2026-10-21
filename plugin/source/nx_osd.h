// Menu en pantalla / contador de FPS / modo comparar. Contexto liviano (pocos
// KiB): funciona aunque no se haya podido reservar la memoria de los efectos,
// para que el menu siempre este disponible (modo seguro, poca memoria).
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "gcn_desc.h"
#include "nx_dispatch.h"
#include "pm4.h"

typedef struct {
    uint64_t text_gpu, attr_gpu, font_gpu;   // grilla Latin-1, color por celda, atlas (nx_osd_font.h)
    int cols, rows, x0, y0, scale, sel_row;
} nx_osd_t;

// tamano del panel en pixeles (para centrarlo)
void nx_osd_panel_size(int cols, int rows, int scale, int* w, int* h);

#define NX_OSD_RINGS 10

typedef struct {
    const nx_shader_t* sh_draw;
    const nx_shader_t* sh_copy;
    const nx_shader_t* sh_fill;
    uint64_t code_draw, code_copy, code_fill;
    nx_ring_t ring[NX_OSD_RINGS];   // kernargs + descriptores: 0-1 motor, 2-5 generacion VCFG (nx_fg8), 6-9 su post-proceso
} nx_osd_ctx_t;

size_t nx_osd_mem_size(void);
// mem: memoria CPU/GPU de al menos nx_osd_mem_size() bytes, alineada a 256
int nx_osd_init(nx_osd_ctx_t* o, uint8_t* cpu, uint64_t gpu);
void nx_osd_begin(nx_osd_ctx_t* o, int slot);   // vacia la zona de kernargs del slot
int nx_osd_draw(nx_osd_ctx_t* o, pm4_t* c, int slot, const nx_image_t* img, const nx_osd_t* const* osds, int n);
// copia la mitad izquierda de src en dst (modo comparar)
// copia el rectangulo (x0, y0, w, h) de src a la misma posicion en dst
int nx_osd_copy_rect(nx_osd_ctx_t* o, pm4_t* c, int slot, const nx_image_t* src, const nx_image_t* dst, int x0, int y0,
                     int w, int h);
// copia src entero sobre dst (mismo tamano o el menor)
int nx_osd_copy(nx_osd_ctx_t* o, pm4_t* c, int slot, const nx_image_t* src, const nx_image_t* dst);
int nx_osd_copy_left(nx_osd_ctx_t* o, pm4_t* c, int slot, const nx_image_t* src, const nx_image_t* dst);
// rellena (x0, y0, w, h) de dst con un color (linea divisoria del modo comparar)
int nx_osd_fill_rect(nx_osd_ctx_t* o, pm4_t* c, int slot, const nx_image_t* dst, int x0, int y0, int w, int h,
                     float r, float g, float b);
