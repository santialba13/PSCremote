// Despacho de un kernel de nx_shaders.h en un DCB (comun a pipeline y OSD).
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "../include/nx_shaders.h"
#include "pm4.h"

// zona de memoria CPU/GPU para kernargs y descriptores de un slot
typedef struct {
    uint8_t* cpu;
    uint64_t gpu;
    size_t cap, used;
} nx_ring_t;

const nx_shader_t* nx_find_shader(const char* name);
// copia n bytes a la zona (alineado a "align"); 0 si no hay lugar
uint64_t nx_ring_put(nx_ring_t* r, const void* data, size_t n, size_t align);
// emite SET_SH_REG + DISPATCH_DIRECT + barrera; tx, ty = hilos totales (grupos de 8x8).
// Devuelve 0, o < 0 si los argumentos no coinciden con lo que espera el kernel.
int nx_dispatch(pm4_t* c, nx_ring_t* ka, const nx_shader_t* s, uint64_t code_gpu,
                const void* args, size_t args_size, int tx, int ty);
