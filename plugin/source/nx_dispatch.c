#define NX_SHADERS_IMPL
#include "nx_dispatch.h"

#include <string.h>

const nx_shader_t* nx_find_shader(const char* name)
{
    for (int i = 0; i < NX_NUM_SHADERS; ++i)
        if (!strcmp(nx_shaders[i].name, name)) return &nx_shaders[i];
    return NULL;
}

uint64_t nx_ring_put(nx_ring_t* r, const void* data, size_t n, size_t align)
{
    size_t at = (r->used + align - 1) / align * align;
    if (at + n > r->cap) return 0;
    memcpy(r->cpu + at, data, n);
    r->used = at + n;
    return r->gpu + at;
}

int nx_dispatch(pm4_t* c, nx_ring_t* ka, const nx_shader_t* s, uint64_t code_gpu,
                const void* args, size_t args_size, int tx, int ty)
{
    // el struct C puede tener relleno al final (sizeof redondea a 8): se copian
    // exactamente los bytes que declara el kernel; si faltan, el layout no coincide
    if (args_size < s->kernarg_size || args_size > s->kernarg_size + 7) return -2;
    uint64_t kp = nx_ring_put(ka, args, s->kernarg_size, 64);
    if (!kp) return -1;
    uint32_t pg[2] = { (uint32_t)(code_gpu >> 8), (uint32_t)(code_gpu >> 40) };
    uint32_t rs[2] = { s->rsrc1, s->rsrc2 };
    uint32_t nt[3] = { 8, 8, 1 };
    // ABI HSA: s[0:3] = private segment buffer (sin scratch -> 0), s[4:5] = kernarg ptr
    uint32_t ud[6] = { 0, 0, 0, 0, (uint32_t)kp, (uint32_t)(kp >> 32) };
    pm4_set_sh(c, R_COMPUTE_PGM_LO, pg, 2);
    pm4_set_sh(c, R_COMPUTE_PGM_RSRC1, rs, 2);
    pm4_set_sh(c, R_COMPUTE_NUM_THREAD_X, nt, 3);
    pm4_set_sh(c, R_COMPUTE_USER_DATA_0, ud, s->user_sgprs < 6 ? s->user_sgprs : 6);
    pm4_dispatch(c, (uint32_t)(tx + 7) / 8, (uint32_t)(ty + 7) / 8, 1);
    pm4_cs_barrier(c);
    return 0;
}
