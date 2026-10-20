#include "pm4.h"

void pm4_init(pm4_t* c, uint32_t* buf, size_t cap_dwords)
{
    c->buf = buf;
    c->cap = cap_dwords;
    c->n = 0;
    c->overflow = 0;
}

static void emit(pm4_t* c, uint32_t v)
{
    if (c->n >= c->cap) {
        c->overflow = 1;
        return;
    }
    c->buf[c->n++] = v;
}

void pm4_set_sh(pm4_t* c, uint32_t reg, const uint32_t* vals, unsigned count)
{
    emit(c, PM4_TYPE3(PM4_SET_SH_REG, count + 1) | PM4_SHADER_COMPUTE);
    emit(c, (reg - SH_REG_BASE) >> 2);
    for (unsigned i = 0; i < count; ++i)
        emit(c, vals[i]);
}

void pm4_set_sh1(pm4_t* c, uint32_t reg, uint32_t val)
{
    pm4_set_sh(c, reg, &val, 1);
}

void pm4_dispatch(pm4_t* c, uint32_t gx, uint32_t gy, uint32_t gz)
{
    emit(c, PM4_TYPE3(PM4_DISPATCH_DIRECT, 4) | PM4_SHADER_COMPUTE);
    emit(c, gx);
    emit(c, gy);
    emit(c, gz);
    emit(c, 1); // COMPUTE_DISPATCH_INITIATOR.COMPUTE_SHADER_EN
}

void pm4_cs_barrier(pm4_t* c)
{
    emit(c, PM4_TYPE3(PM4_EVENT_WRITE, 1));
    emit(c, EV_CS_PARTIAL_FLUSH | (4u << 8));   // EVENT_INDEX = 4
    // SURFACE_SYNC: CP_COHER_CNTL, CP_COHER_SIZE (todo), CP_COHER_BASE (0), POLL_INTERVAL
    emit(c, PM4_TYPE3(PM4_SURFACE_SYNC, 4));
    emit(c, COHER_TCL1_ACTION_ENA | COHER_SH_KCACHE_ACTION);
    emit(c, 0xFFFFFFFFu);
    emit(c, 0);
    emit(c, 10);
}

void pm4_end_of_job(pm4_t* c, uint64_t label_gpu_addr, uint32_t value)
{
    // EVENT_WRITE_EOP con flush+inv de caches (L2 incluido) y escritura de 32 bits
    emit(c, PM4_TYPE3(PM4_EVENT_WRITE_EOP, 5));
    emit(c, EV_CACHE_FLUSH_AND_INV_TS | (5u << 8));          // EVENT_INDEX = 5 (TS)
    emit(c, (uint32_t)(label_gpu_addr & 0xFFFFFFFCu));
    emit(c, (uint32_t)((label_gpu_addr >> 32) & 0xFFFF) | (1u << 29)); // DATA_SEL = 1 (32 bits)
    emit(c, value);
    emit(c, 0);
}

void pm4_nop_pad(pm4_t* c, unsigned align_dwords)
{
    while (c->n % align_dwords) {
        unsigned left = align_dwords - (unsigned)(c->n % align_dwords);
        if (left == 1) {
            emit(c, 0x80000000u); // paquete tipo 2 (relleno de 1 dword)
        } else {
            emit(c, PM4_TYPE3(PM4_NOP, left - 1));
            for (unsigned i = 0; i < left - 1; ++i)
                emit(c, 0);
        }
    }
}
