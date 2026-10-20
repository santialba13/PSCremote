// Constructor minimo de paquetes PM4 para despachar compute shaders desde un
// DCB (draw command buffer) del anillo grafico de la PS4 (GCN 1.1 / CIK).
//
// Registros y opcodes segun la documentacion publica de AMD "Sea Islands"
// y src/amd/common/sid.h de Mesa (radeonsi). Este archivo es C portable para
// poder testearlo en el PC (plugin/test/host_test.c).
#pragma once
#include <stdint.h>
#include <stddef.h>

// --- opcodes PM4 tipo 3
#define PM4_NOP              0x10
#define PM4_DISPATCH_DIRECT  0x15
#define PM4_WRITE_DATA       0x37
#define PM4_SURFACE_SYNC     0x43
#define PM4_EVENT_WRITE      0x46
#define PM4_EVENT_WRITE_EOP  0x47
#define PM4_SET_SH_REG       0x76

#define PM4_TYPE3(op, ndw_body) \
    ((3u << 30) | ((((uint32_t)(ndw_body) - 1) & 0x3FFF) << 16) | (((op) & 0xFF) << 8))
#define PM4_SHADER_COMPUTE   (1u << 1)

// --- registros SH de compute (direccion de byte; SET_SH_REG usa (reg - 0xB000) / 4)
#define SH_REG_BASE                       0xB000
#define R_COMPUTE_DISPATCH_INITIATOR      0xB800
#define R_COMPUTE_NUM_THREAD_X            0xB81C
#define R_COMPUTE_PGM_LO                  0xB830
#define R_COMPUTE_PGM_RSRC1               0xB848
#define R_COMPUTE_RESOURCE_LIMITS         0xB854
#define R_COMPUTE_STATIC_THREAD_MGMT_SE0  0xB858
#define R_COMPUTE_TMPRING_SIZE            0xB860
#define R_COMPUTE_USER_DATA_0             0xB900

// --- eventos
#define EV_CS_PARTIAL_FLUSH     0x07
#define EV_CACHE_FLUSH_AND_INV_TS 0x14
#define EV_BOTTOM_OF_PIPE_TS    0x28

// --- CP_COHER_CNTL (SURFACE_SYNC)
#define COHER_TCL1_ACTION_ENA   (1u << 22)   // invalida L1 vectorial
#define COHER_TC_ACTION_ENA     (1u << 23)   // escribe/invalida L2
#define COHER_SH_KCACHE_ACTION  (1u << 27)   // invalida cache escalar (K$)
#define COHER_SH_ICACHE_ACTION  (1u << 29)   // invalida cache de instrucciones

typedef struct {
    uint32_t* buf;
    size_t cap;   // en dwords
    size_t n;     // dwords escritos
    int overflow;
} pm4_t;

void pm4_init(pm4_t* c, uint32_t* buf, size_t cap_dwords);
void pm4_set_sh(pm4_t* c, uint32_t reg, const uint32_t* vals, unsigned count);
void pm4_set_sh1(pm4_t* c, uint32_t reg, uint32_t val);
void pm4_dispatch(pm4_t* c, uint32_t gx, uint32_t gy, uint32_t gz);
// espera a que terminen los dispatches previos e invalida caches L1/K$
void pm4_cs_barrier(pm4_t* c);
// al final del DCB: vuelca L2 (para que VideoOut escanee la imagen) y escribe
// "value" en label_gpu_addr cuando el GPU termino (la CPU puede reusar buffers)
void pm4_end_of_job(pm4_t* c, uint64_t label_gpu_addr, uint32_t value);
void pm4_nop_pad(pm4_t* c, unsigned align_dwords);
