// Generacion de fotogramas del motor clasico de VCFG ("mode 8"), portada de
// github.com/Melnikoff07/vcfg (licencia MIT, ver plugin/vcfg8/LICENSE):
// classic/source/main.c (build_dcb_flow8), fg8_table.h y fg_res.h. Es la que
// lleva GTA V de 30 a 60 en una PS4 base: ~3-4 ms de GPU por fotograma.
//
// Piramide de luma fija (960x540 / 480x270 / 240x135, solo la del fotograma
// actual: la anterior queda por paridad), estimacion de movimiento de grueso a
// fino (me3 -> me2 -> gm8 -> fgs8 -> me1 -> mf8), fase y warp a resolucion
// completa sobre el buffer intermedio (mismo formato y tiling que los del juego).
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "gcn_desc.h"
#include "pm4.h"

#define NX_FG8_SLOTS 4
#define NX_FG8_SLAB 0x20000u     // codigo + tablas + labels + DCBs + colas (CPU escribe, GPU lee)
#define NX_FG8_WORK 0x190000u    // piramides y campos de movimiento (solo GPU)
#define NX_FG8_DCB_DWORDS 4096
#define NX_FG8_TAIL_DWORDS 64

typedef struct {
    uint8_t* slab;               // NX_FG8_SLAB bytes (la direccion de CPU es la que ve la GPU)
    uint8_t* work;               // NX_FG8_WORK bytes
    uint32_t sh_off[9];          // pyr8 me3 me2 gm8 fgs8 me1 mf8 phase8 warp8
    int ok;
} nx_fg8_t;

int nx_fg8_init(nx_fg8_t* f, uint8_t* slab, uint8_t* work);
uint32_t* nx_fg8_dcb(nx_fg8_t* f, int slot);             // DCB del slot (NX_FG8_DCB_DWORDS)
uint32_t* nx_fg8_tail(nx_fg8_t* f, int slot);            // cola de flip para el flip del real
volatile uint32_t* nx_fg8_label(nx_fg8_t* f, int slot);  // fin del trabajo del slot (= seq)
volatile uint64_t* nx_fg8_ts(nx_fg8_t* f, int slot);     // [inicio, fin] del reloj de la GPU del slot

typedef struct {
    const nx_image_t* prev;      // P: fotograma anterior del juego
    const nx_image_t* cur;       // C: fotograma actual del juego
    const nx_image_t* mid;       // M: buffer intermedio (destino del warp)
    uint32_t mid_index;          // indice de VideoOut de M (para WaitFlipDone)
    int32_t handle;
    uint32_t seq;                // contador de fotogramas generados (la paridad elige las piramides)
    uint32_t t_q16;              // fase del intermedio (0x8000 = mitad)
    int repeat;                  // 1 = sin fotograma anterior valido: el intermedio repite C
    int osd_x, osd_y, osd_w, osd_h;   // rectangulo del menu/contador (0 si no hay): el warp no lo deforma
    // espera en la GPU a que M no este en pantalla (sceGnmInsertWaitFlipDone, 7 dwords); NULL = no
    int (*wait_flip_done)(uint32_t* cmd, uint32_t ndw, int32_t handle, uint32_t index);
    uint64_t ts0;                // si no es 0: reloj de la GPU al empezar nuestros pases (tras el prologo)
} nx_fg8_job_t;

// arma los pases de VCFG en c (prologo de espera al juego incluido). No escribe el
// final del trabajo ni la cola: el que llama agrega el OSD y despues
// nx_fg8_end(). Devuelve 0 o < 0.
int nx_fg8_build(nx_fg8_t* f, pm4_t* c, int slot, const nx_fg8_job_t* j);
// por separado (para la generacion propia con la presentacion de VCFG): el prologo
// (espera al fin del trabajo del juego + caches; ts0 = reloj de la GPU despues) y la
// espera a que el intermedio no este en pantalla
void nx_fg8_prologue(nx_fg8_t* f, pm4_t* c, int slot, uint32_t seq, const nx_image_t* cur, uint64_t ts0);
void nx_fg8_wait_mid(pm4_t* c, const nx_fg8_job_t* j);
// label de fin + espera del CP + invalidacion de caches (como el epilogo de VCFG)
void nx_fg8_end(nx_fg8_t* f, pm4_t* c, int slot, uint32_t seq);
