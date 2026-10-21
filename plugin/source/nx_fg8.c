// Generacion de fotogramas del motor clasico de VCFG: ver nx_fg8.h.
// Portado de classic/source/main.c (build_dcb_flow8), fg8_table.h y fg_res.h de
// github.com/Melnikoff07/vcfg (MIT, Copyright (c) 2026 VCFG contributors).
// Sin cambios en los pases ni en la tabla que leen los shaders; quedan afuera las
// partes de VCFG que este plugin no usa (vectores del juego, volcados, segundo
// intermedio, copia de presentacion, rcas8/osd8: el menu lo dibuja nx_osd).
#include "nx_fg8.h"

#include <string.h>

#include "../include/vcfg8_shaders.h"

// ---- disposicion del slab
#define SLOT_BASE 0xA000u   // + slot * SLOT_SZ: tabla (0x400), labels, estadisticas, cola
#define SLOT_SZ 0x1000u
#define SLOT_LBL 0x800u     // fin de nuestro trabajo
#define SLOT_LBL2 0x808u    // fin del trabajo del juego (prologo)
#define SLOT_STATS 0x840u   // estadisticas que escribe phase8 (32 B)
#define SLOT_TS 0x880u      // reloj de la GPU: inicio y fin de nuestros pases
#define SLOT_TAIL 0x900u    // cola de flip (64 dwords)
#define MON_OFF 0xE000u     // valores del monitor (16 B, la fase adaptativa no se usa: ceros)
#define DCB_BASE 0x10000u   // + slot * 0x4000
_Static_assert(DCB_BASE + NX_FG8_SLOTS * NX_FG8_DCB_DWORDS * 4 <= NX_FG8_SLAB, "slab");
_Static_assert(SLOT_BASE + NX_FG8_SLOTS * SLOT_SZ <= MON_OFF, "slots");

// ---- fg_res.h (VCFG): la piramide y la grilla de movimiento quedan fijas en la de 1080p
#define FG_BX(w) (((w) + 15u) / 16u)
#define FG_BY(h) (((h) + 15u) / 16u)
#define FG_PX(w) ((w) / 64u)
#define FG_PY(h) FG_BY(h)
#define FG_M3Y(h) ((FG_BY(h) + 3u) / 4u)
#define FG_M2X(w) ((w) / 128u)
#define FG_M2Y(h) (FG_BY(h) / 2u)
#define FG_WX(w) (((w) + 7u) / 8u)
#define FG_WY(h) (((h) + 7u) / 8u)

// ---- fg8_table.h (VCFG): disposicion de la memoria de trabajo
#define W8_L1A 0x000000u
#define W8_L1B 0x07F000u
#define W8_L2A 0x0FE000u
#define W8_L2B 0x11E000u
#define W8_L3A 0x13E000u
#define W8_L3B 0x146000u
#define W8_V3 0x14E000u
#define W8_V2 0x14F000u
#define W8_FA 0x151000u
#define W8_FB 0x161000u
#define W8_MISC 0x171000u
#define W8_S2 0x172000u
#define W8_GK 0x174000u
#define W8_FT 0x17C000u
#define W8_END 0x18C000u
_Static_assert(W8_END <= NX_FG8_WORK, "work");

#define PKT3(op, nbody, st) (0xC0000000u | (((uint32_t)(nbody)-1u) << 16) | ((uint32_t)(op) << 8) | ((uint32_t)(st) << 1))

static void make_t(uint32_t* t, const nx_image_t* im)
{
    uint32_t tidx = im->tile_index;
    uint32_t dfmt = im->format == NX_FMT_RGB10A2_UNORM ? 9u : 10u;   // 2_10_10_10 / 8_8_8_8
    uint64_t a = im->gpu_addr;
    t[0] = (uint32_t)(a >> 8);
    t[1] = (uint32_t)((a >> 40) & 0xFFu) | (dfmt << 20);
    t[2] = (im->width - 1u) | ((im->height - 1u) << 14);
    t[3] = 4u | (5u << 3) | (6u << 6) | (7u << 9) | (tidx << 20) | (9u << 28);
    t[4] = (im->pitch - 1u) << 13;
    t[5] = 0;
    t[6] = nx_desc_neo && tidx != 8u ? 1u << 24 : 0u;   // PS4 Pro: tabla de macro-tiles de 16 pipes
    t[7] = 0;
}

static void make_v(uint32_t* v, uint64_t a, uint32_t nbytes)
{
    v[0] = (uint32_t)a;
    v[1] = (uint32_t)(a >> 32) & 0xFFFFu;
    v[2] = nbytes;
    v[3] = 0x00027FACu;
}

int nx_fg8_init(nx_fg8_t* f, uint8_t* slab, uint8_t* work)
{
    memset(f, 0, sizeof *f);
    if (!slab || !work) return -1;
    f->slab = slab;
    f->work = work;
    memset(slab, 0, NX_FG8_SLAB);
    const uint32_t* code[9] = { pyr8_code, me3_code, me2_code, gm8_code, fgs8_code, me1_code, mf8_code, phase8_code,
                                warp8_code };
    const uint32_t nw[9] = { PYR8_NWORDS, ME3_NWORDS, ME2_NWORDS, GM8_NWORDS, FGS8_NWORDS, ME1_NWORDS, MF8_NWORDS,
                             PHASE8_NWORDS, WARP8_NWORDS };
    uint32_t off = 0;
    for (int i = 0; i < 9; ++i) {
        if (off + nw[i] * 4 > SLOT_BASE) return -2;
        memcpy(slab + off, code[i], nw[i] * 4);
        f->sh_off[i] = off;
        off = (off + nw[i] * 4 + 0xFFu) & ~0xFFu;
    }
    f->ok = 1;
    return 0;
}

static uint8_t* slotb(nx_fg8_t* f, int slot) { return f->slab + SLOT_BASE + (uint32_t)slot * SLOT_SZ; }
uint32_t* nx_fg8_dcb(nx_fg8_t* f, int slot) { return (uint32_t*)(f->slab + DCB_BASE + (uint32_t)slot * NX_FG8_DCB_DWORDS * 4); }
uint32_t* nx_fg8_tail(nx_fg8_t* f, int slot) { return (uint32_t*)(slotb(f, slot) + SLOT_TAIL); }
volatile uint32_t* nx_fg8_label(nx_fg8_t* f, int slot) { return (volatile uint32_t*)(slotb(f, slot) + SLOT_LBL); }
volatile uint64_t* nx_fg8_ts(nx_fg8_t* f, int slot) { return (volatile uint64_t*)(slotb(f, slot) + SLOT_TS); }

// tabla que leen los shaders (fill_table8_wh + fill_table8_ext de VCFG)
static void fill_table(uint32_t* tb, const nx_fg8_job_t* j, uint8_t* work, volatile uint32_t* mon,
                       volatile uint32_t* stats)
{
    uint32_t r = j->seq & 1u;
    memset(tb, 0, 0x400);
    make_t(tb + 0, j->prev);    // +0x00 T#prev
    make_t(tb + 8, j->cur);     // +0x20 T#cur
    make_t(tb + 16, j->mid);    // +0x40 T#out
    // +0x60 S#: clamp, FORCE_UNNORMALIZED, bilineal, sin mips
    tb[24] = 2u | (2u << 3) | (2u << 6) | (1u << 15);
    tb[26] = (1u << 20) | (1u << 22);
    tb[28] = r;
    tb[29] = (j->mid_index & 0x7FFFu) | ((j->t_q16 & 0xFFFFu) << 16);
    tb[30] = j->repeat ? 2u : 0u;   // bit 1: sin generacion (repite C)
    tb[31] = 0;                     // sin rectangulo de HUD del juego
#define WA(o) ((uint64_t)(uintptr_t)(work + (o)))
    make_v(tb + 32, WA(r ? W8_L1B : W8_L1A), 960 * 540 + 16);
    make_v(tb + 36, WA(r ? W8_L1A : W8_L1B), 960 * 540 + 16);
    make_v(tb + 40, WA(r ? W8_L2B : W8_L2A), 480 * 270 + 16);
    make_v(tb + 44, WA(r ? W8_L2A : W8_L2B), 480 * 270 + 16);
    make_v(tb + 48, WA(r ? W8_L3B : W8_L3A), 240 * 135 + 16);
    make_v(tb + 52, WA(r ? W8_L3A : W8_L3B), 240 * 135 + 16);
    make_v(tb + 56, WA(W8_V3), 4096);
    make_v(tb + 60, WA(W8_V2), 60 * 34 * 4);
    make_v(tb + 64, WA(r ? W8_FB : W8_FA), 120 * 68 * 8);
    make_v(tb + 68, WA(r ? W8_FA : W8_FB), 120 * 68 * 8);
    make_v(tb + 72, WA(W8_MISC), 32);
    make_v(tb + 80, WA(W8_S2), 60 * 34 * 4);
    make_v(tb + 84, WA(W8_GK), 120 * 68 * 4);
    make_v(tb + 96, WA(W8_FT), 120 * 68 * 8);
#undef WA
    make_v(tb + 76, (uint64_t)(uintptr_t)mon, 16);
    make_v(tb + 92, (uint64_t)(uintptr_t)stats, 32);
    // +0x228/0x22C: rectangulo del menu (x0 | y0 << 16, celdas de 16 px): el warp
    // toma ahi el fotograma actual en lugar de deformar el anterior (que lo trae grabado)
    if (j->osd_w > 0 && j->osd_h > 0) {
        uint32_t cols = (uint32_t)(j->osd_w + 15) / 16u, rows = (uint32_t)(j->osd_h + 15) / 16u;
        if (cols > 255) cols = 255;
        if (rows > 255) rows = 255;
        tb[138] = ((uint32_t)j->osd_x & 4095u) | (((uint32_t)j->osd_y & 4095u) << 16);
        tb[139] = cols | (rows << 8);
    }
}

#define EMIT(v) (c->n < c->cap ? (void)(c->buf[c->n++] = (v)) : (void)(c->overflow = 1))

static void acquire(pm4_t* c, uint32_t cntl, uint32_t size256, uint64_t base)
{
    EMIT(PKT3(0x58, 6, 0)); EMIT(cntl); EMIT(size256); EMIT(0); EMIT((uint32_t)(base >> 8)); EMIT((uint32_t)(base >> 40));
    EMIT(0x0A);
}

static void barrier(pm4_t* c)   // PASS_BARRIER de VCFG
{
    EMIT(PKT3(0x46, 1, 0)); EMIT(0x00000407);
    acquire(c, 0x80400000u, 1, 0);
}

static void set_pass(pm4_t* c, uint64_t pgm, uint32_t r1, uint32_t r2)
{
    EMIT(PKT3(0x76, 3, 1)); EMIT(0x20C); EMIT((uint32_t)(pgm >> 8)); EMIT((uint32_t)(pgm >> 40));
    EMIT(PKT3(0x76, 3, 1)); EMIT(0x212); EMIT(r1); EMIT(r2);
}

static void dispatch(pm4_t* c, uint32_t x, uint32_t y)
{
    EMIT(PKT3(0x15, 4, 1)); EMIT(x); EMIT(y); EMIT(1); EMIT(1);
}

static void eop_label(pm4_t* c, uint64_t lbl, uint32_t seq)
{
    EMIT(PKT3(0x47, 5, 0)); EMIT(0x00000528); EMIT((uint32_t)lbl); EMIT(0x40000000u | ((uint32_t)(lbl >> 32) & 0xFFFFu));
    EMIT(seq); EMIT(0);
}

static void wait_label(pm4_t* c, uint64_t lbl, uint32_t seq)
{
    EMIT(PKT3(0x3C, 6, 0)); EMIT(0x13); EMIT((uint32_t)lbl); EMIT((uint32_t)(lbl >> 32)); EMIT(seq); EMIT(0xFFFFFFFFu);
    EMIT(0x0A);
}

void nx_fg8_prologue(nx_fg8_t* f, pm4_t* c, int slot, uint32_t seq, const nx_image_t* cur, uint64_t ts0)
{
    uint8_t* sb = slotb(f, slot);
    uint64_t lbl2 = (uint64_t)(uintptr_t)(sb + SLOT_LBL2);
    *(volatile uint64_t*)(sb + SLOT_LBL2) = 0;
    *(volatile uint64_t*)(sb + SLOT_LBL) = 0;
    ((volatile uint64_t*)(sb + SLOT_TS))[0] = ((volatile uint64_t*)(sb + SLOT_TS))[1] = 0;
    uint32_t nbytes = cur->pitch * cur->height * 4u;
    // el juego termino su fotograma (el CP espera, no la CPU) y sus escrituras de
    // color estan en memoria; caches de constantes/instrucciones limpias
    EMIT(PKT3(0x46, 1, 0)); EMIT(0x0000002E);
    EMIT(PKT3(0x46, 1, 0)); EMIT(0x00000407);
    eop_label(c, lbl2, seq);
    wait_label(c, lbl2, seq);
    acquire(c, 0x82003FC0u, nbytes >> 8, cur->gpu_addr);
    acquire(c, 0x80C40000u, 1, 0);
    acquire(c, 0x88400000u, 1, 0);
    if (ts0) pm4_timestamp(c, ts0);
}

void nx_fg8_wait_mid(pm4_t* c, const nx_fg8_job_t* j)
{
    int r = -1;
    if (c->n + 7 <= c->cap && j->wait_flip_done) r = j->wait_flip_done(c->buf + c->n, 7, j->handle, j->mid_index);
    if (r != 0 && c->n + 7 <= c->cap) {
        c->buf[c->n] = PKT3(0x10, 6, 0);
        for (int i = 1; i < 7; ++i) c->buf[c->n + i] = 0;
    }
    if (c->n + 7 <= c->cap) c->n += 7; else c->overflow = 1;
}

int nx_fg8_build(nx_fg8_t* f, pm4_t* c, int slot, const nx_fg8_job_t* j)
{
    if (!f->ok || slot < 0 || slot >= NX_FG8_SLOTS || !j->prev || !j->cur || !j->mid) return -1;
    uint8_t* sb = slotb(f, slot);
    uint32_t* tb = (uint32_t*)sb;
    uint64_t tbl = (uint64_t)(uintptr_t)tb;
    uint64_t gb = (uint64_t)(uintptr_t)f->slab;
    volatile uint32_t* stats = (volatile uint32_t*)(sb + SLOT_STATS);
    for (int i = 0; i < 8; ++i) stats[i] = 0xFFFFFFFFu;
    fill_table(tb, j, f->work, (volatile uint32_t*)(f->slab + MON_OFF), stats);
    uint32_t W = j->cur->width, H = j->cur->height;
    if (!j->no_prologue) nx_fg8_prologue(f, c, slot, j->seq, j->cur, j->ts0);
    EMIT(PKT3(0x76, 4, 1)); EMIT(0x207); EMIT(64); EMIT(1); EMIT(1);
    EMIT(PKT3(0x76, 3, 1)); EMIT(0x240); EMIT((uint32_t)tbl); EMIT((uint32_t)(tbl >> 32));

    uint32_t px = FG_PX(1920u), py = FG_PY(1080u);
    set_pass(c, gb + f->sh_off[0], PYR8_RSRC1, PYR8_RSRC2); dispatch(c, px, py); barrier(c);
    set_pass(c, gb + f->sh_off[1], ME3_RSRC1, ME3_RSRC2); dispatch(c, FG_PX(1920u), FG_M3Y(1080u)); barrier(c);
    set_pass(c, gb + f->sh_off[2], ME2_RSRC1, ME2_RSRC2); dispatch(c, FG_M2X(1920u), FG_M2Y(1080u)); barrier(c);
    set_pass(c, gb + f->sh_off[3], GM8_RSRC1, GM8_RSRC2); dispatch(c, 1, 1); barrier(c);
    set_pass(c, gb + f->sh_off[4], FGS8_RSRC1, FGS8_RSRC2); dispatch(c, 32, 1); barrier(c);
    set_pass(c, gb + f->sh_off[5], ME1_RSRC1, ME1_RSRC2); dispatch(c, px, py); barrier(c);
    set_pass(c, gb + f->sh_off[6], MF8_RSRC1, MF8_RSRC2); dispatch(c, px, py); barrier(c);
    // recien antes de escribir el intermedio: que no este en pantalla
    nx_fg8_wait_mid(c, j);
    acquire(c, 0x80C40000u, 1, 0);
    set_pass(c, gb + f->sh_off[7], PHASE8_RSRC1, PHASE8_RSRC2); dispatch(c, 1, 1); barrier(c);
    set_pass(c, gb + f->sh_off[8], WARP8_RSRC1, WARP8_RSRC2); dispatch(c, FG_WX(W), FG_WY(H)); barrier(c);
    return c->overflow ? -2 : 0;
}

void nx_fg8_end(nx_fg8_t* f, pm4_t* c, int slot, uint32_t seq)
{
    uint64_t lbl = (uint64_t)(uintptr_t)(slotb(f, slot) + SLOT_LBL);
    eop_label(c, lbl, seq);
    wait_label(c, lbl, seq);
    acquire(c, 0x80C40000u, 1, 0);
}
