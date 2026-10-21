#include "nx_hook.h"

#include <stdio.h>
#include <string.h>

#include "HDE64.h"

#define PAGE 0x4000
#define STUB_OFF 32
#define FAR_LIMIT 0x7FF00000LL   // un poco menos de 2 GB

static int fits32(int64_t v) { return v >= -2147483647LL - 1 && v <= 2147483647LL; }

// jmp [rip+0]; dq dst  (14 bytes, absoluto)
static int put_abs_jmp(uint8_t* p, uint64_t dst)
{
    p[0] = 0xFF; p[1] = 0x25; p[2] = p[3] = p[4] = p[5] = 0;
    memcpy(p + 6, &dst, 8);
    return 14;
}

// call [rip+2]; jmp +8; dq dst  (16 bytes, absoluto)
static int put_abs_call(uint8_t* p, uint64_t dst)
{
    p[0] = 0xFF; p[1] = 0x15; p[2] = 2; p[3] = p[4] = p[5] = 0;
    p[6] = 0xEB; p[7] = 8;
    memcpy(p + 8, &dst, 8);
    return 16;
}

// sigue jmp rel32 / jmp corto / jmp [rip+x] (thunks) hasta la funcion de verdad
static uint8_t* follow(uint8_t* p)
{
    for (int i = 0; i < 4; ++i) {
        if (p[0] == 0xE9) {
            int32_t rel;
            memcpy(&rel, p + 1, 4);
            p = p + 5 + rel;
        } else if (p[0] == 0xEB) {
            p = p + 2 + (int8_t)p[1];
        } else if (p[0] == 0xFF && p[1] == 0x25) {
            int32_t rel;
            memcpy(&rel, p + 2, 4);
            uint64_t dst;
            memcpy(&dst, p + 6 + rel, 8);
            p = (uint8_t*)(uintptr_t)dst;
        } else {
            break;
        }
    }
    return p;
}

int nx_hook_install(nx_hook_t* h, void* target, void* hook)
{
    memset(h, 0, sizeof *h);
    if (!target || !hook) { snprintf(h->why, sizeof h->why, "puntero nulo"); return -1; }
    uint8_t* fn = follow((uint8_t*)target);
    h->target = fn;
    uint8_t* mem = nx_hook_map_near(fn, PAGE);
    if (!mem) { snprintf(h->why, sizeof h->why, "sin memoria ejecutable cerca de %p", (void*)fn); return -2; }
    int64_t dist = (int64_t)(uintptr_t)mem - (int64_t)(uintptr_t)fn;
    if (dist > FAR_LIMIT || dist < -FAR_LIMIT) {
        nx_hook_unmap(mem, PAGE);
        snprintf(h->why, sizeof h->why, "memoria del puente demasiado lejos");
        return -3;
    }
    // copia (y corrige) las instrucciones que pisa el jmp de 5 bytes
    uint8_t* stub = mem + STUB_OFF;
    int in = 0, out = 0;
    while (in < 5) {
        hde64s hs;
        unsigned len = hde64_disasm(fn + in, &hs);
        if (!len || (hs.flags & F_ERROR) || in + (int)len > (int)sizeof h->saved) {
            nx_hook_unmap(mem, PAGE);
            snprintf(h->why, sizeof h->why, "instruccion no reconocida en +%d", in);
            return -4;
        }
        uint8_t* src = fn + in;
        uint8_t* dst = stub + out;
        if (hs.flags & F_RELATIVE) {
            int64_t rel = (hs.flags & F_IMM32) ? (int32_t)hs.imm.imm32 : (int8_t)hs.imm.imm8;
            uint64_t abs = (uint64_t)(uintptr_t)(src + len) + (uint64_t)rel;
            if (hs.opcode == 0xE8) out += put_abs_call(dst, abs);
            else if (hs.opcode == 0xE9 || hs.opcode == 0xEB) out += put_abs_jmp(dst, abs);
            else {
                nx_hook_unmap(mem, PAGE);
                snprintf(h->why, sizeof h->why, "salto condicional al principio (op %02x)", hs.opcode);
                return -5;
            }
        } else if ((hs.flags & F_MODRM) && hs.modrm_mod == 0 && hs.modrm_rm == 5) {
            // direccionamiento relativo a RIP: se ajusta el desplazamiento
            int imm = (hs.flags & F_IMM64) ? 8 : (hs.flags & F_IMM32) ? 4 : (hs.flags & F_IMM16) ? 2
                    : (hs.flags & F_IMM8) ? 1 : 0;
            int doff = (int)len - imm - 4;
            int32_t disp;
            memcpy(&disp, src + doff, 4);
            int64_t abs = (int64_t)(uintptr_t)(src + len) + disp;
            int64_t nd = abs - (int64_t)(uintptr_t)(dst + len);
            if (!fits32(nd)) {
                nx_hook_unmap(mem, PAGE);
                snprintf(h->why, sizeof h->why, "instruccion relativa a RIP fuera de alcance");
                return -6;
            }
            memcpy(dst, src, len);
            int32_t nd32 = (int32_t)nd;
            memcpy(dst + doff, &nd32, 4);
            out += (int)len;
        } else {
            memcpy(dst, src, len);
            out += (int)len;
        }
        in += (int)len;
    }
    out += put_abs_jmp(stub + out, (uint64_t)(uintptr_t)(fn + in));   // vuelta a la original
    put_abs_jmp(mem, (uint64_t)(uintptr_t)hook);                      // puente al hook
    // recien ahora se toca la funcion: jmp rel32 al puente (cerca) + nops
    memcpy(h->saved, fn, (size_t)in);
    h->nsaved = in;
    if (nx_hook_make_writable(fn, (size_t)in) != 0) {
        nx_hook_unmap(mem, PAGE);
        snprintf(h->why, sizeof h->why, "no se pudo escribir la funcion");
        return -7;
    }
    uint8_t patch[32];
    int32_t rel = (int32_t)((int64_t)(uintptr_t)mem - (int64_t)(uintptr_t)(fn + 5));
    patch[0] = 0xE9;
    memcpy(patch + 1, &rel, 4);
    for (int i = 5; i < in; ++i) patch[i] = 0x90;
    memcpy(fn, patch, (size_t)in);
    h->mem = mem;
    h->stub = stub;
    return 0;
}

void nx_hook_remove(nx_hook_t* h)
{
    if (!h->stub || !h->target) return;
    if (nx_hook_make_writable(h->target, (size_t)h->nsaved) == 0) memcpy(h->target, h->saved, (size_t)h->nsaved);
    h->stub = NULL;   // el puente queda mapeado: algun hilo podria estar adentro
}
