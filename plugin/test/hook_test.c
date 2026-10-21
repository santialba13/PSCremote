// Prueba en el PC (x86-64, como la PS4) del enganche propio nx_hook: engancha
// funciones de verdad y verifica que el hook se llame y que la original siga
// funcionando a traves del stub, incluso si empieza con instrucciones relativas.
//   make -C plugin/test hook
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "../source/nx_hook.h"

// --- plataforma (en la PS4 esta en main.c con sceKernelMmap/Mprotect)
void* nx_hook_map_near(void* near, size_t len)
{
    uintptr_t base = (uintptr_t)near & ~(uintptr_t)0xFFFF;
    for (int i = 1; i < 64; ++i) {
        uintptr_t hint = (i & 1) ? base - (uintptr_t)(i / 2 + 1) * (32u << 20) : base + (uintptr_t)(i / 2) * (32u << 20);
        void* p = mmap((void*)hint, len, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) continue;
        int64_t d = (int64_t)(uintptr_t)p - (int64_t)(uintptr_t)near;
        if (d < 0x7FF00000LL && d > -0x7FF00000LL) return p;
        munmap(p, len);
    }
    return NULL;
}
void nx_hook_unmap(void* p, size_t len) { munmap(p, len); }
int nx_hook_make_writable(void* p, size_t len)
{
    uintptr_t a = (uintptr_t)p & ~(uintptr_t)0xFFF, e = ((uintptr_t)p + len + 0xFFF) & ~(uintptr_t)0xFFF;
    return mprotect((void*)a, e - a, PROT_READ | PROT_WRITE | PROT_EXEC);
}

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("FALLO: " __VA_ARGS__); printf("\n"); ++fails; } } while (0)

// funciones objetivo escritas en asm para controlar exactamente sus primeros bytes
long g_value = 1234;
long helper(long x) { return x * 3; }
__asm__(
    ".text\n"
    // 1. prologo comun
    ".globl f_plain\n f_plain:\n push %rbp\n mov %rsp,%rbp\n lea 7(%rdi),%rax\n pop %rbp\n ret\n"
    // 2. empieza leyendo un global relativo a RIP
    ".globl f_riprel\n f_riprel:\n mov g_value(%rip),%rax\n add %rdi,%rax\n ret\n"
    // 3. empieza con call rel32
    ".globl f_call\n f_call:\n call helper\n add $1,%rax\n ret\n"
    // 4. thunk: jmp rel32 a la funcion real
    ".globl f_thunk\n f_thunk:\n .byte 0xE9\n .long f_plain2 - . - 4\n"
    ".globl f_thunk8\n f_thunk8:\n jmp f_plain2\n"
    ".globl f_plain2\n f_plain2:\n push %rbp\n mov %rsp,%rbp\n lea 100(%rdi),%rax\n pop %rbp\n ret\n"
    // 5. salto condicional corto al principio: no se puede enganchar
    ".globl f_jcc\n f_jcc:\n test %rdi,%rdi\n je 1f\n mov %rdi,%rax\n ret\n1:\n xor %eax,%eax\n ret\n"
);
long f_plain(long), f_riprel(long), f_call(long), f_thunk(long), f_plain2(long), f_jcc(long);

static nx_hook_t H[5];
typedef long (*fn_t)(long);
static long hk_plain(long x) { return ((fn_t)H[0].stub)(x) + 1000; }
static long hk_riprel(long x) { return ((fn_t)H[1].stub)(x) + 2000; }
static long hk_call(long x) { return ((fn_t)H[2].stub)(x) + 3000; }
static long hk_thunk(long x) { return ((fn_t)H[3].stub)(x) + 4000; }
static long hk_jcc(long x) { return x; }

int main(void)
{
    // los punteros se llaman a traves de volatile para que el compilador no los "adivine"
    fn_t volatile p_plain = f_plain, p_rip = f_riprel, p_call = f_call, p_thunk = f_thunk, p_jcc = f_jcc;
    CHECK(p_plain(1) == 8 && p_rip(1) == 1235 && p_call(2) == 7 && p_thunk(1) == 101, "antes de enganchar");
    CHECK(nx_hook_install(&H[0], (void*)f_plain, (void*)hk_plain) == 0, "plain: %s", H[0].why);
    CHECK(nx_hook_install(&H[1], (void*)f_riprel, (void*)hk_riprel) == 0, "riprel: %s", H[1].why);
    CHECK(nx_hook_install(&H[2], (void*)f_call, (void*)hk_call) == 0, "call: %s", H[2].why);
    CHECK(nx_hook_install(&H[3], (void*)f_thunk, (void*)hk_thunk) == 0, "thunk: %s", H[3].why);
    CHECK(H[3].target == (void*)f_plain2, "el thunk no se siguio hasta la funcion real");
    int rj = nx_hook_install(&H[4], (void*)f_jcc, (void*)hk_jcc);
    CHECK(rj < 0, "un salto condicional al principio deberia rechazarse");
    printf("rechazado como corresponde: %s\n", H[4].why);
    CHECK(p_plain(1) == 1008, "hook plain: %ld", p_plain(1));
    CHECK(p_rip(1) == 3235, "hook riprel (el global se lee bien desde el stub): %ld", p_rip(1));
    g_value = 10;
    CHECK(p_rip(1) == 2011, "riprel tras cambiar el global: %ld", p_rip(1));
    CHECK(p_call(2) == 3007, "hook call: %ld", p_call(2));
    CHECK(f_plain2(1) == 4101 && p_thunk(1) == 4101, "hook thunk: %ld", p_thunk(1));
    CHECK(p_jcc(0) == 0 && p_jcc(5) == 5, "la funcion rechazada quedo intacta");
    for (int i = 0; i < 4; ++i) {
        int64_t d = (int64_t)(uintptr_t)H[i].mem - (int64_t)(uintptr_t)H[i].target;
        CHECK(d < 0x7FF00000LL && d > -0x7FF00000LL, "puente %d lejos (%lld)", i, (long long)d);
    }
    nx_hook_remove(&H[0]);
    CHECK(p_plain(1) == 8, "al quitarlo vuelve la original");
    printf(fails ? "%d FALLOS\n" : "OK: enganche propio (plain, RIP-relativo, call, thunk, rechazo)\n", fails);
    return fails != 0;
}
